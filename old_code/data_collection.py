"""Finger-tracking data collection: MediaPipe hand detection, paper-relative
coordinate mapping, left/right identity tracking, and the resulting session
data (trails + a full per-frame log).

finger_tracking_manual_objects.py owns finding the paper's 4 corners each
frame; this module owns everything about the fingers once those corners are
known, and is meant to be called into every frame, not run on its own.

Coordinate mapping note: a fingertip's raw video-frame pixel is meaningless
on its own, because the physical paper moves (slides, rotates, tilts) in
every frame. Each frame's already-known 4 corners are used to build a fresh
homography straight to the reference map's own fixed pixel space, and the
fingertip is mapped through it directly. This is recomputed from scratch
every frame from that frame's own corners -- it never nudges or reuses a
previous frame's point -- so there is nothing to accumulate error into, and
it is exact under rotation and perspective/tilt changes, not just sliding.
"""

import json
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FINGER_SMOOTHING = 0.35          # lower = smoother
MAX_FINGER_JUMP = 140.0          # map pixels; rejects detections after occlusion
MAX_MISSED_FRAMES = 20
MIN_VISIT_MS = 1000              # a dwell shorter than this isn't a real "visit"
EXIT_GRACE_MS = 400              # a finger must be out of a box this long
                                 # before the visit ends -- MediaPipe drops a
                                 # hand for a few frames at a time, and that
                                 # shouldn't split one visit into several


def create_hand_landmarker(model_path="hand_landmarker.task"):
    """MediaPipe HandLandmarker configured for per-frame video processing."""
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
            min_tracking_confidence=0.35))


def page_hand_crop(frame, corners):
    """Return an enlarged page-centered ROI and its frame-coordinate mapping."""
    h, w = frame.shape[:2]
    p = np.asarray(corners, np.float32)
    x0, y0 = np.floor(p.min(axis=0)).astype(int)
    x1, y1 = np.ceil(p.max(axis=0)).astype(int)
    margin_x = int(0.22 * max(x1 - x0, 1))
    margin_y = int(0.30 * max(y1 - y0, 1))
    x0, y0 = max(0, x0 - margin_x), max(0, y0 - margin_y)
    x1, y1 = min(w, x1 + margin_x), min(h, y1 + margin_y)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame, 0, 0, 1.0
    # Small hands are the common reason only one is detected. Upscale the ROI,
    # while capping size to keep inference responsive.
    scale = min(2.5, max(1.0, 1280.0 / max(crop.shape[:2])))
    if scale > 1.01:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    return crop, x0, y0, scale


class FingerTrack:
    """One physical hand's smoothed point + rendered trail history."""

    def __init__(self, point, name):
        self.point = np.asarray(point, np.float32)
        self.name = name
        self.missed = 0
        self.samples = []       # None marks a break in the rendered path
        # Running tally of MediaPipe's per-frame "Left"/"Right" guesses for
        # this physical hand. MediaPipe's label can flip on any single frame,
        # but its majority over a hand's whole history is reliable.
        self.votes = {"Left": 0, "Right": 0}

    def vote(self, handedness):
        self.votes[handedness] = self.votes.get(handedness, 0) + 1

    def lean(self):
        """> 0 leans Left, < 0 leans Right, 0 undecided."""
        return self.votes.get("Left", 0) - self.votes.get("Right", 0)

    def update(self, point):
        point = np.asarray(point, np.float32)
        distance = np.linalg.norm(point - self.point)
        if distance > MAX_FINGER_JUMP:
            # After a real occlusion, the hand may reappear far from its last
            # location. Reacquire instead of leaving this track stuck forever.
            if self.missed > MAX_MISSED_FRAMES:
                if self.samples and self.samples[-1] is not None:
                    self.samples.append(None)
                self.point = point
                self.missed = 0
                self.samples.append(tuple(np.rint(self.point).astype(int)))
                return
            self.miss()
            return
        self.point = ((1.0 - FINGER_SMOOTHING) * self.point +
                      FINGER_SMOOTHING * point)
        self.missed = 0
        self.samples.append(tuple(np.rint(self.point).astype(int)))

    def miss(self):
        self.missed += 1
        if self.samples and self.samples[-1] is not None:
            self.samples.append(None)


def assign_detections(tracks, detections):
    """Associate by position, not MediaPipe handedness (which often flips).

    Returns owners, where owners[j] is the track that detection j was
    assigned to (or None if it couldn't be given to any track).
    """
    owners = [None] * len(detections)
    if not tracks:
        for j in sorted(range(len(detections)), key=lambda k: detections[k][0]):
            owners[j] = FingerTrack(detections[j], f"Finger {len(tracks) + 1}")
            tracks.append(owners[j])
        return owners
    # Solve the two-hand assignment jointly. Greedy matching can let the first
    # track steal the second hand's detection and make the other marker vanish.
    if len(tracks) == 2 and len(detections) == 2:
        direct = (np.linalg.norm(detections[0] - tracks[0].point) +
                  np.linalg.norm(detections[1] - tracks[1].point))
        crossed = (np.linalg.norm(detections[1] - tracks[0].point) +
                   np.linalg.norm(detections[0] - tracks[1].point))
        order = (0, 1) if direct <= crossed else (1, 0)
        for track, j in zip(tracks, order):
            track.update(detections[j])
            owners[j] = track
        return owners
    if len(tracks) == 2 and len(detections) == 1:
        chosen = min(range(2), key=lambda i: np.linalg.norm(
            detections[0] - tracks[i].point))
        tracks[chosen].update(detections[0])
        tracks[1 - chosen].miss()
        owners[0] = tracks[chosen]
        return owners
    unused = set(range(len(detections)))
    for track in tracks:
        if not unused:
            track.miss()
            continue
        j = min(unused, key=lambda k: np.linalg.norm(detections[k] - track.point))
        if np.linalg.norm(detections[j] - track.point) <= MAX_FINGER_JUMP:
            track.update(detections[j])
            owners[j] = track
            unused.remove(j)
        else:
            track.miss()
    for j in sorted(unused):
        if len(tracks) < 2:
            owners[j] = FingerTrack(detections[j], f"Finger {len(tracks) + 1}")
            tracks.append(owners[j])
    return owners


def label_detections(owners, raw_handedness):
    """Stable "Left"/"Right" label for each owned detection this frame.

    Each label comes from its track's accumulated vote, not from MediaPipe's
    guess on this one frame, so a one-frame handedness flip can't swap which
    trail/log/box timer a point lands in. When two hands are visible they
    always get different labels: whichever track leans more Left gets Left.
    A track with no lean yet (a tie) falls back to this frame's raw label.
    """
    labels = [None] * len(owners)
    active = [j for j, t in enumerate(owners) if t is not None]
    if len(active) == 1:
        j = active[0]
        lean = owners[j].lean()
        labels[j] = raw_handedness[j] if lean == 0 else (
            "Left" if lean > 0 else "Right")
    elif len(active) == 2:
        a, b = active
        diff = owners[a].lean() - owners[b].lean()
        if diff == 0:
            a_left = raw_handedness[a] == "Left" or raw_handedness[b] == "Right"
        else:
            a_left = diff > 0
        labels[a], labels[b] = ("Left", "Right") if a_left else ("Right", "Left")
    return labels


def in_box(pt, box):
    x0, y0, x1, y1 = box
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def load_symbol_boxes(map_path, ref_w, ref_h):
    """Named box regions (symbols + rooms) in ref-map pixel space, from
    label_symbols.py's <map>.symbols.json. Each entry carries a "type"
    ("symbol" or "room") alongside its box -- any number of each, not a
    fixed count, each becoming its own independent dwell timer below."""
    json_path = os.path.splitext(map_path)[0] + ".symbols.json"
    if not os.path.exists(json_path):
        print(f"No {json_path} -- run label_symbols.py on this map first. "
              "Continuing without box stats.")
        return {}
    with open(json_path) as f:
        raw = json.load(f)
    boxes = {}
    for name, entry in raw.items():
        if isinstance(entry, (list, tuple)):
            # Pre-room/symbol format: name -> [fx0, fy0, fx1, fy1], symbols only.
            entry = {"type": "symbol", "box": entry}
        fx0, fy0, fx1, fy1 = entry["box"]
        boxes[name] = {
            "type": entry["type"],
            "box": (fx0 * ref_w, fy0 * ref_h, fx1 * ref_w, fy1 * ref_h),
        }
    n_symbols = sum(1 for e in boxes.values() if e["type"] == "symbol")
    n_rooms = sum(1 for e in boxes.values() if e["type"] == "room")
    print(f"Loaded {len(boxes)} box(es) from {json_path} "
          f"({n_symbols} symbol(s), {n_rooms} room(s))")
    return boxes


class BoxTimer:
    """Enter/exit timer for one labeled box (symbol or room).

    Timed per *user*, not per hand: the box is occupied while any fingertip
    is inside it, so two hands in the same room count once, not twice. A
    visit starts the first frame a finger is inside, and ends at the last
    frame one was inside -- once no finger has been in it for EXIT_GRACE_MS.
    That grace period also covers a hand simply vanishing (lifted off the
    page, or MediaPipe missing it), which previously left the timer running
    until the hand happened to be seen again somewhere else.

    Visits shorter than MIN_VISIT_MS are just a finger passing over the box:
    counted as "brushes", not visits, and kept out of total time.
    """

    def __init__(self, name, box_type, box):
        self.name = name
        self.type = box_type
        self.box = box
        self.visits = []        # {"enter_ms", "exit_ms", "duration_ms", "hands"}
        self.brushes = 0
        self._start = None
        self._last_inside = None
        self._hands = set()

    def step(self, hands_inside, timestamp_ms):
        """hands_inside: set of hand labels whose fingertip is in the box."""
        if hands_inside:
            if self._start is None:
                self._start = timestamp_ms
                self._hands = set()
            self._hands |= hands_inside
            self._last_inside = timestamp_ms
        elif (self._start is not None and
              timestamp_ms - self._last_inside > EXIT_GRACE_MS):
            self.close()

    def close(self):
        """End any open visit (also called once when the video ends)."""
        if self._start is None:
            return
        duration = self._last_inside - self._start
        if duration >= MIN_VISIT_MS:
            self.visits.append({"enter_ms": self._start,
                                "exit_ms": self._last_inside,
                                "duration_ms": duration,
                                "hands": sorted(self._hands)})
        else:
            self.brushes += 1
        self._start = None

    @property
    def occupied(self):
        return self._start is not None

    def summary(self):
        durations = [v["duration_ms"] for v in self.visits]
        touched = bool(self.visits) or self.brushes > 0
        status = ("visited" if self.visits else
                  "brushed" if touched else "missed")
        return {
            "type": self.type,
            "box_px": [round(float(v), 1) for v in self.box],
            # visited = at least one visit >= MIN_VISIT_MS; brushed = touched
            # but never that long; missed = the finger never entered at all
            "status": status,
            "touched": touched,
            "missed": not touched,
            "visits": len(self.visits),
            "brushes": self.brushes,
            "total_ms": int(sum(durations)),
            "longest_ms": int(max(durations, default=0)),
            "first_enter_ms": self.visits[0]["enter_ms"] if self.visits else None,
            "events": self.visits,
        }


def draw_trail(canvas, samples, color):
    previous = None
    for point in samples:
        if point is None:
            previous = None
        elif previous is not None:
            cv2.line(canvas, previous, point, color, 4, cv2.LINE_AA)
            previous = point
        else:
            previous = point


def _warp_points(samples, H):
    """Transform a list of points (with possible None gaps) through a
    homography in one batched call rather than one cv2 call per point --
    matters here since a trail can grow into the thousands of points over a
    long video and this runs every displayed frame."""
    idx = [i for i, p in enumerate(samples) if p is not None]
    if not idx:
        return list(samples)
    pts = np.float32([samples[i] for i in idx]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)[:, 0, :]
    out = [None] * len(samples)
    for j, i in enumerate(idx):
        out[i] = tuple(np.rint(warped[j]).astype(int))
    return out


class FingerDataCollector:
    """Runs MediaPipe on each frame, maps any detected fingertip into
    paper-relative (map) space using that frame's already-known corners, and
    accumulates both a drawable trail and a full per-frame session log.
    """

    def __init__(self, ref_w, ref_h, model_path="hand_landmarker.task",
                symbol_boxes=None):
        self.ref_w = ref_w
        self.ref_h = ref_h
        self.ref_corners = np.float32([[0, 0], [ref_w - 1, 0],
                                       [ref_w - 1, ref_h - 1], [0, ref_h - 1]])
        # label_symbols.py regions in ref-map pixels; empty = no box stats
        self.symbol_boxes = symbol_boxes or {}
        self.detector = create_hand_landmarker(model_path)
        self.tracks = []
        self.raw_paths = {"Left": [], "Right": []}
        self.session_log = []
        # one independent enter/exit timer per labeled box -- any number of
        # symbols and rooms, each its own instance
        self.box_timers = {
            name: BoxTimer(name, entry["type"], entry["box"])
            for name, entry in self.symbol_boxes.items()
        }

    def update(self, frame, corners, frame_index, timestamp_ms, page_source=None):
        """Detect fingertips in this frame and record them. Returns
        (display_markers, detection_count) for the caller to draw --
        display_markers is a list of ((frame_x, frame_y), handedness) for
        only the hands actually seen this exact frame.

        page_source is how the paper's corners were found this frame
        (edge/PNG/flow/held/lost...), logged so the dashboard can show how
        trustworthy each stretch of the session's data is.
        """
        corners = np.asarray(corners, np.float32)
        H_frame_to_map = cv2.getPerspectiveTransform(corners, self.ref_corners)
        hand_image, crop_x, crop_y, crop_scale = page_hand_crop(frame, corners)
        result = self.detector.detect_for_video(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)), timestamp_ms)

        detections = []
        frame_points = []
        raw_handedness = []
        if result.hand_landmarks:
            for hand_i, landmarks in enumerate(result.hand_landmarks):
                tip = landmarks[8]
                # Convert normalized enlarged-crop coordinates back to the
                # original full video frame before applying the homography.
                frame_x = crop_x + tip.x * hand_image.shape[1] / crop_scale
                frame_y = crop_y + tip.y * hand_image.shape[0] / crop_scale
                p = np.float32([[[frame_x, frame_y]]])
                mapped = cv2.perspectiveTransform(p, H_frame_to_map)[0, 0]
                margin_x, margin_y = 0.05 * self.ref_w, 0.05 * self.ref_h
                if (-margin_x <= mapped[0] < self.ref_w + margin_x and
                        -margin_y <= mapped[1] < self.ref_h + margin_y):
                    detections.append(mapped)
                    frame_points.append((float(frame_x), float(frame_y)))
                    raw_handedness.append(
                        result.handedness[hand_i][0].category_name)

        # Identity comes from position continuity (which physical hand was
        # closest last frame), then each hand's Left/Right label from that
        # track's accumulated vote -- so MediaPipe flipping its label on one
        # frame can no longer swap which trail, log entry or timer it feeds.
        owners = assign_detections(self.tracks, detections)
        for track, handedness in zip(owners, raw_handedness):
            if track is not None:
                track.vote(handedness)
        labels = label_detections(owners, raw_handedness)

        display_markers = []
        seen_hands = set()
        frame_positions = {}
        for mapped, (frame_x, frame_y), raw, handedness in zip(
                detections, frame_points, raw_handedness, labels):
            if handedness is None:
                # No track could take this detection (rare: a 3rd candidate
                # while 2 hands are already tracked) -- recording it under a
                # guessed label could put it on the other hand's trail.
                continue
            display_markers.append(((int(round(frame_x)), int(round(frame_y))),
                                    handedness))
            self.raw_paths.setdefault(handedness, []).append(
                tuple(np.rint(mapped).astype(int)))
            seen_hands.add(handedness)
            frame_positions[handedness] = {
                "frame_x": frame_x, "frame_y": frame_y,
                "map_x": float(mapped[0]), "map_y": float(mapped[1]),
                "mediapipe_label": raw,
            }
        # A None creates a visible break rather than connecting across an
        # interval in which MediaPipe did not actually see that hand.
        for handedness, samples in self.raw_paths.items():
            if (handedness not in seen_hands and samples and
                    samples[-1] is not None):
                samples.append(None)

        # Every box steps every frame, even with no hand in view, so a visit
        # ends when the finger leaves (or disappears), not whenever it's next seen.
        for timer in self.box_timers.values():
            inside = {hand for hand, pos in frame_positions.items()
                      if in_box((pos["map_x"], pos["map_y"]), timer.box)}
            timer.step(inside, timestamp_ms)

        self.session_log.append({
            "frame": frame_index,
            "t_ms": timestamp_ms,
            "page_source": page_source,
            "Left": frame_positions.get("Left"),
            "Right": frame_positions.get("Right"),
        })

        return display_markers, len(display_markers)

    def occupied_boxes(self):
        """Names of boxes a finger is currently in (for the live overlay)."""
        return [name for name, t in self.box_timers.items() if t.occupied]

    def draw_trails(self, canvas):
        draw_trail(canvas, self.raw_paths.get("Left", []), (0, 0, 255))
        draw_trail(canvas, self.raw_paths.get("Right", []), (255, 0, 0))

    def draw_trails_in_frame(self, canvas, h_map_to_frame):
        """Same as draw_trails, but for overlaying onto a live *video* frame
        instead of the flat reference map. The trail is stored in
        paper-relative (map) space; a fixed spot on the paper lands at a
        different screen pixel every frame as the paper moves, so each point
        has to be warped through this frame's current map->frame homography
        before drawing -- recomputed fresh every call, same as everywhere
        else in this pipeline, so nothing here accumulates drift either."""
        left = _warp_points(self.raw_paths.get("Left", []), h_map_to_frame)
        right = _warp_points(self.raw_paths.get("Right", []), h_map_to_frame)
        draw_trail(canvas, left, (0, 0, 255))
        draw_trail(canvas, right, (255, 0, 0))

    def draw_boxes_in_frame(self, canvas, h_map_to_frame):
        """Outline every labeled box on the live video frame (warped through
        this frame's map->frame homography, like the trails), filled in
        while a finger is currently inside it, so the timers can be checked
        against what's actually happening on screen."""
        type_colors = {"symbol": (0, 200, 0), "room": (255, 180, 0)}
        for name, timer in self.box_timers.items():
            x0, y0, x1, y1 = timer.box
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            pts = cv2.perspectiveTransform(quad[None], h_map_to_frame)[0]
            pts = np.int32(np.rint(pts))
            color = type_colors.get(timer.type, (200, 200, 200))
            if timer.occupied:
                fill = canvas.copy()
                cv2.fillConvexPoly(fill, pts, color)
                cv2.addWeighted(fill, 0.35, canvas, 0.65, 0, dst=canvas)
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

    def save_session_log(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.session_log, f, indent=2)

    def get_path_points(self, handedness):
        """Lightweight (x, y, t_ms) polyline for `handedness`, extracted from
        the full per-frame session log -- a None marks a real gap (the hand
        wasn't seen that frame) rather than connecting across it, same as
        the on-screen trail. This is what a dashboard would actually plot."""
        points = []
        for record in self.session_log:
            pos = record.get(handedness)
            if pos is None:
                points.append(None)
            else:
                points.append({"x": pos["map_x"], "y": pos["map_y"],
                               "t_ms": record["t_ms"]})
        return points

    def save_dashboard_data(self, path, final_timestamp_ms, map_path,
                            video_path=None, fps=None):
        """One consolidated, dashboard-ready JSON: every labeled box with its
        type, pixel coordinates, status (visited / only brushed / missed),
        >=1s visit count, total + longest dwell time and every enter/exit
        event; the chronological order boxes were visited in; and each
        hand's full timestamped path -- everything dashboard.py needs
        without re-deriving anything from the raw per-frame log.

        All map coordinates are in the tracker's map space: the reference
        PNG rotated 180 degrees (the same space label_symbols.py labels in).
        """
        # close out any visit still open when the video ended/quit
        for timer in self.box_timers.values():
            timer.close()

        boxes_out = {name: t.summary() for name, t in self.box_timers.items()}
        sequence = sorted(
            ({"name": name, "type": t.type, **v}
             for name, t in self.box_timers.items() for v in t.visits),
            key=lambda e: e["enter_ms"])

        frames = len(self.session_log)
        with_hand = sum(1 for r in self.session_log
                        if r.get("Left") or r.get("Right"))
        data = {
            "video": os.path.basename(video_path) if video_path else None,
            "map": os.path.basename(map_path),
            "coordinate_space": "reference PNG rotated 180 degrees",
            "ref_w": self.ref_w,
            "ref_h": self.ref_h,
            "fps": fps,
            "duration_ms": final_timestamp_ms,
            "frames": frames,
            "frames_with_hand": with_hand,
            "min_visit_ms": MIN_VISIT_MS,
            "boxes": boxes_out,
            "sequence": sequence,
            "page_source": [r.get("page_source") for r in self.session_log],
            "path": {
                "Left": self.get_path_points("Left"),
                "Right": self.get_path_points("Right"),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {path}")
        if boxes_out:
            missed = [n for n, b in boxes_out.items() if b["missed"]]
            visited = [n for n, b in boxes_out.items() if b["status"] == "visited"]
            print(f"  {len(visited)}/{len(boxes_out)} box(es) visited for "
                  f">={MIN_VISIT_MS}ms; missed entirely: "
                  f"{missed if missed else 'none'}")
            for name, b in sorted(boxes_out.items(),
                                  key=lambda kv: -kv[1]["total_ms"]):
                print(f"  [{b['type']}] {name}: {b['status']}, "
                      f"{b['visits']} visit(s), {b['total_ms']}ms total")
        return data

    def close(self):
        self.detector.close()
