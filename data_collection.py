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
    """Associate by position, not MediaPipe handedness (which often flips)."""
    if not tracks:
        for i, p in enumerate(sorted(detections, key=lambda q: q[0])):
            tracks.append(FingerTrack(p, f"Finger {i + 1}"))
        return
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
        return
    if len(tracks) == 2 and len(detections) == 1:
        chosen = min(range(2), key=lambda i: np.linalg.norm(
            detections[0] - tracks[i].point))
        tracks[chosen].update(detections[0])
        tracks[1 - chosen].miss()
        return
    unused = set(range(len(detections)))
    for track in tracks:
        if not unused:
            track.miss()
            continue
        j = min(unused, key=lambda k: np.linalg.norm(detections[k] - track.point))
        if np.linalg.norm(detections[j] - track.point) <= MAX_FINGER_JUMP:
            track.update(detections[j])
            unused.remove(j)
        else:
            track.miss()
    for j in unused:
        if len(tracks) < 2:
            tracks.append(FingerTrack(detections[j], f"Finger {len(tracks) + 1}"))


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


def update_symbol_timer(state, stats, handedness, in_region, timestamp_ms, symbol):
    # start/stop per hand per box; closed intervals roll into stats as a
    # visit count + total dwell time, but only once they clear MIN_VISIT_MS --
    # a shorter touch doesn't count as a real visit, just passing over it.
    # "entered" is separate and set the instant the finger is in the box at
    # all, regardless of duration, so a box that was only ever brushed isn't
    # reported as having been missed entirely.
    entry = stats[symbol]
    key = (handedness, symbol)
    if in_region:
        entry["entered"] = True
        if key not in state:
            state[key] = timestamp_ms
    elif key in state:
        start = state.pop(key)
        duration = timestamp_ms - start
        if duration >= MIN_VISIT_MS:
            entry["visits"] += 1
            entry["total_ms"] += duration


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
        self.symbol_timers = {}
        self.symbol_stats = {
            name: {"type": entry["type"], "entered": False,
                  "visits": 0, "total_ms": 0}
            for name, entry in self.symbol_boxes.items()
        }

    def update(self, frame, corners, frame_index, timestamp_ms):
        """Detect fingertips in this frame and record them. Returns
        (display_markers, detection_count) for the caller to draw --
        display_markers is a list of ((frame_x, frame_y), handedness) for
        only the hands actually seen this exact frame.
        """
        corners = np.asarray(corners, np.float32)
        H_frame_to_map = cv2.getPerspectiveTransform(corners, self.ref_corners)
        hand_image, crop_x, crop_y, crop_scale = page_hand_crop(frame, corners)
        result = self.detector.detect_for_video(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)), timestamp_ms)

        detections = []
        display_markers = []
        seen_hands = set()
        frame_positions = {}
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
                    handedness = result.handedness[hand_i][0].category_name
                    display_markers.append(((int(round(frame_x)),
                                             int(round(frame_y))),
                                            handedness))
                    self.raw_paths.setdefault(handedness, []).append(
                        tuple(np.rint(mapped).astype(int)))
                    seen_hands.add(handedness)
                    frame_positions[handedness] = {
                        "frame_x": float(frame_x), "frame_y": float(frame_y),
                        "map_x": float(mapped[0]), "map_y": float(mapped[1]),
                    }
                    for name, entry in self.symbol_boxes.items():
                        update_symbol_timer(self.symbol_timers, self.symbol_stats,
                                            handedness, in_box(mapped, entry["box"]),
                                            timestamp_ms, name)
        # A None creates a visible break rather than connecting across an
        # interval in which MediaPipe did not actually see that hand.
        for handedness, samples in self.raw_paths.items():
            if (handedness not in seen_hands and samples and
                    samples[-1] is not None):
                samples.append(None)

        assign_detections(self.tracks, detections)

        self.session_log.append({
            "frame": frame_index,
            "t_ms": timestamp_ms,
            "Left": frame_positions.get("Left"),
            "Right": frame_positions.get("Right"),
        })

        return display_markers, len(detections)

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

    def save_dashboard_data(self, path, final_timestamp_ms, map_path):
        """One consolidated, dashboard-ready JSON: every labeled box with its
        type, pixel coordinates, whether it was ever entered at all (so a
        box nobody ever touched is reported as missed, not silently absent),
        >=1s visit count + total dwell time, and each hand's full timestamped
        path -- everything a dashboard needs without re-deriving anything
        from the raw per-frame log."""
        # close out any dwell timers still open when the video ended/quit
        for (handedness, symbol), start in list(self.symbol_timers.items()):
            duration = final_timestamp_ms - start
            if duration >= MIN_VISIT_MS:
                entry = self.symbol_stats[symbol]
                entry["visits"] += 1
                entry["total_ms"] += duration
        self.symbol_timers.clear()

        boxes_out = {}
        for name, box_entry in self.symbol_boxes.items():
            stats = self.symbol_stats[name]
            boxes_out[name] = {
                "type": stats["type"],
                "box_px": [round(v, 1) for v in box_entry["box"]],
                "entered": stats["entered"],
                "missed": not stats["entered"],
                "visits": stats["visits"],
                "total_ms": stats["total_ms"],
            }

        data = {
            "map": os.path.basename(map_path),
            "ref_w": self.ref_w,
            "ref_h": self.ref_h,
            "boxes": boxes_out,
            "path": {
                "Left": self.get_path_points("Left"),
                "Right": self.get_path_points("Right"),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {path}")
        missed = [n for n, b in boxes_out.items() if b["missed"]]
        print(f"  {len(boxes_out) - len(missed)}/{len(boxes_out)} box(es) entered; "
              f"missed: {missed if missed else 'none'}")
        for name, b in sorted(boxes_out.items(), key=lambda kv: -kv[1]["total_ms"]):
            print(f"  [{b['type']}] {name}: {b['visits']} visit(s) >=1s, "
                  f"{b['total_ms']}ms total, entered={b['entered']}")
        return data

    def close(self):
        self.detector.close()
