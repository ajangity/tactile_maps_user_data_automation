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

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FINGER_SMOOTHING = 0.35          # lower = smoother
MAX_FINGER_JUMP = 140.0          # map pixels; rejects detections after occlusion
MAX_MISSED_FRAMES = 20
STAIRS_BOX_FRAC = (0.05, 0.05, 0.95, 0.95)  # wide test box for tonight, shrink once we know real stairs coords


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


def update_symbol_timer(state, handedness, in_region, timestamp_ms, symbol="stairs"):
    # start/stop per hand per symbol, prints when it fires
    key = (handedness, symbol)
    if in_region and key not in state:
        state[key] = timestamp_ms
        print(f"[{symbol}] {handedness} start {timestamp_ms}ms")
    elif not in_region and key in state:
        start = state.pop(key)
        print(f"[{symbol}] {handedness} stop {timestamp_ms}ms, dur {timestamp_ms - start}ms")


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


def draw_points(canvas, samples, color, radius=3):
    """One dot per recorded frame, not connected by lines -- unlike
    draw_trail, consecutive frames are not joined, so gaps (None, where the
    hand wasn't detected) need no special handling here."""
    for point in samples:
        if point is not None:
            cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)


class FingerDataCollector:
    """Runs MediaPipe on each frame, maps any detected fingertip into
    paper-relative (map) space using that frame's already-known corners, and
    accumulates both a drawable trail and a full per-frame session log.
    """

    def __init__(self, ref_w, ref_h, model_path="hand_landmarker.task",
                stairs_box_frac=STAIRS_BOX_FRAC):
        self.ref_w = ref_w
        self.ref_h = ref_h
        self.ref_corners = np.float32([[0, 0], [ref_w - 1, 0],
                                       [ref_w - 1, ref_h - 1], [0, ref_h - 1]])
        self.stairs_box = (stairs_box_frac[0] * ref_w, stairs_box_frac[1] * ref_h,
                           stairs_box_frac[2] * ref_w, stairs_box_frac[3] * ref_h)
        self.detector = create_hand_landmarker(model_path)
        self.tracks = []
        self.raw_paths = {"Left": [], "Right": []}
        self.session_log = []
        self.symbol_timers = {}

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
                    update_symbol_timer(self.symbol_timers, handedness,
                                        in_box(mapped, self.stairs_box), timestamp_ms)
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
        # Left: one continuous connected line (a "trail"). Right: one dot
        # per recorded frame, left unconnected -- a per-frame point cloud
        # rather than a smoothed path, so individual frame positions (and
        # any pauses, where dots cluster) stay directly visible.
        draw_trail(canvas, self.raw_paths.get("Left", []), (0, 0, 255))
        draw_points(canvas, self.raw_paths.get("Right", []), (255, 0, 0))

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
        draw_points(canvas, right, (255, 0, 0))

    def save_session_log(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.session_log, f, indent=2)

    def close(self):
        self.detector.close()
