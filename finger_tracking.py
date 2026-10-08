"""Finger tracking (pipeline steps 22-24): where are the fingertips on the map.

Step 22  hand_crop / MediaPipe   -- crop around the paper (plus margin),
                                    enlarge up to 2.5x, find the index
                                    fingertip of up to 2 hands.
Step 23  map the fingertip       -- crop pixels -> frame pixels -> map
                                    pixels, through a homography built
                                    fresh every frame from that frame's 4
                                    corners (nothing carries over, so
                                    nothing accumulates error). Each tip is
                                    flagged on_paper if it lands on the map.
Step 24  hand identity           -- match each tip to whichever hand was
                                    closest last frame (both hands solved
                                    together), then label Left/Right by each
                                    hand's running majority of MediaPipe's
                                    guesses, so one bad frame can't swap them.

All coordinates on the map are the PNG's own pixels.
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FINGER_SMOOTHING = 0.35      # smoothing of each hand's identity-tracking point
MAX_FINGER_JUMP = 140.0      # map px; a bigger jump isn't the same hand (unless it's been gone a while)
MAX_MISSED_FRAMES = 20
PATH_MARGIN = 0.05           # tips this close outside the map (fraction of its size) still draw on the trail
TRACK_MARGIN = 0.50          # tips further out than this are ignored entirely
PATH_BREAK_JUMP = 140.0      # map px; a trail never draws a straight line across a jump this
                             # big between frames (a hand reappearing elsewhere, or a tip
                             # mapped through a briefly-wrong crop) -- it starts a new stroke


def create_hand_landmarker(model_path="hand_landmarker.task"):
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
            min_tracking_confidence=0.35))


# ---------------------------------------------------------------- step 22

def hand_crop(frame, corners):
    """Enlarged paper-centered crop and how to map it back to the frame.
    Small hands are the usual reason only one is detected, so the crop is
    upscaled (capped at 2.5x / ~1280 px to keep MediaPipe fast)."""
    h, w = frame.shape[:2]
    p = np.asarray(corners, np.float32)
    x0, y0 = np.floor(p.min(axis=0)).astype(int)
    x1, y1 = np.ceil(p.max(axis=0)).astype(int)
    mx, my = int(0.22 * max(x1 - x0, 1)), int(0.30 * max(y1 - y0, 1))
    x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame, 0, 0, 1.0
    scale = min(2.5, max(1.0, 1280.0 / max(crop.shape[:2])))
    if scale > 1.01:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, x0, y0, scale


# ---------------------------------------------------------------- step 24

class HandTrack:
    """One physical hand: a smoothed position for matching, plus a running
    tally of MediaPipe's Left/Right guesses for it."""

    def __init__(self, point):
        self.point = np.asarray(point, np.float32)
        self.missed = 0
        self.votes = {"Left": 0, "Right": 0}

    def update(self, point):
        point = np.asarray(point, np.float32)
        if np.linalg.norm(point - self.point) > MAX_FINGER_JUMP and self.missed <= MAX_MISSED_FRAMES:
            self.miss()
            return
        jumped = np.linalg.norm(point - self.point) > MAX_FINGER_JUMP
        self.point = point if jumped else (
            (1 - FINGER_SMOOTHING) * self.point + FINGER_SMOOTHING * point)
        self.missed = 0

    def miss(self):
        self.missed += 1

    def lean(self):
        """> 0 leans Left, < 0 leans Right."""
        return self.votes["Left"] - self.votes["Right"]


def assign_detections(tracks, detections):
    """owners[j] = the HandTrack detection j belongs to (or None).
    Matched by position, never by MediaPipe's label."""
    owners = [None] * len(detections)
    if not tracks:
        for j in sorted(range(len(detections)), key=lambda k: detections[k][0]):
            owners[j] = HandTrack(detections[j])
            tracks.append(owners[j])
        return owners
    if len(tracks) == 2 and len(detections) == 2:
        # Solve both together, so one hand can't steal the other's detection.
        direct = (np.linalg.norm(detections[0] - tracks[0].point) +
                  np.linalg.norm(detections[1] - tracks[1].point))
        crossed = (np.linalg.norm(detections[1] - tracks[0].point) +
                   np.linalg.norm(detections[0] - tracks[1].point))
        for track, j in zip(tracks, (0, 1) if direct <= crossed else (1, 0)):
            track.update(detections[j])
            owners[j] = track
        return owners
    if len(tracks) == 2 and len(detections) == 1:
        k = min(range(2), key=lambda i: np.linalg.norm(detections[0] - tracks[i].point))
        tracks[k].update(detections[0])
        tracks[1 - k].miss()
        owners[0] = tracks[k]
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
            owners[j] = HandTrack(detections[j])
            tracks.append(owners[j])
    return owners


def label_hands(owners, raw_labels):
    """Stable Left/Right per detection from each hand's vote tally. Two
    visible hands always get different labels; a tie falls back to this
    frame's MediaPipe label."""
    labels = [None] * len(owners)
    active = [j for j, t in enumerate(owners) if t is not None]
    if len(active) == 1:
        j = active[0]
        lean = owners[j].lean()
        labels[j] = raw_labels[j] if lean == 0 else ("Left" if lean > 0 else "Right")
    elif len(active) == 2:
        a, b = active
        diff = owners[a].lean() - owners[b].lean()
        a_left = (raw_labels[a] == "Left" or raw_labels[b] == "Right") if diff == 0 else diff > 0
        labels[a], labels[b] = ("Left", "Right") if a_left else ("Right", "Left")
    return labels


# ---------------------------------------------------------------- all together

class FingerTracker:
    """Runs steps 22-24 every frame and keeps the per-frame log and trails."""

    def __init__(self, ref_w, ref_h, model_path="hand_landmarker.task"):
        self.ref_w, self.ref_h = ref_w, ref_h
        self.ref_corners = np.float32([[0, 0], [ref_w - 1, 0], [ref_w - 1, ref_h - 1], [0, ref_h - 1]])
        self.detector = create_hand_landmarker(model_path)
        self.tracks = []
        self.paths = {"Left": [], "Right": []}   # map points; None = a gap
        self.session_log = []

    def update(self, frame, corners, frame_index, timestamp_ms, page_source=None, page_score=None):
        """Returns this frame's fingertips: a list of dicts with hand,
        frame_x/frame_y, map_x/map_y and on_paper."""
        tips = []
        if corners is not None:
            tips = self._detect(frame, np.asarray(corners, np.float32), timestamp_ms)
        seen = set()
        for tip in tips:
            x, y = tip["map_x"], tip["map_y"]
            mx, my = PATH_MARGIN * self.ref_w, PATH_MARGIN * self.ref_h
            if -mx <= x < self.ref_w + mx and -my <= y < self.ref_h + my:
                path = self.paths[tip["hand"]]
                if path and path[-1] is not None and \
                        np.hypot(x - path[-1][0], y - path[-1][1]) > PATH_BREAK_JUMP:
                    path.append(None)
                path.append((int(round(x)), int(round(y))))
                seen.add(tip["hand"])
        for hand, pts in self.paths.items():
            if hand not in seen and pts and pts[-1] is not None:
                pts.append(None)
        self.session_log.append({
            "frame": frame_index, "t_ms": timestamp_ms,
            "page_source": page_source,
            "page_score": None if page_score is None else round(float(page_score), 3),
            "corners": None if corners is None else np.round(corners, 1).tolist(),
            "Left": next((t for t in tips if t["hand"] == "Left"), None),
            "Right": next((t for t in tips if t["hand"] == "Right"), None),
        })
        return tips

    def _detect(self, frame, corners, timestamp_ms):
        H = cv2.getPerspectiveTransform(corners, self.ref_corners)
        image, cx, cy, scale = hand_crop(frame, corners)                        # step 22
        result = self.detector.detect_for_video(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), timestamp_ms)
        found = []
        for i, landmarks in enumerate(result.hand_landmarks or []):
            tip = landmarks[8]                                                  # index fingertip
            fx = cx + tip.x * image.shape[1] / scale                            # step 23
            fy = cy + tip.y * image.shape[0] / scale
            mx, my = cv2.perspectiveTransform(np.float32([[[fx, fy]]]), H)[0, 0]
            if (-TRACK_MARGIN * self.ref_w <= mx <= (1 + TRACK_MARGIN) * self.ref_w and
                    -TRACK_MARGIN * self.ref_h <= my <= (1 + TRACK_MARGIN) * self.ref_h):
                found.append((np.float32([mx, my]), float(fx), float(fy),
                              result.handedness[i][0].category_name))
        owners = assign_detections(self.tracks, [f[0] for f in found])          # step 24
        for track, f in zip(owners, found):
            if track is not None:
                track.votes[f[3]] = track.votes.get(f[3], 0) + 1
        labels = label_hands(owners, [f[3] for f in found])
        tips = []
        for (m, fx, fy, raw), hand in zip(found, labels):
            if hand is None:
                continue
            tips.append({"hand": hand, "frame_x": round(fx, 1), "frame_y": round(fy, 1),
                         "map_x": round(float(m[0]), 1), "map_y": round(float(m[1]), 1),
                         "on_paper": bool(0 <= m[0] < self.ref_w and 0 <= m[1] < self.ref_h),
                         "mediapipe_label": raw})
        return tips

    # ------------------------------------------------------------ drawing
    def draw_trails(self, canvas, H_map_to_canvas=None):
        """Left trail red, right trail blue -- on the map (H None) or on a
        video frame (H = this frame's map -> frame homography)."""
        for hand, color in (("Left", (0, 0, 255)), ("Right", (255, 0, 0))):
            pts = self.paths[hand]
            if H_map_to_canvas is not None:
                pts = _warp_points(pts, H_map_to_canvas)
            prev = None
            for p in pts:
                if p is not None and prev is not None:
                    cv2.line(canvas, prev, p, color, 4, cv2.LINE_AA)
                prev = p

    def close(self):
        self.detector.close()


def _warp_points(points, H):
    idx = [i for i, p in enumerate(points) if p is not None]
    out = [None] * len(points)
    if idx:
        warped = cv2.perspectiveTransform(np.float32([points[i] for i in idx])[:, None], H)[:, 0]
        for j, i in enumerate(idx):
            out[i] = (int(round(warped[j, 0])), int(round(warped[j, 1])))
    return out
