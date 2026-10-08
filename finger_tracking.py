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
Step 24  hand identity           -- match each hand to whichever hand's
                                    wrist was closest last frame (both hands
                                    solved together), then label Left/Right
                                    from a short, fading, confidence-weighted
                                    memory of MediaPipe's own Left/Right calls.
                                    One low-confidence frame can't swap the
                                    labels, and a wrong label can't stick.

About MediaPipe's Left/Right: the hand landmark model labels every hand,
every frame, from that hand's own image (it doesn't track identity). It
assumes a mirrored, selfie-style image. A camera across the table looking
at the backs of the hands gives the same handedness as a mirrored selfie,
so its labels are right as they come. On S-19_Elevator.mp4 they were right
for 3588 of 3592 two-hand detections. So step 24 follows MediaPipe and only
smooths over its occasional low-confidence flips. Don't replace its labels
with a long-running vote: if the two tracks swap hands once, a vote like
that keeps the wrong labels for minutes. If a recording is mirrored, every
label will come out swapped.

All coordinates on the map are the PNG's own pixels.
"""

import itertools

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LABEL_MEMORY = 0.7           # per frame: share of a hand's Left/Right evidence kept from last frame
MAX_WRIST_JUMP = 200.0       # map px; a wrist moving further than this between frames is a new hand
MAX_MISSED_FRAMES = 15       # a hand unseen this many frames is forgotten (label memory and all)
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
    """One physical hand: where its wrist was last frame, and a short,
    fading memory of MediaPipe's Left/Right calls for it."""

    def __init__(self, wrist):
        self.wrist = wrist
        self.missed = 0
        self.evidence = 0.0      # > 0 leans Left, < 0 leans Right

    def update(self, wrist, label, score):
        """score is MediaPipe's handedness confidence: a 0.55 call barely
        counts, a 0.99 call counts fully. Old calls fade by LABEL_MEMORY each
        frame, so if this track ever ends up on the other hand, the label
        follows MediaPipe again within a couple of frames."""
        self.wrist = wrist
        self.missed = 0
        vote = 2.0 * score - 1.0
        self.evidence = LABEL_MEMORY * self.evidence + (vote if label == "Left" else -vote)


def match_hands(tracks, wrists):
    """owners[j] = the HandTrack wrist j belongs to. Both hands are solved
    together (most matches, then least total movement); a wrist further
    than MAX_WRIST_JUMP from every track starts a new one, and a track
    unseen for MAX_MISSED_FRAMES is forgotten."""
    dist = lambda j, k: float(np.linalg.norm(wrists[j] - tracks[k].wrist))
    pairs = [(j, k) for j in range(len(wrists)) for k in range(len(tracks))
             if dist(j, k) <= MAX_WRIST_JUMP]
    options = [()] + [(p,) for p in pairs] + [
        (p, q) for p, q in itertools.combinations(pairs, 2) if p[0] != q[0] and p[1] != q[1]]
    best = min(options, key=lambda o: (-len(o), sum(dist(j, k) for j, k in o)))
    owners = [None] * len(wrists)
    for j, k in best:
        owners[j] = tracks[k]
    for track in tracks:
        if track not in owners:
            track.missed += 1
    tracks[:] = [t for t in tracks if t.missed <= MAX_MISSED_FRAMES]
    for j, owner in enumerate(owners):
        if owner is None:
            owners[j] = HandTrack(wrists[j])
            tracks.append(owners[j])
    tracks.sort(key=lambda t: t.missed)    # at most 2 hands: drop the stalest
    del tracks[2:]
    return owners


def label_hands(owners, raw_labels):
    """Left/Right per detection from each hand's evidence. Two visible hands
    always get different labels (the one leaning more Left is Left); no
    evidence at all falls back to this frame's MediaPipe label."""
    if len(owners) == 1:
        ev = owners[0].evidence
        return [raw_labels[0] if ev == 0 else "Left" if ev > 0 else "Right"]
    if len(owners) == 2:
        a_left = owners[0].evidence > owners[1].evidence
        return ["Left", "Right"] if a_left else ["Right", "Left"]
    return []


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
            # index fingertip (8) and wrist (0): crop -> frame -> map         step 23
            frame_pts = np.float32([[cx + landmarks[k].x * image.shape[1] / scale,
                                     cy + landmarks[k].y * image.shape[0] / scale] for k in (8, 0)])
            (mx, my), wrist = cv2.perspectiveTransform(frame_pts[None], H)[0]
            fx, fy = frame_pts[0]
            if (-TRACK_MARGIN * self.ref_w <= mx <= (1 + TRACK_MARGIN) * self.ref_w and
                    -TRACK_MARGIN * self.ref_h <= my <= (1 + TRACK_MARGIN) * self.ref_h):
                category = result.handedness[i][0]
                found.append((np.float32([mx, my]), float(fx), float(fy), wrist,
                              category.category_name, category.score))
        owners = match_hands(self.tracks, [f[3] for f in found])                # step 24
        for track, f in zip(owners, found):
            track.update(f[3], f[4], f[5])
        labels = label_hands(owners, [f[4] for f in found])
        tips = []
        for (m, fx, fy, _, raw, _), hand in zip(found, labels):
            tips.append({"hand": hand, "frame_x": round(fx, 1), "frame_y": round(fy, 1),
                         "map_x": round(float(m[0]), 1), "map_y": round(float(m[1]), 1),
                         "on_paper": bool(0 <= m[0] < self.ref_w and 0 <= m[1] < self.ref_h),
                         "mediapipe_label": raw})
        return tips

    # ------------------------------------------------------------ drawing
    def draw_trails(self, canvas, H_map_to_canvas=None, hands=("Left", "Right")):
        """Left trail red, right trail blue -- on the map (H None) or on a
        video frame (H = this frame's map -> frame homography). hands picks
        which trails to draw."""
        for hand, color in (("Left", (0, 0, 255)), ("Right", (255, 0, 0))):
            if hand not in hands:
                continue
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
