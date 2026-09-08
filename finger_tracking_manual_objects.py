import argparse
import math
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Estimate a fresh PNG -> video homography on every frame.  Unlike four independent
# patch trackers, all correspondences must agree on one projective transform.
TRACK_PAGE_MOTION = True
MIN_PAGE_INLIERS = 14
MIN_INLIER_COVERAGE = 0.035      # fraction of reference-map area
MAX_REPROJECTION_ERROR = 3.0     # pixels in the video frame
MAX_FRAME_MOTION = 0.12          # fraction of frame diagonal per frame
PAGE_POSE_SMOOTHING = 0.65       # high = responsive; lower if the camera is noisy
DEFAULT_PLAYBACK_SPEED = 1.5     # processes every frame; only display timing changes
FINGER_SMOOTHING = 0.35          # lower = smoother
MAX_FINGER_JUMP = 140.0          # map pixels; rejects detections after occlusion
MAX_MISSED_FRAMES = 20

STAIRS_BOX_FRAC = (0.05, 0.05, 0.95, 0.95)  # wide test box for tonight, shrink once we know real stairs coords

ui_mode = "align"
drag_pts = []
active_pt_idx = -1
is_paused = True


def mouse_handler(event, x, y, flags, param):
    global active_pt_idx
    if ui_mode != "align" and not is_paused:
        return
    can_edit = (ui_mode == "align" or is_paused)
    if event == cv2.EVENT_LBUTTONDOWN and can_edit:
        if not drag_pts:
            return
        distances = [math.hypot(x - pt[0], y - pt[1]) for pt in drag_pts]
        nearest = int(np.argmin(distances))
        # During initial alignment, clicking anywhere pulls the nearest handle.
        # This is easier than having to hit a small circle precisely.
        if ui_mode == "align" or is_paused or distances[nearest] < 35:
            active_pt_idx = nearest
            drag_pts[active_pt_idx] = [x, y]
    elif (event == cv2.EVENT_MOUSEMOVE and can_edit and
          (flags & cv2.EVENT_FLAG_LBUTTON)):
        # Some HighGUI backends miss LBUTTONDOWN while frames are being shown.
        # Recover by acquiring the nearest handle during the drag itself.
        if active_pt_idx == -1 and drag_pts:
            active_pt_idx = int(np.argmin([
                math.hypot(x - pt[0], y - pt[1]) for pt in drag_pts]))
        if active_pt_idx != -1:
            drag_pts[active_pt_idx] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        active_pt_idx = -1


def valid_quad(points, frame_w, frame_h):
    """Reject mirrored, collapsed, or wildly moving page estimates."""
    p = np.asarray(points, np.float32)
    area = cv2.contourArea(p)
    return (area > 0.03 * frame_w * frame_h and
            np.all(p[:, 0] > -20) and np.all(p[:, 0] < frame_w + 20) and
            np.all(p[:, 1] > -20) and np.all(p[:, 1] < frame_h + 20))


class PagePose:
    """Register the original map directly to each frame (no accumulated drift)."""
    def __init__(self, reference, frame, corners):
        self.reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.ref_h, self.ref_w = self.reference_gray.shape
        self.ref_corners = np.float32([[0, 0], [self.ref_w - 1, 0],
                                       [self.ref_w - 1, self.ref_h - 1],
                                       [0, self.ref_h - 1]])
        self.corners = np.asarray(corners, np.float32)
        # SIFT is much more stable than CSRT/ORB under perspective, scale and glare.
        if hasattr(cv2, "SIFT_create"):
            self.feature = cv2.SIFT_create(nfeatures=3500,
                                           contrastThreshold=0.025,
                                           edgeThreshold=12)
            norm = cv2.NORM_L2
            self.ratio = 0.74
        else:
            self.feature = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
            norm = cv2.NORM_HAMMING
            self.ratio = 0.72
        self.ref_kp, self.ref_des = self.feature.detectAndCompute(
            self.reference_gray, None)
        self.matcher = cv2.BFMatcher(norm)
        self.inliers = 0
        self.error = float("inf")
        self.coverage = 0.0
        self.source = "initial"
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.flow_points = self._seed_flow(self.prev_gray)

    def _seed_flow(self, gray):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(self.corners), 255)
        mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
        return cv2.goodFeaturesToTrack(gray, 700, 0.008, 7, mask=mask,
                                       blockSize=7)

    def reset(self, frame, corners):
        """Accept a paused manual correction as the new tracking state."""
        self.corners = np.asarray(corners, np.float32).copy()
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.flow_points = self._seed_flow(self.prev_gray)
        self.source = "manual"
        self.inliers = 0
        self.error = 0.0
        self.coverage = 0.0

    def _reference_update(self, frame):
        if self.ref_des is None or len(self.ref_kp) < MIN_PAGE_INLIERS:
            return self.corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Search only near the last page polygon.  This removes most hands, faces,
        # clothing and background before descriptor matching even begins.
        search = np.zeros(gray.shape, np.uint8)
        centre = self.corners.mean(axis=0)
        expanded = centre + 1.12 * (self.corners - centre)
        cv2.fillConvexPoly(search, np.int32(expanded), 255)
        kp, des = self.feature.detectAndCompute(gray, search)
        if des is None or len(kp) < MIN_PAGE_INLIERS:
            return self.corners
        pairs = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and
                pair[0].distance < self.ratio * pair[1].distance]
        if len(good) < MIN_PAGE_INLIERS:
            return self.corners
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 2.5,
                                     maxIters=10000, confidence=0.999)
        if H is None or mask is None:
            return self.corners
        inlier_mask = mask.ravel().astype(bool)
        self.inliers = int(inlier_mask.sum())
        if self.inliers < MIN_PAGE_INLIERS:
            return self.corners

        projected = cv2.perspectiveTransform(src[inlier_mask, None], H)[:, 0]
        errors = np.linalg.norm(projected - dst[inlier_mask], axis=1)
        self.error = float(np.median(errors))
        # A tight cluster can produce a mathematically valid but unstable H.
        hull = cv2.convexHull(src[inlier_mask])
        self.coverage = cv2.contourArea(hull) / float(self.ref_w * self.ref_h)
        candidate = cv2.perspectiveTransform(self.ref_corners[None], H)[0]
        step = np.linalg.norm(candidate - self.corners, axis=1)
        h, w = gray.shape
        max_step = MAX_FRAME_MOTION * math.hypot(w, h)
        old_area = abs(cv2.contourArea(self.corners))
        new_area = abs(cv2.contourArea(candidate))
        area_ratio = new_area / max(old_area, 1.0)
        accepted = (valid_quad(candidate, w, h) and
                    self.error <= MAX_REPROJECTION_ERROR and
                    self.coverage >= MIN_INLIER_COVERAGE and
                    np.max(step) <= max_step and
                    0.65 <= area_ratio <= 1.55)
        if accepted:
            self.corners = ((1.0 - PAGE_POSE_SMOOTHING) * self.corners +
                            PAGE_POSE_SMOOTHING * candidate)
            self.source = "PNG"
        return self.corners

    def update(self, frame):
        """Prefer absolute PNG registration; use robust dense flow if obscured."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        before = self.corners.copy()
        self.source = "held"
        self._reference_update(frame)
        png_accepted = np.max(np.linalg.norm(self.corners - before, axis=1)) > 0.01

        if (not png_accepted and self.flow_points is not None and
                len(self.flow_points) >= MIN_PAGE_INLIERS):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.flow_points, None,
                winSize=(25, 25), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          30, 0.01))
            if nxt is None or status is None:
                self.prev_gray = gray
                self.flow_points = self._seed_flow(gray)
                return self.corners
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.prev_gray, nxt, None, winSize=(25, 25), maxLevel=3)
            if back is None or back_status is None:
                self.prev_gray = gray
                self.flow_points = self._seed_flow(gray)
                return self.corners
            fb = np.linalg.norm(self.flow_points[:, 0] - back[:, 0], axis=1)
            keep = ((status[:, 0] == 1) & (back_status[:, 0] == 1) & (fb < 1.5))
            old = self.flow_points[keep, 0]
            new = nxt[keep, 0]
            if len(old) >= MIN_PAGE_INLIERS:
                delta, mask = cv2.findHomography(
                    old, new, cv2.USAC_MAGSAC, 2.0,
                    maxIters=5000, confidence=0.999)
                if delta is not None and mask is not None:
                    inside = mask.ravel().astype(bool)
                    candidate = cv2.perspectiveTransform(
                        before[None], delta)[0]
                    projected = cv2.perspectiveTransform(
                        old[inside, None], delta)[:, 0]
                    error = float(np.median(np.linalg.norm(
                        projected - new[inside], axis=1)))
                    hull_area = (cv2.contourArea(cv2.convexHull(old[inside]))
                                 if inside.sum() >= 3 else 0.0)
                    page_area = max(abs(cv2.contourArea(before)), 1.0)
                    coverage = hull_area / page_area
                    h, w = gray.shape
                    step = np.linalg.norm(candidate - before, axis=1)
                    if (inside.sum() >= MIN_PAGE_INLIERS and error < 2.0 and
                            coverage > 0.08 and valid_quad(candidate, w, h) and
                            np.max(step) < MAX_FRAME_MOTION * math.hypot(w, h)):
                        self.corners = (0.15 * before + 0.85 * candidate)
                        self.inliers = int(inside.sum())
                        self.error = error
                        self.coverage = coverage
                        self.source = "flow"

        # Re-detect hundreds of page points each frame. RANSAC can tolerate an
        # occluding arm as long as visible map texture remains the majority.
        self.prev_gray = gray
        self.flow_points = self._seed_flow(gray)
        return self.corners


class FingerTrack:
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


def manual_keyframe_tracking(video_path, reference_image_path,
                             output_path="finger_paths.png"):
    global drag_pts, is_paused, ui_mode
    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        raise FileNotFoundError(reference_image_path)
    ref_img = cv2.rotate(ref_img, cv2.ROTATE_180)
    ref_h, ref_w = ref_img.shape[:2]
    ref_corners = np.float32([[0, 0], [ref_w - 1, 0],
                              [ref_w - 1, ref_h - 1], [0, ref_h - 1]])
    stairs_box = (STAIRS_BOX_FRAC[0] * ref_w, STAIRS_BOX_FRAC[1] * ref_h,
                 STAIRS_BOX_FRAC[2] * ref_w, STAIRS_BOX_FRAC[3] * ref_h)

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read {video_path}")
    frame_h, frame_w = frame.shape[:2]
    drag_pts = [[100, 100], [frame_w - 100, 100],
                [frame_w - 100, frame_h - 100], [100, frame_h - 100]]

    # AUTOSIZE keeps mouse coordinates in the same pixel coordinate system as
    # the video frame, which is essential for an accurate homography.
    cv2.namedWindow("Video Tracker", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Video Tracker", mouse_handler)
    print("Click or drag near each map corner; the nearest green handle will follow. "
          "Press Enter when aligned.")
    ui_mode = "align"
    while True:
        shown = frame.copy()
        corners = np.asarray(drag_pts, np.float32)
        H = cv2.getPerspectiveTransform(ref_corners, corners)
        overlay = cv2.warpPerspective(ref_img, H, (frame_w, frame_h))
        mask = cv2.warpPerspective(np.full((ref_h, ref_w), 255, np.uint8), H,
                                   (frame_w, frame_h))
        blended = cv2.addWeighted(shown, 0.55, overlay, 0.45, 0)
        shown[mask > 0] = blended[mask > 0]
        for p in drag_pts:
            cv2.circle(shown, tuple(np.int32(p)), 8, (0, 255, 0), -1)
        cv2.polylines(shown, [corners.astype(np.int32)], True,
                      (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(shown, "Click/drag each map corner, then press Enter",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Video Tracker", shown)
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13):
            break
        if key == ord("q"):
            cap.release(); cv2.destroyAllWindows(); return

    pose = PagePose(ref_img, frame, drag_pts)
    tracks = []
    raw_paths = {"Left": [], "Right": []}
    symbol_timers = {}
    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
            min_tracking_confidence=0.35))

    ui_mode = "track"
    is_paused = False
    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    detection_count = 0
    display_markers = []
    playback_speed = DEFAULT_PLAYBACK_SPEED
    print("Tracking. Space pauses/resumes; q saves and quits.")
    while True:
        loop_started = time.perf_counter()
        if not is_paused:
            ok, frame = cap.read()
            if not ok:
                break
            if TRACK_PAGE_MOTION:
                drag_pts = pose.update(frame).tolist()

            corners = np.asarray(drag_pts, np.float32)
            H_frame_to_map = cv2.getPerspectiveTransform(corners, ref_corners)
            hand_image, crop_x, crop_y, crop_scale = page_hand_crop(frame, corners)
            frame_index += 1
            timestamp_ms = int(round(1000.0 * frame_index / fps))
            result = detector.detect_for_video(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)), timestamp_ms)
            detections = []
            display_markers = []
            seen_hands = set()
            if result.hand_landmarks:
                for hand_i, landmarks in enumerate(result.hand_landmarks):
                    tip = landmarks[8]
                    # Convert normalized enlarged-crop coordinates back to the
                    # original full video frame before applying the homography.
                    frame_x = crop_x + tip.x * hand_image.shape[1] / crop_scale
                    frame_y = crop_y + tip.y * hand_image.shape[0] / crop_scale
                    p = np.float32([[[frame_x, frame_y]]])
                    mapped = cv2.perspectiveTransform(p, H_frame_to_map)[0, 0]
                    margin_x, margin_y = 0.05 * ref_w, 0.05 * ref_h
                    if (-margin_x <= mapped[0] < ref_w + margin_x and
                            -margin_y <= mapped[1] < ref_h + margin_y):
                        detections.append(mapped)
                        handedness = result.handedness[hand_i][0].category_name
                        display_markers.append(((int(round(frame_x)),
                                                 int(round(frame_y))),
                                                handedness))
                        raw_paths.setdefault(handedness, []).append(
                            tuple(np.rint(mapped).astype(int)))
                        seen_hands.add(handedness)
                        update_symbol_timer(symbol_timers, handedness,
                                            in_box(mapped, stairs_box), timestamp_ms)
            # A None creates a visible break rather than connecting across an
            # interval in which MediaPipe did not actually see that hand.
            for handedness, samples in raw_paths.items():
                if (handedness not in seen_hands and samples and
                        samples[-1] is not None):
                    samples.append(None)
            detection_count = len(detections)
            assign_detections(tracks, detections)

        corners = np.asarray(drag_pts, np.float32)
        H_map_to_frame = cv2.getPerspectiveTransform(ref_corners, corners)
        overlay = cv2.warpPerspective(ref_img, H_map_to_frame, (frame_w, frame_h))
        mask = cv2.warpPerspective(np.full((ref_h, ref_w), 255, np.uint8),
                                   H_map_to_frame, (frame_w, frame_h))
        shown = frame.copy()
        blended = cv2.addWeighted(shown, 0.65, overlay, 0.35, 0)
        shown[mask > 0] = blended[mask > 0]
        for p in corners:
            cv2.circle(shown, tuple(np.int32(p)), 7, (0, 255, 0), -1)
        cv2.putText(shown,
                    f"map: {pose.source}  matches: {pose.inliers}  error: {pose.error:.1f}px  "
                    f"coverage: {100 * pose.coverage:.1f}%  hands: {detection_count}  "
                    f"speed: {playback_speed:.2f}x",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (40, 255, 40), 2, cv2.LINE_AA)
        if is_paused:
            cv2.putText(shown,
                        "PAUSED: click/drag a corner; Space resumes",
                        (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2, cv2.LINE_AA)
        # Show only fingertips detected in this frame. Persisted/smoothed tracks
        # are for output traces and must not create a duplicate marker on one hand.
        for point, handedness in display_markers:
            color = (0, 0, 255) if handedness == "Left" else (255, 0, 0)
            cv2.circle(shown, point, 9, color, -1)
            cv2.putText(shown, handedness[0], (point[0] + 10, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                        cv2.LINE_AA)
        cv2.imshow("Video Tracker", shown)
        if is_paused:
            wait_ms = 10
        else:
            target_ms = 1000.0 / (fps * playback_speed)
            processing_ms = 1000.0 * (time.perf_counter() - loop_started)
            wait_ms = max(1, int(round(target_ms - processing_ms)))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord(" "):
            was_paused = is_paused
            is_paused = not is_paused
            if was_paused and not is_paused:
                pose.reset(frame, drag_pts)
        elif key in (ord("]"), ord("="), ord("+")):
            playback_speed = min(4.0, playback_speed + 0.25)
            print(f"Playback speed: {playback_speed:.2f}x")
        elif key in (ord("["), ord("-"), ord("_")):
            playback_speed = max(0.25, playback_speed - 0.25)
            print(f"Playback speed: {playback_speed:.2f}x")
        elif key == ord("q"):
            break

    final_canvas = ref_img.copy()
    colors = [(0, 0, 255), (255, 0, 0)]
    draw_trail(final_canvas, raw_paths.get("Left", []), colors[0])
    draw_trail(final_canvas, raw_paths.get("Right", []), colors[1])
    # The video overlay is intentionally rotated 180 degrees, but the saved
    # result should match the original PNG orientation.
    final_canvas = cv2.rotate(final_canvas, cv2.ROTATE_180)
    cv2.imwrite(output_path, final_canvas)
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="S-18 5-12-26 PT2 E.mp4")
    parser.add_argument("map", nargs="?", default="distractor_floorplan_E.png")
    parser.add_argument("-o", "--output", default="finger_paths.png")
    args = parser.parse_args()
    manual_keyframe_tracking(args.video, args.map, args.output)