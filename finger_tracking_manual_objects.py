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
MAX_UNCONFIRMED_FRAMES = 45      # ~1.5s @30fps of pure flow/hold before
                                 # distrusting the lock (it may have drifted
                                 # onto a hand or the wrong sheet) and
                                 # re-searching the whole frame
RECOVERY_RETRY_INTERVAL = 5      # frames between full-frame reacquisition
                                 # attempts while lost (bounds CPU cost)

STAIRS_BOX_FRAC = (0.05, 0.05, 0.95, 0.95)  # wide test box for tonight, shrink once we know real stairs coords

# Automatic paper-edge detection: segment the white page from the desk and
# hands via an Otsu-adaptive brightness split (see _paper_mask), fit a line
# to each visible side, and intersect adjacent lines to recover all 4
# corners even when one is hidden under a hand.
PAPER_SAT_MAX = 55        # HSV saturation ceiling for "paper" pixels; helps
                          # reject skin (measured ~28-41) even where its
                          # brightness rivals the paper's
EDGE_ROI_MARGIN = 0.35    # expand the search region beyond the prior quad
EDGE_MAX_FRAME_MOTION = 0.08  # fraction of frame diagonal; a shape-valid
                          # quad that jumped further than this from last
                          # frame is more likely a bad read than real motion
EDGE_MIN_SEGMENT_LEN = 25 # px, drop short Hough segments (creases, shadows)
EDGE_HOUGH_THRESHOLD = 40
EDGE_ANGLE_TOLERANCE = 20.0        # deg, segment-to-expected-side angle slop
EDGE_CORNER_ANGLE_TOLERANCE = 25.0 # deg, how far a corner may be from 90
EDGE_ASPECT_TOLERANCE = 0.35       # relative aspect-ratio slop vs reference
MIN_COLD_START_MATCHES = 20        # SIFT/ORB good matches needed to auto-lock

ui_mode = "align"
drag_pts = []
active_pt_idx = -1
is_paused = True
show_edge_debug = False


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


def _paper_mask(frame_bgr):
    """Binary mask of pixels that look like a bright printed page.

    A fixed brightness cutoff doesn't hold up: desk brightness varies with
    lighting/position (measured ~76-120 across one real frame) and can
    exceed any reasonable fixed threshold in bright spots, while paper stays
    reliably ~200+. Otsu's method re-derives the paper/background split from
    each frame's own histogram instead, which tracks that variation.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[s > PAPER_SAT_MAX] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def _isolate_nearest_component(mask, roi_bounds, prior_center):
    """Crop to roi_bounds, keeping only the connected paper-colored blob
    nearest prior_center -- a second sheet that has drifted into the same
    search window is a separate component and gets zeroed out, so its real,
    straight, paper-colored edges can never leak into this frame's line fit.
    """
    x0, y0, x1, y1 = roi_bounds
    crop = mask[y0:y1, x0:x1]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        crop, connectivity=8)
    if num <= 1:
        return None
    local_center = np.asarray(prior_center, np.float32) - [x0, y0]
    best_label, best_dist = None, None
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] < 400:
            continue
        dist = np.linalg.norm(centroids[lbl] - local_center)
        if best_dist is None or dist < best_dist:
            best_label, best_dist = lbl, dist
    if best_label is None:
        return None
    return np.where(labels == best_label, np.uint8(255), np.uint8(0))


def _line_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _side_expected_angles(corners):
    tl, tr, br, bl = corners
    return {
        "top": _line_angle(tl[0], tl[1], tr[0], tr[1]),
        "bottom": _line_angle(bl[0], bl[1], br[0], br[1]),
        "left": _line_angle(tl[0], tl[1], bl[0], bl[1]),
        "right": _line_angle(tr[0], tr[1], br[0], br[1]),
    }


def _side_center(corners, side):
    tl, tr, br, bl = corners
    pairs = {"top": (tl, tr), "bottom": (bl, br), "left": (tl, bl), "right": (tr, br)}
    a, b = pairs[side]
    return (np.asarray(a, np.float32) + np.asarray(b, np.float32)) / 2.0


def _fit_line_through_points(points):
    """Robust infinite-line fit; returns (point_on_line, unit_direction)."""
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)


def _intersect_lines(line_a, line_b):
    """Intersection of two infinite lines given as (point, direction)."""
    (p0, d0), (p1, d1) = line_a, line_b
    a = np.array([[d0[0], -d1[0]], [d0[1], -d1[1]]], np.float32)
    b = np.array([p1[0] - p0[0], p1[1] - p0[1]], np.float32)
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    t = np.linalg.solve(a, b)[0]
    return p0 + t * d0


def _sample_segment_points(x1, y1, x2, y2, step=4.0):
    """Points evenly spaced along a segment, not just its 2 endpoints.

    A straight line is fully determined by 2 points, so this adds no new
    information for a single segment in isolation -- what it does is give a
    long, reliable segment proportionally more points (and so more say) than
    a short, noisy one when multiple segments are pooled and fit together.
    Without it, a clean 200px edge and a noisy 25px stub used to count
    equally (2 points each), which is backwards.
    """
    length = math.hypot(x2 - x1, y2 - y1)
    n = max(2, int(length // step) + 1)
    t = np.linspace(0.0, 1.0, n)
    return list(zip(x1 + t * (x2 - x1), y1 + t * (y2 - y1)))


def _classify_with_prior(segments, expected, centers):
    """Assign each Hough segment to a side using the last known quad as a
    prior: match by angle first, then by which side's center it's nearest."""
    sides = {"top": [], "bottom": [], "left": [], "right": []}
    for x1, y1, x2, y2 in segments:
        angle = _line_angle(x1, y1, x2, y2)
        mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], np.float32)
        best_side, best_dist = None, None
        for side, exp_angle in expected.items():
            if _angle_diff(angle, exp_angle) > EDGE_ANGLE_TOLERANCE:
                continue
            dist = float(np.linalg.norm(mid - centers[side]))
            if best_dist is None or dist < best_dist:
                best_side, best_dist = side, dist
        if best_side is not None:
            sides[best_side].extend(_sample_segment_points(x1, y1, x2, y2))
    return sides


def _order_quad_points(pts):
    """Label 4 unordered points as TL/TR/BR/BL by position (sum/diff trick):
    TL has the smallest x+y, BR the largest; TR has the smallest y-x, BL the
    largest. Absolute top/bottom is still arbitrary with no prior -- this
    just gives _classify_with_prior a consistent, geometrically sane quad to
    refine, with the real top/bottom/left/right decided later by content."""
    p = np.asarray(pts, np.float32)
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)],
                     p[np.argmax(s)], p[np.argmax(d)]], np.float32)


def _quad_relabelings(quad):
    """The 4 ways to relabel a quad's TL/TR/BR/BL corners that a cold-start
    fit (no prior to say which visible side is really 'top') could have
    produced: as-is, top/bottom swapped, left/right swapped, and both."""
    q = np.asarray(quad, np.float32)
    return [q[[0, 1, 2, 3]], q[[3, 2, 1, 0]], q[[1, 0, 3, 2]], q[[2, 3, 0, 1]]]


def _valid_quad_geometry(quad, w, h, ref_aspect=None):
    """Beyond valid_quad's area/bounds check: corners must be roughly square
    and, if a reference aspect ratio is known, the quad must roughly match it."""
    p = np.asarray(quad, np.float32)
    if not valid_quad(p, w, h):
        return False
    for i in range(4):
        a, b, c = p[(i - 1) % 4], p[i], p[(i + 1) % 4]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            return False
        cos_ang = np.dot(v1, v2) / (n1 * n2)
        ang = math.degrees(math.acos(np.clip(cos_ang, -1.0, 1.0)))
        if abs(ang - 90) > EDGE_CORNER_ANGLE_TOLERANCE:
            return False
    if ref_aspect is not None:
        w_top, w_bot = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3])
        h_left, h_right = np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1])
        quad_h = (h_left + h_right) / 2.0
        if quad_h < 1.0:
            return False
        aspect = ((w_top + w_bot) / 2.0) / quad_h
        if abs(aspect - ref_aspect) / ref_aspect > EDGE_ASPECT_TOLERANCE:
            return False
    return True


def _refine_corners_subpix(frame, quad, refine_mask):
    """Sub-pixel-snap the selected corners onto the real image using
    cv2.cornerSubPix, which iteratively finds the point whose neighborhood
    gradients are most orthogonal to the vectors pointing at it -- the
    standard OpenCV tool for turning an approximate corner into a precise
    one. Guarded by a small max-shift: cornerSubPix converges to *some*
    nearby corner-like feature, and if our estimate was already off by more
    than a few pixels it's safer to trust the line intersection than risk
    snapping onto an unrelated feature (a fingernail, a desk seam)."""
    idx = [i for i, keep in enumerate(refine_mask) if keep]
    if not idx:
        return quad
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = quad[idx].reshape(-1, 1, 2).astype(np.float32)
    try:
        refined = cv2.cornerSubPix(
            gray, pts, (15, 15), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    except cv2.error:
        return quad
    out = quad.copy()
    for j, i in enumerate(idx):
        if np.linalg.norm(refined[j, 0] - quad[i]) < 8.0:
            out[i] = refined[j, 0]
    return out


class PaperEdgeDetector:
    """Locate the physical page's 4 corners from its visible edges alone.

    Segments the page from the desk/hands by color, fits a line to each
    visible side with Hough transform + robust line fitting, then intersects
    adjacent lines. A corner hidden under a hand is still recovered because
    the two sides that meet there are each defined by their own visible
    portion elsewhere along the edge.
    """

    def __init__(self, ref_aspect):
        self.ref_aspect = ref_aspect
        self.last_valid = {}  # side -> (point, direction), carried across frames

    def track(self, frame, prior_corners, strict=True):
        """Per-frame update, biased toward the previously known quad.

        strict=False skips the jump-from-last-frame sanity check -- pass it
        when prior_corners itself isn't yet trusted (e.g. still recovering
        from drift), so a correct detection can immediately snap to the true
        position instead of being rejected for disagreeing with a baseline
        that was already wrong.
        """
        h, w = frame.shape[:2]
        mask = _paper_mask(frame)
        p = np.asarray(prior_corners, np.float32)
        centre = p.mean(axis=0)
        expanded = centre + (1.0 + EDGE_ROI_MARGIN) * (p - centre)
        x0, y0 = np.floor(expanded.min(axis=0)).astype(int)
        x1, y1 = np.ceil(expanded.max(axis=0)).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        # The search window can contain a second, separate sheet of paper --
        # isolate just the blob nearest our last known position so its edges
        # (real, straight, paper-colored) never leak into this frame's fit.
        crop_mask = _isolate_nearest_component(mask, (x0, y0, x1, y1), centre)
        if crop_mask is None:
            return None
        quad, lines = self._fit_quad(frame, crop_mask, x0, y0, prior_corners,
                                     self.last_valid)
        if quad is None:
            return None
        # A shape-valid quad can still be positionally wrong (motion blur, a
        # stray shadow briefly read as an edge) -- reject an implausible jump
        # from last frame rather than let it flash through for one frame.
        step = np.linalg.norm(quad - p, axis=1)
        if strict and np.max(step) > EDGE_MAX_FRAME_MOTION * math.hypot(w, h):
            return None
        self.last_valid = lines
        return quad

    def find_candidate_quads(self, frame):
        """Cold-start search: every sufficiently large paper-colored blob.

        There's no previous frame to say which visible side is really the
        top, so each blob's rough orientation comes from cv2.minAreaRect
        instead of a from-scratch line classifier -- then that rough quad is
        fed as a *prior* into the same angle+position side classifier
        _fit_quad already uses for per-frame tracking, so line fitting is as
        robust here as it is once locked on. The resulting quad is expanded
        into all 4 axis relabelings (top/bottom and left/right are still an
        arbitrary choice at this point) so the caller can pick the one
        that's actually right-side up, e.g. via feature matching.
        """
        h, w = frame.shape[:2]
        mask = _paper_mask(frame)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_area = w * h
        candidates = []
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < 0.03 * frame_area:
                continue
            blob = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            rough_quad = _order_quad_points(
                cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea))))
            x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            pad_x, pad_y = int(0.15 * bw) + 5, int(0.15 * bh) + 5
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
            # blob, not mask: keep this candidate isolated from any other
            # paper-colored blob whose padded bbox happens to overlap here.
            quad, _ = self._fit_quad(frame, blob[y0:y1, x0:x1], x0, y0,
                                     rough_quad, {})
            if quad is not None:
                candidates.extend(_quad_relabelings(quad))
        return candidates

    def reset(self):
        self.last_valid = {}

    def _fit_quad(self, frame, crop_mask, x0, y0, prior, last_valid):
        if crop_mask.shape[1] < 20 or crop_mask.shape[0] < 20:
            return None, {}
        edges = cv2.Canny(crop_mask, 50, 150)
        segments = cv2.HoughLinesP(edges, 1, np.pi / 180, EDGE_HOUGH_THRESHOLD,
                                    minLineLength=EDGE_MIN_SEGMENT_LEN, maxLineGap=12)
        if segments is None or len(segments) < 4:
            return None, {}
        segments = segments.reshape(-1, 4).astype(np.float32)
        segments[:, [0, 2]] += x0
        segments[:, [1, 3]] += y0

        prior = np.asarray(prior, np.float32)
        expected = _side_expected_angles(prior)
        centers = {s: _side_center(prior, s) for s in expected}
        sides = _classify_with_prior(segments, expected, centers)

        lines = {}
        fresh = {}
        for side, pts in sides.items():
            if len(pts) >= 2:
                lines[side] = _fit_line_through_points(pts)
                fresh[side] = True
            elif side in last_valid:
                lines[side] = last_valid[side]
                fresh[side] = False
        if len(lines) < 4:
            return None, {}

        try:
            tl = _intersect_lines(lines["top"], lines["left"])
            tr = _intersect_lines(lines["top"], lines["right"])
            br = _intersect_lines(lines["bottom"], lines["right"])
            bl = _intersect_lines(lines["bottom"], lines["left"])
        except np.linalg.LinAlgError:
            return None, {}
        if any(pt is None for pt in (tl, tr, br, bl)):
            return None, {}
        quad = np.array([tl, tr, br, bl], np.float32)
        h, w = frame.shape[:2]
        if not _valid_quad_geometry(quad, w, h, self.ref_aspect):
            return None, {}

        # A corner whose both adjacent sides were actually seen this frame is
        # a real, unoccluded corner -- cv2.cornerSubPix can snap the
        # line-intersection estimate to the true corner using local image
        # gradients. A corner with a fallback side is, by definition, hidden
        # this frame; there's no real corner in the image there to refine
        # onto, so leave that estimate as the (still fully valid) extrapolation.
        corner_sides = (("top", "left"), ("top", "right"),
                        ("bottom", "right"), ("bottom", "left"))
        refine = [fresh.get(a, False) and fresh.get(b, False)
                 for a, b in corner_sides]
        quad = _refine_corners_subpix(frame, quad, refine)
        return quad, lines


def build_feature_matcher(reference_gray):
    """Shared SIFT/ORB setup used both for continuous page tracking and for
    scoring cold-start corner candidates."""
    if hasattr(cv2, "SIFT_create"):
        feature = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.025,
                                  edgeThreshold=12)
        norm = cv2.NORM_L2
        ratio = 0.74
    else:
        feature = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
        norm = cv2.NORM_HAMMING
        ratio = 0.72
    ref_kp, ref_des = feature.detectAndCompute(reference_gray, None)
    matcher = cv2.BFMatcher(norm)
    return feature, matcher, ratio, ref_kp, ref_des


def score_candidate_quad(frame_gray, quad, ref_gray, ref_corners, feature, matcher,
                         ratio, ref_des):
    """Warp a candidate quad back to map space and count good feature matches
    against the reference map -- distinguishes the real floorplan sheet from
    a second blank/lightly-printed sheet (e.g. a legend page) in frame."""
    if ref_des is None:
        return 0
    ref_h, ref_w = ref_gray.shape
    H = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), ref_corners)
    warped = cv2.warpPerspective(frame_gray, H, (ref_w, ref_h))
    kp, des = feature.detectAndCompute(warped, None)
    if des is None or len(kp) < 8:
        return 0
    pairs = matcher.knnMatch(ref_des, des, k=2)
    good = [pr[0] for pr in pairs if len(pr) == 2 and pr[0].distance < ratio * pr[1].distance]
    return len(good)


def auto_locate_page(frame, ref_gray, ref_corners, ref_aspect):
    """Try to find the map's 4 corners automatically, before falling back to
    manual alignment. Returns a quad (TL, TR, BR, BL) or None."""
    detector = PaperEdgeDetector(ref_aspect)
    candidates = detector.find_candidate_quads(frame)
    if not candidates:
        return None
    feature, matcher, ratio, _, ref_des = build_feature_matcher(ref_gray)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_quad, best_score = None, 0
    for quad in candidates:
        score = score_candidate_quad(frame_gray, quad, ref_gray, ref_corners,
                                     feature, matcher, ratio, ref_des)
        if score > best_score:
            best_quad, best_score = quad, score
    if best_score < MIN_COLD_START_MATCHES:
        return None
    return best_quad


class PagePose:
    """Register the original map directly to each frame (no accumulated drift)."""
    def __init__(self, reference, frame, corners):
        self.reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.ref_h, self.ref_w = self.reference_gray.shape
        self.ref_corners = np.float32([[0, 0], [self.ref_w - 1, 0],
                                       [self.ref_w - 1, self.ref_h - 1],
                                       [0, self.ref_h - 1]])
        self.ref_aspect = self.ref_w / float(self.ref_h)
        self.corners = np.asarray(corners, np.float32)
        # SIFT is much more stable than CSRT/ORB under perspective, scale and glare.
        (self.feature, self.matcher, self.ratio,
         self.ref_kp, self.ref_des) = build_feature_matcher(self.reference_gray)
        self.edge_detector = PaperEdgeDetector(self.ref_aspect)
        self.inliers = 0
        self.error = float("inf")
        self.coverage = 0.0
        self.source = "initial"
        self.frames_since_confirmed = 0
        self.recovery_tick = 0
        self.lost = False
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
        self.edge_detector.reset()
        self.source = "manual"
        self.frames_since_confirmed = 0
        self.recovery_tick = 0
        self.lost = False
        self.inliers = 0
        self.error = 0.0
        self.coverage = 0.0

    def _reference_update(self, frame, strict=True):
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
        geometry_ok = (valid_quad(candidate, w, h) and
                      self.error <= MAX_REPROJECTION_ERROR and
                      self.coverage >= MIN_INLIER_COVERAGE)
        # Only demand agreement with our current corners when they're
        # already trusted -- see the comment on the `trusted` flag in
        # update(). A confident SIFT match shouldn't be discarded just for
        # correctly disagreeing with a position we already suspect is wrong.
        motion_ok = (not strict) or (np.max(step) <= max_step and
                                     0.65 <= area_ratio <= 1.55)
        accepted = geometry_ok and motion_ok
        if accepted:
            self.corners = ((1.0 - PAGE_POSE_SMOOTHING) * self.corners +
                            PAGE_POSE_SMOOTHING * candidate)
            self.source = "PNG"
        return self.corners

    def _note_confirmed(self):
        """edge/PNG both check the frame against real page content (Hough
        lines against the true page brightness, or SIFT against the actual
        reference image) -- either one succeeding means the lock is real."""
        self.frames_since_confirmed = 0
        self.recovery_tick = 0
        self.lost = False

    def _note_unconfirmed(self):
        """Optical flow / holding stale corners has no ground truth check --
        it will happily keep 'tracking' a hand or the wrong sheet forever.
        Past MAX_UNCONFIRMED_FRAMES of that, treat the lock as suspect."""
        self.frames_since_confirmed += 1
        self.recovery_tick += 1
        if self.frames_since_confirmed >= MAX_UNCONFIRMED_FRAMES:
            self.lost = True

    def _attempt_recovery(self, frame):
        """Full-frame reacquisition for when tracking has likely drifted onto
        the wrong object. Unlike the per-frame chain (which only searches
        near the stale corners), this searches the whole frame again, same
        as the initial cold start, so it can find the real page wherever it
        actually is."""
        candidates = self.edge_detector.find_candidate_quads(frame)
        if not candidates:
            return None
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_quad, best_score = None, 0
        for quad in candidates:
            score = score_candidate_quad(frame_gray, quad, self.reference_gray,
                                         self.ref_corners, self.feature,
                                         self.matcher, self.ratio, self.ref_des)
            if score > best_score:
                best_quad, best_score = quad, score
        if best_score < MIN_COLD_START_MATCHES:
            return None
        return best_quad

    def update(self, frame):
        """Prefer automatic edge detection; fall back to PNG registration or
        dense flow when the page's visible edges aren't enough this frame.
        If neither has confirmed the lock in a while, periodically re-search
        the whole frame rather than trust indefinite optical-flow drift."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        before = self.corners.copy()
        self.source = "held"
        # Only require a fresh detection to agree with our current corners
        # when those corners are themselves already trusted. Otherwise a
        # correct detection that jumps away from an already-drifted position
        # -- the exact case where we most need it to win -- would get
        # rejected for "disagreeing" with a baseline that was wrong to begin
        # with, and tracking could never self-correct.
        trusted = self.frames_since_confirmed == 0

        edge_quad = self.edge_detector.track(frame, self.corners, strict=trusted)
        if edge_quad is not None:
            self.corners = ((1.0 - PAGE_POSE_SMOOTHING) * before +
                            PAGE_POSE_SMOOTHING * edge_quad)
            self.source = "edge"
            self.prev_gray = gray
            self.flow_points = self._seed_flow(gray)
            self._note_confirmed()
            return self.corners

        corners = self._fallback_track(frame, gray, before, strict=trusted)
        if self.source == "PNG":
            self._note_confirmed()
            return corners

        self._note_unconfirmed()
        if self.lost and self.recovery_tick % RECOVERY_RETRY_INTERVAL == 0:
            recovered = self._attempt_recovery(frame)
            if recovered is not None:
                self.corners = recovered
                self.edge_detector.reset()
                self.flow_points = self._seed_flow(gray)
                self.source = "recovered"
                self._note_confirmed()
                return self.corners
        if self.lost:
            self.source = "lost"
        return corners

    def _fallback_track(self, frame, gray, before, strict):
        """PNG (SIFT) registration, else robust dense optical flow. Unchanged
        from before edge detection existed -- update() now only reaches this
        when edge detection can't see enough of the page's sides this frame."""
        self._reference_update(frame, strict=strict)
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


def draw_edge_debug(shown, frame, corners, frame_w, frame_h):
    """Overlay the paper mask and side-classified Hough segments so the
    automatic corner detector's thresholds can be tuned against real footage."""
    mask = _paper_mask(frame)
    p = np.asarray(corners, np.float32)
    centre = p.mean(axis=0)
    expanded = centre + (1.0 + EDGE_ROI_MARGIN) * (p - centre)
    x0, y0 = np.floor(expanded.min(axis=0)).astype(int)
    x1, y1 = np.ceil(expanded.max(axis=0)).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_w, x1), min(frame_h, y1)
    mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(shown, 0.7, mask_overlay, 0.3, 0, dst=shown)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return
    crop_mask = mask[y0:y1, x0:x1]
    edges = cv2.Canny(crop_mask, 50, 150)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, EDGE_HOUGH_THRESHOLD,
                               minLineLength=EDGE_MIN_SEGMENT_LEN, maxLineGap=12)
    if segments is None:
        return
    expected = _side_expected_angles(p)
    centers = {s: _side_center(p, s) for s in expected}
    side_colors = {"top": (0, 0, 255), "bottom": (255, 0, 0),
                  "left": (0, 255, 0), "right": (0, 255, 255)}
    for xs1, ys1, xs2, ys2 in segments.reshape(-1, 4):
        xs1, xs2 = xs1 + x0, xs2 + x0
        ys1, ys2 = ys1 + y0, ys2 + y0
        angle = _line_angle(xs1, ys1, xs2, ys2)
        mid = np.array([(xs1 + xs2) / 2.0, (ys1 + ys2) / 2.0], np.float32)
        best_side, best_dist = None, None
        for side, exp_angle in expected.items():
            if _angle_diff(angle, exp_angle) > EDGE_ANGLE_TOLERANCE:
                continue
            dist = float(np.linalg.norm(mid - centers[side]))
            if best_dist is None or dist < best_dist:
                best_side, best_dist = side, dist
        color = side_colors.get(best_side, (200, 200, 200))
        cv2.line(shown, (int(xs1), int(ys1)), (int(xs2), int(ys2)), color, 2)
    cv2.putText(shown, "Edge debug (key 'e' to toggle): red=top blue=bottom "
               "green=left yellow=right", (18, frame_h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


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
    global drag_pts, is_paused, ui_mode, show_edge_debug
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
    ref_gray_for_lookup = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    auto_quad = auto_locate_page(frame, ref_gray_for_lookup, ref_corners,
                                 ref_w / float(ref_h))
    if auto_quad is not None:
        drag_pts = auto_quad.tolist()
        print("Automatically located the page corners from its visible edges. "
              "Press Enter to accept, or drag a corner to correct it first.")
    else:
        drag_pts = [[100, 100], [frame_w - 100, 100],
                    [frame_w - 100, frame_h - 100], [100, frame_h - 100]]
        print("Could not automatically locate the page. Click or drag near each "
              "map corner; the nearest green handle will follow. Press Enter "
              "when aligned.")

    # AUTOSIZE keeps mouse coordinates in the same pixel coordinate system as
    # the video frame, which is essential for an accurate homography.
    cv2.namedWindow("Video Tracker", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Video Tracker", mouse_handler)
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
        align_msg = ("Auto-located: press Enter to accept, or drag a corner to fix"
                    if auto_quad is not None else
                    "Click/drag each map corner, then press Enter")
        cv2.putText(shown, align_msg, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
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
    print("Tracking. Space pauses/resumes; e toggles the edge-detection debug "
          "overlay; q saves and quits.")
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
        if show_edge_debug:
            draw_edge_debug(shown, frame, corners, frame_w, frame_h)
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
        elif key == ord("e"):
            show_edge_debug = not show_edge_debug
            print(f"Edge debug overlay: {'on' if show_edge_debug else 'off'}")
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