"""Crop checker: is a crop believable, and what to do when no crop is.

Step 14  check_quad      -- sanity checks every candidate crop must pass:
                            corners within 25 degrees of square, width/height
                            within 35% of the PNG's, big enough and not
                            hanging off the frame, and (while the lock is
                            trusted) no corner jumping more than 8% of the
                            frame's diagonal since last frame.
         outline_agreement -- and the crop's edges must sit on the paper's
                            real edges (paper inside, desk outside).
Step 18  PngBackup       -- Backup #1: SIFT-match the frame against the PNG
                            near the last crop and solve for the whole map.
Step 19  FlowBackup      -- Backup #2: follow hundreds of paper points from
                            last frame with optical flow.
Step 21  correct_outlier_corner / corner_visible
                         -- the paper is rigid: one corner can't jump while
                            the other 3 stay put; and which corners are
                            actually in view (green/orange dots).

Both backups must also pass step 14 and the alignment score (the caller
passes its score function in), so neither can drift onto a hand or the
wrong sheet unnoticed.
"""

import math

import cv2
import numpy as np

from paper_locator import paper_mask, PAPER_SAT_MAX

# Step 14
CORNER_ANGLE_TOLERANCE = 25.0    # degrees from 90
ASPECT_TOLERANCE = 0.35          # relative to the PNG's width/height
MIN_AREA_FRACTION = 0.03         # of the frame
EDGE_SLACK = 20                  # px a corner may sit outside the frame
MAX_JUMP = 0.08                  # fraction of the frame diagonal per frame
# Steps 18-19
MIN_MATCHES = 14
MAX_REPROJECTION_ERROR = 3.0     # px
MIN_INLIER_COVERAGE = 0.035      # of the map's area
BACKUP_MAX_JUMP = 0.12           # fraction of frame diagonal
BACKUP_MIN_SCORE = 0.35          # alignment score a backup crop must reach
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.65, 1.55   # PNG backup: crop area vs last frame's
# Step 21
OUTLIER_OTHERS_MAX = 6.0         # px the other 3 corners may move
OUTLIER_MIN_MOVE = 18.0          # px the odd corner must have moved
OUTLIER_RATIO = 3.0              # ...and this many times more than the others


# ======================================================================
# Step 14: sanity checks
# ======================================================================

def check_quad(quad, frame_shape, ref_aspect, prior=None, max_jump=None):
    """Return (ok, reason). reason names the first check that failed."""
    h, w = frame_shape[:2]
    p = np.asarray(quad, np.float32)
    if not np.all(np.isfinite(p)):
        return False, "not finite"
    if cv2.contourArea(p) < MIN_AREA_FRACTION * w * h:
        return False, "too small"
    if (np.any(p[:, 0] < -EDGE_SLACK) or np.any(p[:, 0] > w + EDGE_SLACK) or
            np.any(p[:, 1] < -EDGE_SLACK) or np.any(p[:, 1] > h + EDGE_SLACK)):
        return False, "off the frame"
    for i in range(4):
        v1, v2 = p[i - 1] - p[i], p[(i + 1) % 4] - p[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3:
            return False, "collapsed"
        angle = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
        if abs(angle - 90) > CORNER_ANGLE_TOLERANCE:
            return False, "corner not square"
    if ref_aspect is not None:
        width = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
        height = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
        if height < 1 or abs(width / height - ref_aspect) / ref_aspect > ASPECT_TOLERANCE:
            return False, "wrong aspect ratio"
    if prior is not None and max_jump is not None:
        if np.max(np.linalg.norm(p - np.asarray(prior, np.float32), axis=1)) > \
                max_jump * math.hypot(w, h):
            return False, "jumped too far"
    return True, "ok"


OUTLINE_MIN = 0.65               # outline_agreement a crop needs to be accepted
OUTLINE_OFFSET = 8               # px inside / outside the edge to compare
OUTLINE_CONTRAST = 25            # gray levels brighter inside than outside


def outline_agreement(gray, quad, samples_per_side=40):
    """Do the crop's 4 edges sit on the paper's real edges?

    At points along each edge, compares the image just inside the edge
    with just outside it: on a real paper edge, inside (paper, or a hand
    resting on it) is clearly brighter than outside (desk). Returns the
    share of points where that holds.

    This catches a failure the line score can't: on a floorplan with
    evenly sized rooms, a crop shifted by exactly one room still lines most
    walls up with *other* walls, and the dots still match, so it can score
    0.5-0.6 while being wrong. Measured on S-19: correct crops 0.81-0.82
    with hands on the page, crops stuck one room off 0.47-0.57.
    """
    q = np.asarray(quad, np.float32)
    centre = q.mean(axis=0)
    h, w = gray.shape[:2]
    good = total = 0
    t = np.linspace(0.05, 0.95, samples_per_side)[:, None]
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        normal = np.float32([-(b - a)[1], (b - a)[0]])
        normal /= np.linalg.norm(normal) + 1e-6
        if np.dot(centre - (a + b) / 2, normal) < 0:
            normal = -normal                                  # make it point inward
        pts = a + t * (b - a)
        pin = np.rint(pts + OUTLINE_OFFSET * normal).astype(int)
        pout = np.rint(pts - OUTLINE_OFFSET * normal).astype(int)
        valid = ((pin[:, 0] >= 0) & (pin[:, 0] < w) & (pin[:, 1] >= 0) & (pin[:, 1] < h) &
                 (pout[:, 0] >= 0) & (pout[:, 0] < w) & (pout[:, 1] >= 0) & (pout[:, 1] < h))
        if not valid.any():
            continue
        inside = gray[pin[valid, 1], pin[valid, 0]].astype(int)
        outside = gray[pout[valid, 1], pout[valid, 0]].astype(int)
        good += int(np.sum(inside - outside > OUTLINE_CONTRAST))
        total += int(valid.sum())
    return good / total if total else 0.0


# ======================================================================
# Step 18: Backup #1 -- SIFT against the PNG
# ======================================================================

class PngBackup:
    """Match SIFT features between the PNG and the frame (only near the
    last crop, which keeps hands, faces and the background out), and solve
    for the homography that carries the whole map into the frame."""

    def __init__(self, ref):
        self.ref = ref
        if hasattr(cv2, "SIFT_create"):
            self.feature = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.025,
                                           edgeThreshold=12)
            self.matcher, self.ratio = cv2.BFMatcher(cv2.NORM_L2), 0.74
        else:
            self.feature = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
            self.matcher, self.ratio = cv2.BFMatcher(cv2.NORM_HAMMING), 0.72
        self.ref_kp, self.ref_des = self.feature.detectAndCompute(ref.gray, None)
        self.stats = {"matches": 0, "error": float("inf"), "coverage": 0.0}

    def locate(self, frame_gray, prior, strict, score_fn):
        """Return (quad, score) or (None, 0)."""
        if self.ref_des is None or len(self.ref_kp) < MIN_MATCHES:
            return None, 0.0
        search = np.zeros(frame_gray.shape, np.uint8)
        c = prior.mean(axis=0)
        cv2.fillConvexPoly(search, np.int32(c + 1.12 * (prior - c)), 255)
        kp, des = self.feature.detectAndCompute(frame_gray, search)
        if des is None or len(kp) < MIN_MATCHES:
            return None, 0.0
        pairs = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < MIN_MATCHES:
            return None, 0.0
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 2.5,
                                     maxIters=10000, confidence=0.999)
        if H is None or mask is None:
            return None, 0.0
        inl = mask.ravel().astype(bool)
        self.stats["matches"] = int(inl.sum())
        if inl.sum() < MIN_MATCHES:
            return None, 0.0
        proj = cv2.perspectiveTransform(src[inl, None], H)[:, 0]
        self.stats["error"] = float(np.median(np.linalg.norm(proj - dst[inl], axis=1)))
        self.stats["coverage"] = cv2.contourArea(cv2.convexHull(src[inl])) / float(self.ref.w * self.ref.h)
        if (self.stats["error"] > MAX_REPROJECTION_ERROR or
                self.stats["coverage"] < MIN_INLIER_COVERAGE):
            return None, 0.0
        quad = cv2.perspectiveTransform(self.ref.corners[None], H)[0]
        ok, _ = check_quad(quad, frame_gray.shape, self.ref.aspect, prior,
                           BACKUP_MAX_JUMP if strict else None)
        if not ok:
            return None, 0.0
        if strict:
            # A trusted crop can't suddenly grow or shrink a lot either.
            ratio = abs(cv2.contourArea(quad)) / max(abs(cv2.contourArea(prior)), 1.0)
            if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
                return None, 0.0
        score = score_fn(quad)
        return (quad, score) if score >= BACKUP_MIN_SCORE else (None, score)


# ======================================================================
# Step 19: Backup #2 -- optical flow
# ======================================================================

class FlowBackup:
    """Follow up to 700 distinctive points on the paper from last frame to
    this one and move the crop the way they moved. Points are only seeded
    on paper-colored pixels inside the crop, so they don't sit on a hand.

    remember() just keeps last frame and its crop; the points are only
    picked (from that remembered frame) when this backup actually runs.
    Picking them every frame was ~30% of all per-frame time, for a backup
    that's needed on a few percent of frames."""

    def __init__(self):
        self.prev = None   # (frame_bgr, gray, corners)

    def remember(self, frame_bgr, gray, corners):
        self.prev = (frame_bgr, gray, np.asarray(corners, np.float32).copy())

    def _seed(self):
        frame_bgr, gray, corners = self.prev
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(corners), 255)
        mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
        x0, y0, w, h = cv2.boundingRect(np.int32(corners))
        x0, y0 = max(0, x0), max(0, y0)
        region = mask[y0:y0 + h, x0:x0 + w]
        region &= paper_mask(frame_bgr[y0:y0 + h, x0:x0 + w])
        return cv2.goodFeaturesToTrack(gray, 700, 0.008, 7, mask=mask, blockSize=7)

    def locate(self, gray, prior, ref_aspect, score_fn):
        """Return (quad, score) or (None, 0)."""
        if self.prev is None:
            return None, 0.0
        self.prev_gray = self.prev[1]
        self.points = self._seed()
        if self.points is None or len(self.points) < MIN_MATCHES:
            return None, 0.0
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None, winSize=(25, 25), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if nxt is None:
            return None, 0.0
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, nxt, None,
                                                winSize=(25, 25), maxLevel=3)
        if back is None:
            return None, 0.0
        fb = np.linalg.norm(self.points[:, 0] - back[:, 0], axis=1)
        keep = (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < 1.5)
        old, new = self.points[keep, 0], nxt[keep, 0]
        if len(old) < MIN_MATCHES:
            return None, 0.0
        delta, mask = cv2.findHomography(old, new, cv2.USAC_MAGSAC, 2.0,
                                         maxIters=5000, confidence=0.999)
        if delta is None or mask is None:
            return None, 0.0
        inl = mask.ravel().astype(bool)
        if inl.sum() < MIN_MATCHES:
            return None, 0.0
        err = float(np.median(np.linalg.norm(
            cv2.perspectiveTransform(old[inl, None], delta)[:, 0] - new[inl], axis=1)))
        hull = cv2.contourArea(cv2.convexHull(old[inl])) if inl.sum() >= 3 else 0.0
        if err >= 2.0 or hull / max(abs(cv2.contourArea(prior)), 1.0) <= 0.08:
            return None, 0.0
        quad = cv2.perspectiveTransform(prior[None], delta)[0]
        ok, _ = check_quad(quad, gray.shape, ref_aspect, prior, BACKUP_MAX_JUMP)
        if not ok:
            return None, 0.0
        score = score_fn(quad)
        return (quad, score) if score >= BACKUP_MIN_SCORE else (None, score)


# ======================================================================
# Step 21: rigid-paper check and corner visibility
# ======================================================================

def correct_outlier_corner(new, prior):
    """A sheet of paper is rigid: if 3 corners barely moved but the 4th
    jumped, the 4th is a measurement error. Replace it with where the
    motion of the other 3 says it should be."""
    new = np.asarray(new, np.float32)
    prior = np.asarray(prior, np.float32)
    moves = np.linalg.norm(new - prior, axis=1)
    odd = int(np.argmax(moves))
    others = [i for i in range(4) if i != odd]
    if not (moves[odd] >= OUTLIER_MIN_MOVE and
            moves[odd] >= OUTLIER_RATIO * (np.max(moves[others]) + 1e-3) and
            np.all(moves[others] <= OUTLIER_OTHERS_MAX)):
        return new
    M = cv2.getAffineTransform(prior[others], new[others])
    fixed = new.copy()
    fixed[odd] = M @ np.float32([prior[odd, 0], prior[odd, 1], 1.0])
    return fixed


def corner_visible(frame, point, window=24):
    """Is the paper actually visible at this corner right now (not under a
    hand)? A local brightness split around the point; the bright side must
    be paper-colored (low saturation), and both sides must really exist."""
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x - window), max(0, y - window), min(w, x + window), min(h, y + window)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return False
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    _, bright = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(bright > 0) < 20 or np.sum(bright == 0) < 20:
        return False
    return float(np.median(s[bright > 0])) <= PAPER_SAT_MAX
