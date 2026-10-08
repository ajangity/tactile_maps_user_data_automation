"""Auto-crop, part 2 (pipeline steps 9-17): turn edges into exact corners.

Step 9   weighted_points          -- points every 4 px along each edge
                                     segment, so long clean edges outweigh
                                     short stubs.
Step 10  fit_side_line            -- robust line of best fit per side.
Step 12  fit_quad                 -- extend the 4 side lines until they meet
                                     -> 4 corners (works even when a corner
                                     is hidden, as long as its 2 sides show).
Step 13  refine_with_traced_lines -- edge tracing improves the crop: line
                                     the PNG's printed lines up with the
                                     lines traced in the frame (ICP).
(Step 14, the sanity checks, is in crop_checker.py.)
Step 15  snap_corners             -- sub-pixel snap of clearly visible corners.
Step 16  keep_if_stationary       -- don't apply re-fit noise when the paper
                                     didn't move.
Step 17  pick_best / smooth       -- choose the best-scoring candidate and
                                     blend it with last frame's crop.

(The old step 11, reusing a side's line from an earlier frame when the side
isn't seen, is gone: it kept stale lines alive even once the real edge was
visible again. A frame where a side is hidden now relies on step 13
instead, which doesn't need the paper's edges at all.)
"""

import math

import cv2
import numpy as np

import edge_tracing
from paper_locator import SIDE_ANGLE_TOLERANCE, SIDES, classify_segments

MIN_SEGMENT_LENGTH = 25       # px; shorter Hough segments are creases/noise
HOUGH_THRESHOLD = 40
POINT_SPACING = 4.0           # px between sampled points along a segment
ICP_TOLERANCES = (14, 10, 7, 5, 4)   # map px; match radius shrinks each round
ICP_MAX_POINTS = 2500
ICP_REGION_MARGIN = 0.10      # only frame ink within the crop grown by 10%
FRAME_DOT_MAX = 9             # px; traced marks smaller than this in the frame are dots
SNAP_MAX_SHIFT = 8.0          # px; ignore a sub-pixel snap that moves further
STABLE_CORNER_TOLERANCE = 2.0 # px
MIN_STABLE_CORNERS = 2
SMOOTHING = 0.65              # share of the new crop vs last frame's (1 = no smoothing)


# ---------------------------------------------------------------- step 9

def weighted_points(x1, y1, x2, y2, angle_error):
    """Points every POINT_SPACING px along a segment. A segment whose angle
    only loosely matched its side gets fewer points (down to 15%), so a long
    but questionable segment can't overpower a short clean one."""
    weight = max(0.15, 1.0 - angle_error / SIDE_ANGLE_TOLERANCE)
    length = math.hypot(x2 - x1, y2 - y1)
    n = max(2, int(weight * length // POINT_SPACING) + 1)
    t = np.linspace(0.0, 1.0, n)
    return np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], axis=1)


# ---------------------------------------------------------------- step 10

def fit_side_line(points):
    """Robust infinite line through points -> (point_on_line, direction).

    Fits once, drops points that are clear outliers from that fit (beyond
    median + 4 * MAD of the perpendicular distances -- e.g. a segment that
    was put on the wrong side), then refits on the rest."""
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if len(pts) >= 6:
        flat = pts.reshape(-1, 2)
        off = flat - [x0, y0]
        perp = np.abs(off[:, 0] * vy - off[:, 1] * vx)
        med = np.median(perp)
        keep = perp <= med + 4.0 * (np.median(np.abs(perp - med)) + 1e-6)
        if 4 <= keep.sum() < len(flat):
            vx, vy, x0, y0 = cv2.fitLine(flat[keep].reshape(-1, 1, 2), cv2.DIST_HUBER,
                                         0, 0.01, 0.01).ravel()
    return np.float32([x0, y0]), np.float32([vx, vy])


# ---------------------------------------------------------------- step 12

def intersect(line_a, line_b):
    (p0, d0), (p1, d1) = line_a, line_b
    a = np.array([[d0[0], -d1[0]], [d0[1], -d1[1]]], np.float32)
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    t = np.linalg.solve(a, np.float32([p1[0] - p0[0], p1[1] - p0[1]]))[0]
    return p0 + t * d0


def fit_quad(crop_mask, x0, y0, prior):
    """Corners of one paper blob, using `prior` (last frame's crop, or the
    blob's rough outline on the first frame) to tell the sides apart.

    Returns (quad, visible): visible[i] is True when both sides meeting at
    corner i were actually seen this frame (a real, unhidden corner).
    Returns (None, None) unless all 4 sides were seen.
    """
    if crop_mask.shape[0] < 20 or crop_mask.shape[1] < 20:
        return None, None
    edges = cv2.Canny(crop_mask, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                          minLineLength=MIN_SEGMENT_LENGTH, maxLineGap=12)
    if raw is None or len(raw) < 4:
        return None, None
    segments = raw.reshape(-1, 4).astype(np.float32) + [x0, y0, x0, y0]
    by_side = classify_segments(segments, prior)          # step 8
    lines = {}
    for side in SIDES:
        if by_side[side]:
            pts = np.concatenate([weighted_points(*s) for s in by_side[side]])
            lines[side] = fit_side_line(pts)                 # steps 9-10
    if len(lines) < 4:
        return None, None
    corners = [intersect(lines["top"], lines["left"]),
               intersect(lines["top"], lines["right"]),
               intersect(lines["bottom"], lines["right"]),
               intersect(lines["bottom"], lines["left"])]
    if any(c is None for c in corners):
        return None, None
    return np.float32(corners), [True, True, True, True]


# ---------------------------------------------------------------- step 13

def refine_with_traced_lines(ref, frame_ink, H_frame_to_map, region_quad):
    """Fine-tune a crop so the PNG's printed lines sit exactly on the lines
    edge tracing found in the frame (iterative closest point).

    Each round: send the frame's traced ink pixels (only those inside the
    crop, grown by ICP_REGION_MARGIN) through the current crop into PNG
    space, pair each with the nearest printed pixel if it's within the
    round's radius, and re-fit the crop's homography to those pairs with
    RANSAC (which also ignores hand outlines and other strays). The radius
    shrinks each round (14 -> 4 map px) as the crop locks on.

    Uses every printed line on the map, not just the paper's border, so it
    works even with a corner or a whole side under a hand. Returns the
    refined frame -> map homography (or the input one if it can't improve).
    """
    q = np.asarray(region_quad, np.float32)
    centre = q.mean(axis=0)
    grown = np.int32(centre + (1 + ICP_REGION_MARGIN) * (q - centre))
    h, w = frame_ink.shape
    bx, by, bw, bh = cv2.boundingRect(grown)
    x0, y0, x1, y1 = max(0, bx), max(0, by), min(w, bx + bw), min(h, by + bh)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return H_frame_to_map
    region = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(region, grown - [x0, y0], 255)
    ink = cv2.bitwise_and(frame_ink[y0:y1, x0:x1], region)
    # Leave out the traced dots (tiny marks), matching the PNG side.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    big = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= FRAME_DOT_MAX
    big[0] = False
    ys, xs = np.nonzero(big[labels])
    xs, ys = xs + x0, ys + y0
    if len(xs) < 50:
        return H_frame_to_map
    if len(xs) > ICP_MAX_POINTS:
        pick = np.random.default_rng(0).choice(len(xs), ICP_MAX_POINTS, replace=False)
        xs, ys = xs[pick], ys[pick]
    pts = np.stack([xs, ys], 1).astype(np.float32)
    H = H_frame_to_map
    for radius in ICP_TOLERANCES:
        m = cv2.perspectiveTransform(pts[None], H)[0]
        inside = (m[:, 0] >= 0) & (m[:, 0] < ref.w) & (m[:, 1] >= 0) & (m[:, 1] < ref.h)
        xi = np.clip(np.rint(m[:, 0]).astype(int), 0, ref.w - 1)
        yi = np.clip(np.rint(m[:, 1]).astype(int), 0, ref.h - 1)
        keep = inside & (ref.ink_dist[yi, xi] <= radius)
        if keep.sum() < 30:
            break
        targets = ref.ink_xy[ref.ink_label[yi[keep], xi[keep]] - 1]
        H_new, _ = cv2.findHomography(pts[keep], targets, cv2.RANSAC, 3.0)
        if H_new is None:
            break
        H = H_new
    return H


SHIFT_SEARCH_MAX = 0.35       # shifts up to 35% of the map's size each way
SHIFT_SEARCH_PEAKS = 5


def shift_candidates(ref, frame_ink, H_frame_to_map):
    """Ways to slide a crop that has slipped off the real alignment.

    Warps the frame's traced walls and symbols into PNG space through the
    current crop, then cross-correlates them with the PNG's walls and
    symbols over every shift up to SHIFT_SEARCH_MAX (one FFT, ~50 ms).
    Returns the best few shifted homographies; each still needs step 13's
    refinement and the usual checks."""
    if not hasattr(ref, "_shift_fft"):
        printed = ((ref.trace.ink > 0) & (ref.trace.classes != edge_tracing.CLASS_DOT))
        printed = cv2.dilate(printed.astype(np.float32), np.ones((5, 5), np.uint8))
        ref._shift_pad = (2 * ref.h, 2 * ref.w)
        ref._shift_fft = np.fft.rfft2(printed, ref._shift_pad)
    warped = cv2.warpPerspective(frame_ink, H_frame_to_map, (ref.w, ref.h),
                                 flags=cv2.INTER_NEAREST)
    cls = edge_tracing.ink_classes(warped)
    seen = ((cls == edge_tracing.CLASS_WALL) | (cls == edge_tracing.CLASS_SYMBOL)).astype(np.float32)
    if seen.sum() < 50:
        return []
    seen = cv2.dilate(seen, np.ones((5, 5), np.uint8))
    pad = ref._shift_pad
    corr = np.fft.fftshift(np.fft.irfft2(ref._shift_fft * np.conj(np.fft.rfft2(seen, pad)), pad))
    cy, cx = pad[0] // 2, pad[1] // 2
    my, mx = int(SHIFT_SEARCH_MAX * ref.h), int(SHIFT_SEARCH_MAX * ref.w)
    window = corr[cy - my:cy + my + 1, cx - mx:cx + mx + 1].copy()
    floor = float(window.min())
    out = []
    for _ in range(SHIFT_SEARCH_PEAKS):
        y, x = np.unravel_index(np.argmax(window), window.shape)
        T = np.float64([[1, 0, x - mx], [0, 1, y - my], [0, 0, 1]])
        out.append(T @ H_frame_to_map)
        cv2.circle(window, (int(x), int(y)), 25, floor, -1)
    return out


def corners_from_homography(H_frame_to_map, ref):
    return cv2.perspectiveTransform(ref.corners[None], np.linalg.inv(H_frame_to_map))[0]


# ---------------------------------------------------------------- step 15

def snap_corners(frame_gray, quad, visible):
    """Sub-pixel snap (cv2.cornerSubPix) of corners whose 2 sides were both
    seen; ignored if the snap would move a corner more than SNAP_MAX_SHIFT
    px (it would be locking onto something else, like a fingernail)."""
    idx = [i for i, v in enumerate(visible or []) if v]
    if not idx:
        return quad
    pts = quad[idx].reshape(-1, 1, 2).astype(np.float32)
    try:
        refined = cv2.cornerSubPix(frame_gray, pts, (15, 15), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    except cv2.error:
        return quad
    out = quad.copy()
    for j, i in enumerate(idx):
        if np.linalg.norm(refined[j, 0] - quad[i]) < SNAP_MAX_SHIFT:
            out[i] = refined[j, 0]
    return out


# ---------------------------------------------------------------- step 16

def keep_if_stationary(quad, prior, trusted_corners=None):
    """If at least MIN_STABLE_CORNERS of the trusted corners landed within
    STABLE_CORNER_TOLERANCE px of last frame, the paper didn't move: return
    last frame's crop unchanged instead of this frame's re-fit noise."""
    trusted = trusted_corners if trusted_corners is not None else [True] * 4
    still = sum(1 for i in range(4) if trusted[i] and
                np.linalg.norm(quad[i] - prior[i]) <= STABLE_CORNER_TOLERANCE)
    return prior.copy() if still >= MIN_STABLE_CORNERS else quad


# ---------------------------------------------------------------- step 17

def pick_best(scored, min_score):
    """scored: list of (score, quad, info). Highest score at or above
    min_score wins; None if nothing qualifies."""
    good = [s for s in scored if s[0] >= min_score]
    return max(good, key=lambda s: s[0]) if good else None


def smooth(prior, new, amount=SMOOTHING):
    return (1.0 - amount) * np.asarray(prior, np.float32) + amount * np.asarray(new, np.float32)
