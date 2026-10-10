"""Auto-crop, part 1 (pipeline steps 1-8): know the map, find the paper.

Step 1  load_reference_map    -- read the PNG and write its map JSON: every
                                 black line as a graph (nodes + edges), the
                                 rooms, the symbols.
Step 2  paper_mask            -- bright, low-color pixels = paper.
Step 3  paper_blob_quads      -- rough 4-corner outline of each paper blob
        traced_map_quads      -- ...and of each cluster of printed walls
                                 found by edge tracing (works even when the
                                 paper's own outline is broken by a hand or
                                 merged with another sheet).
Step 4  orientations          -- the 4 ways to say which corner is the
                                 map's top-left, for one outline.
        alignment_score       -- the scoring rubric: how well the frame's
                                 traced lines match the PNG's lines for a
                                 given crop. Picks the orientation, rejects
                                 the wrong sheet, and grades every crop.
Step 6  search_window         -- after the first lock, only look near the
                                 last crop.
Step 7  candidate_blobs       -- every separate paper blob in that window.
Step 8  classify_segments     -- sort a blob's straight edges into top /
                                 bottom / left / right using last frame's crop.

Steps 9-17 (turning those edges into corners) are in corner_fitting.py.
"""

import json
import math
import os

import cv2
import numpy as np

import edge_tracing
import symbol_identification
from room_tracking import RoomMap, detect_rooms, detect_symbols
from units import PixelsToInches

# Step 2
PAPER_SAT_MAX = 55        # HSV saturation ceiling for "paper"; skin measured ~28-41+
# Step 3
MIN_BLOB_FRACTION = 0.03  # of the frame; smaller bright blobs aren't a sheet
TRACED_WALL_MIN = 40      # px in the frame; traced marks this long count as walls
# Step 4
SCORE_TOLERANCE = 5.0     # map px; a traced line this close to a printed line "matches"
CLASS_WEIGHTS = {edge_tracing.CLASS_WALL: 0.4, edge_tracing.CLASS_SYMBOL: 0.4,
                 edge_tracing.CLASS_DOT: 0.2}
# Step 6
SEARCH_MARGIN = 0.35      # grow last frame's crop by 35%. Tested 20% on S-19 (20-50 s):
                          # 3x more frames where the crop couldn't be confirmed, and
                          # slower overall (lost frames trigger whole-frame searches).
                          # Not about motion -- 99.9% of frames move < 2.4% of the
                          # paper's diagonal -- but the paper/desk brightness split is
                          # computed inside this window, and more desk in view keeps it
                          # stable when hands cover the page. Smaller saves ~3 ms/frame.
# Step 7
MIN_CANDIDATE_AREA = 400  # px
MAX_CANDIDATES = 3
# Step 8
SIDE_ANGLE_TOLERANCE = 20.0   # degrees between a segment and the side it's assigned to


# ======================================================================
# Step 1: the reference map and its JSON
# ======================================================================

class ReferenceMap:
    """The PNG plus everything derived from it once, up front.

    Coordinates everywhere in this project are the PNG's own pixels, in the
    PNG's own orientation: (0, 0) is its top-left corner. (The old 180-degree
    pre-rotation is gone: orientation is now worked out per video by step 4,
    so no fixed assumption about how the paper sits is needed.)
    """

    def __init__(self, path, json_path=None, names_from=None):
        """json_path: where to write the map JSON (default: next to the PNG).
        names_from: an earlier map JSON whose hand-edited room/symbol names
        should be kept (default: json_path itself, if it already exists)."""
        self.path = path
        self.image = edge_tracing.load_image(path)   # transparent PNGs onto white
        if self.image is None:
            raise FileNotFoundError(path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.gray.shape
        self.aspect = self.w / float(self.h)
        self.corners = np.float32([[0, 0], [self.w - 1, 0],
                                   [self.w - 1, self.h - 1], [0, self.h - 1]])

        self.trace = edge_tracing.Trace(self.gray)
        label_image, rooms, gap, _ = detect_rooms(self.trace.walls)
        symbols = symbol_identification.detect_symbols(self.trace, label_image, path)
        self.json_path = json_path or os.path.splitext(path)[0] + ".map.json"
        rooms, symbols = _keep_custom_names(names_from or self.json_path, rooms, symbols)
        symbol_identification.restore_auto_names(names_from or self.json_path, symbols)
        self.inches = PixelsToInches(self.w, self.h)
        for room in rooms:   # (width, height) of the room's bounding box on the paper
            room["size_in"] = list(self.inches.bbox_size(room["bbox"]))
        self.rooms = RoomMap(self.w, self.h, rooms, symbols)
        ys, xs = np.nonzero(self.trace.walls)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        # The printed outer border: where the wall network's outline sits on
        # the page. Used to turn a traced border in the frame into a crop.
        self.border = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        self.data = self.trace.to_json(
            image=os.path.basename(path),
            border=self.border.tolist(),
            room_gap_px=int(gap),
            **self.inches.to_json(),
            rooms=rooms,
            symbols=symbols,
        )
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        self._prepare_scoring()

    def _prepare_scoring(self):
        """Lookup tables for alignment_score (step 4) and the traced-line
        refinement (step 13): distance from every map pixel to the nearest
        printed line of each class, and which printed pixel that is."""
        self.class_present = {}
        self.class_dist = {}
        for c in CLASS_WEIGHTS:
            here = self.trace.classes == c
            self.class_present[c] = here
            self.class_dist[c] = cv2.distanceTransform(
                np.where(here, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
        # Step 13 only lines up walls and symbols, never the corridor's dots:
        # the dots repeat every few pixels, so a crop shifted by a few dot
        # spacings still "fits" them, and a fit that leans on them can slide
        # sideways when hands cover the walls.
        structure = np.where((self.trace.ink > 0) &
                             (self.trace.classes != edge_tracing.CLASS_DOT), 255, 0).astype(np.uint8)
        self.ink_dist, self.ink_label = cv2.distanceTransformWithLabels(
            255 - structure, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        zy, zx = np.nonzero(structure == 255)   # same raster order as the labels
        self.ink_xy = np.stack([zx, zy], 1).astype(np.float32)


def _keep_custom_names(json_path, rooms, symbols):
    """If a previous map JSON had rooms/symbols renamed by hand, keep those
    names (matched by id) when the JSON is regenerated."""
    if not json_path or not os.path.exists(json_path):
        return rooms, symbols
    try:
        with open(json_path, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        return rooms, symbols
    for key, new_list in (("rooms", rooms), ("symbols", symbols)):
        old_list = old.get(key) or []
        if len(old_list) == len(new_list):
            for new, prev in zip(new_list, old_list):
                if prev.get("id") == new["id"] and prev.get("name"):
                    new["name"] = prev["name"]
    return rooms, symbols


# ======================================================================
# Step 2: which pixels are paper
# ======================================================================

def paper_mask(frame_bgr):
    """Binary mask of pixels that look like a bright printed page.

    Otsu re-derives the paper/background brightness split from each frame's
    own histogram (desk brightness varies a lot with lighting). Bright but
    colorful pixels are thrown out too -- skin can be nearly as bright as
    paper but is noticeably more saturated -- so a hand between two sheets
    can't bridge them into one blob. That saturation cutoff can only get
    tighter than PAPER_SAT_MAX, never looser.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_max = PAPER_SAT_MAX
    bright_sat = s[mask > 0]
    if bright_sat.size > 200:
        sat_max = min(PAPER_SAT_MAX, float(cv2.threshold(
            bright_sat.reshape(-1, 1), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))
    mask[s > sat_max] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


INK_BORDER_THRESHOLD = 180   # gray level; darker = ink or desk


def ink_border_mask(frame_bgr):
    """Fallback for paper_mask (from Vivaan's ink-border auto-crop): mark
    everything darker than the paper -- the printed lines *and* the desk --
    so the paper reads as the bright region inside it. Its edges come from
    the sharp printed border and the paper/desk boundary rather than from
    paper_mask's brightness+color split, which fails when the desk is nearly
    as bright as the page or a hand bridges two sheets. Fed into the exact
    same edge-fitting (corner_fitting.fit_quad) when paper_mask's blobs
    don't give a usable outline."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, INK_BORDER_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return mask


# ======================================================================
# Step 3: rough outlines to start from (first frame / recovery only)
# ======================================================================

def order_quad_points(pts):
    """Label 4 points TL/TR/BR/BL by where they sit in the image (smallest
    x+y is TL, etc). Which one is the *map's* top-left is step 4's job."""
    p = np.asarray(pts, np.float32)
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)],
                     p[np.argmax(s)], p[np.argmax(d)]], np.float32)


def four_corner_outline(points):
    """Best 4-corner outline of a point cloud: its convex hull simplified to
    4 vertices (follows perspective, unlike a fitted rectangle), falling
    back to the min-area rectangle if the hull won't reduce to 4."""
    hull = cv2.convexHull(np.asarray(points, np.float32).reshape(-1, 1, 2))
    perimeter = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return order_quad_points(approx.reshape(4, 2))
    return order_quad_points(cv2.boxPoints(cv2.minAreaRect(hull)))


def paper_blob_quads(frame):
    """Every sizable paper blob in the whole frame, as (rough_quad, blob_mask,
    bounding box). corner_fitting.fit_quad turns each into precise corners."""
    h, w = frame.shape[:2]
    mask = paper_mask(frame)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_FRACTION * w * h:
            continue
        blob = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        rough = order_quad_points(cv2.boxPoints(
            cv2.minAreaRect(max(contours, key=cv2.contourArea))))
        x, y, bw, bh = stats[i, :4]
        pad_x, pad_y = int(0.15 * bw) + 5, int(0.15 * bh) + 5
        box = (max(0, x - pad_x), max(0, y - pad_y),
               min(w, x + bw + pad_x), min(h, y + bh + pad_y))
        out.append((rough, blob, box))
    return out


def traced_map_quads(frame_ink, ref):
    """Rough page outlines from edge tracing alone: cluster the long traced
    strokes (walls), take each cluster's 4-corner outline as the printed
    border, and convert border corners to page corners using where the
    border sits on the PNG. Doesn't need the paper's own edges at all."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(frame_ink, connectivity=8)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (extent >= TRACED_WALL_MIN).astype(np.uint8)
    keep[0] = 0
    walls = keep[labels] * 255
    clusters = cv2.dilate(walls, np.ones((31, 31), np.uint8))
    m, cl_labels, _, _ = cv2.connectedComponentsWithStats(clusters, connectivity=8)
    quads = []
    for i in range(1, m):
        ys, xs = np.nonzero((cl_labels == i) & (walls > 0))
        if len(xs) < 300:
            continue
        border = four_corner_outline(np.stack([xs, ys], 1))
        for labeled in orientations(border):
            H = cv2.getPerspectiveTransform(ref.border, labeled)   # map -> frame
            quads.append(cv2.perspectiveTransform(ref.corners[None], H)[0])
    return quads


# ======================================================================
# Step 4: orientation + the scoring rubric
# ======================================================================

def orientations(quad):
    """The 4 ways to label one outline's corners as the map's TL/TR/BR/BL.

    These are NOT 4 guesses at the paper's angle -- the outline already
    has the paper's exact angle, whatever it is (37 degrees, tilted in
    perspective, anything). A rectangle just looks the same from all four
    sides, so the outline alone can't say which corner is the map's
    top-left. Each labeling is the true angle plus 0/90/180/270 degrees;
    the scoring rubric decides which one is real.
    """
    q = np.asarray(quad, np.float32)
    return [np.roll(q, -k, axis=0) for k in range(4)]


def homography_for(quad, ref):
    """Frame -> map homography for a crop given as 4 frame corners."""
    return cv2.getPerspectiveTransform(np.asarray(quad, np.float32), ref.corners)


def alignment_score(ref, frame_ink, H_frame_to_map, tol=SCORE_TOLERANCE):
    """The scoring rubric: how well does this crop line the frame's traced
    lines up with the PNG's printed lines? 0 = nothing matches, 1 = perfect.

    The frame's traced ink is warped into the PNG's pixel space through the
    crop, then compared class by class -- walls to walls, symbols to
    symbols, dots to dots -- because each class answers something different:
      walls   -- is the crop geometrically right?
      symbols -- is it the right way round? (A floorplan's walls are often
                 nearly symmetric; its letters and icons never are.)
      dots    -- texture; confirms the corridor lines up.
    Matching class-to-class also stops the dense dot field from "matching"
    any stray stroke that lands near it.

    For each class:
      precision = share of the frame's traced pixels that land within `tol`
                  px of a printed pixel of that class (is what we see on the
                  map?) -- unaffected by hands covering part of the map
      recall    = share of the PNG's printed pixels that have a traced pixel
                  within `tol` px (how much of the map do we see?) -- drops
                  when hands cover it
      F         = weighted harmonic mean, precision counting 2x (F-beta,
                  beta = 0.5), so occlusion costs less than mismatches.
    score = 0.4 * F_walls + 0.4 * F_symbols + 0.2 * F_dots (weights are
    renormalized if the map has no dots, say).

    Measured on the S-19 video: right orientation 0.98-0.99; same map
    rotated 180 degrees ~0.40; 90/270 degrees 0.10-0.14; the decoy sheet
    0.03-0.04; a blank page < 0.07.
    """
    warped = cv2.warpPerspective(frame_ink, H_frame_to_map, (ref.w, ref.h),
                                 flags=cv2.INTER_NEAREST)
    result = {"score": 0.0}
    if np.count_nonzero(warped) < 50:
        return result
    classes = edge_tracing.ink_classes(warped)
    total, weight_sum = 0.0, 0.0
    for c, weight in CLASS_WEIGHTS.items():
        printed = ref.class_present[c]
        if not printed.any():
            continue
        seen = classes == c
        precision = float((ref.class_dist[c][seen] <= tol).mean()) if seen.any() else 0.0
        seen_dist = cv2.distanceTransform(np.where(seen, 0, 255).astype(np.uint8),
                                          cv2.DIST_L2, 3)
        recall = float((seen_dist[printed] <= tol).mean())
        f = 1.25 * precision * recall / max(1e-6, 0.25 * precision + recall)
        name = {edge_tracing.CLASS_WALL: "walls", edge_tracing.CLASS_SYMBOL: "symbols",
                edge_tracing.CLASS_DOT: "dots"}[c]
        result[name] = {"precision": round(precision, 3), "recall": round(recall, 3),
                        "f": round(f, 3)}
        total += weight * f
        weight_sum += weight
    result["score"] = round(total / weight_sum, 4) if weight_sum else 0.0
    return result


# ======================================================================
# Steps 6-7: where to look next frame, and which blobs are there
# ======================================================================

def search_window(prior_corners, frame_shape, margin=SEARCH_MARGIN):
    """Bounding box (x0, y0, x1, y1) of last frame's crop grown by `margin`
    (0.20 = 20% bigger, i.e. ~10% of the paper's size added on every side)."""
    h, w = frame_shape[:2]
    p = np.asarray(prior_corners, np.float32)
    centre = p.mean(axis=0)
    grown = centre + (1.0 + margin) * (p - centre)
    x0, y0 = np.floor(grown.min(axis=0)).astype(int)
    x1, y1 = np.ceil(grown.max(axis=0)).astype(int)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def candidate_blobs(frame, window, prior_corners):
    """Each separate paper blob inside the search window as its own mask
    (cropped to the window), nearest last frame's crop first. A second
    sheet is a separate blob, so its edges can never leak into the real
    page's line fit. The paper mask is computed on the window only."""
    x0, y0, x1, y1 = window
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []
    mask = paper_mask(frame[y0:y1, x0:x1])
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centre = np.asarray(prior_corners, np.float32).mean(axis=0) - [x0, y0]
    ranked = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_CANDIDATE_AREA),
                    key=lambda i: np.linalg.norm(centroids[i] - centre))
    return [np.where(labels == i, np.uint8(255), np.uint8(0))
            for i in ranked[:MAX_CANDIDATES]]


# ======================================================================
# Step 8: which side of the paper is each straight edge on
# ======================================================================

SIDES = ("top", "right", "bottom", "left")
SIDE_CORNERS = {"top": (0, 1), "right": (1, 2), "bottom": (3, 2), "left": (0, 3)}


def line_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def point_to_line_distance(pt, a, b):
    """Perpendicular distance from pt to the infinite line through a and b."""
    a, b, pt = (np.asarray(v, np.float32) for v in (a, b, pt))
    d = b - a
    norm = float(np.linalg.norm(d))
    if norm < 1e-6:
        return float(np.linalg.norm(pt - a))
    v = pt - a
    return abs(float(d[0] * v[1] - d[1] * v[0])) / norm


def classify_segment(x1, y1, x2, y2, corners):
    """(side, angle_error) for one segment, or (None, None).

    First the angle must be within SIDE_ANGLE_TOLERANCE of that side's angle
    last frame; of the sides that pass, the segment goes to whichever side's
    line it sits closest to (distance to the whole line, not the side's
    midpoint, which is unreliable near corners)."""
    angle = line_angle(x1, y1, x2, y2)
    mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    best = (None, None, None)
    for side in SIDES:
        i, j = SIDE_CORNERS[side]
        a, b = corners[i], corners[j]
        err = angle_diff(angle, line_angle(a[0], a[1], b[0], b[1]))
        if err > SIDE_ANGLE_TOLERANCE:
            continue
        dist = point_to_line_distance(mid, a, b)
        if best[1] is None or dist < best[1]:
            best = (side, dist, err)
    return best[0], best[2]


def classify_segments(segments, corners):
    """{side: [(x1, y1, x2, y2, angle_error), ...]} using last frame's crop
    as the guide. Segments matching no side are dropped."""
    corners = np.asarray(corners, np.float32)
    sides = {s: [] for s in SIDES}
    for x1, y1, x2, y2 in segments:
        side, err = classify_segment(x1, y1, x2, y2, corners)
        if side is not None:
            sides[side].append((x1, y1, x2, y2, err))
    return sides


def draw_paper_debug(shown, frame, corners):
    """'e' key overlay: the paper mask in the search window, and every
    straight paper edge colored by the side step 8 assigned it to."""
    window = search_window(corners, frame.shape)
    x0, y0, x1, y1 = window
    if x1 - x0 < 20 or y1 - y0 < 20:
        return
    mask = paper_mask(frame[y0:y1, x0:x1])
    region = shown[y0:y1, x0:x1]
    cv2.addWeighted(region, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0, dst=region)
    segs = cv2.HoughLinesP(cv2.Canny(mask, 50, 150), 1, np.pi / 180, 40,
                           minLineLength=25, maxLineGap=12)
    colors = {"top": (0, 0, 255), "bottom": (255, 0, 0),
              "left": (0, 255, 0), "right": (0, 255, 255)}
    if segs is not None:
        segs = segs.reshape(-1, 4).astype(np.float32) + [x0, y0, x0, y0]
        for x1_, y1_, x2_, y2_ in segs:
            side, _ = classify_segment(x1_, y1_, x2_, y2_, np.asarray(corners, np.float32))
            cv2.line(shown, (int(x1_), int(y1_)), (int(x2_), int(y2_)),
                     colors.get(side, (200, 200, 200)), 2)
    cv2.rectangle(shown, (x0, y0), (x1, y1), (0, 255, 255), 1)
    cv2.putText(shown, "Paper edges (key 'e'): red=top blue=bottom green=left "
                "yellow=right, box=search window", (18, shown.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
