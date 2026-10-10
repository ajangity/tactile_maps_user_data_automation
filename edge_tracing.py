"""Edge tracing: find the printed black lines in an image, on their own.

This file is self-contained on purpose: it doesn't import or depend on the
auto-crop pipeline. It takes an image (the reference PNG, or a video frame)
and, optionally, a region polygon to stay inside, and returns what ink it
found there. The caller passes the paper's crop as that region once the map
is locked, so on video frames tracing only ever processes the paper's own
pixels -- not the desk, the other sheet, or hands off the page. (Before the
map has been found there is no crop yet, so the whole frame is traced to
find it.) The auto-crop pipeline then *uses* these results to score and
fine-tune its crop (paper_locator.alignment_score,
corner_fitting.refine_with_traced_lines).

What it does:
  1. ink_mask         -- every thin dark stroke (walls, symbols, dots, text).
                         Uses a morphological "black-hat" filter: it keeps
                         dark marks that are thinner than ~9 px and ignores
                         large dark areas, so the desk, shadows and the
                         gap around the paper don't show up as "ink" even
                         though they're dark. No threshold on absolute
                         brightness, so it works on the full video frame
                         with no crop at all.
  2. ink_classes      -- splits that ink into dots / symbols / walls by the
                         size of each connected mark.
  3. trace_segments   -- straight wall segments (Hough), merged so one
                         printed line becomes one segment.
  4. build_line_graph -- turns the segments into a graph: nodes are line
                         ends, corners and T-junctions; edges are the lines
                         between them. This is what gets written to JSON.

Run on its own to inspect what it sees:
    python edge_tracing.py distractor_floorplan_E.png
    python edge_tracing.py S-19_Elevator.mp4 --time 105
"""

import argparse
import json
import math
import os

import cv2
import numpy as np

BLACKHAT_KERNEL = 9       # px; strokes thinner than this count as ink
INK_THRESHOLD = 30        # black-hat response (0-255) needed to count as ink

# Size classes, in pixels of the image being classified. For a video frame
# these are only used for the JSON graph/debug view; scoring classifies in
# map space instead (see paper_locator.alignment_score), where sizes are fixed.
DOT_MAX_EXTENT = 12       # marks no bigger than this are texture dots
WALL_MIN_EXTENT = 80      # marks at least this long are walls

CLASS_DOT, CLASS_SYMBOL, CLASS_WALL = 1, 2, 3


# ---------------------------------------------------------------- loading

def load_image(path):
    """Read an image as BGR on a white background. A transparent PNG has
    its transparent areas composited onto white (read naively they come
    out black, which would trace as one giant ink blob); grayscale becomes
    BGR."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        bgr = raw[:, :, :3].astype(np.float32)
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        return (bgr * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return raw


# ---------------------------------------------------------------- 1. ink

def ink_mask(gray, kernel=BLACKHAT_KERNEL, threshold=INK_THRESHOLD, region=None):
    """Binary mask (0/255) of thin dark strokes in a grayscale image.

    region: optional polygon (N x 2 points). If given, only pixels inside it
    are processed at all -- the filter runs on just that polygon's bounding
    box, and everything outside the polygon is left empty. The caller
    decides what the region is (e.g. the paper's crop); this file never
    works it out itself.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    if region is None:
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
        return (blackhat > threshold).astype(np.uint8) * 255
    h, w = gray.shape[:2]
    poly = np.int32(np.rint(np.asarray(region, np.float32)))
    pad = kernel                      # so the filter has context at the box edge
    x, y, bw, bh = cv2.boundingRect(poly)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    out = np.zeros((h, w), np.uint8)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return out
    blackhat = cv2.morphologyEx(gray[y0:y1, x0:x1], cv2.MORPH_BLACKHAT, se)
    inside = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(inside, [poly - [x0, y0]], 255)
    out[y0:y1, x0:x1] = np.where((blackhat > threshold) & (inside > 0), 255, 0).astype(np.uint8)
    return out


# ---------------------------------------------------------------- 2. classes

def ink_classes(mask, dot_max=DOT_MAX_EXTENT, wall_min=WALL_MIN_EXTENT):
    """Per-pixel class image: 0 background, 1 dot, 2 symbol, 3 wall.

    Each connected mark is classed by its bounding-box extent (its longer
    side): tiny marks are the corridor's texture dots, long marks are walls
    (one connected wall network is a single, very long mark), and
    everything in between -- letters, stars, stairs, circles -- is a symbol.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    lut = np.where(extent <= dot_max, CLASS_DOT,
                   np.where(extent >= wall_min, CLASS_WALL, CLASS_SYMBOL))
    lut = lut.astype(np.uint8)
    lut[0] = 0
    return lut[labels]


# ---------------------------------------------------------------- 3. segments

def trace_segments(wall_mask, min_len=25):
    """Straight segments (x1, y1, x2, y2) along the walls, with the many
    near-duplicate Hough hits on one thick printed line merged into one."""
    raw = cv2.HoughLinesP(wall_mask, 1, np.pi / 180, threshold=20,
                          minLineLength=min_len, maxLineGap=4)
    if raw is None:
        return []
    return merge_segments(raw.reshape(-1, 4).astype(np.float32))


def merge_segments(segments, angle_tol=3.0, offset_tol=4.0, gap_tol=8.0):
    """Merge segments that lie on the same printed line: same angle (within
    angle_tol degrees), same perpendicular offset (within offset_tol px),
    and touching or overlapping along their length (gap <= gap_tol)."""
    groups = []   # each: [angle_deg, unit_dir, normal, offset, t0, t1, weight]
    order = np.argsort(-np.hypot(segments[:, 2] - segments[:, 0],
                                 segments[:, 3] - segments[:, 1]))
    for i in order:
        x1, y1, x2, y2 = segments[i]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-6:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        for g in groups:
            diff = abs(angle - g[0]) % 180
            if min(diff, 180 - diff) > angle_tol:
                continue
            d, n = g[1], g[2]
            if abs(n @ np.float32([x1, y1]) - g[3]) > offset_tol:
                continue
            if abs(n @ np.float32([x2, y2]) - g[3]) > offset_tol:
                continue
            a, b = sorted((d @ np.float32([x1, y1]), d @ np.float32([x2, y2])))
            if a > g[5] + gap_tol or b < g[4] - gap_tol:
                continue
            g[4], g[5] = min(g[4], a), max(g[5], b)
            g[6] += length
            break
        else:
            rad = math.radians(angle)
            d = np.float32([math.cos(rad), math.sin(rad)])
            n = np.float32([-d[1], d[0]])
            a, b = sorted((d @ np.float32([x1, y1]), d @ np.float32([x2, y2])))
            groups.append([angle, d, n, float(n @ np.float32([x1, y1])), a, b, length])
    merged = []
    for _, d, n, offset, t0, t1, _ in groups:
        p0 = d * t0 + n * offset
        p1 = d * t1 + n * offset
        merged.append((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])))
    return merged


# ---------------------------------------------------------------- 4. graph

def build_line_graph(segments, node_tol=8.0):
    """Nodes + edges from wall segments.

    Nodes are where lines end, meet at a corner, or butt into another line
    (a T-junction, which splits that other line in two). Endpoints within
    node_tol px of each other are the same node. Edges are the straight
    lines between nodes, with their length and angle.
    """
    segs = [np.float32(s).reshape(2, 2) for s in segments]
    # T-junctions: an endpoint resting on another segment's interior splits it
    cut_points = [[] for _ in segs]
    for i, s in enumerate(segs):
        for p in s:
            for j, t in enumerate(segs):
                if i == j:
                    continue
                d = t[1] - t[0]
                length = float(np.linalg.norm(d))
                if length < 1e-6:
                    continue
                u = float(np.dot(p - t[0], d) / (length * length))
                if not (0.0 < u < 1.0):
                    continue
                if min(u, 1 - u) * length <= node_tol:
                    continue  # near that segment's end: a corner, not a T
                v = p - t[0]
                perp = abs(float(d[0] * v[1] - d[1] * v[0])) / length
                if perp <= node_tol:
                    cut_points[j].append(u)
    pieces = []
    for s, cuts in zip(segs, cut_points):
        us = [0.0] + sorted(cuts) + [1.0]
        for a, b in zip(us, us[1:]):
            if b - a > 1e-3:
                pieces.append((s[0] + a * (s[1] - s[0]), s[0] + b * (s[1] - s[0])))

    nodes = []    # [sum_x, sum_y, count]

    def node_for(p):
        p = (float(p[0]), float(p[1]))
        for k, (sx, sy, c) in enumerate(nodes):
            if math.hypot(sx / c - p[0], sy / c - p[1]) <= node_tol:
                nodes[k] = [sx + p[0], sy + p[1], c + 1]
                return k
        nodes.append([float(p[0]), float(p[1]), 1])
        return len(nodes) - 1

    edges = []
    for a, b in pieces:
        ia, ib = node_for(a), node_for(b)
        if ia != ib and not any({e[0], e[1]} == {ia, ib} for e in edges):
            edges.append((ia, ib))
    degree = [0] * len(nodes)
    for ia, ib in edges:
        degree[ia] += 1
        degree[ib] += 1
    node_list = [{"id": k, "x": round(sx / c, 1), "y": round(sy / c, 1),
                  "degree": degree[k]} for k, (sx, sy, c) in enumerate(nodes)]
    edge_list = []
    for k, (ia, ib) in enumerate(edges):
        a, b = node_list[ia], node_list[ib]
        edge_list.append({
            "id": k, "from": ia, "to": ib,
            "length": round(math.hypot(b["x"] - a["x"], b["y"] - a["y"]), 1),
            "angle": round(math.degrees(math.atan2(b["y"] - a["y"],
                                                   b["x"] - a["x"])) % 180, 1),
        })
    return {"nodes": node_list, "edges": edge_list}


# ---------------------------------------------------------------- all at once

class Trace:
    """Everything edge tracing found in one image."""

    def __init__(self, gray, dot_max=DOT_MAX_EXTENT, wall_min=WALL_MIN_EXTENT,
                 with_graph=True, region=None):
        self.shape = gray.shape
        self.region = region
        self.ink = ink_mask(gray, region=region)
        self.classes = ink_classes(self.ink, dot_max, wall_min)
        self.walls = np.where(self.classes == CLASS_WALL, 255, 0).astype(np.uint8)
        self.segments = trace_segments(self.walls) if with_graph else []
        self.graph = build_line_graph(self.segments) if with_graph else None

    def symbol_marks(self):
        """Bounding boxes (x0, y0, x1, y1) of symbol marks, with marks that
        belong together (a letter's pieces, a circle and its centre dot)
        grouped into one."""
        sym = np.where(self.classes == CLASS_SYMBOL, 255, 0).astype(np.uint8)
        grouped = cv2.dilate(sym, np.ones((7, 7), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h = stats[i, :4]
            boxes.append((int(x) + 3, int(y) + 3, int(x + w) - 3, int(y + h) - 3))
        return boxes

    def to_json(self, **extra):
        counts = {name: int((self.classes == c).sum()) for c, name in
                  ((CLASS_DOT, "dot_px"), (CLASS_SYMBOL, "symbol_px"),
                   (CLASS_WALL, "wall_px"))}
        return {"width": int(self.shape[1]), "height": int(self.shape[0]),
                "ink": counts, "lines": self.graph, **extra}


def classify_shape(mask, box):
    """Cheap rule-based shape label for a symbol: filled vs outline, round
    vs angular, roughly how many straight sides. Contour geometry only."""
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = mask[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "?"
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    fill_ratio = area / max((x1 - x0) * (y1 - y0), 1)
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
    if fill_ratio > 0.75 and circularity > 0.7:
        return "circle"
    if circularity > 0.7:
        return "ring"
    if fill_ratio > 0.75:
        return "filled"
    if len(approx) <= 5:
        return f"{len(approx)}-gon"
    return "text"


def draw_trace(canvas, trace, label=True):
    """Draw what edge tracing found onto canvas (same size as the traced
    image): wall pixels, each straight wall line (red), graph nodes
    (yellow), dots (green), and each symbol boxed in blue, plus the totals.
    Symbols aren't named here: symbol_identification.py names them, from
    the map PNG's JSON."""
    canvas[trace.walls > 0] = (0, 0, 160)
    canvas[trace.classes == CLASS_DOT] = (0, 160, 0)
    for x1, y1, x2, y2 in trace.segments:
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA)
    symbols = trace.symbol_marks()
    for x0, y0, x1, y1 in symbols:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 0), 2)
    if trace.graph:
        for n in trace.graph["nodes"]:
            cv2.circle(canvas, (int(n["x"]), int(n["y"])), 4, (0, 220, 255), -1)
    if trace.region is not None:
        cv2.polylines(canvas, [np.int32(trace.region)], True, (0, 220, 0), 1, cv2.LINE_AA)
    if label:
        g = trace.graph or {"nodes": [], "edges": []}
        where = "inside the crop" if trace.region is not None else "whole image"
        cv2.putText(canvas, f"Edge tracing (key 'w', {where}): {len(trace.segments)} wall lines, "
                    f"{len(g['nodes'])} nodes, {len(symbols)} symbols", (18, canvas.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Trace printed lines in an image or video frame.")
    parser.add_argument("source", help="a PNG/JPG, or a video")
    parser.add_argument("--time", type=float, default=0.0,
                        help="for a video: seconds into it to trace")
    parser.add_argument("-o", "--output", help="output prefix (default: next to the source)")
    args = parser.parse_args()

    image = load_image(args.source)
    if image is None:
        cap = cv2.VideoCapture(args.source)
        cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000.0)
        ok, image = cap.read()
        if not ok:
            raise SystemExit(f"Could not read an image or video frame from {args.source}")
        # a frame's map is smaller than the PNG, so its walls are shorter
        trace = Trace(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), wall_min=40)
    else:
        trace = Trace(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    prefix = args.output or os.path.splitext(args.source)[0]
    with open(prefix + ".lines.json", "w", encoding="utf-8") as f:
        json.dump(trace.to_json(source=os.path.basename(args.source)), f, indent=1)
    vis = image.copy()
    draw_trace(vis, trace)
    cv2.imwrite(prefix + ".lines.png", vis)
    g = trace.graph
    symbols = trace.symbol_marks()
    print(f"{len(g['edges'])} wall lines, {len(g['nodes'])} nodes, {len(symbols)} symbol marks")
    for i, (x0, y0, x1, y1) in enumerate(symbols):
        print(f"  symbol {i}: center=({(x0 + x1) // 2},{(y0 + y1) // 2}) size={x1 - x0}x{y1 - y0}")
    print(f"saved {prefix}.lines.json and {prefix}.lines.png")


if __name__ == "__main__":
    main()
