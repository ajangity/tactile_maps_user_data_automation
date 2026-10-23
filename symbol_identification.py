"""This file matches every detected symbol mark to the tactile-symbol
legend and names it. Symbols drawn on walls are handled as room entrances.

Symbol identification: what each symbol on the map *is*, by name.

edge_tracing.py finds where the symbols are, but only as anonymous marks.
This file matches each one to the tactile-symbol legend and names it after
the legend entry -- "Elevator #11", "Toilet #2", "Door / Emergency exit
#15" -- the names the timers, the on-screen timer list, map.json and
the CSVs all use.

THE LEGEND lives in legends/: symbol_legend.json lists every entry -- its
ID, its code in the study ("P27", the name of its map
distractor_floorplan_P27.png), referent ("Elevator"), description and
placement -- and legends/symbols/<code>.png is its page: the symbol printed
twice (regular and small) the way the maps print it. The templates are cut
out of those pages (load_legend): a free symbol from inside each square
frame, a wall symbol from the room outline's walls (the mark, with the
wall under it, as a doorway on a map is cut out), an area pattern as the
texture inside its outline. No shape is hard-coded here. To add a symbol,
add its page and a line to symbol_legend.json. (#5, Nick Giudice's stair
symbol, has no page yet; its template still comes from the old legend
screenshot symbol_legend_1.png.)

How a symbol gets its name:

  0. TEXTURE FIRST. Area patterns -- a corridor's dots, dashes or diagonal
     lines, a hatched dangerous area -- are found before anything else
     (texture_mask): dots and dashes, and hatching (thin straight diagonal strokes that come in a family at
     one angle). That ink is never a symbol. Bold hatching touches the
     walls, so room tracking gets the walls without it (room_walls), or a
     hatched room would be cut into strips.

  1. FIND. Free-standing symbols come from edge tracing's symbol marks.
     Symbols drawn *on* a wall (a circle on a wall, a tick across it, a
     letter straddling it) are invisible to edge tracing, because they're
     connected to the wall network and so count as "wall". find_wall_symbols
     recovers them: erase every long straight run of the walls, and whatever
     ink is left over is a symbol. Small hollow rings (which edge tracing
     files under "dots") are picked up too. Doorways are found as well
     (find_wall_gaps): a short break along a wall line. A break and the wall
     symbols beside it are one door; a break with no symbol is a door too.

  2. RECOGNIZE. Each symbol (a whole doorway, once merged) is matched
     against the legend: its ink and every legend template are scaled into
     a 48x48 box and thinned to 1 px centre lines, then compared by chamfer
     distance -- on average, how many pixels apart the two drawings' lines
     are. The closest template wins if it's close enough (MATCH_MAX);
     otherwise the symbol is "Unknown symbol". A doorway that is only a gap
     is legend #14 ("Gap or break in wall").
     Orientation is part of a symbol's meaning in the legend (a square open
     at the top is a toilet, open at the right an elevator), so:
       - a free-standing symbol is compared upright; a turned match only
         wins if it beats every upright one by ROTATION_PENALTY.
       - a symbol on a wall is compared relative to its wall, since the
         legend draws every wall horizontal: on a vertical wall it's turned
         a quarter turn first (either way round, as a door can face either
         room). That's what tells a circle with the wall running through it
         (#18) from one with a line across the wall (#19).

  3. NAME IT: "<referent> #<legend id>", e.g. "Elevator #11". A name that
     occurs more than once is numbered in reading order ("Elevator #11
     (1)", "Elevator #11 (2)"), because every timer needs its own name.
     A map can rename legend entries with its own <map>.legend.json:

         {"28": "Hallway", "13": "Front door"}

  4. WRITE IT TO THE MAP JSON. detect_symbols builds the "symbols" section
     of map.json: every symbol with its name, legend id, referent,
     description, room and box. Every room filled with a texture is
     matched to the legend's area patterns too (room_patterns: what share
     of the texture is dots, dashes and lines, how the dots are laid out,
     how bold the lines are, against the same numbers for each pattern's
     page), and paper_locator stores that on the room as "pattern", so it's
     labeled "Corridor #26", "Dangerous / Inaccessible area #25".

  5. TIME IT. SymbolTiming gives every symbol its own timer. A symbol on a
     wall marks a room's entrance, whatever its referent (a toilet symbol
     on a toilet's wall marks the way into it), so its timer is an
     "entrance" timer: it covers a little space around the mark and records
     which room the entrance leads into. An area pattern fills its room, so
     the room's own timer is its timer.

Symbols in the video are the same printed symbols: the crop maps every
video pixel onto the PNG, so a fingertip on a symbol in the video lands on
that symbol's box here, and is timed under its name.

Run on its own to check what it sees on a map:
    python symbol_identification.py distractor_floorplan_E.png
"""

import argparse
import json
import os

import cv2
import numpy as np

import edge_tracing
from edge_tracing import CLASS_DOT, CLASS_SYMBOL
from room_tracking import detect_rooms
from timing import RegionTimer, Timing

# ---- finding symbols drawn on walls
WALL_RUN_MIN = 45         # px; a straight wall run at least this long is "wall", not symbol
WALL_SYMBOL_MIN = 10      # px; smaller leftovers are corner/junction crumbs
WALL_SYMBOL_MAX = 80      # px; bigger leftovers are diagonal walls / hatching, not a symbol
GROUP_GAP = 9             # px; leftover pieces this close together are one symbol

# ---- finding doorways (gaps in a wall)
DOOR_GAP_MIN = 10         # px; a shorter break in a wall is a print flaw, not a doorway
DOOR_GAP_MAX = 70         # px; a longer break is two separate walls
DOOR_JOIN = 20            # px; wall symbols this close to a gap belong to that doorway

# ---- timing
SYMBOL_PADDING = 6        # px added around each symbol's box for its timer
ENTRANCE_REACH = 18       # px around an entrance symbol that count as "at the entrance"

# ---- recognizing shapes
THIN_SIZE = 96            # symbols are enlarged to this before thinning, to keep detail
NORM_SIZE = 48            # symbols and templates are compared in a box this size
NORM_INK = 40             # ... with the shape's longer side scaled to this
MATCH_MAX = 1.5           # px (in the 48x48 box); a worse best match is "unknown"
ROTATION_PENALTY = 1.0    # px added to a turned match of a free-standing symbol

# ---- area textures (dots, dashes, hatching)
DOT_MAX = 8               # px; a texture mark no longer than this is a dot
DASH_MAX = 20             # px; ... up to this (and twice as long as wide) a dash; longer, a line
HATCH_MIN_LEN = 15        # px; a hatch stroke is at least this long
HATCH_MAX_WIDTH = 7       # px; ... and at most this thick
HATCH_ANGLE = (25, 65)    # degrees from horizontal: hatching is diagonal
HATCH_FAMILY = 5          # strokes at the same angle (+- HATCH_ANGLE_TOL) needed to be hatching
HATCH_ANGLE_TOL = 10      # degrees
TEXTURE_MIN_MARKS = 6     # a room with fewer texture marks than this has no pattern
AREA_MATCH_MAX = 0.5      # texture distance; a worse best match is no pattern

# ---- the legend
LEGEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legends")
LEGEND_JSON = os.path.join(LEGEND_DIR, "symbol_legend.json")
LEGEND_INK = 140          # legend screenshot pixels darker than this are ink
GAP_LEGEND_ID = 14        # "Gap or break in wall": a doorway with no mark
PAGE_GROUP = 41           # px; ink this close on a legend page is one drawing
MARK_REACH = 25           # px either side of a wall line searched for its mark
CORNER_CLEAR = 7          # px; a wall mark closer than this to a corner isn't used


# ---------------------------------------------------------------- 0. texture

def _strokes(mask):
    """(labels, [(label, length, width, angle 0..180, pixels)]): every
    connected mark in mask, measured by its tightest rotated box."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        if len(xs) < 3:
            out.append((i, 1.0, 1.0, 0.0, area))
            continue
        (_, _), (a, b), angle = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        if a < b:
            a, b, angle = b, a, angle + 90
        out.append((i, float(a) + 1, float(b) + 1, angle % 180, area))
    return labels, out


def _straight_runs(walls):
    return (cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((1, WALL_RUN_MIN), np.uint8)) |
            cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((WALL_RUN_MIN, 1), np.uint8)))


def hatching(trace):
    """Mask of hatch strokes: thin, straight, diagonal marks that come in a
    family of HATCH_FAMILY or more at the same angle (a lone diagonal line,
    or the stroke of an X, is not hatching). Light hatching is printed loose
    (edge tracing calls each stroke a symbol); bold hatching touches the
    room's walls (it's part of the wall network), so it's looked for there
    too, with the long straight wall runs taken away."""
    if getattr(trace, "_hatching", None) is not None:
        return trace._hatching
    off_walls = cv2.subtract(trace.walls, cv2.dilate(_straight_runs(trace.walls), np.ones((5, 5), np.uint8)))
    candidates = off_walls | np.where(trace.classes == CLASS_SYMBOL, 255, 0).astype(np.uint8)
    labels, strokes = _strokes(candidates)
    lo, hi = HATCH_ANGLE
    diagonal = [s for s in strokes if s[1] >= HATCH_MIN_LEN and s[2] <= HATCH_MAX_WIDTH
                and (lo <= s[3] <= hi or 180 - hi <= s[3] <= 180 - lo)]
    keep = [s[0] for s in diagonal
            if sum(abs((t[3] - s[3] + 90) % 180 - 90) <= HATCH_ANGLE_TOL for t in diagonal) >= HATCH_FAMILY]
    trace._hatching = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)
    return trace._hatching


def texture_mask(trace):
    """All area-texture ink: solid dots and dashes (edge tracing's dot
    class, without the small hollow rings, which are door marks) and
    hatching."""
    if getattr(trace, "_texture", None) is not None:
        return trace._texture
    dots = np.where(trace.classes == CLASS_DOT, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(dots, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i, (_, _, child, parent) in enumerate(hierarchy[0]):
            if parent < 0 and child >= 0:
                cv2.drawContours(dots, contours, i, 0, -1)
    trace._texture = dots | hatching(trace)
    return trace._texture


def room_walls(trace):
    """The walls without any bold hatching. Room tracking must use these:
    hatch strokes that touch the walls would otherwise count as walls and
    cut a hatched room into strips (or hand it to its neighbours)."""
    if getattr(trace, "_room_walls", None) is None:
        hatch = cv2.dilate(hatching(trace), np.ones((5, 5), np.uint8))
        hatch[_straight_runs(trace.walls) > 0] = 0
        trace._room_walls = cv2.subtract(trace.walls, hatch)
    return trace._room_walls


def texture_features(texture, region=None):
    """What a patch of texture looks like, as numbers from 0 to 1: which
    share of its ink is dots, dashes and lines; of the dots, how many have
    their nearest neighbour straight across (rows and columns) rather than
    diagonal; and how thick the lines are. None if it has too few marks."""
    if region is not None:
        texture = np.where(region > 0, texture, 0).astype(np.uint8)
    labels, strokes = _strokes(texture)
    if len(strokes) < TEXTURE_MIN_MARKS:
        return None
    total = float(sum(s[4] for s in strokes))
    dots = [s for s in strokes if s[1] <= DOT_MAX]
    dashes = [s for s in strokes if DOT_MAX < s[1] <= DASH_MAX and s[1] >= 2 * s[2]]
    lines = [s for s in strokes if s[1] > DASH_MAX]
    straight = 0.5
    if len(dots) >= 3:
        _, _, _, centres = cv2.connectedComponentsWithStats(
            np.where(np.isin(labels, [s[0] for s in dots]), 255, 0).astype(np.uint8), connectivity=8)
        pts = centres[1:]
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        np.fill_diagonal(d, np.inf)
        step = pts[d.argmin(axis=1)] - pts
        angle = np.degrees(np.arctan2(np.abs(step[:, 1]), np.abs(step[:, 0])))   # 0..90
        straight = float(np.mean(np.minimum(angle, 90 - angle) < 22.5))
    thick = min(1.0, sum(s[4] for s in lines) / sum(s[1] for s in lines) / 6.0) if lines else 0.0
    return {"dots": sum(s[4] for s in dots) / total, "dashes": sum(s[4] for s in dashes) / total,
            "lines": sum(s[4] for s in lines) / total, "straight": straight, "thick": thick}


def texture_distance(a, b):
    """How different two textures are: mostly what they're made of (dots,
    dashes, lines), then how the dots are laid out and how bold the lines
    are."""
    d = sum(abs(a[k] - b[k]) for k in ("dots", "dashes", "lines"))
    d += min(a["dots"], b["dots"]) * abs(a["straight"] - b["straight"])
    d += 2 * min(a["lines"], b["lines"]) * abs(a["thick"] - b["thick"])
    return d


# ---------------------------------------------------------------- 1. find

def find_wall_symbols(trace):
    """(mask, boxes) of the symbols edge tracing counted as wall or dots.

    mask is their ink (0/255); boxes are (x0, y0, x1, y1), one per symbol.
    """
    walls = room_walls(trace)
    runs = (cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((1, WALL_RUN_MIN), np.uint8)) |
            cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((WALL_RUN_MIN, 1), np.uint8)))
    leftover = cv2.subtract(walls, cv2.dilate(runs, np.ones((3, 3), np.uint8)))
    leftover = cv2.morphologyEx(leftover, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Hollow rings small enough to be classed as dots (texture dots are solid).
    dots = np.where(trace.classes == CLASS_DOT, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(dots, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = np.zeros_like(dots)
    if hierarchy is not None:
        for i, (_, _, child, parent) in enumerate(hierarchy[0]):
            if parent < 0 and child >= 0:
                cv2.drawContours(rings, contours, i, 255, -1)
        rings &= dots

    mask = leftover | rings
    grouped = cv2.dilate(mask, np.ones((GROUP_GAP, GROUP_GAP), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    keep = np.zeros_like(mask)
    boxes = []
    for i in range(1, n):
        ys, xs = np.nonzero((labels == i) & (mask > 0))
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if WALL_SYMBOL_MIN <= max(x1 - x0, y1 - y0) <= WALL_SYMBOL_MAX:
            # the symbol's own ink includes the bits that lie along the wall
            # (a letter's bar, the wall crossing a tick), so take it all back
            keep[y0:y1 + 1, x0:x1 + 1] = (walls | rings)[y0:y1 + 1, x0:x1 + 1]
            boxes.append((x0, y0, x1, y1))
    return keep, boxes


def find_wall_gaps(trace):
    """Doorways: boxes (x0, y0, x1, y1) of the breaks along each wall line.

    Every long straight run of wall belongs to a wall line (runs at the same
    x, or the same y, are one line). Walking along a line, a stretch with no
    ink between DOOR_GAP_MIN and DOOR_GAP_MAX px long is a doorway. A symbol
    sitting in the doorway (four circles, a double circle) doesn't hide it:
    only ink on the line itself counts.
    """
    walls = room_walls(trace)
    gaps = []
    for vertical in (True, False):
        kernel = np.ones((WALL_RUN_MIN, 1) if vertical else (1, WALL_RUN_MIN), np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(walls, cv2.MORPH_OPEN, kernel))
        runs = sorted((x + w / 2.0, y, y + h) if vertical else (y + h / 2.0, x, x + w)
                      for x, y, w, h, _ in stats[1:])
        lines = []   # [position across, start along, end along]
        for across, start, end in runs:
            if lines and abs(lines[-1][0] - across) <= 3:
                lines[-1][1:] = [min(lines[-1][1], start), max(lines[-1][2], end)]
            else:
                lines.append([across, start, end])
        for across, start, end in lines:
            c, start, end = int(round(across)), int(start), int(end)
            band = walls[start:end, max(0, c - 3):c + 4] if vertical else walls[max(0, c - 3):c + 4, start:end]
            empty = np.flatnonzero(band.max(axis=1 if vertical else 0) == 0)
            if len(empty) == 0:
                continue
            for stretch in np.split(empty, np.flatnonzero(np.diff(empty) > 1) + 1):
                if DOOR_GAP_MIN <= len(stretch) <= DOOR_GAP_MAX:
                    a, b = int(stretch[0]) + start, int(stretch[-1]) + start
                    gaps.append((c - 3, a, c + 3, b) if vertical else (a, c - 3, b, c + 3))
    return gaps


def merge_doorways(symbols, gaps):
    """One entry per doorway. A gap and the wall symbols within DOOR_JOIN px
    of it are the same door (the two ticks either side of a gap, the four
    circles around one), so they become a single symbol covering all of
    them, recognized as a whole afterwards. parts = how many wall symbols
    it absorbed; a gap with none is legend #14, a plain gap in the wall."""
    symbols = list(symbols)
    for gap in gaps:
        reach = (gap[0] - DOOR_JOIN, gap[1] - DOOR_JOIN, gap[2] + DOOR_JOIN, gap[3] + DOOR_JOIN)
        parts = [s for s in symbols if s["on_wall"] and _overlap(s["box"], reach)]
        boxes = [s["box"] for s in parts] + [gap]
        door = {"on_wall": True, "gap": gap, "parts": len(parts),
                "box": (min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes))}
        symbols = [s for s in symbols if not any(s is part for part in parts)] + [door]
    return symbols


def wall_runs(trace):
    """(vertical, horizontal): the walls' long straight runs, each way."""
    walls = room_walls(trace)
    return (cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((WALL_RUN_MIN, 1), np.uint8)),
            cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((1, WALL_RUN_MIN), np.uint8)))


def wall_is_vertical(symbol, runs, reach=12):
    """Which way the wall under a wall symbol runs: a doorway's gap says so
    directly (it's a thin box along the wall); otherwise, whichever runs
    there are more of around the symbol."""
    if symbol.get("gap"):
        x0, y0, x1, y1 = symbol["gap"]
        return (y1 - y0) > (x1 - x0)
    x0, y0, x1, y1 = symbol["box"]
    area = (slice(max(0, y0 - reach), y1 + reach + 1), slice(max(0, x0 - reach), x1 + reach + 1))
    return int(runs[0][area].sum()) > int(runs[1][area].sum())


def _overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def symbol_marks(trace):
    """(ink, boxes) of the free-standing symbol marks: edge tracing's symbol
    class without the area texture (light hatch strokes are symbol-sized),
    with marks that belong together (a letter's pieces, a circle and its
    centre dot) grouped into one box, as in Trace.symbol_marks."""
    ink = cv2.subtract(np.where(trace.classes == CLASS_SYMBOL, 255, 0).astype(np.uint8), texture_mask(trace))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((7, 7), np.uint8)), connectivity=8)
    return ink, [(int(x) + 3, int(y) + 3, int(x + w) - 3, int(y + h) - 3) for x, y, w, h, _ in stats[1:]]


def find_symbols(trace):
    """Every symbol on the map: (ink_mask, [(box, on_wall), ...]).

    A free-standing mark that overlaps a wall symbol (the inner rings of a
    bullseye whose outer ring touches a wall) is merged into it.
    """
    wall_ink, wall_boxes = find_wall_symbols(trace)
    free_ink, free_boxes = symbol_marks(trace)
    ink = free_ink | wall_ink
    found = [(box, True) for box in wall_boxes]
    for box in free_boxes:
        for k, (other, _) in enumerate(found):
            if _overlap(box, other):
                found[k] = ((min(box[0], other[0]), min(box[1], other[1]),
                             max(box[2], other[2]), max(box[3], other[3])), True)
                break
        else:
            found.append((box, False))
    return ink, found


# ---------------------------------------------------------------- 2. recognize

def normalize(mask):
    """A symbol's ink as 1 px centre lines, scaled into the NORM_SIZE box
    with its proportions kept. Thinning happens before the final scaling,
    so stroke weight doesn't change how big the shape comes out."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = THIN_SIZE / float(max(crop.shape))
    big = cv2.resize(crop, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ys, xs = np.nonzero(thin(np.pad(np.where(big > 100, 255, 0).astype(np.uint8), 1)))
    if len(xs) == 0:
        return None
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    scale = NORM_INK / float(max(w, h, 1))
    px = np.rint((xs - xs.min() - w / 2.0) * scale + NORM_SIZE / 2.0).astype(int)
    py = np.rint((ys - ys.min() - h / 2.0) * scale + NORM_SIZE / 2.0).astype(int)
    out = np.zeros((NORM_SIZE, NORM_SIZE), np.uint8)
    out[np.clip(py, 0, NORM_SIZE - 1), np.clip(px, 0, NORM_SIZE - 1)] = 255
    return out


def thin(mask):
    """Zhang-Suen thinning: every stroke down to a 1 px centre line, so a
    bold printed letter and a thin drawn template compare like for like."""
    img = (mask > 0).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
                 p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]   # N, NE, E, ... NW
            count = sum(n)
            turns = sum((n[k] == 0) & (n[(k + 1) % 8] == 1) for k in range(8))
            if step == 0:
                corner = (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
            else:
                corner = (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            remove = (img == 1) & (count >= 2) & (count <= 6) & (turns == 1) & corner
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            return img * 255


def _chamfer(a, b):
    """Average distance (px) from each drawing's lines to the other's, both ways."""
    da = cv2.distanceTransform(np.where(a > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform(np.where(b > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    return 0.5 * (float(db[a > 0].mean()) + float(da[b > 0].mean()))


def _page_groups(ink):
    """Boxes (x0, y0, x1, y1) of the separate drawings on a legend page."""
    g = PAGE_GROUP
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((g, g), np.uint8)))
    return [(x + g // 2, y + g // 2, x + w - g // 2, y + h - g // 2)
            for x, y, w, h, _ in stats[1:] if max(w, h) - g >= 20]


def frame_cells(ink):
    """Free symbols on a legend page: the ink inside each square frame (one
    regular-size, one small print). A frame is the biggest kind of square
    hole on the page; the top row of them is the symbol's (the "S" page has
    a stray drawing further down)."""
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if (hierarchy[0][i][3] >= 0 and w >= 30 and 0.8 <= w / float(h) <= 1.25
                and cv2.contourArea(c) > 0.85 * w * h):
            holes.append((x, y, x + w, y + h))
    if not holes:
        return []
    biggest = max(b[2] - b[0] for b in holes)
    holes = [b for b in holes if b[2] - b[0] > 0.6 * biggest]
    top = min(b[1] for b in holes)
    return [ink[y0 + 3:y1 - 3, x0 + 3:x1 - 3] for x0, y0, x1, y1 in holes if y0 < top + 40]


def top_wall_marks(ink):
    """Door marks on the top wall of the room outline drawn in ink: the
    ink (wall included, side walls left out) of each stretch of the wall
    with ink off the line or a break in it, as a doorway on a map is cut
    out. A mark that runs into a corner is skipped: its crop would take
    the corner along, which no doorway on a map has."""
    ys, xs = np.nonzero(ink)
    span_w, span_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    rows = np.flatnonzero((ink > 0).sum(axis=1) > 0.4 * span_w)   # the outline's sides:
    cols = np.flatnonzero((ink > 0).sum(axis=0) > 0.4 * span_h)   # mostly-inked lines
    if len(rows) == 0 or len(cols) == 0:
        return []
    top, left, right = int(rows.min()), int(cols.min()), int(cols.max())
    line = np.flatnonzero((ink[top:top + 8] > 0).sum(axis=1) > 0.4 * (right - left)) + top
    y_line, half = int(round(line.mean())), max(2, (line.max() - line.min()) // 2 + 2)
    clean = ink.copy()
    for x in (left, right):   # the side walls, above and below the top wall
        clean[y_line + half + 1:, max(0, x - 3):x + 4] = 0
        clean[:max(0, y_line - half), max(0, x - 3):x + 4] = 0
    y0 = max(0, y_line - MARK_REACH)
    band = clean[y0:y_line + MARK_REACH + 1]
    off = band.copy()
    off[y_line - half - y0:y_line + half - y0 + 1] = 0
    on_line = ink[y_line - half:y_line + half + 1].max(axis=0)
    marked = set(np.flatnonzero(off.max(axis=0))) | {x for x in range(left + 3, right - 2) if on_line[x] == 0}
    marked = sorted(x for x in marked if left - MARK_REACH <= x <= right + MARK_REACH)
    out = []
    for piece in (np.split(np.array(marked), np.flatnonzero(np.diff(marked) > DOOR_JOIN) + 1) if marked else []):
        a, b = int(piece[0]), int(piece[-1])
        if b - a < 3 or a - left < CORNER_CLEAR or right - b < CORNER_CLEAR:
            continue
        r = np.flatnonzero(band[:, a:b + 1].max(axis=1))
        r0 = min(y0 + int(r.min()), y_line - half) if len(r) else y_line - half
        r1 = max(y0 + int(r.max()), y_line + half) if len(r) else y_line + half
        out.append(clean[r0:r1 + 1, a:b + 1])
    return out


def wall_cells(ink):
    """Wall symbols on a legend page: the door marks on each room outline's
    top wall, and on its right wall turned a quarter turn so that wall runs
    horizontally too (the legend's convention)."""
    out = []
    for x0, y0, x1, y1 in _page_groups(ink):
        r = MARK_REACH
        drawing = np.ascontiguousarray(ink[max(0, y0 - r):y1 + r, max(0, x0 - r):x1 + r])
        out += top_wall_marks(drawing) + top_wall_marks(np.ascontiguousarray(np.rot90(drawing, 1)))
    return out


_legend_cache = None


def load_legend():
    """legends/symbol_legend.json's entries ({"id", "code", "referent",
    "description", "placement", "image"}), each with what it's matched by:

      templates  free and wall symbols: every drawing of the symbol on its
                 page in legends/symbols/, as normalized centre lines
      texture    area patterns: texture_features of the page's pattern

    The plain gap (#14) needs neither. An entry with a "box" is cropped
    from that box of its image instead (a legend screenshot)."""
    global _legend_cache
    if _legend_cache is None:
        with open(LEGEND_JSON, encoding="utf-8") as f:
            entries = json.load(f)["symbols"]
        for e in entries:
            e["templates"], e["texture"] = [], None
            path = os.path.join(LEGEND_DIR, e["image"])
            if e.get("box"):
                x0, y0, x1, y1 = e["box"]
                cell = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
                cells = [np.where(cell < LEGEND_INK, 255, 0).astype(np.uint8)]
            elif e["placement"] in ("free", "wall", "area"):
                gray = cv2.cvtColor(edge_tracing.load_image(path), cv2.COLOR_BGR2GRAY)
                if e["placement"] == "area":
                    e["texture"] = texture_features(texture_mask(edge_tracing.Trace(gray, with_graph=False)))
                    continue
                ink = edge_tracing.ink_mask(gray)
                cells = frame_cells(ink) if e["placement"] == "free" else wall_cells(ink)
            else:
                continue
            e["templates"] = [normalize(c) for c in cells if c.any()]
        _legend_cache = entries
    return _legend_cache


def legend_entry(legend_id):
    return next((e for e in load_legend() if e["id"] == legend_id), None)


def _best_match(shape, turns, placements=None):
    """(legend_id, quarter turns, distance) of the closest template."""
    best = (None, 0, float("inf"))
    for entry in load_legend():
        if placements and entry["placement"] not in placements:
            continue
        for template in entry["templates"]:
            for quarter, penalty in turns.items():
                d = _chamfer(np.rot90(shape, quarter), template) + penalty
                if d < best[2]:
                    best = (entry["id"], quarter, d)
    return best


def recognize(ink, box, on_wall=False, wall_vertical=False):
    """(legend_id, rotation_degrees, distance) for the symbol ink inside box.

    legend_id is None if no legend template is within MATCH_MAX. rotation
    is the quarter turns (anticlockwise) that bring the symbol onto the
    legend's drawing.

    A mark on a wall is compared relative to its wall (the legend draws
    walls horizontal).
    """
    x0, y0, x1, y1 = [int(v) for v in box]
    shape = normalize(ink[max(0, y0):y1 + 1, max(0, x0):x1 + 1])
    if shape is None:
        return None, 0, float("inf")
    if on_wall:   # relative to the wall; the legend draws walls horizontal
        best = _best_match(shape, {1: 0.0, 3: 0.0} if wall_vertical else {0: 0.0, 2: 0.0})
    else:
        best = _best_match(shape, {0: 0.0, 1: ROTATION_PENALTY, 2: ROTATION_PENALTY, 3: ROTATION_PENALTY})
    if best[2] > MATCH_MAX:
        return None, 0, best[2]
    return best[0], best[1] * 90, best[2]


# ---------------------------------------------------------------- 3. name

def load_overrides(map_path=None):
    """<map>.legend.json ({legend id: name}), if the map has one."""
    path = os.path.splitext(map_path)[0] + ".legend.json" if map_path else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {str(k): v for k, v in json.load(f).items()}
    return {}


def legend_name(legend_id, overrides=None):
    """A legend entry's display name: "<referent> #<id>", or the map's own
    name for it from <map>.legend.json."""
    if overrides and str(legend_id) in overrides:
        return overrides[str(legend_id)]
    entry = legend_entry(legend_id)
    return f"{entry['referent']} #{legend_id}" if entry else "Unknown symbol"


def is_default_name(name):
    """True for the placeholder names ("Symbol 7") this file replaces."""
    head, _, tail = str(name).rpartition(" ")
    return head == "Symbol" and tail.isdigit()


def name_symbols(symbols, overrides=None):
    """Give every symbol dict (with a "legend_id") a unique "name", in place."""
    base = [legend_name(s["legend_id"], overrides) for s in symbols]
    seen = {}
    for s, name in zip(symbols, base):
        if base.count(name) > 1:
            seen[name] = seen.get(name, 0) + 1
            s["name"] = f"{name} ({seen[name]})"
        else:
            s["name"] = name
    return symbols


def identify_symbols(trace, map_path=None):
    """Every symbol on the map, found, recognized and named.

    Returns dicts with legend_id, referent, description, name, rotation,
    match (chamfer distance; lower is surer), on_wall and box (x0, y0, x1,
    y1), in reading order. A doorway also has gap, the box of its break in
    the wall, when it has one.
    """
    ink, found = find_symbols(trace)
    symbols = merge_doorways([{"on_wall": on_wall, "box": box} for box, on_wall in found],
                             find_wall_gaps(trace))
    runs = wall_runs(trace)
    door_ink = ink | room_walls(trace)   # a doorway is matched with the wall stubs inside it
    for s in symbols:
        if s.get("gap") and not s["parts"]:
            legend_id, rotation, distance = GAP_LEGEND_ID, 0, 0.0
        elif s["on_wall"]:
            legend_id, rotation, distance = recognize(door_ink if s.get("gap") else ink, s["box"],
                                                      True, wall_is_vertical(s, runs))
        else:
            legend_id, rotation, distance = recognize(ink, s["box"])
        entry = legend_entry(legend_id) or {}
        s.update(legend_id=legend_id, referent=entry.get("referent"),
                 description=entry.get("description"), rotation=rotation, match=round(distance, 2))
    symbols.sort(key=lambda s: (s["box"][1] // 40, s["box"][0]))
    return name_symbols(symbols, load_overrides(map_path))


# ---------------------------------------------------------------- 4. map JSON

def recognize_texture(features):
    """(legend_id, distance) of the area pattern closest to features, or
    (None, distance) if none is within AREA_MATCH_MAX."""
    best = (None, float("inf"))
    for entry in load_legend():
        if entry["texture"] is not None:
            d = texture_distance(features, entry["texture"])
            if d < best[1]:
                best = (entry["id"], d)
    return best if best[1] <= AREA_MATCH_MAX else (None, best[1])


def room_patterns(trace, label_image, map_path=None):
    """{room id: {legend_id, name, referent, description, match}} for every
    room filled with an area pattern: the corridor's dots, dashes or
    diagonal lines, a hatched dangerous area. The room's texture is
    compared with each pattern's page in legends/symbols/ (texture_features)."""
    texture = texture_mask(trace)
    overrides = load_overrides(map_path)
    out = {}
    for room in np.unique(label_image[texture > 0]):
        if room <= 0:
            continue
        features = texture_features(texture, label_image == room)
        legend_id, distance = recognize_texture(features) if features else (None, 0.0)
        if legend_id is None:
            continue
        entry = legend_entry(legend_id)
        out[int(room)] = {"legend_id": legend_id, "name": legend_name(legend_id, overrides),
                          "referent": entry["referent"], "description": entry["description"],
                          "match": round(distance, 2)}
    return out


def corridor_id(trace, label_image, patterns=None):
    """The corridor's room id, or 0: the room with a corridor pattern
    (the most textured one if there are several); failing that, the room
    that holds most of the texture dots."""
    patterns = room_patterns(trace, label_image) if patterns is None else patterns
    texture = texture_mask(trace)
    corridors = [r for r, p in patterns.items() if p["referent"] == "Corridor"]
    if corridors:
        return max(corridors, key=lambda r: int(((label_image == r) & (texture > 0)).sum()))
    ids = label_image[trace.classes == CLASS_DOT]
    ids = ids[ids > 0]
    if len(ids) == 0:
        return 0
    values, counts = np.unique(ids, return_counts=True)
    return int(values[counts.argmax()]) if counts.max() > 0.5 * len(ids) else 0


def detect_symbols(trace, label_image, map_path=None):
    """The "symbols" section of the map JSON: every symbol, named and
    tagged with its room. Replaces room_tracking.detect_symbols.

    A symbol drawn on a wall sits at a room's entrance, so it has no room
    of its own; instead it gets entrance=True and entrance_to, the ids of
    the rooms it leads into (the rooms touching it, leaving out the
    corridor when there is another).

    auto_name is the name as generated; "name" starts out the same and is
    what a person may edit in the map JSON.
    """
    h, w = label_image.shape
    corridor = corridor_id(trace, label_image)
    symbols = []
    for k, found in enumerate(identify_symbols(trace, map_path), 1):
        x0, y0, x1, y1 = found["box"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        room = int(label_image[min(cy, h - 1), min(cx, w - 1)])
        p = SYMBOL_PADDING
        symbol = {
            "id": k,
            "name": found["name"],
            "auto_name": found["name"],
            "legend_id": found["legend_id"],
            "referent": found["referent"],
            "description": found["description"],
            "rotation": found["rotation"],
            "match": found["match"],
            "on_wall": found["on_wall"],
            "room": room or None,
            "bbox": [max(0, x0 - p), max(0, y0 - p), min(w - 1, x1 + p), min(h - 1, y1 + p)],
        }
        if found["on_wall"]:
            r = ENTRANCE_REACH
            near = np.unique(label_image[max(0, y0 - r):y1 + r + 1, max(0, x0 - r):x1 + r + 1])
            rooms = [int(i) for i in near if i > 0]
            symbol["room"] = None
            symbol["entrance"] = True
            symbol["entrance_to"] = [i for i in rooms if i != corridor] or rooms
            symbol["entrance_zone"] = [max(0, x0 - r), max(0, y0 - r), min(w - 1, x1 + r), min(h - 1, y1 + r)]
            if found.get("gap"):
                symbol["gap"] = list(found["gap"])
        symbols.append(symbol)
    return symbols


def restore_auto_names(previous_json, symbols):
    """Undo the carry-over of names nobody edited.

    paper_locator copies every name from the previous run's map JSON onto
    the new symbols. That is right for a name someone typed in by hand,
    but a name that was still the generated one (or an old "Symbol 7"
    placeholder) should follow symbol identification instead, so those
    are set back to this run's auto_name. So is every name from a map JSON
    written before the legend (its symbols have no legend_id).
    """
    if not previous_json or not os.path.exists(previous_json):
        return symbols
    try:
        with open(previous_json, encoding="utf-8") as f:
            old = json.load(f).get("symbols") or []
    except (OSError, ValueError):
        return symbols
    if len(old) == len(symbols):
        for new, prev in zip(symbols, old):
            if (prev.get("name") == prev.get("auto_name") or is_default_name(prev.get("name"))
                    or "legend_id" not in prev):
                new["name"] = new["auto_name"]
    return symbols


# ---------------------------------------------------------------- 5. timers

class SymbolTimer(RegionTimer):
    """A symbol's timer. Same visit rules as every other timer (see
    timing.py: only stays of 1 s or more count); it also writes what the
    symbol is into the results JSON.

    For an entrance symbol the timer is of type "entrance" and covers the
    symbol plus ENTRANCE_REACH px around it: a fingertip exploring a
    doorway rests beside the mark as often as on it, and the mark alone
    (a tick across a wall is 3 px tall) is too small a target.
    """

    def __init__(self, symbol, room_names):
        self.symbol = symbol
        self.entrance = bool(symbol.get("entrance"))
        if self.entrance:
            self.leads_to = [room_names[i] for i in symbol.get("entrance_to", []) if i in room_names]
            zone, kind, room = symbol["entrance_zone"], "entrance", " / ".join(self.leads_to) or None
        else:
            zone, kind, room = symbol["bbox"], "symbol", room_names.get(symbol["room"])
        x0, y0, x1, y1 = zone
        super().__init__(symbol["name"], kind, lambda x, y, tip: x0 <= x <= x1 and y0 <= y <= y1,
                         {"bbox": zone}, room=room)

    def summary(self):
        out = super().summary()
        out["symbol"] = {k: self.symbol.get(k) for k in
                         ("id", "legend_id", "referent", "description", "rotation", "on_wall")}
        if self.entrance:
            out["entrance_to"] = self.leads_to
            out["symbol_bbox"] = self.symbol["bbox"]
        return out


class SymbolTiming(Timing):
    """timing.Timing with the symbols' timers swapped for SymbolTimers, so
    entrance symbols are timed as entrances. Everything else is unchanged."""

    def __init__(self, room_map, manual_boxes=None):
        super().__init__(room_map, manual_boxes)
        self.symbols = [SymbolTimer(s, room_map.room_names) for s in room_map.symbols]

    def entrances(self):
        return [t for t in self.symbols if t.entrance]

# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description="Match the symbols on a map PNG to the legend.")
    parser.add_argument("map", help="the map PNG")
    parser.add_argument("-o", "--output", help="output picture (default: <map>.symbols.png)")
    args = parser.parse_args()

    image = edge_tracing.load_image(args.map)
    if image is None:
        raise SystemExit(f"Could not read {args.map}")
    trace = edge_tracing.Trace(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    symbols = identify_symbols(trace, args.map)
    for i, s in enumerate(symbols, 1):
        x0, y0, x1, y1 = s["box"]
        print(f"  {i:2d}  {s['name']:<34} {s['description'] or '':<44} match={s['match']:<5} "
              f"rot={s['rotation']:<3} {'on wall ' if s['on_wall'] else ''}at ({(x0 + x1) // 2},{(y0 + y1) // 2})")
        color = (0, 0, 220) if s["legend_id"] is None else (200, 0, 0)
        cv2.rectangle(image, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), color, 2)
        cv2.putText(image, s["name"], (x0 - 6, max(12, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    label_image = detect_rooms(room_walls(trace))[0]
    for room, p in sorted(room_patterns(trace, label_image, args.map).items()):
        print(f"  pattern: Room {room} = {p['name']} ({p['description']}) match={p['match']}")
        ys, xs = np.nonzero(label_image == room)
        cv2.putText(image, p["name"], (int(xs.min()) + 6, int(ys.max()) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 140, 0), 1, cv2.LINE_AA)
    out = args.output or os.path.splitext(args.map)[0] + ".symbols.png"
    cv2.imwrite(out, image)
    unknown = sum(s["legend_id"] is None for s in symbols)
    print(f"{len(symbols)} symbols, {unknown} not matched to the legend. Saved {out}")


if __name__ == "__main__":
    main()
