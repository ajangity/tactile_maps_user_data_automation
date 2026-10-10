"""Symbol identification: what each symbol on the map *is*, by name.

edge_tracing.py finds where the symbols are, but only as anonymous marks
("Symbol 1", "Symbol 2", ...). This file gives each one a real name
("Stairs zigzag", "Elevator E 2", "Door P6 1"), so the video overlay, rooms.png and the
dashboard say what the viewer is looking at.

How a symbol gets its name, in three steps:

  1. FIND. Free-standing symbols come from edge tracing's symbol marks.
     Symbols drawn *on* a wall (a circle on a wall, a tick across it, a
     letter straddling it) are invisible to edge tracing, because they're
     connected to the wall network and so count as "wall". find_wall_symbols
     recovers them: erase every long straight run of the walls, and whatever
     ink is left over is a symbol. Small hollow rings (which edge tracing
     files under "dots") are picked up too.

  2. RECOGNIZE THE SHAPE. Each symbol's ink is scaled into a 48x48 box and
     thinned to 1 px centre lines, then compared with a small library of
     shape templates drawn in code (TEMPLATES below): stairs, E, star, plus,
     ring, ... The comparison is a
     chamfer distance -- on average, how many pixels apart are the two
     drawings' lines -- tried at all four 90-degree rotations. The closest
     template wins if it's close enough (MATCH_MAX); otherwise the symbol
     is "unknown" and keeps a generic name.

  3. NAME IT. LEGEND turns a shape into a display name. A shape only has a
     *meaning* by convention, and ours is the symbol catalogue of the
     tactile-symbols user study: each name is the referent the shape stands
     for plus its catalogue code ("Elevator P27", "Door P6"). That table is
     the one place to edit, and a map can override it with its own
     <map>.legend.json:

         {"star": "Emergency exit", "bar@wall": "Window"}

     If the same name occurs more than once it's numbered in reading order
     ("Elevator E 1", "Elevator E 2"), because every timer needs its own name.

  4. WRITE IT TO THE MAP JSON. detect_symbols builds the "symbols" section
     of map.json: every symbol with its name, shape, room and box.

  5. TIME IT. SymbolTiming gives every symbol its own timer. A symbol on a
     wall marks a room's entrance (a door circle, a tick, the sideways B),
     so its timer is an "entrance" timer: it covers a little space around
     the mark and records which room the entrance leads into. The results
     land in dashboard.json under "boxes", like every other timer.

Orientation is part of some symbols: a square missing its top side is a
toilet (P26), missing its right side an elevator (P27), missing its bottom
side a door (P29); an upright E is the elevator's first letter, the same E
turned clockwise is the separate symbol P72.

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
from timing import RegionTimer, Timing

# ---- finding symbols drawn on walls
WALL_RUN_MIN = 45         # px; a straight wall run at least this long is "wall", not symbol
WALL_SYMBOL_MIN = 10      # px; smaller leftovers are corner/junction crumbs
WALL_SYMBOL_MAX = 80      # px; bigger leftovers are diagonal walls / hatching, not a symbol
GROUP_GAP = 9             # px; leftover pieces this close together are one symbol

# ---- timing
SYMBOL_PADDING = 6        # px added around each symbol's box for its timer
ENTRANCE_REACH = 18       # px around an entrance symbol that count as "at the entrance"

# ---- recognizing shapes
THIN_SIZE = 96            # symbols are enlarged to this before thinning, to keep detail
NORM_SIZE = 48            # symbols and templates are compared in a box this size
NORM_INK = 40             # ... with the shape's longer side scaled to this
MATCH_MAX = 1.5           # px (in the 48x48 box); a worse best match is "unknown"

# Shape -> display name: what the shape stands for, plus its code in the
# study's symbol catalogue ("Tactile Symbols User Study", Referents &
# Symbols table). Codes are the regular-size ones; the smaller prints of
# the same shape have their own codes there (P30 for P26, P31 for P27...).
# A key ending in "@wall" is used instead when the symbol is drawn on a wall.
LEGEND = {
    "stairs": "Stairs zigzag",              # Nick Giudice's suggested stair symbol
    "E": "Elevator E",                      # embossed first letter
    "E-down": "Elevator P72",               # E turned 90 degrees clockwise
    "E-rotated": "Rotated E",
    "open-box-up": "Toilet P26",            # square without its top side
    "open-box-right": "Elevator P27",       # square without its right side
    "open-box-down": "Door P29",            # square without its bottom side
    "open-box-left": "Open box",
    "split-box": "Elevator P54",            # two rectangles side by side
    "bullseye": "Toilet P9",                # circle with a filled circle inside
    "L": "Toilet P34",
    "rectangle": "Stairs P40",
    "open-rectangle": "Stairs P35",         # rectangle without one short side
    "four-boxes": "Stairs P7",              # four rectangles side by side
    "three-lines": "Stairs P84",
    "four-rings": "Emergency exit P99",     # four small circles in a diamond
    "ring-line": "Emergency exit P48",      # circle with a line through its centre
    "ring-line@wall": "Door P6",            # ... unless that line is the wall it sits on
    "ring": "Door P6",
    "T": "Door P95",
    "bar@wall": "Door P95",                 # the T's bar crossing a wall
    "bar": "Bar",
    "caret": "Door P86",
    "ring-slash": "Inaccessible area P47",  # circle with a diagonal line
    "plus": "Dangerous area P10",
    "x": "Dangerous area P98",
    "triangle": "Dangerous area P17",
    "diamond": "Dangerous area P42",
    "star": "Reception P41",
    "H": "Reception P74",
    "square": "Reception P11",
    "B": "Letter B",
    "B-rotated": "Rotated B",
    "unknown": "Unknown symbol",
}


# ---------------------------------------------------------------- 1. find

def find_wall_symbols(trace):
    """(mask, boxes) of the symbols edge tracing counted as wall or dots.

    mask is their ink (0/255); boxes are (x0, y0, x1, y1), one per symbol.
    """
    walls = trace.walls
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


def _overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def find_symbols(trace):
    """Every symbol on the map: (ink_mask, [(box, on_wall), ...]).

    A free-standing mark that overlaps a wall symbol (the inner rings of a
    bullseye whose outer ring touches a wall) is merged into it.
    """
    wall_ink, wall_boxes = find_wall_symbols(trace)
    ink = np.where(trace.classes == CLASS_SYMBOL, 255, 0).astype(np.uint8) | wall_ink
    found = [(box, True) for box in wall_boxes]
    for box in trace.symbol_marks():
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


def _lines(*strokes, size=(200, 200), closed=False, thickness=12):
    canvas = np.zeros((size[1], size[0]), np.uint8)
    for stroke in strokes:
        cv2.polylines(canvas, [np.int32(stroke)], closed, 255, thickness, cv2.LINE_AA)
    return canvas


def _rings(*circles, size=(200, 200), thickness=10):
    canvas = np.zeros((size[1], size[0]), np.uint8)
    for cx, cy, r in circles:
        cv2.circle(canvas, (cx, cy), r, 255, thickness, cv2.LINE_AA)
    return canvas


def _star(*sizes):
    stars = []
    for size in sizes:
        pts = []
        for k in range(10):
            r = size if k % 2 == 0 else size * 0.42
            a = -np.pi / 2 + k * np.pi / 5
            pts.append((100 + r * np.cos(a), 100 + r * np.sin(a)))
        stars.append(pts)
    return _lines(*stars, closed=True, thickness=6)


def _letter(ch):
    canvas = np.zeros((260, 260), np.uint8)
    cv2.putText(canvas, ch, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 8, 255, 14, cv2.LINE_AA)
    return canvas


def _e(width):
    # spine + three bars; the middle bar is a little shorter, like the print
    return _lines([(width, 0), (0, 0), (0, 200), (width, 200)], [(0, 100), (width * 0.9, 100)],
                  size=(width, 200))


def _open_box(height):
    return _lines([(0, 0), (0, height), (200, height), (200, 0)], size=(200, height))


def _grid(columns):
    w = 60 * columns
    return _lines([(0, 0), (w, 0), (w, 200), (0, 200)], closed=True, size=(w, 200)) | \
        _lines(*[[(60 * k, 0), (60 * k, 200)] for k in range(1, columns)], size=(w, 200))


UPRIGHT = {0: ""}                                    # any other way round: "<kind>-rotated"
OPENING = {0: "-up", 1: "-right", 2: "-down", 3: "-left"}

# kind -> (drawings, orientation). A kind may have several drawings
# (different proportions of the same shape). orientation is None when the
# shape means the same any way round; otherwise it maps the number of
# quarter turns (anticlockwise) that bring the symbol onto the drawing to a
# suffix for the kind, because on these maps a turned shape is a different
# symbol: an open box is a toilet, an elevator or a door depending on which
# side is open.
TEMPLATES = {
    "stairs": ([_lines([(0, 0), (100, 0), (100, 100), (200, 100), (200, 200)]),   # two steps, or three
                _lines([(0, 0), (66, 0), (66, 66), (133, 66), (133, 133), (200, 133), (200, 200)])], None),
    "E": ([_e(150), _e(100)], {0: "", 1: "-down"}),
    "B": ([_letter("B")], UPRIGHT),
    "open-box": ([_open_box(200)], OPENING),
    "open-rectangle": ([_open_box(500)], None),
    "split-box": ([_grid(2), _lines([(0, 0), (200, 0), (200, 200), (0, 200)], closed=True) |
                   _lines([(100, 0), (100, 200)])], None),
    "four-boxes": ([_grid(4)], None),
    "three-lines": ([_lines([(0, 0), (200, 0)], [(0, 70), (200, 70)], [(0, 140), (200, 140)],
                            size=(200, 140))], None),
    "L": ([_lines([(0, 0), (0, 200), (130, 200)], size=(130, 200))], None),
    "H": ([_lines([(0, 0), (0, 200)], [(160, 0), (160, 200)], [(0, 100), (160, 100)], size=(160, 200))], None),
    "T": ([_lines([(0, 0), (0, 200)], [(0, 100), (150, 100)], size=(150, 200))], None),
    "caret": ([_lines([(0, 100), (100, 0), (200, 100)], size=(200, 100))], None),
    "plus": ([_lines([(100, 0), (100, 200)], [(0, 100), (200, 100)])], None),
    "x": ([_lines([(0, 0), (200, 200)], [(200, 0), (0, 200)])], None),
    "triangle": ([_lines([(100, 0), (200, 180), (0, 180)], closed=True, size=(200, 180))], None),
    "diamond": ([_lines([(100, 0), (200, 100), (100, 200), (0, 100)], closed=True)], None),
    "star": ([_star(95), _star(95, 45)], None),   # outline, or a star in a star
    "rectangle": ([_lines([(0, 0), (200, 0), (200, 100), (0, 100)], size=(200, 100), closed=True)], None),
    "square": ([_lines([(0, 0), (200, 0), (200, 200), (0, 200)], closed=True)], None),
    "ring": ([_rings((100, 100, 90))], None),
    "ring-line": ([_rings((100, 100, 90)) | _lines([(100, 10), (100, 190)], thickness=10)], None),
    "ring-slash": ([_rings((100, 100, 90)) | _lines([(36, 36), (164, 164)], thickness=10)], None),
    "bullseye": ([_rings((100, 100, 90), (100, 100, 28), (100, 100, 8))], None),
    "four-rings": ([_rings((100, 22, 20), (100, 178, 20), (22, 100, 20), (178, 100, 20))], None),
    "bar": ([_lines([(0, 6), (200, 6)], size=(200, 12))], None),
}

_template_cache = None


def _templates():
    global _template_cache
    if _template_cache is None:
        _template_cache = [(kind, normalize(drawing), orientation)
                           for kind, (drawings, orientation) in TEMPLATES.items()
                           for drawing in drawings]
    return _template_cache


def _chamfer(a, b):
    """Average distance (px) from each drawing's lines to the other's, both ways."""
    da = cv2.distanceTransform(np.where(a > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform(np.where(b > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    return 0.5 * (float(db[a > 0].mean()) + float(da[b > 0].mean()))


def recognize(ink, box):
    """(kind, rotation_degrees, distance) for the symbol ink inside box.

    kind is a key of LEGEND; "unknown" if nothing in TEMPLATES is close.
    """
    x0, y0, x1, y1 = [int(v) for v in box]
    shape = normalize(ink[max(0, y0):y1 + 1, max(0, x0):x1 + 1])
    if shape is None:
        return "unknown", 0, float("inf")
    best = ("unknown", 0, float("inf"))
    for kind, template, orientation in _templates():
        for quarter in range(4):
            d = _chamfer(np.rot90(shape, quarter), template)
            if d < best[2]:
                suffix = "" if orientation is None else orientation.get(quarter, "-rotated")
                best = (kind + suffix, quarter * 90, d)
    if best[2] > MATCH_MAX:
        return "unknown", 0, best[2]
    return best


# ---------------------------------------------------------------- 3. name

def load_legend(map_path=None):
    """LEGEND, with <map>.legend.json's entries laid over it if that exists."""
    legend = dict(LEGEND)
    path = os.path.splitext(map_path)[0] + ".legend.json" if map_path else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            legend.update(json.load(f))
    return legend


def is_default_name(name):
    """True for the placeholder names ("Symbol 7") this file replaces."""
    head, _, tail = str(name).rpartition(" ")
    return head == "Symbol" and tail.isdigit()


def name_symbols(symbols, legend=None):
    """Give every symbol dict (with a "kind") a unique "name", in place."""
    legend = legend or LEGEND
    base = [(s.get("on_wall") and legend.get(s["kind"] + "@wall")) or legend.get(s["kind"], s["kind"])
            for s in symbols]
    seen = {}
    for s, name in zip(symbols, base):
        if base.count(name) > 1:
            seen[name] = seen.get(name, 0) + 1
            s["name"] = f"{name} {seen[name]}"
        else:
            s["name"] = name
    return symbols


def identify_symbols(trace, map_path=None):
    """Every symbol on the map, found, recognized and named.

    Returns dicts with kind, name, rotation, match (chamfer distance; lower
    is surer), on_wall and box (x0, y0, x1, y1), in reading order.
    """
    ink, found = find_symbols(trace)
    symbols = []
    for box, on_wall in sorted(found, key=lambda f: (f[0][1] // 40, f[0][0])):
        kind, rotation, distance = recognize(ink, box)
        symbols.append({"kind": kind, "rotation": rotation, "match": round(distance, 2),
                        "on_wall": on_wall, "box": box})
    return name_symbols(symbols, load_legend(map_path))


# ---------------------------------------------------------------- 4. map JSON

def corridor_id(trace, label_image):
    """The room that holds most of the texture dots (the corridor), or 0."""
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
            "kind": found["kind"],
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
        symbols.append(symbol)
    return symbols


def restore_auto_names(previous_json, symbols):
    """Undo the carry-over of names nobody edited.

    paper_locator copies every name from the previous run's map JSON onto
    the new symbols. That is right for a name someone typed in by hand,
    but a name that was still the generated one (or an old "Symbol 7"
    placeholder) should follow symbol identification instead, so those
    are set back to this run's auto_name.
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
            if prev.get("name") == prev.get("auto_name") or is_default_name(prev.get("name")):
                new["name"] = new["auto_name"]
    return symbols


# ---------------------------------------------------------------- 5. timers

class SymbolTimer(RegionTimer):
    """A symbol's timer. Same visit rules as every other timer (see
    timing.py); it also writes what the symbol is into the results JSON,
    and touch_ms: all the time a finger was on it, including the touches
    shorter than a visit that total_ms leaves out (passing through a
    doorway rarely takes a full second).

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
        self.touch_ms = 0   # all time inside, brushes included

    def close(self):
        if self._start is not None:
            self.touch_ms += self._last_inside - self._start
        super().close()

    def summary(self):
        out = super().summary()
        out["symbol"] = {k: self.symbol[k] for k in ("id", "kind", "rotation", "on_wall")}
        out["touch_ms"] = int(self.touch_ms)
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
    parser = argparse.ArgumentParser(description="Find and name the symbols on a map PNG.")
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
        print(f"  {i:2d}  {s['name']:<22} kind={s['kind']:<14} match={s['match']:<5} "
              f"rot={s['rotation']:<3} {'on wall ' if s['on_wall'] else ''}at ({(x0 + x1) // 2},{(y0 + y1) // 2})")
        color = (0, 0, 220) if s["kind"] == "unknown" else (200, 0, 0)
        cv2.rectangle(image, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), color, 2)
        cv2.putText(image, s["name"], (x0 - 6, max(12, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    out = args.output or os.path.splitext(args.map)[0] + ".symbols.png"
    cv2.imwrite(out, image)
    unknown = sum(s["kind"] == "unknown" for s in symbols)
    print(f"{len(symbols)} symbols, {unknown} not recognized. Saved {out}")


if __name__ == "__main__":
    main()
