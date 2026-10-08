"""Room tracking: which exact pixels of the map belong to which room.

Rooms aren't labeled anywhere in the PNG -- they're just the white spaces
enclosed by the printed walls. The problem is the doorways: every room has
gaps in its walls, so if you flood-fill the white space, all the rooms
leak into each other (and into the corridor) through those gaps.

How this finds them (from the wall mask edge_tracing.py produces):
  1. Thicken the walls just enough to seal the doorway gaps (a "closing
     gap" in px). Now every room is a sealed white area.
  2. Each sealed white area that doesn't touch the image border, and isn't
     tiny, is one room. The thickened walls have eaten into each room's
     edges, so these are only the room *seeds*.
  3. Grow every seed back out to the real walls: each non-wall pixel inside
     the map's outer border goes to whichever room seed it's closest to.
     Doorway pixels are split between the two rooms on either side.

The result is a label image the size of the PNG: label_image[y, x] is the
room id (1..N) that pixel belongs to, or 0 for walls / outside the map.
That's the exact per-pixel answer the room timers need.

The doorway gap size is picked automatically: too small and rooms leak
into each other (fewer, bigger rooms), too big and narrow rooms vanish.
The room count stays flat over the range of "right" sizes, so this tries a
range and keeps the smallest gap from the most common (stable) count.
Pass gap=... to override.

Symbols are the symbol marks edge tracing finds, each tagged with the room
it sits in, so a symbol's timer can say "the Elevator symbol in Room 4".
"""

import cv2
import numpy as np

from edge_tracing import CLASS_WALL, classify_shape

MIN_ROOM_AREA_FRACTION = 0.002   # of the map's area; smaller sealed areas are noise
SYMBOL_PADDING = 6               # px added around each symbol's box for its timer


def _rooms_for_gap(walls, gap):
    h, w = walls.shape
    sealed = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (sealed == 0).astype(np.uint8), connectivity=4)
    touches_border = set(np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    min_area = MIN_ROOM_AREA_FRACTION * h * w
    seeds = [i for i in range(1, n)
             if i not in touches_border and stats[i, cv2.CC_STAT_AREA] >= min_area]
    return labels, seeds


def choose_gap(walls, gaps=None):
    """Smallest doorway-sealing gap within the most stable room count."""
    h, w = walls.shape
    if gaps is None:
        base = min(h, w)
        gaps = sorted({max(5, int(round(f * base))) | 1
                       for f in np.linspace(0.015, 0.09, 16)})
    counts = [len(_rooms_for_gap(walls, g)[1]) for g in gaps]
    values, freq = np.unique(counts, return_counts=True)
    stable = values[freq == freq.max()].max()   # ties -> more rooms
    return gaps[counts.index(stable)], dict(zip(gaps, counts))


def detect_rooms(walls, gap=None):
    """Return (label_image, rooms, gap_used, tried) for a wall mask.

    rooms: list of dicts with id, name, area_px, centroid, bbox and polygon
    (the room's outline in PNG pixel coordinates).
    """
    h, w = walls.shape
    tried = None
    if gap is None:
        gap, tried = choose_gap(walls)
    labels, seeds = _rooms_for_gap(walls, gap)

    # Name rooms in reading order (top-to-bottom, then left-to-right).
    centres = {i: np.argwhere(labels == i).mean(axis=0) for i in seeds}
    row_band = max(1.0, 0.04 * h)
    seeds.sort(key=lambda i: (round(centres[i][0] / row_band), centres[i][1]))
    seed_img = np.zeros((h, w), np.int32)
    for room_id, i in enumerate(seeds, 1):
        seed_img[labels == i] = room_id

    # Map outline: the filled outer contour of the (sealed) wall network.
    sealed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
    contours, _ = cv2.findContours(sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inside_map = np.zeros((h, w), np.uint8)
    if contours:
        cv2.drawContours(inside_map, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    # Grow seeds back to the walls: nearest-seed label for every pixel.
    _, nearest = cv2.distanceTransformWithLabels(
        (seed_img == 0).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_CCOMP)
    lut = np.zeros(int(nearest.max()) + 1, np.int32)
    ys, xs = np.nonzero(seed_img)
    lut[nearest[ys, xs]] = seed_img[ys, xs]
    label_image = lut[nearest]
    label_image[(inside_map == 0) | (walls > 0)] = 0

    rooms = []
    for room_id in range(1, len(seeds) + 1):
        mask = (label_image == room_id).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max(cs, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(outline)
        yy, xx = np.nonzero(mask)
        rooms.append({
            "id": room_id,
            "name": f"Room {room_id}",
            "area_px": int(mask.sum()),
            "centroid": [round(float(xx.mean()), 1), round(float(yy.mean()), 1)],
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "polygon": outline.reshape(-1, 2).tolist(),
        })
    return label_image, rooms, gap, tried


def detect_symbols(trace, label_image):
    """Symbols from edge tracing's symbol marks, each tagged with its room."""
    h, w = label_image.shape
    symbols = []
    for k, (x0, y0, x1, y1) in enumerate(sorted(
            trace.symbol_marks(), key=lambda b: (b[1] // 40, b[0])), 1):
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        room = int(label_image[min(cy, h - 1), min(cx, w - 1)])
        p = SYMBOL_PADDING
        symbols.append({
            "id": k,
            "name": f"Symbol {k}",
            "shape": classify_shape(trace.ink, (x0, y0, x1, y1)),
            "room": room or None,
            "bbox": [max(0, x0 - p), max(0, y0 - p), min(w - 1, x1 + p), min(h - 1, y1 + p)],
        })
    return symbols


class RoomMap:
    """Rooms + symbols of one map, with fast per-pixel lookups.

    Built from the "rooms" and "symbols" sections of the map JSON (see
    paper_locator.load_reference_map), so the lookup always matches exactly
    what's saved in that file.
    """

    def __init__(self, width, height, rooms, symbols):
        self.width, self.height = width, height
        self.rooms = rooms
        self.symbols = symbols
        self.label_image = np.zeros((height, width), np.int32)
        for room in rooms:
            cv2.fillPoly(self.label_image, [np.int32(room["polygon"])], room["id"])
        self.room_names = {r["id"]: r["name"] for r in rooms}

    def room_at(self, x, y):
        """Room id at map pixel (x, y), or 0 if it's a wall / off the map."""
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return int(self.label_image[yi, xi])
        return 0

    def symbols_at(self, x, y):
        return [s["id"] for s in self.symbols
                if s["bbox"][0] <= x <= s["bbox"][2] and s["bbox"][1] <= y <= s["bbox"][3]]

    def draw(self, image):
        """Colored room overlay with names, for checking the detection."""
        rng = np.random.default_rng(7)
        colors = rng.integers(70, 240, (len(self.rooms) + 1, 3)).astype(np.uint8)
        colors[0] = 0
        tint = colors[self.label_image]
        out = image.copy()
        has_room = self.label_image > 0
        out[has_room] = (0.55 * out[has_room] + 0.45 * tint[has_room]).astype(np.uint8)
        for room in self.rooms:
            cx, cy = (int(v) for v in room["centroid"])
            cv2.putText(out, room["name"], (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2, cv2.LINE_AA)
        for s in self.symbols:
            x0, y0, x1, y1 = s["bbox"]
            cv2.rectangle(out, (x0, y0), (x1, y1), (200, 0, 0), 2)
            cv2.putText(out, s["name"], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 0, 0), 1, cv2.LINE_AA)
        return out
