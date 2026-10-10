"""Timing: one enter/exit timer per region the user's finger can be in.

Timers are created for:
  - the paper itself  ("On paper", the main timer: the total time the user
                       interacts with the map -- see InteractionTimer)
  - every room        (from the map JSON's rooms -- the exact pixels
                       room_tracking.py assigned to each room)
  - every symbol      (from the map JSON's symbols -- hover time)
  - any boxes drawn by hand with label_symbols.py (optional extras)

Every room / symbol timer works the same way (RegionTimer):
  - a visit starts the first frame any fingertip is inside the region
  - it ends once no fingertip has been inside for EXIT_GRACE_MS (MediaPipe
    drops a hand for a few frames at a time; a lifted hand also ends it),
    and the visit's end time is the last moment a finger was actually in
  - two hands inside at once count as one visit, not two
  - only visits of at least MIN_VISIT_MS (1 second) count as visits;
    shorter ones are "brushes" (a finger passing over)
"""

import json
import os

MIN_VISIT_MS = 1000
EXIT_GRACE_MS = 400


class RegionTimer:
    def __init__(self, name, kind, contains, geometry, room=None):
        self.name = name
        self.kind = kind              # "paper", "room", "symbol", or a manual box's type
        self.contains = contains      # fn(map_x, map_y, tip) -> bool
        self.geometry = geometry      # {"polygon": [...]} or {"bbox": [...]}
        self.room = room
        self.visits = []              # {"enter_ms", "exit_ms", "duration_ms", "hands"}
        self.brushes = 0
        self._start = None
        self._last_inside = None
        self._hands = set()

    @property
    def occupied(self):
        return self._start is not None

    def step(self, hands_inside, timestamp_ms):
        if hands_inside:
            if self._start is None:
                self._start, self._hands = timestamp_ms, set()
            self._hands |= hands_inside
            self._last_inside = timestamp_ms
        elif self._start is not None and timestamp_ms - self._last_inside > EXIT_GRACE_MS:
            self.close()

    def close(self):
        if self._start is None:
            return
        duration = self._last_inside - self._start
        if duration >= MIN_VISIT_MS:
            self.visits.append({"enter_ms": self._start, "exit_ms": self._last_inside,
                                "duration_ms": duration, "hands": sorted(self._hands)})
        else:
            self.brushes += 1
        self._start = None

    def summary(self):
        durations = [v["duration_ms"] for v in self.visits]
        touched = bool(self.visits) or self.brushes > 0
        return {
            "type": self.kind,
            **self.geometry,
            "room": self.room,
            # visited = at least one visit >= 1 s; brushed = only shorter
            # touches; missed = never touched at all
            "status": "visited" if self.visits else "brushed" if touched else "missed",
            "touched": touched,
            "missed": not touched,
            "visits": len(self.visits),
            "brushes": self.brushes,
            "total_ms": int(sum(durations)),
            "longest_ms": int(max(durations, default=0)),
            "first_enter_ms": self.visits[0]["enter_ms"] if self.visits else None,
            "events": self.visits,
        }


class InteractionTimer(RegionTimer):
    """The main timer: total time the user interacts with the map.

    A stopwatch with no grace period and no minimum: it runs on every frame
    at least one index fingertip is inside the map's pixels, and pauses on
    the first frame none is (hands lifted, off the paper, or the map not in
    the frame at all). Each run from start to pause is one "visit", and
    total_ms is their sum.
    """

    def __init__(self, width, height):
        super().__init__("On paper", "paper", lambda x, y, tip: tip["on_paper"],
                         {"bbox": [0, 0, width - 1, height - 1]})
        self.now_ms = 0

    @property
    def running(self):
        return self._start is not None

    def elapsed_ms(self):
        """Total so far, including the run in progress."""
        done = sum(v["duration_ms"] for v in self.visits)
        return done + (self.now_ms - self._start if self.running else 0)

    def step(self, hands_inside, timestamp_ms):
        self.now_ms = timestamp_ms
        if hands_inside:
            if self._start is None:
                self._start, self._hands = timestamp_ms, set()
            self._hands |= hands_inside
        elif self._start is not None:
            self.close()   # paused as of this frame

    def close(self):
        if self._start is None:
            return
        if self.now_ms > self._start:
            self.visits.append({"enter_ms": self._start, "exit_ms": self.now_ms,
                                "duration_ms": self.now_ms - self._start,
                                "hands": sorted(self._hands)})
        self._start = None


def _in_bbox(bbox):
    x0, y0, x1, y1 = bbox
    return lambda x, y, tip: x0 <= x <= x1 and y0 <= y <= y1


def load_manual_boxes(map_path, ref_w, ref_h):
    """Boxes drawn with label_symbols.py (<map>.symbols.json), in PNG pixels.
    Older files were drawn on the map rotated 180 degrees (no "space": "png"
    field) and are rotated back here."""
    path = os.path.splitext(map_path)[0] + ".symbols.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    boxes = {}
    for name, entry in raw.items():
        if isinstance(entry, (list, tuple)):
            entry = {"type": "symbol", "box": entry}
        fx0, fy0, fx1, fy1 = entry["box"]
        if entry.get("space") != "png":
            fx0, fy0, fx1, fy1 = 1 - fx1, 1 - fy1, 1 - fx0, 1 - fy0
        boxes[name] = {"type": entry.get("type", "symbol"),
                       "bbox": [fx0 * ref_w, fy0 * ref_h, fx1 * ref_w, fy1 * ref_h]}
    print(f"Loaded {len(boxes)} hand-drawn box(es) from {path}")
    return boxes


class Timing:
    """All the timers for one session, stepped once per frame."""

    def __init__(self, room_map, manual_boxes=None):
        self.room_map = room_map
        w, h = room_map.width, room_map.height
        self.paper = InteractionTimer(w, h)
        self.rooms = {}
        for r in room_map.rooms:
            rid = r["id"]
            self.rooms[rid] = RegionTimer(
                r["name"], "room", lambda x, y, tip, rid=rid: room_map.room_at(x, y) == rid,
                {"polygon": r["polygon"]})
        self.symbols = [RegionTimer(s["name"], "symbol", _in_bbox(s["bbox"]),
                                    {"bbox": s["bbox"]},
                                    room=room_map.room_names.get(s["room"]))
                        for s in room_map.symbols]
        self.manual = [RegionTimer(name, b["type"], _in_bbox(b["bbox"]),
                                   {"bbox": [round(v, 1) for v in b["bbox"]]})
                       for name, b in (manual_boxes or {}).items()]

    def all(self):
        return [self.paper, *self.rooms.values(), *self.symbols, *self.manual]

    def step(self, tips, timestamp_ms):
        """tips: this frame's fingertips from FingerTracker.update."""
        for timer in self.all():
            inside = {t["hand"] for t in tips if timer.contains(t["map_x"], t["map_y"], t)}
            timer.step(inside, timestamp_ms)

    def occupied(self):
        return [t for t in self.all() if t.occupied]

    def finish(self):
        for timer in self.all():
            timer.close()

    def summaries(self):
        """{name: summary} for every timer except the paper one."""
        return {t.name: t.summary() for t in self.all() if t is not self.paper}

    def sequence(self):
        """Every >= 1 s room/symbol visit, in the order they happened."""
        events = [{"name": t.name, "type": t.kind, **v}
                  for t in self.all() if t is not self.paper for v in t.visits]
        return sorted(events, key=lambda e: e["enter_ms"])
