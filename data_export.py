"""This file writes the run's final data files, the CSV tables and
corner-motion log among them, into the run folder at the end of a run.

The run's final data files, for analysis outside this project.

Written at the end of every run (run_pipeline.py calls save_data_files),
into the run folder next to map.json and finger_paths.png:

  summary.csv          the whole map, then one row per room, then any symbol
                       that belongs to no room:
                         Room Symbol, Room Symbol Visits, Room Symbol Time,
                         Entrance Symbol, Entrance Symbol Visits, Entrance Symbol Time,
                         Room Visits, Room Time, Room Dimensions
                       Room Symbol: the symbol(s) inside the room, or the area
                       pattern the room is filled with (a corridor's dots,
                       dashes or lines, a hatched area), with the room's name:
                       "Stairs #5 (Room 1)", "Corridor #28 (Room 6)". A
                       pattern covers the whole room, so its visits and time
                       are the room's own. Entrance Symbol: the mark(s) on the
                       walls leading into the room. Each of the three has its
                       own timer; two symbols in one cell are added together.
                       Times in seconds, dimensions in inches (width, height).
  room_order.csv       every room visit in order, revisits included:
                         Order, Time Entered (s), Time Exited (s)
  corner_motion.csv    a few times a second, how far each paper corner moved
                       since the previous sample (frame pixels):
                         Frame #, Timestamp (s), TL dx, TL dy, ... BL dx, BL dy
  corner_motion.json   the corner vectors, one list per corner: {"TL": [[dx, dy], ...], ...}
  finger_positions.json  every frame's index fingertips in map (PNG) pixels
  floorplan.png        a copy of the map PNG the run used

Each CSV is one plain table (header row, then data), so it opens as-is in
Google Sheets or Excel. Times are video time in plain seconds (12.34), not
M:SS.ss, which Sheets misreads as a clock time. Visits follow timing.py's
rules (a visit is >= 1 s; shorter touches aren't counted). The "Entire
map" row's Room Time is the main timer: total time at least one index
finger was on the map (timing.InteractionTimer).

Room labels (summary.csv's Room Symbol, room_order.csv's Order): a room is
named after the free-standing symbol(s) inside it, or, failing that, its
area pattern, followed by the room's own name from map.json -- e.g.
"Stairs #5 (Room 1)", "Corridor #28 (Room 6)"; a room with neither is just
"Room 4". Symbol and pattern names come from the legend (see
symbol_identification.py). Renaming a room in map.json (say to "Office")
carries through: "Toilet #2 (Office)".
"""

import csv
import json
import os
import shutil

import numpy as np

CORNER_NAMES = ("TL", "TR", "BR", "BL")   # the map's own corners, as AutoCrop orders them
CORNER_SAMPLES_PER_SECOND = 4


def clock(ms):
    """Milliseconds of video time -> "M:SS.ss"."""
    if ms is None:
        return ""
    minutes, seconds = divmod(ms / 1000.0, 60)
    return f"{int(minutes)}:{seconds:05.2f}"


def seconds(ms):
    """Milliseconds of video time -> seconds, 2 decimals, for the CSVs."""
    return "" if ms is None else round(ms / 1000.0, 2)


# ---------------------------------------------------------------- corner motion

class CornerMotion:
    """Samples the crop's 4 corners CORNER_SAMPLES_PER_SECOND times a second
    and keeps, for each sample, how far each corner moved (in whole frame
    pixels) since the sample before. A sample where the map isn't in the
    frame -- or the first one after it comes back -- has no vectors (None)."""

    def __init__(self, samples_per_second=CORNER_SAMPLES_PER_SECOND):
        self.every_ms = 1000.0 / samples_per_second
        self.next_ms = 0.0
        self.last = None
        self.samples = []   # {"frame", "t_ms", "vectors": [(dx, dy) x 4] or None}

    def update(self, frame_index, timestamp_ms, corners):
        if timestamp_ms < self.next_ms:
            return
        while self.next_ms <= timestamp_ms:
            self.next_ms += self.every_ms
        now = None if corners is None else np.rint(np.asarray(corners, np.float64)).astype(int)
        vectors = None
        if now is not None and self.last is not None:
            vectors = [(int(dx), int(dy)) for dx, dy in now - self.last]
        self.samples.append({"frame": frame_index, "t_ms": timestamp_ms, "vectors": vectors})
        self.last = now

    def to_json(self):
        """{corner name: [[dx, dy] or None per sample]} -- one list per corner,
        lined up with the rows of corner_motion.csv."""
        return {name: [None if s["vectors"] is None else list(s["vectors"][i]) for s in self.samples]
                for i, name in enumerate(CORNER_NAMES)}

    def rows(self):
        out = []
        for s in self.samples:
            vec = [""] * 8 if s["vectors"] is None else [d for v in s["vectors"] for d in v]
            out.append([s["frame"], seconds(s["t_ms"])] + vec)
        return out


# ---------------------------------------------------------------- summary + order

def room_symbols(room_map):
    """({room id: [symbol ids inside it]}, {room id: [entrance symbol ids
    leading into it]})."""
    inside = {r["id"]: [] for r in room_map.rooms}
    leading_in = {r["id"]: [] for r in room_map.rooms}
    for s in room_map.symbols:
        if s.get("entrance"):
            for rid in s.get("entrance_to", []):
                if rid in leading_in:
                    leading_in[rid].append(s["id"])
        elif s.get("room") in inside:
            inside[s["room"]].append(s["id"])
    return inside, leading_in


def room_labels(room_map):
    """({room id: label}, {room id: [ids of the symbols inside it]}) -- see
    the module notes."""
    inside, _ = room_symbols(room_map)
    names_by_id = {s["id"]: s["name"] for s in room_map.symbols}
    labels = {}
    for r in room_map.rooms:
        names = (" / ".join(names_by_id[i] for i in inside[r["id"]])
                 or (r.get("pattern") or {}).get("name"))
        labels[r["id"]] = f"{names} ({r['name']})" if names else r["name"]
    return labels, inside


def _visits_and_time(timers):
    visits = sum(len(t.visits) for t in timers)
    total = sum(v["duration_ms"] for t in timers for v in t.visits)
    return visits, total


def _cells(name, timers):
    """[name, visits, time] for one symbol cell; blanks if there's no symbol."""
    if not timers:
        return ["", "", ""]
    visits, ms = _visits_and_time(timers)
    return [name, visits, seconds(ms)]


def summary_rows(timing, room_map, inches):
    labels, _ = room_labels(room_map)
    inside, leading_in = room_symbols(room_map)
    symbol_timers = {t.symbol["id"]: t for t in timing.symbols}
    rows = [["Entire map", "", "", "", "", "", "", seconds(timing.paper.elapsed_ms()),
             f"({inches.width_in:g}, {inches.height_in:g})"]]
    used = set()
    for r in room_map.rooms:
        rid = r["id"]
        room_timer = timing.rooms[rid]
        own = [symbol_timers[i] for i in inside[rid] if i in symbol_timers]
        doors = [symbol_timers[i] for i in leading_in[rid] if i in symbol_timers]
        used.update(inside[rid] + leading_in[rid])
        if own:
            room_symbol = _cells(labels[rid], own)
        elif r.get("pattern"):   # the pattern fills the room: its timer is the room's
            room_symbol = _cells(labels[rid], [room_timer])
        else:
            room_symbol = [labels[rid], "", ""]
        w, h = r.get("size_in") or inches.bbox_size(r["bbox"])
        rows.append(room_symbol + _cells(" / ".join(t.name for t in doors), doors)
                    + [len(room_timer.visits), seconds(sum(v["duration_ms"] for v in room_timer.visits)),
                       f"({w:g}, {h:g})"])
    for sid, t in symbol_timers.items():   # symbols that belong to no room
        if sid not in used:
            cells = _cells(t.name, [t])
            rows.append((["", "", ""] + cells if t.entrance else cells + ["", "", ""]) + ["", "", ""])
    return rows


def order_rows(timing, room_map):
    labels, _ = room_labels(room_map)
    visits = sorted(((v["enter_ms"], v["exit_ms"], labels[rid])
                     for rid, t in timing.rooms.items() for v in t.visits))
    return [[label, seconds(enter), seconds(exit_)] for enter, exit_, label in visits]


# ---------------------------------------------------------------- writing

SUMMARY_HEADER = ["Room Symbol", "Room Symbol Visits", "Room Symbol Time",
                  "Entrance Symbol", "Entrance Symbol Visits", "Entrance Symbol Time",
                  "Room Visits", "Room Time", "Room Dimensions"]
ORDER_HEADER = ["Order", "Time Entered (s)", "Time Exited (s)"]
CORNER_HEADER = ["Frame #", "Timestamp (s)"] + [f"{name} {axis}" for name in CORNER_NAMES
                                                for axis in ("dx", "dy")]


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def finger_positions(tracker):
    """Every frame's index fingertips in map pixels; None = that hand not seen."""
    def at(tip):
        return None if tip is None else [tip["map_x"], tip["map_y"]]
    return [{"frame": r["frame"], "t_ms": r["t_ms"], "Left": at(r["Left"]), "Right": at(r["Right"])}
            for r in tracker.session_log]


def save_data_files(run_dir, timing, ref, tracker, corner_motion):
    """Write every file listed in the module notes into run_dir. Call after
    timing.finish()."""
    out = lambda name: os.path.join(run_dir, name)
    _write_csv(out("summary.csv"), SUMMARY_HEADER, summary_rows(timing, ref.rooms, ref.inches))
    _write_csv(out("room_order.csv"), ORDER_HEADER, order_rows(timing, ref.rooms))
    _write_csv(out("corner_motion.csv"), CORNER_HEADER, corner_motion.rows())
    with open(out("corner_motion.json"), "w", encoding="utf-8") as f:
        json.dump(corner_motion.to_json(), f)
    with open(out("finger_positions.json"), "w", encoding="utf-8") as f:
        json.dump(finger_positions(tracker), f)
    shutil.copyfile(ref.path, out("floorplan" + os.path.splitext(ref.path)[1]))
