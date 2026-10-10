"""The run's final data files, for analysis outside this project.

Written at the end of every run (run_pipeline.py calls save_data_files),
into the run folder next to map.json and finger_paths.png:

  summary.csv          one row per room (labeled by the symbol in it), then
                       one per symbol not used as a label, plus the whole map:
                         Element/Symbol, Symbol Visits, Symbol Time,
                         Room Visits, Room Time, Room Dimension (inches)
  room_order.csv       every room visit in order, revisits included:
                         Order, Time Entered, Time Exited
  corner_motion.csv    a few times a second, how far each paper corner moved
                       since the previous sample:
                         Frame #, Timestamp, Corner Vectors
  corner_motion.json   the same vectors, one list per corner: {"TL": [[dx, dy], ...], ...}
  finger_positions.json  every frame's index fingertips in map (PNG) pixels
  floorplan.png        a copy of the map PNG the run used

Times are video time, written M:SS.ss. Visits follow timing.py's rules (a
visit is >= 1 s; shorter touches aren't counted). The "Entire map" row's
Room Time is the main timer: total time at least one index finger was on
the map (timing.InteractionTimer).

Room labels: a room is named after the free-standing symbol(s) inside it,
or, failing that, the entrance symbol(s) leading into it, followed by the
room's own name from map.json -- e.g. "Stairs zigzag (Room 1)". A room
with no symbol (the corridor) keeps its own name. Renaming a room in
map.json (say to "Office") carries through: "Toilet P26 (Office)".
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
            vec = "" if s["vectors"] is None else "; ".join(
                f"{name} ({dx}, {dy})" for name, (dx, dy) in zip(CORNER_NAMES, s["vectors"]))
            out.append([s["frame"], clock(s["t_ms"]), vec])
        return out


# ---------------------------------------------------------------- summary + order

def room_labels(room_map):
    """({room id: label}, {room id: [symbol ids]}) -- see the module notes."""
    inside = {r["id"]: [] for r in room_map.rooms}
    leading_in = {r["id"]: [] for r in room_map.rooms}
    for s in room_map.symbols:
        if s.get("entrance"):
            for rid in s.get("entrance_to", []):
                if rid in leading_in:
                    leading_in[rid].append(s)
        elif s.get("room") in inside:
            inside[s["room"]].append(s)
    labels, symbol_ids = {}, {}
    for r in room_map.rooms:
        syms = inside[r["id"]] or leading_in[r["id"]]
        symbol_ids[r["id"]] = [s["id"] for s in syms]
        names = " / ".join(s["name"] for s in syms)
        labels[r["id"]] = f"{names} ({r['name']})" if names else r["name"]
    return labels, symbol_ids


def _visits_and_time(timers):
    visits = sum(len(t.visits) for t in timers)
    total = sum(v["duration_ms"] for t in timers for v in t.visits)
    return visits, total


def summary_rows(timing, room_map, inches):
    labels, symbol_ids = room_labels(room_map)
    symbol_timers = {t.symbol["id"]: t for t in timing.symbols}
    rows = [["Entire map", "", "", "", clock(timing.paper.elapsed_ms()),
             f"({inches.width_in:g}, {inches.height_in:g})"]]
    used = set()
    for r in room_map.rooms:
        rid = r["id"]
        room_timer = timing.rooms[rid]
        timers = [symbol_timers[i] for i in symbol_ids[rid] if i in symbol_timers]
        used.update(symbol_ids[rid])
        s_visits, s_ms = _visits_and_time(timers)
        w, h = r.get("size_in") or inches.bbox_size(r["bbox"])
        rows.append([labels[rid], s_visits if timers else "", clock(s_ms) if timers else "",
                     len(room_timer.visits), clock(sum(v["duration_ms"] for v in room_timer.visits)),
                     f"({w:g}, {h:g})"])
    for sid, t in symbol_timers.items():
        if sid in used:
            continue
        name = t.name + (f" (entrance to {t.room})" if t.entrance and t.room else "")
        s_visits, s_ms = _visits_and_time([t])
        rows.append([name, s_visits, clock(s_ms), "", "", ""])
    return rows


def order_rows(timing, room_map):
    labels, _ = room_labels(room_map)
    visits = sorted(((v["enter_ms"], v["exit_ms"], labels[rid])
                     for rid, t in timing.rooms.items() for v in t.visits))
    return [[label, clock(enter), clock(exit_)] for enter, exit_, label in visits]


# ---------------------------------------------------------------- writing

SUMMARY_HEADER = ["Element/Symbol", "Symbol Visits", "Symbol Time", "Room Visits", "Room Time",
                  "Room Dimension (in)"]
ORDER_HEADER = ["Order", "Time Entered", "Time Exited"]
CORNER_HEADER = ["Frame #", "Timestamp", "Corner Vectors"]


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
