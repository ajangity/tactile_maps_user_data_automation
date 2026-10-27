"""This file checks every run in data/ and lists the ones whose tracking
probably went wrong, so they can be looked at and reprocessed.

    python flag_runs.py                 newest run of each video
    python flag_runs.py --all           every run folder
    python flag_runs.py --data-dir D    run folders somewhere else

A run is flagged when:
  never found  -- the map was never locked onto (no crop in any frame)
  low map time -- the participant's index finger(s) were on the map for
                  less than MIN_MAP_SHARE (50%) of the video
  cut short    -- the run covers less than MIN_PROCESSED (95%) of the
                  video's frames (stopped with q, or crashed)

The video's length comes from the video itself (mp4_user_data/<name>.mp4)
when it's there, otherwise from the run's own frame count. Time on the map
is summary.csv's "Entire map" Room Time (timing.InteractionTimer).

Writes data/flagged_runs.csv: one row per run checked, flagged or not, so
it also opens as a full overview in Sheets.
"""

import argparse
import csv
import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_MAP_SHARE = 0.50
MIN_PROCESSED = 0.95
HEADER = ["Run", "Video", "Map", "Video Length (s)", "Processed (s)", "Map Time (s)",
          "Map Time (% of video)", "First Found (s)", "Flags"]


def _video_seconds(video):
    path = os.path.join(HERE, "mp4_user_data", video + ".mp4")
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames / fps if frames and fps else None


def _map_time(run_dir):
    """summary.csv's Entire map Room Time, in seconds (None if missing)."""
    path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    col = rows[0].index("Room Time")
    entire = next((r for r in rows[1:] if r and r[0] == "Entire map"), None)
    return float(entire[col]) if entire and entire[col] else 0.0


def _first_found(run_dir, positions):
    """Seconds into the video the map was first found; None if never.
    From tracking_quality.json, or (older runs) the first sample of
    corner_motion.csv with a crop."""
    path = os.path.join(run_dir, "tracking_quality.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["first_found_s"]
    path = os.path.join(run_dir, "corner_motion.csv")
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in list(csv.reader(f))[1:]:
                if len(row) > 2 and row[2] != "":
                    return float(row[1])
    return None


def check(run_dir):
    name = os.path.basename(run_dir)
    parts = name.split("__")
    map_name, video = (parts[1], parts[2]) if len(parts) == 3 else ("", name)
    path = os.path.join(run_dir, "finger_positions.json")
    if not os.path.exists(path):
        return [name, video, map_name, "", "", "", "", "", "incomplete (no data files)"]
    with open(path, encoding="utf-8") as f:
        positions = json.load(f)
    processed = positions[-1]["t_ms"] / 1000.0 if positions else 0.0
    length = _video_seconds(video) or processed
    map_time = _map_time(run_dir) or 0.0
    found = _first_found(run_dir, positions)
    flags = []
    if found is None:
        flags.append("never found")
    if length and map_time < MIN_MAP_SHARE * length:
        flags.append(f"low map time ({map_time / length:.0%})")
    if length and processed < MIN_PROCESSED * length:
        flags.append(f"cut short ({processed:.0f} of {length:.0f} s)")
    share = f"{100 * map_time / length:.1f}" if length else ""
    return [name, video, map_name, round(length, 2), round(processed, 2), round(map_time, 2),
            share, "" if found is None else found, "; ".join(flags)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--all", action="store_true", help="every run, not just the newest per video")
    args = parser.parse_args()
    runs = sorted(d for d in os.listdir(args.data_dir)
                  if os.path.isdir(os.path.join(args.data_dir, d)) and "__" in d)
    if not args.all:   # folder names start with the date and time, so the last one is the newest
        newest = {}
        for d in runs:
            newest[d.split("__")[-1]] = d
        runs = sorted(newest.values())
    rows = [check(os.path.join(args.data_dir, d)) for d in runs]
    out = os.path.join(args.data_dir, "flagged_runs.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    flagged = [r for r in rows if r[-1]]
    print(f"Checked {len(rows)} run(s); {len(flagged)} flagged:")
    for r in flagged:
        print(f"  {r[1]:24} {r[-1]}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
