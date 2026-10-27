"""This file lets you place the map's corners by hand, once per video, for
the videos where the pipeline can't find the map by itself. The next run
of that video starts tracking from your corners and keeps them to the end.

    python place_corners.py S-15_Elevator S-1_Door     these videos
    python place_corners.py --flagged                  every video flagged in
                                                       data/flagged_runs.csv

For each video, a frame is shown. Pick a frame where the map is on the
table, as early as possible -- tracking can't start before it:
    Right / Left   one second later / earlier      (also d / a)
    Up / Down      ten seconds later / earlier     (also w / s)
    Enter / Space  place the corners on this frame (the usual corner editor:
                   drag TL, TR, BR, BL -- TL is the map's own top-left)
    n              skip this video        q   save and quit

The corners are saved in manual_corners.json next to this file:
    {"S-15_Elevator.mp4": {"frame": 61, "corners": [[x, y], x4]}}
run_pipeline.py reads it (see starting_corners) and loop_pipeline.py
therefore uses it too. Delete a video's entry to go back to automatic.
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np

import edge_tracing
import manual_crop

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(HERE, "manual_corners.json")


def load():
    if not os.path.exists(SAVED):
        return {}
    with open(SAVED, encoding="utf-8") as f:
        return json.load(f)


def starting_corners(video_path):
    """(frame number, 4 corners) placed by hand for this video, or None."""
    entry = load().get(os.path.basename(video_path))
    return None if entry is None else (entry["frame"], np.float32(entry["corners"]))


def _save(saved):
    with open(SAVED, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=1)


def _read(cap, index):
    """Frame `index` (counting from 1, as run_pipeline does)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
    ok, frame = cap.read()
    return frame if ok else None


def place(video, map_name, saved, window):
    """Returns False to stop everything."""
    cap = cv2.VideoCapture(os.path.join(HERE, "mp4_user_data", video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_image = edge_tracing.load_image(os.path.join(HERE, "png_reference_maps", map_name))
    index = saved.get(video, {}).get("frame", int(fps))
    corners = saved.get(video, {}).get("corners")
    # arrow keys as cv2.waitKeyEx reports them on Windows and on Linux, and letters
    later, earlier, much_later, much_earlier = fps, -fps, 10 * fps, -10 * fps
    steps = {2555904: later, 65363: later, ord("d"): later,
             2424832: earlier, 65361: earlier, ord("a"): earlier,
             2490368: much_later, 65362: much_later, ord("w"): much_later,
             2621440: much_earlier, 65364: much_earlier, ord("s"): much_earlier}
    while True:
        frame = _read(cap, index)
        if frame is None:
            index = max(1, index - int(fps))
            continue
        window_ = window(frame)
        shown = frame.copy()
        text = (f"{video}  {index / fps:.1f}s / {total / fps:.0f}s  -  arrows (a/d, w/s): move   "
                "Enter/Space: place corners   n: skip   q: quit")
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(shown, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 255, 40), 2, cv2.LINE_AA)
        window_.show(shown)
        key = cv2.waitKeyEx(0)
        code = key & 0xFF
        if code == ord("q"):
            return False
        if code == ord("n"):
            return True
        if code in (10, 13, 32):
            new, action = manual_crop.adjust_corners(window_, frame, corners, ref_image)
            if action == "quit":
                return False
            if action == "accept":
                saved[video] = {"frame": index, "corners": np.round(new, 1).tolist()}
                _save(saved)
                print(f"{video}: corners saved at {index / fps:.1f}s (frame {index})")
                return True
            continue
        step = steps.get(key) or steps.get(code)
        if step:
            index = int(min(max(1, index + step), total))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("videos", nargs="*", help="video names, e.g. S-15_Elevator")
    parser.add_argument("--flagged", action="store_true", help="every video in data/flagged_runs.csv")
    args = parser.parse_args()
    from loop_pipeline import PAIRS
    maps = dict(PAIRS)
    names = [v if v.endswith(".mp4") else v + ".mp4" for v in args.videos]
    if args.flagged:
        with open(os.path.join(HERE, "data", "flagged_runs.csv"), newline="", encoding="utf-8") as f:
            names += [r["Video"] + ".mp4" for r in csv.DictReader(f) if r["Flags"]]
    if not names:
        parser.error("name some videos, or use --flagged")
    saved = load()
    state = {}

    def window(frame):
        if "w" not in state:
            state["w"] = manual_crop.VideoWindow("Place the map's corners", frame.shape[1], frame.shape[0])
        return state["w"]

    for video in dict.fromkeys(names):
        if video not in maps:
            print(f"{video}: not in loop_pipeline.PAIRS, skipped")
            continue
        if not place(video, maps[video], saved, window):
            break
    cv2.destroyAllWindows()
    print(f"{len(saved)} video(s) have hand-placed corners in {SAVED}")


if __name__ == "__main__":
    main()
