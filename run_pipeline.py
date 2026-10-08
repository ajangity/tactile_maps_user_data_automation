"""Run the whole pipeline on one video. This is the file you run:

    python run_pipeline.py "S-19_Elevator.mp4" distractor_floorplan_E.png
    python run_pipeline.py video.mp4 map.png --no-display     (no window; batch)

Everything a run creates is saved in its own folder:
    data/<YYYY-MM-DD_HH-MM-SS>__<map name>__<video name>/

What happens, in order (each step lives in its own file):
  paper_locator.py   read the map PNG -> map.json (its lines as a graph,
                     its rooms and symbols) and rooms.png
  auto_crop.py       every frame: find / follow the paper's 4 corners
                     (edge_tracing, paper_locator, corner_fitting, crop_checker)
  finger_tracking.py every frame: fingertips -> map pixels, Left/Right
  timing.py          every frame: step the paper / room / symbol timers
  dashboard.py       at the end: dashboard.json + dashboard.html

Keys while the video plays:
  c       show / hide the list of commands (on screen)
  Space   stop and fix the crop by hand (manual_crop.py)
  t       show trails + rooms on the video      e   paper-edge debug view
  l / r   (while t is on) left / right trail on/off
  w       edge-tracing debug view               ] / [   faster / slower
  q       stop, save everything, quit
"""

import argparse
import json
import os
import pathlib
import time
import webbrowser

import cv2
import numpy as np

import edge_tracing
import manual_crop
from auto_crop import AutoCrop
from dashboard import build_dashboard, save_dashboard_data
from finger_tracking import FingerTracker
from paper_locator import ReferenceMap, draw_paper_debug
from timing import Timing, load_manual_boxes

DEFAULT_PLAYBACK_SPEED = 1.5   # display only; every frame is always processed


def draw_crop(shown, corners, H_map_to_frame, ref, crop):
    """The crop, made obvious: the PNG's printed lines drawn in cyan where
    the crop says they are (if the crop is right they sit exactly on the
    real printed lines), a green outline, and labeled corners (green =
    corner visible, orange = covered)."""
    h, w = shown.shape[:2]
    lines = cv2.warpPerspective(ref.trace.ink, H_map_to_frame, (w, h), flags=cv2.INTER_NEAREST)
    shown[lines > 0] = (255, 255, 0)
    good = crop.source in ("found", "lines", "edges", "realigned", "PNG", "recovered", "manual")
    color = (0, 220, 0) if good else (0, 165, 255)
    cv2.polylines(shown, [np.int32(corners)], True, color, 3, cv2.LINE_AA)
    for p, name, vis in zip(corners, ("TL", "TR", "BR", "BL"), crop.corner_visible):
        c = (int(p[0]), int(p[1]))
        cv2.circle(shown, c, 8, (0, 220, 0) if vis else (0, 165, 255), -1)
        cv2.putText(shown, name, (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)


def banner(shown, title, subtitle, row=0):
    """Big readable message across the top of the frame."""
    y = 60 + row * 85
    cv2.rectangle(shown, (0, y - 45), (shown.shape[1], y + 35), (0, 0, 0), -1)
    if title:
        cv2.putText(shown, title, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(shown, subtitle, (20, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)


COMMANDS = [   # (key, what it does, the show / trail_hands entry it toggles, if any)
    ("c", "show / hide this list of commands", None),
    ("Space", "pause and drag the crop's corners by hand", None),
    ("t", "trails, room outlines and symbol boxes on the video", "trails"),
    ("l", "left finger's trail on / off (while t is on)", "Left"),
    ("r", "right finger's trail on / off (while t is on)", "Right"),
    ("e", "paper-edge debug view (paper mask, search window, edges)", "edges"),
    ("w", "edge-tracing debug view (inside the crop)", "trace"),
    ("] / [", "play faster / slower (every frame is still processed)", None),
    ("q", "stop, save everything and quit", None),
]


def draw_commands(shown, open_, state):
    """Top-right corner: a "press c" hint, or (once c is pressed) every key
    and what it does, with the on/off state of the ones that toggle."""
    font, scale, pad, line = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 12, 30
    if open_:
        rows = [("Commands", "", None)] + [
            (key, text + ("" if name is None else "   [on]" if state[name] else "   [off]"), name)
            for key, text, name in COMMANDS]
    else:
        rows = [("", "Press c to view commands", None)]
    key_w = max(cv2.getTextSize(k, font, scale, 2)[0][0] for k, _, _ in rows[open_:]) + (20 if open_ else 0)
    text_w = max(cv2.getTextSize(t, font, scale, 1)[0][0] for _, t, _ in rows)
    w, h = key_w + text_w + 2 * pad, len(rows) * line + pad
    x0 = shown.shape[1] - w - 10
    fill = shown.copy()
    cv2.rectangle(fill, (x0, 10), (x0 + w, 10 + h), (0, 0, 0), -1)
    cv2.addWeighted(fill, 0.75, shown, 0.25, 0, dst=shown)
    for i, (key, text, name) in enumerate(rows):
        y = 10 + pad + 15 + i * line
        on = name is not None and state[name]
        cv2.putText(shown, key, (x0 + pad, y), font, scale, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(shown, text, (x0 + pad + key_w, y), font, scale,
                    (40, 255, 40) if on else (255, 255, 255), 1, cv2.LINE_AA)


def draw_regions(shown, H_map_to_frame, timing):
    """Room outlines + symbol boxes on the live frame; filled while a finger
    is inside (so the timers can be checked against what's on screen)."""
    for timer in [*timing.rooms.values(), *timing.symbols, *timing.manual]:
        geo = timer.geometry
        if "polygon" in geo:
            pts = np.float32(geo["polygon"])
        else:
            x0, y0, x1, y1 = geo["bbox"]
            pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        pts = np.int32(np.rint(cv2.perspectiveTransform(pts[None], H_map_to_frame)[0]))
        color = (255, 180, 0) if timer.kind == "room" else (0, 200, 0)
        if timer.occupied:
            fill = shown.copy()
            cv2.fillPoly(fill, [pts], color)
            cv2.addWeighted(fill, 0.35, shown, 0.65, 0, dst=shown)
        cv2.polylines(shown, [pts], True, color, 2 if timer.kind == "room" else 1, cv2.LINE_AA)
        if timer.kind == "room":
            c = pts.mean(axis=0).astype(int)
            cv2.putText(shown, timer.name, (int(c[0]) - 30, int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        else:   # symbols and hand-drawn boxes are labeled too
            x, y = pts.min(axis=0)
            cv2.putText(shown, timer.name, (int(x), int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _safe(name):
    """A file name's stem, with characters Windows won't allow in folder names removed."""
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join("_" if c in '<>:"/\\|?*' else c for c in stem).strip() or "unnamed"


def make_run_dir(map_path, video_path, data_dir=DATA_DIR):
    """data/<YYYY-MM-DD_HH-MM-SS>__<map name>__<video name>/ for this run."""
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(data_dir, f"{stamp}__{_safe(map_path)}__{_safe(video_path)}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def previous_map_json(map_path, data_dir=DATA_DIR, exclude=None):
    """The newest earlier run's map.json for the same floorplan, so room and
    symbol names edited by hand there carry over to this run."""
    if not os.path.isdir(data_dir):
        return None
    tag = f"__{_safe(map_path)}__"
    runs = sorted((d for d in os.listdir(data_dir)
                   if tag in d and os.path.join(data_dir, d) != exclude), reverse=True)
    for d in runs:
        path = os.path.join(data_dir, d, "map.json")
        if os.path.exists(path):
            return path
    return None


def run(video_path, map_path, data_dir=DATA_DIR, display=True, open_dashboard=True):
    """Everything this run creates goes in one new folder (see make_run_dir):
        map.json          the PNG's lines (nodes + edges), rooms and symbols
        rooms.png         picture of the detected rooms and symbols
        frame_lines.json  lines traced from the video frame the map was found in
        finger_paths.png  left (red) / right (blue) trails on the map
        session.json      every frame: crop, crop source + score, fingertips
        dashboard.json    the results (timers, visits, order, paths)
        dashboard.html    the dashboard
    """
    run_dir = make_run_dir(map_path, video_path, data_dir)
    out = lambda name: os.path.join(run_dir, name)
    ref = ReferenceMap(map_path, json_path=out("map.json"),
                       names_from=previous_map_json(map_path, data_dir, exclude=run_dir))
    cv2.imwrite(out("rooms.png"), ref.rooms.draw(ref.image))
    print(f"Saving this run to {run_dir}")
    print(f"Map: {len(ref.rooms.rooms)} rooms, {len(ref.rooms.symbols)} symbols, "
          f"{len(ref.data['lines']['edges'])} wall lines")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop = AutoCrop(ref)
    tracker = FingerTracker(ref.w, ref.h)
    timing = Timing(ref.rooms, load_manual_boxes(map_path, ref.w, ref.h))
    window = manual_crop.VideoWindow("Video Tracker", frame_w, frame_h) if display else None
    show = {"trails": False, "edges": False, "trace": False, "commands": False}
    trail_hands = {"Left": True, "Right": True}   # l / r, under t
    speed = DEFAULT_PLAYBACK_SPEED
    frame_index, timestamp_ms, tips = 0, 0, []
    print("Searching for the map..." + ("" if display else " (batch mode, no window)"))

    while True:
        started = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        timestamp_ms = int(round(1000.0 * frame_index / fps))
        was_locked = crop.locked
        corners = crop.update(frame, frame_index)
        if crop.locked and not was_locked:
            print(f"Found the map at {timestamp_ms / 1000:.1f}s (score {crop.score:.2f})")
        tips = tracker.update(frame, corners, frame_index, timestamp_ms, crop.source, crop.score)
        timing.step(tips, timestamp_ms)

        if not display:
            if frame_index % int(round(10 * fps)) == 0:
                print(f"  {timestamp_ms / 1000:.0f}s / {total / fps:.0f}s   crop: {crop.source} "
                      f"(score {crop.score:.2f})")
            continue

        shown = frame.copy()
        if corners is not None:
            H = cv2.getPerspectiveTransform(ref.corners, np.float32(corners))
            draw_crop(shown, corners, H, ref, crop)
            if show["trails"]:
                draw_regions(shown, H, timing)
                tracker.draw_trails(shown, H, [h for h, on in trail_hands.items() if on])
            if show["edges"]:
                draw_paper_debug(shown, frame, corners)
        else:
            banner(shown, "Looking for the map...  (it isn't face-up on the table yet)",
                   "Press Space to place the corners by hand.")
            if show["trails"]:
                banner(shown, "", "Trails and rooms appear once the map has been found.", row=2)
        if show["trace"]:
            if corners is not None:
                # Only the paper's own pixels: the crop is the trace region.
                edge_tracing.draw_trace(shown, edge_tracing.Trace(
                    crop.gray, wall_min=40, region=corners))
            else:
                banner(shown, "", "Edge tracing runs inside the crop - there's no crop yet.", row=1)
        for tip in tips:
            color = (0, 0, 255) if tip["hand"] == "Left" else (255, 0, 0)
            p = (int(tip["frame_x"]), int(tip["frame_y"]))
            cv2.circle(shown, p, 9, color, -1)
            cv2.putText(shown, tip["hand"][0], (p[0] + 10, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        inside = ", ".join(t.name for t in timing.occupied()) or "-"
        sift = ""
        if crop.source == "PNG":   # backup #1's own numbers, as the old status line showed
            st = crop.png.stats
            sift = (f"  matches: {st['matches']}  error: {st['error']:.1f}px  "
                    f"coverage: {100 * st['coverage']:.1f}%")
        status = (f"crop: {crop.source} (score {crop.score:.2f}){sift}  hands: {len(tips)}  "
                  f"in: {inside}  speed: {speed:.2f}x   [Space] fix crop  [t] trails  [w] tracing  [q] quit")
        y = shown.shape[0] - 15
        if show["trails"]:
            trails = "trails:  " + "   ".join(f"{h} {'on' if on else 'off'}"
                                            for h, on in trail_hands.items()) + "     [l] / [r] toggle"
            (tw, _), _ = cv2.getTextSize(trails, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(shown, (0, y - 62), (tw + 28, y - 28), (0, 0, 0), -1)
            cv2.putText(shown, trails, (14, y - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 40), 2,
                        cv2.LINE_AA)
        cv2.rectangle(shown, (0, y - 28), (shown.shape[1], shown.shape[0]), (0, 0, 0), -1)
        cv2.putText(shown, status, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 40), 2, cv2.LINE_AA)
        draw_commands(shown, show["commands"], {**show, **trail_hands})
        window.show(shown)

        wait = max(1, int(round(1000.0 / (fps * speed) - 1000.0 * (time.perf_counter() - started))))
        key = cv2.waitKey(wait) & 0xFF
        if key == ord(" "):
            new_corners, action = manual_crop.adjust_corners(window, frame, corners, ref.image)
            if action == "accept":
                crop.reset(frame, new_corners)
            elif action == "quit":
                break
        elif key in (ord("]"), ord("="), ord("+")):
            speed = min(4.0, speed + 0.25)
        elif key in (ord("["), ord("-"), ord("_")):
            speed = max(0.25, speed - 0.25)
        elif key == ord("c"):
            show["commands"] = not show["commands"]
        elif key == ord("t"):
            show["trails"] = not show["trails"]
        elif key in (ord("l"), ord("r")) and show["trails"]:
            hand = "Left" if key == ord("l") else "Right"
            trail_hands[hand] = not trail_hands[hand]
        elif key == ord("e"):
            show["edges"] = not show["edges"]
        elif key == ord("w"):
            show["trace"] = not show["trace"]
        elif key == ord("q"):
            break

    # ---- everything below runs automatically when the video ends (or on q)
    cap.release()
    if display:
        cv2.destroyAllWindows()
    tracker.close()
    trails = ref.image.copy()
    tracker.draw_trails(trails)
    cv2.imwrite(out("finger_paths.png"), trails)
    with open(out("session.json"), "w", encoding="utf-8") as f:
        json.dump(tracker.session_log, f, indent=1)
    if crop.lock_info:
        with open(out("frame_lines.json"), "w", encoding="utf-8") as f:
            json.dump(crop.lock_info, f, indent=1)
    data = save_dashboard_data(out("dashboard.json"), tracker, timing, ref,
                               video_path, fps, timestamp_ms)
    html = build_dashboard(data, map_path, out("dashboard.html"))
    print(f"Saved everything to {run_dir}:")
    for name in sorted(os.listdir(run_dir)):
        print(f"  {name}")
    if open_dashboard:
        webbrowser.open(pathlib.Path(html).resolve().as_uri())
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video")
    parser.add_argument("map")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="where run folders are created (default: data/ next to this file)")
    parser.add_argument("--no-display", action="store_true", help="run with no window")
    parser.add_argument("--no-open", action="store_true", help="don't open the dashboard at the end")
    args = parser.parse_args()
    run(args.video, args.map, args.data_dir, display=not args.no_display,
        open_dashboard=not args.no_open)


if __name__ == "__main__":
    main()
