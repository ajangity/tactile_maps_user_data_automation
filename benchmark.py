"""This file is the crop tracker's test set: 10 videos picked at random, a
few random frames from each with the paper's true corners, and a runner
that scores the tracker against them. Any change to auto-crop should be
checked here, so a fix for one participant can't quietly break another.

    python benchmark.py make             pick the videos and frames (once; run
                                         again to re-extract missing frames)
    python benchmark.py run --tag NAME   run the tracker on all 10 videos
    python benchmark.py run --tag NAME --quick
                                         only the QUICK_WARMUP seconds before
                                         each labeled frame (minutes, not an
                                         hour; corner error only)
    python benchmark.py label --from NAME
                                         check / fix the true corners by hand,
                                         starting from run NAME's corners
    python benchmark.py compare A B      two runs side by side
    python benchmark.py score NAME       re-grade run NAME against the current
                                         labels (after labels were added)

Everything lives in benchmark/:
    benchmark.json          the videos, their maps and the picked frames
    frames/                 each picked frame as a PNG (to look at) -- not in
                            git (large); `make` re-extracts them from the videos
    labels.json             true corners per picked frame: "checked" (exact,
                            from a crop verified against the printed lines),
                            "rough" (placed by eye where no crop lined up --
                            the map's there, corners within ~25 px), or
                            "absent" (the map isn't in view / face-up)
    runs/<tag>/report.json  one run's numbers (not in git); overlays/ has every picked
                            frame with the tracker's crop (green) and the
                            true crop (magenta)

Two kinds of numbers come out of a run:
  * corner error on the labeled frames -- how far the tracker's corners
    are from the true ones (needs labels.json);
  * whole-video health, which needs no labels -- the share of frames where
    the crop was confirmed by its alignment score, followed by optical
    flow only, held, or lost, and the longest stretch without a
    confirmed crop.
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benchmark")
SEED = 20261024
VIDEOS = 10
FRAMES_PER_VIDEO = 8
CONFIRMED = {"found", "lines", "edges", "realigned", "PNG", "manual"}   # = data_export.CONFIRMED_SOURCES
QUICK_WARMUP = 6.0    # seconds tracked before each labeled frame in --quick runs
CLOSE_PX = 10.0       # mean corner error this small counts as right
OFF_PX = 25.0         # ...and this big counts as wrong
ROUGH_PX = 40.0       # a "rough" label counts as found within this


def _path(*parts):
    return os.path.join(BENCH, *parts)


def _load(name, default=None):
    try:
        with open(_path(name), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _save(name, data):
    os.makedirs(os.path.dirname(_path(name)), exist_ok=True)
    with open(_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


# ---------------------------------------------------------------- make

def make():
    """Pick VIDEOS random video/map pairs (of those on disk) and
    FRAMES_PER_VIDEO random frames from each, and save the frames."""
    sys.path.insert(0, HERE)
    from loop_pipeline import MAP_DIR, PAIRS, VIDEO_DIR
    if _load("benchmark.json"):
        extract_frames()
        return
    rng = random.Random(SEED)
    on_disk = [(v, m) for v, m in PAIRS
               if os.path.exists(os.path.join(HERE, VIDEO_DIR, v))
               and os.path.exists(os.path.join(HERE, MAP_DIR, m))]
    picked = rng.sample(on_disk, VIDEOS)
    entries = []
    os.makedirs(_path("frames"), exist_ok=True)
    for video, map_name in picked:
        cap = cv2.VideoCapture(os.path.join(HERE, VIDEO_DIR, video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Frame numbers as run_pipeline counts them: the first frame is 1.
        frames = sorted(rng.sample(range(1, total + 1), FRAMES_PER_VIDEO))
        index, want = 0, set(frames)
        while want:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index in want:
                cv2.imwrite(_path("frames", f"{video[:-4]}__{index:05d}.png"), frame)
                want.discard(index)
        cap.release()
        entries.append({"video": f"{VIDEO_DIR}/{video}", "map": f"{MAP_DIR}/{map_name}",
                        "frames": [f for f in frames if f not in want]})
        print(f"{video}: frames {entries[-1]['frames']} of {total}")
    _save("benchmark.json", {"seed": SEED, "videos": entries})


def extract_frames():
    """Re-extract any picked frame missing from frames/ (they aren't in git).
    To pick new videos and frames instead, delete benchmark.json first."""
    os.makedirs(_path("frames"), exist_ok=True)
    for e in _load("benchmark.json")["videos"]:
        name = os.path.basename(e["video"])[:-4]
        want = {f for f in e["frames"] if not os.path.exists(_path("frames", f"{name}__{f:05d}.png"))}
        cap, index = cv2.VideoCapture(os.path.join(HERE, e["video"])), 0
        while want:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index in want:
                cv2.imwrite(_path("frames", f"{name}__{index:05d}.png"), frame)
                want.discard(index)
        cap.release()
        print(f"{name}: frames ready" + (f" (missing from the video: {sorted(want)})" if want else ""))


# ---------------------------------------------------------------- run

def _track(job):
    """Run auto-crop over one video (no display, no fingers) -- all of it,
    or frames start..end. Returns the per-frame log and the picked frames'
    corners."""
    entry, overlay_dir, start, end = job
    cv2.setNumThreads(2)
    sys.path.insert(0, HERE)
    if os.environ.get("BENCH_CODE"):   # try out a modified copy of the pipeline
        sys.path.insert(0, os.environ["BENCH_CODE"])
    from auto_crop import AutoCrop
    from paper_locator import ReferenceMap
    tmp = _path("runs", ".maps")
    os.makedirs(tmp, exist_ok=True)
    name = os.path.basename(entry["video"])[:-4]
    ref = ReferenceMap(os.path.join(HERE, entry["map"]),
                       json_path=os.path.join(tmp, f"{name}_{os.getpid()}.map.json"))
    crop = AutoCrop(ref)
    cap = cv2.VideoCapture(os.path.join(HERE, entry["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    want = {f for f in entry["frames"] if start <= f <= (end or f)}
    index = 0
    if start > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        index = start - 1
    log, picked = [], {}
    started = time.perf_counter()
    while not end or index < end:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        corners = crop.update(frame, index)
        log.append((crop.source, round(float(crop.score), 3)))
        if index in want:
            saved = cv2.imread(_path("frames", f"{name}__{index:05d}.png"))
            if saved is not None and np.abs(saved.astype(int) - frame).mean() > 2:
                print(f"WARNING: {name} frame {index} doesn't match the saved frame (seek error?)")
            picked[str(index)] = None if corners is None else np.round(corners, 1).tolist()
            shown = frame.copy()
            if corners is not None:
                cv2.polylines(shown, [np.int32(corners)], True, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(shown, tuple(int(v) for v in corners[0]), 8, (0, 255, 0), -1)
            cv2.putText(shown, f"{crop.source} {crop.score:.2f}", (14, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(overlay_dir, f"{name}__{index:05d}.jpg"), shown,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()
    seconds = time.perf_counter() - started
    return {"video": entry["video"], "fps": fps, "log": log, "picked": picked,
            "ms_per_frame": round(1000 * seconds / max(1, len(log)), 1)}


def _health(log, fps):
    """Whole-video numbers from the per-frame sources (no labels needed),
    counted from the first lock on."""
    first = next((i for i, (s, _) in enumerate(log) if s != "searching"), len(log))
    after = log[first:]
    n = max(1, len(after))
    count = lambda names: sum(1 for s, _ in after if s in names)
    longest = run = 0
    for s, _ in after:
        run = 0 if s in CONFIRMED else run + 1
        longest = max(longest, run)
    scores = [sc for s, sc in after if s in CONFIRMED]
    return {"first_lock_s": round(first / fps, 1) if first < len(log) else None,
            "tracked": round(sum(1 for s, _ in log if s in CONFIRMED) / max(1, len(log)), 4),
            "confirmed": round(count(CONFIRMED) / n, 4),
            "flow": round(count({"flow"}) / n, 4),
            "held": round(count({"held"}) / n, 4),
            "lost": round(count({"searching"}) / n, 4),
            "longest_gap_s": round(longest / fps, 1),
            "median_score": round(float(np.median(scores)), 3) if scores else 0.0}


def _corner_error(got, true):
    """Mean corner distance in px. The corners are in map order (map
    top-left first), so a crop that's the wrong way round is far off."""
    return float(np.mean(np.linalg.norm(np.float32(got) - np.float32(true), axis=1)))


def _accuracy(picked, labels):
    """Per labeled frame: 'close' / 'near' / 'off' by corner error, 'miss'
    (no crop on a visible map), 'phantom' (a crop on an absent map) or
    'absent' (correctly no crop). A rough label only says 'found' (within
    ROUGH_PX) or 'off'."""
    out = {}
    for frame, got in picked.items():
        label = labels.get(frame)
        if not label or label["status"] == "unchecked":
            continue
        if label["status"] == "absent":
            out[frame] = {"result": "absent" if got is None else "phantom"}
        elif got is None:
            out[frame] = {"result": "miss"}
        elif label["status"] == "rough":
            err = _corner_error(got, label["corners"])
            out[frame] = {"result": "found" if err <= ROUGH_PX else "off", "rough_error": round(err, 1)}
        else:
            err = _corner_error(got, label["corners"])
            out[frame] = {"result": "close" if err <= CLOSE_PX else
                          "near" if err <= OFF_PX else "off", "error": round(err, 1)}
    return out


def run(tag, workers, quick=False):
    bench = _load("benchmark.json")
    if not bench:
        raise SystemExit("Run `python benchmark.py make` first.")
    labels = _load("labels.json", {})
    out_dir = _path("runs", tag)
    overlay_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    if quick:   # one short job per labeled frame
        jobs = []
        for e in bench["videos"]:
            fps = cv2.VideoCapture(os.path.join(HERE, e["video"])).get(cv2.CAP_PROP_FPS) or 30.0
            jobs += [(e, overlay_dir, max(1, f - int(QUICK_WARMUP * fps)), f) for f in e["frames"]]
    else:
        jobs = [(e, overlay_dir, 1, None) for e in bench["videos"]]
    started = time.perf_counter()
    with Pool(min(workers, len(jobs))) as pool:
        results = pool.map(_track, jobs)
    report = {"tag": tag, "quick": quick,
              "minutes": round((time.perf_counter() - started) / 60, 1), "videos": {}}
    merged = {}
    for r in results:   # quick runs: several jobs per video
        m = merged.setdefault(r["video"], {"video": r["video"], "fps": r["fps"], "log": [],
                                           "picked": {}, "ms": []})
        m["log"] += r["log"]
        m["picked"].update(r["picked"])
        m["ms"].append(r["ms_per_frame"])
    for r in merged.values():
        name = os.path.basename(r["video"])
        acc = _accuracy(r["picked"], labels.get(name, {}))
        report["videos"][name] = {"health": _health(r["log"], r["fps"]), "accuracy": acc,
                                  "corners": r["picked"],
                                  "ms_per_frame": round(float(np.mean(r["ms"])), 1),
                                  "sources": [] if quick else [s for s, _ in r["log"]]}
        _draw_labels(overlay_dir, name, labels.get(name, {}))
    _save(os.path.join("runs", tag, "report.json"), report)
    print_report(report)


def _draw_labels(overlay_dir, name, labels):
    for frame, label in labels.items():
        if label.get("status") not in ("checked", "fixed"):
            continue
        path = os.path.join(overlay_dir, f"{name[:-4]}__{int(frame):05d}.jpg")
        img = cv2.imread(path)
        if img is None:
            continue
        cv2.polylines(img, [np.int32(label["corners"])], True, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _totals(report):
    results = [a for v in report["videos"].values() for a in v["accuracy"].values()]
    count = lambda r: sum(1 for a in results if a["result"] == r)
    errs = [a["error"] for a in results if "error" in a]
    health = [v["health"] for v in report["videos"].values()]
    mean = lambda k: float(np.mean([h[k] for h in health]))
    return {"labeled": len(results), "close": count("close"), "near": count("near"),
            "found": count("found"),
            "tracked": round(mean("tracked"), 4),
            "off": count("off"), "miss": count("miss"), "phantom": count("phantom"),
            "median_err": round(float(np.median(errs)), 1) if errs else None,
            "confirmed": round(mean("confirmed"), 4), "flow": round(mean("flow"), 4),
            "held": round(mean("held"), 4), "lost": round(mean("lost"), 4),
            "worst_gap_s": max(h["longest_gap_s"] for h in health),
            "ms_per_frame": round(float(np.mean([v["ms_per_frame"]
                                                 for v in report["videos"].values()])), 1)}


def print_report(report):
    print(f"\nRun '{report['tag']}' ({report['minutes']} min)")
    print(f"{'video':26} {'track':>6} {'lock s':>6} {'conf':>6} {'flow':>6} {'held':>6} {'lost':>6} "
          f"{'gap s':>6} {'score':>6} {'ms/f':>6}  labeled frames")
    for name, v in sorted(report["videos"].items()):
        h = v["health"]
        acc = " ".join(a["result"] + (f"({a['error']:.0f})" if "error" in a else "")
                       for _, a in sorted(v["accuracy"].items(), key=lambda kv: int(kv[0])))
        lock = "never" if h["first_lock_s"] is None else f"{h['first_lock_s']:.1f}"
        print(f"{name:26} {h['tracked']:6.1%} {lock:>6} {h['confirmed']:6.1%} {h['flow']:6.1%} "
              f"{h['held']:6.1%} {h['lost']:6.1%} "
              f"{h['longest_gap_s']:6.1f} {h['median_score']:6.2f} {v['ms_per_frame']:6.0f}  {acc}")
    t = _totals(report)
    print(f"\nLabeled frames: {t['labeled']}  close (<= {CLOSE_PX:.0f}px): {t['close']}  "
          f"near: {t['near']}  found (rough labels): {t['found']}  off: {t['off']}  miss: {t['miss']}  "
          f"phantom: {t['phantom']}  median error: {t['median_err']}px")
    print(f"Mean over videos: tracked {t['tracked']:.1%} of all frames;  after the first lock: "
          f"confirmed {t['confirmed']:.1%}  flow {t['flow']:.1%}  "
          f"held {t['held']:.1%}  lost {t['lost']:.1%}  worst gap {t['worst_gap_s']}s  "
          f"{t['ms_per_frame']} ms/frame")


def score(tag):
    report = _load(os.path.join("runs", tag, "report.json"))
    labels = _load("labels.json", {})
    for name, v in report["videos"].items():
        v["accuracy"] = _accuracy(v["corners"], labels.get(name, {}))
    _save(os.path.join("runs", tag, "report.json"), report)
    print_report(report)


def compare(a, b):
    ra, rb = (_load(os.path.join("runs", t, "report.json")) for t in (a, b))
    ta, tb = _totals(ra), _totals(rb)
    print(f"{'':16} {a:>12} {b:>12}")
    for k in ta:
        print(f"{k:16} {str(ta[k]):>12} {str(tb[k]):>12}")
    print(f"\n{'video':26} {'tracked (all frames)':>21} {'longest gap s':>21}")
    for name in sorted(ra["videos"]):
        ha, hb = ra["videos"][name]["health"], rb["videos"].get(name, {}).get("health", {})
        print(f"{name:26} {ha.get('tracked', 0):10.1%} {hb.get('tracked', 0):10.1%} "
              f"{ha['longest_gap_s']:10.1f} {hb.get('longest_gap_s', 0):10.1f}")


# ---------------------------------------------------------------- label

def label(source_tag, only_unchecked):
    """Step through every picked frame with its proposed corners (the
    existing label, or run source_tag's crop) and accept, fix or mark
    absent -- the same corner editor as Space in run_pipeline.py."""
    sys.path.insert(0, HERE)
    import manual_crop
    bench = _load("benchmark.json")
    labels = _load("labels.json", {})
    report = _load(os.path.join("runs", source_tag, "report.json")) if source_tag else None
    window = None
    for entry in bench["videos"]:
        name = os.path.basename(entry["video"])
        ref_image = cv2.imread(os.path.join(HERE, entry["map"]))
        mine = labels.setdefault(name, {})
        for f in entry["frames"]:
            key = str(f)
            current = mine.get(key)
            if only_unchecked and current and current["status"] != "unchecked":
                continue
            frame = cv2.imread(_path("frames", f"{name[:-4]}__{f:05d}.png"))
            if window is None:
                window = manual_crop.VideoWindow("Benchmark labels", frame.shape[1], frame.shape[0])
            corners = (current or {}).get("corners")
            if corners is None and report:
                corners = report["videos"].get(name, {}).get("corners", {}).get(key)
            shown = frame.copy()
            cv2.putText(shown, f"{name} frame {f}  -  [Enter] edit/accept  [a] map absent  [q] quit",
                        (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 255, 40), 2, cv2.LINE_AA)
            window.show(shown)
            key_in = cv2.waitKey(0) & 0xFF
            if key_in == ord("q"):
                break
            if key_in == ord("a"):
                mine[key] = {"status": "absent", "corners": None}
                _save("labels.json", labels)
                continue
            new, action = manual_crop.adjust_corners(window, frame, corners, ref_image)
            if action == "quit":
                break
            if action == "accept":
                changed = corners is None or _corner_error(new, corners) > 0.5
                mine[key] = {"status": "fixed" if changed else "checked",
                             "corners": np.round(np.float32(new), 1).tolist()}
                _save("labels.json", labels)
        else:
            continue
        break
    _save("labels.json", labels)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    p = sub.add_parser("run")
    p.add_argument("--tag", required=True)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--quick", action="store_true",
                   help=f"only the {QUICK_WARMUP:.0f} s before each labeled frame")
    p = sub.add_parser("label")
    p.add_argument("--from", dest="source", default=None)
    p.add_argument("--unchecked", action="store_true", help="only frames not yet checked")
    p = sub.add_parser("score")
    p.add_argument("tag")
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = parser.parse_args()
    if args.cmd == "make":
        make()
    elif args.cmd == "run":
        run(args.tag, args.workers, args.quick)
    elif args.cmd == "label":
        label(args.source, args.unchecked)
    elif args.cmd == "score":
        score(args.tag)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
