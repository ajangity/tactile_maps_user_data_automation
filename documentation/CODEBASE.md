# Codebase reference

Technical reference for the tactile-map finger-tracking pipeline: what every file does, how data moves between them, the file formats a run produces, the settings that control behavior, and what has been measured so far.

- **Interactive diagram:** [`architecture.html`](architecture.html). Open it in a browser and click any box.
- **Narrative walk-through of the 26 pipeline steps:** [`README.md`](README.md).

---

## 1. What the project does

Given a **video** of someone exploring a printed tactile map and the map's **floorplan PNG**, the pipeline:

1. Finds the printed map in each video frame, even when it's rotated, tilted in perspective, moved, partly covered by hands, or next to a decoy sheet. This is the **crop**: the map's 4 corners in the frame.
2. Tracks the index fingertip of up to two hands with MediaPipe, and maps each fingertip onto the PNG's own pixel coordinates.
3. Works out the map's rooms and symbols automatically from the PNG. It then times every visit to the paper, each room and each symbol; a visit counts if it lasts at least 1 second.
4. Saves everything into a per-run folder, including an interactive HTML dashboard.

---

## 2. Quick start

```bash
# from the project folder
python run_pipeline.py "S-19_Elevator.mp4" distractor_floorplan_E.png              # with a window
python run_pipeline.py "S-19_Elevator.mp4" distractor_floorplan_E.png --no-display  # batch, no window
python dashboard.py "data/<run folder>/dashboard.json"                             # rebuild a dashboard
python edge_tracing.py distractor_floorplan_E.png                                  # inspect line tracing
python label_symbols.py distractor_floorplan_E.png                                 # optional extra boxes
```

| Option | Meaning |
|---|---|
| `--no-display` | No window; runs start to finish unattended. |
| `--no-open` | Don't open the dashboard when finished. |
| `--data-dir PATH` | Where run folders are created. Default: `data/` next to the code. |

**Keys while the video plays**

| Key | Action |
|---|---|
| `c` | Show / hide the on-screen list of every command (a "Press c" hint is always in the top-right corner) |
| Space | Stop the video and drag the crop's corners by hand (`manual_crop.py`) |
| `t` | Trails, room outlines and symbol boxes on the video |
| `l` / `r` | While `t` is on: turn the left / right finger's trail on or off |
| `e` | Paper-edge debug view: paper mask, search window, edges colored by side |
| `w` | Edge-tracing view, inside the crop only |
| `[` / `]` | Slower / faster display. Every frame is still processed. |
| `q` | Stop, save everything, quit |

**Requirements:** Python with `opencv-python` (or `opencv-contrib-python`), `mediapipe` and `numpy`, plus `hand_landmarker.task` in the project folder.

> **Windows:** if Controlled Folder Access (ransomware protection) is on, it blocks `python.exe` and `git.exe` from writing inside `Documents`. Python then reports a misleading `FileNotFoundError: No such file or directory` when creating the run folder. To fix it, allow both under **Windows Security → Ransomware protection → Allow an app → Recently blocked apps**.

---

## 3. Folder layout

```
tactile_maps_user_data_automation/
├── run_pipeline.py        entry point
├── auto_crop.py           crop running order (steps 2–21 each frame)
├── paper_locator.py       auto-crop part 1 (steps 1–8)
├── corner_fitting.py      auto-crop part 2 (steps 9–17)
├── crop_checker.py        checks + backups (steps 14, 18, 19, 21)
├── edge_tracing.py        printed-line tracing (PNG + frames)
├── room_tracking.py       rooms + symbols from the PNG
├── finger_tracking.py     fingertips (steps 22–24)
├── timing.py              enter/exit timers (step 25)
├── dashboard.py           results JSON + HTML dashboard (step 26)
├── manual_crop.py         Space-key corner dragging + video window
├── label_symbols.py       optional: hand-drawn extra boxes
├── hand_landmarker.task   MediaPipe model
├── distractor_floorplan_E.png, S-19_Elevator.mp4   inputs
├── data/                  one folder per run (created automatically)
├── documentation/         README.md, CODEBASE.md, architecture.html
└── old_code/              the pre-refactor version, kept for reference
```

---

## 4. Architecture

The pipeline runs in 8 layers, top to bottom. Layers 4–6 repeat for every video frame.

```
1 Inputs          map PNG · video · hand_landmarker.task · (<map>.symbols.json)
                         │
2 Start           run_pipeline.py ── creates data/<date_time>__<map>__<video>/
                         │
3 Know the map    paper_locator.ReferenceMap ◄── edge_tracing (PNG lines → graph)
  (once)                 │              ◄── room_tracking (rooms, symbols)
                         └─► map.json, rooms.png
   ┌──────────────── every frame ────────────────────────────────────────┐
4  │ auto_crop.py ─ calls ─► edge_tracing (ink inside crop + 35%)        │
   │      │                  paper_locator (steps 2–8, scoring rubric)   │
   │      │                  corner_fitting (steps 9–17, ICP, shift)     │
   │      │                  crop_checker (14, 18, 19, 21)               │
   │      │ ◄── manual_crop.py (Space)          └─► frame_lines.json     │
   │      ▼ 4 corners                                                    │
5  │ finger_tracking.py ◄── hand_landmarker.task ─► session log, trails  │
   │      ▼ fingertips (PNG px, on_paper)                                │
6  │ timing.py ◄── RoomMap (room pixels, symbol boxes), extra boxes      │
   └─────────────────────────────────────────────────────────────────────┘
7 Results         dashboard.py ─► dashboard.json, dashboard.html
8 Run folder      all 7 files in data/<date_time>__<map>__<video>/
```

**Dependency rules**

- `edge_tracing.py` imports nothing from the crop code. Callers can pass it a region to stay inside: the crop once the map is locked, or nothing before that.
- `paper_locator.py` (File 1) has no dependency on `corner_fitting.py` (File 2); File 2 imports File 1. `auto_crop.py` imports both, so the running order lives in one place.
- `run_pipeline.py` is the only file that touches the window, the keyboard, or the run folder.

---

## 5. Module reference

### `run_pipeline.py`: entry point

Creates the run folder and builds the `ReferenceMap`, which writes `map.json`; it also saves `rooms.png`. It then loops over frames: `AutoCrop.update` → `FingerTracker.update` → `Timing.step`. In display mode it draws the crop, the overlays and a status bar. At the end it writes all outputs and opens the dashboard.

| Function | Purpose |
|---|---|
| `run(video, map, data_dir, display, open_dashboard)` | Whole run; returns the run folder |
| `make_run_dir`, `previous_map_json` | Run folder naming; carry hand-edited names over from the last run of the same map |
| `draw_crop`, `draw_regions`, `banner` | Live-window drawing |

### `paper_locator.py`: auto-crop part 1, steps 1–8

| Step | Function | What it does |
|---|---|---|
| 1 | `ReferenceMap(path, json_path, names_from)` | Loads the PNG, compositing transparency onto white. Traces it, detects rooms and symbols, and writes `map.json`. Precomputes scoring tables: per-class distance transforms, plus nearest walls-and-symbols pixel lookups for ICP. |
| 2 | `paper_mask` / `ink_border_mask` | Otsu brightness split minus colorful (skin) pixels; fallback: everything darker than gray 180 |
| 3 | `paper_blob_quads`, `traced_map_quads` | Starting outlines from paper blobs, and from clusters of traced walls (needs no paper edge) |
| 4 | `orientations`, `alignment_score` | 4 corner labelings per outline; the scoring rubric (§7.1) |
| 6 | `search_window` | Last frame's crop grown 35% |
| 7 | `candidate_blobs` | Up to 3 separate paper blobs in the window, nearest first |
| 8 | `classify_segments` | Each straight paper edge assigned to top/right/bottom/left (±20°, nearest line) |
| — | `draw_paper_debug` | The `e` overlay |

### `corner_fitting.py`: auto-crop part 2, steps 9–17

| Step | Function | What it does |
|---|---|---|
| 9 | `weighted_points` | Points every 4 px along a segment; loose angle match → fewer points (min 15%) |
| 10 | `fit_side_line` | Huber line fit, drop outliers beyond median + 4×MAD, refit |
| 12 | `fit_quad` | Canny + Hough on a mask → sides → intersections → 4 corners |
| 13 | `refine_with_traced_lines` | ICP: PNG walls+symbols onto traced frame ink, RANSAC homography, radius 14→4 px (§7.2) |
| 13 | `shift_candidates` | FFT cross-correlation over ±35% shifts → restart points when the crop slipped (§7.4) |
| 15 | `snap_corners` | `cornerSubPix`; ignored if it moves a corner > 8 px |
| 16 | `keep_if_stationary` | ≥ 2 corners within 2 px of last frame → keep last frame's crop |
| 17 | `pick_best`, `smooth` | Highest score ≥ threshold; 65% new / 35% old |

Step 11 was removed: it reused a side's line from an earlier frame when that side looked hidden.

### `crop_checker.py`: steps 14, 18, 19, 21

| Function | What it does |
|---|---|
| `check_quad` | Corners within 25° of square; aspect within 35% of the PNG's; area ≥ 3% of the frame; corners ≤ 20 px off-frame; when trusted, no corner jumps > 8% of the frame diagonal |
| `outline_agreement` | Share of 160 edge samples where the image is ≥ 25 gray levels brighter 8 px inside the crop than 8 px outside. A crop needs ≥ 0.65 (§7.3). |
| `PngBackup.locate` | SIFT/ORB vs the PNG near the last crop. MAGSAC homography with ≥ 14 inliers, ≤ 3 px error and ≥ 3.5% coverage. When trusted, the area must be within 0.65–1.55× last frame's. Then score ≥ 0.35 plus the outline check. |
| `FlowBackup.remember` / `locate` | Pyramidal LK flow on ≤ 700 paper points, forward-backward error < 1.5 px, homography error < 2 px. Points are only picked when the backup runs. |
| `correct_outlier_corner` | If 3 corners moved ≤ 6 px and 1 moved ≥ 18 px (and ≥ 3× the others), the odd one is predicted from the other three |
| `corner_visible` | Local Otsu split around a corner; the bright side must be paper-colored |

### `auto_crop.py`: running order, steps 2–21 each frame

`AutoCrop.update(frame, i)`:

1. **Not found yet:** trace the whole frame; every 10 frames run `locate`. Locking needs a score ≥ 0.60 and the outline check.
2. **Locked:** trace inside the last crop grown 35%. **Candidate A** is the last crop refined by ICP.
3. If A scores below 0.60, add **candidates B**: each blob's `fit_quad`, or the ink-border fallback if no blob works, then snap and ICP.
4. Every candidate goes through `check_quad` + `judge` (outline check, then the rubric). The best one scoring ≥ 0.40 wins; then `keep_if_stationary`, then `smooth`.
5. **Nothing passed:** every 3 frames, `_realign` (shift search + ICP); then the PNG backup; then the flow backup; otherwise hold the crop.
6. After 20 unconfirmed frames the crop is **lost**: a whole-frame `locate` runs every 3 frames, and results within 85% of the best count as a tie (the one nearest the last crop wins).
7. `correct_outlier_corner`, then `corner_visible`.

**Source labels** (logged per frame): `searching`, `found`, `lines` (candidate A), `edges` (candidate B), `realigned`, `PNG`, `flow`, `held`, `lost`, `recovered`, `manual`.

### `edge_tracing.py`: line tracing

| Function | What it does |
|---|---|
| `ink_mask(gray, region=None)` | Morphological black-hat (9 px), threshold 30. With a region, only that polygon's pixels are processed. |
| `ink_classes` | Per-pixel class by mark size: dot ≤ 12 px, wall ≥ 80 px, symbol in between |
| `trace_segments`, `merge_segments` | Hough on walls; merge segments within 3°, 4 px offset and 8 px gap |
| `build_line_graph` | Nodes at line ends, corners and T-junctions (8 px tolerance); edges with length and angle |
| `Trace(gray, region=…)` | All of the above; `symbol_marks()`, `to_json()` |
| `load_image`, `classify_shape`, `draw_trace` | Transparent PNGs onto white; shape labels; the `w` overlay |
| CLI `main` | `<name>.lines.json` + `.lines.png` + a per-symbol listing |

On video frames the wall threshold is 40 px, since the map is smaller in the frame than in the PNG.

### `room_tracking.py`: rooms and symbols

| Function | What it does |
|---|---|
| `detect_rooms(walls)` | Seal doorways by thickening the walls, take enclosed white areas as seeds, then grow the seeds back to the walls (nearest seed), giving a label image + polygons |
| `choose_gap(walls)` | Tries 16 sealing widths (1.5–9% of the map); picks the smallest with the most common room count |
| `detect_symbols(trace, labels)` | Symbol marks + 6 px padding, tagged with their room |
| `RoomMap` | Rasterizes the saved polygons; `room_at(x, y)` is one array lookup; `draw()` → `rooms.png` |

### `finger_tracking.py`: steps 22–24

| Function | What it does |
|---|---|
| `hand_crop` | Paper bounding box + 22% / 30% margins, upscaled ≤ 2.5× (≈ 1280 px cap) |
| `FingerTracker.update` | MediaPipe (2 hands, VIDEO mode) → landmark 8 → frame px → PNG px through this frame's homography. Sets `on_paper`; ignores tips > 50% off the map; appends the session-log record. |
| `match_hands`, `label_hands`, `HandTrack` | Joint two-hand wrist matching (max jump 200 px), Left/Right from a fading, confidence-weighted memory of MediaPipe's labels |
| `draw_trails` | Trails on the PNG or warped onto the video; they break at gaps and jumps > 140 px |

### `timing.py`: step 25

| Item | Detail |
|---|---|
| Timers | `On paper`; one per room (`RoomMap.room_at`); one per symbol (bbox); one per extra box |
| Visit rules | Starts when any fingertip is inside. Ends after 0.4 s with none inside, at the last moment one was inside. Two hands count as one visit. ≥ 1 s = visit, shorter = brush. |
| Status | `visited` (≥ 1 visit), `brushed` (touched, no visit), `missed` (never touched) |
| `sequence()` | Every visit in time order: the path through the rooms |
| `load_manual_boxes` | Reads `<map>.symbols.json`; files without `"space": "png"` are rotated back from the old 180° convention |

### `dashboard.py`: step 26

| Function | What it does |
|---|---|
| `save_dashboard_data` | Closes open visits, writes `dashboard.json`, and prints a per-box summary |
| `build_dashboard` | Self-contained `dashboard.html` (map + data embedded, works offline, light and dark mode) |
| CLI `main` | Rebuilds the HTML from a saved `dashboard.json`; finds the PNG via `map_path` |

The dashboard has summary tiles; the map with rooms and symbols colored by status plus both hands' paths, with a time slider and Play; a table; an enter/exit timeline (with rows for crop reliability and time on paper); and the visit order.

### `manual_crop.py`

`adjust_corners` freezes the frame, overlays the PNG on the crop, labels the corners TL/TR/BR/BL, and makes the nearest corner follow the mouse. Space or Enter accepts (`AutoCrop.reset`), Esc cancels, `q` quits. `VideoWindow` letterboxes the frame to the window size and maps clicks back to frame pixels.

### `label_symbols.py` (optional)

Press `s`/`r` to start a symbol or room box, click two corners, type a name, then Enter. `u` undoes, `d` saves `<map>.symbols.json` in PNG orientation.

---

## 6. Data formats

All map coordinates are the **PNG's own pixels, in the PNG's own orientation** (`(0, 0)` = top-left). Frame coordinates are video pixels.

### `map.json`

```jsonc
{
  "width": 816, "height": 1056, "image": "distractor_floorplan_E.png",
  "ink": {"dot_px": 32761, "symbol_px": 3424, "wall_px": 21170},
  "lines": {
    "nodes": [{"id": 0, "x": 56.5, "y": 34.5, "degree": 2}],
    "edges": [{"id": 0, "from": 0, "to": 1, "length": 133.5, "angle": 90.1}]
  },
  "border": [[54.0, 34.0], [759.0, 34.0], [759.0, 1002.0], [54.0, 1002.0]],
  "room_gap_px": 33,
  "rooms":   [{"id": 1, "name": "Room 1", "area_px": 27043, "centroid": [159.7, 102.0],
               "bbox": [57, 37, 265, 168], "polygon": [[x, y], ...]}],
  "symbols": [{"id": 1, "name": "Symbol 1", "shape": "text", "room": 1,
               "bbox": [130, 73, 190, 133]}]
}
```

Renaming a room or symbol here carries over to the next run of the same map; names are matched by id.

### `frame_lines.json`

Same graph format, traced inside the crop of the frame where the map was first found. Each node also has `map_x`/`map_y`, so it can be compared with `map.json` directly. Extra fields: `frame_index`, `crop` (4 corners), and `alignment` (the score breakdown, per class: precision, recall, F).

### `session.json` (one record per frame)

```jsonc
{"frame": 631, "t_ms": 21054, "page_source": "lines", "page_score": 0.94,
 "corners": [[x, y], [x, y], [x, y], [x, y]],
 "Left":  {"hand": "Left", "frame_x": 0, "frame_y": 0, "map_x": 0, "map_y": 0,
           "on_paper": true, "mediapipe_label": "Left"},
 "Right": null}
```

### `dashboard.json`

```jsonc
{
  "video": "...", "map": "...", "map_path": "...", "coordinate_space": "png",
  "ref_w": 816, "ref_h": 1056, "fps": 29.97, "duration_ms": 112612,
  "frames": 3375, "frames_with_hand": 0, "min_visit_ms": 1000,
  "paper": { /* summary, see below */ },
  "boxes": {"Room 1": {
      "type": "room", "polygon": [[x, y]], "room": null,
      "status": "visited", "touched": true, "missed": false,
      "visits": 2, "brushes": 1, "total_ms": 7274, "longest_ms": 5400,
      "first_enter_ms": 25300,
      "events": [{"enter_ms": 25300, "exit_ms": 30700, "duration_ms": 5400, "hands": ["Left", "Right"]}]}},
  "sequence": [{"name": "Room 6", "type": "room", "enter_ms": 23900, "exit_ms": 25300,
                "duration_ms": 1400, "hands": ["Left", "Right"]}],
  "page_source": ["searching", "...", "lines"],
  "path": {"Left": [{"x": 0, "y": 0, "t_ms": 0}, null], "Right": []}
}
```

Symbols use `"bbox"` instead of `"polygon"`, and `"room"` is the name of the room they're in. In `path`, `null` marks a gap: the hand wasn't seen, it was too far off the map, or it jumped further than a hand can move in one frame.

---

## 7. Key algorithms

### 7.1 Scoring rubric (`paper_locator.alignment_score`)

The frame's traced ink is warped into PNG space through the candidate crop and classified (dots, symbols, walls) using PNG-scale sizes. Each class is then compared only with the PNG's ink of the same class:

- **Precision:** the share of traced pixels within 5 px of a printed pixel of the same class.
- **Recall:** the share of printed pixels within 5 px of a traced pixel.
- **F:** F-beta with β = 0.5, so precision counts twice as much. Hands hide ink, which costs recall, not precision.
- **Score** = 0.4·F(walls) + 0.4·F(symbols) + 0.2·F(dots), renormalized if a class is absent from the map.

Measured on S-19:

| Candidate | Score |
|---|---|
| Right orientation | 0.98–0.99 |
| Same map rotated 180° | ~0.40 |
| Rotated 90° or 270° | 0.10–0.14 |
| Decoy sheet | 0.03–0.04 |
| Blank page | < 0.07 |

Symbols decide the orientation, because walls are near-symmetric.

### 7.2 Traced-line refinement (`corner_fitting.refine_with_traced_lines`)

ICP over a homography:

1. Take traced frame pixels inside the crop + 10%, minus marks < 9 px (dots), up to 2,500 of them.
2. Push them into PNG space.
3. Pair each with the nearest printed **wall or symbol** pixel within the round's radius (14, 10, 7, 5, 4 px).
4. Re-fit with RANSAC (3 px).

Dots are excluded on both sides, because their regular spacing let a shifted crop still fit.

### 7.3 Outline check (`crop_checker.outline_agreement`)

On this map's evenly sized rooms, a crop shifted (or shrunk) by one room still lines most walls up with other walls, and can score 0.5–0.6. The outline check catches this from the image alone: 40 samples per edge, each comparing 8 px inside with 8 px outside, where the inside must be ≥ 25 gray levels brighter.

| Crop | Outline agreement |
|---|---|
| Correct (hands on the page) | 0.81–0.82 |
| Stuck one room off | 0.47–0.57 |

The threshold is 0.65.

### 7.4 Shift search (`corner_fitting.shift_candidates`)

Warp the traced walls and symbols into PNG space through the current crop, then cross-correlate them with the PNG's walls and symbols over shifts up to ±35%, using one FFT. Each of the 5 strongest peaks is a restart point for ICP, and the best result that passes both checks wins. This runs every 3 frames while nothing passes; it is what recovers a crop that slipped.

### 7.5 Room detection (`room_tracking.detect_rooms`)

1. Dilate the walls by the sealing width, so doorways close.
2. Enclosed white components that don't touch the image border and are ≥ 0.2% of the map become seeds.
3. A nearest-seed distance transform assigns every non-wall pixel inside the map outline to a room.
4. Each room is saved as an outline polygon.

On map E this finds 11 rooms (10 rooms + the corridor). The sealing width it picks automatically is 33 px.

---

## 8. Settings worth knowing

| Setting | File | Value | Effect |
|---|---|---|---|
| `LOCK_SCORE` | auto_crop | 0.60 | Score needed for first lock / recovery |
| `TRACK_SCORE` | auto_crop | 0.40 | Per-frame acceptance |
| `STRONG_SCORE` | auto_crop | 0.60 | Above this, skip paper-edge candidates (~75 ms saved) |
| `SEARCH_EVERY` | auto_crop | 10 | Frames between searches before first lock |
| `MAX_UNCONFIRMED` | auto_crop | 20 | Frames before "lost" |
| `SEARCH_MARGIN` | paper_locator | 0.35 | Crop growth for the search window and tracing region (20% tested: 3× worse) |
| `SCORE_TOLERANCE` | paper_locator | 5 px | Rubric match distance |
| `PAPER_SAT_MAX` | paper_locator | 55 | Skin rejection ceiling |
| `OUTLINE_MIN` | crop_checker | 0.65 | Outline check threshold |
| `MAX_JUMP` | crop_checker | 8% diag | Max corner move per trusted frame |
| `ICP_TOLERANCES` | corner_fitting | 14…4 px | ICP pairing radii |
| `SMOOTHING` | corner_fitting | 0.65 | Share of the new crop |
| `BLACKHAT_KERNEL` / `INK_THRESHOLD` | edge_tracing | 9 px / 30 | What counts as ink |
| `MIN_VISIT_MS` / `EXIT_GRACE_MS` | timing | 1000 / 400 ms | Visit rules |
| `PATH_BREAK_JUMP` | finger_tracking | 140 px | Trail breaks |

---

## 9. Measured results (S-19_Elevator.mp4, map E)

| Measure | Result |
|---|---|
| Map found | 21.0 s (score 0.92). The sheet is face-down until then, and the blank side is correctly rejected. |
| Crop confirmed while the map is on the table (21–106 s) | All but 15 frames (0.5 s); 99.4%. Visually checked frame-by-frame through two-handed slides. |
| After 106 s | Sheets removed from the table, so the crop is correctly "lost" |
| Paper motion between frames | Median 0%, 99th percentile 1.1%, 99.9th percentile 2.7% of the paper's diagonal |
| Crop cost | ~80–100 ms per frame typically (realignment ~0.5 s when it runs) |
| Whole video, batch | ~10 minutes for 1:53 of video, including MediaPipe |
| Rooms | 11 found automatically; 10 visited, 1 only brushed (one earlier run) |

---

## 10. Known limitations

- **Skin can pass as paper.** For this participant, pale skin passes the paper filter, so blob outlines grow up the arm. The crop then relies on printed lines (ICP, outline check) when hands cover the page.
- **Tuned on one video and one map.** A map without symbols, very different line weights, or a different camera setup may need the score thresholds adjusted.
- **Speed.** Processing runs at ~6 fps; the first 21 s of S-19 stutter in display mode because of whole-frame searches every 10 frames.
- **Display mode** was checked with a screenshot harness, not driven on a real screen.
- **Auto-generated names.** Rooms and symbols are named `Room N` / `Symbol N` in reading order; rename them in `map.json`.

---

## 11. History: from the old code to the new modules

The pre-refactor files are kept in `old_code/`:

| Old | New |
|---|---|
| `finger_tracking_manual_objects.py`: `PaperEdgeDetector`, `PagePose`, UI | `paper_locator`, `corner_fitting`, `crop_checker`, `auto_crop`, `manual_crop`, `run_pipeline` |
| `data_collection.py`: fingers, box timers, dashboard data | `finger_tracking`, `timing`, `dashboard` |
| `trace_map.py`: threshold ink, Hough walls, symbol blobs | `edge_tracing` (black-hat ink, classes, merged segments, line graph, region restriction) |

**Behavior changes in the refactor**

- **Orientation:** chosen by the line-scoring rubric instead of SIFT text-weighted matching, which a blank page could fool. There is no fixed 180° rotation.
- **Step 5 (startup manual alignment):** became the Space-key tool.
- **Step 11 (stale side lines):** removed.
- **New:** traced-line refinement, outline check, shift search, and automatic rooms and symbols.
- **Restored from the old code:** page-restricted edge tracing and its `w` overlay labels; the ink-border fallback; the SIFT backup's area-ratio check; SIFT diagnostics in the status bar; the 85% recovery tie-break; transparent-PNG loading; labels on every box; the per-box console summary; the per-symbol CLI listing.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError ... map.json` / `index.lock` at startup | Windows Controlled Folder Access is blocking `python.exe` or `git.exe` (§2) |
| "Looking for the map…" for a long time | The map isn't face-up, or is mostly covered. Press Space to place the corners by hand. |
| Crop outline on the wrong sheet or in the wrong place | Press Space, drag the corners, then Space again. Check that the cyan lines sit on the printed lines. |
| Rooms look wrong | Open `rooms.png` in the run folder. Doorway sealing is automatic; `detect_rooms(walls, gap=…)` can override it. |
| Only one hand detected | Lighting or motion blur; the crop is upscaled up to 2.5× already |
