# Automated Analysis of Tactile Map Exploration from Video

Project writeup: what the system does, how the pipeline works stage by stage, what it outputs, and how well it performs on the full study dataset.

> For per-function developer reference see [`CODEBASE.md`](CODEBASE.md). For an interactive diagram, open [`architecture.html`](architecture.html) in a browser.

---

## Contents

1. [Problem and motivation](#1-problem-and-motivation)
2. [The study and its data](#2-the-study-and-its-data)
3. [System overview](#3-system-overview)
4. [Pipeline in depth](#4-pipeline-in-depth)
   - 4.1 [Understanding the map (once per run)](#41-stage-1--understanding-the-map-once-per-run)
   - 4.2 [Finding and tracking the paper (every frame)](#42-stage-2--finding-and-tracking-the-paper-every-frame)
   - 4.3 [Finger tracking (every frame)](#43-stage-3--finger-tracking-every-frame)
   - 4.4 [Timing visits (every frame)](#44-stage-4--timing-visits-every-frame)
   - 4.5 [Exporting results (end of run)](#45-stage-5--exporting-results-end-of-run)
5. [Outputs](#5-outputs)
6. [Batch processing, quality control and human-in-the-loop tools](#6-batch-processing-quality-control-and-human-in-the-loop-tools)
7. [Evaluation and results](#7-evaluation-and-results)
8. [Limitations](#8-limitations)
9. [Future work](#9-future-work)
10. [Appendix A: key parameters](#appendix-a-key-parameters)
11. [Appendix B: file map](#appendix-b-file-map)
12. [Appendix C: symbol legend](#appendix-c-symbol-legend)

---

## 1. Problem and motivation

Tactile maps are raised-line floorplans read by touch, mainly by blind and low-vision people. Designing good tactile symbols (which mark means "elevator", "stairs", "emergency exit"?) requires user studies. In those studies a participant explores a printed map with their fingers while a camera records from above. The researcher then wants to know:

- How long did the participant spend on the map at all?
- Which rooms and symbols did they touch, for how long, and how many times?
- In what order did they move through the rooms?
- Did they use one hand or both, and what did each hand do?

Answering these by hand means scrubbing through every video with a stopwatch. That's slow, subjective and hard to repeat. This study has 178 videos, about 4.5 hours of footage.

This project automates that coding. Given a session video and the floorplan PNG that was printed for that session, the pipeline:

1. finds the printed map in every frame, even when it's rotated, tilted, sliding, partly off-camera, covered by hands, or lying next to a decoy sheet;
2. tracks the index fingertip of each hand and projects it onto the map's own pixel coordinates;
3. works out the map's rooms and symbols automatically from the PNG, with no per-map setup;
4. times every visit to the paper, each room and each symbol, overall and per hand;
5. writes spreadsheet-ready data files, and flags any run whose tracking needs a human check.

The design goal throughout was **no per-participant or per-map tuning**. Lighting, skin tone, sleeves, jewelry, camera angle and map design all vary between sessions. The system learns what it needs (the paper's colour, the map's rooms and symbols) from each video and each PNG.

---

## 2. The study and its data

### 2.1 Design

- **20 participants**, `S-1` to `S-20`.
- **9 referents** (the thing the participant looks for): Corridor, Door, Emergency Exit, Stair, Elevator, Bathroom, Inaccessible Area, Dangerous Area, Reception.
- Each participant chose a symbol for each referent. The session for that referent uses a **distractor floorplan** built around that symbol: `distractor_floorplan_<symbol>.png`. Each floorplan contains the target symbol several times among other symbols. Map E, for example, has three "E" (elevator) symbols, matching a "find all 3" task.
- Planned sessions: 20 × 9 = 180. **178 videos** exist. S-13's Stair and Reception sessions were recorded but never uploaded.

### 2.2 Inputs

| Input | Location | Notes |
|---|---|---|
| Session videos | `mp4_user_data/S-<N>_<Referent>.mp4` | Overhead camera. Too large for git; copied from the lab Drive. |
| Floorplan PNGs | `png_reference_maps/distractor_floorplan_<code>.png` | 34 files, 30 used in the study. 816 × 1056 px = US Letter at 96 px/in. |
| Symbol legend | `legends/symbol_legend.json` + `legends/symbols/<code>.png` | 32 legend entries (Appendix C). Each has a printed page with the symbol drawn at regular and small size. |
| Hand model | `hand_landmarker.task` | Google MediaPipe hand landmark model. |
| Video → map pairing | `loop_pipeline.PAIRS` | From the study's Results sheet (column B = the symbol the participant chose). |

### 2.3 Dataset at a glance

| Measure | Value |
|---|---|
| Videos processed | 178 (all on disk), one run each |
| Total footage | 4.52 h, 487,244 frames |
| Video length | median 46 s, range 7.7 s – 14.3 min |
| Distinct floorplans | 30 |
| Rooms per map (auto-detected) | 8–12, median 11 |
| Symbols per map (auto-detected) | 0–22, median 20. The 0s are the corridor-pattern maps (A9, A11, L4), where the target is an area texture, not a symbol. |

Median video length by referent: Emergency 153 s and Door 98 s were the longest tasks. Elevator (33 s), Bathroom (35 s) and Inaccessible (35 s) were the shortest.

---

## 3. System overview

The pipeline is written in Python with OpenCV, NumPy and MediaPipe. It runs in five stages; stages 2–4 repeat for every video frame.

```
 INPUTS     floorplan PNG ─────┐        video ──────┐     hand_landmarker.task
                               ▼                    │              │
 STAGE 1    Understand the map (once)               │              │
            edge_tracing ─► room_tracking ─► symbol_identification │
            (ink → walls/symbols/dots → rooms → named symbols)     │
                               │  map.json                         │
           ┌───────────────────┼────────── every frame ────────────┼────────┐
 STAGE 2   │  Find / track the paper (auto_crop)                   │        │
           │  paper_locator · corner_fitting · crop_checker ·      │        │
           │  paper_model · edge_tracing                           │        │
           │                   │ 4 corners → homography            │        │
 STAGE 3   │  Finger tracking (finger_tracking) ◄──────────────────┘        │
           │                   │ fingertips in map pixels, Left / Right     │
 STAGE 4   │  Timing (timing, symbol_identification.SymbolTiming)           │
           │  paper timer · room timers · symbol timers · per-hand timers   │
           └───────────────────┼────────────────────────────────────────────┘
 STAGE 5    Export (data_export) ─► summary.csv · room_order.csv · corner_motion.csv
                                    finger_positions.json · tracking_quality.json · ...
 AROUND     loop_pipeline (all 178 sessions) · flag_runs (QA) · place_corners (manual fix)
            google_sheets/import_run.gs (one Sheet tab per run) · benchmark (crop test set)
```

Two ideas run through the whole design:

1. **Everything is measured in the PNG's own coordinates.** Once the four paper corners are known in a frame, a homography maps any frame pixel to a map pixel. Rooms, symbols and fingertips then all live in one fixed coordinate system, whatever the camera angle or paper motion. The homography is rebuilt from scratch every frame, so error never accumulates.
2. **The printed lines verify the crop.** A candidate crop is accepted only if the ink seen in the frame, warped into map space, lines up with the ink in the PNG. This one scoring rubric picks the map's orientation, rejects the decoy sheet and the blank back of the page, and grades every frame's crop.

---

## 4. Pipeline in depth

### 4.1 Stage 1 — Understanding the map (once per run)

`paper_locator.ReferenceMap` loads the PNG and builds everything later stages need. It writes the result to `map.json` in the run folder.

#### 4.1.1 Ink tracing (`edge_tracing.py`)

- **Ink mask.** A morphological black-hat (9 px kernel) highlights thin dark strokes against the white page, then a threshold of 30 grey levels marks ink. Black-hat ignores slow lighting gradients, so the same code works on the clean PNG and on camera frames. Transparent PNGs are composited onto white first.
- **Classes by mark size.** Each connected ink mark is classed by its extent. Dots are ≤ 12 px, walls are ≥ 80 px (40 px on video frames, where the map is smaller), and symbols fall in between.
- **Line graph.** A Hough transform finds wall segments. Segments within 3°, 4 px offset and 8 px gap are merged. Nodes go at line ends, corners and T-junctions (8 px tolerance), giving the map's walls as a graph of nodes and edges with length and angle.

#### 4.1.2 Textured areas

Some maps fill a room with a pattern: hatching, dots or dashes marking a corridor or a dangerous area. Through the camera these blur into a grey smear and never resolve as fine lines. `texture_region` finds them automatically: wherever ink covers more than 11% of a 41 px window, over an area of at least 4000 px, pulled 12 px back from the surrounding walls. Scoring and alignment ignore these areas. Before this was added, the map with a hatched corridor (P54) could never lock, because two thirds of its "walls" were hatch lines.

#### 4.1.3 Room detection (`room_tracking.py`)

Rooms are found with no hand labelling:

1. **Seal the doorways.** The walls are thickened by a sealing width so doorway gaps close.
2. **Seed.** Each enclosed white region that doesn't touch the image border and covers at least 0.2% of the map becomes a room seed.
3. **Grow.** A nearest-seed distance transform assigns every non-wall pixel inside the map outline to a room, so rooms reach back out to the real (unthickened) walls.
4. **Outline.** Each room is saved as a polygon with area, centroid, bounding box and physical size in inches.

The sealing width is chosen per map. `choose_gap` tries 16 widths (1.5–9% of the map) and picks the smallest one that gives the most common room count, which is stable to small changes in line weight. On map E it picks 33 px and finds 11 rooms (10 rooms plus the corridor).

#### 4.1.4 Symbol finding and recognition (`symbol_identification.py`)

Symbols come in three placements, and each is found differently:

| Placement | Examples | How it's found |
|---|---|---|
| **Free-standing** (inside a room) | E, B, stair steps, star, X | Symbol-sized ink marks, minus any area texture |
| **On a wall** (marks a door / exit) | circle on a wall, circle with a line | Erase every straight wall run ≥ 45 px (morphological opening, horizontal and vertical). Leftovers 10–80 px across are wall symbols. |
| **Gap** (a doorway) | break in the wall | Along each wall line, a 10–70 px stretch with no ink. A gap and the wall marks within 20 px of it are merged into one door. |

**Recognition.** Each mark is thinned to one-pixel centre lines (Zhang–Suen) and scaled into a 48 × 48 box with its proportions kept. It is then compared by **chamfer distance** against templates cut automatically from the legend pages:

- Free-standing symbols are compared upright. Other rotations carry a 1 px penalty, because orientation can change meaning: legend #2 (square open at the top) vs #12 (square open at the right).
- Wall symbols are rotated so their wall is horizontal, as the legend draws them, and tried both ways round.
- The best match wins if its distance is ≤ 1.5 px. Otherwise the mark is labelled "Unknown symbol".

**Area patterns** (legend #25–29) are recognised by texture statistics rather than shape: the share of the texture that is dots, dashes or lines; whether dots sit in rows or on diagonals; and line thickness. The nearest legend pattern wins if within 0.5. The room is then labelled with it, e.g. "Corridor #28 (Room 6)".

**Entrances.** A symbol on a wall is treated as an entrance. It records `entrance_to` (the rooms it touches, minus the corridor) and an `entrance_zone` (the mark plus 18 px), which is the region its timer watches.

**Naming.** Each symbol is named `<referent> #<legend id>`, e.g. "Elevator #11". Repeats are numbered in reading order. Names edited by hand in a previous run's `map.json` carry over to the next run of the same map.

Nothing about any shape is hard-coded. Adding a symbol means adding its legend page and one JSON entry.

#### 4.1.5 Scoring tables

Finally, `ReferenceMap` precomputes the lookups used every frame: a distance transform per ink class (walls, symbols, dots), and nearest-ink lookups for the alignment step. All of them exclude the textured areas.

---

### 4.2 Stage 2 — Finding and tracking the paper (every frame)

This is the hardest part of the problem and most of the code. The output each frame is the **crop**: the map's four corners in the frame, ordered as the map's own top-left, top-right, bottom-right and bottom-left.

What makes it hard:

- The sheet often starts face-down or off-camera.
- A **decoy sheet** may lie next to the map.
- Hands, arms, sleeves and heads cover large parts of the page, sometimes including several corners.
- The page slides and rotates while covered, and can slide partly out of frame.
- Many maps have evenly sized rooms, so a crop shifted by exactly one room still lines up most walls with *other* walls.

The running order lives in `auto_crop.AutoCrop.update`. The image processing lives in four modules: `paper_locator` (finding the paper), `corner_fitting` (corners and alignment), `crop_checker` (checks and backups) and `paper_model` (what's covering the page).

#### 4.2.1 The scoring rubric (the core idea)

`paper_locator.alignment_score` grades any candidate crop:

1. Warp the frame's traced ink into PNG space through the candidate's homography.
2. Classify the warped ink into walls, symbols and dots, using PNG-scale size thresholds.
3. For each class, compare it only with the PNG's ink of the same class, with a 5 px tolerance:
   - **precision**: the share of traced pixels near printed pixels;
   - **recall**: the share of printed pixels near traced pixels;
   - **F-beta with β = 0.5**, so precision counts twice as much as recall. A hand hides ink, which costs recall but not precision, so occlusion is penalised gently while misalignment is penalised hard.
4. **Score = 0.4·F(walls) + 0.4·F(symbols) + 0.2·F(dots)**, renormalised if a class is missing from the map.

Three refinements keep the score fair across maps and participants:

- **Textures are ignored** (§4.1.2).
- **Symbols and dots can stand in for each other.** Blur merges a dashed line's dashes into symbol-sized marks that flip between the two classes from frame to frame.
- **Only visible ink is expected.** Printed pixels under a hand, sleeve or object (from the paper model, §4.2.3) don't count against recall. A class with less than 15% of its pixels in view is skipped.

Measured on session S-19 (map E):

| Candidate crop | Score |
|---|---|
| Correct orientation | 0.98–0.99 |
| Same map rotated 180° | ~0.40 |
| Rotated 90° / 270° | 0.10–0.14 |
| The decoy sheet | 0.03–0.04 |
| Blank back of the page | < 0.07 |

Walls are nearly symmetric, so the symbols are what decide orientation. Once tracking, a crop needs a score of at least **0.40**; first lock and recovery need **0.60**.

#### 4.2.2 Before the first lock: searching the whole frame

Until the map is found, every 10th frame (`SEARCH_EVERY`) runs a whole-frame search (`AutoCrop.locate`):

1. **Paper mask** (`paper_mask`): an Otsu brightness split, minus colourful (skin-like) pixels. A second mask comes from the paper's colour (`PaperModel.from_frame`), which keeps pale arms out of the paper blobs.
2. **Candidate outlines.** Each paper blob is fit with a quadrilateral (Canny + Hough → four side lines → intersections). A side hidden under an arm is completed from the PNG's aspect ratio at 0.85×, 1× and 1.15× its distance, since perspective shortens the far side. Further candidates come from clusters of traced walls (`traced_map_quads`), which works even when the paper's outline is broken or merged with the other sheet.
3. **Four orientations** of each outline (which corner is the map's top-left).
4. **Sanity checks** (`check_quad`): corners within 25° of square, aspect ratio within 35% of the PNG's, area at least 3% of the frame.
5. **Pre-score** all candidates. The best three are refined with ICP (§4.2.4), using a one-off paper model built from their own pixels so hands on the real map don't sink its score.
6. **Lock** if the best scores ≥ 0.60 *and* passes the outline check (§4.2.5). If part of the map is off-frame, the stricter 0.75 applies. On the benchmark frames, no wrong crop ever scored above 0.72.

#### 4.2.3 The paper model: what's covering the page

`paper_model.PaperModel` learns this video's paper colour instead of relying on fixed thresholds.

- **Learning.** Every 15 confirmed frames scoring ≥ 0.60, it samples the crop's non-ink pixels (brightest half, trimmed to 3×MAD) in **LAB** colour space and blends in their lightness median and a*/b* mean and spread, 25% new. This follows lighting changes over the session.
- **Classifying.** A pixel is paper if its a*/b* lies within 3.5σ of the paper's colour and it isn't much darker than the paper. Skin is separated on the a* (red–green) axis: paper measured 126–128, even very pale skin 138–142. Dark sleeves and shadows fail the lightness test.
- **Occluders.** Non-paper areas left after an opening with an 11 px disc (which removes the thin printed strokes) are occluders: hands, arms, sleeves, bottles, phones. An 8 px margin is added, because a hand's outline and its shadow trace as long dark "walls".
- **Use.** Ink on or along occluders is dropped *before* any candidate is fit or scored, so hand outlines can't pull or grade a crop. The visibility mask also tells the scoring rubric which printed lines to expect.

This step is computed at half resolution (~12 ms).

#### 4.2.4 After the lock: tracking each frame

With the map locked, each frame:

1. **Trace only near the paper.** Edge tracing is limited to last frame's crop grown by 35%. The paper can move between frames, and a 20% margin tested 3× worse.
2. **Remove occluders** using the paper model.
3. **Candidate A — line refinement.** Last frame's crop is refined by **ICP over a homography** (`refine_with_traced_lines`). Up to 2,500 traced pixels are pushed into PNG space and each is paired with the nearest printed wall or symbol pixel. The homography is re-fit with RANSAC (3 px). Pairing radii shrink over five rounds: 14, 10, 7, 5, 4 px. Dots are left out on both sides, because their regular spacing let a shifted crop still fit.
4. **Candidate B — paper edges** (only if A scores below 0.60). Each paper blob's straight edges are sorted into top, right, bottom and left using last frame's crop (±20°). Points are sampled every 4 px and weighted by angle agreement, then each side gets a Huber line fit with outliers beyond median + 4×MAD dropped. The sides are intersected into corners, snapped to sub-pixel corners (`cornerSubPix`, ignored if it moves a corner more than 8 px), then ICP-refined. With only three sides visible, the fourth is completed from last frame's proportions.
5. **Check and judge** every candidate. `check_quad` adds a jump limit (no corner may move more than 8% of the frame diagonal per frame), then come the outline check and the rubric. The best candidate scoring ≥ 0.40 wins.
6. **Stabilise.** If at least two corners are within 2 px of last frame, the old crop is kept (`keep_if_stationary`), which removes jitter. Otherwise the new crop is blended 65% new / 35% old.
7. **Outlier corner fix.** If three corners moved ≤ 6 px and one moved ≥ 18 px (and at least 3× the others), the odd one is predicted from the other three.

#### 4.2.5 The outline check

The rubric alone has one blind spot. On maps with evenly sized rooms, a crop shifted by one room still lines most walls up with other walls, and scores 0.5–0.6. `outline_agreement` catches this from the image alone: it takes 160 samples around the crop's border, and at each compares the brightness 8 px inside with 8 px outside. Inside (paper) must be at least 25 grey levels brighter than outside (desk). A correct crop with hands on the page scores 0.81–0.82; a crop stuck one room off scores 0.47–0.57. The threshold is 0.65.

Sometimes there's no paper/desk contrast: the map lies on top of the decoy sheet, or a sleeve covers the edge. A crop that fails the outline check is then still accepted if either:

- the edge points that *can* be judged pass and the lines score ≥ 0.60, or
- while tracking, the lines score ≥ 0.75 and no corner moved more than 2% of the frame diagonal.

On the benchmark, 463 deliberately wrong crops (shifted by a room, refined, rescored) scored at most 0.72, all at least 129 px off, so neither rule lets a wrong crop through. This change is what let the S-15 videos, where the map overlaps the decoy, confirm a crop at all.

#### 4.2.6 Recovery when nothing passes

If no candidate passes, the tracker tries these in order:

1. **Shift search** (`shift_candidates`, every 3 frames). The traced walls and symbols are warped into PNG space and cross-correlated with the PNG's walls and symbols over shifts up to ±35%, using one FFT. The 5 strongest peaks each seed an ICP refinement, and the best result that passes both checks wins. This recovers a crop that slipped.
2. **PNG backup** (`PngBackup`). SIFT/ORB features are matched between the frame and the PNG near the last crop, and a MAGSAC homography is fit. It needs ≥ 14 inliers, ≤ 3 px error and ≥ 3.5% coverage, an area within 0.65–1.55× last frame's, and a score ≥ 0.35 plus the outline check.
3. **Optical-flow backup** (`FlowBackup`). Pyramidal Lucas–Kanade flow on up to 700 paper feature points, with forward–backward error < 1.5 px and homography error < 2 px.
4. **Hold** the last crop. While holding, a whole-frame search still runs every 10 frames, in case the paper was turned or slid fast under the hands.

**Once found, the crop is never dropped.** Until the end of the video there is always a crop, either confirmed or held. Whether the participant is touching the map is then decided by the fingertips alone.

#### 4.2.7 Map partly out of frame

After the first lock, the page may slide partly off the frame. The crop keeps four corners; those outside the frame are placed by the printed lines (ICP), the paper's remaining edges, or the backups. This is allowed while at least 35% of the map is in view (`check_quad(..., partial=True)`). The outline check skips edge points that are off-frame, and the rubric only expects lines on the part of the map in view. The live window shows an off-screen corner as a hollow red ring on the frame's edge.

#### 4.2.8 Source labels

Every frame records how its crop was obtained. These labels drive the quality metrics:

| Label | Meaning | Confirmed? |
|---|---|---|
| `searching` | Map not found yet | — |
| `found` | Whole-frame search locked on | yes |
| `lines` | Candidate A (ICP on last crop) | yes |
| `edges` | Candidate B (paper edges) | yes |
| `realigned` | Shift search | yes |
| `PNG` | SIFT/ORB backup | yes |
| `manual` | Corners placed by hand | yes |
| `flow` | Optical flow only | no |
| `held` | Last crop kept | no |

---

### 4.3 Stage 3 — Finger tracking (every frame)

`finger_tracking.FingerTracker`:

1. **Crop and enlarge.** Small hands are the usual reason MediaPipe misses one. The frame is cropped around the paper with generous margins (22% horizontally, 30% vertically, so arms coming in are included) and upscaled up to 2.5× (about 1280 px max).
2. **Detect.** MediaPipe Hand Landmarker runs in VIDEO mode with up to 2 hands and low detection thresholds (0.25 / 0.25 / 0.35). The **index fingertip (landmark 8)** of each hand is kept.
3. **Project.** The fingertip goes crop px → frame px → **map px** through this frame's homography (from the four corners). Each tip is flagged `on_paper` if it lands inside the map. Tips more than 50% of the map's size outside it are ignored.
4. **Keep identities stable.** Hands are matched to last frame's hands by wrist position, solving both hands jointly (most matches, then least total movement; a jump over 200 px is a new hand). Left/Right labels come from a **fading, confidence-weighted memory** of MediaPipe's own handedness calls, with 70% kept per frame. One low-confidence frame can't swap the labels, and a wrong label can't stick. Two visible hands always get different labels. On S-19, MediaPipe's raw labels were right for 3588 of 3592 two-hand detections, so the system smooths MediaPipe rather than overriding it.

Trails are drawn per hand (left red, right blue). They break at gaps and at jumps over 140 px, so a briefly wrong crop or a re-appearing hand never draws a false straight line.

---

### 4.4 Stage 4 — Timing visits (every frame)

`timing.Timing` (extended by `symbol_identification.SymbolTiming`) runs one timer per region:

| Timer | Region | Rule |
|---|---|---|
| **Map interaction** (`InteractionTimer`, the main timer) | The whole map | A plain stopwatch: runs on every frame at least one index fingertip is on the map; pauses on the first frame none is. No grace period, no minimum. |
| **Room** | The room's exact pixels from room detection (one array lookup) | Visit rules below |
| **Symbol** | The symbol's bounding box (+6 px) | Visit rules below |
| **Entrance symbol** | The mark + 18 px (`entrance_zone`) | Visit rules below |
| **Video timer** | — | Just the current video time, for the live display. It isn't a map timer and isn't exported. |

**Visit rules** for rooms and symbols:

- A visit starts on the first frame any fingertip is inside.
- It ends once no fingertip has been inside for **0.4 s**. This bridges MediaPipe dropping a hand for a few frames. The exit time is the last moment a finger was actually inside.
- Two hands inside at once count as **one** visit.
- A visit must last **≥ 1 s** to count. Shorter contacts are **brushes** (a finger passing over).
- Each region ends with a status: **visited** (≥ 1 visit), **brushed** (touched, never a full visit), or **missed** (never touched).

**Per-hand timing.** Every timer is also run twice more, once seeing only the left index fingertip and once only the right, with the same rules. So each room has a Left time and a Right time. These can add up to more than the combined time, because when both fingers are in a room the combined timer counts that once.

---

### 4.5 Stage 5 — Exporting results (end of run)

When the video ends (or the user presses `q`), open visits are closed and `data_export.save_data_files` writes the run folder (§5). Every CSV is one plain table, so it opens directly in Excel or Google Sheets. Times are plain seconds (`12.34`), because Sheets misreads `M:SS.ss` as a time of day.

---

## 5. Outputs

Each run creates one folder:

```
data/<YYYY-MM-DD_HH-MM-SS>__<map name>__<video name>/
```

The timestamp comes first so folders sort chronologically. Re-running a video creates a new folder; nothing is overwritten.

| File | Contents |
|---|---|
| `summary.csv` | Per-room and per-symbol visits and time, overall and per hand, plus room size in inches |
| `room_order.csv` | Every room visit in time order, revisits included: the participant's path through the map |
| `corner_motion.csv` | 4 times a second: how far each paper corner moved (frame px), and each index fingertip's position and movement (map px) |
| `corner_motion.json` | The corner motion vectors as one list per corner |
| `finger_positions.json` | Every frame's left and right index fingertip in map pixels (or `null`) |
| `map.json` | The map as the pipeline understood it: wall graph, rooms (polygons, sizes, patterns), symbols (names, legend IDs, match distances, boxes, entrances) |
| `floorplan.png` | Copy of the PNG used |
| `finger_paths.png` | Left (red) and right (blue) finger trails drawn on the floorplan |
| `tracking_quality.json` | Whether the run needs a human check, with reasons and per-source frame counts |

### 5.1 `summary.csv`

One row for the whole map, one per room, then one per symbol that belongs to no room. Columns:

| Group | Columns | Meaning |
|---|---|---|
| Room symbol | name, visits, time, time (Left), time (Right) | The free-standing symbol(s) in the room, or the room's area pattern, labelled `"<symbol> (<room>)"` |
| Entrance symbol | name, visits, time, (Left), (Right) | The wall mark(s) leading into the room |
| Room | visits, time, (Left), (Right), dimensions | The room itself; dimensions in inches (width, height) |

The `Entire map` row's Room Time is the main interaction timer. Cells that don't apply are blank, not 0.

**Example** (S-1, Corridor task, map L4 — a corridor-pattern map with no symbols):

| Room Symbol | Visits | Time (s) | Left | Right | … | Room Time | Left | Right | Dimensions |
|---|---|---|---|---|---|---|---|---|---|
| Entire map | | | | | | 70.07 | 66.30 | 60.06 | (8.5, 11) |
| Corridor #26 (Room 2) | 3 | 9.24 | 2.23 | 8.58 | | 9.24 | 2.23 | 8.58 | (2.15, 1.38) |
| Corridor #26 (Room 10) | 5 | 26.99 | 3.34 | 26.56 | | 26.99 | 3.34 | 26.56 | (2.15, 3.65) |
| Room 6 | | | | | | 30.30 | 30.26 | 0.00 | (2.95, 10.03) |
| Room 11 | | | | | | 0.00 | 0.00 | 0.00 | (2.16, 1.74) |

From this one table a researcher can read that the participant spent 70 s on the map, that Room 10 (a dashed-line corridor) was mostly explored with the right hand, that Room 6 was explored only with the left, and that Room 11 was never visited.

### 5.2 `room_order.csv`

```
Order,                    Time Entered (s), Time Exited (s), Total Time (s)
Corridor #26 (Room 10),   7.81,             10.24,           2.44
Room 6,                   9.38,             10.81,           1.44
Dangerous / Inaccessible area #25 (Room 5), 10.38, 11.61,    1.24
...
```

Rows can overlap in time when the two hands are in different rooms.

### 5.3 `corner_motion.csv`

Sampled 4 times per second: frame number, timestamp, `dx`/`dy` for each of the four corners, each index fingertip's `(x, y)` on the map, and its `dx`/`dy` since the last sample. Blank corner cells mean the map hadn't been found yet. Blank fingertip cells mean that hand wasn't seen. This supports analyses such as "did the participant move the paper?" and "did they sweep with both hands, or anchor one finger as a reference point?"

### 5.4 `tracking_quality.json`

```json
{ "status": "ok", "reasons": [], "first_found_s": 6.6,
  "tracked_after_found": 0.998, "longest_unconfirmed_s": 0.1,
  "frames": {"lines": 2069, "edges": 30, "flow": 4, "realigned": 2,
             "found": 1, "held": 1, "searching": 199} }
```

A run is marked `"check"` if the map was never found, fewer than 70% of frames after the first lock had a confirmed crop, or the crop went unconfirmed for more than 10 s at a stretch.

### 5.5 Google Sheets import

`google_sheets/import_run.gs` is a Google Apps Script that adds a **Tactile Maps → Import run CSVs…** menu to a Sheet. Pick the whole `data/` folder (or individual runs). It creates one tab per run, named `<map> - <video>`, with `summary.csv`, `room_order.csv` and `corner_motion.csv` laid side by side as native Sheets tables. It imports one run per server call to stay inside Apps Script's 6-minute limit. It reads whatever columns each CSV has, so changing the export format needs no change to the script.

---

## 6. Batch processing, quality control and human-in-the-loop tools

| Tool | Purpose |
|---|---|
| `run_pipeline.py <video> <map> [--no-display]` | One session. With a window, it shows the crop (the PNG's lines in cyan over the real print), fingertips, live timers (`t`), trails and room outlines (`f`), and debug views (`e`, `w`). Space pauses to fix the crop by hand. |
| `loop_pipeline.py` | Runs every pair in `PAIRS` (all 178 sessions) one after another, each as its **own process** so one crash doesn't stop the batch. Skips pairs whose files are missing and lists every flagged run at the end. |
| `flag_runs.py` | Reviews every run and writes `data/flagged_runs.csv`. It flags runs where the map was never found, the finger was on the map for under 50% of the video, or under 95% of the video was processed. |
| `place_corners.py` | For runs the tracker can't start on its own: step through the video, pick a frame, drag the four corners. They're saved in `manual_corners.json`, and the next run starts tracking from them. |
| `manual_crop.py` | The corner editor (used by Space and `place_corners`): freezes the frame, overlays the PNG, and the nearest corner follows the mouse. |
| `benchmark.py` | Regression test for the crop tracker (§7.2). Any change to auto-crop is checked here, so a fix for one participant can't quietly break another. |
| `label_symbols.py` | Optional: hand-draw extra symbol or room boxes for a map. |
| `units.py` | Pixel → inch conversion (US Letter, 96 px/in for the 816 × 1056 maps). |
| `dashboard.py` | Legacy: rebuilds the old interactive HTML dashboard from older runs. |

The intended workflow is: **batch everything → flag → fix the flagged runs by hand → re-run only those.** Automation does most of the work, and human time goes only where it's needed.

---

## 7. Evaluation and results

### 7.1 Full study run (178 sessions)

All 178 videos were processed with the current code (runs dated 26–28 Oct 2026).

| Measure | Result |
|---|---|
| Runs completed | **178 / 178** |
| Passed automatic quality check | **160 (89.9%)** |
| Flagged for a human check | 18 (10.1%) |
| — map never found | 4: S-1_Door, S-3_Stair, S-11_Corridor, S-15_Corridor |
| — < 70% of frames confirmed after lock | 7 |
| — unconfirmed > 10 s at a stretch | 9 (2 runs hit both) |
| Time to first find the map | median **1.8 s** (mean 5.7 s, max 84.7 s) |
| Share of frames with a confirmed crop after first lock | median **100%**, mean 94% |
| Longest unconfirmed stretch | median 0.5 s |
| Finger-on-map time as share of video | median 90% |
| Room visits per session | median 17 (max 250) |

**How each frame's crop was obtained**, across all 487,244 frames:

| Source | Share of frames |
|---|---|
| `lines` (ICP refinement of the last crop) | 83.8% |
| `searching` (map not yet found, including the 4 never-found runs) | 7.0% |
| `held` (last crop kept) | 5.0% |
| `edges` (paper-edge fit) | 2.8% |
| `realigned` (shift search) | 0.7% |
| `flow` (optical flow) | 0.6% |
| `PNG` (feature-matching backup), `found` | < 0.1% |

So the primary method, printed-line ICP, carries the vast majority of frames. The fallbacks matter for the hard minority.

**Flagged runs.** Two of the four never-found runs use corridor-pattern maps (L4, A11). On those maps the only print besides the walls is a texture the camera blurs, which leaves the rubric little to match. Four of participant S-18's nine runs are flagged, which suggests something about that session (camera placement or lighting) rather than about the maps. These are hypotheses: the flagged runs haven't all been reviewed by hand yet.

### 7.2 Crop accuracy benchmark

`benchmark.py` holds a fixed test set: **10 randomly chosen videos** (seed 20261024), 8 random frames each, **80 labelled frames** with the paper's true corners. Labels are *checked* (exact, verified against the printed lines), *rough* (placed by eye where no crop lined up; within ~25 px), or *absent* (map not in view). Grading uses the mean corner error: *close* ≤ 10 px, *near* ≤ 25 px, *off* > 25 px. A rough label counts as *found* within 40 px. A *miss* is no crop on a visible map; a *phantom* is a crop on an absent map.

Versions of the tracker over development:

| Run | Correct (close + near + found) | Missed | Wrong (off) | Phantom | Mean share of frames tracked | Worst unconfirmed gap |
|---|---|---|---|---|---|---|
| `baseline` | 47 / 75 | 28 | 0 | 0 / 5 | 56.6% | 22.5 s |
| `v5` (paper model, texture-aware scoring, hidden-side completion) | 66 / 75 | 8 | 1 | 0 / 5 | 77.4% | 12.9 s |
| `v6` (outline-check bypass, held-crop re-search) | **72 / 75** | 2 | 1 | 1 / 5 | 87.0% | 12.8 s |
| `v7_partial` (map partly off-frame; current) | **72 / 75** | 2 | 1 | 1 / 5 | 86.8% | 12.8 s |

Median corner error on exactly labelled frames is about 3 px. Per video (current version), the share of confirmed frames after the first lock ranges from 77% (S-6_Elevator, the hatched-corridor map that originally could never lock) to 100% (S-16_Door).

**Speed.** Crop cost rose from about 157 to about 300 ms per frame between `baseline` and `v6`. The extra time goes to heavily covered videos, where the tracker now keeps trying (backups, re-search) instead of giving up; easy videos run about as fast as before. The `v7_partial` figure (666 ms/frame) was measured with 10 videos running in parallel on one machine and isn't comparable. Including MediaPipe, a typical video processes at roughly 3–6 frames per second: about 10 minutes for a 2-minute session.

The benchmark has one caveat. Many "checked" labels were taken from verified crops of these same runs, so small corner errors slightly favour the runs they came from. The miss, wrong and phantom counts don't depend on that.

### 7.3 Map understanding

Across the 30 floorplans, the automatic analysis found 8–12 rooms per map (328 total) and 527 symbols. **35 symbols (6.6%) matched no legend entry** and were labelled "Unknown symbol".

As a consistency check, count how many symbols on each map were recognised as that map's **target symbol** (the one the floorplan is named after), including area patterns. The study design places it three times on each map.

| Target recognised | Maps |
|---|---|
| 3 of 3 | 25 of 30 |
| 2 of 3 | P29, doors_P29, P95, P99 (all door/exit wall symbols) |
| 1 of 3 | P9 (circle with a dot) |

All three corridor-pattern maps (A9, A11, L4) and the hatched-area map (A10) had their target pattern recognised in all three rooms. Hand checks on maps E (20 symbols, all correct) and Stair (17 of 19 correct) agree with this.

### 7.4 Detailed single-session check (S-19_Elevator, map E)

| Measure | Result |
|---|---|
| Map found | at 21.0 s (score 0.92). The sheet is face-down until then, and the blank side is correctly rejected. |
| Crop confirmed while the map was on the table (21–106 s) | 99.4% of frames (all but 15), visually checked frame by frame through two-handed slides |
| Paper motion between frames | median 0%, 99th percentile 1.1% of the paper's diagonal |
| Rooms | 11 found automatically |

---

## 8. Limitations

- **Coverage isn't correctness for fingers.** The benchmark measures crop accuracy. Fingertip-to-room accuracy hasn't been validated against hand-coded timings; it inherits crop error plus MediaPipe's landmark error. A small validation set of hand-coded room visits would close this gap.
- **Hard sessions still need a human.** 18 of 178 runs are flagged; 4 never locked on automatically. `place_corners.py` exists for exactly these, but they need manual attention.
- **Low-texture maps are harder.** Maps whose only distinguishing print is an area pattern give the alignment score little to work with once the pattern blurs.
- **Symbol recognition has borderline cases.** Similar legend entries (#13 circle vs #18 circle-with-line, #21 triangle vs #23 diamond) are separated by small margins. A free symbol drawn on a wall is only compared against wall marks. Wall symbols and doorways are only found on horizontal or vertical walls.
- **Video → map pairing has open assumptions.** P29 vs doors_P29, A9 vs corridor_A9, and A11 vs A11_staggered are each guessed (see `png_reference_maps/README.md`) and should be confirmed against the Results sheet.
- **Only the index fingertip is tracked.** Exploration with other fingers or the palm isn't counted.
- **Handedness assumes an unmirrored camera.** A mirrored recording would swap every Left/Right label.
- **Symbol first-touch time isn't in the CSVs.** It can be computed from `finger_positions.json` and the symbol boxes in `map.json`.
- **Interrupted runs aren't saved.** Only `q` saves early; Ctrl+C or a crash leaves just `map.json`.
- **Speed.** Processing runs at a few frames per second, so the full study is an overnight batch, not real time.

---

## 9. Future work

- **Validate timings against human coders** on a sample of sessions (inter-rater agreement between the pipeline and a person).
- **Report symbol first-touch / time-to-find** directly in `summary.csv`. This is likely the key measure for a "find all 3" task.
- **Aggregate analysis across sessions**: per-referent and per-symbol comparisons of search time, hit rate (target symbols visited vs missed) and exploration strategy (one- vs two-handed).
- **Speed**: run MediaPipe and crop tracking in parallel, or process at reduced resolution when the crop is stable.
- **Track more fingers** (all fingertips, or the palm) to capture more of the exploration.

---

## Appendix A: key parameters

| Parameter | Value | Effect |
|---|---|---|
| `LOCK_SCORE` / `PARTIAL_LOCK_SCORE` | 0.60 / 0.75 | Score needed for first lock (map fully / partly in frame) |
| `TRACK_SCORE` | 0.40 | Per-frame acceptance once locked |
| `SEARCH_EVERY` | 10 frames | Whole-frame search interval |
| `SEARCH_MARGIN` | 35% | Crop growth for the tracing region |
| `SCORE_TOLERANCE` | 5 px | Rubric match distance |
| Rubric weights | 0.4 walls / 0.4 symbols / 0.2 dots, F-β (β = 0.5) | |
| `OUTLINE_MIN` | 0.65 | Outline check threshold |
| `MAX_JUMP` | 8% of frame diagonal | Max corner move per frame |
| `MIN_IN_FRAME` | 35% | Share of map that must stay in view |
| ICP radii | 14 → 10 → 7 → 5 → 4 px | |
| `SMOOTHING` | 65% new | Crop blending |
| Paper model | 3.5σ chroma, 8 px occluder margin, learns every 15 frames | |
| `MIN_VISIT_MS` / `EXIT_GRACE_MS` | 1000 / 400 ms | Visit rules |
| `QUALITY_MIN_TRACKED` / `QUALITY_MAX_GAP_S` | 70% / 10 s | When a run is flagged |
| `CORNER_SAMPLES_PER_SECOND` | 4 | Corner-motion sampling |
| Symbol `MATCH_MAX` | 1.5 px chamfer | Above this, "Unknown symbol" |

## Appendix B: file map

| File | Role |
|---|---|
| `run_pipeline.py` | Entry point: one video + one map; live window; writes the run folder |
| `paper_locator.py` | Reference map, paper masks, candidate outlines, scoring rubric |
| `corner_fitting.py` | Line fits, quad fitting, ICP refinement, shift search, smoothing |
| `crop_checker.py` | Sanity checks, outline check, PNG and optical-flow backups |
| `paper_model.py` | Per-video paper colour; occluder detection |
| `auto_crop.py` | Per-frame running order of the crop pipeline |
| `edge_tracing.py` | Ink mask, ink classes, wall segments, line graph |
| `room_tracking.py` | Room detection and room lookup |
| `symbol_identification.py` | Symbol finding, legend matching, area patterns, entrance timing |
| `finger_tracking.py` | MediaPipe, projection to map pixels, hand identity, trails |
| `timing.py` | Visit timers (map, rooms, symbols, per hand) |
| `data_export.py` | CSV / JSON outputs, tracking quality |
| `units.py` | Pixels → inches |
| `loop_pipeline.py` | Batch runner for all sessions |
| `flag_runs.py` / `place_corners.py` / `manual_crop.py` | QA and manual correction |
| `benchmark.py` | Crop-tracker test set and regression runner |
| `google_sheets/import_run.gs` | Google Sheets importer |

## Appendix C: symbol legend

| ID | Code | Referent | Description | Placement |
|---|---|---|---|---|
| 1 | P9 | Toilet | Circle with a dot in the middle | free |
| 2 | P26 | Toilet | Square with open side on top edge | free |
| 3 | P34 | Toilet | Square with open sides on top and right edge | free |
| 4 | B | Toilet | Print letter "B" | free |
| 5 | Stair | Stairs | Lines in a step-like pattern | free |
| 6 | P7 | Stairs | Four adjacent rectangles | free |
| 7 | P35 | Stairs | Rectangle | free |
| 8 | S | Stairs | Print letter "S" | free |
| 9 | P54 | Elevator | Two adjacent rectangles | free |
| 10 | P72 | Elevator | Top half of two adjacent rectangles | free |
| 11 | E | Elevator | Print letter "E" | free |
| 12 | P27 | Elevator | Square with open side on right edge | free |
| 13 | P6 | Door / Emergency exit | Circle | wall |
| 14 | gap | Door / Emergency exit | Gap or break in wall | gap |
| 15 | P95 | Door / Emergency exit | Gap or break with vertical lines | wall |
| 16 | P99 | Door / Emergency exit | Four dots in a cross-like pattern | wall |
| 17 | P29 | Door / Emergency exit | Square with open side on bottom edge | wall |
| 18 | P48 | Door / Emergency exit | Circle with horizontal line through center | wall |
| 19 | P48_flipped | Door / Emergency exit | Circle with vertical line through center | wall |
| 20 | P10 | Dangerous / Inaccessible area | Cross | free |
| 21 | P17 | Dangerous / Inaccessible area | Triangle | free |
| 22 | P98 | Dangerous / Inaccessible area | X | free |
| 23 | P42 | Dangerous / Inaccessible area | Diamond | free |
| 24 | P47 | Dangerous / Inaccessible area | Circle with diagonal line through center | free |
| 25 | A10 | Dangerous / Inaccessible area | Area filled with diagonal lines | area |
| 26 | L4 | Corridor | Dashed line | area |
| 27 | A9 | Corridor | Diagonal lines | area |
| 28 | A11 | Corridor | Horizontal dots | area |
| 29 | A11_staggered | Corridor | Diagonal dots | area |
| 30 | R | Reception | Print letter "R" | free |
| 31 | P41 | Reception | Star | free |
| 32 | P11 | Reception | Square | free |
