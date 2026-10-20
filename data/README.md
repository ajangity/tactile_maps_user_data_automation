# Tactile map study data: how to read it

This folder holds the output of the finger-tracking pipeline. Each subfolder is **one run**: one session video of a participant exploring one printed tactile map, processed from start to finish. Each run records where the participant's index fingertips were on the map, which rooms and symbols they touched, for how long and in what order, and how much the paper moved.

This guide explains how the folder is organized, what every file and column means, how the numbers are counted, and what to watch for when you analyze them. You don't need to read any code.

---

## Contents

1. [Quick start: open the data in Google Sheets](#1-quick-start-open-the-data-in-google-sheets)
2. [How the folder is organized](#2-how-the-folder-is-organized)
3. [Key ideas behind the numbers](#3-key-ideas-behind-the-numbers)
4. [File by file](#4-file-by-file)
   - [summary.csv](#summarycsv): visits and time per room and symbol
   - [room_order.csv](#room_ordercsv): the path through the rooms
   - [corner_motion.csv](#corner_motioncsv): how much the paper moved
   - [corner_motion.json](#corner_motionjson)
   - [finger_positions.json](#finger_positionsjson): every fingertip position, every frame
   - [finger_paths.png](#finger_pathspng)
   - [map.json](#mapjson): the map's rooms and symbols
   - [floorplan.png](#floorplanpng)
5. [Answering common questions](#5-answering-common-questions)
6. [Data quality: what to check and what to watch for](#6-data-quality-what-to-check-and-what-to-watch-for)
7. [Units and coordinates cheat sheet](#7-units-and-coordinates-cheat-sheet)

---

## 1. Quick start: open the data in Google Sheets

The three CSV files in each run open in Excel or Google Sheets as they are. To bring **every run in this folder** into one spreadsheet at once, use the importer script, [`google_sheets/import_run.gs`](../google_sheets/import_run.gs).

**One-time setup, per spreadsheet:**

1. Create a new Google Sheet.
2. Click **Extensions → Apps Script**.
3. Delete what's in the editor, paste in the whole of `google_sheets/import_run.gs`, and save (Ctrl+S).
4. In the editor's left sidebar, click **+** next to **Services**, choose **Google Sheets API**, and click **Add**. This lets the script turn each CSV into a proper Google Sheets table.
5. Go back to the sheet and refresh the page. A **Tactile Maps** menu appears next to Help.

**Importing:**

1. Click **Tactile Maps → Import run CSVs...**
2. The first time, Google asks for permission: click **Continue**, choose your account, then **Advanced → Go to (project name) (unsafe) → Allow**. The script only reads the files you pick and writes to this spreadsheet. Then open the menu again.
3. Under **Folder**, pick this whole `data` folder to import every run, or pick a single run folder. Your browser will say it is "uploading" files. That only means the page can read them, and only the CSVs are used.
4. Click **Import**. The dialog shows progress ("Importing run 12 of 210...") and then a summary.

**What you get:**

- **One tab per run**, named `<map> - <video>`, for example `distractor_floorplan_E - S-19_Elevator`.
- On each tab, **three tables side by side**:
  - **Summary** in columns A–F
  - **Room Order** in columns H–J
  - **Corner Motion** in columns L–U
- Each table has its own header row, filter buttons and sorting.

Good to know:

- **Re-importing** replaces a run's tab and leaves your other tabs alone.
- **Repeated runs:** if the folder holds several runs of the same map and video (the same video processed twice), the **newest** run gets the tab, and the dialog says how many older runs it replaced.
- **Old runs** that only have a `data.csv` (the format before the CSVs were split) are skipped and listed.
- **If import fails with `PERMISSION_DENIED`,** you are signed in to more than one Google account. Use an Incognito window signed in to just the account that owns the sheet.

---

## 2. How the folder is organized

```
data/
├── README.md                                                ← this file
├── 2026-10-19_00-32-29__distractor_floorplan_E__S-19_Elevator/
│   ├── summary.csv           visits and time per room and symbol
│   ├── room_order.csv        every room visit, in time order
│   ├── corner_motion.csv     paper movement, 4 samples a second
│   ├── corner_motion.json    the same movement data, as JSON
│   ├── finger_positions.json every frame's fingertip positions on the map
│   ├── finger_paths.png      picture of both fingers' paths on the map
│   ├── map.json              the map's rooms and symbols, with coordinates
│   └── floorplan.png         copy of the map image the run used
├── 2026-10-19_22-31-54__distractor_floorplan_L4__S-1_Corridor/
│   └── ...
└── ...
```

### Run folder names

Every run folder is named `<date>_<time>__<map>__<video>`, with two underscores between the parts:

| Part | Example | Meaning |
|---|---|---|
| Date and time | `2026-10-19_00-32-29` | When the run was **processed** (not when the session was recorded). Folders sort chronologically. |
| Map | `distractor_floorplan_E` | The floorplan image used, from `png_reference_maps/`. The part after `distractor_floorplan_` is the **symbol ID** the participant was tested on (here `E`). |
| Video | `S-19_Elevator` | The session video, from `mp4_user_data/`. **`S-19`** is the participant; **`Elevator`** is the referent (what the symbol stands for). |

The referents are Corridor, Door, Emergency (S-1's files say EmergencyExit), Stair, Elevator, Bathroom, Inaccessible, Dangerous and Reception.

How each video is matched to its map is documented in [`png_reference_maps/README.md`](../png_reference_maps/README.md). A few of those pairings are still marked as assumptions; see [section 6](#6-data-quality-what-to-check-and-what-to-watch-for).

### Complete and incomplete runs

| What's in the folder | What it means |
|---|---|
| All 8 files | A finished run. |
| Only `map.json` | The run is **still processing**, or it **crashed or was force-stopped** before the end. Every other file is written only when a run ends. Don't analyze it. |
| `data.csv` instead of the three CSVs | A run from before the CSV split. The numbers are valid, but times are written as `M:SS.ss` and all tables are side by side in one file. The Sheets importer skips these. |

A run stopped early by pressing `q` is still saved. Its data covers only the part of the video before `q` was pressed, so check that the times reach the end of the session (see [section 6](#6-data-quality-what-to-check-and-what-to-watch-for)).

Processing the same video again creates a **new** folder; nothing is ever overwritten. When there are duplicates, use the newest folder unless you know otherwise.

---

## 3. Key ideas behind the numbers

Read this once. Every file depends on these definitions.

### Time is video time, in seconds

Every time in the data is **seconds since the start of the video file**, with two decimals: `23.89` means 23.89 s into the video. It is **not** time since the participant started, and not time since the map appeared. Sessions often start with the sheet face-down or off the table; in S-19, for example, the map only becomes visible at about 21 s. To get time since the map appeared, subtract the first time in the run's data, such as the first visit in `room_order.csv`.

### Only the index fingertips are tracked

The pipeline follows the **tip of the index finger of each hand**, up to two hands, using Google's MediaPipe hand model. Other fingers, palms and thumbs are not tracked. If a participant reads with their middle finger, that contact is missed.

Each fingertip is labeled **Left** or **Right**. The labels are kept stable over time, but they are a model's guess; see [section 6](#6-data-quality-what-to-check-and-what-to-watch-for).

### Everything is placed on the map, not in the video

The paper is found again in every video frame, even when it is rotated, slid around or partly covered by hands. Each fingertip is then converted from video pixels into **map pixels**: positions on the floorplan image itself. So a position means "this spot on the printed map" no matter where the paper was on the table. See [section 7](#7-units-and-coordinates-cheat-sheet) for converting pixels to inches.

### Rooms

The rooms are found automatically from the floorplan image. They are the areas enclosed by walls, with doorways treated as closed. Every map also has one **corridor**: the large dotted area connecting the rooms. Rooms are numbered `Room 1`, `Room 2`, ... in reading order (top-left first).

In the CSVs each room is **labeled by the symbol inside it**, followed by its number:

- `Stairs #5 (Room 1)`: Room 1 contains the Stairs symbol, legend entry #5.
- `Corridor #28 (Room 6)`: the corridor, named after its dot texture (legend #28 = horizontal dots, #29 = diagonal).
- `Elevator #11 (2) (Room 4)`: the second of several Elevator #11 symbols on this map. The `(1)`, `(2)`, `(3)` numbering follows reading order.
- If a room has no symbol inside it, it is labeled by the symbol on its doorway. If it has neither, it's just `Room N`.

### Symbols and the legend

Every symbol on the map is matched to the **tactile symbol legend** in `legends/`, which has 32 entries, and named `<what it stands for> #<legend ID>`. Examples are `Toilet #2`, `Elevator #11` and `Door / Emergency exit #14`. The legend ID tells you exactly which drawing it is, since several drawings can stand for the same thing. For example, Toilet has entries #1, #2 and #4.

Symbols come in two kinds:

- **Free-standing symbols** are drawn inside a room. A symbol is "touched" when a fingertip is inside the symbol's box, which is the symbol plus a 6-pixel margin (about 1/16 inch).
- **Entrance symbols** are drawn **on a wall**, like a doorway gap, a door mark, or a symbol marking an entrance. They are listed as `... (entrance to Room N)`. Because they're thin marks on a wall, they're touched over a slightly larger **entrance zone**: the mark plus 18 pixels (about 3/16 inch) around it.

### Visits, brushes and the 1-second rule

Rooms and symbols use the same rules to decide what counts as a **visit**:

1. A visit **starts** on the first frame that any tracked fingertip is inside the room or symbol.
2. It **ends** once no fingertip has been inside for 0.4 s. The 0.4 s allowance covers the model briefly losing a hand. The recorded exit time is the last moment a finger was actually inside, not the end of that 0.4 s.
3. **Two hands inside at the same time count as one visit**, not two.
4. Only visits lasting **at least 1 second** count. Anything shorter is a **brush**, a finger passing over, and is **not** included in any visit count or time in the CSVs.

So "Room Visits = 3, Room Time = 10.01" means three separate stays of at least 1 s each, totaling 10.01 s, with brushes left out.

### The main timer: time on the map

The `Entire map` row in `summary.csv` uses a different, stricter timer. It runs on every frame that **at least one fingertip is on the map** and stops on the first frame none is. There's no 1-second minimum and no 0.4 s allowance. This is the participant's total hands-on time with the map. It also stops while the map is out of view (face-down, lifted, covered or off the table), because nothing can be placed on the map then.

---

## 4. File by file

### `summary.csv`

**Visits and time for each room and symbol.** This is the main results table, with one row per room or symbol.

| Column | Meaning |
|---|---|
| **Element/Symbol** | What the row is about (see "Row types" below). |
| **Symbol Visits** | Number of visits (≥ 1 s) to the symbol(s). On a room row, this counts the symbol(s) that label the room. |
| **Symbol Time (s)** | Total seconds of those symbol visits. |
| **Room Visits** | Number of visits (≥ 1 s) to the whole room. |
| **Room Time (s)** | Total seconds of those room visits. On the `Entire map` row, this is the **main timer**: total time any fingertip was on the map. |
| **Room Dimension (in)** | The room's size in inches as `(width, height)`: the width and height of the smallest box around the room, not its area. On the `Entire map` row, it's the sheet: `(8.5, 11)`. |

**Row types, in order:**

1. **`Entire map`**: one row. Only *Room Time (s)* (time on the map) and *Room Dimension (in)* (the sheet size) are filled.
2. **Room rows**, one per room in room-number order, for example `Stairs #5 (Room 1)`. The symbol columns describe the symbol(s) in the room's label. If a room is labeled by two symbols (`A / B (Room 3)`), their visits and times are added together. The corridor and unlabeled rooms have **blank** symbol columns.
3. **Remaining symbol rows**: every symbol not already used as a room's label. These are mostly the entrance symbols (`Door / Emergency exit #16 (1) (entrance to Room 3)`). Only the symbol columns are filled.

**Example** (S-19, map E; first rows):

```
Element/Symbol,Symbol Visits,Symbol Time (s),Room Visits,Room Time (s),Room Dimension (in)
Entire map,,,,39.57,"(8.5, 11)"
Stairs #5 (Room 1),2,2.97,2,7.27,"(2.17, 1.36)"
Toilet #2 (Room 2),0,0.0,1,2.04,"(2.15, 1.38)"
Corridor #28 (Room 6),,,3,4.37,"(2.99, 10.03)"
Door / Emergency exit #16 (1) (entrance to Room 3),1,1.47,,,
```

How to read it:

- The participant had fingers on the map for **39.57 s** in total.
- They visited Room 1 (the Stairs room) **twice**, spending **7.27 s** in it. Of that, **2.97 s** over 2 visits was on the Stairs symbol itself.
- They entered Room 2 once (2.04 s) but **never stayed on its Toilet symbol for 1 s or more** (0 visits). They may have brushed it.

**Things to keep in mind:**

- **Blank and 0 are different.** `0` means "this has a timer, and it was never visited". Blank means "this column doesn't apply to this row".
- **Room times don't add up to the map time.** Two hands can be in two rooms at once, which counts toward both rooms. Brushes and moments on walls or margins count toward neither. Symbol time is part of room time, since the symbol is inside the room, but it is timed separately.
- **Visits are counted independently per room or symbol.** One visit to a symbol usually happens during one visit to its room, but the two counts can differ.

### `room_order.csv`

**Every room visit, in the order it happened, revisits included.** This is the participant's path through the map.

| Column | Meaning |
|---|---|
| **Order** | The room visited, with the same labels as `summary.csv`. The first row is the first room visited. |
| **Time Entered (s)** | Video time the visit started. |
| **Time Exited (s)** | Video time the visit ended (the last moment a finger was inside). |

```
Order,Time Entered (s),Time Exited (s)
Corridor #28 (Room 6),23.89,25.29
Stairs #5 (Room 1),25.32,30.73
Elevator #11 (2) (Room 4),27.16,28.93
Elevator #11 (2) (Room 4),29.6,34.07
```

- Rows are sorted by entry time. Only visits of at least 1 s appear here; brushes don't.
- **Visits can overlap.** In the example, Room 1 (25.32–30.73) and Room 4 (27.16–28.93) overlap because the participant had one hand in each room. This is real two-handed exploration, not an error.
- The same room appears again each time it's revisited. The number of rows for a room equals its *Room Visits* in `summary.csv`.
- Visit length = *Time Exited* − *Time Entered*.
- Only **rooms** appear here, not symbols.

### `corner_motion.csv`

**How much the paper moved over the session.** Four times a second, the position of each of the map's four corners in the video is compared with its position a quarter-second earlier.

| Column | Meaning |
|---|---|
| **Frame #** | Video frame number of this sample (counting from 1). |
| **Timestamp (s)** | Video time of this sample. |
| **TL dx, TL dy** | How far the map's **top-left** corner moved since the previous sample, in **video pixels**. `dx` is right (+) or left (−); `dy` is down (+) or up (−). |
| **TR dx, TR dy** | Same for the **top-right** corner. |
| **BR dx, BR dy** | Same for the **bottom-right** corner. |
| **BL dx, BL dy** | Same for the **bottom-left** corner. |

- **The corners are the map's own corners**, matching the floorplan image. "Top-left" is the map's top-left even if the participant turned the paper upside down on the table.
- **All zeros** means the paper didn't move. Values of ±1–2 are normal tracking jitter; larger values mean the paper was pushed, slid or rotated.
- **Blank cells** mean the map wasn't in view for that sample (face-down, off the table or not found yet). The first sample after the map reappears is also blank, since there's no earlier position to compare it to.
- How far a corner moved = √(dx² + dy²). If all four corners move by about the same amount in the same direction, the paper was **slid**. If they move different amounts or directions, it was **rotated or tilted**.
- These are **video pixels**, not map pixels or inches, so the scale depends on how far the camera was from the table. Compare movement between sessions recorded with the same camera setup, or divide by the paper's size in the video.

### `corner_motion.json`

The same movement data as `corner_motion.csv`, for scripts:

```json
{"TL": [null, null, [0, -2], [0, 0], ...],
 "TR": [...], "BR": [...], "BL": [...]}
```

Each corner has one `[dx, dy]` pair per sample, in the same order as the CSV rows. `null` means the same as a blank CSV row.

### `finger_positions.json`

**Every video frame's fingertip positions on the map.** Use this for anything the CSVs don't cover: custom regions, distances traveled, speed, heat maps, time to first touch a particular symbol, and so on.

```json
[
  {"frame": 1,    "t_ms": 33,    "Left": null,           "Right": null},
  {"frame": 1501, "t_ms": 50083, "Left": [664.0, 445.4], "Right": [640.0, 245.6]},
  ...
]
```

| Field | Meaning |
|---|---|
| `frame` | Video frame number (from 1). There's one entry for every frame of the video. |
| `t_ms` | Video time in **milliseconds**. Divide by 1000 to match the CSVs. |
| `Left`, `Right` | `[x, y]` position of that hand's index fingertip **in map pixels**, or `null` if that hand wasn't found on this frame. |

- `[0, 0]` is the **top-left corner of the floorplan image**; x increases to the right and y increases downward.
- A fingertip can be slightly **outside** the map (negative, or larger than the image), for example when it rests on the margin or the table just off the sheet. Positions more than half a map-width off the map are dropped as `null`. To keep only touches on the map, keep positions with 0 ≤ x < image width and 0 ≤ y < image height (`width` and `height` are in `map.json`).
- `null` happens when the hand is out of view, covered, lifted off the table, or the model missed it for a few frames. It also happens on **every** frame while the map itself isn't in view.
- Paired with `map.json`, you can tell which room or symbol any position falls in: compare it to the room `polygon`s and symbol `bbox`es.

### `finger_paths.png`

A picture of the floorplan with both fingertips' complete paths drawn on it: **Left in red, Right in blue**. Gaps in a line are moments the hand wasn't seen or jumped too far between frames. This is the quickest way to sanity-check a run: you should see the paths following the walls, symbols and corridors the participant actually explored.

### `map.json`

**The map's layout as the pipeline understood it**: every room and symbol, with coordinates in map pixels. You need this to connect `finger_positions.json` to rooms and symbols, or to check that rooms and symbols were found correctly.

Most useful fields:

| Field | Meaning |
|---|---|
| `width`, `height` | Size of the floorplan image in pixels (typically 816 × 1056). |
| `page_in`, `px_per_inch` | Sheet size in inches and pixels per inch (96 for the 816 × 1056 maps). |
| `rooms[]` | One entry per room. |
| ↳ `id`, `name` | Room number and name (`"Room 1"`). |
| ↳ `polygon` | The room's outline as `[[x, y], ...]` points in map pixels. |
| ↳ `bbox`, `size_in` | Bounding box `[x0, y0, x1, y1]` and its size in inches (the CSV's *Room Dimension*). |
| ↳ `centroid`, `area_px` | Centre point and area in pixels. Divide the area by `px_per_inch`² for square inches. |
| ↳ `pattern` | Only on the corridor: its legend dot pattern (`"Corridor #28"`). |
| `symbols[]` | One entry per symbol. |
| ↳ `id`, `name` | Symbol number and name (`"Stairs #5"`, `"Elevator #11 (2)"`). |
| ↳ `legend_id`, `referent`, `description` | Which legend entry it matched, what it stands for, and how the legend describes the drawing. |
| ↳ `match` | How closely it matched the legend drawing: 0 is perfect, and anything above 1.5 counts as no match (`"Unknown symbol"`). Values near 1.5 are worth checking by eye. |
| ↳ `room` | The room number it's in (`null` for symbols on walls). |
| ↳ `bbox` | The symbol's box `[x0, y0, x1, y1]` in map pixels, including the 6-pixel margin used for timing. |
| ↳ `on_wall`, `entrance`, `entrance_to`, `entrance_zone` | For symbols on a wall: the room(s) it leads into and the larger zone used to time it. |
| `lines`, `border`, `ink` | The traced walls and border, used internally. |

**Renaming rooms:** room names are generated (`Room 1`, ...). To give them meaningful names (say `"Office"`), edit `name` in the most recent run's `map.json` for that map. The **next** run of the same map picks up the new names, and its CSVs then say, for example, `Toilet #2 (Office)`. Existing CSVs aren't changed.

### `floorplan.png`

An exact copy of the map image the run used, so the folder is complete without the rest of the project. All map-pixel coordinates in `map.json` and `finger_positions.json` refer to this image.

---

## 5. Answering common questions

| Question | Where to look |
|---|---|
| How long did the participant spend on the map overall? | `summary.csv` → `Entire map` → *Room Time (s)* |
| Which rooms did they explore, and which did they skip? | `summary.csv` → *Room Visits*. 0 means never visited for ≥ 1 s. |
| How much attention did the target symbol get? | `summary.csv` → that symbol's *Symbol Visits* and *Symbol Time (s)* |
| What order did they explore the rooms in? | `room_order.csv`, top to bottom |
| When did they first reach a room? | `room_order.csv` → the first row with that room's label → *Time Entered (s)* |
| How long after the map appeared did they reach it? | That *Time Entered* minus the first *Time Entered* in `room_order.csv` |
| How often did they come back to a room? | `summary.csv` → *Room Visits* − 1, or count its rows in `room_order.csv` |
| When did they first touch a particular **symbol**? | Not in the CSVs. Use `finger_positions.json`: the first frame where either fingertip is inside that symbol's `bbox` (or `entrance_zone`) from `map.json`. |
| Did they use one hand or two? | `finger_positions.json` (frames where both are non-null), or overlapping rows in `room_order.csv` |
| Did they move or rotate the paper? | `corner_motion.csv`: non-zero rows, and whether the four corners moved alike |
| How do maps or symbols compare across participants? | Import the whole folder into Google Sheets ([section 1](#1-quick-start-open-the-data-in-google-sheets)) and compare the tabs for the same map, or the same referent across maps |

Before comparing **symbols across different maps**, check which legend ID each map's target symbol has (`legend_id` in `map.json`, or the `#` number in the labels). Maps for the same referent use different symbols. That's the point of the study.

---

## 6. Data quality: what to check and what to watch for

**Quick checks for each run:**

1. **Look at `finger_paths.png`.** The paths should sit on the map's features. Paths that look shifted, mirrored or scattered mean the paper tracking struggled.
2. **Check the run covers the whole session.** The last *Time Exited* in `room_order.csv`, or the last non-null entry in `finger_positions.json`, should be near the end of the participant's exploration. If it stops early, the run was likely stopped with `q`.
3. **Check the time on the map is plausible** against how long the participant actually explored.

**Known limitations:**

- **No data while the map is out of view.** Before the map is face-up, while it's lifted or carried, or if it's almost fully covered, no positions are recorded and every timer pauses.
- **Hands can be missed.** The hand model occasionally loses a hand for a few frames, especially with fast motion, poor lighting, or a hand mostly hidden under another. The 0.4 s allowance stops this from splitting visits, but a fingertip resting somewhere for a long time while undetected won't be counted.
- **Left and Right can be wrong.** The labels come from the hand model and are smoothed over time, but crossed or overlapping hands can swap them. Use the combined data (either hand) unless handedness matters for your question.
- **Index fingers only.** Exploration with other fingers or the palm isn't recorded.
- **Rooms and symbols are found automatically.** This was checked closely on map E, where all 20 symbols (doorways included) were matched to the right legend entry. Other maps haven't all been checked by hand. If a label looks wrong, check the symbol's `match` value and compare `map.json` with `floorplan.png`. Very similar legend drawings (#13 vs #18, #21 vs #23) are the most likely to be confused. Area patterns (#25–#27) aren't detected as symbols, and doorways and wall symbols are only found on horizontal or vertical walls.
- **Some video-to-map pairings are assumptions.** Videos involving P29, A9 and A11 use a best guess for which of two similar map files was shown; see [`png_reference_maps/README.md`](../png_reference_maps/README.md). Confirm these before relying on runs for those maps.
- **Tuned on one session.** The tracking settings were developed on S-19 (map E). Sessions with very different lighting, camera angle or map style may track less reliably; `finger_paths.png` is the fastest way to spot this.

---

## 7. Units and coordinates cheat sheet

| Quantity | Unit | Notes |
|---|---|---|
| All times in the CSVs | seconds of video time | Two decimals; counted from the start of the video file |
| `t_ms` in `finger_positions.json` | milliseconds of video time | ÷ 1000 → seconds |
| Room Dimension | inches | Bounding-box width × height; the sheet is 8.5 × 11 in (US Letter) |
| Positions in `finger_positions.json` and `map.json` | map pixels | `[0, 0]` = top-left of `floorplan.png`; x right, y down |
| Map pixels → inches | ÷ 96 | For 816 × 1056 maps; otherwise use `px_per_inch` from `map.json` |
| Corner movement (`corner_motion.*`) | video pixels per ¼ second | Depends on camera distance; not comparable to map pixels |
| Visit | ≥ 1 s continuous presence | With a 0.4 s allowance for brief tracking dropouts; two hands at once = one visit |
| Video frame rate | ~30 fps | One `finger_positions.json` entry per frame |
