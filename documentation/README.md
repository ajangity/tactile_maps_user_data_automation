# tactile_maps_user_data_automation

Tracks where a user's fingers go on a printed tactile map in a video, and reports how long they spent in each room and on each symbol.

**More documentation in this folder:**
- [`architecture.html`](architecture.html): an interactive diagram of every file, its inputs and outputs, layer by layer. Open it in a browser and click a box.
- [`CODEBASE.md`](CODEBASE.md): the technical reference (modules, data formats, algorithms, settings, measured results, known limitations).

Run every command below from the project folder (one level up from this file).

## Run it

```bash
python run_pipeline.py "S-19_Elevator.mp4" distractor_floorplan_E.png
python run_pipeline.py video.mp4 map.png --no-display      # batch mode, no window
python dashboard.py "data/<run folder>/dashboard.json"     # rebuild a dashboard later
python edge_tracing.py distractor_floorplan_E.png          # see what edge tracing finds
```

Keys while the video plays: **c** shows or hides the on-screen list of commands, **Space** stops the video so you can fix the crop by hand, **t** shows trails and rooms (then **l** / **r** turn the left / right trail on and off), **e** shows the paper-edge debug view, **w** shows the edge-tracing debug view, **[ / ]** changes speed, and **q** saves and quits.

Requires `opencv-python`, `mediapipe`, `numpy`, and `hand_landmarker.task` in this folder.

## Files

| File | What it does |
|---|---|
| `run_pipeline.py` | The file you run. Calls everything below in order, every frame, then saves the results. |
| `paper_locator.py` | **Auto-crop part 1 (steps 1–8).** Builds the map JSON, finds the paper, scores orientation, sets the search window, sorts edges by side. |
| `corner_fitting.py` | **Auto-crop part 2 (steps 9–17).** Turns edges into exact corners, refines the crop with the traced lines, picks the best crop. |
| `edge_tracing.py` | Finds the printed black lines in an image, optionally only inside a region you give it. Once the map is locked, it only processes the pixels of the paper's crop (plus the 35% search margin while tracking, since the paper moves between frames). Before the map is found it traces the whole frame, because that's how the paper is found. |
| `crop_checker.py` | Sanity checks (step 14), Backup #1 PNG/SIFT (18), Backup #2 optical flow (19), and the rigid-paper and visible-corner checks (21). |
| `auto_crop.py` | The running order of the crop steps. Also searches for the map before it's found and recovers when it's lost (step 20). |
| `manual_crop.py` | Stop the video and drag the crop's corners. Only runs when you press Space. |
| `finger_tracking.py` | Steps 22–24: MediaPipe, mapping fingertips onto the map, and keeping Left/Right straight. |
| `room_tracking.py` | Works out exactly which PNG pixels belong to each room, and which room each symbol is in. |
| `timing.py` | Enter/exit timers: one for the paper, one per room, one per symbol. |
| `dashboard.py` | Saves the results JSON and builds the HTML dashboard. |
| `label_symbols.py` | Optional. Draws extra boxes by hand, each of which gets a timer too. |

## Outputs

Every run creates its own folder, named with the date and time, the floorplan PNG, and the video:

```
data/
  2026-10-08_14-30-05__distractor_floorplan_E__S-19_Elevator/
    map.json  rooms.png  frame_lines.json  finger_paths.png
    session.json  dashboard.json  dashboard.html
```

| File | Contents |
|---|---|
| `map.json` | Every black line on the PNG as nodes and edges, plus the printed border, the rooms (exact outlines) and the symbols. If you rename rooms or symbols in it, the next run of the same floorplan picks those names up. |
| `rooms.png` | Picture of the detected rooms and symbols, for checking. |
| `frame_lines.json` | Same graph format, traced (inside the crop) from the video frame where the map was first found, in frame pixels and in PNG pixels. Includes the score breakdown for that crop. |
| `finger_paths.png` | Left trail in red and right trail in blue, drawn on the map. |
| `session.json` | Every frame: crop corners, how the crop was found, its score, and each fingertip. |
| `dashboard.json` / `dashboard.html` | The results, and the dashboard (opens automatically). |

Use `--data-dir somewhere/else` to put the run folders somewhere other than `data/`.

All map coordinates are the PNG's own pixels, in the PNG's own orientation.

## The pipeline

1. Loads the tactile map PNG. **It's no longer rotated 180 degrees.** The orientation is now worked out from the video itself (step 4), so we don't need to assume how the paper sits. Edge tracing reads every black line on the PNG and saves them as a graph in `map.json`: nodes are line ends, corners and T-junctions, and edges are the lines between them. Room tracking then works out the rooms. It thickens the walls just enough to seal the doorways, so each room becomes a sealed white area. Then it grows each room back out to the real walls, so every pixel inside the map belongs to exactly one room. It finds the right doorway width by itself, by testing a range of widths and keeping the one where the room count stays stable. On map E that's 11 rooms (10 rooms plus the corridor). Every symbol is tagged with the room it's in.
2. Using Otsu's method, the script splits bright pixels (paper) from darker ones (table, hands, etc.). It also throws out bright pixels that are too colorful, since skin can be nearly as bright as paper but is more saturated. That stops a hand resting between two papers from joining them into one blob.
3. *(First frame, and when lost.)* The script makes rough 4-corner outlines to start from, in two independent ways:
   - **Paper edges:** each big white blob, with its corners fitted the same way as steps 8–12.
   - **Edge tracing:** each cluster of printed walls, using the outline of the printed border. This works even when the paper's own edge is broken by a hand or touching the other sheet.
4. *(First frame, and when lost.)* **Orientation.** One outline already captures the paper's exact angle, whatever it is (37 degrees, tilted in perspective, anything). What it can't tell you is which corner is the map's top-left, because a rectangle looks the same from all four sides. So each outline gets 4 labelings, the true angle plus 0/90/180/270 degrees, and the **scoring rubric** picks the real one:
   - The frame's traced lines are pushed through the candidate crop into the PNG's pixels and compared class by class: walls to walls, symbols to symbols, dots to dots.
   - For each class, **precision** is the share of what we traced that lands on a printed line, and **recall** is the share of the printed lines we traced. They're combined so precision counts twice as much, because a hand covering the map should cost less than tracing the wrong thing.
   - Score = 0.4 × walls + 0.4 × symbols + 0.2 × dots, from 0 to 1.

   Symbols decide the orientation, because walls are often symmetric but letters and icons never are. The best 3 candidates are fine-tuned against the traced lines (step 13) and re-scored, and a score of at least 0.6 locks on. The video can start with the sheet face-down, so the script keeps searching every 10 frames until something scores that high. On S-19:

   | Candidate | Score |
   |---|---|
   | Right orientation | 0.98 |
   | Same map rotated 180° | 0.40 |
   | Rotated 90° or 270° | 0.10–0.14 |
   | The decoy sheet | 0.04 |
   | A blank page | under 0.07 |

5. *(Removed: manual crop is no longer a pipeline step. Press Space at any time to stop the video and drag the corners; see `manual_crop.py`.)*
6. On every frame after that, the script only looks near last frame's crop, grown by 35%.
   - **I tested 20%.** The crop couldn't be confirmed on 3× as many frames, and processing was slower, because losing the map triggers the expensive whole-frame search.
   - **It isn't about motion.** On S-19, 99.9% of frames move less than 2.4% of the paper's diagonal. The paper/desk brightness split is computed inside this window, and more desk in view keeps it stable when hands cover the page.
   - **Going smaller would only save about 3 ms per frame.**
7. Every separate white blob in that window is handled on its own, closest to last frame first, up to 3. A second sheet's edges can never leak into the real page's line fit.
8. Each straight edge segment is assigned to the top, bottom, left or right side. Its angle must be within 20 degrees of that side's angle last frame, and it goes to whichever side's line it sits closest to.
9. Points are placed every 4 px along each segment, so long clean edges count for more. Segments with a loose angle match get fewer points.
10. Each side gets a robust line of best fit: fit once, throw out points that are clearly off, then fit again.
11. *(Removed: reusing a side's line from an earlier frame when the side looked hidden. It kept stale lines alive even once the real edge was visible again. Hidden sides are now covered by step 13, which doesn't need the paper's edges at all.)*
12. The 4 side lines are extended until they meet, which gives 4 corners even if a corner is hidden. If no paper blob gives a usable outline, the same window is re-fitted on the **ink-border mask** (everything darker than the paper), which uses the sharp printed border and the paper/desk boundary instead. This is Vivaan's fallback, used both per frame and on the first lock.
13. **Edge tracing improves the crop, and this is now the main way the crop is found each frame.** Starting from last frame's crop, the script lines up the PNG's printed lines with the lines traced in the frame, using an iterative closest-point (ICP) fit.
    - Each traced pixel is paired with the nearest printed pixel, and the crop is re-fit to those pairs. The radius for pairing shrinks from 14 to 4 px over 5 rounds.
    - RANSAC ignores hand outlines and other strays.
    - It uses every printed line on the map, not just the border, so a hidden corner or side doesn't matter.

    If this crop scores at least 0.6, steps 6–12 are skipped for that frame. On S-19 that was about 92% of frames, and it saves about 75 ms each time.
14. Every candidate crop has to pass the checks in `crop_checker.py`:
    - every corner within 25 degrees of square
    - width/height ratio within 35% of the PNG's
    - big enough, and not hanging off the frame
    - while the lock is trusted, no corner jumping more than 8% of the frame's diagonal
15. For paper-edge crops, corners whose two sides were both seen are snapped to the exact corner pixel. The snap is ignored if it would move the corner more than 8 px.
16. If at least 2 trusted corners are within 2 px of last frame, the paper didn't move, so last frame's crop is kept as-is with no jitter.
17. The best-scoring candidate wins if it scores at least 0.4. It's blended 65% new / 35% old.
18. **Backup #1 (PNG):** SIFT-match the frame against the PNG near the last crop. The result must pass step 14 and score at least 0.35.
19. **Backup #2 (optical flow):** follow up to 700 paper points from last frame. Same checks. The points are only picked when this backup actually runs (it used to cost 30% of every frame). If this fails too, the crop is held.
20. Only steps 13/17 and Backup #1 count as confirming the crop. After 20 frames without confirmation the crop is "lost", and the whole-frame search (steps 2–4) runs every 3 frames. If several results are close, the one nearest the last crop wins.
21. The paper is rigid: if 3 corners stayed put and 1 jumped, the jump is corrected. Each corner is drawn green if it's visible and orange if it's covered.
22. The script crops around the paper plus some margin, enlarges the crop up to 2.5×, and runs MediaPipe for the index fingertip of up to 2 hands.
23. The fingertip goes crop → frame → PNG pixels, through a homography built fresh every frame from that frame's corners. It's marked `on_paper` if it lands on the map. The trail breaks instead of drawing a straight line across any jump over 140 px.
24. Each hand is matched to whichever hand's wrist was closest last frame, solving both hands together. Then it's labeled Left/Right from a short, fading memory of MediaPipe's own Left/Right calls, weighted by MediaPipe's confidence. One low-confidence frame can't swap the labels, and a wrong label can't stick (MediaPipe's labels are reliable for a camera across the table).
25. **Timers** (`timing.py`): one for the paper, one per room, and one per symbol (plus any boxes drawn with `label_symbols.py`). Every timer works the same way:
    - **Start:** a finger enters the room's exact pixels.
    - **End:** no finger has been in it for 0.4 s, either because it left or because the hand vanished. The visit ends at the last moment a finger was actually inside.
    - **Two hands at once:** still one visit.
    - **Minimum length:** only visits of 1 second or more count. Shorter ones are "brushes".
    - **Result:** each room is *visited*, *brushed only*, or *missed* (never touched).
26. When the video ends (or on q), everything in **Outputs** is saved and the dashboard opens.
