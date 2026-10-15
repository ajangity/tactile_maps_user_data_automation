# png_reference_maps

Every tactile floorplan a participant might be shown. Names are `distractor_floorplan_<SymbolID>.png`. The symbol IDs are the old IDs from the "Referents & Symbols" tab of the symbols doc.

## Matching a video to its map

`S-<N>_<Referent>.mp4` → participant S-N's row for that referent, column B of the Results sheet → `distractor_floorplan_<value>.png`

Example: S-19 chose `E` for Elevator, so `S-19_Elevator.mp4` uses `distractor_floorplan_E.png`. That map has three "E" symbols, which fits the "find all 3" task.

| Column B value | PNG |
|---|---|
| P7, P9, P54, E, R, B, … | `distractor_floorplan_<value>.png` (same name) |
| P48 Flipped | `distractor_floorplan_P48_flipped.png` |
| Gap | `distractor_floorplan_gap.png` |
| Stair (Nick Giudice's stair symbol) | `distractor_floorplan_Stair.png` (**not** `S.png`, which is the letter S) |

- A10, P98, P42 and P10 use the same file whether the referent is Inaccessible or Dangerous.
- P95, P99, P48 and Gap use the same file whether the referent is Door or Emergency Exit.

## Still unconfirmed

These are the places where one sheet value could mean two different PNGs:

1. **P29**: there are two maps, `P29.png` and `doors_P29.png`. The guess is that `doors_P29` goes with Door (S-1, S-7, S-15) and `P29` goes with Emergency Exit (S-3, S-17, S-20).
2. **A9 (Corridor)**: there are two maps, `A9.png` and `corridor_A9.png`. The guess is that every run used `A9.png`, which shows the pattern in three rooms.
3. **A11**: the sheet only says A11. The guess is that every run used `A11.png` and never `A11_staggered.png`.

## Notes

- The Drive folder has duplicate shortcuts, including two for P41. Match on filename only.
- The Drive copies have transparent backgrounds. Flatten them onto white before running the pipeline, or the background shows up as black.
