"""This file is a temporary helper that runs run_pipeline.py on every
video/map pair in the data folders, one after another.

Temporary helper: run run_pipeline.py on every video with its paired map, one at a time.

Usage:
    python loop_pipeline.py

Videos are read from mp4_user_data/ and maps from png_reference_maps/.
Pairs whose video isn't on disk yet are skipped. Ctrl+C stops the batch.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "mp4_user_data"
MAP_DIR = "png_reference_maps"

# ---------------------------------------------------------------------------
# VIDEO -> REFERENCE MAP LOOKUP (all 20 participants x 9 referents)
#
# Rule: S-<N>_<Referent>.mp4 -> symbol participant S-<N> chose for that
# referent (Results sheet, column B) -> png_reference_maps/distractor_floorplan_<symbol>.png
#
# Special cases:
#   "P48 Flipped" -> P48_flipped.png   |  "Gap" -> gap.png
#   "Stair" (Nick Giudice's symbol) -> Stair.png (S.png is the letter S)
#   P29 for Door -> doors_P29.png; P29 for Emergency Exit -> P29.png   (ASSUMPTION - confirm)
#   A9 corridor -> A9.png (not corridor_A9.png); A11 -> A11.png (not A11_staggered.png) (ASSUMPTION)
#   S-1 uses "EmergencyExit"; every other participant uses "Emergency".
#   Unused maps: A11_staggered, corridor_A9, S, P72.
#   [NOT ON DRIVE] = recorded (times in Results) but never uploaded.
# ---------------------------------------------------------------------------
PAIRS = [
    ("S-1_Corridor.mp4", "distractor_floorplan_L4.png"),
    ("S-1_Door.mp4", "distractor_floorplan_doors_P29.png"),
    ("S-1_EmergencyExit.mp4", "distractor_floorplan_P48_flipped.png"),
    ("S-1_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-1_Elevator.mp4", "distractor_floorplan_E.png"),
    ("S-1_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-1_Inaccessible.mp4", "distractor_floorplan_A10.png"),
    ("S-1_Dangerous.mp4", "distractor_floorplan_P98.png"),
    ("S-1_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-2_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-2_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-2_Emergency.mp4", "distractor_floorplan_P48_flipped.png"),
    ("S-2_Stair.mp4", "distractor_floorplan_Stair.png"),
    ("S-2_Elevator.mp4", "distractor_floorplan_P27.png"),
    ("S-2_Bathroom.mp4", "distractor_floorplan_P26.png"),
    ("S-2_Inaccessible.mp4", "distractor_floorplan_P98.png"),
    ("S-2_Dangerous.mp4", "distractor_floorplan_P42.png"),
    ("S-2_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-3_Corridor.mp4", "distractor_floorplan_A11.png"),
    ("S-3_Door.mp4", "distractor_floorplan_gap.png"),
    ("S-3_Emergency.mp4", "distractor_floorplan_P29.png"),
    ("S-3_Stair.mp4", "distractor_floorplan_Stair.png"),
    ("S-3_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-3_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-3_Inaccessible.mp4", "distractor_floorplan_P10.png"),
    ("S-3_Dangerous.mp4", "distractor_floorplan_P98.png"),
    ("S-3_Reception.mp4", "distractor_floorplan_P11.png"),
    ("S-4_Corridor.mp4", "distractor_floorplan_A11.png"),
    ("S-4_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-4_Emergency.mp4", "distractor_floorplan_P48.png"),
    ("S-4_Stair.mp4", "distractor_floorplan_Stair.png"),
    ("S-4_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-4_Bathroom.mp4", "distractor_floorplan_B.png"),
    ("S-4_Inaccessible.mp4", "distractor_floorplan_A10.png"),
    ("S-4_Dangerous.mp4", "distractor_floorplan_P98.png"),
    ("S-4_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-5_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-5_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-5_Emergency.mp4", "distractor_floorplan_P48.png"),
    ("S-5_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-5_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-5_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-5_Inaccessible.mp4", "distractor_floorplan_P98.png"),           # [downloading]
    ("S-5_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-5_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-6_Corridor.mp4", "distractor_floorplan_A11.png"),
    ("S-6_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-6_Emergency.mp4", "distractor_floorplan_P48.png"),
    ("S-6_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-6_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-6_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-6_Inaccessible.mp4", "distractor_floorplan_A10.png"),
    ("S-6_Dangerous.mp4", "distractor_floorplan_P98.png"),
    ("S-6_Reception.mp4", "distractor_floorplan_P41.png"),
    ("S-7_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-7_Door.mp4", "distractor_floorplan_doors_P29.png"),
    ("S-7_Emergency.mp4", "distractor_floorplan_P48.png"),
    ("S-7_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-7_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-7_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-7_Inaccessible.mp4", "distractor_floorplan_P47.png"),
    ("S-7_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-7_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-8_Corridor.mp4", "distractor_floorplan_A11.png"),
    ("S-8_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-8_Emergency.mp4", "distractor_floorplan_P48_flipped.png"),
    ("S-8_Stair.mp4", "distractor_floorplan_Stair.png"),
    ("S-8_Elevator.mp4", "distractor_floorplan_E.png"),
    ("S-8_Bathroom.mp4", "distractor_floorplan_B.png"),
    ("S-8_Inaccessible.mp4", "distractor_floorplan_A10.png"),
    ("S-8_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-8_Reception.mp4", "distractor_floorplan_R.png"),                # [downloading]
    ("S-9_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-9_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-9_Emergency.mp4", "distractor_floorplan_P99.png"),
    ("S-9_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-9_Elevator.mp4", "distractor_floorplan_E.png"),
    ("S-9_Bathroom.mp4", "distractor_floorplan_P34.png"),
    ("S-9_Inaccessible.mp4", "distractor_floorplan_P47.png"),
    ("S-9_Dangerous.mp4", "distractor_floorplan_A10.png"),
    ("S-9_Reception.mp4", "distractor_floorplan_P41.png"),
    ("S-10_Corridor.mp4", "distractor_floorplan_L4.png"),
    ("S-10_Door.mp4", "distractor_floorplan_P48.png"),                  # [downloading]
    ("S-10_Emergency.mp4", "distractor_floorplan_P95.png"),
    ("S-10_Stair.mp4", "distractor_floorplan_P35.png"),
    ("S-10_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-10_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-10_Inaccessible.mp4", "distractor_floorplan_A10.png"),
    ("S-10_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-10_Reception.mp4", "distractor_floorplan_P11.png"),
    ("S-11_Corridor.mp4", "distractor_floorplan_L4.png"),
    ("S-11_Door.mp4", "distractor_floorplan_gap.png"),
    ("S-11_Emergency.mp4", "distractor_floorplan_P48.png"),
    ("S-11_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-11_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-11_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-11_Inaccessible.mp4", "distractor_floorplan_P98.png"),
    ("S-11_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-11_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-12_Corridor.mp4", "distractor_floorplan_L4.png"),
    ("S-12_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-12_Emergency.mp4", "distractor_floorplan_P48.png"),
    ("S-12_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-12_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-12_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-12_Inaccessible.mp4", "distractor_floorplan_P47.png"),
    ("S-12_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-12_Reception.mp4", "distractor_floorplan_P41.png"),
    ("S-13_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-13_Door.mp4", "distractor_floorplan_P6.png"),                   # [downloading]
    ("S-13_Emergency.mp4", "distractor_floorplan_P48_flipped.png"),
    ("S-13_Stair.mp4", "distractor_floorplan_P7.png"),                  # [NOT ON DRIVE]
    ("S-13_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-13_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-13_Inaccessible.mp4", "distractor_floorplan_A10.png"),
    ("S-13_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-13_Reception.mp4", "distractor_floorplan_P41.png"),             # [NOT ON DRIVE]
    ("S-14_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-14_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-14_Emergency.mp4", "distractor_floorplan_P48_flipped.png"),
    ("S-14_Stair.mp4", "distractor_floorplan_Stair.png"),
    ("S-14_Elevator.mp4", "distractor_floorplan_E.png"),
    ("S-14_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-14_Inaccessible.mp4", "distractor_floorplan_A10.png"),
    ("S-14_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-14_Reception.mp4", "distractor_floorplan_P41.png"),
    ("S-15_Corridor.mp4", "distractor_floorplan_A11.png"),
    ("S-15_Door.mp4", "distractor_floorplan_doors_P29.png"),            # [downloading]
    ("S-15_Emergency.mp4", "distractor_floorplan_P48.png"),
    ("S-15_Stair.mp4", "distractor_floorplan_Stair.png"),               # [downloading]
    ("S-15_Elevator.mp4", "distractor_floorplan_E.png"),
    ("S-15_Bathroom.mp4", "distractor_floorplan_P9.png"),
    ("S-15_Inaccessible.mp4", "distractor_floorplan_P42.png"),
    ("S-15_Dangerous.mp4", "distractor_floorplan_A10.png"),
    ("S-15_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-16_Corridor.mp4", "distractor_floorplan_A11.png"),
    ("S-16_Door.mp4", "distractor_floorplan_P99.png"),
    ("S-16_Emergency.mp4", "distractor_floorplan_gap.png"),
    ("S-16_Stair.mp4", "distractor_floorplan_Stair.png"),
    ("S-16_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-16_Bathroom.mp4", "distractor_floorplan_P26.png"),
    ("S-16_Inaccessible.mp4", "distractor_floorplan_P98.png"),
    ("S-16_Dangerous.mp4", "distractor_floorplan_P42.png"),
    ("S-16_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-17_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-17_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-17_Emergency.mp4", "distractor_floorplan_P29.png"),             # [downloading]
    ("S-17_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-17_Elevator.mp4", "distractor_floorplan_E.png"),
    ("S-17_Bathroom.mp4", "distractor_floorplan_B.png"),
    ("S-17_Inaccessible.mp4", "distractor_floorplan_P98.png"),
    ("S-17_Dangerous.mp4", "distractor_floorplan_A10.png"),
    ("S-17_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-18_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-18_Door.mp4", "distractor_floorplan_P6.png"),
    ("S-18_Emergency.mp4", "distractor_floorplan_P95.png"),
    ("S-18_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-18_Elevator.mp4", "distractor_floorplan_E.png"),
    ("S-18_Bathroom.mp4", "distractor_floorplan_B.png"),
    ("S-18_Inaccessible.mp4", "distractor_floorplan_P10.png"),
    ("S-18_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-18_Reception.mp4", "distractor_floorplan_R.png"),
    ("S-19_Corridor.mp4", "distractor_floorplan_A11.png"),
    ("S-19_Door.mp4", "distractor_floorplan_gap.png"),
    ("S-19_Emergency.mp4", "distractor_floorplan_P48_flipped.png"),
    ("S-19_Stair.mp4", "distractor_floorplan_P7.png"),
    ("S-19_Elevator.mp4", "distractor_floorplan_E.png"),
    ("S-19_Bathroom.mp4", "distractor_floorplan_B.png"),
    ("S-19_Inaccessible.mp4", "distractor_floorplan_P42.png"),
    ("S-19_Dangerous.mp4", "distractor_floorplan_A10.png"),
    ("S-19_Reception.mp4", "distractor_floorplan_P41.png"),             # [downloading]
    ("S-20_Corridor.mp4", "distractor_floorplan_A9.png"),
    ("S-20_Door.mp4", "distractor_floorplan_P95.png"),
    ("S-20_Emergency.mp4", "distractor_floorplan_P29.png"),             # [downloading]
    ("S-20_Stair.mp4", "distractor_floorplan_Stair.png"),
    ("S-20_Elevator.mp4", "distractor_floorplan_P54.png"),
    ("S-20_Bathroom.mp4", "distractor_floorplan_P34.png"),
    ("S-20_Inaccessible.mp4", "distractor_floorplan_P10.png"),
    ("S-20_Dangerous.mp4", "distractor_floorplan_P17.png"),
    ("S-20_Reception.mp4", "distractor_floorplan_R.png"),
]


def main():
    ran, skipped, failed = 0, [], []
    try:
        for i, (video_name, map_name) in enumerate(PAIRS, 1):
            video = os.path.join(VIDEO_DIR, video_name)
            map_path = os.path.join(MAP_DIR, map_name)
            missing = [p for p in (video, map_path) if not os.path.exists(os.path.join(HERE, p))]
            if missing:
                print(f"[{i}/{len(PAIRS)}] SKIP {video_name}: missing {', '.join(missing)}", flush=True)
                skipped.append(video_name)
                continue

            print(f"\n=== [{i}/{len(PAIRS)}] {video_name} + {map_name} ===", flush=True)
            cmd = [sys.executable, os.path.join(HERE, "run_pipeline.py"), video, map_path]
            result = subprocess.run(cmd, cwd=HERE)
            ran += 1
            if result.returncode != 0:
                failed.append(video_name)
            print(f"=== {video_name} exited with code {result.returncode} ===", flush=True)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")

    print(f"\nRan {ran}, skipped {len(skipped)}, failed {len(failed)}.")
    if skipped:
        print("Skipped: " + ", ".join(skipped))
    if failed:
        print("Failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
