"""Temporary helper: rerun run_pipeline.py on the same inputs until Ctrl+C.

Usage:
    python loop_pipeline.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = "S-19_Elevator.mp4"
MAP = "distractor_floorplan_E.png"


def main():
    cmd = [sys.executable, os.path.join(HERE, "run_pipeline.py"), VIDEO, MAP]
    run_num = 0
    try:
        while True:
            run_num += 1
            print(f"\n=== Run {run_num} ===", flush=True)
            result = subprocess.run(cmd, cwd=HERE)
            print(f"=== Run {run_num} exited with code {result.returncode} ===", flush=True)
    except KeyboardInterrupt:
        print(f"\nStopped after {run_num} run(s).")


if __name__ == "__main__":
    main()
