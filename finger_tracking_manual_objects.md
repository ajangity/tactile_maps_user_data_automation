# finger_tracking_manual_objects.py

## What this script does

`finger_tracking_manual_objects.py` tracks index-finger movement over a video of a printed floorplan or map, then saves the traced paths onto the original map image.

The script combines:

- A manual first-frame alignment step, where you drag the four green map corners onto the video frame.
- Automatic page tracking, using OpenCV feature matching and optical flow to keep the map aligned as the camera or paper moves.
- MediaPipe hand tracking, using the index-finger tip landmark from up to two detected hands.
- Perspective mapping, which converts fingertip locations from video-frame coordinates back into map-image coordinates.
- Final path rendering, which draws left-hand and right-hand traces onto the map and saves them as a PNG.

By default, it uses:

- Video: `S-18 5-12-26 PT2 E.mp4`
- Map image: `distractor_floorplan_E.png`
- Output image: `finger_paths.png`

## How it works

1. The reference map image is loaded and rotated 180 degrees for the live overlay.
2. The first video frame opens in an OpenCV window named `Video Tracker`.
3. Four green handles appear near the corners of the frame.
4. You click or drag the handles until the overlaid map lines up with the physical map in the video.
5. After you press `Enter`, the script starts processing the video.
6. For each frame, the script estimates the current map position:
   - It first tries to match visual features from the reference map to the video frame.
   - If feature matching is not strong enough, it falls back to optical flow from the previous frame.
7. The script crops around the map area and runs MediaPipe hand landmark detection.
8. For each detected hand, it reads landmark `8`, the index-finger tip.
9. The fingertip position is transformed from video coordinates into reference-map coordinates.
10. The mapped fingertip samples are stored as paths, with breaks inserted when a hand is not detected.
11. When the video ends, or when you quit, the script draws the traced paths onto the original map orientation and saves the output image.

## Controls

During initial alignment:

- Mouse click or drag: move the nearest green corner handle.
- `Enter`: accept the alignment and start tracking.
- `q`: quit without producing a tracked output.

During tracking:

- `Space`: pause or resume.
- Mouse click or drag while paused: manually correct a map corner.
- `Space` after correcting corners: resume and reset the page tracker from the corrected alignment.
- `]`, `=`, or `+`: increase playback display speed.
- `[`, `-`, or `_`: decrease playback display speed.
- `q`: stop early, save the output image, and quit.

## Dependencies

The script requires Python and these Python packages:

- `opencv-python` or `opencv-contrib-python`
- `mediapipe`
- `numpy`

Use a Python version supported by MediaPipe. If `pip install mediapipe` fails for your current Python version, create the virtual environment with an earlier supported Python release and reinstall the dependencies there.

It also requires these local input files:

- `hand_landmarker.task`, the MediaPipe hand landmark model file.
- A video file, such as `S-18 5-12-26 PT2 E.mp4`.
- A reference map image, such as `distractor_floorplan_E.png`.

OpenCV must be able to open a desktop GUI window because the script uses `cv2.imshow`, mouse callbacks, and keyboard input. Run it from a local desktop session rather than a headless terminal.

## Install dependencies

From this folder, create and activate a virtual environment if desired:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the required packages:

```bash
python3 -m pip install opencv-python mediapipe numpy
```

If you need OpenCV's fuller contributed feature set, install `opencv-contrib-python` instead of `opencv-python`:

```bash
python3 -m pip install opencv-contrib-python mediapipe numpy
```

## How to run

Run with the default video, default map, and default output filename:

```bash
python3 finger_tracking_manual_objects.py
```

Run with a specific video and map:

```bash
python3 finger_tracking_manual_objects.py "S-18 5-12-26 PT2 E.mp4" distractor_floorplan_E.png
```

Run with a custom output path:

```bash
python3 finger_tracking_manual_objects.py "S-18 5-12-26 PT2 E.mp4" distractor_floorplan_E.png -o S-18_manual_path.png
```

## Output

The output is a PNG image containing the original map plus the detected finger traces.

Default output:

```text
finger_paths.png
```

The script draws:

- Left hand path in red.
- Right hand path in blue.

The color names are based on the script's MediaPipe handedness labels. MediaPipe handedness can sometimes flip during tracking, so the script also keeps position-based internal tracks to reduce duplicate or disappearing markers during live processing.

## Notes and troubleshooting

- If the script cannot find `hand_landmarker.task`, make sure the model file is in the same folder where you run the script.
- If you see `ModuleNotFoundError: No module named 'mediapipe'`, install the dependencies in the active Python environment.
- If the video or map path contains spaces, wrap the path in quotes.
- If the OpenCV window does not appear, confirm that you are running in a desktop environment with GUI support.
- If tracking drifts, press `Space`, drag the green corners back into place, then press `Space` again to continue.
- If only one hand is detected, improve lighting, reduce motion blur, or slow the playback display speed with `[` or `-`.
- The script processes every video frame. Playback speed only changes how quickly frames are displayed while processing.
