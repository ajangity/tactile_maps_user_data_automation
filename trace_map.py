"""Trace wall lines and locate symbol blobs on a flattened map image.

Works on any top-down, hand-free view of the map: the clean reference PNG
directly, or a video frame already warped into map space via the tracker's
own homography (see finger_tracking_manual_objects.py). Same pipeline
either way -- only the input image changes.
"""
import sys

import cv2
import numpy as np


def ink_mask(gray, thresh=180):
    """Binary mask of printed black ink vs white paper."""
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return mask


def trace_walls(mask, min_len=25):
    """Straight wall/corridor segments as (x1, y1, x2, y2) tuples."""
    edges = cv2.Canny(mask, 50, 150)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                               minLineLength=min_len, maxLineGap=6)
    if segments is None:
        return []
    return [tuple(s) for s in segments.reshape(-1, 4)]


def find_symbol_blobs(mask, wall_mask, min_area=150, max_area=8000):
    """Small ink blobs that aren't part of the traced wall lines --
    candidate symbols. Subtracting the wall mask first keeps a room's own
    border from being picked up as a "symbol" sitting inside it."""
    interior = cv2.bitwise_and(mask, cv2.bitwise_not(wall_mask))
    interior = cv2.dilate(interior, np.ones((3, 3), np.uint8))
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        interior, connectivity=8)
    blobs = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            blobs.append((x, y, w, h, tuple(centroids[i])))
    return blobs


def classify_shape(mask, box):
    """Cheap rule-based shape label for a symbol blob: filled vs outline,
    round vs angular, roughly how many straight sides -- no ML model,
    just contour geometry, matches how the rest of this codebase works."""
    x, y, w, h, _ = box
    crop = mask[y:y + h, x:x + w]
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "?"
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    box_area = max(w * h, 1)
    fill_ratio = area / box_area
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0

    if fill_ratio > 0.75 and circularity > 0.7:
        return "circle"
    if circularity > 0.7:
        return "ring"
    if fill_ratio > 0.75:
        return "filled"
    if len(approx) <= 5:
        return f"{len(approx)}-gon"
    return "text"


def main():
    if len(sys.argv) < 2:
        print("usage: python3 trace_map.py <image.png> [output.png]")
        sys.exit(1)
    image_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "traced_" + image_path.split("/")[-1]

    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(image_path)
    if raw.shape[2] == 4:
        bgr, a = raw[:, :, :3].astype(np.float32), raw[:, :, 3:4].astype(np.float32) / 255.0
        img = (bgr * a + 255.0 * (1 - a)).astype(np.uint8)
    else:
        img = raw
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = ink_mask(gray)

    walls = trace_walls(mask)
    wall_mask = np.zeros_like(mask)
    for x1, y1, x2, y2 in walls:
        cv2.line(wall_mask, (x1, y1), (x2, y2), 255, 3)

    blobs = find_symbol_blobs(mask, wall_mask)

    shown = img.copy()
    for x1, y1, x2, y2 in walls:
        cv2.line(shown, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for i, box in enumerate(blobs):
        x, y, w, h, (cx, cy) = box
        label = classify_shape(mask, box)
        cv2.rectangle(shown, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(shown, f"{i}:{label}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, shown)
    print(f"traced {len(walls)} wall segments, {len(blobs)} candidate symbol blobs")
    for i, box in enumerate(blobs):
        x, y, w, h, (cx, cy) = box
        print(f"  blob {i}: center=({cx:.0f},{cy:.0f}) size={w}x{h} "
              f"shape='{classify_shape(mask, box)}'")
    print(f"saved visualization to {out_path}")


if __name__ == "__main__":
    main()