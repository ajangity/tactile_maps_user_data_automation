"""Map pixels -> real-world inches.

The map PNG is the whole printed sheet, edge to edge, and every tactile map
is printed on US Letter paper (8.5 x 11 inches). So the PNG's longer side is
11 inches and its shorter side 8.5, whichever way round the PNG is, and any
distance in PNG pixels converts to inches with one scale per axis. (For the
816 x 1056 floorplan PNGs that is exactly 96 px per inch both ways.)

Every coordinate in this project is already in PNG pixels (finger tracking
maps fingertips through the crop into them), so this one conversion covers
rooms, symbols and finger paths alike.

    python units.py distractor_floorplan_E.png      # print each room's size
"""

import argparse

PAGE_INCHES = (8.5, 11.0)   # (short side, long side)


class PixelsToInches:
    def __init__(self, width_px, height_px, page_inches=PAGE_INCHES):
        short, long_ = page_inches
        self.width_in, self.height_in = (short, long_) if height_px >= width_px else (long_, short)
        self.x_scale = self.width_in / float(width_px)    # inches per pixel, across
        self.y_scale = self.height_in / float(height_px)  # inches per pixel, down

    def point(self, x, y):
        """A PNG pixel position -> inches from the sheet's top-left corner."""
        return x * self.x_scale, y * self.y_scale

    def size(self, w_px, h_px, digits=2):
        """A (width, height) in PNG pixels -> (width, height) in inches."""
        return round(w_px * self.x_scale, digits), round(h_px * self.y_scale, digits)

    def bbox_size(self, bbox, digits=2):
        """(width, height) in inches of an (x0, y0, x1, y1) box, x1/y1 exclusive
        as cv2.boundingRect gives them (room_tracking's room boxes)."""
        x0, y0, x1, y1 = bbox
        return self.size(x1 - x0, y1 - y0, digits)

    def to_json(self):
        return {"page_in": [self.width_in, self.height_in],
                "px_per_inch": [round(1 / self.x_scale, 3), round(1 / self.y_scale, 3)]}


def main():
    import json
    import os
    import cv2
    parser = argparse.ArgumentParser(description="Print each room's size in inches.")
    parser.add_argument("map", help="the map PNG (its map.json is read from a run folder or "
                                    "next to the PNG)")
    parser.add_argument("--json", help="map.json to read rooms from (default: <map>.map.json)")
    args = parser.parse_args()
    h, w = cv2.imread(args.map).shape[:2]
    inches = PixelsToInches(w, h)
    print(f"{w} x {h} px = {inches.width_in} x {inches.height_in} in "
          f"({1 / inches.x_scale:.1f} x {1 / inches.y_scale:.1f} px per inch)")
    path = args.json or os.path.splitext(args.map)[0] + ".map.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for room in json.load(f)["rooms"]:
                print(f"  {room['name']:<10} {inches.bbox_size(room['bbox'])}")


if __name__ == "__main__":
    main()
