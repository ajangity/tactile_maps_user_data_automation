import json
import sys
import cv2

if len(sys.argv) < 2:
    print("usage: python3 label_symbols.py <reference_map.png>")
    sys.exit(1)

image_path = sys.argv[1]
out_path = image_path.rsplit(".", 1)[0] + ".symbols.json"

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(image_path)
img = cv2.rotate(img, cv2.ROTATE_180)  # matches how the tracker loads it
h, w = img.shape[:2]

boxes = {}
current_name = None
clicks = []


def redraw():
    shown = img.copy()
    for name, (fx0, fy0, fx1, fy1) in boxes.items():
        p0 = (int(fx0 * w), int(fy0 * h))
        p1 = (int(fx1 * w), int(fy1 * h))
        cv2.rectangle(shown, p0, p1, (0, 255, 0), 2)
        cv2.putText(shown, name, (p0[0], p0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    prompt = f"labeling: {current_name}" if current_name else "type a symbol name in the terminal"
    cv2.putText(shown, prompt, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("label_symbols", shown)


def on_click(event, x, y, flags, param):
    global clicks
    if event != cv2.EVENT_LBUTTONDOWN or current_name is None:
        return
    clicks.append((x, y))
    print(f"  clicked ({x}, {y})")
    if len(clicks) == 2:
        (x0, y0), (x1, y1) = clicks
        boxes[current_name] = (min(x0, x1) / w, min(y0, y1) / h,
                               max(x0, x1) / w, max(y0, y1) / h)
        print(f"  saved box for '{current_name}'")
        clicks = []
    redraw()


cv2.namedWindow("label_symbols", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("label_symbols", on_click)
redraw()

print("For each symbol: type its name and press Enter, then click its two opposite")
print("corners in the image window. Type 'done' when finished.")

while True:
    redraw()
    key = cv2.waitKey(20) & 0xFF
    if key == 13 or key == 10:  # Enter in the terminal doesn't reach cv2, this
        pass                    # loop only exists to keep the window responsive
    name = input("symbol name (or 'done'): ").strip()
    if name.lower() == "done":
        break
    if not name:
        continue
    current_name = name
    clicks = []
    redraw()
    print(f"click the two opposite corners of '{name}' in the image window")
    while len(clicks) < 2:
        cv2.waitKey(20)
    current_name = None

cv2.destroyAllWindows()

with open(out_path, "w") as f:
    json.dump(boxes, f, indent=2)
print(f"\nSaved {len(boxes)} symbol(s) to {out_path}")
