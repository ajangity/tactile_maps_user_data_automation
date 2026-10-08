"""Optional extra boxes, drawn by hand on the reference map PNG.

Rooms and symbols are now found automatically (room_tracking.py, saved in
<map>.map.json), so this is only needed for an extra region the automatic
detection doesn't cover. Each box drawn here gets its own timer too.

Click boxes on the PNG and mark
each one "symbol" (a legend icon) or "room" (a space the user's finger
moves through). Press 's' or 'r', then click the box's two opposite
corners. You can then type a name for it right in the image window
("Elevator", "Stairs", ...) and press Enter -- or just press Enter (or Esc)
to keep the auto-numbered name ("Symbol 1", "Room 1", ...). Press 'u' to
undo, 'd' when done. Saves <map>.symbols.json, with each box both as
fractions of the image (what the tracker loads) and in PNG pixel coords.

Everything happens inside the image window via keyboard, with no blocking
terminal input() call -- that was what made the window look "Not
Responding" on macOS before: a GUI window whose event loop stops being
pumped while Python waits on terminal input gets flagged unresponsive by
the OS even though it isn't actually stuck. cv2.waitKey() alone keeps it
pumped continuously.
"""
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
h, w = img.shape[:2]

boxes = {}             # name -> {"type":, "box":}
order = []              # insertion order, so undo removes the right one
counts = {"symbol": 0, "room": 0}
pending_type = None     # "symbol" or "room" while mid-click, else None
clicks = []
naming = None           # name of the just-drawn box while it's being renamed
name_buffer = ""

TYPE_COLORS = {"symbol": (0, 255, 0), "room": (255, 180, 0)}


def redraw():
    shown = img.copy()
    for name, entry in boxes.items():
        fx0, fy0, fx1, fy1 = entry["box"]
        color = TYPE_COLORS[entry["type"]]
        p0 = (int(fx0 * w), int(fy0 * h))
        p1 = (int(fx1 * w), int(fy1 * h))
        cv2.rectangle(shown, p0, p1, color, 2)
        cv2.putText(shown, name, (p0[0], p0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if naming:
        prompt = (f"name for {naming}: {name_buffer}_   "
                  "(Enter = save, Esc = keep auto name)")
    elif pending_type:
        prompt = (f"click 2 opposite corners for the next {pending_type} "
                 f"({len(clicks)}/2 so far)")
    else:
        prompt = "s = new symbol box   r = new room box   u = undo   d = done"
    cv2.putText(shown, prompt, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(shown, f"{counts['symbol']} symbol(s), {counts['room']} room(s) so far",
                (18, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("label_symbols", shown)


def finish_box():
    global pending_type, clicks, naming, name_buffer
    (x0, y0), (x1, y1) = clicks
    counts[pending_type] += 1
    name = f"{pending_type.capitalize()} {counts[pending_type]}"
    while name in boxes:   # a custom name may already have taken it
        name += "+"
    px = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    boxes[name] = {
        "type": pending_type,
        "box": (px[0] / w, px[1] / h, px[2] / w, px[3] / h),
        # same box in pixels of the rotated PNG shown in this window
        "box_px": px,
        "space": "png",   # PNG's own orientation (older files were rotated 180)
    }
    order.append(name)
    pending_type = None
    clicks = []
    naming, name_buffer = name, ""


def finish_naming(keep_typed):
    """Rename the just-drawn box to whatever was typed (if anything)."""
    global naming, name_buffer
    new = name_buffer.strip()
    if keep_typed and new and new != naming:
        if new in boxes:
            print(f"  '{new}' is already used -- keeping {naming}")
        else:
            boxes[new] = boxes.pop(naming)
            order[order.index(naming)] = new
            naming = new
    print(f"  saved {naming}")
    naming, name_buffer = None, ""


def undo():
    global pending_type, clicks
    if pending_type is not None:
        # cancel an in-progress box rather than undo an already-finished one
        pending_type = None
        clicks = []
        print("  cancelled in-progress box")
        return
    if not order:
        return
    name = order.pop()
    entry = boxes.pop(name)
    counts[entry["type"]] -= 1
    print(f"  undid {name}")


def on_click(event, x, y, flags, param):
    global clicks
    if event != cv2.EVENT_LBUTTONDOWN or pending_type is None or naming:
        return
    clicks.append((x, y))
    if len(clicks) == 2:
        finish_box()


cv2.namedWindow("label_symbols", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("label_symbols", on_click)

print("s = start a symbol box, r = start a room box, then click its two opposite")
print("corners in the image window. Then type a name in the window and press")
print("Enter (or just Enter/Esc to keep the auto number). u = undo, d = done.")

while True:
    redraw()
    key = cv2.waitKey(20) & 0xFF
    if naming:
        # While naming, every key is text, so s/r/u/d don't trigger commands.
        if key in (10, 13):
            finish_naming(keep_typed=True)
        elif key == 27:
            finish_naming(keep_typed=False)
        elif key in (8, 127):
            name_buffer = name_buffer[:-1]
        elif 32 <= key < 127:
            name_buffer += chr(key)
        continue
    if key == ord("s") and pending_type is None:
        pending_type = "symbol"
        clicks = []
    elif key == ord("r") and pending_type is None:
        pending_type = "room"
        clicks = []
    elif key == ord("u"):
        undo()
    elif key == ord("d"):
        break

cv2.destroyAllWindows()

with open(out_path, "w") as f:
    json.dump(boxes, f, indent=2)
print(f"\nSaved {len(boxes)} box(es) to {out_path} "
      f"({counts['symbol']} symbol(s), {counts['room']} room(s))")
