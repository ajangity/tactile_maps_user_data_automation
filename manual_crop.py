"""Manual crop: stop the video and drag the crop's corners by hand.

Not part of the automatic pipeline. It only runs when you press Space
while the video is playing: the video freezes, the PNG is overlaid on the
current crop, and you drag the 4 corners until it lines up. Space or
Enter accepts (tracking continues from your corners), Esc cancels
(tracking continues from where it was), q quits.

Also holds VideoWindow, the resizable window the video plays in, which
keeps track of how the frame was scaled into it so a click in the window
maps back to the right frame pixel.
"""

import math

import cv2
import numpy as np


class VideoWindow:
    def __init__(self, name, frame_w, frame_h):
        self.name = name
        self.scale, self.offset = 1.0, (0, 0)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, frame_w, frame_h)

    def show(self, image):
        """Show image letterboxed to the window's current size (resizing the
        window scales the frame, never crops it)."""
        h, w = image.shape[:2]
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(self.name)
        except cv2.error:
            win_w = win_h = 0
        if win_w < 2 or win_h < 2:
            self.scale, self.offset = 1.0, (0, 0)
            cv2.imshow(self.name, image)
            return
        scale = min(win_w / w, win_h / h)
        dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        canvas = np.zeros((win_h, win_w, 3), np.uint8)
        ox, oy = (win_w - dw) // 2, (win_h - dh) // 2
        canvas[oy:oy + dh, ox:ox + dw] = cv2.resize(
            image, (dw, dh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        self.scale, self.offset = scale, (ox, oy)
        cv2.imshow(self.name, canvas)

    def to_frame(self, x, y):
        return (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale


def overlay_map(frame, corners, ref_image, alpha=0.45):
    """The PNG warped onto the crop and blended over the frame."""
    h, w = frame.shape[:2]
    rh, rw = ref_image.shape[:2]
    src = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]])
    H = cv2.getPerspectiveTransform(src, np.float32(corners))
    warped = cv2.warpPerspective(ref_image, H, (w, h))
    inside = cv2.warpPerspective(np.full((rh, rw), 255, np.uint8), H, (w, h)) > 0
    out = frame.copy()
    out[inside] = cv2.addWeighted(frame, 1 - alpha, warped, alpha, 0)[inside]
    return out


def adjust_corners(window, frame, corners, ref_image):
    """Let the user drag the 4 corners. Returns (corners, action) where
    action is "accept", "cancel" or "quit"."""
    h, w = frame.shape[:2]
    if corners is None:
        corners = [[w * 0.3, h * 0.2], [w * 0.7, h * 0.2], [w * 0.7, h * 0.8], [w * 0.3, h * 0.8]]
    pts = [list(map(float, p)) for p in corners]
    state = {"active": -1}

    def on_mouse(event, x, y, flags, _):
        fx, fy = window.to_frame(x, y)
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and
                                              flags & cv2.EVENT_FLAG_LBUTTON):
            if state["active"] == -1:   # grab whichever corner is nearest the click
                state["active"] = int(np.argmin([math.hypot(fx - p[0], fy - p[1]) for p in pts]))
            pts[state["active"]] = [fx, fy]
        elif event == cv2.EVENT_LBUTTONUP:
            state["active"] = -1

    cv2.setMouseCallback(window.name, on_mouse)
    labels = ("TL", "TR", "BR", "BL")
    try:
        while True:
            shown = overlay_map(frame, pts, ref_image)
            cv2.polylines(shown, [np.int32(pts)], True, (0, 255, 0), 2, cv2.LINE_AA)
            for p, name in zip(pts, labels):
                cv2.circle(shown, (int(p[0]), int(p[1])), 8, (0, 255, 0), -1)
                cv2.putText(shown, name, (int(p[0]) + 10, int(p[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(shown, "STOPPED: drag the corners (TL = the map's top-left). "
                        "Space/Enter = use these, Esc = cancel, q = quit",
                        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            window.show(shown)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord(" "), 10, 13):
                return np.float32(pts), "accept"
            if key == 27:
                return None, "cancel"
            if key == ord("q"):
                return None, "quit"
    finally:
        cv2.setMouseCallback(window.name, lambda *a: None)
