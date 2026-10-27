"""This file learns what the paper looks like in each video, from that
video's own frames, so nothing has to be tuned per participant. Anything
on top of the page that doesn't look like this paper -- a hand of any skin
tone, a sleeve, a watch, a ring, a bottle -- is an occluder.

A crop that scores well is a free labeled example: the pixels inside it
that aren't printed ink are (mostly) paper. Their color is summarized
robustly (brightest half, then median and spread, so hands covering part
of the page don't skew it) and blended into a running model, so it follows
the lighting as it drifts. Before the map has been found there's no model
yet, so each candidate crop gets a throwaway one built from its own pixels
(PaperModel.from_crop).

Paper is told apart by color first: in LAB, paper sits at almost the same
a* (red-green) value everywhere on the page, while skin -- even very pale
skin, measured a* 138-142 against paper's 126-128 -- is clearly redder.
Dark sleeves and shadows are caught by brightness instead.

What the model is used for (see auto_crop.py):
  visible()   -- which parts of the crop aren't covered right now.
                 alignment_score only expects printed lines where the map is
                 visible, so a hand covering most of the page doesn't drag a
                 correct crop's score down.
  clean_ink() -- drops traced "ink" on or right next to an occluder:
                 knuckle creases, fingernails, a hand's outline and the thin
                 shadow along it, watch straps and sleeve folds aren't
                 printed lines, and they used to count against a crop.
"""

import cv2
import numpy as np

LEARN_EVERY = 15          # frames between model updates (while confirmed)
LEARN_MIN_SCORE = 0.60    # only crops at least this good teach the model
BLEND = 0.25              # weight of each new sample in the running model
SAMPLE_STEP = 3           # learn from every 3rd pixel each way (enough, and fast)
MIN_SAMPLES = 400
CHROMA_SIGMAS = 3.5       # a pixel whose color is further than this from paper isn't paper
CHROMA_FLOOR = 1.5        # LAB a/b units; the spread is never treated as tighter than this
LIGHT_SIGMAS = 5.0        # how much darker than the paper's median a lit pixel may be
LIGHT_FLOOR = 0.70        # ...and in any case no darker than 70% of the paper's L
STROKE_MAX = 11           # px; non-paper marks thinner than this are printed lines, not occluders
OCCLUDER_MARGIN = 8       # px around an occluder where traced ink is its outline / shadow


class PaperModel:
    def __init__(self):
        self.ready = False
        self._since = LEARN_EVERY

    @classmethod
    def from_crop(cls, frame, corners, frame_ink):
        """A one-off model from a single crop (before the first lock)."""
        model = cls()
        model._learn(frame, corners, frame_ink)
        return model

    @classmethod
    def from_frame(cls, frame, rough_paper):
        """A one-off model from a whole frame (before the first lock): every
        sheet on the table is the same paper, so the brightest of the pixels
        a rough paper mask (paper_locator.paper_mask) accepts -- which can
        include pale skin -- are the paper's color."""
        model = cls()
        lab = cv2.cvtColor(frame[::SAMPLE_STEP, ::SAMPLE_STEP], cv2.COLOR_BGR2LAB).astype(np.float32)
        model._fit(lab[rough_paper[::SAMPLE_STEP, ::SAMPLE_STEP] > 0])
        return model

    # ------------------------------------------------------------ learning
    def maybe_learn(self, frame, corners, frame_ink, score):
        """Called after every confirmed frame; learns every LEARN_EVERY
        frames from crops that score at least LEARN_MIN_SCORE."""
        self._since += 1
        if score >= LEARN_MIN_SCORE and self._since >= LEARN_EVERY:
            self._since = 0
            self._learn(frame, corners, frame_ink)

    def _learn(self, frame, corners, frame_ink):
        box, inside = _region(frame.shape, corners, 0.92)
        if box is None:
            return
        x0, y0, x1, y1 = box
        lab = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
        ink = cv2.dilate(frame_ink[y0:y1, x0:x1], np.ones((5, 5), np.uint8))
        take = (inside > 0) & (ink == 0)
        self._fit(lab[::SAMPLE_STEP, ::SAMPLE_STEP][take[::SAMPLE_STEP, ::SAMPLE_STEP]])

    def _fit(self, px):
        """Blend the paper color of these LAB pixels into the model."""
        if len(px) < MIN_SAMPLES:
            return
        # Paper is the brightest large part of the crop: hands, sleeves and
        # shadows are darker. Keep the brighter half before measuring color,
        # then trim anything still far from the median.
        px = px[px[:, 0] >= np.median(px[:, 0])]
        med = np.median(px, axis=0)
        mad = 1.4826 * np.median(np.abs(px - med), axis=0) + 1e-3
        px = px[np.all(np.abs(px - med) <= 3 * mad, axis=1)]
        if len(px) < MIN_SAMPLES // 4:
            return
        light = float(np.median(px[:, 0]))
        light_sd = float(1.4826 * np.median(np.abs(px[:, 0] - light))) + 1.0
        mean = px[:, 1:].mean(axis=0)
        cov = np.cov(px[:, 1:].T) + np.eye(2) * CHROMA_FLOOR ** 2
        if not self.ready:
            self.light, self.light_sd, self.mean, self.cov = light, light_sd, mean, cov
        else:
            b = BLEND
            self.light = (1 - b) * self.light + b * light
            self.light_sd = (1 - b) * self.light_sd + b * light_sd
            self.mean = (1 - b) * self.mean + b * mean
            self.cov = (1 - b) * self.cov + b * cov
        self.cov_inv = np.linalg.inv(self.cov)
        self.ready = True

    # ------------------------------------------------------------ using it
    def paper_pixels(self, bgr):
        """Mask (0/255) of pixels in this image that look like the paper."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        da, db = lab[:, :, 1] - self.mean[0], lab[:, :, 2] - self.mean[1]
        ci = self.cov_inv
        chroma = ci[0, 0] * da * da + 2 * ci[0, 1] * da * db + ci[1, 1] * db * db
        floor = min(self.light - LIGHT_SIGMAS * self.light_sd, LIGHT_FLOOR * self.light)
        return ((chroma <= CHROMA_SIGMAS ** 2) & (lab[:, :, 0] >= floor)).astype(np.uint8) * 255

    def sheet_mask(self, bgr):
        """paper_pixels cleaned up into whole sheets, the same way as
        paper_locator.paper_mask: specks removed, printed lines filled in.
        Unlike that brightness + saturation mask, pale skin isn't paper
        here, so a hand can't merge the page with the arm or another sheet."""
        mask = self.paper_pixels(bgr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    def visible(self, frame, corners, scale=1.0):
        """Frame-sized mask, 0 where something covers the paper (and in a
        margin of OCCLUDER_MARGIN px around it), 255 everywhere else.
        Occluders are only looked for inside the crop itself -- the desk
        around the paper isn't paper either, but it isn't covering it --
        and their margin may reach out into the crop grown by `scale`.

        Occluders are the sizable non-paper areas: an opening with a
        STROKE_MAX disc removes thin printed strokes, which aren't
        paper-colored either but belong to the map. The margin is there
        because a hand's outline and the thin shadow it casts trace as long
        dark "walls" right beside it. None if the model isn't ready."""
        if not self.ready:
            return None
        box, _ = _region(frame.shape, corners, scale)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        inside = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(inside, np.int32(np.rint(np.asarray(corners, np.float32) - [x0, y0])), 255)
        # Occluders are big, so half resolution is plenty (and 4x cheaper).
        small = cv2.resize(frame[y0:y1, x0:x1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        paper = self.paper_pixels(small)
        disc = lambda d: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        occluder = cv2.morphologyEx(cv2.bitwise_not(paper), cv2.MORPH_OPEN, disc(STROKE_MAX // 2 + 1))
        occluder = cv2.resize(occluder, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        occluder = cv2.dilate(occluder & inside, disc(2 * OCCLUDER_MARGIN + 1))
        out = np.full(frame.shape[:2], 255, np.uint8)
        out[y0:y1, x0:x1] = cv2.bitwise_not(occluder)
        return out


def clean_ink(frame_ink, visible):
    """frame_ink with marks on or right next to an occluder removed."""
    return frame_ink if visible is None else cv2.bitwise_and(frame_ink, visible)


def _region(shape, corners, scale):
    """Bounding box of the crop scaled about its centre, and the crop's
    filled polygon (0/255) inside that box."""
    h, w = shape[:2]
    q = np.asarray(corners, np.float32)
    c = q.mean(axis=0)
    q = c + scale * (q - c)
    x0, y0 = np.maximum(np.floor(q.min(axis=0)), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(q.max(axis=0)) + 1, [w, h]).astype(int)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None, None
    inside = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(inside, np.int32(np.rint(q - [x0, y0])), 255)
    return (x0, y0, x1, y1), inside
