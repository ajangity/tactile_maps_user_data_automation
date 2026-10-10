"""Auto-crop: runs the crop pipeline once per frame, in order.

This file holds no image-processing logic of its own -- it's the running
order, calling into:
    edge_tracing.py    trace the printed lines (independent of the crop)
    paper_locator.py   steps 1-8
    corner_fitting.py  steps 9-17
    crop_checker.py    step 14 checks, the two backups, step 21

Before the map has been found (the video can start with the sheet
face-down, or off-camera), it searches the whole frame every
SEARCH_EVERY frames until a crop scores LOCK_SCORE. After that, each frame:
    trace lines -> candidates (traced-line refinement of last frame's crop,
    plus each paper blob's fitted corners) -> sanity checks -> score ->
    best one wins -> otherwise PNG backup -> optical-flow backup -> hold.
If nothing has confirmed the crop for MAX_UNCONFIRMED frames, the map has
left the frame (or been turned over): the crop is dropped and it goes back
to exactly the start-of-video state -- no crop, whole-frame search every
SEARCH_EVERY frames until the map is found again.
"""

import numpy as np
import cv2

import edge_tracing
import paper_locator
from paper_locator import (alignment_score, candidate_blobs, homography_for, ink_border_mask,
                           orientations, paper_blob_quads, search_window, traced_map_quads)
from corner_fitting import (corners_from_homography, fit_quad, keep_if_stationary, pick_best,
                            refine_with_traced_lines, shift_candidates, smooth, snap_corners)
from crop_checker import (MAX_JUMP, OUTLINE_MIN, FlowBackup, PngBackup, check_quad,
                          corner_visible, correct_outlier_corner, outline_agreement)

LOCK_SCORE = 0.60        # first lock / recovery: a crop must score this
TRACK_SCORE = 0.40       # per frame: a candidate must score this (orientation is already known)
STRONG_SCORE = 0.60      # a traced-line crop this good skips the paper-edge candidates
SEARCH_EVERY = 10        # frames between whole-frame searches before the first lock
MAX_UNCONFIRMED = 20     # frames of backup-only tracking before the map counts as gone
REALIGN_EVERY = 3        # frames between shift searches while the crop is unconfirmed
COLD_START_REFINE = 3    # best N pre-scored candidates get the traced-line refinement


def grown(corners, margin=None):
    """The crop grown about its centre by the search margin (step 6) -- the
    region edge tracing is limited to once the map is locked."""
    p = np.asarray(corners, np.float32)
    c = p.mean(axis=0)
    return c + (1.0 + (paper_locator.SEARCH_MARGIN if margin is None else margin)) * (p - c)


class AutoCrop:
    def __init__(self, ref):
        self.ref = ref
        self.png = PngBackup(ref)
        self.flow = FlowBackup()
        self.corners = None          # 4 frame points (map TL, TR, BR, BL), None until found
        self.source = "searching"
        self.score = 0.0
        self.outline = 0.0
        self.detail = {}
        self.corner_visible = [False] * 4
        self.frames_since_confirmed = 0
        self.lost = False
        self.lock_info = None        # filled the first time the map is found
        self._tick = 0

    # ------------------------------------------------------------ helpers
    def _trace(self, frame, region=None):
        """Edge-trace this frame. Once the map is locked, region is last
        frame's crop grown by the search margin (the paper can move a little
        between frames), so only the paper's own pixels are processed. With
        no region -- before the first lock, or when searching again after
        losing it -- the whole frame is traced, since that's how the paper
        is found in the first place."""
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_ink = edge_tracing.ink_mask(self.gray, region=region)
        self.traced_region = region

    def score_quad(self, quad):
        return alignment_score(self.ref, self.frame_ink, homography_for(quad, self.ref))["score"]

    def judge(self, quad, H=None):
        """(score, detail, outline) for a crop. A crop whose edges don't sit
        on the paper's real edges gets score 0, however well its lines match
        (see crop_checker.outline_agreement)."""
        outline = outline_agreement(self.gray, quad)
        if outline < OUTLINE_MIN:
            return 0.0, {"score": 0.0, "rejected": "edges not on the paper's edges"}, outline
        detail = alignment_score(self.ref, self.frame_ink,
                                 H if H is not None else homography_for(quad, self.ref))
        return detail["score"], detail, outline

    def checked_score(self, quad):
        return self.judge(quad)[0]

    @property
    def locked(self):
        return self.corners is not None

    # ------------------------------------------------------------ steps 2-4: whole-frame search
    def locate(self, frame):
        """Search the whole frame for the map. Returns (quad, score, detail)
        or None."""
        if not hasattr(self, "frame_ink"):
            self._trace(frame)
        shape = frame.shape
        candidates = []
        for rough, blob, (x0, y0, x1, y1) in paper_blob_quads(frame):       # step 3
            quad, _ = fit_quad(blob[y0:y1, x0:x1], x0, y0, rough)          # step 12
            if quad is None:   # ink-border fallback, same padded region
                quad, _ = fit_quad(ink_border_mask(frame[y0:y1, x0:x1]), x0, y0, rough)
            candidates.extend(orientations(quad if quad is not None else rough))  # step 4
        candidates.extend(traced_map_quads(self.frame_ink, self.ref))      # step 3 (traced)
        candidates = [q for q in candidates
                      if check_quad(q, shape, self.ref.aspect)[0]]         # step 14
        if not candidates:
            return None
        pre = sorted(((self.score_quad(q), q) for q in candidates), key=lambda s: -s[0])
        results = []
        for _, q in pre[:COLD_START_REFINE]:                               # step 13
            H = refine_with_traced_lines(self.ref, self.frame_ink, homography_for(q, self.ref), q)
            q2 = corners_from_homography(H, self.ref)
            if not check_quad(q2, shape, self.ref.aspect)[0]:
                q2, H = q, homography_for(q, self.ref)
            score, detail, outline = self.judge(q2, H)
            detail["outline"] = round(outline, 3)
            results.append((score, q2, detail))
        best = max(results, key=lambda r: r[0])
        if best[0] < LOCK_SCORE:
            return None
        return best

    def _lock(self, frame, quad, score, detail, frame_index, source):
        self.corners = np.asarray(quad, np.float32)
        self.score, self.detail, self.source = score, detail, source
        self._confirmed()
        self.flow.remember(frame, self.gray, self.corners)
        if self.lock_info is None:
            self.lock_info = self._frame_lines_json(frame_index)

    def _frame_lines_json(self, frame_index):
        """Edge tracing's graph of the frame the map was first found in, in
        frame pixels and also carried into PNG pixels through the crop --
        directly comparable with the PNG's own map JSON."""
        trace = edge_tracing.Trace(self.gray, wall_min=40, region=self.corners)
        H = homography_for(self.corners, self.ref)
        nodes = trace.graph["nodes"]
        if nodes:
            pts = np.float32([[n["x"], n["y"]] for n in nodes])[None]
            mapped = cv2.perspectiveTransform(pts, H)[0]
            for n, (mx, my) in zip(nodes, mapped):
                n["map_x"], n["map_y"] = round(float(mx), 1), round(float(my), 1)
        return trace.to_json(frame_index=frame_index, crop=self.corners.tolist(),
                             alignment=self.detail)

    # ------------------------------------------------------------ bookkeeping
    def _confirmed(self):
        self.frames_since_confirmed = 0
        self._tick = 0
        self.lost = False

    def _unconfirmed(self):
        self.frames_since_confirmed += 1
        self._tick += 1
        if self.frames_since_confirmed >= MAX_UNCONFIRMED:
            self.lost = True

    def _unlock(self):
        """The map has left the frame: drop the crop and go back to the
        start-of-video search (see update)."""
        self.corners = None
        self.source = "searching"
        self.score = 0.0
        self.corner_visible = [False] * 4
        self.flow.prev = None
        self.frames_since_confirmed = 0
        self._tick = 0

    def reset(self, frame, corners):
        """Accept corners placed by hand (manual_crop.py) as the new crop."""
        self._trace(frame, region=grown(corners))
        self.corners = np.asarray(corners, np.float32).copy()
        self.source = "manual"
        self.score = self.score_quad(self.corners)
        self._confirmed()
        self.flow.remember(frame, self.gray, self.corners)

    # ------------------------------------------------------------ per frame
    def update(self, frame, frame_index):
        """Run the crop pipeline on one frame. Returns the 4 corners, or None
        while the map hasn't been found yet."""
        self._trace(frame, region=grown(self.corners) if self.locked else None)
        if not self.locked:
            self.source = "searching"
            if frame_index % SEARCH_EVERY == 0:
                found = self.locate(frame)
                if found:
                    self._lock(frame, found[1], found[0], found[2], frame_index, "found")
            return self.corners

        prior = self.corners.copy()
        trusted = self.frames_since_confirmed == 0
        jump = MAX_JUMP if trusted else None
        shape = frame.shape
        scored = []

        # Candidate from edge tracing: last frame's crop, re-fit to the traced lines.
        H = refine_with_traced_lines(self.ref, self.frame_ink, homography_for(prior, self.ref), prior)
        quad = corners_from_homography(H, self.ref)
        if check_quad(quad, shape, self.ref.aspect, prior, jump)[0]:
            score, detail, outline = self.judge(quad, H)
            scored.append((score, quad, {"source": "lines", "visible": None,
                                         "detail": detail, "outline": outline}))

        # Candidates from the paper's own edges (steps 6-12) -- only needed
        # when the traced-line candidate isn't already strong (it is on most
        # frames, and this saves ~75 ms/frame there).
        strong = scored and scored[0][0] >= STRONG_SCORE
        window = search_window(prior, shape)
        masks = [] if strong else candidate_blobs(frame, window, prior)
        fitted = []
        for blob in masks:
            quad, visible = fit_quad(blob, window[0], window[1], prior)
            if quad is not None:
                fitted.append((quad, visible))
        if not strong and not fitted:
            # No paper blob gave a usable outline: retry the same window on
            # the ink-border mask, through the identical fit (step 12).
            wx0, wy0, wx1, wy1 = window
            quad, visible = fit_quad(ink_border_mask(frame[wy0:wy1, wx0:wx1]), wx0, wy0, prior)
            if quad is not None:
                fitted.append((quad, visible))
        for quad, visible in fitted:
            if quad is None or not check_quad(quad, shape, self.ref.aspect, prior, jump)[0]:
                continue
            quad = snap_corners(self.gray, quad, visible)                   # step 15
            # The paper's outline can't be fooled by repeating patterns, so
            # it's the right starting point when the paper was slid fast and
            # last frame's crop is too far off for step 13 to recover alone.
            H = refine_with_traced_lines(self.ref, self.frame_ink,
                                         homography_for(quad, self.ref), quad)  # step 13
            refined = corners_from_homography(H, self.ref)
            if check_quad(refined, shape, self.ref.aspect, prior, jump)[0]:
                quad = refined
            else:
                H = homography_for(quad, self.ref)
            score, detail, outline = self.judge(quad, H)
            scored.append((score, quad, {"source": "edges", "visible": visible,
                                         "detail": detail, "outline": outline}))

        best = pick_best(scored, TRACK_SCORE)                               # step 17
        if best is None and self.frames_since_confirmed % REALIGN_EVERY == 0:
            best = self._realign(prior, shape)                              # step 13, wider
        if best is not None:
            score, quad, info = best
            quad = keep_if_stationary(quad, prior, info["visible"])         # step 16
            self.corners = smooth(prior, quad) if info["source"] != "realigned" else quad
            self.source, self.score, self.detail = info["source"], score, info["detail"]
            self.outline = info["outline"]
            self._confirmed()
        else:
            self._backups(frame, prior, trusted)
            if not self.locked:
                return None
        self.flow.remember(frame, self.gray, self.corners)

        self.corners = correct_outlier_corner(self.corners, prior)          # step 21
        self.corner_visible = [corner_visible(frame, p) for p in self.corners]
        return self.corners

    def _realign(self, prior, shape):
        """The crop slipped (nothing passed this frame): try sliding it to
        each of the best few alignments a shift search finds, refine each
        against the traced lines, and keep the best one that passes."""
        H_prior = homography_for(prior, self.ref)
        best = None
        for H0 in shift_candidates(self.ref, self.frame_ink, H_prior):
            H = refine_with_traced_lines(self.ref, self.frame_ink, H0,
                                         corners_from_homography(H0, self.ref))
            quad = corners_from_homography(H, self.ref)
            if not check_quad(quad, shape, self.ref.aspect)[0]:
                continue
            score, detail, outline = self.judge(quad, H)
            if score >= TRACK_SCORE and (best is None or score > best[0]):
                best = (score, quad, {"source": "realigned", "visible": None,
                                      "detail": detail, "outline": outline})
        return best

    def _backups(self, frame, prior, trusted):
        quad, score = self.png.locate(self.gray, prior, trusted, self.checked_score)  # step 18
        if quad is not None:
            self.corners, self.source, self.score = smooth(prior, quad), "PNG", score
            self._confirmed()
            return
        quad, score = self.flow.locate(self.gray, prior, self.ref.aspect, self.checked_score)  # 19
        if quad is not None:
            self.corners, self.source, self.score = smooth(prior, quad, 0.85), "flow", score
        else:
            self.source = "held"
        self._unconfirmed()
        if self.lost:                                                        # step 20
            # Nothing has confirmed the crop for MAX_UNCONFIRMED frames: the
            # map has left the frame. Back to the start-of-video search,
            # which locks on again once it reappears.
            self._unlock()
