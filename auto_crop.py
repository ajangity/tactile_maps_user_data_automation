"""This file finds the paper's four corners every frame and keeps tracking
them as it moves. It calls into paper_locator, corner_fitting,
crop_checker, and edge_tracing in order, with backup methods if the
normal detection fails.

Auto-crop: runs the crop pipeline once per frame, in order.

This file holds no image-processing logic of its own -- it's the running
order, calling into:
    edge_tracing.py    trace the printed lines (independent of the crop)
    paper_locator.py   steps 1-8
    corner_fitting.py  steps 9-17
    crop_checker.py    step 14 checks, the two backups, step 21
    paper_model.py     this video's paper color: what's covering the page

Before the map has been found (the video can start with the sheet
face-down, or off-camera), it searches the whole frame every
SEARCH_EVERY frames until a crop scores LOCK_SCORE. After that, each frame:
    trace lines -> candidates (traced-line refinement of last frame's crop,
    plus each paper blob's fitted corners) -> sanity checks -> score ->
    best one wins -> otherwise PNG backup -> optical-flow backup -> hold
    (and every SEARCH_EVERY held frames, a whole-frame search, in case the
    paper was turned or slid while covered).
Hands, sleeves and anything else on the page are found from the paper's
color first (paper_model.py): the ink they cast is dropped, and the score
only expects printed lines where the page is visible.
Once the map has been found, the crop is never dropped: until the end of
the video there is always a crop, confirmed or held. (A held crop is still
re-checked by a whole-frame search every SEARCH_EVERY frames, so a paper
that was moved is found again.) Whether the participant is touching the
map is then decided by their fingertips alone -- see timing.py.
"""

import math

import numpy as np
import cv2

import edge_tracing
import paper_locator
from paper_locator import (alignment_score, candidate_blobs, homography_for, ink_border_mask,
                           orientations, paper_blob_quads, search_window, traced_map_quads)
from paper_model import PaperModel, clean_ink
from corner_fitting import (corners_from_homography, fit_quad, fit_quad_options,
                            keep_if_stationary, pick_best,
                            refine_with_traced_lines, shift_candidates, smooth, snap_corners)
from crop_checker import (MAX_JUMP, OUTLINE_MIN, FlowBackup, PngBackup, check_quad,
                          corner_visible, correct_outlier_corner, outline_agreement,
                          outline_agreement_seen)

LOCK_SCORE = 0.60        # first lock / recovery: a crop must score this
TRACK_SCORE = 0.40       # per frame: a candidate must score this (orientation is already known)
STRONG_SCORE = 0.60      # a traced-line crop this good skips the paper-edge candidates
SEARCH_EVERY = 10        # frames between whole-frame searches before the first lock
REALIGN_EVERY = 3        # frames between shift searches while the crop is unconfirmed
OUTLINE_BYPASS_SCORE = 0.60   # a crop failing the outline check still counts if its lines
                              # score this and its judgeable edges pass (see judge)
OUTLINE_TRUST_SCORE = 0.75    # ...or, while tracking, if they score this and the crop
OUTLINE_TRUST_JUMP = 0.02     # moved no corner more than this share of the frame diagonal
COLD_START_REFINE = 3    # best N pre-scored candidates get the traced-line refinement
COLD_START_STRETCHES = (0.85, 1.0, 1.15)   # a hidden side's distance, x the paper's shape
VISIBLE_MARGIN = 0.10    # occluders are looked for in last frame's crop grown by 10%


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
        self.lock_info = None        # filled the first time the map is found
        self._tick = 0
        self.paper = PaperModel()    # this video's paper color, learned as it runs
        self.visible = None          # this frame's uncovered-paper mask (paper_model.py)
        self._seen_by = None         # the paper model that made it

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
        self.visible = self._seen_by = None

    def _see(self, frame, quad, model=None):
        """Find what covers the paper around this crop (paper_model.py) and
        drop the traced ink on and along it -- before any candidate is fit
        or scored, so hand outlines can't pull or grade a crop."""
        model = model or self.paper
        self.visible = model.visible(frame, quad, 1.0 + VISIBLE_MARGIN)
        self.frame_ink = clean_ink(self.frame_ink, self.visible)
        self._seen_by, self.frame = model, frame

    def score_quad(self, quad):
        return alignment_score(self.ref, self.frame_ink, homography_for(quad, self.ref),
                               visible=self.visible)["score"]

    def judge(self, quad, H=None, prior=None):
        """(score, detail, outline) for a crop. A crop whose edges don't sit
        on the paper's real edges gets score 0, however well its lines match
        (see crop_checker.outline_agreement) -- unless the edges simply
        can't be seen: where the map lies on the other sheet, or under a
        sleeve, there's no paper/desk contrast. Then either
          - the edge points that can be judged must still pass, and the
            lines must score at least OUTLINE_BYPASS_SCORE, or
          - while tracking (prior = last frame's crop), the lines must score
            at least OUTLINE_TRUST_SCORE and no corner may have moved more
            than OUTLINE_TRUST_JUMP of the frame diagonal.
        The outline check is there for crops one room off, which line most
        walls up with other walls. On the benchmark frames, 463 such crops
        (refined and rescored) scored at most 0.72, and every one was 129 px
        or more from the true crop; none passes either rule."""
        outline = outline_agreement(self.gray, quad)
        H = H if H is not None else homography_for(quad, self.ref)
        if outline < OUTLINE_MIN:
            reject = 0.0, {"score": 0.0, "rejected": "edges not on the paper's edges"}, outline
            seen = (outline_agreement_seen(self.frame, self.gray, quad, self._seen_by, self.visible)
                    if self._seen_by is not None else 0.0)
            near = prior is not None and np.max(np.linalg.norm(
                np.asarray(quad, np.float32) - prior, axis=1)) <= OUTLINE_TRUST_JUMP * math.hypot(
                *self.gray.shape[:2])
            if seen < OUTLINE_MIN and not near:
                return reject
            detail = alignment_score(self.ref, self.frame_ink, H, visible=self.visible)
            if not ((seen >= OUTLINE_MIN and detail["score"] >= OUTLINE_BYPASS_SCORE) or
                    (near and detail["score"] >= OUTLINE_TRUST_SCORE)):
                return reject
            detail["outline_seen"] = round(seen, 3)
            return detail["score"], detail, outline
        detail = alignment_score(self.ref, self.frame_ink, H, visible=self.visible)
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
        # Paper blobs from the plain brightness + saturation mask, and again
        # from this video's paper color (learned so far, or from this frame's
        # sheets), which keeps pale hands and arms out of the blobs.
        rough_paper = paper_locator.paper_mask(frame)
        model = self.paper if self.paper.ready else PaperModel.from_frame(frame, rough_paper)
        blobs = paper_blob_quads(frame, lambda f: rough_paper)
        if model.ready:
            blobs += paper_blob_quads(frame, model.sheet_mask)
        for rough, blob, (x0, y0, x1, y1) in blobs:                         # step 3
            fits = fit_quad_options(blob[y0:y1, x0:x1], x0, y0, rough,      # step 12
                                    self.ref.aspect, COLD_START_STRETCHES)
            if not fits:   # ink-border fallback, same padded region
                fits = fit_quad_options(ink_border_mask(frame[y0:y1, x0:x1]), x0, y0, rough,
                                        self.ref.aspect, COLD_START_STRETCHES)
            for quad in [q for q, _ in fits] or [rough]:
                candidates.extend(orientations(quad))                       # step 4
        candidates.extend(traced_map_quads(self.frame_ink, self.ref))      # step 3 (traced)
        candidates = [q for q in candidates
                      if check_quad(q, shape, self.ref.aspect)[0]]         # step 14
        if not candidates:
            return None
        pre = sorted(((self.score_quad(q), q) for q in candidates), key=lambda s: -s[0])
        raw_ink = self.frame_ink
        results = []
        for _, q in pre[:COLD_START_REFINE]:                               # step 13
            # No paper model yet: each candidate judges its own pixels, so
            # hands covering the real map don't sink its score.
            self.frame_ink = raw_ink
            self._see(frame, q, self.paper if self.paper.ready else
                      PaperModel.from_crop(frame, q, raw_ink))
            H = refine_with_traced_lines(self.ref, self.frame_ink, homography_for(q, self.ref), q)
            q2 = corners_from_homography(H, self.ref)
            if not check_quad(q2, shape, self.ref.aspect)[0]:
                q2, H = q, homography_for(q, self.ref)
            score, detail, outline = self.judge(q2, H)
            detail["outline"] = round(outline, 3)
            results.append((score, q2, detail))
        self.frame_ink, self.visible, self._seen_by = raw_ink, None, None
        best = max(results, key=lambda r: r[0])
        if best[0] < LOCK_SCORE:
            return None
        return best

    def _lock(self, frame, quad, score, detail, frame_index, source):
        self.corners = np.asarray(quad, np.float32)
        self.score, self.detail, self.source = score, detail, source
        self._confirmed()
        if not self.paper.ready:
            self.paper = PaperModel.from_crop(frame, self.corners, self.frame_ink)
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

    def _unconfirmed(self):
        self.frames_since_confirmed += 1
        self._tick += 1

    def reset(self, frame, corners):
        """Accept corners placed by hand (manual_crop.py) as the new crop."""
        self._trace(frame, region=grown(corners))
        self.corners = np.asarray(corners, np.float32).copy()
        self.source = "manual"
        self.score = self.score_quad(self.corners)
        self._confirmed()
        if not self.paper.ready:   # as on an automatic lock (_lock)
            self.paper = PaperModel.from_crop(frame, self.corners, self.frame_ink)
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
        if self.paper.ready:
            self._see(frame, prior)
        trusted = self.frames_since_confirmed == 0
        jump = MAX_JUMP if trusted else None
        shape = frame.shape
        scored = []

        # Candidate from edge tracing: last frame's crop, re-fit to the traced lines.
        H = refine_with_traced_lines(self.ref, self.frame_ink, homography_for(prior, self.ref), prior)
        quad = corners_from_homography(H, self.ref)
        if check_quad(quad, shape, self.ref.aspect, prior, jump)[0]:
            score, detail, outline = self.judge(quad, H, prior)
            scored.append((score, quad, {"source": "lines", "visible": None,
                                         "detail": detail, "outline": outline}))

        # Candidates from the paper's own edges (steps 6-12) -- only needed
        # when the traced-line candidate isn't already strong (it is on most
        # frames, and this saves ~75 ms/frame there).
        strong = scored and scored[0][0] >= STRONG_SCORE
        window = search_window(prior, shape)
        masks = [] if strong else candidate_blobs(frame, window, prior)
        if not strong and self.paper.ready:   # and from the paper's color (no pale skin)
            masks += candidate_blobs(frame, window, prior, self.paper.sheet_mask)
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
            score, detail, outline = self.judge(quad, H, prior)
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
            self.paper.maybe_learn(frame, self.corners, self.frame_ink, score)
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
            # A held crop may be stale: the paper can be turned or slid fast
            # while hands cover it. Every SEARCH_EVERY held frames, look
            # again from scratch -- the whole-frame search doesn't depend on
            # where the paper was.
            if self.frames_since_confirmed % SEARCH_EVERY == SEARCH_EVERY - 1:
                found = self.locate(frame)
                if found:
                    self._lock(frame, found[1], found[0], found[2], None, "found")
                    return
        self._unconfirmed()
