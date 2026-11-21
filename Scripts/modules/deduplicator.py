from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.bgmi_detector import CandidateDetection
from modules.config import DetectionConfig, KillEvent


def normalize_ocr_signature(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def compute_text_similarity(text_a: str, text_b: str) -> float:
    na = normalize_ocr_signature(text_a)
    nb = normalize_ocr_signature(text_b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq_ratio = SequenceMatcher(None, na, nb).ratio()
    tokens_a = set(na.split())
    tokens_b = set(nb.split())
    jaccard = len(tokens_a & tokens_b) / float(max(1, len(tokens_a | tokens_b)))
    return max(seq_ratio, 0.55 * seq_ratio + 0.45 * jaccard)


def make_row_fingerprint(row_bgr: np.ndarray, size: Tuple[int, int] = (96, 20)) -> np.ndarray:
    """Create a resolution-normalized grayscale visual fingerprint of a kill-feed row."""
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    # Zero-mean unit-norm normalization for illumination resilience
    arr = resized.astype(np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std > 1e-4:
        arr = (arr - mean) / std
    else:
        arr = arr - mean
    return arr


def compute_visual_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    if fp_a.shape != fp_b.shape:
        return 0.0
    denom = float(np.linalg.norm(fp_a) * np.linalg.norm(fp_b))
    if denom <= 1e-6:
        return 0.0
    cos_sim = float(np.sum(fp_a * fp_b) / denom)
    return max(0.0, min(1.0, cos_sim))


@dataclass
class _TrackedEntry:
    track_id: int
    first_seen_ts: float
    last_seen_ts: float
    best_ts: float
    best_frame_index: int
    best_confidence: float
    best_candidate: CandidateDetection
    best_full_frame: Optional[np.ndarray]
    best_roi_frame: np.ndarray
    fingerprint: np.ndarray
    ocr_sig: str
    last_victim: str = ""
    seen_count: int = 1
    confirmed: bool = False
    confirmed_event: Optional[KillEvent] = None


class KillFeedDeduplicator:
    """
    Stateful temporal & content-aware event deduplicator.

    Key properties:
      - Tracks each kill-feed banner individually across frames, even when the banner
        scrolls vertically inside the ROI as newer kills arrive!
      - Never relies on a naive global cooldown that would suppress two legitimate
        consecutive kills occurring 0.5s apart.
      - Suppresses duplicate screenshots/events while the same banner persists across
        frames or briefly flickers during its lifetime.
      - Requires `min_confirm_frames` observations before confirming an event,
        preventing single-frame artifacts from triggering false positives.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._next_track_id = 1
        self._next_event_id = 1
        self._active_tracks: List[_TrackedEntry] = []
        self._recently_expired: List[_TrackedEntry] = []
        self.confirmed_events: List[KillEvent] = []

    def _is_distinct_victim(self, cand: CandidateDetection, track: _TrackedEntry) -> bool:
        cv = normalize_ocr_signature(cand.victim)
        cv_alnum = re.sub(r"[^a-z0-9]", "", cv)
        if not cv_alnum:
            return False

        tv_candidates = [track.best_candidate.victim, track.last_victim]
        parsed_tvs: List[Tuple[str, str]] = []
        for raw_tv in tv_candidates:
            tv = normalize_ocr_signature(raw_tv)
            tv_alnum = re.sub(r"[^a-z0-9]", "", tv)
            if tv_alnum:
                parsed_tvs.append((tv, tv_alnum))
        if not parsed_tvs:
            return False

        for tv, tv_alnum in parsed_tvs:
            if cv_alnum in tv_alnum or tv_alnum in cv_alnum:
                return False
            if len(cv_alnum) >= 5 and len(tv_alnum) >= 5:
                if compute_text_similarity(cv, tv) >= 0.50:
                    return False
            elif min(len(cv_alnum), len(tv_alnum)) >= 4:
                if compute_text_similarity(cv_alnum, tv_alnum) >= 0.40:
                    return False

        best_tv_alnum = parsed_tvs[0][1]
        if len(cv_alnum) >= 5 and len(best_tv_alnum) >= 5:
            return True
        if min(len(cv_alnum), len(best_tv_alnum)) >= 4:
            return True
        return False

    def _participant_overlap(self, cand: CandidateDetection, track: _TrackedEntry) -> bool:
        ck = re.sub(r"[^a-z0-9]", "", (cand.killer or "").lower())
        tk = re.sub(r"[^a-z0-9]", "", (track.best_candidate.killer or "").lower())
        if len(ck) >= 4 and len(tk) >= 4:
            if ck in tk or tk in ck or compute_text_similarity(ck, tk) >= 0.65:
                return True
        cv = re.sub(r"[^a-z0-9]", "", (cand.victim or "").lower())
        tv = re.sub(r"[^a-z0-9]", "", (track.best_candidate.victim or "").lower())
        if len(cv) >= 4 and len(tv) >= 4:
            if cv in tv or tv in cv or compute_text_similarity(cv, tv) >= 0.68:
                return True
        return False

    def _matches_track(self, cand: CandidateDetection, cand_fp: np.ndarray, track: _TrackedEntry) -> float:
        """
        Compute combined similarity between a candidate row and an existing tracked event.
        Note: We intentionally do NOT require identical Y-coordinates because BGMI kill-feed
        rows scroll upward/downward when another kill appears!
        """
        if self._is_distinct_victim(cand, track):
            return 0.0
        if cand.is_partial_continuation:
            return 0.78 if self._participant_overlap(cand, track) else 0.0
        vis_sim = compute_visual_similarity(cand_fp, track.fingerprint)
        cand_sig = normalize_ocr_signature(cand.ocr_text)

        cv_alnum = re.sub(r"[^a-z0-9]", "", (cand.victim or "").lower())
        tv_alnum = re.sub(r"[^a-z0-9]", "", (track.best_candidate.victim or "").lower())
        v_sim = 0.0
        if cv_alnum and tv_alnum:
            v_sim = compute_text_similarity(cv_alnum, tv_alnum)
            if min(len(cv_alnum), len(tv_alnum)) >= 3 and (cv_alnum in tv_alnum or tv_alnum in cv_alnum):
                v_sim = max(v_sim, 0.88)

        if cand_sig and track.ocr_sig:
            txt_sim = compute_text_similarity(cand_sig, track.ocr_sig)
            if txt_sim >= self.config.text_similarity_threshold:
                return max(txt_sim, vis_sim, 0.55 * v_sim + 0.45 * txt_sim)
            if self._participant_overlap(cand, track):
                return max(0.72, 0.50 * v_sim + 0.50 * max(txt_sim, vis_sim))
            return 0.60 * txt_sim + 0.40 * vis_sim
        return vis_sim

    def _is_match(self, cand: CandidateDetection, cand_fp: np.ndarray, track: _TrackedEntry) -> bool:
        if self._is_distinct_victim(cand, track):
            return False
        if cand.is_partial_continuation:
            return self._participant_overlap(cand, track)
        cand_sig = normalize_ocr_signature(cand.ocr_text)
        if cand_sig and track.ocr_sig:
            txt_sim = compute_text_similarity(cand_sig, track.ocr_sig)
            if txt_sim >= self.config.text_similarity_threshold:
                return True
            if self._participant_overlap(cand, track):
                return True
        vis_sim = compute_visual_similarity(cand_fp, track.fingerprint)
        return vis_sim >= self.config.visual_similarity_threshold

    def update(
        self,
        timestamp_sec: float,
        frame_index: int,
        candidates: List[CandidateDetection],
        full_frame_bgr: Optional[np.ndarray],
        roi_frame_bgr: np.ndarray,
    ) -> List[KillEvent]:
        """
        Ingest detections from the current frame and return any NEWLY confirmed KillEvents.
        """
        # Expire tracks that have disappeared beyond disappear_grace_sec
        still_active: List[_TrackedEntry] = []
        for tr in self._active_tracks:
            if (timestamp_sec - tr.last_seen_ts) <= self.config.disappear_grace_sec:
                still_active.append(tr)
            elif tr.confirmed:
                # Release large image buffers once expired to keep RAM usage O(1)
                tr.best_full_frame = np.empty((0,), dtype=np.uint8)
                tr.best_roi_frame = np.empty((0,), dtype=np.uint8)
                self._recently_expired.append(tr)
        self._active_tracks = still_active

        # Prune old expired history past reappear_cooldown_sec
        self._recently_expired = [
            tr
            for tr in self._recently_expired
            if (timestamp_sec - tr.last_seen_ts) <= self.config.reappear_cooldown_sec
        ]

        newly_confirmed: List[KillEvent] = []
        matched_track_ids = set()

        # Process full detections before partial continuation heartbeats so stacked full banners always win first.
        ordered_candidates = [c for c in candidates if not c.is_partial_continuation] + [
            c for c in candidates if c.is_partial_continuation
        ]

        for cand in ordered_candidates:
            cand_fp = make_row_fingerprint(cand.row_bgr)

            # 1. Check against active tracks
            best_track: Optional[_TrackedEntry] = None
            best_key: Tuple[int, float, int, float, float] = (-1, -1.0, -1, -1.0, -1.0)
            for tr in self._active_tracks:
                if tr.track_id in matched_track_ids:
                    continue
                if self._is_match(cand, cand_fp, tr):
                    sim = self._matches_track(cand, cand_fp, tr)
                    cand_key = (
                        1 if tr.confirmed else 0,
                        round(sim, 4),
                        1 if (cand.is_partial_continuation and tr.confirmed) else 0,
                        tr.best_confidence,
                        tr.last_seen_ts,
                    )
                    if cand_key > best_key:
                        best_key = cand_key
                        best_track = tr

            if best_track is not None:
                matched_track_ids.add(best_track.track_id)
                best_track.last_seen_ts = timestamp_sec
                if not cand.is_partial_continuation:
                    best_track.seen_count += 1
                else:
                    # Only count a partial continuation toward confirming a high-confidence two-participant banner
                    bc = best_track.best_candidate
                    v_alnum = re.sub(r"[^A-Za-z0-9]", "", bc.victim or "")
                    k_alnum = re.sub(r"[^A-Za-z0-9]", "", bc.killer or "")
                    if best_track.confirmed or (
                        bc.ocr_score >= 0.85
                        and bc.visual_score >= 0.78
                        and len(k_alnum) >= 4
                        and len(v_alnum) >= 4
                    ):
                        best_track.seen_count += 1
                if cand.victim.strip():
                    best_track.last_victim = cand.victim
                cand_q = cand.confidence + 0.008 * min(16, len(re.sub(r"[^A-Za-z0-9]", "", cand.ocr_text or "")))
                best_q = best_track.best_confidence + 0.008 * min(
                    16, len(re.sub(r"[^A-Za-z0-9]", "", best_track.best_candidate.ocr_text or ""))
                )
                if not cand.is_partial_continuation and cand_q > best_q:
                    best_track.best_confidence = cand.confidence
                    best_track.best_ts = timestamp_sec
                    best_track.best_frame_index = frame_index
                    best_track.best_candidate = cand
                    best_track.best_full_frame = full_frame_bgr.copy() if full_frame_bgr is not None and full_frame_bgr.size > 0 else None
                    best_track.best_roi_frame = roi_frame_bgr.copy()
                    best_track.fingerprint = cand_fp
                    if cand.ocr_text.strip():
                        best_track.ocr_sig = normalize_ocr_signature(cand.ocr_text)

                if not best_track.confirmed and best_track.seen_count >= self.config.min_confirm_frames:
                    ev = self._promote_track(best_track)
                    newly_confirmed.append(ev)
                continue

            if cand.is_partial_continuation:
                for tr in self._recently_expired:
                    if not self._is_distinct_victim(cand, tr) and self._participant_overlap(cand, tr):
                        tr.last_seen_ts = timestamp_sec
                        break
                continue

            # 2. Check against recently expired confirmed tracks (prevents re-trigger after brief occlusion)
            reappeared = False
            for tr in self._recently_expired:
                if self._is_match(cand, cand_fp, tr):
                    tr.last_seen_ts = timestamp_sec
                    reappeared = True
                    break
            if reappeared:
                continue

            # 3. Create a brand-new track
            new_track = _TrackedEntry(
                track_id=self._next_track_id,
                first_seen_ts=timestamp_sec,
                last_seen_ts=timestamp_sec,
                best_ts=timestamp_sec,
                best_frame_index=frame_index,
                best_confidence=cand.confidence,
                best_candidate=cand,
                best_full_frame=full_frame_bgr.copy() if full_frame_bgr is not None and full_frame_bgr.size > 0 else None,
                best_roi_frame=roi_frame_bgr.copy(),
                fingerprint=cand_fp,
                ocr_sig=normalize_ocr_signature(cand.ocr_text),
                last_victim=cand.victim,
                seen_count=1,
            )
            self._next_track_id += 1
            self._active_tracks.append(new_track)

            if self.config.min_confirm_frames <= 1:
                ev = self._promote_track(new_track)
                newly_confirmed.append(ev)

        return newly_confirmed

    def flush_end_of_stream(self, end_timestamp_sec: float) -> List[KillEvent]:
        """
        If the video ends while a high-confidence, two-participant kill banner is still on screen
        with 1 frame remaining before EOF, promote it instead of silently discarding the final kill.
        Mid-video single-frame artifacts remain suppressed.
        """
        if self.config.min_confirm_frames > 2:
            return []

        eof_window = (1.0 / max(0.5, float(self.config.sample_fps))) + 0.25
        newly_confirmed: List[KillEvent] = []
        for tr in self._active_tracks:
            if tr.confirmed:
                continue
            if (end_timestamp_sec - tr.last_seen_ts) > eof_window:
                continue
            bc = tr.best_candidate
            if bc.is_partial_continuation:
                continue
            k_alnum = re.sub(r"[^A-Za-z0-9]", "", bc.killer or "")
            v_alnum = re.sub(r"[^A-Za-z0-9]", "", bc.victim or "")
            if (
                bc.confidence >= max(0.85, self.config.confidence_threshold)
                and bc.ocr_score >= 0.88
                and bc.visual_score >= 0.78
                and len(k_alnum) >= 4
                and len(v_alnum) >= 4
                and k_alnum.lower() != v_alnum.lower()
            ):
                # Do not re-trigger if identical kill already confirmed recently
                if any(
                    self._is_match(bc, tr.fingerprint, prev)
                    for prev in (self._active_tracks + self._recently_expired)
                    if prev.confirmed and prev.track_id != tr.track_id
                ):
                    continue
                ev = self._promote_track(tr)
                newly_confirmed.append(ev)
        return newly_confirmed

    def _promote_track(self, track: _TrackedEntry) -> KillEvent:
        track.confirmed = True
        cand = track.best_candidate
        ev = KillEvent(
            event_id=self._next_event_id,
            timestamp_sec=track.first_seen_ts,
            frame_index=track.best_frame_index,
            confidence=track.best_confidence,
            ocr_text=cand.ocr_text,
            killer=cand.killer,
            victim=cand.victim,
            weapon_or_action=cand.weapon_or_action,
            row_bbox_in_roi=cand.row_bbox_in_roi,
            is_highlighted=cand.is_highlighted,
            notes=cand.reason,
        )
        self._next_event_id += 1
        track.confirmed_event = ev
        self.confirmed_events.append(ev)
        return ev

    def get_track_frames(self, event_id: int) -> Optional[Tuple[Optional[np.ndarray], np.ndarray]]:
        """Retrieve the cleanest captured full frame & ROI crop for a confirmed event."""
        for tr in self._active_tracks + self._recently_expired:
            if tr.confirmed_event and tr.confirmed_event.event_id == event_id:
                return tr.best_full_frame, tr.best_roi_frame
        return None

    def get_track_timestamp(self, event_id: int) -> Optional[float]:
        """Retrieve the cleanest timestamp for a confirmed event (used for lazy full-frame capture)."""
        for tr in self._active_tracks + self._recently_expired:
            if tr.confirmed_event and tr.confirmed_event.event_id == event_id:
                return tr.best_ts
        return None
