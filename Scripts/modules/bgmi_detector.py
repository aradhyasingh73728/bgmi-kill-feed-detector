from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from modules.calibration import CalibrationManager
from modules.config import DetectionConfig, ROIRegion
from modules.ocr_engine import OCREngine, OCRResult, OCRToken


# Words strongly associated with BGMI kill-feed notifications
BGMI_KILL_KEYWORDS = {
    "killed",
    "knocked",
    "eliminated",
    "finished",
    "headshot",
    "finally",
    "with",
    "by",
    "playzone",
    "redzone",
    "falling",
    "grenade",
    "frag",
    "molotov",
    "vehicle",
    "pan",
    "machete",
    "crowbar",
    "sickle",
    "fist",
    "bleed",
}

BGMI_WEAPONS = {
    "m416",
    "akm",
    "scar",
    "scarl",
    "m762",
    "beryl",
    "dp28",
    "ump",
    "ump45",
    "vector",
    "uzi",
    "bizon",
    "thompson",
    "awm",
    "kar98",
    "kar98k",
    "k98",
    "m24",
    "amr",
    "mini14",
    "slr",
    "sks",
    "mk14",
    "qbu",
    "qbz",
    "g36c",
    "groza",
    "aug",
    "famas",
    "ace32",
    "honeybadger",
    "mg3",
    "m249",
    "dbs",
    "s12k",
    "s686",
    "s1897",
    "ns2000",
    "p90",
    "mp5k",
    "vss",
    "win94",
    "mk12",
    "p1911",
    "p92",
    "deagle",
    "skorpion",
    "crossbow",
    "panzerfaust",
    "m16a4",
    "mk47",
}

# Non-kill UI/HUD/menu/telemetry noise patterns to reject conservatively
UI_NOISE_REGEX = re.compile(
    r"\b("
    r"fps|ping|alive|lobby|inventory|settings|classic|arena|tdm|arcade|"
    r"matchmaking|ready|cancel|start|season|mission|royale|shop|events|"
    r"helmet|vest|ammo|reload|medkit|bandage|energy\s*drink|painkiller|"
    r"zone\s*shrinking|match\s*starting|auto\s*pick|teammate|mic|speaker|"
    r"classic\s*mode|ranked|unranked|tier|recruit|backpack|bag|map|"
    r"players\s+in\s+area|remaining|eliminations|super\s*smooth"
    r")\b",
    re.IGNORECASE,
)

PURE_TELEMETRY_REGEX = re.compile(
    r"^("
    r"\d{1,2}:\d{2}(:\d{2})?|"          # Timers like 01:25
    r"\d+\s*/\s*\d+|"                    # Ammo / Alive counts like 30/120
    r"[NSEW]{1,2}\s*\d{1,3}|"            # Compass headings like NW 145
    r"\d+\s*(ms|fps|m|km)|"              # Ping/FPS/Distance like 24ms, 60fps, 150m
    r"[\d\W_]+"                          # Pure digits/symbols
    r")$",
    re.IGNORECASE,
)


@dataclass
class CandidateDetection:
    row_bbox_in_roi: Tuple[int, int, int, int]  # (x, y, w, h) relative to ROI
    row_bgr: np.ndarray
    confidence: float
    visual_score: float
    ocr_score: float
    calib_score: float
    is_highlighted: bool
    ocr_text: str
    killer: str
    victim: str
    weapon_or_action: str
    reason: str
    is_partial_continuation: bool = False


def _compute_stroke_activity(roi_bgr: np.ndarray, frame_height: int) -> np.ndarray:
    """
    Isolate small bright foreground text & weapon icon strokes from arbitrary
    semi-transparent backgrounds using morphological Top-Hat + horizontal Sobel edges.
    """
    b, g, r = cv2.split(roi_bgr)
    bright_max = np.maximum(np.maximum(r, g), b)

    # Stroke scale proportional to video resolution (~1.4% of frame height, odd kernel)
    k_size = max(5, int(round(frame_height * 0.014)))
    if k_size % 2 == 0:
        k_size += 1
    k_size = min(21, k_size)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(bright_max, cv2.MORPH_TOPHAT, kernel)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)

    combined = cv2.addWeighted(tophat, 0.65, abs_sobel_x, 0.35, 0)
    return combined


def segment_kill_feed_rows(
    roi_bgr: np.ndarray,
    ref_dim: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Segment individual horizontal kill-feed rows inside the selected ROI.
    `ref_dim` should be min(frame_width, frame_height) so portrait and landscape
    videos size kill-feed bands identically.
    Returns list of (x, y, w, h) bounding boxes relative to roi_bgr.
    """
    rh, rw = roi_bgr.shape[:2]
    if rh < 10 or rw < 30:
        return []

    activity = _compute_stroke_activity(roi_bgr, ref_dim)
    active_mask = (activity > 26).astype(np.uint8)

    row_density = np.mean(active_mask, axis=1)
    row_mean_energy = np.mean(activity, axis=1)

    active_y = (row_density >= 0.045) & (row_mean_energy >= 7.0)

    min_row_h = max(12, int(round(ref_dim * 0.015)))
    max_row_h = max(32, int(round(ref_dim * 0.056)))
    gap_tol = max(1, int(round(ref_dim * 0.004)))

    y_col = active_y.astype(np.uint8).reshape(-1, 1)
    close_k = np.ones((gap_tol * 2 + 1, 1), dtype=np.uint8)
    y_closed = cv2.morphologyEx(y_col, cv2.MORPH_CLOSE, close_k).reshape(-1) > 0

    raw_bands: List[Tuple[int, int]] = []
    in_band = False
    start_y = 0
    for y_idx, val in enumerate(y_closed):
        if val and not in_band:
            in_band = True
            start_y = y_idx
        elif not val and in_band:
            in_band = False
            raw_bands.append((start_y, y_idx))
    if in_band:
        raw_bands.append((start_y, rh))

    # Split tall bands at valleys (stacked rows or background clutter)
    split_bands: List[Tuple[int, int]] = []
    for y0, y1 in raw_bands:
        band_h = y1 - y0
        if band_h <= max_row_h:
            split_bands.append((y0, y1))
        elif band_h <= max_row_h * 3.5:
            sub_energy = row_mean_energy[y0:y1]
            k = max(3, min_row_h // 2)
            smooth = np.convolve(sub_energy, np.ones(k) / k, mode="same")
            cur_start = 0
            pos = min_row_h
            while pos < band_h - min_row_h:
                window = smooth[pos : min(band_h - min_row_h, pos + max_row_h)]
                if len(window) == 0:
                    break
                rel_min = int(np.argmin(window))
                valley = pos + rel_min
                if smooth[valley] < 0.45 * float(np.max(smooth[cur_start:valley + 1]) + 1e-5):
                    split_bands.append((y0 + cur_start, y0 + valley))
                    cur_start = valley + 1
                    pos = cur_start + min_row_h
                else:
                    pos += max(4, min_row_h // 2)
            if band_h - cur_start >= min_row_h:
                split_bands.append((y0 + cur_start, y1))
        else:
            continue

    if not split_bands and rh <= max_row_h * 1.25 and float(np.mean(active_y)) > 0.20:
        split_bands.append((0, rh))

    boxes: List[Tuple[int, int, int, int]] = []
    pad_y = max(2, int(round(ref_dim * 0.003)))
    pad_x = max(4, int(round(rw * 0.015)))

    for y0, y1 in split_bands:
        # Trim weak top/bottom foliage tails when a band is taller than a typical banner
        band_energy = row_mean_energy[y0:y1]
        if len(band_energy) > max_row_h * 0.72:
            peak_e = float(np.max(band_energy))
            core_idx = np.where(band_energy >= peak_e * 0.42)[0]
            if len(core_idx) >= min_row_h:
                y0_t = y0 + int(core_idx[0])
                y1_t = y0 + int(core_idx[-1]) + 1
                if (y1_t - y0_t) >= min_row_h:
                    y0, y1 = y0_t, y1_t

        band_h = y1 - y0
        if band_h > max_row_h * 1.25 and band_h <= max_row_h * 2.2:
            # Extract highest-energy banner strip inside a tall unsplit ROI (e.g. 65px portrait ROI)
            win_h = min(band_h, max(min_row_h, int(round(ref_dim * 0.048))))
            best_start = y0
            best_sum = -1.0
            for s in range(y0, y1 - win_h + 1):
                cur_sum = float(np.sum(row_mean_energy[s : s + win_h]))
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_start = s
            y0, y1 = best_start, best_start + win_h
            band_h = y1 - y0

        if band_h < min_row_h or band_h > max_row_h * 1.25:
            continue

        ry0 = max(0, y0 - pad_y)
        ry1 = min(rh, y1 + pad_y)
        band_mask = active_mask[ry0:ry1, :]
        col_energy = np.sum(band_mask, axis=0)
        active_cols = np.where(col_energy > 0)[0]
        if len(active_cols) < 16:
            continue

        rx0 = max(0, int(active_cols[0]) - pad_x)
        rx1 = min(rw, int(active_cols[-1]) + 1 + pad_x)
        bw = rx1 - rx0
        bh = ry1 - ry0

        if bw < max(48, int(rw * 0.24)) or bh < min_row_h:
            continue
        # Reject thin clipped grass/HUD edge slivers touching ROI top or bottom boundary
        if (ry0 == 0 or ry1 == rh) and bh < max(16, int(round(ref_dim * 0.023))):
            continue
        if (bw / float(max(1, bh))) < 2.6:
            continue

        boxes.append((rx0, ry0, bw, bh))

    return boxes


def _evaluate_visual_structure(
    row_bgr: np.ndarray,
    ref_dim: int,
) -> Tuple[float, bool, str]:
    """
    Evaluate BGMI kill-feed visual geometry, contrast, 3-zone layout, and highlight color.
    Returns (visual_score in [0,1], is_highlighted, reason).
    """
    rh, rw = row_bgr.shape[:2]
    if rh <= 0 or rw <= 0:
        return 0.0, False, "empty row"

    aspect = rw / float(rh)
    if aspect < 2.6:
        return 0.0, False, f"aspect too narrow ({aspect:.2f})"

    b, g, r = cv2.split(row_bgr)
    bright_max = np.maximum(np.maximum(r, g), b)
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)

    bg_luma = float(np.percentile(gray, 25))
    fg_luma = float(np.percentile(bright_max, 93))
    contrast = fg_luma - bg_luma

    if contrast < 36.0:
        return 0.0, False, f"low fg/bg contrast ({contrast:.1f})"
    if bg_luma > 225.0:
        return 0.0, False, f"background too bright ({bg_luma:.1f})"

    contrast_score = min(1.0, max(0.0, (contrast - 34.0) / 105.0))
    dark_bar_score = min(1.0, max(0.25, (215.0 - bg_luma) / 150.0))

    activity = _compute_stroke_activity(row_bgr, ref_dim)
    stroke_bin = (activity > 26).astype(np.float32)

    z_left = float(np.mean(stroke_bin[:, : max(1, int(rw * 0.42))]))
    z_mid = float(np.mean(stroke_bin[:, int(rw * 0.32) : max(int(rw * 0.32) + 1, int(rw * 0.68))]))
    z_right = float(np.mean(stroke_bin[:, int(rw * 0.58) :]))

    has_two_sides = z_left >= 0.032 and z_right >= 0.032
    if not has_two_sides:
        side_score = 0.20 * min(1.0, (z_left + z_right) / 0.12)
    else:
        balance = min(z_left, z_right) / max(z_left, z_right, 1e-5)
        density_ok = min(1.0, (z_left + z_right) / 0.16)
        mid_ok = 1.0 if z_mid >= 0.02 else 0.6
        side_score = 0.55 * density_ok + 0.25 * balance + 0.20 * mid_ok

    overall_stroke_density = float(np.mean(stroke_bin))
    if overall_stroke_density > 0.85:
        return 0.0, False, f"excessive texture noise ({overall_stroke_density:.2f})"

    hsv = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    red_mask = (((h_ch <= 14) | (h_ch >= 165)) & (s_ch >= 55) & (v_ch >= 55))
    gold_mask = ((h_ch >= 15) & (h_ch <= 34) & (s_ch >= 65) & (v_ch >= 80))
    highlight_ratio = float(np.mean(red_mask | gold_mask))
    is_highlighted = highlight_ratio >= 0.08

    aspect_score = min(1.0, max(0.0, (aspect - 2.6) / 5.0))
    visual_score = (
        0.36 * contrast_score
        + 0.14 * dark_bar_score
        + 0.38 * side_score
        + 0.12 * aspect_score
    )
    if is_highlighted:
        visual_score = min(1.0, visual_score + 0.06)

    reason = "ok"
    if overall_stroke_density > 0.63:
        excess = (overall_stroke_density - 0.63) / 0.22
        visual_score *= max(0.55, 1.0 - 0.45 * excess)
        reason = f"high-texture ({overall_stroke_density:.2f})"

    return float(max(0.0, min(1.0, visual_score))), is_highlighted, reason


def _clean_alnum(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text or "")


def _parse_kill_feed_ocr(ocr: OCRResult) -> Tuple[float, str, str, str, str]:
    """
    Validate OCR output against BGMI kill-feed grammar & UI noise filters.
    Strictly rejects random 2-letter background hallucinations (trees/roofs/clouds).
    Returns (ocr_score, killer, weapon_or_action, victim, rejection_reason).
    """
    raw_text = ocr.full_text.strip()
    if not raw_text:
        return 0.0, "", "", "", "no OCR text"

    if UI_NOISE_REGEX.search(raw_text):
        return 0.0, "", "", "", f"UI noise keyword matched: '{raw_text}'"

    if PURE_TELEMETRY_REGEX.match(raw_text):
        return 0.0, "", "", "", f"telemetry/counter string: '{raw_text}'"

    alnum_chars = _clean_alnum(raw_text)
    if len(alnum_chars) < 6:
        return 0.0, "", "", "", f"insufficient alphanumeric characters: '{raw_text}'"

    # Reject garbage lines dominated by punctuation/divider artifacts (e.g. '| = = ne pa')
    if ocr.tokens:
        symbol_tokens = sum(1 for t in ocr.tokens if len(_clean_alnum(t.text)) < 2)
        if symbol_tokens >= max(2, int(len(ocr.tokens) * 0.60)) and len(alnum_chars) < 9:
            return 0.0, "", "", "", f"high symbol/fragment noise: '{raw_text}'"

    # Split fused multi-word tokens (e.g. single wide box 'Killer Victim') proportionally across X
    expanded_tokens: List[OCRToken] = []
    for t in ocr.tokens:
        parts = t.text.split()
        span = t.x_max_norm - t.x_min_norm
        if len(parts) >= 2 and (len(ocr.tokens) <= 2 or span >= 0.35):
            total_chars = max(1, sum(len(p) for p in parts))
            cursor = t.x_min_norm
            for p in parts:
                frac = len(p) / float(total_chars)
                nx0 = cursor
                nx1 = min(t.x_max_norm, cursor + span * frac)
                expanded_tokens.append(
                    OCRToken(
                        text=p,
                        confidence=t.confidence,
                        x_min_norm=nx0,
                        x_max_norm=nx1,
                    )
                )
                cursor = nx1
        else:
            expanded_tokens.append(t)

    words = re.findall(r"[A-Za-z0-9_\-]+", raw_text.lower())
    matched_keywords = [w for w in words if w in BGMI_KILL_KEYWORDS]
    matched_weapons = [w for w in words if w in BGMI_WEAPONS]

    excluded_vocab = set(matched_keywords) | set(matched_weapons)
    left_tokens = [
        t for t in expanded_tokens
        if t.x_center_norm < 0.48
        and len(_clean_alnum(t.text)) >= 2
        and re.sub(r"[^a-z0-9_\-]", "", t.text.lower()) not in excluded_vocab
    ]
    right_tokens = [
        t for t in expanded_tokens
        if t.x_center_norm > 0.52
        and len(_clean_alnum(t.text)) >= 2
        and re.sub(r"[^a-z0-9_\-]", "", t.text.lower()) not in excluded_vocab
    ]

    # If all tokens sit on one side of 0.50 (wide ROI padding on right/left), split at largest horizontal weapon-icon gap
    if not (left_tokens and right_tokens):
        clean_tokens = [
            t for t in expanded_tokens
            if len(_clean_alnum(t.text)) >= 2
            and re.sub(r"[^a-z0-9_\-]", "", t.text.lower()) not in excluded_vocab
        ]
        if len(clean_tokens) >= 2:
            best_gap = 0.0
            best_idx = -1
            for i in range(len(clean_tokens) - 1):
                g = clean_tokens[i + 1].x_min_norm - clean_tokens[i].x_max_norm
                if g > best_gap:
                    best_gap = g
                    best_idx = i + 1
            if best_idx > 0 and best_gap >= 0.06:
                l_part = clean_tokens[:best_idx]
                r_part = clean_tokens[best_idx:]
                l_alnum = sum(len(_clean_alnum(t.text)) for t in l_part)
                r_alnum = sum(len(_clean_alnum(t.text)) for t in r_part)
                if l_alnum >= 3 and r_alnum >= 3:
                    left_tokens = l_part
                    right_tokens = r_part

    killer = " ".join(t.text for t in left_tokens).strip()
    victim = " ".join(t.text for t in right_tokens).strip()
    weapon_or_action = ""

    if matched_weapons:
        weapon_or_action = matched_weapons[0].upper()
    elif matched_keywords:
        weapon_or_action = matched_keywords[0]

    if not killer or not victim:
        verb_match = re.search(
            r"^(.+?)\s+(?:knocked(?:\s+out)?|eliminated|killed|finished)\s+(?:by\s+)?(.+)$",
            raw_text,
            re.IGNORECASE,
        )
        if verb_match:
            killer = verb_match.group(1).strip()
            victim = verb_match.group(2).strip()
            if not weapon_or_action:
                weapon_or_action = "eliminated"

    killer_alnum = _clean_alnum(killer)
    victim_alnum = _clean_alnum(victim)
    has_two_sides = bool(killer_alnum) and bool(victim_alnum) and (killer_alnum.lower() != victim_alnum.lower())
    has_kill_vocab = bool(matched_keywords or matched_weapons)

    if not has_two_sides and not has_kill_vocab:
        return 0.0, killer, weapon_or_action, victim, f"lacks two-participant or kill-feed structure: '{raw_text}'"

    all_clean_tokens = [len(_clean_alnum(t.text)) for t in (left_tokens + right_tokens)]
    max_tok_len = max(all_clean_tokens) if all_clean_tokens else 0

    if has_kill_vocab:
        if (len(killer_alnum) + len(victim_alnum)) < 4:
            return 0.0, killer, weapon_or_action, victim, f"kill keyword without readable participant name: '{raw_text}'"
    else:
        # Icon-only BGMI banner: enforce real player-name length (blocks 'ne pa', 'See ee', 'Bia eS ES')
        if min(len(killer_alnum), len(victim_alnum)) < 3 or max(len(killer_alnum), len(victim_alnum)) < 4:
            return 0.0, killer, weapon_or_action, victim, f"participant name too short/noisy: '{raw_text}'"
        if (len(killer_alnum) + len(victim_alnum)) < 7:
            return 0.0, killer, weapon_or_action, victim, f"total participant text too short: '{raw_text}'"
        if max_tok_len < 4 and (len(killer_alnum) + len(victim_alnum)) < 9:
            return 0.0, killer, weapon_or_action, victim, f"fragmented short tokens only: '{raw_text}'"

    valid_tokens = left_tokens + right_tokens
    valid_conf = (
        float(sum(t.confidence for t in valid_tokens) / len(valid_tokens))
        if valid_tokens
        else ocr.mean_confidence
    )
    if valid_conf < 0.28 and not has_kill_vocab:
        return 0.0, killer, weapon_or_action, victim, f"low participant OCR confidence ({valid_conf:.2f})"

    gap_score = 0.0
    if left_tokens and right_tokens:
        gap = right_tokens[0].x_min_norm - left_tokens[-1].x_max_norm
        if gap >= 0.05:
            gap_score = min(1.0, gap / 0.18)
        elif has_kill_vocab:
            gap_score = 0.75

    total_name_alnum = len(killer_alnum) + len(victim_alnum)
    name_quality = min(1.0, total_name_alnum / 13.0)
    conf_factor = min(1.0, max(0.0, valid_conf) / 0.76)

    if has_two_sides and has_kill_vocab:
        score = 0.42 + 0.30 * conf_factor + 0.28 * name_quality
    elif has_two_sides:
        score = 0.26 + 0.34 * conf_factor + 0.25 * name_quality + 0.15 * gap_score
    else:
        score = 0.35 + 0.35 * conf_factor + 0.20 * name_quality

    if not weapon_or_action and has_two_sides:
        weapon_or_action = "weapon_icon"

    return float(min(1.0, score)), killer, weapon_or_action, victim, "ok"


class BGMIKillFeedDetector:
    """
    Conservative BGMI kill-feed detector operating strictly inside the user-selected ROI.
    Combines:
      1. Multi-row horizontal segmentation
      2. BGMI strip contrast, geometry & 3-zone layout scoring
      3. Pluggable CPU OCR + kill-feed grammar & UI-noise exclusion
      4. Optional user-calibrated BGMI sample matching
    """

    def __init__(
        self,
        config: DetectionConfig,
        ocr_engine: OCREngine,
        calibration_manager: CalibrationManager,
    ) -> None:
        self.config = config
        self.ocr_engine = ocr_engine
        self.calibration = calibration_manager

    def detect_in_frame(
        self,
        frame_bgr: np.ndarray,
        roi: ROIRegion,
    ) -> List[CandidateDetection]:
        """Analyze only `roi` inside `frame_bgr` and return confirmed-threshold row candidates."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        fh, fw = frame_bgr.shape[:2]
        ref_dim = min(fw, fh)
        mapped_roi = roi.map_to_resolution(fw, fh).clamp()
        roi_bgr = frame_bgr[
            mapped_roi.y : mapped_roi.y + mapped_roi.height,
            mapped_roi.x : mapped_roi.x + mapped_roi.width,
        ]
        return self.detect_in_roi_crop(roi_bgr, ref_dim=ref_dim)

    def detect_in_roi_crop(
        self,
        roi_bgr: np.ndarray,
        ref_dim: int,
    ) -> List[CandidateDetection]:
        """Analyze an already-cropped ROI image directly."""
        if roi_bgr is None or roi_bgr.size == 0:
            return []

        row_boxes = segment_kill_feed_rows(roi_bgr, ref_dim)
        if not row_boxes:
            return []

        candidates: List[CandidateDetection] = []
        for rx, ry, rw, rh in row_boxes:
            row_bgr = roi_bgr[ry : ry + rh, rx : rx + rw]
            vis_score, is_highlighted, vis_reason = _evaluate_visual_structure(row_bgr, ref_dim)
            is_high_texture = vis_reason.startswith("high-texture")
            if vis_score < 0.42:
                continue
            if is_high_texture and not self.ocr_engine.is_available:
                continue

            if self.config.prefer_highlighted_rows and not is_highlighted:
                vis_score *= 0.85

            calib_score = 0.0
            calib_label = ""
            if self.calibration.is_calibrated:
                c_match = self.calibration.match_row(row_bgr)
                if c_match is not None:
                    calib_score = c_match.score
                    calib_label = c_match.sample_name

            ocr_score = 0.0
            ocr_text = ""
            killer = ""
            victim = ""
            weapon_or_action = ""

            if self.ocr_engine.is_available:
                ocr_res = self.ocr_engine.recognize_row(row_bgr)
                ocr_text = ocr_res.full_text
                ocr_score, killer, weapon_or_action, victim, ocr_reason = _parse_kill_feed_ocr(ocr_res)

                # Guarded split-zone fallback: run ONLY when the full row ALREADY contains at least
                # one real >=4-char alphanumeric token (or strong calibration match), so empty tree/roof
                # crops never trigger split-zone text hallucination.
                is_known_noise = ocr_reason.startswith("UI noise") or ocr_reason.startswith("telemetry")
                has_real_anchor_token = any(len(_clean_alnum(t.text)) >= 4 for t in ocr_res.tokens)
                if (
                    not is_known_noise
                    and (ocr_score == 0.0 or not killer or not victim)
                    and vis_score >= 0.55
                    and rw >= 90
                    and (has_real_anchor_token or calib_score >= 0.55)
                ):
                    left_crop = row_bgr[:, : max(1, int(rw * 0.44))]
                    right_crop = row_bgr[:, int(rw * 0.56) :]
                    l_res = self.ocr_engine.recognize_row(left_crop)
                    r_res = self.ocr_engine.recognize_row(right_crop)
                    merged_tokens: List[OCRToken] = []
                    for t in l_res.tokens:
                        if len(_clean_alnum(t.text)) >= 2:
                            merged_tokens.append(
                                OCRToken(
                                    text=t.text,
                                    confidence=t.confidence,
                                    x_min_norm=t.x_min_norm * 0.44,
                                    x_max_norm=t.x_max_norm * 0.44,
                                )
                            )
                    for t in r_res.tokens:
                        if len(_clean_alnum(t.text)) >= 2:
                            merged_tokens.append(
                                OCRToken(
                                    text=t.text,
                                    confidence=t.confidence,
                                    x_min_norm=0.56 + t.x_min_norm * 0.44,
                                    x_max_norm=0.56 + t.x_max_norm * 0.44,
                                )
                            )
                    if merged_tokens:
                        confs = [t.confidence for t in merged_tokens]
                        split_res = OCRResult(
                            full_text=" ".join(t.text for t in merged_tokens),
                            tokens=merged_tokens,
                            mean_confidence=float(sum(confs) / len(confs)),
                            backend=ocr_res.backend,
                        )
                        s_score, s_k, s_w, s_v, s_rsn = _parse_kill_feed_ocr(split_res)
                        if s_score > ocr_score:
                            ocr_score, killer, weapon_or_action, victim, ocr_reason = (
                                s_score,
                                s_k,
                                s_w,
                                s_v,
                                s_rsn,
                            )
                            ocr_text = split_res.full_text

                if is_high_texture and ocr_score < 0.72:
                    continue

                if ocr_score == 0.0 and self.config.require_ocr_if_available:
                    # Emit a guarded single-side continuation heartbeat (never creates new events)
                    # when a clean killer/victim name persists on a banner whose opposite half faded out.
                    strong_tokens = [t for t in ocr_res.tokens if len(_clean_alnum(t.text)) >= 5 and t.confidence >= 0.75]
                    if (
                        not is_known_noise
                        and not is_high_texture
                        and vis_score >= 0.72
                        and strong_tokens
                    ):
                        left_k = " ".join(t.text for t in strong_tokens if t.x_center_norm < 0.48).strip()
                        right_v = " ".join(t.text for t in strong_tokens if t.x_center_norm > 0.52).strip()
                        if left_k or right_v:
                            if self.config.only_player_kills and self.config.player_ign.strip():
                                ign = self.config.player_ign.strip().lower()
                                if ign not in f"{left_k} {ocr_text}".lower():
                                    continue
                            candidates.append(
                                CandidateDetection(
                                    row_bbox_in_roi=(rx, ry, rw, rh),
                                    row_bgr=row_bgr.copy(),
                                    confidence=float(min(0.60, 0.34 * vis_score + 0.30 * ocr_res.mean_confidence)),
                                    visual_score=float(vis_score),
                                    ocr_score=0.0,
                                    calib_score=float(calib_score),
                                    is_highlighted=is_highlighted,
                                    ocr_text=ocr_text,
                                    killer=left_k,
                                    victim=right_v,
                                    weapon_or_action=weapon_or_action,
                                    reason=f"continuation vis={vis_score:.2f}",
                                    is_partial_continuation=True,
                                )
                            )
                    if calib_score < 0.85:
                        continue

                if self.config.only_player_kills and self.config.player_ign.strip():
                    ign = self.config.player_ign.strip().lower()
                    haystack = f"{killer} {ocr_text}".lower()
                    if ign not in haystack:
                        continue
            else:
                if self.calibration.is_calibrated and calib_score < 0.58 and vis_score < 0.78:
                    continue
                if not self.calibration.is_calibrated and vis_score < 0.78:
                    continue

            if not weapon_or_action and calib_label:
                weapon_or_action = calib_label

            if self.ocr_engine.is_available and self.calibration.is_calibrated and calib_score > 0.25:
                confidence = max(
                    0.34 * vis_score + 0.66 * ocr_score,
                    0.28 * vis_score + 0.47 * ocr_score + 0.25 * calib_score,
                )
            elif self.ocr_engine.is_available:
                confidence = 0.34 * vis_score + 0.66 * ocr_score
            elif self.calibration.is_calibrated:
                confidence = 0.42 * vis_score + 0.58 * calib_score
            else:
                confidence = 0.90 * vis_score

            if is_highlighted:
                confidence = min(1.0, confidence + 0.04)

            if confidence >= self.config.confidence_threshold:
                candidates.append(
                    CandidateDetection(
                        row_bbox_in_roi=(rx, ry, rw, rh),
                        row_bgr=row_bgr.copy(),
                        confidence=float(confidence),
                        visual_score=float(vis_score),
                        ocr_score=float(ocr_score),
                        calib_score=float(calib_score),
                        is_highlighted=is_highlighted,
                        ocr_text=ocr_text,
                        killer=killer,
                        victim=victim,
                        weapon_or_action=weapon_or_action,
                        reason=f"vis={vis_score:.2f}, ocr={ocr_score:.2f}, cal={calib_score:.2f}",
                    )
                )

        return candidates
