from __future__ import annotations

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class OCRToken:
    text: str
    confidence: float
    x_min_norm: float
    x_max_norm: float

    @property
    def x_center_norm(self) -> float:
        return 0.5 * (self.x_min_norm + self.x_max_norm)


@dataclass
class OCRResult:
    full_text: str
    tokens: List[OCRToken] = field(default_factory=list)
    mean_confidence: float = 0.0
    backend: str = "none"


def _extract_white_gold_luma(upscaled_bgr: np.ndarray) -> np.ndarray:
    """
    Suppress cool/green backgrounds (blue sky, cyan HUD glow, grass, trees) while preserving
    both neutral-white and warm-gold BGMI kill-feed text as high-intensity foreground.
    """
    b, g, r = cv2.split(upscaled_bgr)
    gray = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cool_excess = np.maximum(0.0, np.maximum(b, g).astype(np.float32) - r.astype(np.float32))
    luma = np.clip(gray - 0.50 * cool_excess, 0.0, 255.0).astype(np.uint8)
    return luma


def preprocess_row_for_ocr(row_bgr: np.ndarray) -> np.ndarray:
    """
    Upscale, suppress blue/green backgrounds without amplifying faint texture noise,
    sharpen, and invert so bright white/gold kill-feed text becomes dark ink on clean light paper.
    """
    h, w = row_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return row_bgr

    target_h = 56
    scale = min(3.2, max(1.6, target_h / float(h)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    upscaled = cv2.resize(row_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    luma = _extract_white_gold_luma(upscaled)

    # On bright sky/gradient backgrounds, apply bounded Top-Hat (NO global NORM_MINMAX, which
    # amplifies faint tree/cloud noise up to 255).
    bg_p25 = float(np.percentile(luma, 25))
    if bg_p25 > 80.0:
        k_size = max(21, int(round(new_h * 0.28)))
        if k_size % 2 == 0:
            k_size += 1
        k_size = min(39, k_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        tophat = cv2.morphologyEx(luma, cv2.MORPH_TOPHAT, kernel)
        tophat_bounded = np.clip(tophat.astype(np.float32) * 2.1, 0.0, 255.0).astype(np.uint8)
        combined = cv2.addWeighted(luma, 0.55, tophat_bounded, 0.45, 0)
    else:
        combined = luma

    blurred = cv2.GaussianBlur(combined, (0, 0), 1.0)
    sharpened = cv2.addWeighted(combined, 1.45, blurred, -0.45, 0)

    inverted = 255 - sharpened
    padded = cv2.copyMakeBorder(inverted, 14, 14, 18, 18, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def preprocess_row_threshold(row_bgr: np.ndarray, thresh: int = 172) -> np.ndarray:
    """Crisp binary white/gold text isolation fallback for glowing or noisy BGMI banners."""
    h, w = row_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return row_bgr
    scale = min(3.2, max(1.6, 56.0 / float(h)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    upscaled = cv2.resize(row_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    luma = _extract_white_gold_luma(upscaled)
    _, bin_inv = cv2.threshold(luma, thresh, 255, cv2.THRESH_BINARY_INV)
    padded = cv2.copyMakeBorder(bin_inv, 14, 14, 18, 18, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def _alnum_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


class OCREngine:
    """
    CPU-friendly pluggable OCR wrapper supporting:
      1. rapidocr_onnxruntime (pure pip wheel on Windows 10/11, no external binary)
      2. pytesseract / Tesseract CLI (requires Tesseract executable on PATH)
      3. none (visual/calibration-only fallback with conservative gating)
    """

    def __init__(self, preferred_backend: str = "auto") -> None:
        self.preferred_backend = (preferred_backend or "auto").lower().strip()
        self.backend_name: str = "none"
        self._rapid_ocr = None
        self._pytesseract = None
        self._tesseract_exe: Optional[str] = None
        self._init_backend()

    def _init_backend(self) -> None:
        order = (
            ["rapidocr", "tesseract"]
            if self.preferred_backend == "auto"
            else [self.preferred_backend]
        )
        for candidate in order:
            if candidate == "none":
                self.backend_name = "none"
                return
            if candidate == "rapidocr":
                try:
                    from rapidocr_onnxruntime import RapidOCR  # type: ignore

                    try:
                        self._rapid_ocr = RapidOCR(
                            det_limit_side_len=736,
                            det_limit_type="max",
                        )
                    except TypeError:
                        self._rapid_ocr = RapidOCR()
                    self.backend_name = "rapidocr"
                    return
                except Exception:
                    continue
            if candidate == "tesseract":
                exe = shutil.which("tesseract")
                try:
                    import pytesseract  # type: ignore

                    if exe or getattr(pytesseract.pytesseract, "tesseract_cmd", ""):
                        pytesseract.get_tesseract_version()
                        self._pytesseract = pytesseract
                        self._tesseract_exe = exe
                        self.backend_name = "tesseract"
                        return
                except Exception:
                    pass
                if exe:
                    try:
                        proc = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5)
                        if proc.returncode == 0:
                            self._tesseract_exe = exe
                            self.backend_name = "tesseract"
                            return
                    except Exception:
                        pass
        self.backend_name = "none"

    @property
    def is_available(self) -> bool:
        return self.backend_name != "none"

    def _run_active_backend(self, img_bgr: np.ndarray) -> OCRResult:
        if self.backend_name == "rapidocr":
            return self._run_rapidocr(img_bgr)
        if self.backend_name == "tesseract":
            return self._run_tesseract(img_bgr)
        return OCRResult(full_text="", tokens=[], mean_confidence=0.0, backend="none")

    def recognize_row(self, row_bgr: np.ndarray) -> OCRResult:
        if not self.is_available or row_bgr is None or row_bgr.size == 0:
            return OCRResult(full_text="", tokens=[], mean_confidence=0.0, backend=self.backend_name)

        prep = preprocess_row_for_ocr(row_bgr)
        primary = self._run_active_backend(prep)
        primary_alnum = _alnum_count(primary.full_text)
        if primary_alnum >= 7:
            return primary

        # Skip redundant 2nd OCR pass when primary found <=2 alnum chars and crop lacks crisp white/gold text over darker backdrop
        if not primary.tokens or primary_alnum <= 2:
            luma = _extract_white_gold_luma(row_bgr)
            p98 = float(np.percentile(luma, 98))
            p35 = float(np.percentile(luma, 35))
            if p98 < 168.0 or (p98 - p35) < 62.0:
                return primary

        # Secondary crisp threshold pass helps on bright-glow/aura BGMI banners while staying blank on trees/sky
        thresh_prep = preprocess_row_threshold(row_bgr)
        secondary = self._run_active_backend(thresh_prep)
        sec_alnum = _alnum_count(secondary.full_text)
        if sec_alnum > primary_alnum:
            return secondary
        if sec_alnum == primary_alnum and secondary.mean_confidence > primary.mean_confidence:
            return secondary
        return primary

    def _run_rapidocr(self, img_bgr: np.ndarray) -> OCRResult:
        h, w = img_bgr.shape[:2]
        try:
            try:
                result, _ = self._rapid_ocr(img_bgr, use_cls=False)
            except TypeError:
                result, _ = self._rapid_ocr(img_bgr)
        except Exception:
            return OCRResult(full_text="", tokens=[], mean_confidence=0.0, backend="rapidocr")

        if not result:
            return OCRResult(full_text="", tokens=[], mean_confidence=0.0, backend="rapidocr")

        tokens: List[OCRToken] = []
        confs: List[float] = []
        ordered = sorted(result, key=lambda item: min(pt[0] for pt in item[0]))
        texts: List[str] = []
        for item in ordered:
            box, text, score = item[0], str(item[1]).strip(), float(item[2])
            if not text:
                continue
            xs = [pt[0] for pt in box]
            x_min = max(0.0, min(1.0, min(xs) / float(max(1, w))))
            x_max = max(0.0, min(1.0, max(xs) / float(max(1, w))))
            tokens.append(
                OCRToken(
                    text=text,
                    confidence=score,
                    x_min_norm=x_min,
                    x_max_norm=x_max,
                )
            )
            texts.append(text)
            confs.append(score)

        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return OCRResult(
            full_text=" ".join(texts).strip(),
            tokens=tokens,
            mean_confidence=mean_conf,
            backend="rapidocr",
        )

    def _run_tesseract(self, img_bgr: np.ndarray) -> OCRResult:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        tokens: List[OCRToken] = []
        confs: List[float] = []
        texts: List[str] = []

        if self._pytesseract is not None:
            try:
                data = self._pytesseract.image_to_data(
                    gray,
                    config="--psm 7 --oem 3",
                    output_type=self._pytesseract.Output.DICT,
                )
            except Exception:
                return OCRResult(full_text="", tokens=[], mean_confidence=0.0, backend="tesseract")

            n = len(data.get("text", []))
            for i in range(n):
                raw_txt = str(data["text"][i]).strip()
                if not raw_txt:
                    continue
                try:
                    conf = float(data["conf"][i])
                except Exception:
                    conf = -1.0
                if conf < 25.0:
                    continue
                norm_conf = max(0.0, min(1.0, conf / 100.0))
                left = float(data["left"][i])
                bw = float(data["width"][i])
                x_min = max(0.0, min(1.0, left / float(max(1, w))))
                x_max = max(0.0, min(1.0, (left + bw) / float(max(1, w))))
                tokens.append(OCRToken(text=raw_txt, confidence=norm_conf, x_min_norm=x_min, x_max_norm=x_max))
                texts.append(raw_txt)
                confs.append(norm_conf)
        elif self._tesseract_exe:
            ok, encoded = cv2.imencode(".png", gray)
            if not ok:
                return OCRResult(full_text="", tokens=[], mean_confidence=0.0, backend="tesseract")
            try:
                proc = subprocess.run(
                    [self._tesseract_exe, "stdin", "stdout", "--psm", "7", "--oem", "3", "tsv"],
                    input=encoded.tobytes(),
                    capture_output=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    return OCRResult(full_text="", tokens=[], mean_confidence=0.0, backend="tesseract")
                tsv_text = proc.stdout.decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in reader:
                    raw_txt = (row.get("text") or "").strip()
                    if not raw_txt:
                        continue
                    try:
                        conf = float(row.get("conf", -1))
                    except Exception:
                        conf = -1.0
                    if conf < 25.0:
                        continue
                    norm_conf = max(0.0, min(1.0, conf / 100.0))
                    left = float(row.get("left", 0))
                    bw = float(row.get("width", 0))
                    x_min = max(0.0, min(1.0, left / float(max(1, w))))
                    x_max = max(0.0, min(1.0, (left + bw) / float(max(1, w))))
                    tokens.append(OCRToken(text=raw_txt, confidence=norm_conf, x_min_norm=x_min, x_max_norm=x_max))
                    texts.append(raw_txt)
                    confs.append(norm_conf)
            except Exception:
                return OCRResult(full_text="", tokens=[], mean_confidence=0.0, backend="tesseract")

        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return OCRResult(
            full_text=" ".join(texts).strip(),
            tokens=tokens,
            mean_confidence=mean_conf,
            backend="tesseract",
        )
