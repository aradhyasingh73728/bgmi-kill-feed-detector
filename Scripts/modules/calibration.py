from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from modules.config import ROIRegion
from modules.video_utils import read_preview_frame


DEFAULT_CALIBRATION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration", "samples")
)


@dataclass
class CalibrationSample:
    name: str
    path: str
    image_gray: np.ndarray
    image_bgr: np.ndarray


@dataclass
class CalibrationMatch:
    sample_name: str
    score: float
    scale: float


class CalibrationManager:
    """
    Manages user-calibrated BGMI reference samples (e.g., weapon icons, skull/knock
    indicators, or kill-feed strip crops extracted from real BGMI footage).

    Important:
    No synthetic or non-BGMI templates are loaded by default. The user must supply
    real BGMI crops via the calibration workflow to enable template-assisted scoring.
    """

    def __init__(self, samples_dir: Optional[str] = None) -> None:
        self.samples_dir = os.path.abspath(samples_dir or DEFAULT_CALIBRATION_DIR)
        os.makedirs(self.samples_dir, exist_ok=True)
        self.samples: List[CalibrationSample] = []
        self.reload()

    def reload(self) -> int:
        self.samples.clear()
        if not os.path.isdir(self.samples_dir):
            return 0

        for fname in sorted(os.listdir(self.samples_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            fpath = os.path.join(self.samples_dir, fname)
            img_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img_bgr is None or img_bgr.size == 0:
                continue
            h, w = img_bgr.shape[:2]
            if h < 6 or w < 6:
                continue
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            self.samples.append(
                CalibrationSample(
                    name=os.path.splitext(fname)[0],
                    path=fpath,
                    image_gray=img_gray,
                    image_bgr=img_bgr,
                )
            )
        return len(self.samples)

    @property
    def is_calibrated(self) -> bool:
        return len(self.samples) > 0

    def match_row(
        self,
        row_bgr: np.ndarray,
        scales: Tuple[float, ...] = (0.70, 0.85, 1.0, 1.15, 1.35),
    ) -> Optional[CalibrationMatch]:
        """
        Perform multi-scale normalized cross-correlation against user-calibrated BGMI
        reference samples inside a candidate row.
        """
        if not self.samples or row_bgr is None or row_bgr.size == 0:
            return None

        row_gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
        rh, rw = row_gray.shape[:2]

        # Also compare edge maps for weapon silhouette resilience across background colors
        row_edges = cv2.Canny(row_gray, 50, 150)

        best_match: Optional[CalibrationMatch] = None
        for sample in self.samples:
            sh, sw = sample.image_gray.shape[:2]
            sample_edges = cv2.Canny(sample.image_gray, 50, 150)

            for sc in scales:
                tw = int(round(sw * sc))
                th = int(round(sh * sc))
                if tw < 6 or th < 6 or tw > rw or th > rh:
                    continue

                tmpl_gray = cv2.resize(sample.image_gray, (tw, th), interpolation=cv2.INTER_AREA)
                res_gray = cv2.matchTemplate(row_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
                gray_score = float(cv2.minMaxLoc(res_gray)[1])

                tmpl_edges = cv2.resize(sample_edges, (tw, th), interpolation=cv2.INTER_AREA)
                res_edges = cv2.matchTemplate(row_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
                edge_score = float(cv2.minMaxLoc(res_edges)[1])

                combined = max(gray_score, 0.6 * gray_score + 0.4 * edge_score)
                if best_match is None or combined > best_match.score:
                    best_match = CalibrationMatch(
                        sample_name=sample.name,
                        score=max(0.0, min(1.0, combined)),
                        scale=sc,
                    )
        return best_match

    def capture_sample_interactively(
        self,
        video_path: str,
        timestamp_sec: float,
        label: str = "bgmi_sample",
        roi: Optional[ROIRegion] = None,
    ) -> Optional[str]:
        """
        Open an OpenCV selection window on a frame at `timestamp_sec` so the user can
        crop a real BGMI weapon icon or kill-feed strip and save it as a calibration sample.
        """
        frame, _ = read_preview_frame(video_path, timestamp_sec)
        if roi is not None:
            r = roi.clamp()
            source = frame[r.y : r.y + r.height, r.x : r.x + r.width].copy()
            # Upscale small ROI slightly so cropping icons is comfortable
            scale = 2.0 if source.shape[1] < 600 else 1.0
        else:
            source = frame.copy()
            scale = min(1.0, 1280.0 / float(source.shape[1]))

        disp = (
            cv2.resize(
                source,
                (max(1, int(source.shape[1] * scale)), max(1, int(source.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
            )
            if scale != 1.0
            else source
        )

        win = "PEACE Calibration - Crop BGMI Weapon Icon or Kill Row (ENTER=Save, ESC=Cancel)"
        try:
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            rect = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
        finally:
            try:
                cv2.destroyWindow(win)
            except Exception:
                pass
            cv2.waitKey(1)

        dx, dy, dw, dh = [int(v) for v in rect]
        if dw < 8 or dh < 8:
            return None

        inv = 1.0 / scale
        sx = max(0, int(round(dx * inv)))
        sy = max(0, int(round(dy * inv)))
        sw = max(1, min(source.shape[1] - sx, int(round(dw * inv))))
        sh = max(1, min(source.shape[0] - sy, int(round(dh * inv))))
        crop = source[sy : sy + sh, sx : sx + sw]

        safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label.strip())
        if not safe_label:
            safe_label = "bgmi_sample"

        idx = 1
        out_path = os.path.join(self.samples_dir, f"{safe_label}_{idx:02d}.png")
        while os.path.exists(out_path):
            idx += 1
            out_path = os.path.join(self.samples_dir, f"{safe_label}_{idx:02d}.png")

        cv2.imwrite(out_path, crop)
        self.reload()
        return out_path
