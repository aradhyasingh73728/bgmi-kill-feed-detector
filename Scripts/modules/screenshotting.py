from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from modules.config import KillEvent, ROIRegion


def save_confirmed_event_screenshots(
    event: KillEvent,
    full_frame_bgr: np.ndarray,
    roi_frame_bgr: np.ndarray,
    roi: ROIRegion,
    screenshots_dir: str,
    debug_region_dir: Optional[str] = None,
    save_debug_roi: bool = True,
) -> Tuple[str, str]:
    """
    Save screenshots ONLY after a kill-feed event is confirmed:
      1. Saves the original full gameplay frame to `screenshots/kill_001_00-01-25.png`.
      2. Optionally saves the cropped kill-feed ROI (with detected row box) to `debug/detected_region/`.

    Never saves unconfirmed frames or extracts all frames from the video.
    """
    os.makedirs(screenshots_dir, exist_ok=True)
    ts_tag = event.formatted_timestamp()
    base_name = f"kill_{event.event_id:03d}_{ts_tag}"

    screenshot_path = os.path.join(screenshots_dir, f"{base_name}.png")
    counter = 1
    while os.path.exists(screenshot_path):
        screenshot_path = os.path.join(screenshots_dir, f"{base_name}_{counter}.png")
        counter += 1

    ok = cv2.imwrite(screenshot_path, full_frame_bgr)
    if not ok:
        raise IOError(f"Failed to write confirmed screenshot: '{screenshot_path}'")
    event.screenshot_path = screenshot_path

    debug_roi_path = ""
    if save_debug_roi and debug_region_dir and roi_frame_bgr is not None and roi_frame_bgr.size > 0:
        os.makedirs(debug_region_dir, exist_ok=True)
        annotated_roi = roi_frame_bgr.copy()
        rx, ry, rw, rh = event.row_bbox_in_roi
        cv2.rectangle(annotated_roi, (rx, ry), (rx + rw, ry + rh), (0, 230, 120), 2)
        label = f"#{event.event_id} conf={event.confidence:.2f}"
        cv2.putText(
            annotated_roi,
            label,
            (max(2, rx), max(14, ry - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 230, 120),
            1,
            cv2.LINE_AA,
        )
        debug_roi_path = os.path.join(debug_region_dir, f"{base_name}_roi.png")
        if cv2.imwrite(debug_roi_path, annotated_roi):
            event.debug_roi_path = debug_roi_path

    return screenshot_path, debug_roi_path
