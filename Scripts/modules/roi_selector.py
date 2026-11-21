from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from modules.config import ROIRegion
from modules.video_utils import read_preview_frame


class ROISelectionError(RuntimeError):
    """Raised when the user cancels ROI selection or selects an invalid region."""


def compute_display_scale(
    frame_width: int,
    frame_height: int,
    max_display_width: int = 1280,
    max_display_height: int = 720,
) -> float:
    """Compute a uniform scale factor so high-res videos fit comfortably on any screen."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    scale_w = max_display_width / float(frame_width)
    scale_h = max_display_height / float(frame_height)
    return min(1.0, scale_w, scale_h)


def map_display_rect_to_native(
    rect: Tuple[int, int, int, int],
    scale: float,
    native_width: int,
    native_height: int,
) -> ROIRegion:
    """Convert a display-space rectangle (dx, dy, dw, dh) back to native video coordinates."""
    dx, dy, dw, dh = rect
    if dw <= 0 or dh <= 0:
        raise ROISelectionError(
            "No kill-feed region was selected (selection cancelled or zero-sized). "
            "Processing cannot continue without a valid kill-feed region."
        )

    inv = 1.0 / scale if scale > 0 else 1.0
    nx = int(round(dx * inv))
    ny = int(round(dy * inv))
    nw = int(round(dw * inv))
    nh = int(round(dh * inv))

    roi = ROIRegion(
        x=nx,
        y=ny,
        width=nw,
        height=nh,
        source_width=native_width,
        source_height=native_height,
    ).clamp()

    if not roi.is_valid(min_width=40, min_height=16):
        raise ROISelectionError(
            f"Selected kill-feed region ({roi.width}x{roi.height}px in native "
            f"{native_width}x{native_height} video) is too small or invalid. "
            "Please draw a box covering the full BGMI kill-feed area (at least 40x16px)."
        )

    return roi


def select_kill_feed_roi(
    video_path: str,
    preview_time_sec: Optional[float] = None,
    max_display_width: int = 1280,
    max_display_height: int = 720,
) -> ROIRegion:
    """
    Display a preview frame from the video and prompt the user to draw the rectangular
    BGMI kill-feed region using OpenCV's cv2.selectROI().

    Raises ROISelectionError if cancelled (ESC / 'c' / zero box) or invalid.
    """
    frame, actual_ts = read_preview_frame(video_path, preview_time_sec)
    native_h, native_w = frame.shape[:2]

    scale = compute_display_scale(
        native_w,
        native_h,
        max_display_width=max_display_width,
        max_display_height=max_display_height,
    )

    if scale < 1.0:
        disp_w = max(1, int(round(native_w * scale)))
        disp_h = max(1, int(round(native_h * scale)))
        display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        display_frame = frame.copy()

    # Draw non-shifting instruction pill inside bottom-left corner so (0,0) aligns identically
    hint = (
        f"Select BGMI Kill-Feed Region | Video {native_w}x{native_h} @ {actual_ts:.1f}s | "
        "Drag box -> ENTER/SPACE to confirm | ESC/C to cancel"
    )
    dh, dw = display_frame.shape[:2]
    overlay = display_frame.copy()
    bar_h = min(32, max(22, dh // 20))
    cv2.rectangle(overlay, (0, dh - bar_h), (dw, dh), (15, 18, 25), -1)
    cv2.addWeighted(overlay, 0.72, display_frame, 0.28, 0, display_frame)
    cv2.putText(
        display_frame,
        hint,
        (10, dh - max(8, bar_h // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 244, 248),
        1,
        cv2.LINE_AA,
    )

    window_name = "PEACE - Select BGMI Kill Feed Region (ENTER=Confirm, C/ESC=Cancel)"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        rect = cv2.selectROI(window_name, display_frame, showCrosshair=True, fromCenter=False)
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        cv2.waitKey(1)

    rx, ry, rw, rh = [int(v) for v in rect]
    return map_display_rect_to_native((rx, ry, rw, rh), scale, native_w, native_h)


def render_roi_preview_image(frame: np.ndarray, roi: ROIRegion) -> np.ndarray:
    """Return a copy of frame with the validated ROI highlighted for confirmation preview."""
    vis = frame.copy()
    cv2.rectangle(
        vis,
        (roi.x, roi.y),
        (roi.x + roi.width, roi.y + roi.height),
        (0, 220, 120),
        2,
    )
    label = f"Kill-Feed ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}"
    ty = max(20, roi.y - 8)
    cv2.putText(
        vis,
        label,
        (roi.x, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 120),
        2,
        cv2.LINE_AA,
    )
    return vis
