from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any
import json
import os


@dataclass
class ROIRegion:
    """Represents a rectangular kill-feed Region of Interest in native video coordinates."""
    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int

    def is_valid(self, min_width: int = 40, min_height: int = 16) -> bool:
        return (
            self.width >= min_width
            and self.height >= min_height
            and self.x >= 0
            and self.y >= 0
            and self.source_width > 0
            and self.source_height > 0
            and (self.x + self.width) <= self.source_width
            and (self.y + self.height) <= self.source_height
        )

    def as_normalized(self) -> Tuple[float, float, float, float]:
        """Return (nx, ny, nw, nh) normalized to [0, 1] for resolution-independent transfer."""
        return (
            self.x / float(self.source_width),
            self.y / float(self.source_height),
            self.width / float(self.source_width),
            self.height / float(self.source_height),
        )

    def map_to_resolution(self, target_width: int, target_height: int) -> "ROIRegion":
        """Map normalized ROI accurately onto another video's native resolution."""
        if target_width == self.source_width and target_height == self.source_height:
            return ROIRegion(
                x=self.x,
                y=self.y,
                width=self.width,
                height=self.height,
                source_width=target_width,
                source_height=target_height,
            )
        nx, ny, nw, nh = self.as_normalized()
        rx = max(0, min(target_width - 1, int(round(nx * target_width))))
        ry = max(0, min(target_height - 1, int(round(ny * target_height))))
        rw = max(1, min(target_width - rx, int(round(nw * target_width))))
        rh = max(1, min(target_height - ry, int(round(nh * target_height))))
        return ROIRegion(
            x=rx,
            y=ry,
            width=rw,
            height=rh,
            source_width=target_width,
            source_height=target_height,
        )

    def clamp(self) -> "ROIRegion":
        """Clamp ROI bounds strictly inside source_width x source_height."""
        x = max(0, min(self.source_width - 1, int(self.x)))
        y = max(0, min(self.source_height - 1, int(self.y)))
        w = max(1, min(self.source_width - x, int(self.width)))
        h = max(1, min(self.source_height - y, int(self.height)))
        return ROIRegion(
            x=x,
            y=y,
            width=w,
            height=h,
            source_width=self.source_width,
            source_height=self.source_height,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionConfig:
    """Configuration for conservative BGMI kill-feed detection and highlight clipping."""
    # Sampling & timing
    sample_fps: float = 3.0
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None

    # Conservative detection thresholds
    confidence_threshold: float = 0.65
    min_confirm_frames: int = 2
    require_ocr_if_available: bool = True

    # Optional player-focused filtering (if user enters their BGMI IGN)
    player_ign: str = ""
    only_player_kills: bool = False
    prefer_highlighted_rows: bool = False

    # Temporal & content-aware deduplication
    disappear_grace_sec: float = 1.6
    reappear_cooldown_sec: float = 4.5
    text_similarity_threshold: float = 0.70
    visual_similarity_threshold: float = 0.76

    # Clipping parameters
    pre_event_buffer_sec: float = 8.0
    post_event_buffer_sec: float = 6.0
    merge_gap_sec: float = 4.0
    preserve_audio: bool = True
    fast_stream_copy: bool = False

    # Processing & debugging
    use_gpu_if_available: bool = False
    save_debug_roi: bool = True
    calibration_dir: str = ""
    ocr_backend: str = "auto"  # "auto", "rapidocr", "tesseract", "none"

    def validate(self) -> None:
        if self.sample_fps <= 0 or self.sample_fps > 60:
            raise ValueError("sample_fps must be between 0 (exclusive) and 60.")
        if not (0.30 <= self.confidence_threshold <= 0.99):
            raise ValueError("confidence_threshold must be between 0.30 and 0.99.")
        if self.min_confirm_frames < 1 or self.min_confirm_frames > 10:
            raise ValueError("min_confirm_frames must be between 1 and 10.")
        if self.pre_event_buffer_sec < 0 or self.post_event_buffer_sec < 0:
            raise ValueError("Clip buffers must be >= 0 seconds.")
        if self.start_time_sec is not None and self.start_time_sec < 0:
            raise ValueError("start_time_sec cannot be negative.")
        if (
            self.start_time_sec is not None
            and self.end_time_sec is not None
            and self.end_time_sec <= self.start_time_sec
        ):
            raise ValueError("end_time_sec must be greater than start_time_sec.")

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class KillEvent:
    """A single confirmed, deduplicated BGMI kill-feed event."""
    event_id: int
    timestamp_sec: float
    frame_index: int
    confidence: float
    ocr_text: str
    killer: str
    victim: str
    weapon_or_action: str
    row_bbox_in_roi: Tuple[int, int, int, int]
    is_highlighted: bool
    screenshot_path: str = ""
    debug_roi_path: str = ""
    notes: str = ""

    def formatted_timestamp(self) -> str:
        total = max(0, int(round(self.timestamp_sec)))
        hrs = total // 3600
        mins = (total % 3600) // 60
        secs = total % 60
        return f"{hrs:02d}-{mins:02d}-{secs:02d}"

    def formatted_hms(self) -> str:
        total = max(0.0, float(self.timestamp_sec))
        hrs = int(total // 3600)
        mins = int((total % 3600) // 60)
        secs = total % 60.0
        return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"
