from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional

from modules.config import DetectionConfig
from modules.createDir import SessionPaths
from modules.timeGrouping import ClipSegment
from modules.video_utils import VideoMetadata, check_ffmpeg_available, detect_available_hwaccels


ClipProgressCallback = Callable[[int, int, str], None]


@dataclass
class ClipResult:
    clip_index: int
    output_path: str
    start_sec: float
    end_sec: float
    success: bool
    error_message: str = ""


def _build_ffmpeg_command(
    ffmpeg_path: str,
    video_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: float,
    has_audio: bool,
    preserve_audio: bool,
    fast_stream_copy: bool,
    hwaccel: Optional[str] = None,
) -> List[str]:
    cmd: List[str] = [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        cmd.extend(["-hwaccel", hwaccel])

    # Place -ss before -i for fast seek, and re-encode (by default) for frame-accurate start without black frames
    cmd.extend(["-ss", f"{start_sec:.3f}", "-i", video_path, "-t", f"{duration_sec:.3f}"])

    if fast_stream_copy:
        cmd.extend(["-c:v", "copy"])
        if preserve_audio and has_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.append("-an")
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"])
        if preserve_audio and has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.append("-an")

    cmd.extend(["-movflags", "+faststart", output_path])
    return cmd


def generate_highlight_clips(
    video_meta: VideoMetadata,
    clip_segments: List[ClipSegment],
    session_paths: SessionPaths,
    config: DetectionConfig,
    logger: logging.Logger,
    progress_callback: Optional[ClipProgressCallback] = None,
) -> List[ClipResult]:
    """
    Generate highlight MP4 clips strictly from confirmed, grouped kill-feed intervals.
    Never creates clips when `clip_segments` is empty.
    Preserves audio by default and never passes negative timestamps or exceeds video duration.
    """
    if not clip_segments:
        logger.info("No confirmed kill-feed events; skipping highlight clip generation.")
        return []

    ffmpeg_ok, ffmpeg_info = check_ffmpeg_available()
    if not ffmpeg_ok:
        raise RuntimeError(ffmpeg_info)
    ffmpeg_path = ffmpeg_info

    os.makedirs(session_paths.clips_dir, exist_ok=True)

    selected_hwaccel: Optional[str] = None
    if config.use_gpu_if_available and not config.fast_stream_copy:
        available = detect_available_hwaccels()
        for preferred in ("cuda", "d3d11va", "qsv", "dxva2", "vaapi"):
            if preferred in available:
                selected_hwaccel = preferred
                break
        if selected_hwaccel:
            logger.info("Using optional FFmpeg hwaccel decoder: %s", selected_hwaccel)
        else:
            logger.info("No optional FFmpeg hwaccel detected; running on CPU.")

    results: List[ClipResult] = []
    total = len(clip_segments)

    for idx, seg in enumerate(clip_segments, start=1):
        safe_start = max(0.0, float(seg.start_sec))
        max_end = float(video_meta.duration_sec) if video_meta.duration_sec > 0 else float(seg.end_sec)
        safe_end = min(max_end, float(seg.end_sec))
        duration = safe_end - safe_start

        out_path = os.path.join(session_paths.clips_dir, f"highlight_{idx:03d}.mp4")
        if duration <= 0.1:
            msg = f"Clip {idx} skipped: invalid duration ({duration:.2f}s)."
            logger.warning(msg)
            results.append(ClipResult(idx, out_path, safe_start, safe_end, False, msg))
            continue

        if progress_callback:
            progress_callback(idx, total, f"Rendering highlight_{idx:03d}.mp4 ({safe_start:.1f}s - {safe_end:.1f}s)")

        cmd = _build_ffmpeg_command(
            ffmpeg_path=ffmpeg_path,
            video_path=video_meta.path,
            output_path=out_path,
            start_sec=safe_start,
            duration_sec=duration,
            has_audio=video_meta.has_audio,
            preserve_audio=config.preserve_audio,
            fast_stream_copy=config.fast_stream_copy,
            hwaccel=selected_hwaccel,
        )

        proc = subprocess.run(cmd, capture_output=True, text=True)

        # If optional GPU hwaccel failed on this system/codec, automatically retry on CPU
        if proc.returncode != 0 and selected_hwaccel is not None:
            logger.warning(
                "FFmpeg hwaccel '%s' failed for clip %d (%s); retrying with CPU...",
                selected_hwaccel,
                idx,
                proc.stderr.strip(),
            )
            selected_hwaccel = None
            cpu_cmd = _build_ffmpeg_command(
                ffmpeg_path=ffmpeg_path,
                video_path=video_meta.path,
                output_path=out_path,
                start_sec=safe_start,
                duration_sec=duration,
                has_audio=video_meta.has_audio,
                preserve_audio=config.preserve_audio,
                fast_stream_copy=config.fast_stream_copy,
                hwaccel=None,
            )
            proc = subprocess.run(cpu_cmd, capture_output=True, text=True)

        if proc.returncode != 0 or not os.path.isfile(out_path):
            err = (proc.stderr or proc.stdout or "Unknown FFmpeg error").strip()
            logger.error("Failed to create clip %s: %s", out_path, err)
            results.append(ClipResult(idx, out_path, safe_start, safe_end, False, err))
        else:
            logger.info(
                "Created %s [%.2fs -> %.2fs, %.2fs] for events %s",
                os.path.basename(out_path),
                safe_start,
                safe_end,
                duration,
                seg.event_ids,
            )
            results.append(ClipResult(idx, out_path, safe_start, safe_end, True, ""))

    return results
