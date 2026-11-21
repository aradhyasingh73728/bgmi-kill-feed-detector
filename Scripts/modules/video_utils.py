from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    has_audio: bool
    is_vfr: bool
    codec_name: str


def check_ffmpeg_available() -> Tuple[bool, str]:
    """
    Verify that the FFmpeg executable is installed and accessible on PATH.
    Note: Installing 'ffmpeg-python' via pip does NOT install ffmpeg.exe.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return (
            False,
            "FFmpeg executable ('ffmpeg') was not found on your system PATH.\n"
            "Note: 'pip install ffmpeg-python' only installs the Python wrapper, not FFmpeg itself.\n"
            "Install FFmpeg on Windows PowerShell via:\n"
            "  winget install Gyan.FFmpeg\n"
            "Then restart your terminal and verify with: ffmpeg -version",
        )
    return True, ffmpeg_path


def check_ffprobe_available() -> Optional[str]:
    return shutil.which("ffprobe")


def detect_available_hwaccels() -> List[str]:
    """Probe FFmpeg for supported hardware decoders without requiring CUDA."""
    ok, ffmpeg_path = check_ffmpeg_available()
    if not ok:
        return []
    try:
        proc = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return []
        lines = [ln.strip().lower() for ln in proc.stdout.splitlines()]
        accels = [ln for ln in lines if ln and not ln.startswith("hardware")]
        return accels
    except Exception:
        return []


def _parse_fraction(value: str, fallback: float = 30.0) -> float:
    if not value or value in ("0/0", "N/A"):
        return fallback
    try:
        return float(Fraction(value))
    except Exception:
        try:
            return float(value)
        except Exception:
            return fallback


def get_video_metadata(video_path: str) -> VideoMetadata:
    """
    Inspect video metadata safely using ffprobe (when available) + OpenCV verification.
    Handles variable frame rate (VFR) mobile recordings and diverse resolutions.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: '{video_path}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video file '{video_path}'. "
            "Ensure the file is not corrupted and uses a supported codec."
        )

    try:
        ret, sample_frame = cap.read()
        if not ret or sample_frame is None:
            raise RuntimeError(
                f"Video '{video_path}' opened, but no readable frames were found."
            )
        decoded_h, decoded_w = sample_frame.shape[:2]
        cv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    width = decoded_w
    height = decoded_h
    fps = cv_fps if cv_fps > 0 else 30.0
    frame_count = max(0, cv_frames)
    duration_sec = (frame_count / fps) if (frame_count > 0 and fps > 0) else 0.0
    has_audio = False
    is_vfr = False
    codec_name = "unknown"

    ffprobe_path = check_ffprobe_available()
    if ffprobe_path:
        try:
            proc = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                info = json.loads(proc.stdout)
                streams = info.get("streams", [])
                fmt = info.get("format", {})
                for stream in streams:
                    stype = stream.get("codec_type")
                    if stype == "video":
                        codec_name = stream.get("codec_name", codec_name)
                        r_fps = _parse_fraction(stream.get("r_frame_rate", ""), fps)
                        avg_fps = _parse_fraction(stream.get("avg_frame_rate", ""), r_fps)
                        if avg_fps > 0:
                            fps = avg_fps
                        if r_fps > 0 and avg_fps > 0 and abs(r_fps - avg_fps) / max(r_fps, avg_fps) > 0.05:
                            is_vfr = True
                        if stream.get("duration"):
                            try:
                                duration_sec = max(duration_sec, float(stream["duration"]))
                            except ValueError:
                                pass
                    elif stype == "audio":
                        has_audio = True
                if fmt.get("duration"):
                    try:
                        duration_sec = float(fmt["duration"])
                    except ValueError:
                        pass
        except Exception:
            pass

    if duration_sec <= 0 and frame_count > 0 and fps > 0:
        duration_sec = frame_count / fps

    return VideoMetadata(
        path=os.path.abspath(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        has_audio=has_audio,
        is_vfr=is_vfr,
        codec_name=codec_name,
    )


def read_preview_frame(video_path: str, timestamp_sec: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Read a single preview frame in memory without saving anything to disk.
    If timestamp_sec is None, picks ~15% into the video (capped at 10s) to avoid black intros.
    """
    meta = get_video_metadata(video_path)
    if timestamp_sec is None:
        if meta.duration_sec > 4.0:
            timestamp_sec = min(10.0, meta.duration_sec * 0.15)
        else:
            timestamp_sec = 0.0

    timestamp_sec = max(0.0, min(timestamp_sec, max(0.0, meta.duration_sec - 0.1)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for preview: '{video_path}'")

    try:
        if timestamp_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback to frame 0 if seeking failed
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            actual_ts = 0.0
        else:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            actual_ts = (pos_msec / 1000.0) if pos_msec and pos_msec > 0 else timestamp_sec
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read preview frame from '{video_path}'.")
        return frame, actual_ts
    finally:
        cap.release()


def format_seconds_hms(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hrs = int(total // 3600)
    mins = int((total % 3600) // 60)
    secs = int(total % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"
