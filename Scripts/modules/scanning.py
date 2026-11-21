from __future__ import annotations

import csv
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from modules.bgmi_detector import BGMIKillFeedDetector, CandidateDetection
from modules.calibration import CalibrationManager
from modules.config import DetectionConfig, KillEvent, ROIRegion
from modules.createDir import SessionPaths
from modules.deduplicator import KillFeedDeduplicator
from modules.ocr_engine import OCREngine
from modules.screenshotting import save_confirmed_event_screenshots
from modules.video_utils import VideoMetadata, check_ffmpeg_available, format_seconds_hms


ProgressCallback = Callable[[float, str, int], None]
CancelCheck = Callable[[], bool]


@dataclass
class ScanSummary:
    events: List[KillEvent]
    frames_read: int
    frames_analyzed: int
    corrupt_frames_skipped: int
    elapsed_sec: float
    ocr_backend: str
    calibrated_samples_count: int
    cancelled: bool = False


def write_events_csv(events: List[KillEvent], csv_path: str) -> None:
    fieldnames = [
        "event_id",
        "timestamp_sec",
        "timestamp_hms",
        "frame_index",
        "confidence",
        "killer",
        "weapon_or_action",
        "victim",
        "ocr_text",
        "is_highlighted",
        "screenshot_file",
        "debug_roi_file",
        "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            writer.writerow(
                {
                    "event_id": ev.event_id,
                    "timestamp_sec": f"{ev.timestamp_sec:.3f}",
                    "timestamp_hms": ev.formatted_hms(),
                    "frame_index": ev.frame_index,
                    "confidence": f"{ev.confidence:.4f}",
                    "killer": ev.killer,
                    "weapon_or_action": ev.weapon_or_action,
                    "victim": ev.victim,
                    "ocr_text": ev.ocr_text,
                    "is_highlighted": str(ev.is_highlighted),
                    "screenshot_file": os.path.basename(ev.screenshot_path) if ev.screenshot_path else "",
                    "debug_roi_file": os.path.basename(ev.debug_roi_path) if ev.debug_roi_path else "",
                    "notes": ev.notes,
                }
            )


def _roi_diff_fingerprint(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 24), interpolation=cv2.INTER_AREA)


def _read_exact(stream, num_bytes: int) -> bytes:
    chunks: List[bytes] = []
    remaining = num_bytes
    while remaining > 0:
        part = stream.read(remaining)
        if not part:
            break
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def _seek_full_frame(cap: cv2.VideoCapture, timestamp_sec: float) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        return frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        return frame
    return None


def scan_video_for_kills(
    video_meta: VideoMetadata,
    roi: ROIRegion,
    config: DetectionConfig,
    session_paths: SessionPaths,
    logger: logging.Logger,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> ScanSummary:
    """
    Stream video frames in-memory (no disk extraction of non-kill frames), analyze only
    the user-selected kill-feed ROI, skip unchanged ROI frames via frame-diff gating,
    deduplicate events over time, and save full-resolution screenshots only when confirmed.
    """
    config.validate()
    t_start = time.monotonic()

    ocr_engine = OCREngine(preferred_backend=config.ocr_backend)
    calibration = CalibrationManager(samples_dir=config.calibration_dir or None)
    detector = BGMIKillFeedDetector(config, ocr_engine, calibration)
    deduplicator = KillFeedDeduplicator(config)

    mapped_roi = roi.map_to_resolution(video_meta.width, video_meta.height).clamp()
    ref_dim = min(video_meta.width, video_meta.height)

    logger.info(
        "Starting BGMI kill-feed scan | video=%s | res=%dx%d | fps=%.2f | duration=%.2fs | vfr=%s",
        video_meta.path,
        video_meta.width,
        video_meta.height,
        video_meta.fps,
        video_meta.duration_sec,
        video_meta.is_vfr,
    )
    logger.info(
        "Selected ROI (native): x=%d, y=%d, w=%d, h=%d",
        mapped_roi.x,
        mapped_roi.y,
        mapped_roi.width,
        mapped_roi.height,
    )
    logger.info(
        "Detector settings: sample_fps=%.2f | conf_thresh=%.2f | confirm_frames=%d | ocr=%s | calibrated_samples=%d",
        config.sample_fps,
        config.confidence_threshold,
        config.min_confirm_frames,
        ocr_engine.backend_name,
        len(calibration.samples),
    )
    if not ocr_engine.is_available:
        logger.warning(
            "No OCR backend active (RapidOCR/Tesseract unavailable). "
            "Falling back to strict visual/calibration detection; install rapidocr-onnxruntime for best accuracy."
        )
    if not calibration.is_calibrated:
        logger.warning(
            "No user-calibrated BGMI reference samples loaded in '%s'. "
            "Detection relies on structural + OCR rules. You can add real BGMI samples via Calibration.",
            calibration.samples_dir,
        )

    scan_start_sec = float(config.start_time_sec or 0.0)
    scan_end_sec = (
        min(video_meta.duration_sec, float(config.end_time_sec))
        if config.end_time_sec is not None and video_meta.duration_sec > 0
        else (float(config.end_time_sec) if config.end_time_sec is not None else video_meta.duration_sec)
    )
    if scan_end_sec <= 0 and video_meta.duration_sec > 0:
        scan_end_sec = video_meta.duration_sec

    frames_read = 0
    frames_analyzed = 0
    corrupt_frames_skipped = 0
    ocr_skipped_diff = 0
    cancelled = False

    prev_roi_fp: Optional[np.ndarray] = None
    prev_candidates: List[CandidateDetection] = []
    last_analyzed_ts: float = scan_start_sec

    def _analyze_roi_step(
        current_ts: float,
        frame_index: int,
        roi_crop: np.ndarray,
        full_frame: Optional[np.ndarray],
        lazy_cap_box: List[Optional[cv2.VideoCapture]],
    ) -> None:
        nonlocal frames_analyzed, ocr_skipped_diff, prev_roi_fp, prev_candidates, last_analyzed_ts

        frames_analyzed += 1
        last_analyzed_ts = current_ts
        cur_fp = _roi_diff_fingerprint(roi_crop)
        if prev_roi_fp is not None:
            mad = float(np.mean(np.abs(cur_fp.astype(np.int16) - prev_roi_fp.astype(np.int16))))
        else:
            mad = 999.0

        if mad < 1.8:
            candidates = prev_candidates
            ocr_skipped_diff += 1
        else:
            # Fast bright-pixel check: skip heavy segmentation/OCR on completely dark/flat crops
            if not prev_candidates and float(np.percentile(cur_fp, 97)) < 75.0:
                candidates = []
            else:
                candidates = detector.detect_in_roi_crop(roi_crop, ref_dim=ref_dim)
            prev_roi_fp = cur_fp
            prev_candidates = candidates

        newly_confirmed = deduplicator.update(
            timestamp_sec=current_ts,
            frame_index=frame_index,
            candidates=candidates,
            full_frame_bgr=full_frame,
            roi_frame_bgr=roi_crop,
        )

        for ev in newly_confirmed:
            frames_pair = deduplicator.get_track_frames(ev.event_id)
            best_full: Optional[np.ndarray] = None
            best_roi: np.ndarray = roi_crop
            if frames_pair is not None:
                best_full, best_roi = frames_pair

            if best_full is None or best_full.size == 0:
                if lazy_cap_box[0] is None:
                    lazy_cap_box[0] = cv2.VideoCapture(video_meta.path)
                best_ts = deduplicator.get_track_timestamp(ev.event_id)
                seek_ts = best_ts if best_ts is not None else ev.timestamp_sec
                if lazy_cap_box[0].isOpened():
                    best_full = _seek_full_frame(lazy_cap_box[0], seek_ts)

            if best_full is not None and best_full.size > 0:
                save_confirmed_event_screenshots(
                    event=ev,
                    full_frame_bgr=best_full,
                    roi_frame_bgr=best_roi,
                    roi=mapped_roi,
                    screenshots_dir=session_paths.screenshots_dir,
                    debug_region_dir=session_paths.debug_region_dir,
                    save_debug_roi=config.save_debug_roi,
                )
            logger.info(
                "CONFIRMED KILL #%03d @ %s (%.2fs) | conf=%.2f | text='%s' | saved='%s'",
                ev.event_id,
                format_seconds_hms(ev.timestamp_sec),
                ev.timestamp_sec,
                ev.confidence,
                ev.ocr_text,
                os.path.basename(ev.screenshot_path),
            )

        if progress_callback:
            total_span = scan_end_sec if scan_end_sec > 0 else max(current_ts, 1.0)
            progress_callback(
                current_ts,
                total_span,
                len(deduplicator.confirmed_events),
                f"Scanning {format_seconds_hms(current_ts)} / {format_seconds_hms(total_span)} | "
                f"Confirmed Kills: {len(deduplicator.confirmed_events)}",
            )

    # Use FFmpeg C-level ROI crop pipe for fast streaming when FPS downsampling applies
    ff_ok, ff_exe = check_ffmpeg_available()
    use_ffmpeg_pipe = bool(
        ff_ok
        and video_meta.duration_sec > 4.0
        and video_meta.fps > config.sample_fps * 1.4
        and mapped_roi.width >= 40
        and mapped_roi.height >= 16
    )

    used_ffmpeg_ok = False
    if use_ffmpeg_pipe:
        lazy_cap_box: List[Optional[cv2.VideoCapture]] = [None]
        frame_bytes = mapped_roi.width * mapped_roi.height * 3
        vf = (
            f"fps={config.sample_fps:.6g},"
            f"crop={mapped_roi.width}:{mapped_roi.height}:{mapped_roi.x}:{mapped_roi.y}"
        )
        cmd: List[str] = [ff_exe, "-hide_banner", "-loglevel", "error"]
        if scan_start_sec > 0:
            cmd.extend(["-ss", f"{scan_start_sec:.3f}"])
        cmd.extend(["-i", video_meta.path])
        if scan_end_sec > 0 and (video_meta.duration_sec <= 0 or scan_end_sec < video_meta.duration_sec - 0.05):
            cmd.extend(["-t", f"{max(0.1, scan_end_sec - scan_start_sec):.3f}"])
        cmd.extend(["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])

        logger.info("Using fast FFmpeg ROI crop pipe (sample_fps=%.2f).", config.sample_fps)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=max(frame_bytes * 4, 65536),
            creationflags=creationflags,
        )
        try:
            idx = 0
            assert proc.stdout is not None
            while True:
                if cancel_check and cancel_check():
                    cancelled = True
                    logger.info("Scan cancelled by user.")
                    proc.terminate()
                    break

                raw = _read_exact(proc.stdout, frame_bytes)
                if len(raw) < frame_bytes:
                    break

                roi_crop = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (mapped_roi.height, mapped_roi.width, 3)
                )
                current_ts = scan_start_sec + (idx + 0.5) / float(config.sample_fps)
                if scan_end_sec > 0 and current_ts > scan_end_sec + 0.15:
                    break
                frame_idx = int(round(current_ts * max(1.0, video_meta.fps)))
                frames_read += 1
                idx += 1

                _analyze_roi_step(current_ts, frame_idx, roi_crop, None, lazy_cap_box)
        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.stderr is not None:
                    proc.stderr.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            if lazy_cap_box[0] is not None:
                lazy_cap_box[0].release()

        if frames_analyzed > 0 or cancelled:
            used_ffmpeg_ok = True
        else:
            err_msg = ""
            if proc.stderr is not None:
                try:
                    err_msg = proc.stderr.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
            logger.warning("FFmpeg ROI pipe produced 0 frames (%s); falling back to OpenCV.", err_msg)

    if not used_ffmpeg_ok and not cancelled:
        cap = cv2.VideoCapture(video_meta.path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file for scanning: '{video_meta.path}'")

        if scan_start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, scan_start_sec * 1000.0)

        sample_interval_sec = 1.0 / float(config.sample_fps)
        next_sample_ts = scan_start_sec
        last_ts = max(0.0, scan_start_sec - 1e-3)
        fallback_frame_idx = int(round(scan_start_sec * video_meta.fps))
        consecutive_failures = 0
        lazy_cap_box = [None]

        try:
            while True:
                if cancel_check and cancel_check():
                    cancelled = True
                    logger.info("Scan cancelled by user.")
                    break

                grabbed = cap.grab()
                if not grabbed:
                    consecutive_failures += 1
                    if consecutive_failures <= 3:
                        corrupt_frames_skipped += 1
                        continue
                    break

                consecutive_failures = 0
                frames_read += 1
                fallback_frame_idx += 1

                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if pos_msec > 0.0:
                    current_ts = pos_msec / 1000.0
                else:
                    current_ts = fallback_frame_idx / max(1.0, video_meta.fps)

                if current_ts < last_ts:
                    current_ts = last_ts + (1.0 / max(1.0, video_meta.fps))
                last_ts = current_ts

                if scan_end_sec > 0 and current_ts > scan_end_sec:
                    break

                if current_ts + 1e-4 < next_sample_ts:
                    continue

                while next_sample_ts <= current_ts:
                    next_sample_ts += sample_interval_sec

                ret, frame = cap.retrieve()
                if not ret or frame is None or frame.size == 0:
                    corrupt_frames_skipped += 1
                    logger.warning("Skipped unreadable/corrupt frame around %.2fs.", current_ts)
                    continue

                fh, fw = frame.shape[:2]
                cur_roi = roi.map_to_resolution(fw, fh).clamp()
                roi_crop = frame[
                    cur_roi.y : cur_roi.y + cur_roi.height,
                    cur_roi.x : cur_roi.x + cur_roi.width,
                ]
                _analyze_roi_step(current_ts, fallback_frame_idx, roi_crop, frame, lazy_cap_box)
        finally:
            cap.release()

    if not cancelled and frames_analyzed > 0:
        flushed_eof = deduplicator.flush_end_of_stream(last_analyzed_ts)
        if flushed_eof:
            eof_cap: Optional[cv2.VideoCapture] = None
            try:
                for ev in flushed_eof:
                    frames_pair = deduplicator.get_track_frames(ev.event_id)
                    best_full: Optional[np.ndarray] = None
                    best_roi: Optional[np.ndarray] = None
                    if frames_pair is not None:
                        best_full, best_roi = frames_pair
                    if best_full is None or best_full.size == 0:
                        if eof_cap is None:
                            eof_cap = cv2.VideoCapture(video_meta.path)
                        seek_ts = deduplicator.get_track_timestamp(ev.event_id) or ev.timestamp_sec
                        if eof_cap.isOpened():
                            best_full = _seek_full_frame(eof_cap, seek_ts)
                    if best_full is not None and best_full.size > 0 and best_roi is not None:
                        save_confirmed_event_screenshots(
                            event=ev,
                            full_frame_bgr=best_full,
                            roi_frame_bgr=best_roi,
                            roi=mapped_roi,
                            screenshots_dir=session_paths.screenshots_dir,
                            debug_region_dir=session_paths.debug_region_dir,
                            save_debug_roi=config.save_debug_roi,
                        )
            finally:
                if eof_cap is not None:
                    eof_cap.release()

    events = sorted(deduplicator.confirmed_events, key=lambda e: e.timestamp_sec)
    write_events_csv(events, session_paths.events_csv)
    elapsed = time.monotonic() - t_start

    logger.info(
        "Scan finished in %.2fs | frames_read=%d | analyzed=%d | diff_skipped=%d | skipped_corrupt=%d | confirmed_kills=%d",
        elapsed,
        frames_read,
        frames_analyzed,
        ocr_skipped_diff,
        corrupt_frames_skipped,
        len(events),
    )

    return ScanSummary(
        events=events,
        frames_read=frames_read,
        frames_analyzed=frames_analyzed,
        corrupt_frames_skipped=corrupt_frames_skipped,
        elapsed_sec=elapsed,
        ocr_backend=ocr_engine.backend_name,
        calibrated_samples_count=len(calibration.samples),
        cancelled=cancelled,
    )


def scan_roi_frame_sequence(
    roi_frames: Sequence[Tuple[float, str]],
    video_width: int,
    video_height: int,
    duration_sec: float,
    roi: ROIRegion,
    config: DetectionConfig,
    session_paths: SessionPaths,
    logger: logging.Logger,
    progress_callback: Optional[ProgressCallback] = None,
) -> ScanSummary:
    """
    Scan an ordered sequence of pre-cropped ROI image files (e.g., sampled directly in the browser
    from a huge 30-min HD video without uploading the full multi-GB video file).
    """
    config.validate()
    t_start = time.monotonic()

    ocr_engine = OCREngine(preferred_backend=config.ocr_backend)
    calibration = CalibrationManager(samples_dir=config.calibration_dir or None)
    detector = BGMIKillFeedDetector(config, ocr_engine, calibration)
    deduplicator = KillFeedDeduplicator(config)

    ref_dim = min(max(1, video_width), max(1, video_height))
    mapped_roi = roi.map_to_resolution(max(1, video_width), max(1, video_height)).clamp()

    logger.info(
        "Starting Fast ROI-Only scan | crops=%d | source_res=%dx%d | duration=%.2fs | ocr=%s",
        len(roi_frames),
        video_width,
        video_height,
        duration_sec,
        ocr_engine.backend_name,
    )

    frames_read = 0
    frames_analyzed = 0
    corrupt_frames_skipped = 0
    prev_roi_fp: Optional[np.ndarray] = None
    prev_candidates: List[CandidateDetection] = []
    total_span = max(duration_sec, roi_frames[-1][0] if roi_frames else 1.0, 1.0)

    for idx, (ts_sec, img_path) in enumerate(roi_frames):
        frames_read += 1
        roi_crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if roi_crop is None or roi_crop.size == 0:
            corrupt_frames_skipped += 1
            continue

        frames_analyzed += 1
        cur_fp = _roi_diff_fingerprint(roi_crop)
        if prev_roi_fp is not None:
            mad = float(np.mean(np.abs(cur_fp.astype(np.int16) - prev_roi_fp.astype(np.int16))))
        else:
            mad = 999.0

        if mad < 1.8:
            candidates = prev_candidates
        else:
            if not prev_candidates and float(np.percentile(cur_fp, 97)) < 75.0:
                candidates = []
            else:
                candidates = detector.detect_in_roi_crop(roi_crop, ref_dim=ref_dim)
            prev_roi_fp = cur_fp
            prev_candidates = candidates

        frame_idx = int(round(ts_sec * 30.0))
        newly_confirmed = deduplicator.update(
            timestamp_sec=float(ts_sec),
            frame_index=frame_idx,
            candidates=candidates,
            full_frame_bgr=None,
            roi_frame_bgr=roi_crop,
        )

        for ev in newly_confirmed:
            frames_pair = deduplicator.get_track_frames(ev.event_id)
            best_roi = frames_pair[1] if frames_pair is not None else roi_crop
            # Save debug ROI crop (full-frame PNG is captured locally in browser when full video wasn't uploaded)
            if config.save_debug_roi and best_roi is not None and best_roi.size > 0:
                os.makedirs(session_paths.debug_region_dir, exist_ok=True)
                annotated = best_roi.copy()
                rx, ry, rw, rh = ev.row_bbox_in_roi
                cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 230, 120), 2)
                debug_path = os.path.join(
                    session_paths.debug_region_dir,
                    f"kill_{ev.event_id:03d}_{ev.formatted_timestamp()}_roi.png",
                )
                if cv2.imwrite(debug_path, annotated):
                    ev.debug_roi_path = debug_path

        if progress_callback:
            progress_callback(
                float(idx + 1),
                float(max(1, len(roi_frames))),
                len(deduplicator.confirmed_events),
                f"Scanning ROI crop {idx + 1}/{len(roi_frames)} ({format_seconds_hms(ts_sec)}) | "
                f"Confirmed Kills: {len(deduplicator.confirmed_events)}",
            )

    eof_roi_ts = roi_frames[-1][0] if roi_frames else duration_sec
    for ev in deduplicator.flush_end_of_stream(eof_roi_ts):
        frames_pair = deduplicator.get_track_frames(ev.event_id)
        best_roi = frames_pair[1] if frames_pair is not None else None
        if config.save_debug_roi and best_roi is not None and best_roi.size > 0:
            os.makedirs(session_paths.debug_region_dir, exist_ok=True)
            annotated = best_roi.copy()
            rx, ry, rw, rh = ev.row_bbox_in_roi
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 230, 120), 2)
            debug_path = os.path.join(
                session_paths.debug_region_dir,
                f"kill_{ev.event_id:03d}_{ev.formatted_timestamp()}_roi.png",
            )
            if cv2.imwrite(debug_path, annotated):
                ev.debug_roi_path = debug_path

    events = sorted(deduplicator.confirmed_events, key=lambda e: e.timestamp_sec)
    write_events_csv(events, session_paths.events_csv)
    elapsed = time.monotonic() - t_start

    logger.info(
        "ROI-only scan finished in %.2fs | crops=%d | confirmed_kills=%d",
        elapsed,
        frames_analyzed,
        len(events),
    )

    return ScanSummary(
        events=events,
        frames_read=frames_read,
        frames_analyzed=frames_analyzed,
        corrupt_frames_skipped=corrupt_frames_skipped,
        elapsed_sec=elapsed,
        ocr_backend=ocr_engine.backend_name,
        calibrated_samples_count=len(calibration.samples),
        cancelled=False,
    )
