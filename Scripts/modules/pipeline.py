from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from modules.clipping import ClipResult, generate_highlight_clips
from modules.config import DetectionConfig, KillEvent, ROIRegion
from modules.createDir import SessionPaths, configure_session_logger, create_session_directories
from modules.scanning import ScanSummary, scan_roi_frame_sequence, scan_video_for_kills
from modules.timeGrouping import ClipSegment, group_events_into_clips
from modules.video_utils import VideoMetadata, check_ffmpeg_available, format_seconds_hms, get_video_metadata


StatusCallback = Callable[[float, str, int], None]
CancelCheck = Callable[[], bool]


@dataclass
class PipelineResult:
    session_paths: SessionPaths
    video_meta: VideoMetadata
    roi: ROIRegion
    scan_summary: ScanSummary
    clip_segments: List[ClipSegment]
    clip_results: List[ClipResult]


def write_session_summary(
    session_paths: SessionPaths,
    video_meta: VideoMetadata,
    roi: ROIRegion,
    config: DetectionConfig,
    scan_summary: ScanSummary,
    clip_segments: List[ClipSegment],
    clip_results: List[ClipResult],
) -> None:
    lines = [
        "================================================================",
        "PEACE — BGMI Kill Feed Detection & Highlight Clipping Summary",
        "================================================================",
        f"Source Video       : {video_meta.path}",
        f"Resolution         : {video_meta.width}x{video_meta.height}",
        f"Video FPS / Codec  : {video_meta.fps:.2f} fps ({video_meta.codec_name}), VFR={video_meta.is_vfr}",
        f"Video Duration     : {format_seconds_hms(video_meta.duration_sec)} ({video_meta.duration_sec:.2f}s)",
        f"Audio Stream       : {'Present' if video_meta.has_audio else 'None'}",
        f"Selected ROI       : x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height} (native px)",
        f"OCR Backend        : {scan_summary.ocr_backend}",
        f"Calibrated Samples : {scan_summary.calibrated_samples_count}",
        f"Sampling FPS       : {config.sample_fps:.2f}",
        f"Confidence Thresh  : {config.confidence_threshold:.2f}",
        f"Confirm Frames     : {config.min_confirm_frames}",
        f"Clip Buffers       : -{config.pre_event_buffer_sec:.1f}s / +{config.post_event_buffer_sec:.1f}s",
        f"Frames Read        : {scan_summary.frames_read}",
        f"Frames Analyzed    : {scan_summary.frames_analyzed}",
        f"Corrupt Skipped    : {scan_summary.corrupt_frames_skipped}",
        f"Scan Elapsed       : {scan_summary.elapsed_sec:.2f}s",
        f"Cancelled Early    : {scan_summary.cancelled}",
        "",
        f"Confirmed Kill-Feed Events: {len(scan_summary.events)}",
        "----------------------------------------------------------------",
    ]

    if not scan_summary.events:
        lines.append("No confirmed BGMI kill-feed events were detected.")
        lines.append("No kill screenshots or highlight clips were generated.")
    else:
        for ev in scan_summary.events:
            lines.append(
                f"  #{ev.event_id:03d} | {ev.formatted_hms()} ({ev.timestamp_sec:.2f}s) | "
                f"conf={ev.confidence:.2f} | killer='{ev.killer}' | action='{ev.weapon_or_action}' | "
                f"victim='{ev.victim}' | raw='{ev.ocr_text}'"
            )
            if ev.screenshot_path:
                lines.append(f"         screenshot: {os.path.basename(ev.screenshot_path)}")

    lines.append("")
    lines.append(f"Generated Highlight Clips: {sum(1 for r in clip_results if r.success)} / {len(clip_results)}")
    lines.append("----------------------------------------------------------------")
    for res in clip_results:
        status = "OK" if res.success else f"FAILED ({res.error_message})"
        lines.append(
            f"  {os.path.basename(res.output_path)} | {res.start_sec:.2f}s -> {res.end_sec:.2f}s | {status}"
        )

    lines.append("")
    if scan_summary.calibrated_samples_count == 0:
        lines.append(
            "NOTE: No user-calibrated BGMI reference samples were loaded for this session. "
            "Add real BGMI kill-feed crops in Scripts/calibration/samples/ for improved accuracy."
        )

    with open(session_paths.summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_bgmi_pipeline(
    video_path: str,
    roi: ROIRegion,
    config: DetectionConfig,
    base_output_dir: Optional[str] = None,
    use_current_dir: bool = True,
    status_callback: Optional[StatusCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> PipelineResult:
    """
    Execute the complete BGMI detection and highlight clipping pipeline:
      1. Validate video & FFmpeg availability
      2. Create unique BGMI_Output/Session_Date_Time/ structure
      3. Scan frames in memory within selected ROI
      4. Save screenshots ONLY for confirmed events
      5. Group nearby events into clamped non-negative intervals
      6. Generate MP4 highlight clips (skipped if 0 events)
      7. Write events.csv & summary.txt
    """
    config.validate()
    if not roi.is_valid():
        raise ValueError("Invalid Kill-Feed ROI provided. Cannot start processing.")

    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
    if not ffmpeg_ok:
        raise RuntimeError(ffmpeg_msg)

    video_meta = get_video_metadata(video_path)
    mapped_roi = roi.map_to_resolution(video_meta.width, video_meta.height).clamp()
    if not mapped_roi.is_valid():
        raise ValueError(
            f"ROI {mapped_roi} is invalid for video dimensions {video_meta.width}x{video_meta.height}."
        )

    session_paths = create_session_directories(
        base_output_dir=base_output_dir,
        use_current_dir=use_current_dir,
    )
    logger = configure_session_logger(session_paths.log_file)
    config.to_json(os.path.join(session_paths.debug_dir, "session_config.json"))

    def _scan_progress(cur_sec: float, total_sec: float, confirmed: int, msg: str) -> None:
        if status_callback:
            pct = min(0.80, 0.80 * (cur_sec / max(1e-3, total_sec)))
            status_callback(pct, msg, confirmed)

    scan_summary = scan_video_for_kills(
        video_meta=video_meta,
        roi=mapped_roi,
        config=config,
        session_paths=session_paths,
        logger=logger,
        progress_callback=_scan_progress,
        cancel_check=cancel_check,
    )

    clip_segments: List[ClipSegment] = []
    clip_results: List[ClipResult] = []

    if scan_summary.events and not scan_summary.cancelled:
        clip_segments = group_events_into_clips(
            events=scan_summary.events,
            video_duration_sec=video_meta.duration_sec,
            pre_buffer_sec=config.pre_event_buffer_sec,
            post_buffer_sec=config.post_event_buffer_sec,
            merge_gap_sec=config.merge_gap_sec,
            grouping_txt_path=session_paths.grouping_txt,
            grouping_json_path=session_paths.grouping_json,
        )

        def _clip_progress(idx: int, total: int, msg: str) -> None:
            if status_callback:
                pct = 0.80 + 0.20 * (idx / float(max(1, total)))
                status_callback(pct, msg, len(scan_summary.events))

        clip_results = generate_highlight_clips(
            video_meta=video_meta,
            clip_segments=clip_segments,
            session_paths=session_paths,
            config=config,
            logger=logger,
            progress_callback=_clip_progress,
        )
    else:
        logger.info("No highlight clips generated (confirmed_events=%d, cancelled=%s).",
                    len(scan_summary.events), scan_summary.cancelled)

    write_session_summary(
        session_paths=session_paths,
        video_meta=video_meta,
        roi=mapped_roi,
        config=config,
        scan_summary=scan_summary,
        clip_segments=clip_segments,
        clip_results=clip_results,
    )

    if status_callback:
        status_callback(
            1.0,
            f"Complete! Confirmed kills: {len(scan_summary.events)} | Clips: {sum(1 for c in clip_results if c.success)}",
            len(scan_summary.events),
        )

    return PipelineResult(
        session_paths=session_paths,
        video_meta=video_meta,
        roi=mapped_roi,
        scan_summary=scan_summary,
        clip_segments=clip_segments,
        clip_results=clip_results,
    )


def run_roi_frames_pipeline(
    roi_frames: Sequence[Tuple[float, str]],
    video_name: str,
    video_width: int,
    video_height: int,
    duration_sec: float,
    roi: ROIRegion,
    config: DetectionConfig,
    base_output_dir: Optional[str] = None,
    use_current_dir: bool = False,
    status_callback: Optional[StatusCallback] = None,
) -> PipelineResult:
    """
    Execute BGMI detection + highlight interval grouping on pre-cropped ROI frames
    (sampled directly in the browser without uploading multi-GB full video files).
    """
    config.validate()
    video_meta = VideoMetadata(
        path=video_name,
        width=max(1, int(video_width)),
        height=max(1, int(video_height)),
        fps=float(config.sample_fps),
        frame_count=len(roi_frames),
        duration_sec=max(0.0, float(duration_sec)),
        has_audio=True,
        is_vfr=False,
        codec_name="browser-roi-stream",
    )
    mapped_roi = roi.map_to_resolution(video_meta.width, video_meta.height).clamp()
    session_paths = create_session_directories(
        base_output_dir=base_output_dir,
        use_current_dir=use_current_dir,
    )
    logger = configure_session_logger(session_paths.log_file)
    config.to_json(os.path.join(session_paths.debug_dir, "session_config.json"))

    def _scan_progress(cur_idx: float, total_cnt: float, confirmed: int, msg: str) -> None:
        if status_callback:
            pct = min(0.95, 0.95 * (cur_idx / max(1.0, total_cnt)))
            status_callback(pct, msg, confirmed)

    scan_summary = scan_roi_frame_sequence(
        roi_frames=roi_frames,
        video_width=video_meta.width,
        video_height=video_meta.height,
        duration_sec=video_meta.duration_sec,
        roi=mapped_roi,
        config=config,
        session_paths=session_paths,
        logger=logger,
        progress_callback=_scan_progress,
    )

    clip_segments: List[ClipSegment] = []
    if scan_summary.events:
        clip_segments = group_events_into_clips(
            events=scan_summary.events,
            video_duration_sec=video_meta.duration_sec,
            pre_buffer_sec=config.pre_event_buffer_sec,
            post_buffer_sec=config.post_event_buffer_sec,
            merge_gap_sec=config.merge_gap_sec,
            grouping_txt_path=session_paths.grouping_txt,
            grouping_json_path=session_paths.grouping_json,
        )

    write_session_summary(
        session_paths=session_paths,
        video_meta=video_meta,
        roi=mapped_roi,
        config=config,
        scan_summary=scan_summary,
        clip_segments=clip_segments,
        clip_results=[],
    )

    if status_callback:
        status_callback(
            1.0,
            f"Complete! Confirmed kills: {len(scan_summary.events)} | Highlight windows: {len(clip_segments)}",
            len(scan_summary.events),
        )

    return PipelineResult(
        session_paths=session_paths,
        video_meta=video_meta,
        roi=mapped_roi,
        scan_summary=scan_summary,
        clip_segments=clip_segments,
        clip_results=[],
    )
