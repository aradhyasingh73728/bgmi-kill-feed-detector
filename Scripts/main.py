from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Ensure Scripts/ is always on sys.path whether invoked from repo root or Scripts/
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from modules.calibration import CalibrationManager
from modules.config import DetectionConfig, ROIRegion
from modules.pipeline import run_bgmi_pipeline
from modules.roi_selector import ROISelectionError, select_kill_feed_roi
from modules.video_utils import check_ffmpeg_available, format_seconds_hms, get_video_metadata


def _parse_roi_arg(raw_roi: str, width: int, height: int) -> ROIRegion:
    parts = [int(p.strip()) for p in raw_roi.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be formatted as x,y,width,height (e.g., 1400,60,480,220).")
    x, y, w, h = parts
    roi = ROIRegion(
        x=x,
        y=y,
        width=w,
        height=h,
        source_width=width,
        source_height=height,
    ).clamp()
    if not roi.is_valid():
        raise ValueError(f"Provided ROI {raw_roi} is invalid or too small for {width}x{height}.")
    return roi


def run_cli(args: argparse.Namespace) -> int:
    ff_ok, ff_info = check_ffmpeg_available()
    if not ff_ok:
        print(f"ERROR: {ff_info}")
        return 1

    video_path = args.video
    if not video_path:
        video_path = input("Enter path to BGMI gameplay video: ").strip().strip('"')
    if not os.path.isfile(video_path):
        print(f"ERROR: Video file does not exist: '{video_path}'")
        return 1

    meta = get_video_metadata(video_path)
    print(
        f"Loaded video: {meta.width}x{meta.height} | {meta.fps:.2f} fps | "
        f"duration={format_seconds_hms(meta.duration_sec)} | audio={'Yes' if meta.has_audio else 'No'}"
    )

    if args.calibrate:
        cal = CalibrationManager()
        ts = args.preview_time if args.preview_time is not None else min(10.0, meta.duration_sec * 0.15)
        saved = cal.capture_sample_interactively(video_path, ts, label=args.sample_label)
        if saved:
            print(f"Saved BGMI calibration sample to: {saved}")
            return 0
        print("Calibration cancelled or invalid selection.")
        return 1

    if args.roi:
        roi = _parse_roi_arg(args.roi, meta.width, meta.height)
        print(f"Using CLI ROI (native): x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")
    else:
        print("Opening preview frame for mandatory Kill-Feed Region selection...")
        try:
            roi = select_kill_feed_roi(video_path, preview_time_sec=args.preview_time)
            print(f"Selected ROI (native): x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")
        except ROISelectionError as exc:
            print(f"ERROR: {exc}")
            return 1

    config = DetectionConfig(
        sample_fps=args.sample_fps,
        start_time_sec=args.start_time,
        end_time_sec=args.end_time,
        confidence_threshold=args.confidence,
        min_confirm_frames=args.confirm_frames,
        pre_event_buffer_sec=args.pre_buffer,
        post_event_buffer_sec=args.post_buffer,
        merge_gap_sec=args.merge_gap,
        player_ign=args.player_ign or "",
        only_player_kills=bool(args.only_player_kills),
        preserve_audio=not args.no_audio,
        fast_stream_copy=bool(args.fast_copy),
        use_gpu_if_available=bool(args.gpu),
        save_debug_roi=not args.no_debug_roi,
        ocr_backend=args.ocr_backend,
    )

    def _cli_progress(frac: float, msg: str, count: int) -> None:
        print(f"[{frac * 100:5.1f}%] {msg}")

    result = run_bgmi_pipeline(
        video_path=video_path,
        roi=roi,
        config=config,
        base_output_dir=args.output_dir,
        use_current_dir=(args.output_dir is None),
        status_callback=_cli_progress,
    )

    print("\n=== Processing Complete ===")
    print(f"Confirmed Kill-Feed Events : {len(result.scan_summary.events)}")
    print(f"Generated Highlight Clips  : {sum(1 for c in result.clip_results if c.success)}")
    print(f"Output Folder              : {result.session_paths.root_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PEACE — BGMI Kill Feed Detection & Automatic Highlight Clipping Software"
    )
    parser.add_argument("--cli", action="store_true", help="Run in interactive/command-line mode instead of GUI")
    parser.add_argument("--video", type=str, default=None, help="Path to BGMI gameplay video")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory")
    parser.add_argument("--roi", type=str, default=None, help="Optional native ROI as x,y,w,h (otherwise selectROI opens)")
    parser.add_argument("--preview-time", type=float, default=None, help="Timestamp (sec) for ROI selection preview frame")
    parser.add_argument("--sample-fps", type=float, default=3.0, help="Detection sampling FPS (default: 3.0)")
    parser.add_argument("--start-time", type=float, default=None, help="Optional scan start time in seconds")
    parser.add_argument("--end-time", type=float, default=None, help="Optional scan end time in seconds")
    parser.add_argument("--confidence", type=float, default=0.65, help="Detection confidence threshold (default: 0.65)")
    parser.add_argument("--confirm-frames", type=int, default=2, help="Frames required to confirm an event (default: 2)")
    parser.add_argument("--pre-buffer", type=float, default=8.0, help="Seconds before kill event (default: 8.0)")
    parser.add_argument("--post-buffer", type=float, default=6.0, help="Seconds after kill event (default: 6.0)")
    parser.add_argument("--merge-gap", type=float, default=4.0, help="Merge buffered clips within this gap in sec (default: 4.0)")
    parser.add_argument("--player-ign", type=str, default="", help="Optional BGMI player IGN for filtering")
    parser.add_argument("--only-player-kills", action="store_true", help="Only keep events matching --player-ign")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio in generated clips")
    parser.add_argument("--fast-copy", action="store_true", help="Use FFmpeg stream copy instead of re-encoding")
    parser.add_argument("--gpu", action="store_true", help="Use optional GPU hwaccel decoding if available")
    parser.add_argument("--no-debug-roi", action="store_true", help="Skip saving debug ROI crop images")
    parser.add_argument("--ocr-backend", type=str, default="auto", choices=["auto", "rapidocr", "tesseract", "none"])
    parser.add_argument("--calibrate", action="store_true", help="Capture a real BGMI calibration sample from --video")
    parser.add_argument("--sample-label", type=str, default="bgmi_killfeed", help="Label for calibration sample")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Launch GUI when no CLI-specific flags were passed
    if not args.cli and not args.video and not args.calibrate:
        try:
            from modules.gui import launch_gui

            launch_gui()
            return 0
        except Exception as exc:
            print(f"GUI unavailable ({exc}); falling back to CLI mode.")
            return run_cli(args)

    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
