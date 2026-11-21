from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from typing import Any, Dict

import cv2
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from modules.calibration import CalibrationManager
from modules.config import DetectionConfig, ROIRegion
from modules.ocr_engine import OCREngine
from modules.pipeline import run_bgmi_pipeline, run_roi_frames_pipeline
from modules.video_utils import (
    check_ffmpeg_available,
    check_ffprobe_available,
    get_video_metadata,
    read_preview_frame,
)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _build_detection_config(raw_cfg: Dict[str, Any]) -> DetectionConfig:
    return DetectionConfig(
        sample_fps=float(raw_cfg.get("sample_fps", 3.0)),
        confidence_threshold=float(raw_cfg.get("confidence_threshold", 0.65)),
        min_confirm_frames=int(raw_cfg.get("min_confirm_frames", 2)),
        pre_event_buffer_sec=float(raw_cfg.get("pre_event_buffer_sec", 8.0)),
        post_event_buffer_sec=float(raw_cfg.get("post_event_buffer_sec", 6.0)),
        merge_gap_sec=float(raw_cfg.get("merge_gap_sec", 4.0)),
        player_ign=str(raw_cfg.get("player_ign", "")),
        only_player_kills=bool(raw_cfg.get("only_player_kills", False)),
        preserve_audio=bool(raw_cfg.get("preserve_audio", True)),
        save_debug_roi=bool(raw_cfg.get("save_debug_roi", True)),
        fast_stream_copy=bool(raw_cfg.get("fast_stream_copy", False)),
        use_gpu_if_available=bool(raw_cfg.get("use_gpu_if_available", False)),
    )


def cmd_status() -> int:
    ff_ok, ff_info = check_ffmpeg_available()
    ffprobe_path = check_ffprobe_available()
    ocr = OCREngine("auto")
    cal = CalibrationManager()
    payload = {
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "ffmpeg_available": ff_ok,
        "ffmpeg_info": ff_info,
        "ffprobe_available": bool(ffprobe_path),
        "ocr_backend": ocr.backend_name,
        "ocr_available": ocr.is_available,
        "calibration_dir": cal.samples_dir,
        "calibration_samples": [
            {"name": s.name, "filename": os.path.basename(s.path), "width": int(s.image_bgr.shape[1]), "height": int(s.image_bgr.shape[0])}
            for s in cal.samples
        ],
    }
    print(json.dumps(payload))
    return 0


def cmd_metadata(video_path: str) -> int:
    meta = get_video_metadata(video_path)
    print(json.dumps(asdict(meta)))
    return 0


def cmd_preview(video_path: str, ts: float | None, out_path: str) -> int:
    frame, actual_ts = read_preview_frame(video_path, ts)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ok = cv2.imwrite(out_path, frame)
    if not ok:
        raise RuntimeError(f"Failed to write preview PNG to '{out_path}'")
    h, w = frame.shape[:2]
    print(json.dumps({"out_path": out_path, "actual_ts": actual_ts, "width": w, "height": h}))
    return 0


def cmd_save_calibration(video_path: str | None, image_path: str | None, ts: float, roi_str: str, label: str) -> int:
    parts = [int(p.strip()) for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    if image_path:
        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise RuntimeError(f"Failed to read calibration image '{image_path}'")
    elif video_path:
        frame, _ = read_preview_frame(video_path, ts)
    else:
        raise ValueError("Either --video or --image is required.")
    fh, fw = frame.shape[:2]
    roi = ROIRegion(parts[0], parts[1], parts[2], parts[3], fw, fh).clamp()
    if roi.width < 8 or roi.height < 8:
        raise ValueError("Calibration crop must be at least 8x8px")
    crop = frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    cal = CalibrationManager()
    safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label.strip()) or "bgmi_sample"
    idx = 1
    out_path = os.path.join(cal.samples_dir, f"{safe_label}_{idx:02d}.png")
    while os.path.exists(out_path):
        idx += 1
        out_path = os.path.join(cal.samples_dir, f"{safe_label}_{idx:02d}.png")
    cv2.imwrite(out_path, crop)
    cal.reload()
    print(json.dumps({"saved_path": out_path, "filename": os.path.basename(out_path), "total_samples": len(cal.samples)}))
    return 0


def cmd_generate_demo(out_path: str) -> int:
    import subprocess

    ff_ok, ff_exe = check_ffmpeg_available()
    if not ff_ok:
        raise RuntimeError(ff_exe)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    raw_mp4 = out_path + ".raw.mp4"
    width, height, fps, duration = 1280, 720, 10, 14.0
    writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))

    def draw_banner(frame: np.ndarray, x: int, y: int, w: int, h: int, killer: str, weapon: str, victim: str, red: bool) -> None:
        ov = frame.copy()
        bg = (22, 24, 95) if red else (28, 28, 32)
        cv2.rectangle(ov, (x, y), (x + w, y + h), bg, -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        sc = max(0.45, h / 46.0)
        th = max(1, int(round(sc * 2.0)))
        by = y + int(h * 0.68)
        cv2.putText(frame, killer, (x + int(w * 0.04), by), cv2.FONT_HERSHEY_SIMPLEX, sc, (245, 248, 255), th, cv2.LINE_AA)
        cv2.putText(frame, weapon, (x + int(w * 0.41), by), cv2.FONT_HERSHEY_SIMPLEX, sc * 0.9, (90, 215, 255), th, cv2.LINE_AA)
        cv2.putText(frame, victim, (x + int(w * 0.64), by), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 210, 120), th, cv2.LINE_AA)

    try:
        total_frames = int(fps * duration)
        for idx in range(total_frames):
            t = idx / float(fps)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[: height // 2] = (128, 102, 74)
            frame[height // 2 :] = (46, 82, 54)
            cx = int(width * 0.5 + 36 * np.sin(t * 1.8))
            cv2.circle(frame, (cx, height // 2), 20, (220, 230, 210), 1)
            cv2.putText(frame, "Alive 34", (32, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
            # Non-kill UI noise at t=4.2..5.5
            if 4.2 <= t <= 5.5:
                ov = frame.copy()
                cv2.rectangle(ov, (820, 38), (1240, 72), (24, 24, 28), -1)
                cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
                cv2.putText(frame, "60 FPS  22 ms", (840, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
            # Kill 1 at t=1.2..3.6
            if 1.2 <= t <= 3.6:
                draw_banner(frame, 820, 40, 420, 32, "ViperIN", "M416", "BotAlpha", True)
            # Kill 2 at t=7.0..10.5 + simultaneous Kill 3 at t=8.0..10.5
            if 7.0 <= t <= 10.5:
                draw_banner(frame, 820, 40, 420, 32, "ViperIN", "AKM", "ShadowX", True)
            if 8.0 <= t <= 10.5:
                draw_banner(frame, 820, 80, 420, 32, "SquadMate", "UMP45", "Sniper07", False)
            writer.write(frame)
    finally:
        writer.release()

    proc = subprocess.run(
        [
            ff_exe, "-y", "-hide_banner", "-loglevel", "error",
            "-i", raw_mp4,
            "-f", "lavfi", "-i", f"sine=frequency=520:duration={duration:.2f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", out_path,
        ],
        capture_output=True,
        text=True,
    )
    if os.path.exists(raw_mp4):
        os.remove(raw_mp4)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg demo creation failed: {proc.stderr}")

    meta = get_video_metadata(out_path)
    print(json.dumps({"video_path": out_path, "suggested_roi": {"x": 800, "y": 28, "width": 460, "height": 110}, "metadata": asdict(meta)}))
    return 0


def cmd_run_job(video_path: str, roi_str: str, config_json_path: str, out_base: str, status_file: str) -> int:
    meta = get_video_metadata(video_path)
    parts = [int(p.strip()) for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    roi = ROIRegion(parts[0], parts[1], parts[2], parts[3], meta.width, meta.height).clamp()

    with open(config_json_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    cfg = _build_detection_config(raw_cfg)

    _atomic_write_json(status_file, {"state": "running", "progress": 0.0, "message": "Initializing pipeline...", "confirmed_count": 0})

    def _on_progress(frac: float, msg: str, count: int) -> None:
        _atomic_write_json(
            status_file,
            {"state": "running", "progress": round(float(frac), 4), "message": msg, "confirmed_count": int(count)},
        )

    try:
        res = run_bgmi_pipeline(
            video_path=video_path,
            roi=roi,
            config=cfg,
            base_output_dir=out_base,
            use_current_dir=False,
            status_callback=_on_progress,
        )

        summary_text = ""
        if os.path.isfile(res.session_paths.summary_txt):
            with open(res.session_paths.summary_txt, "r", encoding="utf-8") as sf:
                summary_text = sf.read()

        payload = {
            "state": "completed",
            "scan_mode": "full",
            "progress": 1.0,
            "message": f"Complete! Confirmed Kills: {len(res.scan_summary.events)}",
            "confirmed_count": len(res.scan_summary.events),
            "session_dir": res.session_paths.root_dir,
            "ocr_backend": res.scan_summary.ocr_backend,
            "calibrated_samples_count": res.scan_summary.calibrated_samples_count,
            "frames_analyzed": res.scan_summary.frames_analyzed,
            "elapsed_sec": round(res.scan_summary.elapsed_sec, 2),
            "events": [
                {
                    "event_id": ev.event_id,
                    "timestamp_sec": round(ev.timestamp_sec, 3),
                    "timestamp_hms": ev.formatted_hms(),
                    "confidence": round(ev.confidence, 4),
                    "killer": ev.killer,
                    "weapon_or_action": ev.weapon_or_action,
                    "victim": ev.victim,
                    "ocr_text": ev.ocr_text,
                    "is_highlighted": ev.is_highlighted,
                    "screenshot_file": os.path.basename(ev.screenshot_path) if ev.screenshot_path else "",
                    "debug_roi_file": os.path.basename(ev.debug_roi_path) if ev.debug_roi_path else "",
                }
                for ev in res.scan_summary.events
            ],
            "clips": [
                {
                    "clip_index": c.clip_index,
                    "filename": os.path.basename(c.output_path),
                    "start_sec": round(c.start_sec, 2),
                    "end_sec": round(c.end_sec, 2),
                    "duration_sec": round(max(0.0, c.end_sec - c.start_sec), 2),
                    "success": c.success,
                    "error_message": c.error_message,
                }
                for c in res.clip_results
            ],
            "clip_segments": [
                {
                    "clip_index": seg.clip_index,
                    "start_sec": round(seg.start_sec, 2),
                    "end_sec": round(seg.end_sec, 2),
                    "duration_sec": round(seg.duration_sec, 2),
                    "event_ids": list(seg.event_ids),
                }
                for seg in res.clip_segments
            ],
            "summary_text": summary_text,
        }
        _atomic_write_json(status_file, payload)
        return 0
    except Exception as exc:
        _atomic_write_json(
            status_file,
            {
                "state": "failed",
                "progress": 0.0,
                "message": str(exc),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def cmd_scan_roi_frames(
    manifest_path: str,
    video_name: str,
    width: int,
    height: int,
    duration: float,
    roi_str: str,
    config_json_path: str,
    out_base: str,
    status_file: str,
) -> int:
    parts = [int(p.strip()) for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    roi = ROIRegion(parts[0], parts[1], parts[2], parts[3], max(1, width), max(1, height)).clamp()

    with open(manifest_path, "r", encoding="utf-8") as mf:
        raw_manifest = json.load(mf)
    roi_frames = sorted(
        [(float(item["ts"]), str(item["path"])) for item in raw_manifest],
        key=lambda x: x[0],
    )

    with open(config_json_path, "r", encoding="utf-8") as cf:
        raw_cfg = json.load(cf)
    cfg = _build_detection_config(raw_cfg)

    _atomic_write_json(status_file, {"state": "running", "progress": 0.0, "message": "Initializing Fast ROI-Only scan...", "confirmed_count": 0})

    def _on_progress(frac: float, msg: str, count: int) -> None:
        _atomic_write_json(
            status_file,
            {"state": "running", "progress": round(float(frac), 4), "message": msg, "confirmed_count": int(count)},
        )

    try:
        res = run_roi_frames_pipeline(
            roi_frames=roi_frames,
            video_name=video_name,
            video_width=width,
            video_height=height,
            duration_sec=duration,
            roi=roi,
            config=cfg,
            base_output_dir=out_base,
            use_current_dir=False,
            status_callback=_on_progress,
        )

        summary_text = ""
        if os.path.isfile(res.session_paths.summary_txt):
            with open(res.session_paths.summary_txt, "r", encoding="utf-8") as sf:
                summary_text = sf.read()

        payload = {
            "state": "completed",
            "scan_mode": "roi_only",
            "progress": 1.0,
            "message": f"Complete! Confirmed Kills: {len(res.scan_summary.events)}",
            "confirmed_count": len(res.scan_summary.events),
            "session_dir": res.session_paths.root_dir,
            "ocr_backend": res.scan_summary.ocr_backend,
            "calibrated_samples_count": res.scan_summary.calibrated_samples_count,
            "frames_analyzed": res.scan_summary.frames_analyzed,
            "elapsed_sec": round(res.scan_summary.elapsed_sec, 2),
            "events": [
                {
                    "event_id": ev.event_id,
                    "timestamp_sec": round(ev.timestamp_sec, 3),
                    "timestamp_hms": ev.formatted_hms(),
                    "confidence": round(ev.confidence, 4),
                    "killer": ev.killer,
                    "weapon_or_action": ev.weapon_or_action,
                    "victim": ev.victim,
                    "ocr_text": ev.ocr_text,
                    "is_highlighted": ev.is_highlighted,
                    "screenshot_file": "",
                    "debug_roi_file": os.path.basename(ev.debug_roi_path) if ev.debug_roi_path else "",
                }
                for ev in res.scan_summary.events
            ],
            "clips": [],
            "clip_segments": [
                {
                    "clip_index": seg.clip_index,
                    "start_sec": round(seg.start_sec, 2),
                    "end_sec": round(seg.end_sec, 2),
                    "duration_sec": round(seg.duration_sec, 2),
                    "event_ids": list(seg.event_ids),
                }
                for seg in res.clip_segments
            ],
            "summary_text": summary_text,
        }
        _atomic_write_json(status_file, payload)
        return 0
    except Exception as exc:
        _atomic_write_json(
            status_file,
            {
                "state": "failed",
                "progress": 0.0,
                "message": str(exc),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p_meta = sub.add_parser("metadata")
    p_meta.add_argument("--video", required=True)

    p_prev = sub.add_parser("preview")
    p_prev.add_argument("--video", required=True)
    p_prev.add_argument("--ts", type=float, default=None)
    p_prev.add_argument("--out", required=True)

    p_cal = sub.add_parser("save-calibration")
    p_cal.add_argument("--video", default=None)
    p_cal.add_argument("--image", default=None)
    p_cal.add_argument("--ts", type=float, default=0.0)
    p_cal.add_argument("--roi", required=True)
    p_cal.add_argument("--label", default="bgmi_sample")

    p_demo = sub.add_parser("generate-demo")
    p_demo.add_argument("--out", required=True)

    p_job = sub.add_parser("run-job")
    p_job.add_argument("--video", required=True)
    p_job.add_argument("--roi", required=True)
    p_job.add_argument("--config", required=True)
    p_job.add_argument("--out-base", required=True)
    p_job.add_argument("--status-file", required=True)

    p_roi = sub.add_parser("scan-roi-frames")
    p_roi.add_argument("--manifest", required=True)
    p_roi.add_argument("--video-name", required=True)
    p_roi.add_argument("--width", type=int, required=True)
    p_roi.add_argument("--height", type=int, required=True)
    p_roi.add_argument("--duration", type=float, required=True)
    p_roi.add_argument("--roi", required=True)
    p_roi.add_argument("--config", required=True)
    p_roi.add_argument("--out-base", required=True)
    p_roi.add_argument("--status-file", required=True)

    args = parser.parse_args()
    try:
        if args.cmd == "status":
            return cmd_status()
        if args.cmd == "metadata":
            return cmd_metadata(args.video)
        if args.cmd == "preview":
            return cmd_preview(args.video, args.ts, args.out)
        if args.cmd == "save-calibration":
            return cmd_save_calibration(args.video, args.image, args.ts, args.roi, args.label)
        if args.cmd == "generate-demo":
            return cmd_generate_demo(args.out)
        if args.cmd == "run-job":
            return cmd_run_job(args.video, args.roi, args.config, args.out_base, args.status_file)
        if args.cmd == "scan-roi-frames":
            return cmd_scan_roi_frames(
                args.manifest,
                args.video_name,
                args.width,
                args.height,
                args.duration,
                args.roi,
                args.config,
                args.out_base,
                args.status_file,
            )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
