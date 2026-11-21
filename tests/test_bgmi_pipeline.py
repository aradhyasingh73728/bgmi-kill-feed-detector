from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import List, Tuple

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "Scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from modules.calibration import CalibrationManager
from modules.config import DetectionConfig, KillEvent, ROIRegion
from modules.pipeline import run_bgmi_pipeline, run_roi_frames_pipeline
from modules.roi_selector import ROISelectionError, compute_display_scale, map_display_rect_to_native
from modules.timeGrouping import group_events_into_clips
from modules.video_utils import check_ffmpeg_available, get_video_metadata


def _draw_bgmi_like_kill_banner(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    killer: str,
    weapon_label: str,
    victim: str,
    highlighted: bool = True,
) -> None:
    """Render a realistic synthetic BGMI-style horizontal kill-feed banner onto `frame`."""
    overlay = frame.copy()
    bg_color = (22, 24, 95) if highlighted else (28, 28, 32)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, h / 46.0)
    thick = max(1, int(round(scale * 2.0)))
    baseline_y = y + int(h * 0.68)

    # Left killer name
    cv2.putText(frame, killer, (x + int(w * 0.04), baseline_y), font, scale, (245, 248, 255), thick, cv2.LINE_AA)
    # Middle weapon/action icon + keyword
    mid_x = x + int(w * 0.41)
    cv2.putText(frame, weapon_label, (mid_x, baseline_y), font, scale * 0.9, (90, 215, 255), thick, cv2.LINE_AA)
    # Right victim name
    right_x = x + int(w * 0.64)
    cv2.putText(frame, victim, (right_x, baseline_y), font, scale, (255, 210, 120), thick, cv2.LINE_AA)


def _write_synthetic_video(
    out_path: str,
    width: int,
    height: int,
    fps: int,
    duration_sec: float,
    frame_drawer,
    with_audio: bool = True,
) -> None:
    """Create an H.264 MP4 (optionally with AAC audio) using OpenCV + FFmpeg."""
    raw_avi = out_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_avi, fourcc, float(fps), (width, height))
    total_frames = int(round(fps * duration_sec))
    try:
        for idx in range(total_frames):
            t_sec = idx / float(fps)
            # Textured gameplay-like background (sky top, terrain bottom, HUD noise)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[: height // 2, :] = (135, 110, 80)
            frame[height // 2 :, :] = (45, 85, 55)
            # Add subtle moving foliage/crosshair noise outside and inside scene
            cx = int((width * 0.5) + 30 * np.sin(t_sec * 2.0))
            cv2.circle(frame, (cx, height // 2), 18, (200, 220, 200), 1)
            cv2.putText(frame, "Alive 42", (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            frame_drawer(frame, t_sec, idx)
            writer.write(frame)
    finally:
        writer.release()

    ff_ok, ff_exe = check_ffmpeg_available()
    assert ff_ok, "FFmpeg is required for integration tests"
    if with_audio:
        cmd = [
            ff_exe, "-y", "-hide_banner", "-loglevel", "error",
            "-i", raw_avi,
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_sec:.2f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", out_path,
        ]
    else:
        cmd = [
            ff_exe, "-y", "-hide_banner", "-loglevel", "error",
            "-i", raw_avi,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            out_path,
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(raw_avi):
        os.remove(raw_avi)
    assert proc.returncode == 0, f"FFmpeg failed building test video: {proc.stderr}"


class TestBGMIKillFeedPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="peace_bgmi_test_")
        self.cal_dir = os.path.join(self.tmp_dir, "cal_samples")
        os.makedirs(self.cal_dir, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_roi_selection_mapping_and_invalid_roi(self) -> None:
        # 2560x1440 video scaled down to max 1280x720 -> scale = 0.5
        scale = compute_display_scale(2560, 1440, 1280, 720)
        self.assertAlmostEqual(scale, 0.5, places=4)
        roi = map_display_rect_to_native((850, 40, 320, 90), scale, 2560, 1440)
        self.assertEqual((roi.x, roi.y, roi.width, roi.height), (1700, 80, 640, 180))
        self.assertTrue(roi.is_valid())

        # Transfer normalized ROI to 1280x720 and 2400x1080 Android ultrawide
        roi_720 = roi.map_to_resolution(1280, 720)
        self.assertEqual((roi_720.x, roi_720.y, roi_720.width, roi_720.height), (850, 40, 320, 90))
        roi_android = roi.map_to_resolution(2400, 1080)
        self.assertTrue(roi_android.is_valid())

        # Invalid / cancelled ROIs must raise ROISelectionError
        with self.assertRaises(ROISelectionError):
            map_display_rect_to_native((0, 0, 0, 0), scale, 2560, 1440)
        with self.assertRaises(ROISelectionError):
            map_display_rect_to_native((100, 100, 10, 6), 1.0, 1920, 1080)

    def test_02_video_loading_and_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            get_video_metadata(os.path.join(self.tmp_dir, "does_not_exist.mp4"))

        vid_path = os.path.join(self.tmp_dir, "meta_test.mp4")
        _write_synthetic_video(vid_path, 1280, 720, 15, 2.0, lambda f, t, i: None, with_audio=True)
        meta = get_video_metadata(vid_path)
        self.assertEqual((meta.width, meta.height), (1280, 720))
        self.assertTrue(meta.has_audio)
        self.assertGreater(meta.duration_sec, 1.5)

    def test_03_no_kill_gameplay_and_ui_noise_ignored(self) -> None:
        vid_path = os.path.join(self.tmp_dir, "no_kills_ui_noise.mp4")

        def draw_non_kill_ui(frame: np.ndarray, t_sec: float, idx: int) -> None:
            # Draw unrelated UI/lobby/telemetry text inside the ROI!
            overlay = frame.copy()
            cv2.rectangle(overlay, (1320, 50), (1860, 110), (20, 20, 24), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            if t_sec < 2.0:
                txt = "60 FPS  24 ms"
            elif t_sec < 4.0:
                txt = "LOBBY  INVENTORY  SHOP"
            else:
                txt = "00:45  NW 145"
            cv2.putText(frame, txt, (1340, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

        _write_synthetic_video(vid_path, 1920, 1080, 10, 6.0, draw_non_kill_ui, with_audio=True)
        roi = ROIRegion(1300, 40, 580, 180, 1920, 1080)
        cfg = DetectionConfig(sample_fps=2.0, min_confirm_frames=2, calibration_dir=self.cal_dir)

        res = run_bgmi_pipeline(vid_path, roi, cfg, base_output_dir=self.tmp_dir, use_current_dir=False)
        self.assertEqual(len(res.scan_summary.events), 0)
        self.assertEqual(len(res.clip_results), 0)
        self.assertEqual(os.listdir(res.session_paths.screenshots_dir), [])
        self.assertEqual(os.listdir(res.session_paths.clips_dir), [])
        self.assertTrue(os.path.isfile(res.session_paths.events_csv))
        self.assertTrue(os.path.isfile(res.session_paths.summary_txt))

    def test_04_genuine_kill_deduplication_and_edge_clipping(self) -> None:
        # Kill 1 persists from 1.0s to 4.5s (near beginning -> must clamp clip start to 0.0!)
        # Kill 2 occurs close at 5.2s..7.0s (different kill!)
        # Kill 3 occurs near end at 16.5s..18.5s of 19s video (must clamp clip end to duration!)
        vid_path = os.path.join(self.tmp_dir, "kills_1080p.mp4")

        def draw_kills(frame: np.ndarray, t_sec: float, idx: int) -> None:
            if 1.0 <= t_sec <= 4.5:
                _draw_bgmi_like_kill_banner(frame, 1320, 55, 540, 36, "AlphaPro", "M416", "BotOne", True)
            if 5.2 <= t_sec <= 7.0:
                _draw_bgmi_like_kill_banner(frame, 1320, 55, 540, 36, "AlphaPro", "AKM", "Shadow99", True)
            if 16.5 <= t_sec <= 18.5:
                _draw_bgmi_like_kill_banner(frame, 1320, 55, 540, 36, "AlphaPro", "AWM", "SniperX", True)

        _write_synthetic_video(vid_path, 1920, 1080, 10, 19.0, draw_kills, with_audio=True)
        roi = ROIRegion(1300, 40, 580, 180, 1920, 1080)
        cfg = DetectionConfig(
            sample_fps=2.0,
            confidence_threshold=0.68,
            min_confirm_frames=2,
            pre_event_buffer_sec=4.0,
            post_event_buffer_sec=4.0,
            merge_gap_sec=1.0,
            preserve_audio=True,
            use_gpu_if_available=False,
            calibration_dir=self.cal_dir,
        )

        res = run_bgmi_pipeline(vid_path, roi, cfg, base_output_dir=self.tmp_dir, use_current_dir=False)

        # Exactly 3 unique events (consecutive frames of each kill deduplicated!)
        self.assertEqual(len(res.scan_summary.events), 3, f"Events: {res.scan_summary.events}")
        screenshots = os.listdir(res.session_paths.screenshots_dir)
        self.assertEqual(len(screenshots), 3)
        # Verify full-frame screenshots (1920x1080, not cropped ROI)
        shot_img = cv2.imread(os.path.join(res.session_paths.screenshots_dir, screenshots[0]))
        self.assertIsNotNone(shot_img)
        self.assertEqual(shot_img.shape[:2], (1080, 1920))

        # Verify non-negative start and duration-clamped end in clips, plus audio preserved
        self.assertGreaterEqual(len(res.clip_results), 2)
        for c in res.clip_results:
            self.assertTrue(c.success, c.error_message)
            self.assertGreaterEqual(c.start_sec, 0.0)
            self.assertLessEqual(c.end_sec, res.video_meta.duration_sec + 0.05)
            clip_meta = get_video_metadata(c.output_path)
            self.assertTrue(clip_meta.has_audio)

    def test_05_simultaneous_multiple_kill_feed_rows_720p(self) -> None:
        # Test 1280x720 video with TWO different kills stacked simultaneously in the ROI!
        vid_path = os.path.join(self.tmp_dir, "multi_row_720p.mp4")

        def draw_stacked(frame: np.ndarray, t_sec: float, idx: int) -> None:
            if 1.0 <= t_sec <= 3.5:
                _draw_bgmi_like_kill_banner(frame, 820, 40, 420, 30, "ViperIN", "M416", "TargetA", True)
                _draw_bgmi_like_kill_banner(frame, 820, 82, 420, 30, "Ninja99", "UMP45", "TargetB", False)

        _write_synthetic_video(vid_path, 1280, 720, 10, 5.0, draw_stacked, with_audio=False)
        roi = ROIRegion(800, 30, 460, 110, 1280, 720)
        cfg = DetectionConfig(
            sample_fps=2.0,
            confidence_threshold=0.68,
            min_confirm_frames=2,
            pre_event_buffer_sec=2.0,
            post_event_buffer_sec=2.0,
            calibration_dir=self.cal_dir,
        )

        res = run_bgmi_pipeline(vid_path, roi, cfg, base_output_dir=self.tmp_dir, use_current_dir=False)
        self.assertEqual(len(res.scan_summary.events), 2, f"Events: {res.scan_summary.events}")

    def test_06_grouping_clamping_logic(self) -> None:
        events = [
            KillEvent(1, 3.0, 30, 0.9, "A M416 B", "A", "B", "M416", (0, 0, 10, 10), True),
            KillEvent(2, 12.0, 120, 0.9, "A AKM C", "A", "C", "AKM", (0, 0, 10, 10), True),
            KillEvent(3, 55.0, 550, 0.9, "A AWM D", "A", "D", "AWM", (0, 0, 10, 10), True),
        ]
        segs = group_events_into_clips(events, video_duration_sec=60.0, pre_buffer_sec=10.0, post_buffer_sec=10.0, merge_gap_sec=4.0)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].start_sec, 0.0)   # Clamped from -7.0 to 0.0
        self.assertEqual(segs[0].end_sec, 22.0)
        self.assertEqual(segs[1].start_sec, 45.0)
        self.assertEqual(segs[1].end_sec, 60.0)   # Clamped from 65.0 to 60.0

    def test_07_roi_frames_pipeline(self) -> None:
        # Exercise pre-cropped ROI frame scanning (used by Fast ROI-Only Web Scan)
        roi = ROIRegion(800, 30, 460, 110, 1280, 720)
        roi_dir = os.path.join(self.tmp_dir, "roi_seq")
        os.makedirs(roi_dir, exist_ok=True)
        sequence: List[Tuple[float, str]] = []

        for idx in range(12):
            t_sec = idx * 0.5
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[:360, :] = (130, 105, 80)
            if 1.0 <= t_sec <= 3.0:
                _draw_bgmi_like_kill_banner(frame, 820, 40, 420, 32, "ViperIN", "M416", "EnemyZero", True)
            crop = frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            img_path = os.path.join(roi_dir, f"roi_{idx:03d}.jpg")
            cv2.imwrite(img_path, crop)
            sequence.append((t_sec, img_path))

        cfg = DetectionConfig(
            sample_fps=2.0,
            confidence_threshold=0.68,
            min_confirm_frames=2,
            pre_event_buffer_sec=2.0,
            post_event_buffer_sec=2.0,
            calibration_dir=self.cal_dir,
        )
        res = run_roi_frames_pipeline(
            roi_frames=sequence,
            video_name="browser_30min.mp4",
            video_width=1280,
            video_height=720,
            duration_sec=6.0,
            roi=roi,
            config=cfg,
            base_output_dir=self.tmp_dir,
            use_current_dir=False,
        )
        self.assertEqual(len(res.scan_summary.events), 1, f"Events: {res.scan_summary.events}")
        self.assertEqual(len(res.clip_segments), 1)
        self.assertGreaterEqual(res.clip_segments[0].start_sec, 0.0)
        self.assertLessEqual(res.clip_segments[0].end_sec, 6.0)
        self.assertTrue(os.path.isfile(res.session_paths.events_csv))
        self.assertTrue(os.path.isfile(res.session_paths.summary_txt))

    def test_08_high_texture_kill_banner_and_nonkill_rejection(self) -> None:
        # Ensure banners over dense high-texture terrain (stroke density ~0.66-0.75) still trigger
        # when OCR strongly reads two participants, while textured noise with no OCR text produces 0 events.
        roi = ROIRegion(12, 318, 303, 65, 720, 1280)
        roi_dir = os.path.join(self.tmp_dir, "roi_high_tex")
        os.makedirs(roi_dir, exist_ok=True)
        sequence: List[Tuple[float, str]] = []

        rng = np.random.default_rng(42)
        for idx in range(10):
            t_sec = idx * (1.0 / 3.0)
            frame = np.zeros((1280, 720, 3), dtype=np.uint8)
            noise = rng.integers(35, 165, size=(roi.height, roi.width, 3), dtype=np.uint8)
            for x in range(0, roi.width, 4):
                noise[:, x, :] = 185 if (x // 4) % 2 == 0 else 45
            frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = noise

            if 1.0 <= t_sec <= 2.2:
                _draw_bgmi_like_kill_banner(frame, 18, 322, 275, 32, "GTRXSURAJ", "AKM", "ShadabP", True)

            crop = frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            img_path = os.path.join(roi_dir, f"tex_{idx:03d}.png")
            cv2.imwrite(img_path, crop)
            sequence.append((t_sec, img_path))

        cfg = DetectionConfig(
            sample_fps=3.0,
            confidence_threshold=0.65,
            min_confirm_frames=2,
            calibration_dir=self.cal_dir,
        )
        res = run_roi_frames_pipeline(
            roi_frames=sequence,
            video_name="portrait_high_tex.mp4",
            video_width=720,
            video_height=1280,
            duration_sec=3.5,
            roi=roi,
            config=cfg,
            base_output_dir=self.tmp_dir,
            use_current_dir=False,
        )
        self.assertEqual(len(res.scan_summary.events), 1, f"Events: {res.scan_summary.events}")

    def test_09_portrait_unsplit_band_and_partial_continuation(self) -> None:
        # Simulate portrait 720x1280 with full banner on frame t=1.0s and faded-victim killer card on t=1.33s..2.0s
        roi = ROIRegion(12, 318, 360, 65, 720, 1280)
        roi_dir = os.path.join(self.tmp_dir, "roi_portrait_cont")
        os.makedirs(roi_dir, exist_ok=True)
        sequence: List[Tuple[float, str]] = []

        for idx in range(9):
            t_sec = idx * (1.0 / 3.0)
            crop = np.full((65, 360, 3), 55, dtype=np.uint8)
            for x in range(10, 350, 6):
                cv2.line(crop, (x, 32), (x + 2, 62), (120, 120, 120), 1)
            if abs(t_sec - 1.0) < 0.08:
                _draw_bgmi_like_kill_banner(crop, 6, 2, 344, 30, "GTRxSURAJ", "AKM", "DragonX", True)
            elif 1.25 <= t_sec <= 2.1:
                _draw_bgmi_like_kill_banner(crop, 6, 2, 344, 30, "GTRxSURAJ", "", "", True)
                # Right-side faded banner emblem / border (non-text horizontal strokes)
                cv2.line(crop, (240, 14), (320, 14), (210, 210, 210), 2)
                cv2.line(crop, (250, 20), (310, 20), (195, 195, 195), 2)

            img_path = os.path.join(roi_dir, f"cont_{idx:03d}.png")
            cv2.imwrite(img_path, crop)
            sequence.append((t_sec, img_path))

        cfg = DetectionConfig(
            sample_fps=3.0,
            confidence_threshold=0.65,
            min_confirm_frames=2,
            calibration_dir=self.cal_dir,
        )
        res = run_roi_frames_pipeline(
            roi_frames=sequence,
            video_name="portrait_end_kill.mp4",
            video_width=720,
            video_height=1280,
            duration_sec=3.0,
            roi=roi,
            config=cfg,
            base_output_dir=self.tmp_dir,
            use_current_dir=False,
        )
        self.assertEqual(len(res.scan_summary.events), 1, f"Events: {res.scan_summary.events}")
        self.assertIn("dragon", res.scan_summary.events[0].victim.lower())

    def test_10_final_frame_kill_cutoff_at_video_end(self) -> None:
        # Mid-video single-frame kill at t=1.0s must remain rejected by min_confirm_frames=2,
        # while a clean two-participant kill banner cut off on the final frame (t=2.67s) must be captured.
        roi = ROIRegion(12, 318, 360, 65, 720, 1280)
        roi_dir = os.path.join(self.tmp_dir, "roi_eof_flush")
        os.makedirs(roi_dir, exist_ok=True)
        sequence: List[Tuple[float, str]] = []

        for idx in range(9):
            t_sec = round(idx * (1.0 / 3.0), 3)
            crop = np.full((65, 360, 3), 50, dtype=np.uint8)
            if idx == 3:
                _draw_bgmi_like_kill_banner(crop, 6, 2, 344, 30, "AlphaOne", "M416", "MidFlash", True)
            elif idx == 8:
                _draw_bgmi_like_kill_banner(crop, 6, 2, 344, 30, "AlphaOne", "AWM", "FinalEnemy", True)
            img_path = os.path.join(roi_dir, f"eof_{idx:03d}.png")
            cv2.imwrite(img_path, crop)
            sequence.append((t_sec, img_path))

        cfg = DetectionConfig(
            sample_fps=3.0,
            confidence_threshold=0.65,
            min_confirm_frames=2,
            calibration_dir=self.cal_dir,
        )
        res = run_roi_frames_pipeline(
            roi_frames=sequence,
            video_name="eof_clip.mp4",
            video_width=720,
            video_height=1280,
            duration_sec=2.7,
            roi=roi,
            config=cfg,
            base_output_dir=self.tmp_dir,
            use_current_dir=False,
        )
        self.assertEqual(len(res.scan_summary.events), 1, f"Events: {res.scan_summary.events}")
        self.assertIn("final", res.scan_summary.events[0].victim.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
