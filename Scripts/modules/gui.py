from __future__ import annotations

import os
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from modules.calibration import CalibrationManager
from modules.config import DetectionConfig, ROIRegion
from modules.ocr_engine import OCREngine
from modules.pipeline import PipelineResult, run_bgmi_pipeline
from modules.roi_selector import ROISelectionError, render_roi_preview_image, select_kill_feed_roi
from modules.video_utils import (
    VideoMetadata,
    check_ffmpeg_available,
    format_seconds_hms,
    get_video_metadata,
    read_preview_frame,
)


def open_folder_in_explorer(folder_path: str) -> None:
    if not folder_path or not os.path.isdir(folder_path):
        return
    system = platform.system()
    if system == "Windows":
        os.startfile(folder_path)  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.Popen(["open", folder_path])
    else:
        subprocess.Popen(["xdg-open", folder_path])


class BGMIDetectorApp:
    """Clean, step-guided Tkinter GUI for PEACE — BGMI Kill Feed Detector."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PEACE — BGMI Kill Feed Detector & Highlight Clipper")
        self.root.geometry("920x740")
        self.root.minsize(840, 680)

        self.video_meta: Optional[VideoMetadata] = None
        self.selected_roi: Optional[ROIRegion] = None
        self.last_result: Optional[PipelineResult] = None
        self._cancel_requested = False
        self._worker_thread: Optional[threading.Thread] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None

        self.calibration = CalibrationManager()
        self.ocr_probe = OCREngine("auto")

        self._build_vars()
        self._build_ui()
        self._refresh_environment_status()

    def _build_vars(self) -> None:
        self.var_video_path = tk.StringVar()
        self.var_output_dir = tk.StringVar(value=os.getcwd())
        self.var_preview_sec = tk.DoubleVar(value=10.0)
        self.var_sample_fps = tk.DoubleVar(value=3.0)
        self.var_confidence = tk.DoubleVar(value=0.65)
        self.var_confirm_frames = tk.IntVar(value=2)
        self.var_pre_buffer = tk.DoubleVar(value=8.0)
        self.var_post_buffer = tk.DoubleVar(value=6.0)
        self.var_merge_gap = tk.DoubleVar(value=4.0)
        self.var_player_ign = tk.StringVar(value="")
        self.var_only_player_kills = tk.BooleanVar(value=False)
        self.var_preserve_audio = tk.BooleanVar(value=True)
        self.var_save_debug_roi = tk.BooleanVar(value=True)
        self.var_fast_copy = tk.BooleanVar(value=False)
        self.var_use_gpu = tk.BooleanVar(value=False)

        self.var_video_info = tk.StringVar(value="No video selected.")
        self.var_roi_info = tk.StringVar(value="No kill-feed region selected (mandatory before processing).")
        self.var_env_info = tk.StringVar(value="")
        self.var_progress_text = tk.StringVar(value="Ready.")
        self.var_events_count = tk.StringVar(value="Confirmed Kills: 0")

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Environment banner
        env_bar = ttk.LabelFrame(main_frame, text="System & Calibration Status", padding=8)
        env_bar.pack(fill=tk.X, **pad)
        ttk.Label(env_bar, textvariable=self.var_env_info, wraplength=860).pack(anchor=tk.W)

        # Step 1 & 2: Video + ROI selection
        top_pane = ttk.Frame(main_frame)
        top_pane.pack(fill=tk.X, **pad)

        left_col = ttk.Frame(top_pane)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        step1 = ttk.LabelFrame(left_col, text="1. Select BGMI Gameplay Video & Output Folder", padding=8)
        step1.pack(fill=tk.X, pady=(0, 8))

        row_vid = ttk.Frame(step1)
        row_vid.pack(fill=tk.X, pady=2)
        ttk.Entry(row_vid, textvariable=self.var_video_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(row_vid, text="Browse Video...", command=self._on_browse_video).pack(side=tk.RIGHT)

        ttk.Label(step1, textvariable=self.var_video_info, foreground="#334155").pack(anchor=tk.W, pady=(4, 6))

        row_out = ttk.Frame(step1)
        row_out.pack(fill=tk.X, pady=2)
        ttk.Label(row_out, text="Output Base:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(row_out, textvariable=self.var_output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(row_out, text="Browse...", command=self._on_browse_output).pack(side=tk.RIGHT)

        step2 = ttk.LabelFrame(left_col, text="2. Preview Frame & Select Kill-Feed Region (Mandatory)", padding=8)
        step2.pack(fill=tk.X)

        row_prev = ttk.Frame(step2)
        row_prev.pack(fill=tk.X, pady=2)
        ttk.Label(row_prev, text="Preview Timestamp (sec):").pack(side=tk.LEFT)
        ttk.Spinbox(row_prev, from_=0.0, to=86400.0, increment=5.0, textvariable=self.var_preview_sec, width=8).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(row_prev, text="Load Preview", command=self._on_load_preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_prev, text="Select Kill-Feed Region", command=self._on_select_roi).pack(side=tk.LEFT, padx=4)

        ttk.Label(step2, textvariable=self.var_roi_info, foreground="#0f766e", wraplength=480).pack(
            anchor=tk.W, pady=(6, 0)
        )

        # Preview thumbnail box
        prev_box = ttk.LabelFrame(top_pane, text="Preview & ROI Verification", padding=6)
        prev_box.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.preview_label = ttk.Label(prev_box, text="Load a video to preview frame & ROI", width=38, anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Step 3: Configuration
        step3 = ttk.LabelFrame(main_frame, text="3. Detection & Highlight Clipping Settings", padding=8)
        step3.pack(fill=tk.X, **pad)

        grid = ttk.Frame(step3)
        grid.pack(fill=tk.X)

        fields = [
            ("Sampling FPS:", self.var_sample_fps, 0.5, 30.0, 0.5, 0, 0),
            ("Confidence Threshold:", self.var_confidence, 0.40, 0.95, 0.02, 0, 2),
            ("Confirm Frames:", self.var_confirm_frames, 1, 5, 1, 0, 4),
            ("Pre-Kill Buffer (s):", self.var_pre_buffer, 0.0, 60.0, 1.0, 1, 0),
            ("Post-Kill Buffer (s):", self.var_post_buffer, 0.0, 60.0, 1.0, 1, 2),
            ("Merge Gap (s):", self.var_merge_gap, 0.0, 30.0, 1.0, 1, 4),
        ]
        for label, var, f_min, f_max, inc, r_idx, c_idx in fields:
            ttk.Label(grid, text=label).grid(row=r_idx, column=c_idx, sticky=tk.W, padx=(4, 4), pady=3)
            ttk.Spinbox(grid, from_=f_min, to=f_max, increment=inc, textvariable=var, width=8).grid(
                row=r_idx, column=c_idx + 1, sticky=tk.W, padx=(0, 14), pady=3
            )

        ign_row = ttk.Frame(step3)
        ign_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(ign_row, text="Optional Player IGN Filter:").pack(side=tk.LEFT)
        ttk.Entry(ign_row, textvariable=self.var_player_ign, width=22).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(ign_row, text="Only clips matching Player IGN", variable=self.var_only_player_kills).pack(
            side=tk.LEFT, padx=6
        )

        chk_row = ttk.Frame(step3)
        chk_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Checkbutton(chk_row, text="Preserve Audio", variable=self.var_preserve_audio).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(chk_row, text="Save Debug ROI Crops", variable=self.var_save_debug_roi).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(chk_row, text="Fast Stream Copy (keyframe-aligned)", variable=self.var_fast_copy).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Checkbutton(chk_row, text="Use GPU Decode if Available", variable=self.var_use_gpu).pack(side=tk.LEFT)

        cal_row = ttk.Frame(step3)
        cal_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(cal_row, text="Calibrate: Capture BGMI Kill-Feed Sample", command=self._on_capture_calibration).pack(
            side=tk.LEFT
        )
        ttk.Button(cal_row, text="Open Calibration Folder", command=lambda: open_folder_in_explorer(self.calibration.samples_dir)).pack(
            side=tk.LEFT, padx=8
        )

        # Step 4: Actions & Progress
        step4 = ttk.LabelFrame(main_frame, text="4. Process Video & Results", padding=8)
        step4.pack(fill=tk.BOTH, expand=True, **pad)

        btn_row = ttk.Frame(step4)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        self.btn_start = ttk.Button(btn_row, text="Start Processing", command=self._on_start_processing)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_cancel = ttk.Button(btn_row, text="Cancel", command=self._on_cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=(0, 12))
        self.btn_open_out = ttk.Button(btn_row, text="Open Output Folder", command=self._on_open_output, state=tk.DISABLED)
        self.btn_open_out.pack(side=tk.RIGHT)

        ttk.Label(btn_row, textvariable=self.var_events_count, font=("Segoe UI", 10, "bold")).pack(side=tk.RIGHT, padx=16)

        self.progress_bar = ttk.Progressbar(step4, mode="determinate", maximum=100.0)
        self.progress_bar.pack(fill=tk.X, pady=4)
        ttk.Label(step4, textvariable=self.var_progress_text).pack(anchor=tk.W, pady=(0, 6))

        self.log_text = tk.Text(step4, height=9, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _refresh_environment_status(self) -> None:
        ff_ok, ff_info = check_ffmpeg_available()
        ff_label = "FFmpeg OK" if ff_ok else "FFmpeg MISSING (install required)"
        ocr_label = f"OCR: {self.ocr_probe.backend_name}" if self.ocr_probe.is_available else "OCR: Not installed (visual/calibration only)"
        cal_count = self.calibration.reload()
        cal_label = f"BGMI Calibration Samples: {cal_count} ({'Active' if cal_count > 0 else 'Uncalibrated - add real BGMI crops for best accuracy'})"
        self.var_env_info.set(f"{ff_label}  |  {ocr_label}  |  {cal_label}")

    def _update_preview_thumbnail(self, bgr_img) -> None:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(320.0 / float(max(1, w)), 190.0 / float(max(1, h)))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self._preview_photo = ImageTk.PhotoImage(pil_img)
        self.preview_label.configure(image=self._preview_photo, text="")

    def _on_browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select BGMI Gameplay Video",
            filetypes=[
                ("Video Files", "*.mp4 *.mkv *.mov *.avi *.webm *.ts"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        self.var_video_path.set(path)
        self.selected_roi = None
        self.var_roi_info.set("Video changed — please select the kill-feed region.")
        try:
            self.video_meta = get_video_metadata(path)
            default_prev = min(10.0, self.video_meta.duration_sec * 0.15) if self.video_meta.duration_sec > 4 else 0.0
            self.var_preview_sec.set(round(default_prev, 1))
            self.var_video_info.set(
                f"{self.video_meta.width}x{self.video_meta.height} | "
                f"{self.video_meta.fps:.2f} fps | "
                f"{format_seconds_hms(self.video_meta.duration_sec)} | "
                f"Audio: {'Yes' if self.video_meta.has_audio else 'No'}"
                f"{' (VFR)' if self.video_meta.is_vfr else ''}"
            )
            self._on_load_preview()
        except Exception as exc:
            self.video_meta = None
            messagebox.showerror("Video Error", str(exc))

    def _on_browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Select Base Output Directory")
        if folder:
            self.var_output_dir.set(folder)

    def _on_load_preview(self) -> None:
        video_path = self.var_video_path.get().strip()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showwarning("No Video", "Please select a valid video file first.")
            return
        try:
            frame, actual_ts = read_preview_frame(video_path, float(self.var_preview_sec.get()))
            if self.selected_roi is not None:
                frame = render_roi_preview_image(frame, self.selected_roi)
            self._update_preview_thumbnail(frame)
            self._log(f"Loaded preview frame at {actual_ts:.2f}s.")
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def _on_select_roi(self) -> None:
        video_path = self.var_video_path.get().strip()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showwarning("No Video", "Please select a valid video file first.")
            return
        try:
            roi = select_kill_feed_roi(video_path, preview_time_sec=float(self.var_preview_sec.get()))
            self.selected_roi = roi
            self.var_roi_info.set(
                f"Selected ROI (native {roi.source_width}x{roi.source_height}): "
                f"x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}"
            )
            frame, _ = read_preview_frame(video_path, float(self.var_preview_sec.get()))
            self._update_preview_thumbnail(render_roi_preview_image(frame, roi))
            self._log(f"ROI confirmed: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")
        except ROISelectionError as exc:
            self.selected_roi = None
            self.var_roi_info.set("Kill-feed region selection cancelled or invalid.")
            messagebox.showerror("Kill-Feed Region Required", str(exc))
        except Exception as exc:
            messagebox.showerror("ROI Error", str(exc))

    def _on_capture_calibration(self) -> None:
        video_path = self.var_video_path.get().strip()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showwarning("No Video", "Select a BGMI video and set preview timestamp to a frame showing a kill feed.")
            return
        try:
            saved_path = self.calibration.capture_sample_interactively(
                video_path=video_path,
                timestamp_sec=float(self.var_preview_sec.get()),
                label="bgmi_killfeed",
                roi=self.selected_roi,
            )
            self._refresh_environment_status()
            if saved_path:
                self._log(f"Saved BGMI calibration sample: {saved_path}")
                messagebox.showinfo("Sample Saved", f"Saved calibration sample:\n{saved_path}")
            else:
                self._log("Calibration sample capture cancelled.")
        except Exception as exc:
            messagebox.showerror("Calibration Error", str(exc))

    def _on_start_processing(self) -> None:
        video_path = self.var_video_path.get().strip()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showwarning("Missing Video", "Please select a valid BGMI gameplay video.")
            return
        if self.selected_roi is None or not self.selected_roi.is_valid():
            messagebox.showwarning(
                "Kill-Feed Region Required",
                "You must select the kill-feed region before processing.",
            )
            return

        ff_ok, ff_msg = check_ffmpeg_available()
        if not ff_ok:
            messagebox.showerror("FFmpeg Not Found", ff_msg)
            return

        try:
            cfg = DetectionConfig(
                sample_fps=float(self.var_sample_fps.get()),
                confidence_threshold=float(self.var_confidence.get()),
                min_confirm_frames=int(self.var_confirm_frames.get()),
                pre_event_buffer_sec=float(self.var_pre_buffer.get()),
                post_event_buffer_sec=float(self.var_post_buffer.get()),
                merge_gap_sec=float(self.var_merge_gap.get()),
                player_ign=self.var_player_ign.get().strip(),
                only_player_kills=bool(self.var_only_player_kills.get()),
                preserve_audio=bool(self.var_preserve_audio.get()),
                save_debug_roi=bool(self.var_save_debug_roi.get()),
                fast_stream_copy=bool(self.var_fast_copy.get()),
                use_gpu_if_available=bool(self.var_use_gpu.get()),
            )
            cfg.validate()
        except Exception as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            return

        out_base = self.var_output_dir.get().strip() or os.getcwd()
        self._cancel_requested = False
        self.last_result = None
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_cancel.configure(state=tk.NORMAL)
        self.btn_open_out.configure(state=tk.DISABLED)
        self.progress_bar["value"] = 0.0
        self.var_events_count.set("Confirmed Kills: 0")
        self._log("Starting BGMI kill-feed analysis...")

        roi_copy = self.selected_roi

        def _worker() -> None:
            try:
                def _cb(fraction: float, msg: str, count: int) -> None:
                    self.root.after(0, lambda: self._on_worker_progress(fraction, msg, count))

                res = run_bgmi_pipeline(
                    video_path=video_path,
                    roi=roi_copy,
                    config=cfg,
                    base_output_dir=out_base,
                    use_current_dir=False,
                    status_callback=_cb,
                    cancel_check=lambda: self._cancel_requested,
                )
                self.root.after(0, lambda: self._on_worker_done(res, None))
            except Exception as exc:
                self.root.after(0, lambda: self._on_worker_done(None, exc))

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def _on_worker_progress(self, fraction: float, msg: str, count: int) -> None:
        self.progress_bar["value"] = max(0.0, min(100.0, fraction * 100.0))
        self.var_progress_text.set(msg)
        self.var_events_count.set(f"Confirmed Kills: {count}")

    def _on_worker_done(self, result: Optional[PipelineResult], error: Optional[Exception]) -> None:
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_cancel.configure(state=tk.DISABLED)

        if error is not None:
            self.var_progress_text.set("Processing failed.")
            self._log(f"ERROR: {error}")
            messagebox.showerror("Processing Failed", str(error))
            return

        assert result is not None
        self.last_result = result
        self.progress_bar["value"] = 100.0
        n_kills = len(result.scan_summary.events)
        n_clips = sum(1 for c in result.clip_results if c.success)
        self.var_events_count.set(f"Confirmed Kills: {n_kills}")
        self.var_progress_text.set(
            f"Done! Confirmed Kills: {n_kills} | Highlight Clips Created: {n_clips}"
        )
        self.btn_open_out.configure(state=tk.NORMAL)
        self._log(f"Session folder: {result.session_paths.root_dir}")
        self._log(f"Confirmed kill-feed events: {n_kills}, Highlight clips: {n_clips}")

        if n_kills == 0:
            messagebox.showinfo(
                "Processing Complete",
                f"Processing finished with 0 confirmed kill-feed events.\n"
                f"No screenshots or highlight clips were generated.\n\n"
                f"Session log saved to:\n{result.session_paths.root_dir}",
            )
        else:
            messagebox.showinfo(
                "Processing Complete",
                f"Confirmed Kill-Feed Events: {n_kills}\n"
                f"Highlight Clips Generated: {n_clips}\n\n"
                f"Output saved to:\n{result.session_paths.root_dir}",
            )

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        self.var_progress_text.set("Cancelling...")
        self._log("Cancellation requested by user...")

    def _on_open_output(self) -> None:
        if self.last_result:
            open_folder_in_explorer(self.last_result.session_paths.root_dir)


def launch_gui() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    BGMIDetectorApp(root)
    root.mainloop()
