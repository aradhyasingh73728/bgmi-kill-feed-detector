# PEACE — BGMI Kill Feed Detector & Automatic Highlight Clipping Software

**PEACE (Processing Engine for Adaptive Content Extraction)** has been upgraded into a conservative, CPU-friendly **BGMI (Battlegrounds Mobile India) Kill-Feed Detection and Highlight Clipping** desktop tool for Windows 10/11 and Python 3.11.

---

## What Was Upgraded & Fixed

1. **Mandatory Manual Kill-Feed Region Selection (`roi_selector.py`)**:
   - Removed hardcoded `1415, 74, 364, 183` crop coordinates from `screenshotting.py`.
   - Uses OpenCV `cv2.selectROI()` on a scaled preview frame for every video and maps coordinates accurately back to native video resolution (`720p`, `1080p`, `1440p`, `2400×1080` Android ultrawide, etc.).
   - Cancelling or selecting an invalid/tiny region aborts cleanly with a clear error.

2. **Conservative BGMI Kill-Feed Detection (`bgmi_detector.py`, `ocr_engine.py`, `calibration.py`)**:
   - Analyzes **only** the user-selected kill-feed ROI.
   - Segments individual horizontal kill-feed rows (supporting multiple stacked entries simultaneously).
   - Evaluates BGMI banner contrast/geometry, 3-zone `[Killer] [Weapon/Icon] [Victim]` layout, OCR grammar/keyword validation, UI-noise rejection (lobby, FPS, ping, timers, inventory, compass), and optional user-calibrated BGMI reference matching.
   - Legacy non-BGMI templates from the original repo were archived in `Scripts/KillFeed/legacy_non_bgmi/` and are **not** loaded automatically.

3. **Temporal & Content-Aware Deduplication (`deduplicator.py`)**:
   - Tracks each kill-feed row over time using normalized OCR text similarity and visual row fingerprints.
   - Prevents duplicate screenshots and duplicate clips when a kill banner stays on screen across consecutive frames, while still detecting genuinely different kills occurring close together.

4. **In-Memory Streaming & Confirmed-Only Screenshots (`scanning.py`, `screenshotting.py`)**:
   - No longer extracts every video frame to disk.
   - Reads frames in memory and saves full-resolution gameplay PNGs (`screenshots/kill_001_00-01-25.png`) **only after** an event is confirmed across `min_confirm_frames`.

5. **Safe Highlight Clipping & Grouping (`timeGrouping.py`, `clipping.py`)**:
   - Fixed the original `scanning.py` reversed FPS fraction bug (`fps[::-1]`), `timeGrouping.py` un-updated `last_time` bug, negative clip start times (`-10s` near video start), and exceeding video duration.
   - Preserves audio by default (`-c:a aac` / `-c:v libx264`), skips clipping when 0 events are found, and runs CPU-first (no mandatory CUDA).

---

## Project Structure

```text
PEACE/
├── Scripts/
│   ├── main.py                      # Unified GUI + CLI entry point
│   ├── KillFeed/
│   │   ├── README.txt               # Explains archived legacy templates
│   │   └── legacy_non_bgmi/         # Original non-BGMI Capture*.PNG (disabled)
│   ├── calibration/
│   │   ├── README.txt
│   │   └── samples/                 # Place real BGMI kill-feed crops here
│   └── modules/
│       ├── __init__.py
│       ├── config.py                # ROIRegion, DetectionConfig, KillEvent
│       ├── createDir.py             # BGMI_Output/Session_Date_Time/ setup
│       ├── video_utils.py           # FFmpeg check, VFR-safe metadata, preview
│       ├── roi_selector.py          # Mandatory cv2.selectROI() workflow
│       ├── ocr_engine.py            # CPU OCR (RapidOCR / Tesseract / fallback)
│       ├── calibration.py           # Real BGMI sample capture & matching
│       ├── bgmi_detector.py         # Multi-gate conservative BGMI detector
│       ├── deduplicator.py          # Multi-row temporal deduplication
│       ├── screenshotting.py        # Confirmed-only full-frame screenshots
│       ├── scanning.py              # In-memory streaming & CSV logging
│       ├── timeGrouping.py          # Clamped interval grouping & merging
│       ├── clipping.py              # FFmpeg highlight clipper
│       ├── pipeline.py              # End-to-end session orchestrator
│       └── gui.py                   # Step-guided Tkinter desktop UI
├── ImageLocator/
│   └── locate_image.py              # OpenCV template locator helper
├── tests/
│   └── test_bgmi_pipeline.py        # Automated test suite
├── requirements.txt
└── README.md
```

---

## Windows 10/11 Installation (PowerShell)

> **Important:** `pip install ffmpeg-python` does **not** install the `ffmpeg.exe` binary. You must install FFmpeg separately and have it on your system `PATH`.

Run these exact commands in **Windows PowerShell**:

```powershell
# 1. Navigate to your project folder
cd C:\PEACE-main

# 2. Check Python 3.11 is installed
python --version

# 3. Create and activate a Python 3.11 virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Install FFmpeg (required for video metadata & highlight clipping)
winget install Gyan.FFmpeg

# 6. Close and reopen PowerShell (or refresh PATH), reactivate venv, and verify FFmpeg
.\.venv\Scripts\Activate.ps1
ffmpeg -version
ffprobe -version
```

*(Optional)* If you prefer Tesseract OCR instead of `rapidocr-onnxruntime`:
```powershell
winget install UB-Mannheim.TesseractOCR
```

---

## How to Launch the Software

### Option A: Desktop GUI Mode (`Scripts\main.py`)
```powershell
cd C:\PEACE-main
.\.venv\Scripts\Activate.ps1
python Scripts\main.py
```

### Option B: Local Web UI Mode (Zero-Upload PC Path + Fast ROI Scan)
Requires Node.js 20+:
```powershell
cd C:\PEACE-main
npm install
npm run dev
```
Open `http://localhost:3000` in Chrome/Edge and either:
- Paste your local video path directly (`C:\Videos\bgmi_30min.mp4`) for **0-upload instant scanning & MP4 clipping**, OR
- Pick any video in browser to preview immediately, run **Fast ROI-Only Scan**, or use **6× Parallel Chunk Upload**.

### Option C: Command-Line Mode (CLI)
```powershell
python Scripts\main.py --cli --video "C:\path\to\bgmi_recording.mp4"
```

---

## How to Select the Kill-Feed Region (Mandatory Step)

1. Select your BGMI recording (PC/laptop or Android phone/tablet screen recording).
2. Set **Preview Timestamp (sec)** to a moment where the in-game HUD is visible (not a loading screen) and click **Load Preview**.
3. Click **Select Kill-Feed Region**. An OpenCV preview window opens (automatically scaled to fit your monitor).
4. Click and drag a rectangle covering the area where BGMI kill-feed banners appear on your HUD layout (leave enough vertical height for 2–4 stacked rows if multiple kills happen together).
5. Press **ENTER** or **SPACE** to confirm (or **C** / **ESC** to cancel).
6. Coordinates are automatically converted back to the video's native resolution (`1280×720`, `1920×1080`, `2560×1440`, etc.).

---

## Calibrating with Real BGMI Samples

Because this repository does not invent or ship fake BGMI weapon/icon templates, **real-world BGMI accuracy improves significantly once you add 3–10 real BGMI samples** from your own device/HUD:

1. In the GUI, load your BGMI video and seek **Preview Timestamp** to a frame where a kill-feed banner is visible.
2. Click **Calibrate: Capture BGMI Kill-Feed Sample**.
3. Draw a tight box around a BGMI weapon icon, knock/headshot icon, or kill-feed strip and press **ENTER**.
4. Samples are saved in `Scripts/calibration/samples/` and automatically loaded for multi-scale matching in future sessions.

CLI equivalent:
```powershell
python Scripts\main.py --calibrate --video "C:\path\to\bgmi_recording.mp4" --preview-time 85 --sample-label bgmi_m416
```

---

## Output Folder Structure

Every run creates a unique timestamped session directory:

```text
BGMI_Output/
    Session_2026-11-20_14-30-12/
        screenshots/
            kill_001_00-01-25.png
            kill_002_00-03-42.png
        clips/
            highlight_001.mp4
            highlight_002.mp4
        debug/
            detected_region/
                kill_001_00-01-25_roi.png
            logs/
                session.log
            grouping.txt
            grouping.json
            session_config.json
        events.csv
        summary.txt
```

If **0 confirmed kill-feed events** are detected, `screenshots/` and `clips/` remain empty.

---

## Testing with Your Own BGMI Recordings

1. Run the automated verification suite first:
   ```powershell
   python -m unittest discover -s tests -v
   ```
2. Test on a short 1–2 minute BGMI recording containing:
   - Pure looting / running / lobby / map / inventory footage (confirm 0 false positives).
   - 1–2 confirmed kills (confirm `screenshots/` has full-frame PNGs and `clips/` has ±10s MP4s with audio).
   - Rapid double-kills or squad wipes (confirm separate events in `events.csv` and merged non-overlapping clips).
3. Inspect `debug/detected_region/` and `debug/logs/session.log` if you want to tune `Confidence Threshold` (default `0.72`) or `Sampling FPS` (default `2.0`).

---

## Accuracy & Validation Note

Automated tests in `tests/test_bgmi_pipeline.py` validate the full video pipeline using synthetic BGMI-style banners, UI noise, multi-resolution ROI mapping, deduplication, and FFmpeg clipping. **Real-world BGMI footage accuracy has not yet been validated against live recordings in this environment** because no real BGMI gameplay recordings or BGMI reference templates are included in the repository. Treat default thresholds conservatively and calibrate with samples from your own BGMI recordings for best results.
