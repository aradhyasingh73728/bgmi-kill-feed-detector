from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SessionPaths:
    """All output directories and files for a single BGMI processing session."""
    root_dir: str
    screenshots_dir: str
    clips_dir: str
    debug_dir: str
    debug_region_dir: str
    logs_dir: str
    events_csv: str
    summary_txt: str
    grouping_txt: str
    grouping_json: str
    log_file: str


def create_session_directories(
    base_output_dir: Optional[str] = None,
    use_current_dir: bool = True,
) -> SessionPaths:
    """
    Create a unique session folder matching the required layout:

    BGMI_Output/
        Session_YYYY-MM-DD_HH-MM-SS/
            screenshots/
            clips/
            debug/
                detected_region/
                logs/
            events.csv
            summary.txt
    """
    if use_current_dir or not base_output_dir:
        parent_dir = os.path.join(os.getcwd(), "BGMI_Output")
    else:
        cleaned = os.path.abspath(base_output_dir.strip())
        if not os.path.exists(cleaned):
            raise FileNotFoundError(f"Output base directory does not exist: '{cleaned}'")
        if not os.path.isdir(cleaned):
            raise NotADirectoryError(f"Output base path is not a directory: '{cleaned}'")
        if not os.access(cleaned, os.W_OK):
            raise PermissionError(f"Output base directory is not writable: '{cleaned}'")
        # Nest inside BGMI_Output unless user explicitly chose a folder already named BGMI_Output
        if os.path.basename(cleaned).lower() == "bgmi_output":
            parent_dir = cleaned
        else:
            parent_dir = os.path.join(cleaned, "BGMI_Output")

    os.makedirs(parent_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_name = f"Session_{timestamp}"
    session_root = os.path.join(parent_dir, session_name)

    counter = 1
    while os.path.exists(session_root):
        session_root = os.path.join(parent_dir, f"{session_name}_{counter}")
        counter += 1

    screenshots_dir = os.path.join(session_root, "screenshots")
    clips_dir = os.path.join(session_root, "clips")
    debug_dir = os.path.join(session_root, "debug")
    debug_region_dir = os.path.join(debug_dir, "detected_region")
    logs_dir = os.path.join(debug_dir, "logs")

    for path in (screenshots_dir, clips_dir, debug_region_dir, logs_dir):
        os.makedirs(path, exist_ok=True)

    return SessionPaths(
        root_dir=session_root,
        screenshots_dir=screenshots_dir,
        clips_dir=clips_dir,
        debug_dir=debug_dir,
        debug_region_dir=debug_region_dir,
        logs_dir=logs_dir,
        events_csv=os.path.join(session_root, "events.csv"),
        summary_txt=os.path.join(session_root, "summary.txt"),
        grouping_txt=os.path.join(debug_dir, "grouping.txt"),
        grouping_json=os.path.join(debug_dir, "grouping.json"),
        log_file=os.path.join(logs_dir, "session.log"),
    )


def configure_session_logger(log_file: str, name: str = "peace_bgmi") -> logging.Logger:
    """Configure a file + console logger for the session without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear existing handlers if reusing logger across sessions in GUI
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# Backward-compatible wrapper preserving original createDir signature intent
def createDir(base_directory: Optional[str], use_current_dir: bool) -> Optional[str]:
    try:
        paths = create_session_directories(base_directory, use_current_dir)
        return paths.root_dir
    except Exception as exc:
        print(f"Error creating directory: {exc}")
        return None
