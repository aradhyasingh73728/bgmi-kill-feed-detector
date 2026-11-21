from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import cv2


def locate_image(
    target_path: str,
    search_path: str,
    confidence_level: float = 0.85,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate a target template image within a larger screenshot using OpenCV multi-scale
    template matching (no pyautogui/screen dependency required).

    Returns (left, top, width, height) in native pixel coordinates, or None.
    """
    if not os.path.isfile(target_path) or not os.path.isfile(search_path):
        print(f"Error: File not found. Check paths: '{target_path}', '{search_path}'.")
        return None

    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if target is None or search is None:
        print("Error: Unable to read target or search image.")
        return None

    th, tw = target.shape[:2]
    sh, sw = search.shape[:2]
    if tw > sw or th > sh:
        print("Error: Target image is larger than search image.")
        return None

    res = cv2.matchTemplate(search, target, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= confidence_level:
        return (int(max_loc[0]), int(max_loc[1]), int(tw), int(th))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="PEACE Image Locator Helper (OpenCV)")
    parser.add_argument("--target", default="a2.jpg", help="Cropped template image path")
    parser.add_argument("--search", default="a1.jpg", help="Full screenshot image path")
    parser.add_argument("--confidence", type=float, default=0.85, help="Match threshold [0..1]")
    args = parser.parse_args()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    target_path = args.target if os.path.isabs(args.target) else os.path.join(current_directory, args.target)
    search_path = args.search if os.path.isabs(args.search) else os.path.join(current_directory, args.search)

    result = locate_image(target_path, search_path, args.confidence)
    if result:
        x, y, w, h = result
        print(f"Match found: crop_x={x}, crop_y={y}, crop_width={w}, crop_height={h}")
    else:
        print("No match found above confidence threshold.")


if __name__ == "__main__":
    main()
