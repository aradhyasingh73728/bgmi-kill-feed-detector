LEGACY KILL FEED TEMPLATES NOTICE
=================================

The PNG files previously located directly in Scripts/KillFeed/ (Capture0.PNG .. Capture10.PNG)
from the original PEACE repository have been moved to:

    Scripts/KillFeed/legacy_non_bgmi/

Why?
Those original templates were captured from a non-BGMI layout at fixed pixel dimensions
(~81x20px) and cannot reliably represent BGMI kill-feed entries across different player names,
weapons, devices, and screen resolutions.

They are NOT loaded automatically by the upgraded BGMI Kill-Feed Detector.

To calibrate real BGMI reference samples (weapon icons, knock/headshot icons, or banner strips)
from your own BGMI recordings, use the built-in Calibration Tool in the GUI or run:

    python Scripts/main.py --calibrate --video path/to/your_bgmi_recording.mp4

Samples are stored in:
    Scripts/calibration/samples/
