from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

from modules.config import KillEvent


@dataclass
class ClipSegment:
    clip_index: int
    start_sec: float
    end_sec: float
    event_ids: List[int]
    event_timestamps: List[float]

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def group_events_into_clips(
    events: Sequence[KillEvent],
    video_duration_sec: float,
    pre_buffer_sec: float = 10.0,
    post_buffer_sec: float = 10.0,
    merge_gap_sec: float = 4.0,
    grouping_txt_path: Optional[str] = None,
    grouping_json_path: Optional[str] = None,
) -> List[ClipSegment]:
    """
    Convert confirmed KillEvents into non-negative, duration-clamped highlight clip
    intervals and merge overlapping or nearby intervals cleanly.

    Fixes the original PEACE bugs where `last_time` failed to advance inside a group
    and `start_time` could become negative.
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.timestamp_sec)
    max_dur = float(video_duration_sec) if video_duration_sec and video_duration_sec > 0 else float("inf")

    segments: List[ClipSegment] = []
    for ev in sorted_events:
        raw_start = max(0.0, float(ev.timestamp_sec) - float(pre_buffer_sec))
        raw_end = min(max_dur, float(ev.timestamp_sec) + float(post_buffer_sec))
        if raw_end <= raw_start:
            continue

        if not segments:
            segments.append(
                ClipSegment(
                    clip_index=1,
                    start_sec=raw_start,
                    end_sec=raw_end,
                    event_ids=[ev.event_id],
                    event_timestamps=[ev.timestamp_sec],
                )
            )
        else:
            prev = segments[-1]
            # Merge if buffered clip windows overlap or lie within merge_gap_sec
            if raw_start <= prev.end_sec + float(merge_gap_sec):
                prev.end_sec = max(prev.end_sec, raw_end)
                prev.event_ids.append(ev.event_id)
                prev.event_timestamps.append(ev.timestamp_sec)
            else:
                segments.append(
                    ClipSegment(
                        clip_index=len(segments) + 1,
                        start_sec=raw_start,
                        end_sec=raw_end,
                        event_ids=[ev.event_id],
                        event_timestamps=[ev.timestamp_sec],
                    )
                )

    if grouping_txt_path:
        os.makedirs(os.path.dirname(grouping_txt_path), exist_ok=True)
        with open(grouping_txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                ts_list = [round(t, 3) for t in seg.event_timestamps]
                f.write(
                    f"Highlight-{seg.clip_index:03d}: "
                    f"[{seg.start_sec:.2f}s -> {seg.end_sec:.2f}s] "
                    f"events={seg.event_ids} timestamps={ts_list}\n"
                )

    if grouping_json_path:
        os.makedirs(os.path.dirname(grouping_json_path), exist_ok=True)
        with open(grouping_json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in segments], f, indent=2)

    return segments


# Backward-compatible helper wrapper for legacy integer-timestamp callers
def timeGrouping(
    outputDir: str,
    input_time: List[int],
    video_duration_sec: float = 0.0,
    pre_buffer_sec: float = 10.0,
    post_buffer_sec: float = 10.0,
) -> List[ClipSegment]:
    if not input_time:
        return []
    pseudo_events = [
        KillEvent(
            event_id=idx + 1,
            timestamp_sec=float(t),
            frame_index=0,
            confidence=1.0,
            ocr_text="",
            killer="",
            victim="",
            weapon_or_action="",
            row_bbox_in_roi=(0, 0, 0, 0),
            is_highlighted=False,
        )
        for idx, t in enumerate(sorted(input_time))
    ]
    return group_events_into_clips(
        events=pseudo_events,
        video_duration_sec=video_duration_sec,
        pre_buffer_sec=pre_buffer_sec,
        post_buffer_sec=post_buffer_sec,
        grouping_txt_path=os.path.join(outputDir, "grouping.txt"),
    )
