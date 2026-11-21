export interface CalibrationSampleInfo {
  name: string;
  filename: string;
  width: number;
  height: number;
}

export interface SystemStatus {
  python_version: string;
  opencv_version: string;
  ffmpeg_available: boolean;
  ffmpeg_info: string;
  ffprobe_available: boolean;
  ocr_backend: string;
  ocr_available: boolean;
  calibration_dir: string;
  calibration_samples: CalibrationSampleInfo[];
}

export interface VideoMetadata {
  path: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_sec: number;
  has_audio: boolean;
  is_vfr: boolean;
  codec_name: string;
}

export interface UploadedVideoInfo {
  videoId: string;
  originalName: string;
  isSyntheticDemo: boolean;
  isLocalFile?: boolean;
  serverReady?: boolean;
  localObjectUrl?: string;
  suggestedRoi?: ROIRegion;
  metadata: VideoMetadata;
}

export interface ROIRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionConfigState {
  sample_fps: number;
  confidence_threshold: number;
  min_confirm_frames: number;
  pre_event_buffer_sec: number;
  post_event_buffer_sec: number;
  merge_gap_sec: number;
  player_ign: string;
  only_player_kills: boolean;
  preserve_audio: boolean;
  save_debug_roi: boolean;
  fast_stream_copy: boolean;
}

export interface KillEventItem {
  event_id: number;
  timestamp_sec: number;
  timestamp_hms: string;
  confidence: number;
  killer: string;
  weapon_or_action: string;
  victim: string;
  ocr_text: string;
  is_highlighted: boolean;
  screenshot_file: string;
  debug_roi_file: string;
}

export interface ClipItem {
  clip_index: number;
  filename: string;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  success: boolean;
  error_message: string;
}

export interface ClipSegmentItem {
  clip_index: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  event_ids: number[];
}

export interface JobStatus {
  state: 'running' | 'completed' | 'failed';
  scan_mode?: 'full' | 'roi_only';
  progress: number;
  message: string;
  confirmed_count: number;
  session_dir?: string;
  ocr_backend?: string;
  calibrated_samples_count?: number;
  frames_analyzed?: number;
  elapsed_sec?: number;
  events?: KillEventItem[];
  clips?: ClipItem[];
  clip_segments?: ClipSegmentItem[];
  summary_text?: string;
  error?: string;
}
