import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Crosshair,
  Download,
  Film,
  FolderOpen,
  HardDrive,
  Pause,
  Play,
  Scissors,
  SkipBack,
  SkipForward,
  Sliders,
  Terminal,
  Trash2,
  Upload,
  Video,
  X,
  Zap,
} from 'lucide-react';
import { RoiCanvas } from './components/RoiCanvas';
import type {
  ClipSegmentItem,
  DetectionConfigState,
  JobStatus,
  KillEventItem,
  ROIRegion,
  SystemStatus,
  UploadedVideoInfo,
} from './types';

type ActiveTab = 'workspace' | 'results' | 'calibration' | 'setup';
type BrowserScanMode = 'roi_only' | 'full_upload';

interface UploadProgressState {
  phase: 'sampling_roi' | 'uploading' | 'inspecting';
  filename: string;
  percent: number;
  loadedMB: number;
  totalMB: number;
  detail?: string;
}

interface LocalScreenshotItem {
  dataUrl: string;
  filename: string;
}

interface LocalRecordedClip {
  clip_index: number;
  blobUrl: string;
  filename: string;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
}

const CHUNK_SIZE = 4 * 1024 * 1024; // 4 MB chunks stay well below proxy request limits
const UPLOAD_CONCURRENCY = 6; // Parallel connections saturate bandwidth cleanly
const ROI_BATCH_SIZE = 45;

async function parseJsonResponse<T = Record<string, unknown>>(resp: Response): Promise<T> {
  const raw = await resp.text();
  let parsed: unknown = null;
  if (raw) {
    try {
      parsed = JSON.parse(raw);
    } catch {
      const snippet = raw.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 180);
      throw new Error(
        resp.ok
          ? 'Server returned an unexpected non-JSON response.'
          : `HTTP ${resp.status}: ${snippet || resp.statusText || 'Request failed'}`
      );
    }
  }
  if (!resp.ok) {
    const errObj = parsed as { error?: string; message?: string } | null;
    throw new Error(errObj?.error || errObj?.message || `HTTP ${resp.status}: Request failed`);
  }
  return (parsed ?? {}) as T;
}

const DEFAULT_CONFIG: DetectionConfigState = {
  sample_fps: 3.0,
  confidence_threshold: 0.65,
  min_confirm_frames: 2,
  pre_event_buffer_sec: 8.0,
  post_event_buffer_sec: 6.0,
  merge_gap_sec: 4.0,
  player_ign: '',
  only_player_kills: false,
  preserve_audio: true,
  save_debug_roi: true,
  fast_stream_copy: false,
};

function formatDuration(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  return h > 0
    ? `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(rem).padStart(2, '0')}`
    : `${String(m).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
}

function formatHmsTag(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  return `${String(h).padStart(2, '0')}-${String(m).padStart(2, '0')}-${String(rem).padStart(2, '0')}`;
}

function waitVideoSeek(video: HTMLVideoElement, targetSec: number, timeoutMs = 4500): Promise<void> {
  return new Promise((resolve, reject) => {
    const clamped = Math.max(0, Math.min(Math.max(0, (video.duration || targetSec) - 0.02), targetSec));
    let settled = false;
    const cleanup = () => {
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
      window.clearTimeout(timer);
    };
    const onSeeked = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };
    const onError = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error('Browser could not decode video frame at target timestamp.'));
    };
    const timer = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    }, timeoutMs);

    if (Math.abs(video.currentTime - clamped) < 0.01 && video.readyState >= 2) {
      cleanup();
      resolve();
      return;
    }
    video.addEventListener('seeked', onSeeked, { once: true });
    video.addEventListener('error', onError, { once: true });
    video.currentTime = clamped;
  });
}

async function probeLocalVideoFile(file: File): Promise<{
  objectUrl: string;
  width: number;
  height: number;
  duration: number;
  initialTs: number;
  initialPreviewUrl: string;
}> {
  const objectUrl = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = objectUrl;

  try {
    await new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        video.removeEventListener('loadeddata', onReady);
        video.removeEventListener('error', onErr);
        window.clearTimeout(timer);
      };
      const onReady = () => {
        cleanup();
        resolve();
      };
      const onErr = () => {
        cleanup();
        reject(new Error('Unsupported browser video codec'));
      };
      const timer = window.setTimeout(() => {
        cleanup();
        if (video.videoWidth > 0 && video.duration > 0) {
          resolve();
        } else {
          reject(new Error('Timed out probing local video'));
        }
      }, 6000);

      video.addEventListener('loadeddata', onReady, { once: true });
      video.addEventListener('error', onErr, { once: true });
    });

    const width = video.videoWidth;
    const height = video.videoHeight;
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    if (!width || !height || duration <= 0) {
      throw new Error('Could not read video dimensions locally');
    }

    const initialTs = duration > 4 ? Math.min(10, Number((duration * 0.15).toFixed(1))) : 0;
    if (initialTs > 0) {
      await waitVideoSeek(video, initialTs);
    }

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Canvas 2D context unavailable');
    }
    ctx.drawImage(video, 0, 0, width, height);
    const initialPreviewUrl = canvas.toDataURL('image/png');

    return { objectUrl, width, height, duration, initialTs, initialPreviewUrl };
  } catch (err) {
    URL.revokeObjectURL(objectUrl);
    throw err;
  } finally {
    video.removeAttribute('src');
    video.load();
  }
}

async function captureFrameDataUrl(objectUrl: string, ts: number): Promise<string> {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = objectUrl;
  try {
    await new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        video.removeEventListener('loadeddata', onReady);
        video.removeEventListener('error', onErr);
      };
      const onReady = () => {
        cleanup();
        resolve();
      };
      const onErr = () => {
        cleanup();
        reject(new Error('Failed to decode local frame'));
      };
      video.addEventListener('loadeddata', onReady, { once: true });
      video.addEventListener('error', onErr, { once: true });
    });
    await waitVideoSeek(video, ts);
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D unavailable');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
  } finally {
    video.removeAttribute('src');
    video.load();
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workspace');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [systemError, setSystemError] = useState<string>('');

  const [videoInfo, setVideoInfo] = useState<UploadedVideoInfo | null>(null);
  const [localPathInput, setLocalPathInput] = useState<string>('');
  const [browserScanMode, setBrowserScanMode] = useState<BrowserScanMode>('roi_only');

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgressState | null>(null);
  const [uploadError, setUploadError] = useState<string>('');

  const uploadAbortRef = useRef<AbortController | null>(null);
  const activeUploadIdRef = useRef<string | null>(null);
  const selectedLocalFileRef = useRef<File | null>(null);
  const localObjectUrlRef = useRef<string | null>(null);

  const [previewTs, setPreviewTs] = useState<number>(2.0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [stageMode, setStageMode] = useState<'player' | 'roi'>('roi');
  const [videoTime, setVideoTime] = useState<number>(0);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [videoPlayerError, setVideoPlayerError] = useState<string>('');
  const videoElRef = useRef<HTMLVideoElement | null>(null);

  const [roi, setRoi] = useState<ROIRegion | null>(null);
  const [config, setConfig] = useState<DetectionConfigState>(DEFAULT_CONFIG);

  const [calibrationLabel, setCalibrationLabel] = useState('bgmi_killfeed');
  const [calibrationNotice, setCalibrationNotice] = useState<string>('');

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [processError, setProcessError] = useState<string>('');

  const [localScreenshots, setLocalScreenshots] = useState<Record<number, LocalScreenshotItem>>({});
  const [localClips, setLocalClips] = useState<LocalRecordedClip[]>([]);
  const [clipRecordingMsg, setClipRecordingMsg] = useState<string>('');
  const [isRecordingClips, setIsRecordingClips] = useState(false);

  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [eventSearch, setEventSearch] = useState('');

  const fetchSystemStatus = async () => {
    try {
      const resp = await fetch('/api/system-status');
      const data = await parseJsonResponse<SystemStatus>(resp);
      setSystemStatus(data);
      setSystemError('');
    } catch (err) {
      setSystemError(err instanceof Error ? err.message : 'Unable to connect to backend');
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    return () => {
      if (localObjectUrlRef.current) {
        URL.revokeObjectURL(localObjectUrlRef.current);
      }
    };
  }, []);

  const replaceLocalMedia = (newUrl: string | null, file: File | null) => {
    if (localObjectUrlRef.current && localObjectUrlRef.current !== newUrl) {
      URL.revokeObjectURL(localObjectUrlRef.current);
    }
    localObjectUrlRef.current = newUrl;
    selectedLocalFileRef.current = file;
    setLocalScreenshots({});
    setLocalClips((prev) => {
      prev.forEach((c) => URL.revokeObjectURL(c.blobUrl));
      return [];
    });
  };

  const loadPreviewAt = async (info: UploadedVideoInfo, ts: number) => {
    const clamped = Math.max(0, Number(ts.toFixed(2)));
    setPreviewTs(clamped);

    if (info.localObjectUrl) {
      setIsLoadingPreview(true);
      try {
        const dataUrl = await captureFrameDataUrl(info.localObjectUrl, clamped);
        setPreviewUrl(dataUrl);
        return;
      } catch {
        // Fallback to server preview if serverReady is true
      } finally {
        setIsLoadingPreview(false);
      }
    }

    if (info.serverReady && info.videoId) {
      setIsLoadingPreview(true);
      const url = `/api/preview-frame?videoId=${encodeURIComponent(info.videoId)}&ts=${encodeURIComponent(clamped)}&t=${Date.now()}`;
      setPreviewUrl(url);
    }
  };

  const seekPlayerTo = (targetSec: number) => {
    if (!videoInfo) return;
    const maxDur = Math.max(0, videoInfo.metadata.duration_sec || 0);
    const next = Math.max(0, Math.min(maxDur, Number(targetSec.toFixed(2))));
    setVideoTime(next);
    if (videoElRef.current) {
      videoElRef.current.currentTime = next;
    }
  };

  const togglePlayPause = () => {
    const el = videoElRef.current;
    if (!el) return;
    if (el.paused) {
      void el.play().catch(() => {
        setVideoPlayerError('Browser could not play this video codec inline. Use the Draw Kill-Feed ROI frame scrubber.');
      });
    } else {
      el.pause();
    }
  };

  const capturePlayerFrameToRoi = () => {
    if (!videoInfo) return;
    const el = videoElRef.current;
    const current = el ? el.currentTime : videoTime;
    if (el && !el.paused) {
      el.pause();
    }
    setStageMode('roi');
    void loadPreviewAt(videoInfo, current);
  };

  const handleCancelUpload = async () => {
    uploadAbortRef.current?.abort();
    const pendingId = activeUploadIdRef.current;
    activeUploadIdRef.current = null;
    if (pendingId) {
      try {
        await fetch('/api/upload/cancel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ videoId: pendingId }),
        });
      } catch {
        // Ignore cleanup errors on cancel
      }
    }
  };

  const uploadFullVideoParallel = async (file: File, existingLocalUrl?: string | null): Promise<UploadedVideoInfo> => {
    const controller = new AbortController();
    uploadAbortRef.current = controller;
    activeUploadIdRef.current = null;

    const totalMB = Number((file.size / (1024 * 1024)).toFixed(1));
    const startedAt = performance.now();
    setIsUploading(true);
    setUploadProgress({
      phase: 'uploading',
      filename: file.name,
      percent: 0,
      loadedMB: 0,
      totalMB,
      detail: '0.0 MB/s · 6× parallel',
    });

    try {
      const initResp = await fetch('/api/upload/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, totalSize: file.size }),
        signal: controller.signal,
      });
      const { videoId } = await parseJsonResponse<{ videoId: string }>(initResp);
      activeUploadIdRef.current = videoId;

      const chunks: Array<{ offset: number; end: number }> = [];
      for (let off = 0; off < file.size; off += CHUNK_SIZE) {
        chunks.push({ offset: off, end: Math.min(file.size, off + CHUNK_SIZE) });
      }

      let cursor = 0;
      let completedBytes = 0;

      const worker = async () => {
        while (true) {
          if (controller.signal.aborted) {
            throw new DOMException('Aborted', 'AbortError');
          }
          const idx = cursor++;
          if (idx >= chunks.length) break;
          const { offset, end } = chunks[idx];
          const slice = file.slice(offset, end);
          const resp = await fetch(
            `/api/upload/chunk?videoId=${encodeURIComponent(videoId)}&offset=${offset}`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/octet-stream' },
              body: slice,
              signal: controller.signal,
            }
          );
          await parseJsonResponse(resp);
          completedBytes += end - offset;
          const elapsedSec = Math.max(0.1, (performance.now() - startedAt) / 1000);
          const mbPerSec = (completedBytes / (1024 * 1024)) / elapsedSec;
          const percent = file.size > 0 ? Math.min(100, Math.round((completedBytes / file.size) * 100)) : 100;
          setUploadProgress({
            phase: 'uploading',
            filename: file.name,
            percent,
            loadedMB: Number((completedBytes / (1024 * 1024)).toFixed(1)),
            totalMB,
            detail: `${mbPerSec.toFixed(1)} MB/s · 6× parallel`,
          });
        }
      };

      await Promise.all(
        Array.from({ length: Math.min(UPLOAD_CONCURRENCY, Math.max(1, chunks.length)) }, () => worker())
      );

      setUploadProgress({
        phase: 'inspecting',
        filename: file.name,
        percent: 100,
        loadedMB: totalMB,
        totalMB,
        detail: 'Running ffprobe metadata check...',
      });

      const completeResp = await fetch('/api/upload/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoId }),
        signal: controller.signal,
      });
      const data = await parseJsonResponse<UploadedVideoInfo>(completeResp);
      activeUploadIdRef.current = null;
      return {
        ...data,
        serverReady: true,
        localObjectUrl: existingLocalUrl ?? undefined,
      };
    } finally {
      uploadAbortRef.current = null;
      activeUploadIdRef.current = null;
      setIsUploading(false);
      setUploadProgress(null);
    }
  };

  const handleVideoFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;

    setUploadError('');
    setProcessError('');
    setJobStatus(null);
    setJobId(null);

    try {
      // Try instant 0-upload local probe first!
      const probed = await probeLocalVideoFile(file);
      replaceLocalMedia(probed.objectUrl, file);

      const localInfo: UploadedVideoInfo = {
        videoId: '',
        originalName: file.name,
        isSyntheticDemo: false,
        serverReady: false,
        localObjectUrl: probed.objectUrl,
        metadata: {
          path: file.name,
          width: probed.width,
          height: probed.height,
          fps: 30,
          frame_count: Math.round(probed.duration * 30),
          duration_sec: probed.duration,
          has_audio: true,
          is_vfr: false,
          codec_name: 'local-browser',
        },
      };
      setVideoInfo(localInfo);
      setRoi(null);
      setVideoPlayerError('');
      setIsVideoPlaying(false);
      setVideoTime(probed.initialTs);
      setPreviewTs(probed.initialTs);
      setPreviewUrl(probed.initialPreviewUrl);
      setStageMode('roi');
      setBrowserScanMode('roi_only');
    } catch {
      // Browser cannot natively decode this codec (e.g. MKV/HEVC): fall back cleanly to parallel server upload
      try {
        const uploaded = await uploadFullVideoParallel(file, null);
        replaceLocalMedia(null, file);
        setVideoInfo(uploaded);
        setRoi(null);
        setVideoPlayerError('');
        setIsVideoPlaying(false);
        const initialTs =
          uploaded.metadata.duration_sec > 4
            ? Math.min(10, Number((uploaded.metadata.duration_sec * 0.15).toFixed(1)))
            : 0;
        setVideoTime(initialTs);
        setPreviewTs(initialTs);
        setStageMode('roi');
        void loadPreviewAt(uploaded, initialTs);
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          setUploadError('Upload cancelled.');
        } else {
          setUploadError(err instanceof Error ? err.message : 'Failed to load video');
        }
      }
    }
  };

  const handleOpenLocalPath = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!localPathInput.trim()) {
      setUploadError('Enter a full local file path (for example C:\\Videos\\bgmi_match.mp4).');
      return;
    }
    setIsUploading(true);
    setUploadError('');
    setProcessError('');
    setJobStatus(null);
    setJobId(null);

    try {
      const resp = await fetch('/api/open-local-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filePath: localPathInput.trim() }),
      });
      const data = await parseJsonResponse<UploadedVideoInfo>(resp);
      replaceLocalMedia(null, null);
      const readyInfo: UploadedVideoInfo = { ...data, serverReady: true, isLocalFile: true };
      setVideoInfo(readyInfo);
      setRoi(null);
      setVideoPlayerError('');
      setIsVideoPlaying(false);
      const initialTs =
        readyInfo.metadata.duration_sec > 4
          ? Math.min(10, Number((readyInfo.metadata.duration_sec * 0.15).toFixed(1)))
          : 0;
      setVideoTime(initialTs);
      setPreviewTs(initialTs);
      setStageMode('roi');
      void loadPreviewAt(readyInfo, initialTs);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Failed to open local video file');
    } finally {
      setIsUploading(false);
    }
  };

  const handleGenerateDemoVideo = async () => {
    setIsUploading(true);
    setUploadProgress(null);
    setUploadError('');
    setProcessError('');
    setJobStatus(null);
    setJobId(null);

    try {
      const resp = await fetch('/api/generate-demo-video', { method: 'POST' });
      const data = await parseJsonResponse<UploadedVideoInfo>(resp);
      replaceLocalMedia(null, null);
      const readyInfo: UploadedVideoInfo = { ...data, serverReady: true };

      setVideoInfo(readyInfo);
      setRoi(readyInfo.suggestedRoi ?? null);
      setVideoPlayerError('');
      setIsVideoPlaying(false);
      setVideoTime(2.0);
      setPreviewTs(2.0);
      setStageMode('roi');
      void loadPreviewAt(readyInfo, 2.0);
      setConfig((prev) => ({
        ...prev,
        pre_event_buffer_sec: 3.0,
        post_event_buffer_sec: 3.0,
        merge_gap_sec: 1.5,
      }));
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Failed to generate synthetic demo');
    } finally {
      setIsUploading(false);
    }
  };

  const handleCaptureCalibrationSample = async () => {
    if (!videoInfo || !roi) return;
    setCalibrationNotice('');
    try {
      const bodyPayload =
        videoInfo.serverReady && videoInfo.videoId
          ? {
              videoId: videoInfo.videoId,
              timestampSec: previewTs,
              roi,
              label: calibrationLabel,
            }
          : {
              imageBase64: previewUrl,
              roi,
              label: calibrationLabel,
            };

      const resp = await fetch('/api/calibration/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyPayload),
      });
      const data = await parseJsonResponse<{ filename: string }>(resp);
      setCalibrationNotice(`Saved sample: ${data.filename}`);
      await fetchSystemStatus();
    } catch (err) {
      setCalibrationNotice(err instanceof Error ? err.message : 'Failed to capture calibration sample');
    }
  };

  const handleDeleteCalibrationSample = async (filename: string) => {
    try {
      await fetch(`/api/calibration/samples/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      await fetchSystemStatus();
    } catch {
      // Ignore transient delete errors
    }
  };

  const captureLocalEventScreenshots = async (objectUrl: string, events: KillEventItem[]) => {
    if (!events.length) return;
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    video.src = objectUrl;
    try {
      await new Promise<void>((resolve, reject) => {
        video.addEventListener('loadeddata', () => resolve(), { once: true });
        video.addEventListener('error', () => reject(new Error('Video decode error')), { once: true });
      });
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const captured: Record<number, LocalScreenshotItem> = {};
      for (const ev of events) {
        await waitVideoSeek(video, ev.timestamp_sec);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const filename = `kill_${String(ev.event_id).padStart(3, '0')}_${formatHmsTag(ev.timestamp_sec)}.png`;
        captured[ev.event_id] = {
          dataUrl: canvas.toDataURL('image/png'),
          filename,
        };
      }
      setLocalScreenshots(captured);
    } catch {
      // Ignore transient local screenshot failure
    } finally {
      video.removeAttribute('src');
      video.load();
    }
  };

  const runFastRoiScan = async (info: UploadedVideoInfo, activeRoi: ROIRegion) => {
    if (!info.localObjectUrl) return;
    const controller = new AbortController();
    uploadAbortRef.current = controller;

    const totalDur = Math.max(0.5, info.metadata.duration_sec);
    const stepSec = 1.0 / Math.max(0.5, config.sample_fps);
    const totalExpected = Math.max(1, Math.floor(totalDur / stepSec) + 1);
    let sentMB = 0;

    setIsUploading(true);
    setProcessError('');
    setUploadProgress({
      phase: 'sampling_roi',
      filename: info.originalName,
      percent: 0,
      loadedMB: 0,
      totalMB: 0,
      detail: `Extracting 0 / ${totalExpected} tiny ROI crops (0-video upload)...`,
    });

    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    video.src = info.localObjectUrl;

    try {
      await new Promise<void>((resolve, reject) => {
        video.addEventListener('loadeddata', () => resolve(), { once: true });
        video.addEventListener('error', () => reject(new Error('Unable to read local video frames.')), { once: true });
      });

      const initResp = await fetch('/api/roi-scan/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videoName: info.originalName,
          width: info.metadata.width,
          height: info.metadata.height,
          duration: info.metadata.duration_sec,
        }),
        signal: controller.signal,
      });
      const { scanId } = await parseJsonResponse<{ scanId: string }>(initResp);

      const canvas = document.createElement('canvas');
      canvas.width = activeRoi.width;
      canvas.height = activeRoi.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D unavailable.');

      let batch: Array<{ ts: number; dataUrl: string }> = [];
      let pendingBatchSend: Promise<void> | null = null;
      let count = 0;

      const sendBatch = async (framesToSend: Array<{ ts: number; dataUrl: string }>) => {
        const payload = JSON.stringify({ scanId, frames: framesToSend });
        sentMB += payload.length / (1024 * 1024);
        const resp = await fetch('/api/roi-scan/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: payload,
          signal: controller.signal,
        });
        await parseJsonResponse(resp);
      };

      for (let t = 0; t <= totalDur + 1e-4; t += stepSec) {
        if (controller.signal.aborted) {
          throw new DOMException('Aborted', 'AbortError');
        }
        const clampedT = Math.min(totalDur, Number(t.toFixed(3)));
        await waitVideoSeek(video, clampedT);
        ctx.drawImage(
          video,
          activeRoi.x,
          activeRoi.y,
          activeRoi.width,
          activeRoi.height,
          0,
          0,
          activeRoi.width,
          activeRoi.height
        );
        const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
        batch.push({ ts: clampedT, dataUrl });
        count++;

        if (batch.length >= ROI_BATCH_SIZE) {
          if (pendingBatchSend) await pendingBatchSend;
          const flush = batch;
          batch = [];
          pendingBatchSend = sendBatch(flush);
        }

        if (count % 4 === 0 || count === totalExpected) {
          const pct = Math.min(99, Math.round((count / totalExpected) * 100));
          setUploadProgress({
            phase: 'sampling_roi',
            filename: info.originalName,
            percent: pct,
            loadedMB: Number(sentMB.toFixed(2)),
            totalMB: 0,
            detail: `ROI crop ${count}/${totalExpected} · Only ${sentMB.toFixed(1)} MB sent`,
          });
        }
      }

      if (pendingBatchSend) await pendingBatchSend;
      if (batch.length > 0) {
        await sendBatch(batch);
      }

      setUploadProgress({
        phase: 'inspecting',
        filename: info.originalName,
        percent: 100,
        loadedMB: Number(sentMB.toFixed(2)),
        totalMB: 0,
        detail: `Starting Python BGMI OCR scan (${sentMB.toFixed(1)} MB total ROI data)...`,
      });

      const startResp = await fetch('/api/roi-scan/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scanId, roi: activeRoi, config }),
        signal: controller.signal,
      });
      const data = await parseJsonResponse<{ jobId: string }>(startResp);
      setJobStatus({ state: 'running', progress: 0, message: 'Analyzing ROI kill-feed frames...', confirmed_count: 0 });
      setJobId(data.jobId);
    } finally {
      uploadAbortRef.current = null;
      video.removeAttribute('src');
      video.load();
      setIsUploading(false);
      setUploadProgress(null);
    }
  };

  const handleRunDetection = async () => {
    if (!videoInfo) {
      setProcessError('Select or load a BGMI video first.');
      return;
    }
    if (!roi || roi.width < 40 || roi.height < 16) {
      setProcessError('Draw a valid kill-feed region (at least 40×16px) on the preview frame before processing.');
      return;
    }
    setProcessError('');
    setLocalScreenshots({});
    setLocalClips((prev) => {
      prev.forEach((c) => URL.revokeObjectURL(c.blobUrl));
      return [];
    });

    try {
      // Case 1: Video is already on server (PC local path, synthetic demo, or already uploaded)
      if (videoInfo.serverReady && videoInfo.videoId) {
        setJobStatus({ state: 'running', progress: 0, message: 'Starting scan...', confirmed_count: 0 });
        const resp = await fetch('/api/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            videoId: videoInfo.videoId,
            roi,
            config,
          }),
        });
        const data = await parseJsonResponse<{ jobId: string }>(resp);
        setJobId(data.jobId);
        return;
      }

      // Case 2: Browser-selected file (0-upload previewed): user chose Fast ROI-Only Scan
      if (browserScanMode === 'roi_only' && videoInfo.localObjectUrl) {
        await runFastRoiScan(videoInfo, roi);
        return;
      }

      // Case 3: Browser-selected file + Full Server Upload mode -> 6x parallel upload then process
      const file = selectedLocalFileRef.current;
      if (!file) {
        throw new Error('Local file handle unavailable. Re-select your video.');
      }
      const uploaded = await uploadFullVideoParallel(file, videoInfo.localObjectUrl);
      setVideoInfo(uploaded);
      setJobStatus({ state: 'running', progress: 0, message: 'Starting full FFmpeg + OCR scan...', confirmed_count: 0 });
      const resp = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videoId: uploaded.videoId,
          roi,
          config,
        }),
      });
      const data = await parseJsonResponse<{ jobId: string }>(resp);
      setJobId(data.jobId);
    } catch (err) {
      setJobStatus(null);
      if (err instanceof DOMException && err.name === 'AbortError') {
        setProcessError('Scan cancelled.');
      } else {
        setProcessError(err instanceof Error ? err.message : 'Failed to start detection');
      }
    }
  };

  const handleUpgradeToServerMp4Clips = async () => {
    const file = selectedLocalFileRef.current;
    if (!file || !roi) {
      setProcessError('Local video file or ROI missing.');
      return;
    }
    setProcessError('');
    setActiveTab('workspace');
    try {
      const uploaded = await uploadFullVideoParallel(file, videoInfo?.localObjectUrl);
      setVideoInfo(uploaded);
      setJobStatus({ state: 'running', progress: 0, message: 'Rendering FFmpeg MP4 clips...', confirmed_count: 0 });
      const resp = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videoId: uploaded.videoId,
          roi,
          config,
        }),
      });
      const data = await parseJsonResponse<{ jobId: string }>(resp);
      setJobId(data.jobId);
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        setProcessError(err instanceof Error ? err.message : 'Full video upload failed');
      }
    }
  };

  const handleRecordLocalBrowserClips = async (segments: ClipSegmentItem[]) => {
    const objectUrl = videoInfo?.localObjectUrl;
    if (!objectUrl || segments.length === 0) return;

    if (typeof MediaRecorder === 'undefined') {
      setClipRecordingMsg('MediaRecorder is not supported in this browser. Use "Upload Full Video for FFmpeg MP4 Clips".');
      return;
    }

    setIsRecordingClips(true);
    setClipRecordingMsg('Preparing local browser clip recorder...');
    setLocalClips((prev) => {
      prev.forEach((c) => URL.revokeObjectURL(c.blobUrl));
      return [];
    });

    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.src = objectUrl;

    try {
      await new Promise<void>((resolve, reject) => {
        video.addEventListener('loadeddata', () => resolve(), { once: true });
        video.addEventListener('error', () => reject(new Error('Failed to load video for recording.')), { once: true });
      });

      const w = video.videoWidth;
      const h = video.videoHeight;
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D unavailable.');

      const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) =>
        MediaRecorder.isTypeSupported(m)
      ) || 'video/webm';

      const recorded: LocalRecordedClip[] = [];

      for (const seg of segments) {
        setClipRecordingMsg(`Recording highlight_${String(seg.clip_index).padStart(3, '0')}.webm (${seg.start_sec.toFixed(1)}s → ${seg.end_sec.toFixed(1)}s)...`);
        await waitVideoSeek(video, seg.start_sec);

        const stream = canvas.captureStream(30);
        const chunks: Blob[] = [];
        const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 4_000_000 });
        recorder.ondataavailable = (ev) => {
          if (ev.data && ev.data.size > 0) chunks.push(ev.data);
        };

        const stopped = new Promise<void>((res) => {
          recorder.onstop = () => res();
        });

        let rafId = 0;
        const drawLoop = () => {
          ctx.drawImage(video, 0, 0, w, h);
          if (!video.paused && !video.ended && video.currentTime < seg.end_sec) {
            rafId = requestAnimationFrame(drawLoop);
          }
        };

        recorder.start(250);
        await video.play();
        rafId = requestAnimationFrame(drawLoop);

        await new Promise<void>((res) => {
          const checkTimer = window.setInterval(() => {
            if (video.currentTime >= seg.end_sec || video.ended) {
              window.clearInterval(checkTimer);
              res();
            }
          }, 80);
        });

        cancelAnimationFrame(rafId);
        video.pause();
        recorder.stop();
        await stopped;
        stream.getTracks().forEach((t) => t.stop());

        const blob = new Blob(chunks, { type: 'video/webm' });
        const blobUrl = URL.createObjectURL(blob);
        recorded.push({
          clip_index: seg.clip_index,
          blobUrl,
          filename: `highlight_${String(seg.clip_index).padStart(3, '0')}.webm`,
          start_sec: seg.start_sec,
          end_sec: seg.end_sec,
          duration_sec: seg.duration_sec,
        });
      }

      setLocalClips(recorded);
      setClipRecordingMsg(`Recorded ${recorded.length} highlight WebM clips locally!`);
    } catch (err) {
      setClipRecordingMsg(err instanceof Error ? err.message : 'Local clip recording failed.');
    } finally {
      video.removeAttribute('src');
      video.load();
      setIsRecordingClips(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    const poll = async () => {
      try {
        const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}?_t=${Date.now()}`, {
          cache: 'no-store',
          headers: { 'Cache-Control': 'no-cache' },
        });
        const data = await parseJsonResponse<JobStatus>(resp);
        if (!active) return;
        setJobStatus(data);
        if (data.state === 'completed') {
          if (data.scan_mode === 'roi_only' && data.events && data.events.length > 0 && localObjectUrlRef.current) {
            void captureLocalEventScreenshots(localObjectUrlRef.current, data.events);
          }
          setActiveTab('results');
          return;
        }
        if (data.state === 'failed') {
          setProcessError(data.error || data.message || 'Detection job failed');
          return;
        }
        setTimeout(poll, 600);
      } catch (err) {
        if (!active) return;
        const msg = err instanceof Error ? err.message : 'Lost connection while polling job';
        if (msg.startsWith('HTTP 404')) {
          setJobStatus(null);
          setProcessError('Processing job expired or server restarted. Please run detection again.');
          return;
        }
        setTimeout(poll, 1000);
      }
    };
    poll();
    return () => {
      active = false;
    };
  }, [jobId]);

  const canRun = Boolean(
    videoInfo &&
      roi &&
      roi.width >= 40 &&
      roi.height >= 16 &&
      !isUploading &&
      jobStatus?.state !== 'running'
  );

  const filteredEvents = (jobStatus?.events ?? []).filter((ev) => {
    if (!eventSearch.trim()) return true;
    const q = eventSearch.trim().toLowerCase();
    return (
      ev.killer.toLowerCase().includes(q) ||
      ev.victim.toLowerCase().includes(q) ||
      ev.weapon_or_action.toLowerCase().includes(q) ||
      ev.ocr_text.toLowerCase().includes(q)
    );
  });

  const videoPlayerSrc = videoInfo
    ? videoInfo.localObjectUrl ||
      (videoInfo.serverReady && videoInfo.videoId ? `/api/videos/${encodeURIComponent(videoInfo.videoId)}/stream` : '')
    : '';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Bar Contract: 3 zones separated by gap-8 */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-30">
        <div className="max-w-[1440px] mx-auto px-6 h-16 flex items-center justify-between gap-8">
          <button
            type="button"
            onClick={() => setActiveTab('workspace')}
            className="text-lg font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0 cursor-pointer"
          >
            PEACE
          </button>

          <nav className="flex items-center gap-6 overflow-x-auto">
            {(
              [
                ['workspace', 'Workspace'],
                ['results', 'Session Results'],
                ['calibration', 'Calibration'],
                ['setup', 'Windows Setup'],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setActiveTab(key)}
                className={`py-5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  activeTab === key
                    ? 'border-emerald-500 text-slate-100'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>

          <div className="shrink-0">
            <button
              type="button"
              disabled={!canRun}
              onClick={handleRunDetection}
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-500 text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed"
            >
              <Play className="w-4 h-4" />
              {isUploading
                ? 'Preparing...'
                : jobStatus?.state === 'running'
                ? 'Scanning...'
                : 'Run Detection'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto">
        {systemError && (
          <div className="mx-6 mt-4 p-3 rounded-lg border border-red-800 bg-red-950/50 text-sm text-red-200 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
            <span>{systemError}</span>
          </div>
        )}

        {activeTab === 'workspace' && (
          <div className="flex flex-col lg:flex-row min-h-[calc(100vh-4rem)]">
            {/* Left Sidebar: Video Source & Detection Config */}
            <aside className="w-full lg:w-[350px] shrink-0 border-b lg:border-b-0 lg:border-r border-slate-800 bg-slate-900/40 p-6 space-y-6">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 mb-3">1. Gameplay Video</h2>
                <div className="space-y-2.5">
                  <label className={`flex items-center justify-center gap-2 w-full px-4 py-2.5 text-sm font-medium bg-slate-800 border border-slate-700 rounded-lg transition-colors ${isUploading ? 'opacity-60 cursor-not-allowed' : 'hover:bg-slate-700 cursor-pointer'}`}>
                    <Upload className="w-4 h-4 text-emerald-400" />
                    <span className="whitespace-nowrap">
                      {uploadProgress
                        ? uploadProgress.phase === 'sampling_roi'
                          ? `ROI Scan ${uploadProgress.percent}%`
                          : uploadProgress.phase === 'inspecting'
                          ? 'Inspecting...'
                          : `Uploading ${uploadProgress.percent}%`
                        : isUploading
                        ? 'Preparing Video...'
                        : 'Select BGMI Recording'}
                    </span>
                    <input
                      type="file"
                      accept="video/*,.mp4,.mkv,.mov,.avi,.webm"
                      onChange={handleVideoFileUpload}
                      disabled={isUploading}
                      className="hidden"
                    />
                  </label>

                  {uploadProgress && (
                    <div className="p-3 rounded-lg border border-slate-800 bg-slate-950/70 space-y-2">
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <span className="text-slate-300 truncate">{uploadProgress.filename}</span>
                        <span className="font-mono tabular-nums text-emerald-400 shrink-0">
                          {uploadProgress.percent}%
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all duration-150"
                          style={{ width: `${uploadProgress.percent}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between gap-2 text-[11px] font-mono tabular-nums text-slate-400">
                        <span className="truncate">
                          {uploadProgress.detail ||
                            `${uploadProgress.loadedMB.toFixed(1)} / ${uploadProgress.totalMB.toFixed(1)} MB`}
                        </span>
                        <button
                          type="button"
                          onClick={handleCancelUpload}
                          className="text-red-400 hover:text-red-300 font-sans font-medium shrink-0 cursor-pointer"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Zero-Upload Local PC Path */}
                  <form onSubmit={handleOpenLocalPath} className="pt-1 space-y-1.5">
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                      <HardDrive className="w-3 h-3 text-slate-400" />
                      <span>On Windows/Linux PC (0-Upload Instant Open):</span>
                    </div>
                    <div className="flex gap-1.5">
                      <input
                        type="text"
                        value={localPathInput}
                        onChange={(e) => setLocalPathInput(e.target.value)}
                        placeholder="C:\Videos\bgmi_30min.mp4"
                        disabled={isUploading}
                        className="flex-1 min-w-0 px-2.5 py-1.5 text-xs font-mono bg-slate-950 border border-slate-700 rounded-md text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-emerald-500"
                      />
                      <button
                        type="submit"
                        disabled={isUploading || !localPathInput.trim()}
                        className="px-2.5 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-slate-700 rounded-md text-slate-200 whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        Open
                      </button>
                    </div>
                  </form>

                  <button
                    type="button"
                    onClick={handleGenerateDemoVideo}
                    disabled={isUploading}
                    className="flex items-center justify-center gap-2 w-full px-4 py-2 text-xs font-medium text-slate-300 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 border border-slate-800 rounded-lg transition-colors cursor-pointer disabled:cursor-not-allowed"
                  >
                    <Film className="w-3.5 h-3.5 text-slate-400" />
                    Generate Synthetic Test Video
                  </button>
                </div>

                {uploadError && (
                  <p className="mt-2 text-xs text-red-400">{uploadError}</p>
                )}

                {systemStatus && (
                  <div className="mt-3 text-xs text-slate-400 space-y-1 font-mono tabular-nums">
                    <div>
                      FFmpeg {systemStatus.ffmpeg_available ? 'Ready' : 'Missing'} · OCR{' '}
                      {systemStatus.ocr_available ? systemStatus.ocr_backend : 'Disabled'}
                    </div>
                    <div>
                      Calibration Samples: {systemStatus.calibration_samples.length}
                    </div>
                  </div>
                )}
              </div>

              {/* Scan Workflow Selector when file was opened locally in browser */}
              {videoInfo && videoInfo.localObjectUrl && !videoInfo.serverReady && (
                <div className="border-t border-slate-800 pt-5 space-y-2.5">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
                    <Zap className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Upload & Scan Speed Mode</span>
                  </div>
                  <div className="grid grid-cols-1 gap-2">
                    <button
                      type="button"
                      onClick={() => setBrowserScanMode('roi_only')}
                      className={`p-2.5 text-left rounded-lg border transition-colors cursor-pointer ${
                        browserScanMode === 'roi_only'
                          ? 'bg-emerald-600/15 border-emerald-500 text-slate-100'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="text-xs font-semibold text-emerald-400">
                        Fast ROI-Only Scan (Recommended for 10–60 min HD)
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5 leading-snug">
                        Skips uploading the multi-GB video — sends only tiny kill-feed crops (~100× smaller). Captures full-HD PNGs locally.
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setBrowserScanMode('full_upload')}
                      className={`p-2.5 text-left rounded-lg border transition-colors cursor-pointer ${
                        browserScanMode === 'full_upload'
                          ? 'bg-emerald-600/15 border-emerald-500 text-slate-100'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="text-xs font-semibold text-slate-200">
                        Full Server Upload (6× Parallel) + MP4 Clips
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5 leading-snug">
                        Uploads the entire video over 6 parallel streams so the server cuts x264 MP4 highlight clips with audio.
                      </div>
                    </button>
                  </div>
                </div>
              )}

              <div className="border-t border-slate-800 pt-5">
                <h2 className="text-sm font-semibold text-slate-100 mb-3">2. Preview Timestamp</h2>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    max={videoInfo ? Math.max(0, Math.floor(videoInfo.metadata.duration_sec)) : 86400}
                    step={0.5}
                    value={previewTs}
                    onChange={(e) => setPreviewTs(Math.max(0, Number(e.target.value) || 0))}
                    disabled={!videoInfo}
                    className="w-24 px-3 py-1.5 text-sm font-mono tabular-nums bg-slate-950 border border-slate-700 rounded-md text-slate-100 focus:outline-none focus:border-emerald-500"
                  />
                  <span className="text-xs text-slate-400">sec</span>
                  <button
                    type="button"
                    disabled={!videoInfo || isLoadingPreview}
                    onClick={() => videoInfo && void loadPreviewAt(videoInfo, previewTs)}
                    className="ml-auto px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-slate-700 rounded-md transition-colors cursor-pointer"
                  >
                    Load Frame
                  </button>
                </div>
              </div>

              <div className="border-t border-slate-800 pt-5 space-y-4">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-slate-400" />
                  <h2 className="text-sm font-semibold text-slate-100">3. Detection & Clipping</h2>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setConfig((prev) => ({
                        ...prev,
                        sample_fps: 3.0,
                        confidence_threshold: 0.65,
                        min_confirm_frames: 2,
                        pre_event_buffer_sec: 8.0,
                        post_event_buffer_sec: 6.0,
                        merge_gap_sec: 4.0,
                      }))
                    }
                    className={`px-2.5 py-2 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      config.sample_fps === 3 && config.confidence_threshold === 0.65 && config.min_confirm_frames === 2
                        ? 'bg-emerald-600/20 border-emerald-500 text-emerald-300'
                        : 'bg-slate-950 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    Full Match (HD)
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setConfig((prev) => ({
                        ...prev,
                        sample_fps: 4.0,
                        confidence_threshold: 0.62,
                        min_confirm_frames: 2,
                        pre_event_buffer_sec: 4.0,
                        post_event_buffer_sec: 3.0,
                        merge_gap_sec: 3.0,
                      }))
                    }
                    className={`px-2.5 py-2 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      config.sample_fps === 4 && config.confidence_threshold === 0.62 && config.min_confirm_frames === 2
                        ? 'bg-emerald-600/20 border-emerald-500 text-emerald-300'
                        : 'bg-slate-950 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    Short Clip / Reels
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <label className="text-xs text-slate-400 space-y-1">
                    <span>Sampling FPS</span>
                    <input
                      type="number"
                      min={0.5}
                      max={30}
                      step={0.5}
                      value={config.sample_fps}
                      onChange={(e) => setConfig({ ...config, sample_fps: Number(e.target.value) || 2 })}
                      className="w-full px-2.5 py-1.5 text-sm font-mono tabular-nums bg-slate-950 border border-slate-700 rounded-md text-slate-100"
                    />
                  </label>

                  <label className="text-xs text-slate-400 space-y-1">
                    <span>Confidence</span>
                    <input
                      type="number"
                      min={0.4}
                      max={0.95}
                      step={0.02}
                      value={config.confidence_threshold}
                      onChange={(e) => setConfig({ ...config, confidence_threshold: Number(e.target.value) || 0.65 })}
                      className="w-full px-2.5 py-1.5 text-sm font-mono tabular-nums bg-slate-950 border border-slate-700 rounded-md text-slate-100"
                    />
                  </label>

                  <label className="text-xs text-slate-400 space-y-1">
                    <span>Confirm Frames</span>
                    <input
                      type="number"
                      min={1}
                      max={5}
                      step={1}
                      value={config.min_confirm_frames}
                      onChange={(e) => setConfig({ ...config, min_confirm_frames: Math.max(1, Number(e.target.value) || 2) })}
                      className="w-full px-2.5 py-1.5 text-sm font-mono tabular-nums bg-slate-950 border border-slate-700 rounded-md text-slate-100"
                    />
                  </label>

                  <label className="text-xs text-slate-400 space-y-1">
                    <span>Merge Gap (s)</span>
                    <input
                      type="number"
                      min={0}
                      max={30}
                      step={0.5}
                      value={config.merge_gap_sec}
                      onChange={(e) => setConfig({ ...config, merge_gap_sec: Number(e.target.value) || 0 })}
                      className="w-full px-2.5 py-1.5 text-sm font-mono tabular-nums bg-slate-950 border border-slate-700 rounded-md text-slate-100"
                    />
                  </label>

                  <label className="text-xs text-slate-400 space-y-1">
                    <span>Pre-Kill Buffer (s)</span>
                    <input
                      type="number"
                      min={0}
                      max={60}
                      step={1}
                      value={config.pre_event_buffer_sec}
                      onChange={(e) => setConfig({ ...config, pre_event_buffer_sec: Number(e.target.value) || 0 })}
                      className="w-full px-2.5 py-1.5 text-sm font-mono tabular-nums bg-slate-950 border border-slate-700 rounded-md text-slate-100"
                    />
                  </label>

                  <label className="text-xs text-slate-400 space-y-1">
                    <span>Post-Kill Buffer (s)</span>
                    <input
                      type="number"
                      min={0}
                      max={60}
                      step={1}
                      value={config.post_event_buffer_sec}
                      onChange={(e) => setConfig({ ...config, post_event_buffer_sec: Number(e.target.value) || 0 })}
                      className="w-full px-2.5 py-1.5 text-sm font-mono tabular-nums bg-slate-950 border border-slate-700 rounded-md text-slate-100"
                    />
                  </label>
                </div>

                <label className="block text-xs text-slate-400 space-y-1">
                  <span>Optional Player IGN Filter</span>
                  <input
                    type="text"
                    placeholder="Leave blank to capture all kills in ROI"
                    value={config.player_ign}
                    onChange={(e) => setConfig({ ...config, player_ign: e.target.value })}
                    className="w-full px-2.5 py-1.5 text-sm bg-slate-950 border border-slate-700 rounded-md text-slate-100 placeholder:text-slate-600"
                  />
                </label>

                <div className="space-y-2 pt-1">
                  <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.only_player_kills}
                      onChange={(e) => setConfig({ ...config, only_player_kills: e.target.checked })}
                      className="rounded border-slate-700 bg-slate-950 accent-emerald-600"
                    />
                    Only keep kills matching Player IGN
                  </label>
                  <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.preserve_audio}
                      onChange={(e) => setConfig({ ...config, preserve_audio: e.target.checked })}
                      className="rounded border-slate-700 bg-slate-950 accent-emerald-600"
                    />
                    Preserve original audio in clips
                  </label>
                  <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.save_debug_roi}
                      onChange={(e) => setConfig({ ...config, save_debug_roi: e.target.checked })}
                      className="rounded border-slate-700 bg-slate-950 accent-emerald-600"
                    />
                    Save debug ROI detection crops
                  </label>
                </div>
              </div>
            </aside>

            {/* Main Viewport: Interactive ROI Selection Stage */}
            <section className="flex-1 p-6 lg:p-8 space-y-6">
              {!videoInfo ? (
                <div className="border border-slate-800 bg-slate-900/50 rounded-xl p-10 text-center max-w-2xl mx-auto my-10 space-y-6">
                  <Scissors className="w-10 h-10 text-emerald-500 mx-auto" />
                  <div className="space-y-2">
                    <h1 className="text-2xl font-bold text-slate-100 text-balance">
                      BGMI Kill Feed Detector & Highlight Clipper
                    </h1>
                    <p className="text-sm text-slate-400 max-w-lg mx-auto">
                      Select any 30-min HD BGMI recording to preview instantly with zero upload wait, draw a rectangle
                      over the kill-feed, and extract confirmed kill screenshots & highlight clips.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center justify-center gap-3 pt-1">
                    <label className={`inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold rounded-lg bg-emerald-600 text-white transition-colors ${isUploading ? 'opacity-60 cursor-not-allowed' : 'hover:bg-emerald-500 cursor-pointer'}`}>
                      <Upload className="w-4 h-4" />
                      {uploadProgress
                        ? uploadProgress.phase === 'inspecting'
                          ? 'Inspecting Video...'
                          : `Uploading ${uploadProgress.percent}%`
                        : 'Select BGMI Recording'}
                      <input
                        type="file"
                        accept="video/*,.mp4,.mkv,.mov,.avi,.webm"
                        onChange={handleVideoFileUpload}
                        disabled={isUploading}
                        className="hidden"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={handleGenerateDemoVideo}
                      disabled={isUploading}
                      className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-slate-700 text-slate-200 cursor-pointer disabled:cursor-not-allowed transition-colors"
                    >
                      <Film className="w-4 h-4" />
                      Load Synthetic Test Video
                    </button>
                  </div>

                  <form
                    onSubmit={handleOpenLocalPath}
                    className="max-w-lg mx-auto pt-4 border-t border-slate-800 space-y-2 text-left"
                  >
                    <label className="block text-xs font-medium text-slate-300">
                      Running on Windows / Local PC? Open directly by file path (0-byte upload, instant FFmpeg):
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={localPathInput}
                        onChange={(e) => setLocalPathInput(e.target.value)}
                        placeholder="e.g. C:\Users\You\Videos\bgmi_erangel_30min.mp4"
                        disabled={isUploading}
                        className="flex-1 min-w-0 px-3 py-2 text-xs font-mono bg-slate-950 border border-slate-700 rounded-lg text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-emerald-500"
                      />
                      <button
                        type="submit"
                        disabled={isUploading || !localPathInput.trim()}
                        className="px-4 py-2 text-xs font-semibold bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-slate-700 rounded-lg text-slate-100 whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        Open Local File
                      </button>
                    </div>
                  </form>

                  {uploadProgress && (
                    <div className="max-w-md mx-auto pt-2 space-y-2 text-left">
                      <div className="flex items-center justify-between text-xs font-mono tabular-nums text-slate-300">
                        <span>{uploadProgress.detail || `${uploadProgress.loadedMB.toFixed(1)} / ${uploadProgress.totalMB.toFixed(1)} MB`}</span>
                        <span className="text-emerald-400">{uploadProgress.percent}%</span>
                      </div>
                      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all duration-150"
                          style={{ width: `${uploadProgress.percent}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {/* Clean unboxed metadata bar */}
                  <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
                    <div>
                      <h1 className="text-lg font-semibold text-slate-100">
                        {videoInfo.originalName}
                      </h1>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-400 font-mono tabular-nums">
                        <span>{videoInfo.metadata.width}×{videoInfo.metadata.height}</span>
                        <span aria-hidden="true">·</span>
                        <span>{videoInfo.metadata.fps.toFixed(1)} fps</span>
                        <span aria-hidden="true">·</span>
                        <span>{formatDuration(videoInfo.metadata.duration_sec)}</span>
                        <span aria-hidden="true">·</span>
                        <span>{videoInfo.metadata.has_audio ? 'Audio Present' : 'No Audio'}</span>
                        <span aria-hidden="true">·</span>
                        <span>
                          {videoInfo.isLocalFile
                            ? 'PC Direct Local File (0 Upload)'
                            : videoInfo.localObjectUrl && !videoInfo.serverReady
                            ? 'Instant Browser Preview (0 Upload)'
                            : videoInfo.metadata.codec_name.toUpperCase()}
                        </span>
                        {videoInfo.isSyntheticDemo && (
                          <>
                            <span aria-hidden="true">·</span>
                            <span className="text-amber-400">Synthetic Demo Video</span>
                          </>
                        )}
                      </div>
                    </div>

                    {roi && (
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={calibrationLabel}
                          onChange={(e) => setCalibrationLabel(e.target.value)}
                          placeholder="sample_label"
                          className="w-36 px-2.5 py-1.5 text-xs bg-slate-900 border border-slate-700 rounded-md text-slate-200"
                        />
                        <button
                          type="button"
                          onClick={handleCaptureCalibrationSample}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md text-slate-200 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          <Camera className="w-3.5 h-3.5" />
                          Save Selection as Calibration Sample
                        </button>
                      </div>
                    )}
                  </div>

                  {calibrationNotice && (
                    <div className="text-xs text-emerald-400">{calibrationNotice}</div>
                  )}

                  {/* Stage Mode Switch: Interactive Video Player vs Frame ROI Selector */}
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="inline-flex rounded-lg border border-slate-800 bg-slate-900 p-1">
                      <button
                        type="button"
                        onClick={() => setStageMode('player')}
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                          stageMode === 'player'
                            ? 'bg-emerald-600 text-white'
                            : 'text-slate-300 hover:text-white'
                        }`}
                      >
                        <Play className="w-3.5 h-3.5" />
                        Play & Seek Video
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          if (videoElRef.current && !videoElRef.current.paused) {
                            videoElRef.current.pause();
                          }
                          const ts = videoElRef.current ? videoElRef.current.currentTime : previewTs;
                          setStageMode('roi');
                          if (Math.abs(ts - previewTs) > 0.25) {
                            void loadPreviewAt(videoInfo, ts);
                          }
                        }}
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                          stageMode === 'roi'
                            ? 'bg-emerald-600 text-white'
                            : 'text-slate-300 hover:text-white'
                        }`}
                      >
                        <Crosshair className="w-3.5 h-3.5" />
                        Draw Kill-Feed ROI
                      </button>
                    </div>

                    <div className="text-xs text-slate-400">
                      Tip: Seek to a moment showing the BGMI kill-feed banner, then drag a box around 2–4 rows.
                    </div>
                  </div>

                  {stageMode === 'player' ? (
                    <div className="space-y-3">
                      <div className="w-full bg-slate-950 border border-slate-800 rounded-lg overflow-hidden flex items-center justify-center min-h-[240px]">
                        {videoPlayerError || !videoPlayerSrc ? (
                          <div className="p-8 text-center space-y-3">
                            <p className="text-sm text-amber-300">
                              {videoPlayerError || 'Preview stream unavailable.'}
                            </p>
                            <button
                              type="button"
                              onClick={() => setStageMode('roi')}
                              className="px-4 py-2 text-xs font-semibold rounded-md bg-emerald-600 hover:bg-emerald-500 text-white cursor-pointer"
                            >
                              Switch to Frame Scrubber & ROI Selector
                            </button>
                          </div>
                        ) : (
                          <video
                            ref={videoElRef}
                            key={videoPlayerSrc}
                            src={videoPlayerSrc}
                            controls
                            playsInline
                            preload="metadata"
                            onLoadedMetadata={() => {
                              if (videoElRef.current && videoTime > 0) {
                                videoElRef.current.currentTime = videoTime;
                              }
                            }}
                            onTimeUpdate={() => {
                              if (videoElRef.current) {
                                setVideoTime(videoElRef.current.currentTime);
                              }
                            }}
                            onPlay={() => setIsVideoPlaying(true)}
                            onPause={() => setIsVideoPlaying(false)}
                            onError={() =>
                              setVideoPlayerError(
                                'This video codec cannot play directly in the browser preview. Use Draw Kill-Feed ROI below to scrub any timestamp.'
                              )
                            }
                            className="w-full max-h-[540px] bg-black"
                          />
                        )}
                      </div>

                      {/* Transport & Scrubbing Controls */}
                      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-lg border border-slate-800 bg-slate-900/60">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => seekPlayerTo(videoTime - 10)}
                            className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-mono bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-200 cursor-pointer"
                          >
                            <SkipBack className="w-3 h-3" />
                            10s
                          </button>
                          <button
                            type="button"
                            onClick={() => seekPlayerTo(videoTime - 2)}
                            className="px-2.5 py-1.5 text-xs font-mono bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-200 cursor-pointer"
                          >
                            -2s
                          </button>
                          <button
                            type="button"
                            onClick={() => seekPlayerTo(videoTime - 0.5)}
                            className="px-2.5 py-1.5 text-xs font-mono bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-200 cursor-pointer"
                          >
                            -0.5s
                          </button>
                          <button
                            type="button"
                            onClick={togglePlayPause}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded text-white cursor-pointer"
                          >
                            {isVideoPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                            {isVideoPlaying ? 'Pause' : 'Play'}
                          </button>
                          <button
                            type="button"
                            onClick={() => seekPlayerTo(videoTime + 0.5)}
                            className="px-2.5 py-1.5 text-xs font-mono bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-200 cursor-pointer"
                          >
                            +0.5s
                          </button>
                          <button
                            type="button"
                            onClick={() => seekPlayerTo(videoTime + 2)}
                            className="px-2.5 py-1.5 text-xs font-mono bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-200 cursor-pointer"
                          >
                            +2s
                          </button>
                          <button
                            type="button"
                            onClick={() => seekPlayerTo(videoTime + 10)}
                            className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-mono bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-200 cursor-pointer"
                          >
                            10s
                            <SkipForward className="w-3 h-3" />
                          </button>
                        </div>

                        <div className="flex items-center gap-3">
                          <span className="text-xs font-mono tabular-nums text-slate-300">
                            {videoTime.toFixed(1)}s / {videoInfo.metadata.duration_sec.toFixed(1)}s
                          </span>
                          <button
                            type="button"
                            onClick={capturePlayerFrameToRoi}
                            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-md bg-emerald-600 hover:bg-emerald-500 text-white whitespace-nowrap shrink-0 cursor-pointer"
                          >
                            <Crosshair className="w-3.5 h-3.5" />
                            Use This Frame to Draw ROI
                          </button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <RoiCanvas
                        previewUrl={previewUrl}
                        videoWidth={videoInfo.metadata.width}
                        videoHeight={videoInfo.metadata.height}
                        roi={roi}
                        onChangeRoi={setRoi}
                        isLoadingPreview={isLoadingPreview}
                        onPreviewReady={() => setIsLoadingPreview(false)}
                      />

                      {/* Timeline Frame Scrubber inside ROI Mode */}
                      <div className="p-3 rounded-lg border border-slate-800 bg-slate-900/60 space-y-2.5">
                        <div className="flex items-center gap-3">
                          <span className="text-xs font-mono tabular-nums text-slate-400 shrink-0">
                            {previewTs.toFixed(1)}s
                          </span>
                          <input
                            type="range"
                            min={0}
                            max={Math.max(0.5, videoInfo.metadata.duration_sec)}
                            step={0.2}
                            value={Math.min(previewTs, Math.max(0.5, videoInfo.metadata.duration_sec))}
                            onChange={(e) => setPreviewTs(Number(e.target.value))}
                            onMouseUp={(e) => void loadPreviewAt(videoInfo, Number((e.target as HTMLInputElement).value))}
                            onTouchEnd={(e) => void loadPreviewAt(videoInfo, Number((e.target as HTMLInputElement).value))}
                            className="flex-1 accent-emerald-500 cursor-pointer"
                          />
                          <span className="text-xs font-mono tabular-nums text-slate-400 shrink-0">
                            {videoInfo.metadata.duration_sec.toFixed(1)}s
                          </span>
                        </div>

                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex flex-wrap items-center gap-1.5">
                            {[-10, -5, -1, 1, 5, 10].map((delta) => (
                              <button
                                key={delta}
                                type="button"
                                disabled={isLoadingPreview}
                                onClick={() => {
                                  const next = Math.max(
                                    0,
                                    Math.min(videoInfo.metadata.duration_sec, previewTs + delta)
                                  );
                                  void loadPreviewAt(videoInfo, next);
                                }}
                                className="px-2.5 py-1 text-xs font-mono bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-slate-700 rounded text-slate-200 cursor-pointer"
                              >
                                {delta > 0 ? `+${delta}s` : `${delta}s`}
                              </button>
                            ))}
                          </div>
                          <button
                            type="button"
                            disabled={isLoadingPreview}
                            onClick={() => void loadPreviewAt(videoInfo, previewTs)}
                            className="px-3 py-1 text-xs font-medium bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-slate-700 rounded text-slate-200 cursor-pointer"
                          >
                            Refresh Frame at {previewTs.toFixed(1)}s
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {processError && (
                    <div className="p-3 rounded-lg border border-red-800 bg-red-950/50 text-sm text-red-200 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
                      <span>{processError}</span>
                    </div>
                  )}

                  {jobStatus && (
                    <div className="border border-slate-800 bg-slate-900/60 rounded-lg p-4 space-y-2.5">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium text-slate-200">{jobStatus.message}</span>
                        <span className="font-mono tabular-nums text-emerald-400">
                          {Math.round((jobStatus.progress || 0) * 100)}% · Kills: {jobStatus.confirmed_count}
                        </span>
                      </div>
                      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all duration-200"
                          style={{ width: `${Math.min(100, Math.max(0, (jobStatus.progress || 0) * 100))}%` }}
                        />
                      </div>
                    </div>
                  )}
                </>
              )}
            </section>
          </div>
        )}

        {activeTab === 'results' && (
          <div className="p-6 lg:p-8 space-y-8">
            {!jobStatus || jobStatus.state !== 'completed' ? (
              <div className="border border-slate-800 bg-slate-900/40 rounded-xl p-10 text-center max-w-xl mx-auto my-12 space-y-4">
                <FolderOpen className="w-9 h-9 text-slate-400 mx-auto" />
                <h2 className="text-lg font-semibold text-slate-100">No Completed Session Yet</h2>
                <p className="text-sm text-slate-400">
                  Load a video, select the kill-feed region, and click Run Detection to inspect confirmed events,
                  screenshots, and highlight clips here.
                </p>
                <button
                  type="button"
                  onClick={() => setActiveTab('workspace')}
                  className="px-4 py-2 text-sm font-medium bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg cursor-pointer"
                >
                  Go to Workspace
                </button>
              </div>
            ) : (
              <>
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
                  <div>
                    <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
                      <CheckCircle2 className="w-4 h-4" />
                      <span>
                        {jobStatus.scan_mode === 'roi_only' ? 'Fast ROI-Only Scan Complete' : 'Session Complete'}
                      </span>
                    </div>
                    <h1 className="text-xl font-bold text-slate-100 mt-1">
                      Confirmed BGMI Kill-Feed Results
                    </h1>
                    <p className="text-xs text-slate-400 mt-1">
                      {jobStatus.scan_mode === 'roi_only'
                        ? 'Full-resolution PNG screenshots are captured locally in your browser. Use the clip options below to save highlight clips.'
                        : 'Use the Download buttons below to save PNG screenshots and MP4 clips directly to your PC.'}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    {jobId && (
                      <>
                        <a
                          href={`/api/jobs/${encodeURIComponent(jobId)}/files/root/events.csv`}
                          download="events.csv"
                          className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md text-slate-200"
                        >
                          <Download className="w-3.5 h-3.5" />
                          events.csv
                        </a>
                        <a
                          href={`/api/jobs/${encodeURIComponent(jobId)}/files/root/summary.txt`}
                          download="summary.txt"
                          className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md text-slate-200"
                        >
                          <Download className="w-3.5 h-3.5" />
                          summary.txt
                        </a>
                      </>
                    )}
                  </div>
                </div>

                {/* Single-level metrics grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="border border-slate-800 bg-slate-900/50 rounded-lg p-4">
                    <div className="text-xs text-slate-400">Confirmed Kills</div>
                    <div className="text-2xl font-bold font-mono tabular-nums text-emerald-400 mt-1">
                      {jobStatus.events?.length ?? 0}
                    </div>
                  </div>
                  <div className="border border-slate-800 bg-slate-900/50 rounded-lg p-4">
                    <div className="text-xs text-slate-400">
                      {jobStatus.scan_mode === 'roi_only' ? 'Clip Windows' : 'Highlight Clips'}
                    </div>
                    <div className="text-2xl font-bold font-mono tabular-nums text-slate-100 mt-1">
                      {jobStatus.scan_mode === 'roi_only'
                        ? jobStatus.clip_segments?.length ?? 0
                        : (jobStatus.clips ?? []).filter((c) => c.success).length}
                    </div>
                  </div>
                  <div className="border border-slate-800 bg-slate-900/50 rounded-lg p-4">
                    <div className="text-xs text-slate-400">Frames Analyzed</div>
                    <div className="text-2xl font-bold font-mono tabular-nums text-slate-100 mt-1">
                      {jobStatus.frames_analyzed ?? 0}
                    </div>
                  </div>
                  <div className="border border-slate-800 bg-slate-900/50 rounded-lg p-4">
                    <div className="text-xs text-slate-400">OCR · Elapsed</div>
                    <div className="text-base font-semibold font-mono tabular-nums text-slate-100 mt-2">
                      {jobStatus.ocr_backend} · {jobStatus.elapsed_sec}s
                    </div>
                  </div>
                </div>

                {/* Events Table */}
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <h2 className="text-base font-semibold text-slate-100">Detected Kill-Feed Events</h2>
                    <input
                      type="text"
                      value={eventSearch}
                      onChange={(e) => setEventSearch(e.target.value)}
                      placeholder="Filter by player or weapon..."
                      className="w-64 px-3 py-1.5 text-xs bg-slate-900 border border-slate-700 rounded-md text-slate-200 placeholder:text-slate-500"
                    />
                  </div>

                  {filteredEvents.length === 0 ? (
                    <div className="border border-slate-800 bg-slate-900/40 rounded-lg p-6 space-y-2 text-sm text-slate-300">
                      <div className="font-semibold text-slate-100">
                        {(jobStatus.events?.length ?? 0) === 0
                          ? 'No confirmed kill-feed events were detected — so 0 screenshots and 0 highlight clips were generated.'
                          : 'No events matched your search filter.'}
                      </div>
                      {(jobStatus.events?.length ?? 0) === 0 && (
                        <ul className="list-disc list-inside text-xs text-slate-400 space-y-1">
                          <li>Seek <strong>Preview Timestamp</strong> to a frame where a real BGMI kill-feed banner is visible and verify your ROI rectangle tightly covers it.</li>
                          <li>Try lowering <strong>Confidence</strong> slightly (e.g. 0.58) or setting <strong>Confirm Frames</strong> to 1 if your kill banners appear briefly.</li>
                          <li>Adding 3–5 real BGMI kill-feed crops in <strong>Calibration</strong> improves detection on stylized fonts.</li>
                        </ul>
                      )}
                    </div>
                  ) : (
                    <div className="border border-slate-800 rounded-lg overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead className="bg-slate-900 text-xs text-slate-400 border-b border-slate-800">
                          <tr>
                            <th className="py-2.5 px-4 font-medium">#</th>
                            <th className="py-2.5 px-4 font-medium">Timestamp</th>
                            <th className="py-2.5 px-4 font-medium">Killer</th>
                            <th className="py-2.5 px-4 font-medium">Weapon / Action</th>
                            <th className="py-2.5 px-4 font-medium">Victim</th>
                            <th className="py-2.5 px-4 font-medium text-right">Confidence</th>
                            <th className="py-2.5 px-4 font-medium text-right">Screenshot</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/80">
                          {filteredEvents.map((ev) => {
                            const localShot = localScreenshots[ev.event_id];
                            const shotUrl =
                              jobId && ev.screenshot_file
                                ? `/api/jobs/${encodeURIComponent(jobId)}/files/screenshots/${encodeURIComponent(ev.screenshot_file)}`
                                : localShot?.dataUrl || null;
                            const shotFilename = ev.screenshot_file || localShot?.filename || `kill_${ev.event_id}.png`;

                            return (
                              <tr key={ev.event_id} className="hover:bg-slate-900/50">
                                <td className="py-2.5 px-4 font-mono tabular-nums text-slate-400">
                                  {String(ev.event_id).padStart(3, '0')}
                                </td>
                                <td className="py-2.5 px-4 font-mono tabular-nums text-slate-200">
                                  {ev.timestamp_hms}
                                </td>
                                <td className="py-2.5 px-4 text-slate-100 font-medium">
                                  {ev.killer || '—'}
                                </td>
                                <td className="py-2.5 px-4 text-emerald-400 font-mono text-xs">
                                  {ev.weapon_or_action || '—'}
                                </td>
                                <td className="py-2.5 px-4 text-slate-200">
                                  {ev.victim || '—'}
                                </td>
                                <td className="py-2.5 px-4 text-right font-mono tabular-nums text-slate-300">
                                  {(ev.confidence * 100).toFixed(0)}%
                                </td>
                                <td className="py-2.5 px-4 text-right whitespace-nowrap">
                                  {shotUrl ? (
                                    <div className="inline-flex items-center gap-3">
                                      <button
                                        type="button"
                                        onClick={() => setLightboxUrl(shotUrl)}
                                        className="text-xs font-medium text-emerald-400 hover:underline cursor-pointer"
                                      >
                                        View PNG
                                      </button>
                                      <a
                                        href={shotUrl}
                                        download={shotFilename}
                                        className="inline-flex items-center gap-1 text-xs font-medium text-slate-300 hover:text-white hover:underline"
                                      >
                                        <Download className="w-3 h-3" />
                                        Download
                                      </a>
                                    </div>
                                  ) : (
                                    <span className="text-xs text-slate-500">Capturing...</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* Highlight Clip Windows for Fast ROI-Only Scan */}
                {jobStatus.scan_mode === 'roi_only' && (jobStatus.clip_segments?.length ?? 0) > 0 && (
                  <div className="space-y-4 border border-slate-800 bg-slate-900/40 rounded-lg p-5">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h2 className="text-base font-semibold text-slate-100">
                          Highlight Clip Windows ({jobStatus.clip_segments!.length})
                        </h2>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Fast ROI-Only Scan detected timestamps without uploading the full video. Export WebM locally or upload full video for FFmpeg MP4 clips.
                        </p>
                      </div>
                      <div className="flex flex-wrap items-center gap-2.5">
                        {videoInfo?.localObjectUrl && (
                          <button
                            type="button"
                            disabled={isRecordingClips}
                            onClick={() => void handleRecordLocalBrowserClips(jobStatus.clip_segments!)}
                            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded-md bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-slate-700 text-slate-100 cursor-pointer"
                          >
                            <Video className="w-3.5 h-3.5 text-emerald-400" />
                            {isRecordingClips ? 'Recording Clips...' : 'Record Clips Locally (.webm)'}
                          </button>
                        )}
                        {selectedLocalFileRef.current && (
                          <button
                            type="button"
                            disabled={isUploading}
                            onClick={() => void handleUpgradeToServerMp4Clips()}
                            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white cursor-pointer"
                          >
                            <Upload className="w-3.5 h-3.5" />
                            Upload Full Video (6×) for FFmpeg MP4
                          </button>
                        )}
                      </div>
                    </div>

                    {clipRecordingMsg && (
                      <div className="text-xs text-emerald-400 font-mono">{clipRecordingMsg}</div>
                    )}

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {jobStatus.clip_segments!.map((seg) => (
                        <div key={seg.clip_index} className="p-3 rounded-md border border-slate-800 bg-slate-950/70 text-xs font-mono tabular-nums space-y-1">
                          <div className="text-emerald-400 font-semibold">
                            highlight_{String(seg.clip_index).padStart(3, '0')}
                          </div>
                          <div className="text-slate-200">
                            {formatDuration(seg.start_sec)} ({seg.start_sec.toFixed(1)}s) → {formatDuration(seg.end_sec)} ({seg.end_sec.toFixed(1)}s)
                          </div>
                          <div className="text-slate-400">
                            Duration: {seg.duration_sec.toFixed(1)}s · Kills: {seg.event_ids.map((id) => `#${String(id).padStart(3, '0')}`).join(', ')}
                          </div>
                        </div>
                      ))}
                    </div>

                    {localClips.length > 0 && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                        {localClips.map((clip) => (
                          <div key={clip.clip_index} className="border border-slate-800 bg-slate-900/60 rounded-lg overflow-hidden">
                            <video controls preload="metadata" src={clip.blobUrl} className="w-full aspect-video bg-black" />
                            <div className="p-3.5 flex items-center justify-between gap-4 border-t border-slate-800">
                              <div className="text-xs font-mono tabular-nums text-slate-300">
                                {clip.filename} · {clip.start_sec.toFixed(1)}s → {clip.end_sec.toFixed(1)}s ({clip.duration_sec.toFixed(1)}s)
                              </div>
                              <a
                                href={clip.blobUrl}
                                download={clip.filename}
                                className="inline-flex items-center gap-1 text-xs font-medium text-emerald-400 hover:underline shrink-0"
                              >
                                <Download className="w-3.5 h-3.5" />
                                Download WebM
                              </a>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Highlight Clips (Server FFmpeg MP4s) */}
                {(jobStatus.clips?.length ?? 0) > 0 && jobId && (
                  <div className="space-y-3">
                    <h2 className="text-base font-semibold text-slate-100">Generated Highlight Clips</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {jobStatus.clips!.map((clip) => {
                        const clipUrl = `/api/jobs/${encodeURIComponent(jobId)}/files/clips/${encodeURIComponent(clip.filename)}`;
                        return (
                          <div key={clip.clip_index} className="border border-slate-800 bg-slate-900/50 rounded-lg overflow-hidden">
                            {clip.success ? (
                              <video controls preload="metadata" src={clipUrl} className="w-full aspect-video bg-black" />
                            ) : (
                              <div className="aspect-video flex items-center justify-center text-xs text-red-400 p-4">
                                {clip.error_message || 'Failed to render clip'}
                              </div>
                            )}
                            <div className="p-3.5 flex items-center justify-between gap-4 border-t border-slate-800">
                              <div className="text-xs font-mono tabular-nums text-slate-300">
                                {clip.filename} · {clip.start_sec.toFixed(1)}s → {clip.end_sec.toFixed(1)}s ({clip.duration_sec.toFixed(1)}s)
                              </div>
                              {clip.success && (
                                <a
                                  href={clipUrl}
                                  download={clip.filename}
                                  className="inline-flex items-center gap-1 text-xs font-medium text-emerald-400 hover:underline shrink-0"
                                >
                                  <Download className="w-3.5 h-3.5" />
                                  Download MP4
                                </a>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Confirmed Screenshots & Debug ROI */}
                {(jobStatus.events?.length ?? 0) > 0 && jobId && (
                  <div className="space-y-3">
                    <h2 className="text-base font-semibold text-slate-100">Confirmed Screenshots & Debug ROI Crops</h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                      {jobStatus.events!.map((ev) => {
                        const localShot = localScreenshots[ev.event_id];
                        const shotUrl = ev.screenshot_file
                          ? `/api/jobs/${encodeURIComponent(jobId)}/files/screenshots/${encodeURIComponent(ev.screenshot_file)}`
                          : localShot?.dataUrl || null;
                        const shotFilename = ev.screenshot_file || localShot?.filename || `kill_${ev.event_id}.png`;
                        const roiUrl = ev.debug_roi_file
                          ? `/api/jobs/${encodeURIComponent(jobId)}/files/debug_roi/${encodeURIComponent(ev.debug_roi_file)}`
                          : null;
                        return (
                          <div key={ev.event_id} className="border border-slate-800 bg-slate-900/50 rounded-lg overflow-hidden">
                            {shotUrl ? (
                              <button
                                type="button"
                                onClick={() => setLightboxUrl(shotUrl)}
                                className="block w-full cursor-pointer"
                              >
                                <img
                                  src={shotUrl}
                                  alt={`Confirmed kill screenshot ${ev.event_id}`}
                                  referrerPolicy="no-referrer"
                                  className="w-full aspect-video object-cover"
                                />
                              </button>
                            ) : (
                              <div className="w-full aspect-video bg-slate-950 flex items-center justify-center text-xs text-slate-500">
                                Capturing full-res screenshot...
                              </div>
                            )}
                            <div className="p-3 space-y-2 border-t border-slate-800">
                              <div className="flex items-center justify-between gap-2 text-xs font-mono tabular-nums text-slate-300">
                                <span>#{String(ev.event_id).padStart(3, '0')} · {ev.timestamp_hms}</span>
                                <div className="flex items-center gap-3">
                                  <span>{(ev.confidence * 100).toFixed(0)}%</span>
                                  {shotUrl && (
                                    <a
                                      href={shotUrl}
                                      download={shotFilename}
                                      className="inline-flex items-center gap-1 font-sans font-medium text-emerald-400 hover:underline"
                                    >
                                      <Download className="w-3 h-3" />
                                      PNG
                                    </a>
                                  )}
                                </div>
                              </div>
                              {roiUrl && (
                                <img
                                  src={roiUrl}
                                  alt={`Debug ROI crop for event ${ev.event_id}`}
                                  referrerPolicy="no-referrer"
                                  className="w-full max-h-20 object-contain bg-slate-950 rounded border border-slate-800"
                                />
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {jobStatus.summary_text && (
                  <div className="space-y-2">
                    <h2 className="text-base font-semibold text-slate-100">Session Summary Log</h2>
                    <pre className="p-4 rounded-lg bg-slate-900/60 border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed">
                      {jobStatus.summary_text}
                    </pre>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {activeTab === 'calibration' && (
          <div className="p-6 lg:p-8 max-w-4xl space-y-6">
            <div className="space-y-2 border-b border-slate-800 pb-4">
              <h1 className="text-xl font-bold text-slate-100">BGMI Reference Sample Calibration</h1>
              <p className="text-sm text-slate-400">
                Legacy templates from the original PEACE repository (<code className="text-slate-300">Capture0..10.PNG</code>)
                were archived in <code className="text-slate-300">Scripts/KillFeed/legacy_non_bgmi/</code> and are never loaded automatically.
                Add real BGMI weapon icons or kill-feed strip crops here to improve template-assisted confidence.
              </p>
            </div>

            <div className="border border-slate-800 bg-slate-900/50 rounded-lg p-5 space-y-3">
              <h2 className="text-sm font-semibold text-slate-100">How to Add Real BGMI Calibration Samples</h2>
              <ol className="list-decimal list-inside text-sm text-slate-300 space-y-1.5">
                <li>Load your BGMI gameplay recording in the Workspace tab.</li>
                <li>Seek the Preview Timestamp to a frame where a real BGMI kill-feed banner is visible.</li>
                <li>Drag a tight rectangle around a BGMI weapon icon or kill-feed banner, enter a label (e.g. <code className="font-mono text-emerald-400">bgmi_m416</code>), and click <strong>Save Selection as Calibration Sample</strong>.</li>
              </ol>
            </div>

            <div className="space-y-3">
              <h2 className="text-sm font-semibold text-slate-100">
                Active Samples ({systemStatus?.calibration_samples.length ?? 0})
              </h2>

              {!systemStatus || systemStatus.calibration_samples.length === 0 ? (
                <div className="border border-slate-800 rounded-lg p-6 text-sm text-slate-400">
                  No user-calibrated BGMI reference samples loaded yet. Detection currently relies on structural + OCR rules.
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                  {systemStatus.calibration_samples.map((s) => (
                    <div key={s.filename} className="border border-slate-800 bg-slate-900/50 rounded-lg p-3 space-y-2">
                      <img
                        src={`/api/calibration/samples/${encodeURIComponent(s.filename)}`}
                        alt={s.name}
                        referrerPolicy="no-referrer"
                        className="w-full h-16 object-contain bg-slate-950 rounded border border-slate-800"
                      />
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-mono text-slate-200 truncate">{s.filename}</span>
                        <button
                          type="button"
                          onClick={() => handleDeleteCalibrationSample(s.filename)}
                          className="text-red-400 hover:text-red-300 p-1 cursor-pointer"
                          title="Delete sample"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'setup' && (
          <div className="p-6 lg:p-8 max-w-4xl space-y-6">
            <div className="space-y-2 border-b border-slate-800 pb-4">
              <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
                <Terminal className="w-4 h-4" />
                <span>Windows 10/11 · Local PC Setup (Desktop + Web UI)</span>
              </div>
              <h1 className="text-xl font-bold text-slate-100">Local Installation & Launch Commands</h1>
              <p className="text-sm text-slate-400">
                Download this project to <code className="text-slate-300">C:\PEACE-main</code> and run either the Python Desktop GUI or the zero-upload local Web UI.
              </p>
            </div>

            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-slate-100">1. PowerShell Python & FFmpeg Setup</h2>
              <pre className="p-4 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed">
{`cd C:\\PEACE-main
python --version
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
winget install Gyan.FFmpeg`}
              </pre>
            </div>

            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-slate-100">2A. Launch Python Desktop App (No Browser Needed)</h2>
              <pre className="p-4 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed">
{`.\\.venv\\Scripts\\Activate.ps1
ffmpeg -version
python Scripts\\main.py`}
              </pre>
            </div>

            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-slate-100">2B. Launch Local Web UI (Zero-Upload PC File Path Support)</h2>
              <pre className="p-4 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed">
{`npm install
npm run dev
# Then open http://localhost:3000 and paste your local C:\\Videos\\bgmi.mp4 path`}
              </pre>
            </div>

            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-slate-100">3. Run Automated Verification Suite</h2>
              <pre className="p-4 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed">
{`python -m unittest discover -s tests -v`}
              </pre>
            </div>
          </div>
        )}
      </main>

      {/* Lightbox Modal for Screenshots */}
      {lightboxUrl && (
        <div
          onClick={() => setLightboxUrl(null)}
          className="fixed inset-0 z-50 bg-black/85 flex items-center justify-center p-6"
        >
          <button
            type="button"
            onClick={() => setLightboxUrl(null)}
            className="absolute top-5 right-5 p-2 rounded-lg bg-slate-900 border border-slate-700 text-slate-200 hover:bg-slate-800 cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
          <img
            src={lightboxUrl}
            alt="Full-frame confirmed kill screenshot"
            referrerPolicy="no-referrer"
            onClick={(e) => e.stopPropagation()}
            className="max-w-full max-h-full rounded-lg border border-slate-700"
          />
        </div>
      )}
    </div>
  );
}
