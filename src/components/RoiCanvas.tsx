import React, { useEffect, useRef, useState } from 'react';
import { Crosshair, RotateCcw } from 'lucide-react';
import type { ROIRegion } from '../types';

interface RoiCanvasProps {
  previewUrl: string | null;
  videoWidth: number;
  videoHeight: number;
  roi: ROIRegion | null;
  onChangeRoi: (roi: ROIRegion | null) => void;
  isLoadingPreview: boolean;
  onPreviewReady?: () => void;
}

export const RoiCanvas: React.FC<RoiCanvasProps> = ({
  previewUrl,
  videoWidth,
  videoHeight,
  roi,
  onChangeRoi,
  isLoadingPreview,
  onPreviewReady,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [draftRoi, setDraftRoi] = useState<ROIRegion | null>(null);
  const [imgError, setImgError] = useState(false);

  useEffect(() => {
    setImgError(false);
  }, [previewUrl]);

  const toNativeCoords = (clientX: number, clientY: number): { x: number; y: number } | null => {
    const el = containerRef.current;
    if (!el || videoWidth <= 0 || videoHeight <= 0) return null;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const relX = Math.max(0, Math.min(rect.width, clientX - rect.left));
    const relY = Math.max(0, Math.min(rect.height, clientY - rect.top));
    return {
      x: Math.round((relX / rect.width) * videoWidth),
      y: Math.round((relY / rect.height) * videoHeight),
    };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!previewUrl || imgError) return;
    const pt = toNativeCoords(e.clientX, e.clientY);
    if (!pt) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragStart(pt);
    setDraftRoi({ x: pt.x, y: pt.y, width: 1, height: 1 });
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStart) return;
    const pt = toNativeCoords(e.clientX, e.clientY);
    if (!pt) return;
    const x = Math.min(dragStart.x, pt.x);
    const y = Math.min(dragStart.y, pt.y);
    const width = Math.max(1, Math.abs(pt.x - dragStart.x));
    const height = Math.max(1, Math.abs(pt.y - dragStart.y));
    setDraftRoi({ x, y, width, height });
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStart) return;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // Ignore if pointer capture already released
    }
    if (draftRoi && draftRoi.width >= 8 && draftRoi.height >= 8) {
      onChangeRoi(draftRoi);
    }
    setDragStart(null);
    setDraftRoi(null);
  };

  const activeRoi = draftRoi ?? roi;
  const isValidForDetection = activeRoi ? activeRoi.width >= 40 && activeRoi.height >= 16 : false;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs text-slate-300">
          <Crosshair className="w-4 h-4 text-emerald-500 shrink-0" />
          <span>Click and drag across the preview frame to select the BGMI kill-feed region.</span>
        </div>
        <div className="flex items-center gap-4">
          {activeRoi ? (
            <span className="font-mono text-xs tabular-nums text-slate-300">
              x={activeRoi.x} · y={activeRoi.y} · {activeRoi.width}×{activeRoi.height}px
              {!isValidForDetection && (
                <span className="text-amber-400 ml-2">(Min 40×16px for scan)</span>
              )}
            </span>
          ) : (
            <span className="text-xs text-amber-400">Region required before scanning</span>
          )}
          {roi && (
            <button
              type="button"
              onClick={() => onChangeRoi(null)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Clear Region
            </button>
          )}
        </div>
      </div>

      <div className="w-full bg-slate-950 border border-slate-800 rounded-lg overflow-hidden flex items-center justify-center min-h-[240px]">
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          style={
            videoWidth > 0 && videoHeight > 0
              ? {
                  aspectRatio: `${videoWidth} / ${videoHeight}`,
                  width: '100%',
                  maxWidth: `calc(540px * ${videoWidth} / ${videoHeight})`,
                  maxHeight: '540px',
                }
              : { aspectRatio: '16 / 9', width: '100%', maxHeight: '540px' }
          }
          className="relative bg-slate-950 select-none cursor-crosshair touch-none"
        >
          {previewUrl && !imgError ? (
            <img
              src={previewUrl}
              alt="Video preview frame for kill-feed ROI selection"
              draggable={false}
              referrerPolicy="no-referrer"
              onLoad={() => {
                setImgError(false);
                onPreviewReady?.();
              }}
              onError={() => {
                setImgError(true);
                onPreviewReady?.();
              }}
              className="w-full h-full block pointer-events-none"
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 text-sm px-6 text-center">
              {isLoadingPreview
                ? 'Extracting preview frame from video...'
                : 'Preview frame unavailable for this timestamp.'}
            </div>
          )}

          {isLoadingPreview && previewUrl && (
            <div className="absolute inset-0 bg-slate-950/60 flex items-center justify-center text-xs font-medium text-slate-200">
              Loading frame...
            </div>
          )}

          {activeRoi && videoWidth > 0 && videoHeight > 0 && (
            <div
              style={{
                left: `${(activeRoi.x / videoWidth) * 100}%`,
                top: `${(activeRoi.y / videoHeight) * 100}%`,
                width: `${(activeRoi.width / videoWidth) * 100}%`,
                height: `${(activeRoi.height / videoHeight) * 100}%`,
              }}
              className={`absolute pointer-events-none border-2 ${
                isValidForDetection
                  ? 'border-emerald-400 bg-emerald-500/15'
                  : 'border-amber-400 bg-amber-500/15'
              }`}
            >
              <span className="absolute -top-6 left-0 px-1.5 py-0.5 text-[11px] font-mono tabular-nums bg-slate-950/90 text-emerald-300 border border-emerald-500/40 rounded whitespace-nowrap">
                Kill-Feed ROI ({activeRoi.width}×{activeRoi.height})
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
