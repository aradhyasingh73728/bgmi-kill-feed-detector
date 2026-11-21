import express, { type Request, type Response } from "express";
import { spawn, execFile } from "child_process";
import crypto from "crypto";
import fs from "fs";
import os from "os";
import path from "path";
import { promisify } from "util";
import { fileURLToPath } from "url";

const execFileAsync = promisify(execFile);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WORKSPACE_ROOT = __dirname;
const WEB_BRIDGE_PATH = path.join(WORKSPACE_ROOT, "Scripts", "web_bridge.py");
const CALIBRATION_SAMPLES_DIR = path.join(WORKSPACE_ROOT, "Scripts", "calibration", "samples");

const TEMP_ROOT = path.join(os.tmpdir(), "peace_web_state");
const VIDEOS_DIR = path.join(TEMP_ROOT, "videos");
const PREVIEWS_DIR = path.join(TEMP_ROOT, "previews");
const JOBS_DIR = path.join(TEMP_ROOT, "jobs");
const ROI_SCANS_DIR = path.join(TEMP_ROOT, "roi_scans");

for (const dir of [VIDEOS_DIR, PREVIEWS_DIR, JOBS_DIR, ROI_SCANS_DIR, CALIBRATION_SAMPLES_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

interface StoredVideo {
  id: string;
  originalName: string;
  filePath: string;
  isSyntheticDemo: boolean;
  isLocalFile?: boolean;
  metadata: Record<string, unknown>;
}

interface JobRecord {
  jobId: string;
  videoId: string;
  statusFile: string;
  configFile: string;
  outputBaseDir: string;
}

interface RoiScanRecord {
  scanId: string;
  dir: string;
  videoName: string;
  width: number;
  height: number;
  duration: number;
  frames: Array<{ ts: number; path: string }>;
}

const videoRegistry = new Map<string, StoredVideo>();
const jobRegistry = new Map<string, JobRecord>();
const roiScanRegistry = new Map<string, RoiScanRecord>();

function resolvePythonExecutable(): string {
  const candidates =
    process.platform === "win32"
      ? [
          path.join(WORKSPACE_ROOT, ".venv", "Scripts", "python.exe"),
          path.join(WORKSPACE_ROOT, ".venv", "Scripts", "python"),
          path.join(WORKSPACE_ROOT, ".venv", "bin", "python3"),
          path.join(WORKSPACE_ROOT, ".venv", "bin", "python"),
          "python",
          "python3",
        ]
      : [
          path.join(WORKSPACE_ROOT, ".venv", "bin", "python3"),
          path.join(WORKSPACE_ROOT, ".venv", "bin", "python"),
          path.join(WORKSPACE_ROOT, ".venv", "Scripts", "python.exe"),
          "python3",
          "python",
        ];
  for (const candidate of candidates) {
    if (candidate === "python3" || candidate === "python") {
      return candidate;
    }
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

let pythonDepsVerified = false;
let pythonBootPromise: Promise<void> | null = null;

async function ensurePythonEnvironment(): Promise<void> {
  if (pythonDepsVerified) return;
  if (pythonBootPromise) return pythonBootPromise;

  pythonBootPromise = (async () => {
    const pythonBin = resolvePythonExecutable();
    try {
      await execFileAsync(pythonBin, ["-c", "import cv2, numpy, PIL"], {
        cwd: WORKSPACE_ROOT,
        timeout: 15000,
      });
      pythonDepsVerified = true;
      return;
    } catch {
      // Fall through to auto-install missing Python dependencies
    }

    console.log("Bootstrapping missing Python OpenCV/OCR dependencies...");
    // Strategy 1: If pip is already available (Windows / standard Python env), install from requirements.txt
    let pipInstalled = false;
    try {
      await execFileAsync(pythonBin, ["-m", "pip", "--version"], {
        cwd: WORKSPACE_ROOT,
        timeout: 10000,
      });
      await execFileAsync(
        pythonBin,
        ["-m", "pip", "install", "-q", "-r", path.join(WORKSPACE_ROOT, "requirements.txt")],
        { cwd: WORKSPACE_ROOT, timeout: 240000, maxBuffer: 10 * 1024 * 1024 }
      );
      pipInstalled = true;
    } catch {
      pipInstalled = false;
    }

    // Strategy 2: Linux container where base python3 has no pip bundled (fast ~20s wheel bootstrap, zero dpkg prompts)
    if (!pipInstalled && process.platform === "linux") {
      await execFileAsync(
        "sh",
        [
          "-c",
          "if curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py; then " +
            "python3 /tmp/get-pip.py --break-system-packages -q && rm -f /tmp/get-pip.py && " +
            "pip3 install --break-system-packages -q opencv-python-headless numpy pillow rapidocr-onnxruntime ffmpeg-python; " +
            "else " +
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq -o Dpkg::Options::='--force-confdef' -o Dpkg::Options::='--force-confold' python3-pip python3-opencv python3-numpy python3-pil && " +
            "pip3 install --break-system-packages -q rapidocr-onnxruntime ffmpeg-python; " +
            "fi",
        ],
        { cwd: WORKSPACE_ROOT, timeout: 300000, maxBuffer: 10 * 1024 * 1024 }
      );
    }

    await execFileAsync(pythonBin, ["-c", "import cv2, numpy, PIL"], {
      cwd: WORKSPACE_ROOT,
      timeout: 15000,
    });
    pythonDepsVerified = true;
    console.log("Python OpenCV/OCR environment verified.");
  })().catch((err) => {
    pythonBootPromise = null;
    throw err;
  });

  return pythonBootPromise;
}

async function runBridge(args: string[], timeoutMs = 45000): Promise<Record<string, unknown>> {
  await ensurePythonEnvironment();
  const pythonBin = resolvePythonExecutable();
  try {
    const { stdout, stderr } = await execFileAsync(pythonBin, [WEB_BRIDGE_PATH, ...args], {
      cwd: WORKSPACE_ROOT,
      maxBuffer: 10 * 1024 * 1024,
      timeout: timeoutMs,
    });
    const text = stdout.trim();
    if (!text) {
      throw new Error(stderr.trim() || "Python bridge returned empty response.");
    }
    return JSON.parse(text) as Record<string, unknown>;
  } catch (err: unknown) {
    const e = err as { stderr?: string; stdout?: string; message?: string };
    const stderrText = (e.stderr || "").trim();
    if (stderrText) {
      try {
        const parsed = JSON.parse(stderrText) as { error?: string };
        if (parsed.error) {
          throw new Error(parsed.error);
        }
      } catch (parseErr) {
        if (parseErr instanceof Error && parseErr.message !== stderrText && !parseErr.message.includes("JSON")) {
          throw parseErr;
        }
      }
      throw new Error(stderrText);
    }
    throw err instanceof Error ? err : new Error(String(err));
  }
}

function sanitizeBasename(name: string): string {
  return path.basename(name).replace(/[^a-zA-Z0-9._-]/g, "_");
}

async function createServer() {
  const app = express();
  const PORT = 3000;

  app.get("/api/system-status", async (_req: Request, res: Response) => {
    try {
      res.setHeader("Cache-Control", "no-store");
      const status = await runBridge(["status"]);
      res.json(status);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : "Failed to probe Python status." });
    }
  });

  app.get("/api/calibration/samples/:filename", (req: Request, res: Response) => {
    const filename = sanitizeBasename(String(req.params.filename || ""));
    const fullPath = path.resolve(CALIBRATION_SAMPLES_DIR, filename);
    if (!fullPath.startsWith(path.resolve(CALIBRATION_SAMPLES_DIR) + path.sep) || !fs.existsSync(fullPath)) {
      res.status(404).json({ error: "Calibration sample not found." });
      return;
    }
    res.sendFile(fullPath);
  });

  app.delete("/api/calibration/samples/:filename", (req: Request, res: Response) => {
    const filename = sanitizeBasename(String(req.params.filename || ""));
    const fullPath = path.resolve(CALIBRATION_SAMPLES_DIR, filename);
    if (!fullPath.startsWith(path.resolve(CALIBRATION_SAMPLES_DIR) + path.sep) || !fs.existsSync(fullPath)) {
      res.status(404).json({ error: "Calibration sample not found." });
      return;
    }
    fs.unlinkSync(fullPath);
    res.json({ deleted: filename });
  });

  const pendingUploads = new Map<string, { originalName: string; filePath: string; totalSize: number; receivedBytes: number }>();

  app.post("/api/upload/init", express.json(), (req: Request, res: Response) => {
    try {
      const { filename, totalSize } = req.body as { filename?: string; totalSize?: number };
      const rawName = String(filename || "gameplay.mp4");
      const safeName = sanitizeBasename(rawName) || "gameplay.mp4";
      const ext = path.extname(safeName) || ".mp4";
      const videoId = crypto.randomUUID();
      const filePath = path.join(VIDEOS_DIR, `${videoId}${ext}`);
      fs.writeFileSync(filePath, Buffer.alloc(0));
      pendingUploads.set(videoId, {
        originalName: rawName,
        filePath,
        totalSize: Number(totalSize) || 0,
        receivedBytes: 0,
      });
      res.json({ videoId });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : "Failed to initialize upload." });
    }
  });

  app.post(
    "/api/upload/chunk",
    express.raw({ type: () => true, limit: "16mb" }),
    (req: Request, res: Response) => {
      try {
        const videoId = String(req.query.videoId || "");
        const pending = pendingUploads.get(videoId);
        if (!pending) {
          res.status(404).json({ error: "Upload session not found or expired." });
          return;
        }
        const chunk = req.body as Buffer;
        if (!Buffer.isBuffer(chunk) || chunk.length === 0) {
          res.status(400).json({ error: "Empty chunk payload." });
          return;
        }
        const rawOffset = req.query.offset;
        if (rawOffset !== undefined && String(rawOffset).trim() !== "") {
          const offset = Number(rawOffset);
          if (!Number.isInteger(offset) || offset < 0) {
            res.status(400).json({ error: "Invalid chunk offset." });
            return;
          }
          const fd = fs.openSync(pending.filePath, "r+");
          try {
            fs.writeSync(fd, chunk, 0, chunk.length, offset);
          } finally {
            fs.closeSync(fd);
          }
        } else {
          fs.appendFileSync(pending.filePath, chunk);
        }
        pending.receivedBytes += chunk.length;
        res.json({ receivedBytes: pending.receivedBytes });
      } catch (err) {
        res.status(500).json({ error: err instanceof Error ? err.message : "Failed to write chunk." });
      }
    }
  );

  app.post("/api/upload/complete", express.json(), async (req: Request, res: Response) => {
    const videoId = String((req.body as { videoId?: string })?.videoId || "");
    const pending = pendingUploads.get(videoId);
    if (!pending) {
      res.status(404).json({ error: "Upload session not found." });
      return;
    }
    pendingUploads.delete(videoId);

    try {
      if (!fs.existsSync(pending.filePath) || fs.statSync(pending.filePath).size === 0) {
        throw new Error("Uploaded file is empty.");
      }
      const metadata = await runBridge(["metadata", "--video", pending.filePath]);
      const record: StoredVideo = {
        id: videoId,
        originalName: pending.originalName,
        filePath: pending.filePath,
        isSyntheticDemo: false,
        metadata,
      };
      videoRegistry.set(videoId, record);

      res.json({
        videoId,
        originalName: pending.originalName,
        isSyntheticDemo: false,
        metadata,
      });
    } catch (err) {
      if (fs.existsSync(pending.filePath)) {
        try {
          fs.unlinkSync(pending.filePath);
        } catch {
          // ignore cleanup failure
        }
      }
      res.status(400).json({ error: err instanceof Error ? err.message : "Failed to inspect uploaded video." });
    }
  });

  app.post("/api/upload/cancel", express.json(), (req: Request, res: Response) => {
    const videoId = String((req.body as { videoId?: string })?.videoId || "");
    const pending = pendingUploads.get(videoId);
    if (pending) {
      pendingUploads.delete(videoId);
      if (fs.existsSync(pending.filePath)) {
        try {
          fs.unlinkSync(pending.filePath);
        } catch {
          // ignore cleanup failure
        }
      }
    }
    res.json({ cancelled: true });
  });

  app.post("/api/upload-video", async (req: Request, res: Response) => {
    let savedPath = "";
    try {
      const rawName = String(req.query.filename || req.headers["x-filename"] || "gameplay.mp4");
      const safeName = sanitizeBasename(rawName) || "gameplay.mp4";
      const ext = path.extname(safeName) || ".mp4";
      const videoId = crypto.randomUUID();
      savedPath = path.join(VIDEOS_DIR, `${videoId}${ext}`);

      const bytesWritten = await new Promise<number>((resolve, reject) => {
        let total = 0;
        const ws = fs.createWriteStream(savedPath);
        req.on("data", (chunk: Buffer) => {
          total += chunk.length;
        });
        req.on("aborted", () => {
          ws.destroy();
          reject(new Error("Upload aborted by client."));
        });
        req.on("error", (err) => {
          ws.destroy();
          reject(err);
        });
        ws.on("error", reject);
        ws.on("finish", () => resolve(total));
        req.pipe(ws);
      });

      if (bytesWritten === 0) {
        if (fs.existsSync(savedPath)) fs.unlinkSync(savedPath);
        res.status(400).json({ error: "Empty video payload." });
        return;
      }

      const metadata = await runBridge(["metadata", "--video", savedPath]);
      const record: StoredVideo = {
        id: videoId,
        originalName: rawName,
        filePath: savedPath,
        isSyntheticDemo: false,
        metadata,
      };
      videoRegistry.set(videoId, record);

      res.json({
        videoId,
        originalName: rawName,
        isSyntheticDemo: false,
        metadata,
      });
    } catch (err) {
      if (savedPath && fs.existsSync(savedPath)) {
        try {
          fs.unlinkSync(savedPath);
        } catch {
          // ignore cleanup failure
        }
      }
      res.status(400).json({ error: err instanceof Error ? err.message : "Failed to inspect uploaded video." });
    }
  });

  app.post("/api/generate-demo-video", express.json(), async (_req: Request, res: Response) => {
    try {
      const videoId = crypto.randomUUID();
      const outPath = path.join(VIDEOS_DIR, `${videoId}_synthetic_demo.mp4`);
      const info = await runBridge(["generate-demo", "--out", outPath]);
      const metadata = info.metadata as Record<string, unknown>;
      const record: StoredVideo = {
        id: videoId,
        originalName: "synthetic_bgmi_test_720p.mp4",
        filePath: outPath,
        isSyntheticDemo: true,
        metadata,
      };
      videoRegistry.set(videoId, record);

      res.json({
        videoId,
        originalName: record.originalName,
        isSyntheticDemo: true,
        suggestedRoi: info.suggested_roi,
        metadata,
      });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : "Failed to create synthetic test video." });
    }
  });

  app.post("/api/open-local-video", express.json(), async (req: Request, res: Response) => {
    try {
      const rawInput = String((req.body as { filePath?: string })?.filePath || "").trim();
      const cleaned = rawInput.replace(/^["']|["']$/g, "").trim();
      if (!cleaned) {
        res.status(400).json({ error: "Please provide a valid local video file path." });
        return;
      }
      const resolved = path.resolve(cleaned);
      if (!fs.existsSync(resolved)) {
        res.status(404).json({ error: `File not found on this PC: ${resolved}` });
        return;
      }
      const stat = fs.statSync(resolved);
      if (!stat.isFile()) {
        res.status(400).json({ error: "Specified path is not a video file." });
        return;
      }
      const metadata = await runBridge(["metadata", "--video", resolved]);
      const videoId = crypto.randomUUID();
      const originalName = path.basename(resolved);
      const record: StoredVideo = {
        id: videoId,
        originalName,
        filePath: resolved,
        isSyntheticDemo: false,
        isLocalFile: true,
        metadata,
      };
      videoRegistry.set(videoId, record);

      res.json({
        videoId,
        originalName,
        isSyntheticDemo: false,
        isLocalFile: true,
        metadata,
      });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : "Failed to inspect local video file." });
    }
  });

  app.get("/api/videos/:videoId/stream", (req: Request, res: Response) => {
    const videoId = String(req.params.videoId || "");
    const video = videoRegistry.get(videoId);
    if (!video || !fs.existsSync(video.filePath)) {
      res.status(404).json({ error: "Video not found." });
      return;
    }

    const stat = fs.statSync(video.filePath);
    const fileSize = stat.size;
    const ext = path.extname(video.filePath).toLowerCase();
    const mimeMap: Record<string, string> = {
      ".mp4": "video/mp4",
      ".m4v": "video/mp4",
      ".webm": "video/webm",
      ".mov": "video/quicktime",
      ".mkv": "video/x-matroska",
      ".avi": "video/x-msvideo",
    };
    const contentType = mimeMap[ext] || "video/mp4";

    const range = req.headers.range;
    if (range) {
      const match = /bytes=(\d*)-(\d*)/.exec(range);
      if (!match) {
        res.status(416).setHeader("Content-Range", `bytes */${fileSize}`).end();
        return;
      }
      const start = match[1] ? Number(match[1]) : 0;
      const end = match[2] ? Math.min(fileSize - 1, Number(match[2])) : fileSize - 1;
      if (Number.isNaN(start) || Number.isNaN(end) || start > end || start >= fileSize) {
        res.status(416).setHeader("Content-Range", `bytes */${fileSize}`).end();
        return;
      }
      res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Accept-Ranges": "bytes",
        "Content-Length": end - start + 1,
        "Content-Type": contentType,
        "Cache-Control": "no-store",
      });
      fs.createReadStream(video.filePath, { start, end }).pipe(res);
      return;
    }

    res.writeHead(200, {
      "Content-Length": fileSize,
      "Content-Type": contentType,
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
    });
    fs.createReadStream(video.filePath).pipe(res);
  });

  app.get("/api/preview-frame", async (req: Request, res: Response) => {
    try {
      const videoId = String(req.query.videoId || "");
      const video = videoRegistry.get(videoId);
      if (!video || !fs.existsSync(video.filePath)) {
        res.status(404).json({ error: "Video not found." });
        return;
      }
      const tsParam = req.query.ts !== undefined ? String(req.query.ts) : "";
      const previewPath = path.join(PREVIEWS_DIR, `${videoId}_${Date.now()}.png`);
      const args = ["preview", "--video", video.filePath, "--out", previewPath];
      if (tsParam !== "" && !Number.isNaN(Number(tsParam))) {
        args.push("--ts", String(Math.max(0, Number(tsParam))));
      }
      const info = await runBridge(args);
      res.setHeader("X-Actual-Timestamp", String(info.actual_ts ?? 0));
      res.setHeader("Cache-Control", "no-store");
      res.sendFile(previewPath, () => {
        fs.unlink(previewPath, () => undefined);
      });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : "Preview extraction failed." });
    }
  });

  app.post("/api/calibration/capture", express.json({ limit: "15mb" }), async (req: Request, res: Response) => {
    let tempImgPath = "";
    try {
      const { videoId, timestampSec, imageBase64, roi, label } = req.body as {
        videoId?: string;
        timestampSec?: number;
        imageBase64?: string;
        roi: { x: number; y: number; width: number; height: number };
        label?: string;
      };
      if (!roi || roi.width < 8 || roi.height < 8) {
        res.status(400).json({ error: "Select a region at least 8x8px on the preview frame first." });
        return;
      }
      const roiArg = `${Math.round(roi.x)},${Math.round(roi.y)},${Math.round(roi.width)},${Math.round(roi.height)}`;

      if (imageBase64) {
        const b64 = imageBase64.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, "");
        tempImgPath = path.join(PREVIEWS_DIR, `cal_${crypto.randomUUID()}.png`);
        fs.writeFileSync(tempImgPath, Buffer.from(b64, "base64"));
        const result = await runBridge([
          "save-calibration",
          "--image",
          tempImgPath,
          "--roi",
          roiArg,
          "--label",
          String(label || "bgmi_sample"),
        ]);
        res.json(result);
        return;
      }

      const video = videoId ? videoRegistry.get(videoId) : undefined;
      if (!video) {
        res.status(404).json({ error: "Video not found." });
        return;
      }
      const result = await runBridge([
        "save-calibration",
        "--video",
        video.filePath,
        "--ts",
        String(Math.max(0, Number(timestampSec) || 0)),
        "--roi",
        roiArg,
        "--label",
        String(label || "bgmi_sample"),
      ]);
      res.json(result);
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : "Failed to save calibration sample." });
    } finally {
      if (tempImgPath && fs.existsSync(tempImgPath)) {
        fs.unlink(tempImgPath, () => undefined);
      }
    }
  });

  function spawnJobProcess(args: string[], statusFile: string): void {
    const pythonBin = resolvePythonExecutable();
    const child = spawn(pythonBin, [WEB_BRIDGE_PATH, ...args], {
      cwd: WORKSPACE_ROOT,
      stdio: ["ignore", "ignore", "pipe"],
      detached: false,
    });

    let stderrTail = "";
    child.stderr?.on("data", (chunk: Buffer) => {
      stderrTail = (stderrTail + chunk.toString("utf-8")).slice(-4096);
    });

    child.on("error", (err) => {
      fs.writeFileSync(
        statusFile,
        JSON.stringify({ state: "failed", progress: 0, message: err.message, error: err.message })
      );
    });

    child.on("close", (code) => {
      if (code !== 0) {
        try {
          const current = fs.existsSync(statusFile)
            ? (JSON.parse(fs.readFileSync(statusFile, "utf-8")) as { state?: string })
            : {};
          if (current.state === "running") {
            const msg = stderrTail.trim() || `Python pipeline exited with code ${code}`;
            fs.writeFileSync(
              statusFile,
              JSON.stringify({ state: "failed", progress: 0, message: msg, error: msg })
            );
          }
        } catch {
          // ignore status file read/write error on exit
        }
      }
    });
  }

  app.post("/api/roi-scan/init", express.json(), (req: Request, res: Response) => {
    try {
      const { videoName, width, height, duration } = req.body as {
        videoName?: string;
        width?: number;
        height?: number;
        duration?: number;
      };
      const scanId = crypto.randomUUID();
      const dir = path.join(ROI_SCANS_DIR, scanId);
      fs.mkdirSync(dir, { recursive: true });
      roiScanRegistry.set(scanId, {
        scanId,
        dir,
        videoName: String(videoName || "gameplay.mp4"),
        width: Math.max(1, Math.round(Number(width) || 1280)),
        height: Math.max(1, Math.round(Number(height) || 720)),
        duration: Math.max(0, Number(duration) || 0),
        frames: [],
      });
      res.json({ scanId });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : "Failed to initialize ROI scan." });
    }
  });

  app.post("/api/roi-scan/batch", express.json({ limit: "25mb" }), (req: Request, res: Response) => {
    try {
      const { scanId, frames } = req.body as {
        scanId?: string;
        frames?: Array<{ ts: number; dataUrl: string }>;
      };
      const rec = scanId ? roiScanRegistry.get(scanId) : undefined;
      if (!rec) {
        res.status(404).json({ error: "ROI scan session not found or expired." });
        return;
      }
      if (!Array.isArray(frames) || frames.length === 0) {
        res.status(400).json({ error: "No ROI frame crops provided in batch." });
        return;
      }
      let savedCount = 0;
      for (const item of frames) {
        const ts = Number(item.ts);
        const rawUrl = String(item.dataUrl || "");
        if (!Number.isFinite(ts) || !rawUrl) continue;
        const isPng = rawUrl.startsWith("data:image/png");
        const ext = isPng ? ".png" : ".jpg";
        const b64 = rawUrl.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, "");
        const idx = rec.frames.length + 1;
        const outPath = path.join(rec.dir, `roi_${String(idx).padStart(6, "0")}${ext}`);
        fs.writeFileSync(outPath, Buffer.from(b64, "base64"));
        rec.frames.push({ ts, path: outPath });
        savedCount++;
      }
      res.json({ savedCount, totalFrames: rec.frames.length });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : "Failed to store ROI crop batch." });
    }
  });

  app.post("/api/roi-scan/start", express.json(), async (req: Request, res: Response) => {
    try {
      await ensurePythonEnvironment();
      const { scanId, roi, config } = req.body as {
        scanId?: string;
        roi: { x: number; y: number; width: number; height: number };
        config: Record<string, unknown>;
      };
      const rec = scanId ? roiScanRegistry.get(scanId) : undefined;
      if (!rec) {
        res.status(404).json({ error: "ROI scan session not found." });
        return;
      }
      if (rec.frames.length === 0) {
        res.status(400).json({ error: "No ROI frames were uploaded for this scan." });
        return;
      }
      if (!roi || roi.width < 40 || roi.height < 16) {
        res.status(400).json({ error: "Invalid Kill-Feed ROI (must be at least 40x16px)." });
        return;
      }

      const jobId = crypto.randomUUID();
      const jobDir = path.join(JOBS_DIR, jobId);
      const outputBaseDir = path.join(jobDir, "output");
      fs.mkdirSync(outputBaseDir, { recursive: true });

      const sortedFrames = [...rec.frames].sort((a, b) => a.ts - b.ts);
      const manifestFile = path.join(jobDir, "roi_manifest.json");
      const configFile = path.join(jobDir, "config.json");
      const statusFile = path.join(jobDir, "status.json");

      fs.writeFileSync(manifestFile, JSON.stringify(sortedFrames), "utf-8");
      fs.writeFileSync(configFile, JSON.stringify(config || {}, null, 2), "utf-8");
      fs.writeFileSync(
        statusFile,
        JSON.stringify({ state: "running", progress: 0, message: "Starting Fast ROI-Only scan...", confirmed_count: 0 }),
        "utf-8"
      );

      jobRegistry.set(jobId, { jobId, videoId: `roi:${scanId}`, statusFile, configFile, outputBaseDir });
      roiScanRegistry.delete(rec.scanId);

      const roiArg = `${Math.round(roi.x)},${Math.round(roi.y)},${Math.round(roi.width)},${Math.round(roi.height)}`;
      spawnJobProcess(
        [
          "scan-roi-frames",
          "--manifest",
          manifestFile,
          "--video-name",
          rec.videoName,
          "--width",
          String(rec.width),
          "--height",
          String(rec.height),
          "--duration",
          String(rec.duration),
          "--roi",
          roiArg,
          "--config",
          configFile,
          "--out-base",
          outputBaseDir,
          "--status-file",
          statusFile,
        ],
        statusFile
      );

      res.json({ jobId });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : "Failed to start Fast ROI-Only scan." });
    }
  });

  app.post("/api/process", express.json(), async (req: Request, res: Response) => {
    try {
      await ensurePythonEnvironment();
      const { videoId, roi, config } = req.body as {
        videoId: string;
        roi: { x: number; y: number; width: number; height: number };
        config: Record<string, unknown>;
      };

      const video = videoRegistry.get(videoId);
      if (!video || !fs.existsSync(video.filePath)) {
        res.status(404).json({ error: "Video not found. Upload or generate a video first." });
        return;
      }

      if (!roi || roi.width < 40 || roi.height < 16) {
        res.status(400).json({
          error: "Invalid or missing Kill-Feed Region (must be at least 40x16px in native video coordinates).",
        });
        return;
      }

      const jobId = crypto.randomUUID();
      const jobDir = path.join(JOBS_DIR, jobId);
      const outputBaseDir = path.join(jobDir, "output");
      fs.mkdirSync(outputBaseDir, { recursive: true });

      const configFile = path.join(jobDir, "config.json");
      const statusFile = path.join(jobDir, "status.json");
      fs.writeFileSync(configFile, JSON.stringify(config || {}, null, 2), "utf-8");
      fs.writeFileSync(
        statusFile,
        JSON.stringify({ state: "running", progress: 0, message: "Starting pipeline...", confirmed_count: 0 }),
        "utf-8"
      );

      const job: JobRecord = { jobId, videoId, statusFile, configFile, outputBaseDir };
      jobRegistry.set(jobId, job);

      const roiArg = `${Math.round(roi.x)},${Math.round(roi.y)},${Math.round(roi.width)},${Math.round(roi.height)}`;
      spawnJobProcess(
        [
          "run-job",
          "--video",
          video.filePath,
          "--roi",
          roiArg,
          "--config",
          configFile,
          "--out-base",
          outputBaseDir,
          "--status-file",
          statusFile,
        ],
        statusFile
      );

      res.json({ jobId });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : "Failed to start processing job." });
    }
  });

  function resolveJobStatusFile(rawJobId: string): string | null {
    const safeId = sanitizeBasename(rawJobId);
    if (!safeId || safeId !== rawJobId) {
      return null;
    }
    const memJob = jobRegistry.get(safeId);
    if (memJob && fs.existsSync(memJob.statusFile)) {
      return memJob.statusFile;
    }
    const diskStatusFile = path.resolve(JOBS_DIR, safeId, "status.json");
    if (diskStatusFile.startsWith(path.resolve(JOBS_DIR) + path.sep) && fs.existsSync(diskStatusFile)) {
      return diskStatusFile;
    }
    return null;
  }

  app.get("/api/jobs/:jobId", (req: Request, res: Response) => {
    res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");
    const statusFile = resolveJobStatusFile(String(req.params.jobId || ""));
    if (!statusFile) {
      res.status(404).json({ error: "Processing job not found." });
      return;
    }
    try {
      const raw = fs.readFileSync(statusFile, "utf-8");
      res.json(JSON.parse(raw));
    } catch {
      res.json({ state: "running", progress: 0, message: "Updating status...", confirmed_count: 0 });
    }
  });

  app.get("/api/jobs/:jobId/files/:folder/:filename", (req: Request, res: Response) => {
    const statusFile = resolveJobStatusFile(String(req.params.jobId || ""));
    if (!statusFile) {
      res.status(404).json({ error: "Job not found." });
      return;
    }
    const status = JSON.parse(fs.readFileSync(statusFile, "utf-8")) as { session_dir?: string };
    if (!status.session_dir || !fs.existsSync(status.session_dir)) {
      res.status(404).json({ error: "Session output directory not available." });
      return;
    }

    const folder = String(req.params.folder || "");
    const filename = sanitizeBasename(String(req.params.filename || ""));
    let subDir: string;
    if (folder === "screenshots") {
      subDir = path.join(status.session_dir, "screenshots");
    } else if (folder === "clips") {
      subDir = path.join(status.session_dir, "clips");
    } else if (folder === "debug_roi") {
      subDir = path.join(status.session_dir, "debug", "detected_region");
    } else if (folder === "root") {
      subDir = status.session_dir;
    } else {
      res.status(400).json({ error: "Invalid folder." });
      return;
    }

    const targetPath = path.resolve(subDir, filename);
    if (!targetPath.startsWith(path.resolve(subDir) + path.sep) || !fs.existsSync(targetPath)) {
      res.status(404).json({ error: "File not found." });
      return;
    }
    res.sendFile(targetPath);
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(WORKSPACE_ROOT, "dist");
    app.use(express.static(distDir));
    app.get("*", (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`PEACE BGMI Server running on http://0.0.0.0:${PORT}`);
    void ensurePythonEnvironment().catch((err) => {
      console.error("Background Python bootstrap warning:", err instanceof Error ? err.message : err);
    });
  });
}

createServer();
