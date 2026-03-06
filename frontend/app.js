const API_BASE = "http://localhost:8000";

const SKELETON_CONNECTIONS = [
  ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
  ["LEFT_SHOULDER", "LEFT_ELBOW"],
  ["LEFT_ELBOW", "LEFT_WRIST"],
  ["RIGHT_SHOULDER", "RIGHT_ELBOW"],
  ["RIGHT_ELBOW", "RIGHT_WRIST"],
  ["LEFT_SHOULDER", "LEFT_HIP"],
  ["RIGHT_SHOULDER", "RIGHT_HIP"],
  ["LEFT_HIP", "RIGHT_HIP"],
  ["LEFT_HIP", "LEFT_KNEE"],
  ["LEFT_KNEE", "LEFT_ANKLE"],
  ["RIGHT_HIP", "RIGHT_KNEE"],
  ["RIGHT_KNEE", "RIGHT_ANKLE"],
  ["LEFT_ANKLE", "LEFT_HEEL"],
  ["LEFT_HEEL", "LEFT_FOOT_INDEX"],
  ["RIGHT_ANKLE", "RIGHT_HEEL"],
  ["RIGHT_HEEL", "RIGHT_FOOT_INDEX"],
  ["NOSE", "LEFT_EYE"],
  ["NOSE", "RIGHT_EYE"],
  ["LEFT_EYE", "LEFT_EAR"],
  ["RIGHT_EYE", "RIGHT_EAR"],
];

let analysisData = null;
let selectedFile = null;
let currentAbortController = null;
let overlayRafId = null;

function setStatus(msg, isError = false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : " processing");
}

function showCancelButton(show) {
  const el = document.getElementById("cancel-area");
  if (show) el.classList.remove("hidden");
  else el.classList.add("hidden");
}

function showProgressArea(show) {
  const el = document.getElementById("progress-area");
  if (show) el.classList.remove("hidden");
  else el.classList.add("hidden");
}

function updateProgress(frame, total) {
  const fill = document.getElementById("progress-fill");
  const text = document.getElementById("progress-text");
  const pct = total > 0 ? Math.round((100 * (frame + 1)) / total) : 0;
  fill.style.width = `${Math.min(pct, 100)}%`;
  text.textContent = `Processing frame ${frame + 1} of ${total}`;
}

function setAngleDisplay(angles) {
  const el = document.getElementById("angle-display");
  if (!angles || Object.keys(angles).length === 0) {
    el.textContent = "—";
    return;
  }
  const parts = Object.entries(angles).map(([k, v]) => `${k.replace("_", " ")}: ${Math.round(v)}°`);
  el.textContent = parts.join("  ");
}

function drawOverlay(canvas, video, frameData) {
  const ctx = canvas.getContext("2d");
  const w = video.videoWidth;
  const h = video.videoHeight;
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0, 0, w, h);

  if (!frameData || !frameData.joints) return;

  const joints = frameData.joints;
  const angles = frameData.angles || {};

  const toPixel = (name) => {
    const j = joints[name];
    if (!j) return null;
    return { x: j[0] * w, y: j[1] * h };
  };

  ctx.strokeStyle = "rgb(255, 0, 0)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [a, b] of SKELETON_CONNECTIONS) {
    const pa = toPixel(a);
    const pb = toPixel(b);
    if (pa && pb) {
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
    }
  }
  ctx.stroke();

  ctx.fillStyle = "rgb(0, 255, 0)";
  for (const [name, coord] of Object.entries(joints)) {
    const px = coord[0] * w;
    const py = coord[1] * h;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  setAngleDisplay(angles);
}

function getFrameIndexForTime(video, totalFrames) {
  const duration = video.duration;
  if (duration > 0 && Number.isFinite(duration) && totalFrames > 0) {
    const t = Math.max(0, Math.min(video.currentTime, duration));
    const idx = Math.floor((t / duration) * totalFrames);
    return Math.min(idx, totalFrames - 1);
  }
  const fps = analysisData?.video_info?.fps || 24;
  return Math.min(Math.floor(video.currentTime * fps), totalFrames - 1);
}

function getFrameData(frameIndex) {
  if (!analysisData || !analysisData.frames) return null;
  const frame = analysisData.frames[frameIndex];
  return frame || analysisData.frames[analysisData.frames.length - 1];
}

function syncOverlay() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("overlay");
  if (!video || !canvas || !analysisData?.frames?.length) return;

  const totalFrames = analysisData.frames.length;
  const idx = getFrameIndexForTime(video, totalFrames);
  const frameData = getFrameData(idx);
  drawOverlay(canvas, video, frameData);
}

function startOverlayLoop() {
  function loop() {
    syncOverlay();
    overlayRafId = requestAnimationFrame(loop);
  }
  if (overlayRafId) cancelAnimationFrame(overlayRafId);
  overlayRafId = requestAnimationFrame(loop);
}

function playbackErrorMsg(path) {
  const cli = path ? `python main.py "${path}"` : "python main.py \"path/to/video\"";
  return `Video could not play in the browser (unsupported codec or format). Use the CLI: ${cli} — or re-encode to H.264 MP4.`;
}

function showPlayer(videoSrc, isUpload = false, videoPath = null) {
  const container = document.getElementById("player-container");
  const placeholder = document.getElementById("placeholder");
  const video = document.getElementById("video");

  container.classList.remove("hidden");
  placeholder.classList.add("hidden");
  video.src = videoSrc;
  video.load();

  const showPlaybackError = () => setStatus(playbackErrorMsg(videoPath), true);

  video.removeEventListener("error", video._playbackErrorHandler);
  video._playbackErrorHandler = showPlaybackError;
  video.addEventListener("error", video._playbackErrorHandler);

  const fallbackTimer = setTimeout(() => {
    if (video.readyState >= 2) return;
    showPlaybackError();
  }, 2500);
  video.addEventListener("loadeddata", () => clearTimeout(fallbackTimer), { once: true });
  video.addEventListener("canplay", () => clearTimeout(fallbackTimer), { once: true });

  startOverlayLoop();
  video.addEventListener("loadeddata", () => syncOverlay());
}

function folderLabel(folder) {
  return folder
    .split(/[-_\s]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

async function loadVideoList() {
  const listEl = document.getElementById("video-list");
  try {
    const res = await fetch(`${API_BASE}/api/videos`);
    const data = await res.json();
    if (!data.videos || data.videos.length === 0) {
      listEl.innerHTML = '<span class="loading">No videos found in data/ or dataset/</span>';
      return;
    }

    const byFolder = {};
    for (const v of data.videos) {
      const parts = v.path.replace(/\\/g, "/").split("/");
      const folder = parts.length > 1 ? parts[parts.length - 2] : "Other";
      if (!byFolder[folder]) byFolder[folder] = [];
      byFolder[folder].push(v);
    }

    const folders = Object.keys(byFolder).sort();
    listEl.innerHTML = folders
      .map(
        (folder, i) => `
        <div class="folder-section ${i > 0 ? "collapsed" : ""}" data-folder="${folder}">
          <div class="folder-header">${folderLabel(folder)}</div>
          <div class="folder-content">
            ${byFolder[folder]
              .map(
                (v) =>
                  `<div class="video-item" data-path="${v.path}">${v.name}</div>`
              )
              .join("")}
          </div>
        </div>
      `
      )
      .join("");

    listEl.querySelectorAll(".folder-header").forEach((h) => {
      h.addEventListener("click", () => {
        h.closest(".folder-section").classList.toggle("collapsed");
      });
    });

    listEl.querySelectorAll(".video-item").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        handleSelectExisting(el.dataset.path);
      });
    });
  } catch (err) {
    listEl.innerHTML = `<span class="loading">Failed to load videos. Is the API running?</span>`;
  }
}

async function processStream(res, { onProgress, onDone, onError }) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  const handleEvent = (data) => {
    if (data.type === "progress") onProgress(data.frame, data.total);
    else if (data.type === "done") {
      onDone(data.result);
      finished = true;
    } else if (data.type === "error") {
      onError(data.detail);
      finished = true;
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const event of events) {
        const m = event.match(/^data:\s*(.+)$/m);
        if (!m) continue;
        try {
          handleEvent(JSON.parse(m[1]));
          if (finished) return;
        } catch (_) {}
      }
    }
    if (buffer.trim()) {
      const m = buffer.match(/^data:\s*(.+)$/m);
      if (m) try {
        handleEvent(JSON.parse(m[1]));
      } catch (_) {}
    }
  } catch (err) {
    if (err.name === "AbortError") throw err;
    onError(err.message);
  }
}

async function handleSelectExisting(path) {
  document.querySelectorAll(".video-item").forEach((e) => e.classList.remove("selected"));
  document.querySelector(`[data-path="${path}"]`)?.classList.add("selected");
  selectedFile = null;

  currentAbortController = new AbortController();
  setStatus("Processing...");
  showCancelButton(true);
  showProgressArea(true);
  document.getElementById("progress-fill").style.width = "0%";

  try {
    const res = await fetch(
      `${API_BASE}/api/process-video-stream?path=${encodeURIComponent(path)}`,
      { method: "POST", signal: currentAbortController.signal }
    );
    if (!res.ok) throw new Error(res.statusText);

    await processStream(res, {
      onProgress: (frame, total) => updateProgress(frame, total),
      onDone: (result) => {
        analysisData = result;
        if (!analysisData.frames) return;
        setStatus(`Loaded ${analysisData.frames.length} frames`);
        showPlayer(
          `${API_BASE}/api/videos/file/${encodeURIComponent(path).replace(/%2F/g, "/")}`,
          false,
          path
        );
      },
      onError: (msg) => setStatus(msg || "Failed to process video", true),
    });
  } catch (err) {
    if (err.name === "AbortError") {
      setStatus("Cancelled");
    } else {
      setStatus(err.message || "Failed to process video", true);
    }
  } finally {
    showCancelButton(false);
    showProgressArea(false);
    currentAbortController = null;
  }
}

async function handleUpload(file) {
  if (!file) return;
  selectedFile = file;
  document.querySelectorAll(".video-item").forEach((e) => e.classList.remove("selected"));

  currentAbortController = new AbortController();
  setStatus("Processing...");
  showCancelButton(true);
  showProgressArea(true);
  document.getElementById("progress-fill").style.width = "0%";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/process-video-stream`, {
      method: "POST",
      body: form,
      signal: currentAbortController.signal,
    });
    if (!res.ok) throw new Error(res.statusText);

    await processStream(res, {
      onProgress: (frame, total) => updateProgress(frame, total),
      onDone: (result) => {
        analysisData = result;
        if (!analysisData.frames) return;
        setStatus(`Loaded ${analysisData.frames.length} frames`);
        showPlayer(URL.createObjectURL(file), true, null);
      },
      onError: (msg) => setStatus(msg || "Failed to process video", true),
    });
  } catch (err) {
    if (err.name === "AbortError") {
      setStatus("Cancelled");
    } else {
      setStatus(err.message || "Failed to process video", true);
    }
  } finally {
    showCancelButton(false);
    showProgressArea(false);
    currentAbortController = null;
  }
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`${tab.dataset.tab}-panel`).classList.add("active");
    });
  });
}

function initUpload() {
  const zone = document.getElementById("upload-zone");
  const input = document.getElementById("file-input");

  zone.addEventListener("click", (e) => {
    e.preventDefault();
    input.click();
  });
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });

  input.addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) handleUpload(f);
    e.target.value = "";
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const f = e.dataTransfer?.files?.[0];
    if (f && f.type.startsWith("video/")) handleUpload(f);
  });
}

function init() {
  initTabs();
  initUpload();
  loadVideoList();

  document.getElementById("cancel-btn").addEventListener("click", () => {
    if (currentAbortController) currentAbortController.abort();
  });
}

init();
