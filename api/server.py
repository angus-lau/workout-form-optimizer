"""FastAPI server for workout form optimizer - video analysis API."""

import asyncio
import json
import os
import queue
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# Project root (api/ is inside project)
PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_ROOTS = ["data"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


app = FastAPI(title="Workout Form Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _list_videos() -> list[dict]:
    """List all video files under VIDEO_ROOTS, returning relative paths."""
    videos = []
    seen = set()
    for root_name in VIDEO_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                rel = path.relative_to(PROJECT_ROOT)
                rel_str = str(rel).replace("\\", "/")
                if rel_str not in seen:
                    seen.add(rel_str)
                    videos.append({"path": rel_str, "name": path.name})
    return sorted(videos, key=lambda v: v["path"])


def _resolve_video_path(rel_path: str) -> Path:
    """Resolve a relative path to an absolute path, ensuring it's under a video root."""
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid path")
    abs_path = (PROJECT_ROOT / rel).resolve()
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    if not str(abs_path).startswith(str(PROJECT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return abs_path


@app.get("/api/videos")
def list_videos():
    """List all videos in the data folder."""
    return {"videos": _list_videos()}


@app.get("/api/videos/file/{path:path}")
def serve_video(path: str):
    """Serve a video file by relative path (e.g. data/squat/video.mp4)."""
    try:
        abs_path = _resolve_video_path(path)
    except HTTPException:
        raise
    if not abs_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(abs_path, media_type="video/mp4")


@app.post("/api/process-video")
async def process_video(
    file: UploadFile = File(None),
    path: str | None = Query(None, description="Path to existing video (e.g. data/squat/video.mp4)"),
):
    """
    Process a video and return per-frame pose analysis.

    Either:
    - Upload a file (multipart form with 'file' field), or
    - Provide 'path' (query param) to an existing video in data/.
    """
    from src.ui.opencv_demo import analyze_video

    def run_analyze(video_path: str):
        return analyze_video(video_path)

    video_path = None

    if file and file.filename:
        suffix = Path(file.filename).suffix or ".mp4"
        content = await file.read()
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, content)
            os.close(fd)
            video_path = temp_path
            result = await asyncio.to_thread(run_analyze, video_path)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)

    if path:
        abs_path = _resolve_video_path(path)
        try:
            return await asyncio.to_thread(run_analyze, str(abs_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(
        status_code=400,
        detail="Provide either a file upload or a path to an existing video.",
    )


async def _stream_analyze(video_path: str, cleanup_path: str | None = None):
    """Stream progress events then final result as SSE."""
    from src.ui.opencv_demo import analyze_video

    q = queue.Queue()

    def progress_cb(frame: int, total: int):
        q.put(("progress", frame, total))

    def worker():
        try:
            result = analyze_video(video_path, progress_callback=progress_cb)
            q.put(("done", result))
        except Exception as e:
            q.put(("error", str(e)))

    thread = threading.Thread(target=worker)
    thread.start()

    try:
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if item[0] == "done":
                yield f"data: {json.dumps({'type': 'done', 'result': item[1]})}\n\n"
                break
            if item[0] == "error":
                yield f"data: {json.dumps({'type': 'error', 'detail': item[1]})}\n\n"
                break
            if item[0] == "progress":
                yield f"data: {json.dumps({'type': 'progress', 'frame': item[1], 'total': item[2]})}\n\n"
    finally:
        thread.join()
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass


@app.post("/api/process-video-stream")
async def process_video_stream(
    file: UploadFile = File(None),
    path: str | None = Query(None, description="Path to existing video"),
):
    """
    Process a video and stream progress via Server-Sent Events, then return the result.
    """
    video_path = None

    if file and file.filename:
        suffix = Path(file.filename).suffix or ".mp4"
        content = await file.read()
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, content)
            os.close(fd)
            video_path = temp_path
        except Exception:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            raise

    if path:
        abs_path = _resolve_video_path(path)
        video_path = str(abs_path)

    if not video_path:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file upload or a path to an existing video.",
        )

    cleanup = video_path if (file and file.filename) else None
    return StreamingResponse(
        _stream_analyze(video_path, cleanup_path=cleanup),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
