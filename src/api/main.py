"""
FastAPI application for Workout Form Optimizer.

This module provides REST endpoints for pose estimation, form analysis,
and workout feedback.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import List, Dict, Optional
import io
from datetime import datetime
import uuid
import asyncio
import json
import os
import queue
import tempfile
import threading
from pathlib import Path

from src.api import config

from src.api.schemas import (
    PoseData,
    FormAnalysisRequest,
    FormAnalysisResponse,
    WorkoutFeedback,
    JointAngles,
    Exercise,
    AnalysisResult,
    VideoMetadata,
    VideoListResponse,
    VideoAnalysisResult
)
from src.ui.overlays import Overlays

# feature utilities
from src.features.pose_estimator import PoseEstimator
from src.features.angle_utils import (
    compute_back_angle,
    compute_right_knee_angle,
    compute_left_knee_angle,
    compute_right_hip_angle,
    compute_left_hip_angle
)
from src.features.classifier import FormClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Workout Form Optimizer API",
    description="API for analyzing and optimizing workout form using pose estimation",
    version="1.0.0",
)

# ---------------- video configuration ----------------

# roots under which videos are allowed to live; relative to PROJECT_ROOT
VIDEO_ROOTS = [config.DATA_DIR]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


def _list_videos() -> list[dict]:
    """List all video files under VIDEO_ROOTS, returning relative paths."""
    videos: list[dict] = []
    seen = set()
    for root in VIDEO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                rel = path.relative_to(config.PROJECT_ROOT)
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
    abs_path = (config.PROJECT_ROOT / rel).resolve()
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    if not str(abs_path).startswith(str(config.PROJECT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return abs_path


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pose_estimator = None
overlays = None
classifier: Optional[FormClassifier] = None

# In-memory storage for analysis results (for now - can migrate to database later)
analysis_storage: Dict[str, AnalysisResult] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global pose_estimator, overlays, classifier
    pose_estimator = PoseEstimator()
    pose_estimator.load_model()
    overlays = Overlays()
    # instantiate rule‑based classifier (thresholds can be tuned elsewhere)
    classifier = FormClassifier()

# ==================== Health Check ====================


@app.get("/health")
async def health_check():
    """Check API health and model status."""
    return {
        "status": "healthy",
    }


# ==================== Exercise Endpoints ====================


@app.get("/api/exercises", response_model=List[Exercise])
async def list_exercises() -> List[Exercise]:
    """
    Get the list of available exercises.

    Returns:
        List of exercises with metadata
    """
    return [
        Exercise(
            id="squat",
            name="Squat",
            description="Lower body compound exercise",
            key_areas=["knees", "hips", "back"]
        ),
        Exercise(
            id="deadlift",
            name="Deadlift",
            description="Full body compound exercise",
            key_areas=["knees", "hips", "back"]
        ),
        Exercise(
            id="benchpress",
            name="Bench Press",
            description="Upper body compound exercise",
            key_areas=["back", "shoulders", "chest"]
        ),
    ]


@app.get("/api/exercises/{exercise_id}", response_model=Exercise)
async def get_exercise(exercise_id: str) -> Exercise:
    """
    Get details for a specific exercise.

    Args:
        exercise_id: Exercise identifier (squat, deadlift, benchpress)

    Returns:
        Exercise metadata
    """
    exercises = {
        "squat": Exercise(
            id="squat",
            name="Squat",
            description="Lower body compound exercise targeting quads, glutes, and hamstrings",
            key_areas=["knees", "hips", "back"]
        ),
        "deadlift": Exercise(
            id="deadlift",
            name="Deadlift",
            description="Full body compound exercise targeting posterior chain",
            key_areas=["knees", "hips", "back"]
        ),
        "benchpress": Exercise(
            id="benchpress",
            name="Bench Press",
            description="Upper body compound exercise targeting chest, shoulders, and triceps",
            key_areas=["back", "shoulders", "chest"]
        ),
    }
    
    if exercise_id not in exercises:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    
    return exercises[exercise_id]


# ==================== Analysis Endpoints ====================


@app.get("/api/analyze/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(analysis_id: str) -> AnalysisResult:
    """
    Fetch the results of a past analysis.

    Args:
        analysis_id: Unique identifier of the analysis

    Returns:
        Analysis results with form scores and feedback
    """
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found")
    
    return analysis_storage[analysis_id]


# ==================== Pose Estimation Endpoints ====================


@app.post("/api/analyze/pose/estimate")
async def estimate_pose(file: UploadFile = File(...)) -> Dict:
    """
    Estimate pose from a single image.

    Args:
        file: Image file (JPEG, PNG, etc.)

    Returns:
        Dictionary containing pose landmarks with coordinates
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Estimate pose
        pose = pose_estimator.predict_frame(frame)

        return {"pose": pose, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/pose/batch")
async def estimate_pose_batch(files: List[UploadFile] = File(...)) -> Dict:
    """
    Estimate poses from multiple images.

    Args:
        files: List of image files

    Returns:
        List of pose estimations
    """
    try:
        poses = []
        errors = []

        for i, file in enumerate(files):
            try:
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    errors.append({"index": i, "error": "Invalid image format"})
                    continue

                pose = pose_estimator.predict_frame(frame)
                poses.append({"index": i, "pose": pose})

            except Exception as e:
                errors.append({"index": i, "error": str(e)})

        return {
            "poses": poses,
            "errors": errors,
            "status": "success" if not errors else "partial",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Form Analysis Endpoints ====================


@app.post("/api/analyze/form", response_model=FormAnalysisResponse)
async def analyze_form(request: FormAnalysisRequest) -> FormAnalysisResponse:
    """
    Analyze workout form based on pose data.

    Args:
        request: Form analysis request with pose data and exercise type

    Returns:
        Detailed form analysis with feedback
    """
    try:
        # Extract joint angles
        joint_angles = _calculate_joint_angles(request.pose)

        # Generate feedback based on exercise type
        feedback = _generate_feedback(
            exercise_type=request.exercise_type,
            pose=request.pose,
            joint_angles=joint_angles,
        )

        # Calculate overall form score
        form_score = _calculate_form_score(joint_angles, feedback)

        # Use classifier to annotate quality (if available)
        quality_label = None
        if classifier is not None and joint_angles.knee_angle is not None:
            quality_label = classifier.predict(joint_angles.knee_angle)

        # Store result and get analysis ID
        analysis_id = _store_analysis_result(
            request.exercise_type,
            form_score,
            joint_angles,
            feedback,
            quality=quality_label,
        )

        response = FormAnalysisResponse(
            exercise_type=request.exercise_type,
            form_score=form_score,
            joint_angles=joint_angles,
            feedback=feedback,
            status="success",
            analysis_id=analysis_id,
            quality=quality_label,
        )
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/form/video")
async def analyze_form_video(
    exercise_type: str, file: UploadFile = File(...)
) -> Dict:
    """
    Analyze form throughout a video.

    Args:
        exercise_type: Type of exercise (squat, deadlift, benchpress)
        file: Video file

    Returns:
        Frame-by-frame analysis with aggregated metrics
    """
    try:
        # Read video file
        contents = await file.read()
        video_bytes = io.BytesIO(contents)

        # Create temporary file for video processing
        with open("/tmp/temp_video.mp4", "wb") as temp_file:
            temp_file.write(contents)

        # Process video
        cap = cv2.VideoCapture("/tmp/temp_video.mp4")
        frame_analyses = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Estimate pose
            pose = pose_estimator.predict_frame(frame)

            # Analyze form
            joint_angles = _calculate_joint_angles(pose)
            feedback = _generate_feedback(
                exercise_type=exercise_type, pose=pose, joint_angles=joint_angles
            )

            # classifier quality per frame
            frame_quality = None
            if classifier is not None and joint_angles.knee_angle is not None:
                frame_quality = classifier.predict(joint_angles.knee_angle)

            frame_analyses.append(
                {
                    "frame": frame_count,
                    "joint_angles": joint_angles,
                    "feedback": feedback,
                    "quality": frame_quality,
                }
            )

            frame_count += 1

        cap.release()

        # Calculate aggregate metrics
        avg_form_score = (
            np.mean([_calculate_form_score(fa["joint_angles"], fa["feedback"]) 
                    for fa in frame_analyses])
            if frame_analyses
            else 0
        )

        return {
            "exercise_type": exercise_type,
            "total_frames": frame_count,
            "average_form_score": avg_form_score,
            "frame_analyses": frame_analyses,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Video Management Endpoints ====================

@app.get("/api/videos", response_model=VideoListResponse)
def list_videos() -> VideoListResponse:
    """Return list of all known video files."""
    return VideoListResponse(videos=[VideoMetadata(**v) for v in _list_videos()])


@app.get("/api/videos/file/{path:path}")
def serve_video(path: str):
    """Send a video file by its relative path."""
    try:
        abs_path = _resolve_video_path(path)
    except HTTPException:
        raise
    if not abs_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(abs_path, media_type="video/mp4")


@app.post("/api/process-video", response_model=VideoAnalysisResult)
async def process_video_endpoint(
    file: UploadFile = File(None),
    path: str | None = Query(None, description="Path to existing video (e.g. data/squat/video.mp4)"),
) -> VideoAnalysisResult:
    """Process a video and return pose analysis for every frame.

    Either upload a file or specify `path` to an existing dataset video.
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
            return VideoAnalysisResult(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)

    if path:
        abs_path = _resolve_video_path(path)
        try:
            result = await asyncio.to_thread(run_analyze, str(abs_path))
            return VideoAnalysisResult(**result)
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


# ==================== Exercise-Specific Endpoints ====================


@app.post("/api/exercises/squat")
async def analyze_squat(file: UploadFile = File(...)) -> FormAnalysisResponse:
    """Analyze squat form from an image."""
    request = FormAnalysisRequest(
        pose=pose_estimator.predict_frame(
            cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        ),
        exercise_type="squat",
    )
    return await analyze_form(request)


@app.post("/api/exercises/deadlift")
async def analyze_deadlift(file: UploadFile = File(...)) -> FormAnalysisResponse:
    """Analyze deadlift form from an image."""
    img_array = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR
    )
    pose = pose_estimator.predict_frame(img_array)
    request = FormAnalysisRequest(pose=pose, exercise_type="deadlift")
    return await analyze_form(request)


@app.post("/api/exercises/benchpress")
async def analyze_benchpress(file: UploadFile = File(...)) -> FormAnalysisResponse:
    """Analyze bench press form from an image."""
    img_array = cv2.imdecode(
        np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR
    )
    pose = pose_estimator.predict_frame(img_array)
    request = FormAnalysisRequest(pose=pose, exercise_type="benchpress")
    return await analyze_form(request)


# ==================== Helper Functions ====================


def _store_analysis_result(
    exercise_type: str,
    form_score: float,
    joint_angles: JointAngles,
    feedback: List[WorkoutFeedback],
    quality: Optional[str] = None,
) -> str:
    """
    Store analysis result and return its ID.
    
    Args:
        exercise_type: Type of exercise analyzed
        form_score: Overall form score
        joint_angles: Calculated joint angles
        feedback: Form feedback items
        quality: Optional classifier label describing form quality
    
    Returns:
        Analysis ID for retrieval
    """
    analysis_id = str(uuid.uuid4())
    analysis_result = AnalysisResult(
        analysis_id=analysis_id,
        exercise_type=exercise_type,
        form_score=form_score,
        joint_angles=joint_angles,
        feedback=feedback,
        timestamp=datetime.utcnow().isoformat(),
        status="success",
        quality=quality,
    )
    analysis_storage[analysis_id] = analysis_result
    return analysis_id


def _calculate_joint_angles(pose: Dict) -> JointAngles:
    """Calculate all relevant joint angles from pose data."""
    return JointAngles(
        left_knee_angle=compute_left_knee_angle(pose),
        right_knee_angle=compute_right_knee_angle(pose),
        left_hip_angle=compute_left_hip_angle(pose),
        right_hip_angle=compute_right_hip_angle(pose),
        back_angle=compute_back_angle(pose),
    )


def _generate_feedback(
    exercise_type: str, pose: Dict, joint_angles: JointAngles
) -> List[WorkoutFeedback]:
    """Generate feedback based on exercise type and joint angles."""
    feedback = []

    if exercise_type == "squat":
        feedback.extend(_analyze_squat(joint_angles))
    elif exercise_type == "deadlift":
        feedback.extend(_analyze_deadlift(joint_angles))
    elif exercise_type == "benchpress":
        feedback.extend(_analyze_benchpress(joint_angles))

    return feedback


def _analyze_squat(joint_angles: JointAngles) -> List[WorkoutFeedback]:
    """Squat-specific form analysis."""
    feedback = []

    if joint_angles.left_knee_angle and joint_angles.left_knee_angle < 60:
        feedback.append(
            WorkoutFeedback(
                area="knees",
                severity="warning",
                message="Left knee angle too acute. Ensure knees track over toes.",
            )
        )

    if joint_angles.right_knee_angle and joint_angles.right_knee_angle < 60:
        feedback.append(
            WorkoutFeedback(
                area="knees",
                severity="warning",
                message="Right knee angle too acute. Ensure knees track over toes.",
            )
        )

    if joint_angles.left_hip_angle and joint_angles.left_hip_angle < 70:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="error",
                message="Left hip dropping too low. Maintain a neutral spine.",
            )
        )
    
    if joint_angles.right_hip_angle and joint_angles.right_hip_angle < 70:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="error",
                message="Right hip dropping too low. Maintain a neutral spine.",
            )
        )

    return feedback


def _analyze_deadlift(joint_angles: JointAngles) -> List[WorkoutFeedback]:
    """Deadlift-specific form analysis."""
    feedback = []

    if joint_angles.left_knee_angle and joint_angles.left_knee_angle > 150:
        feedback.append(
            WorkoutFeedback(
                area="knees",
                severity="warning",
                message="Left knees too extended. Keep slight knee bend.",
            )
        )
    
    if joint_angles.right_knee_angle and joint_angles.right_knee_angle > 150:
        feedback.append(
            WorkoutFeedback(
                area="knees",
                severity="warning",
                message="Right knees too extended. Keep slight knee bend.",
            )
        )

    if joint_angles.left_hip_angle and joint_angles.left_hip_angle > 120:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="error",
                message="Left hips too high. Lower hips to proper starting position.",
            )
        )
    
    if joint_angles.right_hip_angle and joint_angles.right_hip_angle > 120:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="error",
                message="Right hips too high. Lower hips to proper starting position.",
            )
        )

    return feedback


def _analyze_benchpress(joint_angles: JointAngles) -> List[WorkoutFeedback]:
    """Bench press-specific form analysis."""
    feedback = []

    if joint_angles.back_angle and joint_angles.back_angle < 160:
        feedback.append(
            WorkoutFeedback(
                area="back",
                severity="warning",
                message="Maintain neutral spine. Excessive arching detected.",
            )
        )

    if joint_angles.left_hip_angle and joint_angles.left_hip_angle > 110:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="warning",
                message="Left hip position unstable. Plant feet firmly.",
            )
        )
    
    if joint_angles.right_hip_angle and joint_angles.right_hip_angle > 110:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="warning",
                message="Right hip position unstable. Plant feet firmly.",
            )
        )

    return feedback


def _calculate_form_score(
    joint_angles: JointAngles, feedback: List[WorkoutFeedback]
) -> float:
    """Calculate overall form score (0-100)."""
    base_score = 100.0

    # Deduct points for each feedback item
    for item in feedback:
        if item.severity == "error":
            base_score -= 20
        elif item.severity == "warning":
            base_score -= 10

    return max(0, min(100, base_score))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
