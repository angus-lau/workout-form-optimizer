"""
FastAPI application for Workout Form Optimizer.

This module provides REST endpoints for pose estimation, form analysis,
and workout feedback.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import List, Dict, Optional
import io

from src.api.schemas import (
    PoseData,
    FormAnalysisRequest,
    FormAnalysisResponse,
    WorkoutFeedback,
    JointAngles,
)
from src.features.pose_estimator import PoseEstimator
from src.features.angle_utils import (
    compute_angle,
    compute_knee_angle,
    compute_hip_angle,
    compute_back_angle,
)
from src.ui.overlays import Overlays

# Initialize FastAPI app
app = FastAPI(
    title="Workout Form Optimizer API",
    description="API for analyzing and optimizing workout form using pose estimation",
    version="1.0.0",
)

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


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global pose_estimator, overlays
    pose_estimator = PoseEstimator()
    pose_estimator.load_model()
    overlays = Overlays()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global pose_estimator
    if pose_estimator:
        pose_estimator.unload_model()


# ==================== Health Check ====================


@app.get("/health")
async def health_check():
    """Check API health and model status."""
    return {
        "status": "healthy",
        "model_loaded": pose_estimator.model_loaded if pose_estimator else False,
    }


# ==================== Pose Estimation Endpoints ====================


@app.post("/api/pose/estimate")
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


@app.post("/api/pose/batch")
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


@app.post("/api/form/analyze", response_model=FormAnalysisResponse)
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

        return FormAnalysisResponse(
            exercise_type=request.exercise_type,
            form_score=form_score,
            joint_angles=joint_angles,
            feedback=feedback,
            status="success",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/form/video")
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

            frame_analyses.append(
                {
                    "frame": frame_count,
                    "joint_angles": joint_angles,
                    "feedback": feedback,
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


def _calculate_joint_angles(pose: Dict) -> JointAngles:
    """Calculate all relevant joint angles from pose data."""
    return JointAngles(
        knee_angle=compute_knee_angle(pose),
        hip_angle=compute_hip_angle(pose),
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

    if joint_angles.knee_angle and joint_angles.knee_angle < 60:
        feedback.append(
            WorkoutFeedback(
                area="knees",
                severity="warning",
                message="Knee angle too acute. Ensure knees track over toes.",
            )
        )

    if joint_angles.hip_angle and joint_angles.hip_angle < 70:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="error",
                message="Hips dropping too low. Maintain a neutral spine.",
            )
        )

    return feedback


def _analyze_deadlift(joint_angles: JointAngles) -> List[WorkoutFeedback]:
    """Deadlift-specific form analysis."""
    feedback = []

    if joint_angles.knee_angle and joint_angles.knee_angle > 150:
        feedback.append(
            WorkoutFeedback(
                area="knees",
                severity="warning",
                message="Knees too extended. Keep slight knee bend.",
            )
        )

    if joint_angles.hip_angle and joint_angles.hip_angle > 120:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="error",
                message="Hips too high. Lower hips to proper starting position.",
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

    if joint_angles.hip_angle and joint_angles.hip_angle > 110:
        feedback.append(
            WorkoutFeedback(
                area="hips",
                severity="warning",
                message="Hip position unstable. Plant feet firmly.",
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
