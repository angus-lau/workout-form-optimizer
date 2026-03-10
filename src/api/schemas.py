"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Tuple, List, Optional


class PoseData(BaseModel):
    """Pose joint coordinates."""

    shoulder: Tuple[float, float, float] = Field(
        ..., description="Shoulder joint (x, y, z) coordinates"
    )
    hip: Tuple[float, float, float] = Field(
        ..., description="Hip joint (x, y, z) coordinates"
    )
    knee: Tuple[float, float, float] = Field(
        ..., description="Knee joint (x, y, z) coordinates"
    )
    ankle: Tuple[float, float, float] = Field(
        ..., description="Ankle joint (x, y, z) coordinates"
    )
    elbow: Optional[Tuple[float, float, float]] = Field(
        None, description="Elbow joint (x, y, z) coordinates"
    )
    wrist: Optional[Tuple[float, float, float]] = Field(
        None, description="Wrist joint (x, y, z) coordinates"
    )


class JointAngles(BaseModel):
    """Joint angle measurements in degrees."""

    left_knee_angle: Optional[float] = Field(None, description="Left knee flexion angle")
    right_knee_angle: Optional[float] = Field(None, description="Right knee flexion angle")
    left_hip_angle: Optional[float] = Field(None, description="Left hip flexion angle")
    right_hip_angle: Optional[float] = Field(None, description="Right hip flexion angle")
    back_angle: Optional[float] = Field(None, description="Back/spinal alignment angle")


class WorkoutFeedback(BaseModel):
    """Feedback item for form correction."""

    area: str = Field(..., description="Body area (knees, hips, shoulders, etc.)")
    severity: str = Field(
        ..., description="Severity level (info, warning, error)", enum=["info", "warning", "error"]
    )
    message: str = Field(..., description="Feedback message")


class FormAnalysisRequest(BaseModel):
    """Request for form analysis."""

    pose: Dict = Field(..., description="Pose data with joint coordinates")
    exercise_type: str = Field(
        ..., description="Type of exercise", enum=["squat", "deadlift", "benchpress"]
    )


class FormAnalysisResponse(BaseModel):
    """Response for form analysis."""

    exercise_type: str = Field(..., description="Type of exercise analyzed")
    form_score: float = Field(..., description="Overall form score (0-100)")
    joint_angles: JointAngles = Field(..., description="Calculated joint angles")
    feedback: List[WorkoutFeedback] = Field(..., description="Form feedback items")
    status: str = Field(..., description="Analysis status")
    analysis_id: Optional[str] = Field(None, description="ID for retrieving this analysis later")
    quality: Optional[str] = Field(
        None,
        description="Classifier assessment of form quality (e.g. good/bad)",
    )


class Exercise(BaseModel):
    """Exercise metadata."""

    id: str = Field(..., description="Exercise identifier (squat, deadlift, benchpress)")
    name: str = Field(..., description="Exercise display name")
    description: str = Field(..., description="Exercise description")
    key_areas: List[str] = Field(..., description="Key body areas for this exercise")


class AnalysisResult(BaseModel):
    """Stored analysis result."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    exercise_type: str = Field(..., description="Type of exercise analyzed")
    form_score: float = Field(..., description="Overall form score (0-100)")
    joint_angles: JointAngles = Field(..., description="Calculated joint angles")
    feedback: List[WorkoutFeedback] = Field(..., description="Form feedback items")
    timestamp: str = Field(..., description="ISO format timestamp of analysis")
    status: str = Field(..., description="Analysis status")
    quality: Optional[str] = Field(
        None,
        description="Classifier assessment of form quality (e.g. good/bad)",
    )


# ------------------ video schemas ------------------


class VideoMetadata(BaseModel):
    """Information about a video file available to the API."""

    path: str = Field(..., description="Relative path to the video file")
    name: str = Field(..., description="Filename")


class VideoListResponse(BaseModel):
    """Response returned by the `/api/videos` endpoint."""

    videos: List[VideoMetadata]


class FrameData(BaseModel):
    """Analysis results for a single video frame."""

    frame_index: int = Field(..., description="Zero‑based frame number")
    joints: Dict[str, List[float]] = Field(..., description="Normalized joint coordinates")
    angles: Dict[str, float] = Field(..., description="Computed joint angles")
    quality: Optional[str] = Field(
        None,
        description="Classifier assessment of form quality for this frame",
    )


class VideoInfo(BaseModel):
    """Metadata about the video stream itself."""

    width: int
    height: int
    fps: int
    total_frames: int


class VideoAnalysisResult(BaseModel):
    """Return value of a video processing request."""

    frames: List[FrameData]
    video_info: VideoInfo

