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

    knee_angle: Optional[float] = Field(None, description="Knee flexion angle")
    hip_angle: Optional[float] = Field(None, description="Hip flexion angle")
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
