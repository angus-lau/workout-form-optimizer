"""
MediaPipe Pose backend adapter stub.

Provides a thin wrapper so PoseEstimator can swap backends without changing callers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

JointMap = Dict[str, Tuple[float, float]]


class MediaPipeBackend:
    """
    Stub for MediaPipe Pose. Replace with real mediapipe imports and inference.
    """

    def __init__(self, model_complexity: int = 1, enable_segmentation: bool = False) -> None:
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.model = None

    def load(self) -> None:
        """Load MediaPipe Pose model (placeholder)."""
        # TODO: import mediapipe and initialize mp.solutions.pose.Pose(...)
        self.model = "mediapipe_pose_loaded"

    def predict_frame(self, frame: np.ndarray) -> Dict[str, JointMap]:
        """
        Run pose on a single frame and return joints + visibility.
        """
        if self.model is None:
            self.load()
        h, w, _ = frame.shape
        center = (w * 0.5, h * 0.5)
        # TODO: replace with real outputs from mediapipe results.pose_landmarks
        return {
            "joints": {
                "LEFT_SHOULDER": center,
                "RIGHT_SHOULDER": (center[0] + 20, center[1]),
            },
            "visibility": {"LEFT_SHOULDER": 0.9, "RIGHT_SHOULDER": 0.9},
        }

    def predict_batch(self, frames: list[np.ndarray]) -> list[Dict[str, JointMap]]:
        """Run pose on a list of frames (iterative stub)."""
        return [self.predict_frame(f) for f in frames]



