"""
MediaPipe Pose backend adapter.

Provides a thin wrapper so PoseEstimator can swap backends without changing callers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import mediapipe as mp


JointMap = Dict[str, Tuple[float, float]]


class MediaPipeBackend:
    """
    MediaPipe Pose backend adapter.

    model_complexity:
        0 = lite
        1 = full (default)
        2 = heavy
    """

    def __init__(
        self,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
    ) -> None:

        if model_complexity not in (0, 1, 2):
            raise ValueError(
                "model_complexity must be 0 (lite), 1 (full), or 2 (heavy)"
            )

        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation

        self.mp_pose = mp.solutions.pose
        self.model = None


    def load(self) -> None:
        """Load MediaPipe Pose model."""
        self.model = self.mp_pose.Pose(
            model_complexity=self.model_complexity,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=0.5,
        )

    def predict_frame(self, frame: np.ndarray) -> Dict[str, JointMap]:
        """
        Run pose on a single frame and return joints + visibility.
        """
        if self.model is None:
            self.load()

        # MediaPipe expects RGB
        rgb_frame = frame[:, :, ::-1]

        results = self.model.process(rgb_frame)

        joints: JointMap = {}
        visibility: Dict[str, float] = {}

        if results.pose_landmarks:
            h, w, _ = frame.shape
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                name = self.mp_pose.PoseLandmark(idx).name
                joints[name] = (landmark.x * w, landmark.y * h)
                visibility[name] = landmark.visibility

        return {
            "joints": joints,
            "visibility": visibility,
        }

    def predict_batch(self, frames: list[np.ndarray]) -> list[Dict[str, JointMap]]:
        """Run pose on a list of frames."""
        return [self.predict_frame(frame) for frame in frames]
