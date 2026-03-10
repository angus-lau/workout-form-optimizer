"""
MediaPipe Pose backend adapter.

Provides a thin wrapper so PoseEstimator can swap backends without changing callers.
Uses MediaPipe Solutions API.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp

JointMap = Dict[str, Tuple[float, float]]

# MediaPipe landmark indices (33 total landmarks)
# See https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md (Fig. 4) for details.
LANDMARK_INDICES = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "MOUTH_LEFT",
    10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}


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

        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)

        Returns:
            Dictionary with 'joints' and 'visibility' keys.
            Joints are pixel coordinates (x, y).
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
                # Map index to a readable name if available; otherwise use index
                name = self.mp_pose.PoseLandmark(idx).name if idx in self.mp_pose.PoseLandmark._member_map_.values() else LANDMARK_INDICES.get(idx, str(idx))
                # Convert normalized to pixel coordinates
                joints[name] = (landmark.x * w, landmark.y * h)
                visibility[name] = landmark.visibility

        return {
            "joints": joints,
            "visibility": visibility,
        }

    def predict_batch(self, frames: list[np.ndarray]) -> list[Dict[str, JointMap]]:
        """Run pose on a list of frames."""
        return [self.predict_frame(frame) for frame in frames]