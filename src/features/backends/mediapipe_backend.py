"""
MediaPipe Pose backend adapter (Tasks API, 0.10.30+).

Provides a thin wrapper so PoseEstimator can swap backends without changing callers.
Outputs pixel-space joints (x, y) and visibility per landmark.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

# Import Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

# Model URLs for pose landmarker (0=lite, 1=full, 2=heavy)
MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def _download_model(model_path: str, model_url: str) -> None:
    """Download the pose landmarker model if it doesn't exist."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)


class MediaPipeBackend:
    """
    MediaPipe Pose backend adapter using the Tasks API.

    model_complexity:
        0 = lite
        1 = full (default)
        2 = heavy
    """

    def __init__(
        self,
        model_complexity: int = 1,
        enable_segmentation: bool = False,  # Not all model variants output masks; kept for API symmetry
    ) -> None:
        if model_complexity not in (0, 1, 2):
            raise ValueError(
                "model_complexity must be 0 (lite), 1 (full), or 2 (heavy)"
            )

        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation

        self.landmarker: Optional[vision.PoseLandmarker] = None
        self._loaded = False

    def load(self) -> None:
        """Load MediaPipe Pose Landmarker (Tasks API)."""
        model_dir = os.path.join(os.path.expanduser("~"), ".mediapipe_models")
        complexity = max(0, min(2, self.model_complexity))
        model_url = MODEL_URLS.get(complexity, MODEL_URLS[1])

        model_name = os.path.basename(model_url)
        model_path = os.path.join(model_dir, model_name)
        _download_model(model_path, model_url)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_segmentation_masks=self.enable_segmentation,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._loaded = True

    def predict_frame(self, frame: np.ndarray) -> Dict[str, JointMap]:
        """
        Run pose on a single frame and return joints + visibility.

        Returns:
            {
                "joints": {LANDMARK_NAME: (x_px, y_px), ...},
                "visibility": {LANDMARK_NAME: float, ...}
            }
        """
        if not self._loaded:
            self.load()

        if frame is None or frame.size == 0:
            return {"joints": {}, "visibility": {}}

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect pose
        assert self.landmarker is not None
        result = self.landmarker.detect(mp_image)

        joints: Dict[str, Tuple[float, float]] = {}
        visibility: Dict[str, float] = {}

        h, w = frame.shape[:2]

        # result.pose_landmarks: List[List[NormalizedLandmark]]
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pose_landmarks = result.pose_landmarks[0]
            for idx, name in LANDMARK_INDICES.items():
                if idx < len(pose_landmarks):
                    lm = pose_landmarks[idx]
                    # Convert from normalized [0..1] to pixel coordinates
                    x_px = float(lm.x) * w
                    y_px = float(lm.y) * h
                    joints[name] = (x_px, y_px)

                    # Visibility may be absent; default to 1.0
                    vis = getattr(lm, "visibility", None)
                    visibility[name] = float(vis) if vis is not None else 1.0

        return {"joints": joints, "visibility": visibility}

    def predict_batch(self, frames: list[np.ndarray]) -> list[Dict[str, JointMap]]:
        """Run pose on a list of frames."""
        return [self.predict_frame(frame) for frame in frames]