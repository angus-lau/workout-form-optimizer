"""
MediaPipe Pose backend adapter.

Provides a thin wrapper so PoseEstimator can swap backends without changing callers.
Uses MediaPipe Tasks API (0.10.30+).
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
<<<<<<< HEAD
import mediapipe as mp

=======
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
>>>>>>> bd5c84d (Woking camera and video input with mediapipe 10.31)

JointMap = Dict[str, Tuple[float, float]]

# MediaPipe landmark indices (33 total landmarks)
#See https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md Fig.4 for details.

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

# Model URLs for pose landmarker (indexed by complexity: 0=lite, 1=full, 2=heavy)
MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def _download_model(model_path: str, model_url: str) -> None:
    """Download the pose landmarker model if it doesn't exist."""
    if not os.path.exists(model_path):
        print(f"Downloading pose landmarker model to {model_path}...")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}") from e


class MediaPipeBackend:
    """
<<<<<<< HEAD
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
=======
    MediaPipe Pose backend implementation using Tasks API.
    Falls back to stub mode if MediaPipe fails to load.
    """

    def __init__(self, model_complexity: int = 1, enable_segmentation: bool = False) -> None:
        """
        Initialize MediaPipe backend.
        
        Args:
            model_complexity: Model complexity level (0=lite, 1=full, 2=heavy)
                - 0: Lite model (fastest, less accurate)
                - 1: Full model (balanced, default)
                - 2: Heavy model (slowest, most accurate)
            enable_segmentation: Whether to enable body segmentation masks
        """
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.landmarker: Optional[vision.PoseLandmarker] = None
        self.last_result: Optional[vision.PoseLandmarkerResult] = None
        self._use_stub = False
        self._load_attempted = False
>>>>>>> bd5c84d (Woking camera and video input with mediapipe 10.31)


    def load(self) -> None:
        """Load MediaPipe Pose model."""
<<<<<<< HEAD
        self.model = self.mp_pose.Pose(
            model_complexity=self.model_complexity,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=0.5,
        )
=======
        model_dir = os.path.join(os.path.expanduser("~"), ".mediapipe_models")
        
        # Select model URL based on complexity setting (clamp to valid range 0-2)
        complexity = max(0, min(2, self.model_complexity))
        model_url = MODEL_URLS.get(complexity, MODEL_URLS[1])
        
        try:
            # Download model if needed
            model_name = os.path.basename(model_url)
            model_path = os.path.join(model_dir, model_name)
            _download_model(model_path, model_url)
            
            # Create landmarker options
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_segmentation_masks=self.enable_segmentation
            )
            
            # Create the landmarker
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            complexity_names = {0: "lite", 1: "full", 2: "heavy"}
            print(f"MediaPipe pose model loaded successfully! (complexity: {complexity_names[complexity]})")
            
        except Exception as e:
            # Fall back to stub mode if loading fails
            print(f"WARNING: MediaPipe failed to load ({e}). Using stub mode.")
            print("For real pose detection, install Visual C++ Redistributables:")
            print("https://aka.ms/vs/17/release/vc_redist.x64.exe")
            self._use_stub = True
>>>>>>> bd5c84d (Woking camera and video input with mediapipe 10.31)

    def predict_frame(self, frame: np.ndarray) -> Dict[str, JointMap]:
        """
        Run pose on a single frame and return joints + visibility.
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary with 'joints' and 'visibility' keys.
            Joints are in normalized coordinates (0-1 range).
        """
<<<<<<< HEAD
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
=======
        if not self._load_attempted:
            try:
                self.load()
            except RuntimeError:
                # Already handled in load(), stub mode activated
                pass
            self._load_attempted = True
        
        # Use stub mode if MediaPipe failed
        if self._use_stub:
            return self._predict_frame_stub(frame)
        
        # Normal MediaPipe detection
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose
        detection_result = self.landmarker.detect(mp_image)
        
        joints: Dict[str, Tuple[float, float]] = {}
        visibility: Dict[str, float] = {}
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            # Get first detected pose
            pose_landmarks = detection_result.pose_landmarks[0]
            
            # Extract landmarks
            for landmark_idx, landmark_name in LANDMARK_INDICES.items():
                if landmark_idx < len(pose_landmarks):
                    landmark = pose_landmarks[landmark_idx]
                    # MediaPipe returns normalized coordinates (0-1)
                    joints[landmark_name] = (landmark.x, landmark.y)
                    visibility[landmark_name] = landmark.visibility
        
        return {
            "joints": joints,
            "visibility": visibility,
        }
    
    def _predict_frame_stub(self, frame: np.ndarray) -> Dict[str, JointMap]:
        """Stub mode: provides basic pose estimation using simple CV."""
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2
        
        # Simple stub: place joints in a basic human pose shape
        # This is just for UI testing - not real detection
        joints = {
            "NOSE": (center_x / w, (center_y - h * 0.15) / h),
            "LEFT_SHOULDER": ((center_x - w * 0.1) / w, center_y / h),
            "RIGHT_SHOULDER": ((center_x + w * 0.1) / w, center_y / h),
            "LEFT_ELBOW": ((center_x - w * 0.15) / w, (center_y + h * 0.1) / h),
            "RIGHT_ELBOW": ((center_x + w * 0.15) / w, (center_y + h * 0.1) / h),
            "LEFT_HIP": ((center_x - w * 0.08) / w, (center_y + h * 0.2) / h),
            "RIGHT_HIP": ((center_x + w * 0.08) / w, (center_y + h * 0.2) / h),
            "LEFT_KNEE": ((center_x - w * 0.08) / w, (center_y + h * 0.35) / h),
            "RIGHT_KNEE": ((center_x + w * 0.08) / w, (center_y + h * 0.35) / h),
            "LEFT_ANKLE": ((center_x - w * 0.08) / w, (center_y + h * 0.5) / h),
            "RIGHT_ANKLE": ((center_x + w * 0.08) / w, (center_y + h * 0.5) / h),
        }
        
        visibility = {name: 0.8 for name in joints.keys()}
        
        return {
            "joints": joints,
            "visibility": visibility,
        }

    def predict_batch(self, frames: list[np.ndarray]) -> list[Dict[str, JointMap]]:
        """Run pose on a list of frames (iterative)."""
        return [self.predict_frame(f) for f in frames]



>>>>>>> bd5c84d (Woking camera and video input with mediapipe 10.31)
