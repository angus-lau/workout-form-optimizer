from typing import Tuple, Dict, List, Optional
import numpy as np

from src.features.backends.mediapipe_backend import MediaPipeBackend


class PoseEstimator:
    """
    High-level pose estimation interface.
    
    This class provides a clean interface for pose estimation, abstracting away
    the backend implementation details. It uses MediaPipeBackend internally
    and transforms the output to a simplified format.
    """
    
    def __init__(self, model_complexity: int = 1, enable_segmentation: bool = False):
        """
        Initialize the PoseEstimator with a backend.
        
        Args:
            model_complexity: Model complexity level (0=lite, 1=full, 2=heavy)
            enable_segmentation: Whether to enable body segmentation masks
        """
        self.backend = MediaPipeBackend(model_complexity=model_complexity, enable_segmentation=enable_segmentation)
        self.model_loaded = False
    
    def load_model(self) -> None:
        """
        Load the pose estimation model.
        
        Initializes and configures the backend model.
        """
        self.backend.load()
        self.model_loaded = True
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """
        Predict the pose for a single numpy frame.
        
        A pose is represented by a dictionary with 4 keys (shoulder, hip, knee, ankle), 
        each with their own corresponding (x, y, z) coordinates ranging from 0 to 1 (inclusive).
        
        Args:
            frame: A NumPy array representing a single image (BGR format from OpenCV)
        
        Returns:
            Dictionary with 4 keys ('shoulder', 'hip', 'knee', 'ankle'), each with 
            a tuple of 3 floats (x, y, z) that represent the predicted pose in normalized coordinates.
            Returns empty dict if pose cannot be detected.
        """
        if not self.model_loaded:
            self.load_model()
        
        # Get pose detection from backend
        result = self.backend.predict_frame(frame)
        joints_dict = result.get("joints", {})
        
        # Convert MediaPipe joints to generic format (averaging left/right)
        # This is for compatibility with the expected return format
        if not joints_dict:
            return {}
        
        # Average left/right joints to create generic joints
        def get_avg_joint(left_key: str, right_key: str) -> Optional[Tuple[float, float, float]]:
            left = joints_dict.get(left_key)
            right = joints_dict.get(right_key)
            
            if left and right:
                x = (left[0] + right[0]) / 2.0
                y = (left[1] + right[1]) / 2.0
                return (x, y, 0.0)
            elif left:
                return (left[0], left[1], 0.0)
            elif right:
                return (right[0], right[1], 0.0)
            return None
        
        shoulder = get_avg_joint("LEFT_SHOULDER", "RIGHT_SHOULDER")
        hip = get_avg_joint("LEFT_HIP", "RIGHT_HIP")
        knee = get_avg_joint("LEFT_KNEE", "RIGHT_KNEE")
        ankle = get_avg_joint("LEFT_ANKLE", "RIGHT_ANKLE")
        
        if shoulder and hip and knee and ankle:
            return {
                "shoulder": shoulder,
                "hip": hip,
                "knee": knee,
                "ankle": ankle
            }
        
        # Return empty dict if no pose detected
        return {}
    
    def predict_batch(self, batch: List[np.ndarray]) -> List[Dict[str, Tuple[float, float, float]]]:
        """
        Predict poses for a batch of numpy frames.
        
        Calls predict_frame repeatedly for each frame in the batch.
        
        Args:
            batch: A list of NumPy frames, each representing a single image
        
        Returns: 
            A list of dictionaries, each with 4 keys and a tuple of 3 floats that 
            represent the predicted pose.
        """
        predictions = []
        
        for frame in batch:
            predictions.append(self.predict_frame(frame))
        
        return predictions
