from typing import Tuple, Dict, List
import numpy as np
import MediaPipe as mp

class PoseEstimator:
    """
    PoseEstimator class for estimating poses in images using MediaPipe. 
    
    This class defines the interface for loading a pose estimation model and predicting 
    poses for single frames or batches of frames.
    """
    
    def __init__(self):
        """
        Initialize the PoseEstimator instance. The actual model loading is not implemented in this method.
        
        Stub implementation.
        """
        self.model = None
        self.model_loaded = False
    
    def load_model(self):
        """
        Load the PoseEstimator model. This method will intialize and configure MediaPipe Pose model, and store it in
        the instanece variable 'self.model'.
        
        Stub implementation.
        """
        self.model_loaded = True
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """
        Predict the pose for a single numpy frame, or returns error if model is not loaded.
        
        A pose is represented by a dictionary with 4 keys (shoulder, hip, knee, ankle), 
        each with their own corresponding (x, y, z) coordinates ranging from 0 to 1 (inclusive).
        
        Parameters:
            frame: np.ndarray:
                A NumPy array representing a single image of a person for pose estimation.
        
        Returns:
            Dict[str, Tuple[float, float, float]]:
                A dictionary with 4 keys, each with a tuple of 3 floats that represent the predicted pose.
        """
        
        if not self.model_loaded:
            raise RuntimeError("PoseEstimator model not loaded. Call load_model() first.")
        
        pose = {
            "shoulder": (0.5, 0.5, 0.5),
            "hip": (0.5, 0.6, 0.5),
            "knee": (0.5, 0.7, 0.5),
            "ankle": (0.5, 0.8, 0.5)
        }
        
        return pose
    
    def predict_batch(self, batch: List[np.ndarray]) -> List[Dict[str, Tuple[float, float, float]]]:
        """
        Predict a pose for a batch of numpy frames.
        
        Calls predict_frame repeatedly for each frame in the batch.
        
        Parameters:
            batch: list[np.ndarray]:
                A list of NumPy frames, each representing a single image of a person for pose estimation.
            
        Returns: 
            list[Dict[str, Tuple[float, float, float]]]:
                A list of dictionaries, each with 4 keys and a tuple of 3 floats that 
                represent the predicted pose.
        """
        predictions = []
        
        for frame in batch:
            predictions.append(self.predict_frame(frame))
        
        return predictions