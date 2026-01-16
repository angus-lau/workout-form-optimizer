from typing import Tuple, Dict, List
import numpy as np
import mediapipe as mp
from features.backends.mediapipe_backend import MediaPipeBackend

class PoseEstimator:
    """
    PoseEstimator class for estimating poses in images using MediaPipe. 
    
    This class defines the interface for loading a pose estimation model and predicting 
    poses for single frames or batches of frames.
    """
    
    def __init__(self):
        """
        Initialize the PoseEstimator instance. 
        """
        self.backend = None
        self.backend_loaded = False
    
    def load_model(self):
        """
        Load and initialize the MediaPipe model into the variable 'self.backend' for use by this PoseEstimator instance. 
        """
        if self.backend_loaded:
            return
        
        self.backend = MediaPipeBackend()
        self.backend.load()  
        self.backend_loaded = True
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Dict]:
        """
        Predict the pose for a single numpy frame.
        
        A pose is represented by a dictionary of 2 nested dictionaries: 'joints' and 'visibility'. 
        
        Both sub-dictionary contains 6 keys:
        - 'LEFT_SHOULDER'
        - 'RIGHT_SHOULDER'
        - 'LEFT_HIP'
        - 'RIGHT_HIP'
        - 'LEFT_ANKLE'
        - 'RIGHT_ANKLE'
        
        'joints' contains the pixel coordinates of the corresponding joints, and 'visibility' 
        contains the visibility [0.0-1.0] of the corresponding joints.
        
        Parameters:
            frame: np.ndarray:
                A NumPy array representing a single image of a person for pose estimation.
        
        Returns:
            Dict[str, Dict]:
                A dictionary with two nested dictionaries, each with 6 keys. each with a tuple of 3 floats that represent the predicted pose.
        """
        if not self.backend_loaded:
            self.backend.load()
            self.backend_loaded = True
            
        pose = self.backend.predict_frame(frame)
        
        return pose
        
        
        
        # pose = {
        #     "shoulder": (0.5, 0.5, 0.5),
        #     "hip": (0.5, 0.6, 0.5),
        #     "knee": (0.5, 0.7, 0.5),
        #     "ankle": (0.5, 0.8, 0.5)
        # }
        
        # return pose
    
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
        if not self.backend_loaded:
            self.backend.load()
            self.backend_loaded = True
             
        predictions = []
        
        for frame in batch:
            predictions.append(self.backend.predict_frame(frame))
        
        return predictions