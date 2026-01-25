from typing import Tuple, Dict, List, Optional
import numpy as np
from .backends.mediapipe_backend import MediaPipeBackend

JointMap = Dict[str, Tuple[float, float]]    # Joint name to (x, y) coordinates
VisibilityMap = Dict[str, float]             # Joint name to visibility score
Pose = Dict[str, JointMap | VisibilityMap]   # Dictionary: {'joints': JointMap, 'visibility': VisibilityMap}

class PoseEstimator:
    """
    High-level pose estimation interface.
    
    This class provides a clean interface for pose estimation, abstracting away
    the backend implementation details. It uses MediaPipeBackend internally
    and transforms the output to a simplified format.
    """
    
    def __init__(self, model_complexity: int = 1, enable_segmentation: bool = False):
        """
        Initialize the PoseEstimator instance. 
        """
        self.backend = None
        self.backend_loaded = False
    
    def load_model(self) -> None:
        """
        Load and initialize the MediaPipe model into the variable 'self.backend' for use by this PoseEstimator instance. 
        """
        if self.backend_loaded:
            return
        
        self.backend = MediaPipeBackend()
        self.backend.load()  
        self.backend_loaded = True
    
    def predict_frame(self, frame: np.ndarray) -> Pose:
        """
        Predict the pose for a single numpy frame.
        
        A pose is represented by a dictionary with 2 keys ('joints' and 'visibility'). Each key mapping to another dictionary. 
        'joints' maps to the joint name and its pixel coordinates as a tuple of (x, y). 'visibility' maps to the 
        joint name and its visibility score as a float [0.0-1.0]. 
        
        Args:
            frame: A NumPy array representing a single image (BGR format from OpenCV)
        
        Returns:
            Pose:
                A dictionary with 2 keys ('joints' and 'visibility') mapping to nested dictionaries.
        """
        if not self.backend_loaded:
            self.load_model()
            
        return self.backend.predict_frame(frame)
    
    def predict_batch(self, batch: List[np.ndarray]) -> List[Pose]:
        """
        Given a batch of numpy frames, predict a pose for each frame in the batch by repeatedly calling predict_frame.
        
        Parameters:
            batch: list[np.ndarray]:
                A list of NumPy arrays, each representing a single image of a person for pose estimation.
            
        Returns: 
            list[Pose]:
                A list of dictionaries, each with 2 keys ('joints' and 'visibility') mapping to nested dictionaries.
        """
        if not self.backend_loaded:
            self.load_model()
             
        return self.backend.predict_batch(batch)
