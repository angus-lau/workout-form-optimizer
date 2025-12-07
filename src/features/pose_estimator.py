import numpy as np

class PoseEstimator:
    """
    Stub PoseEstimator class
    
    This class defines the PoseEstimator interface for loading a model, predicting a pose for a single frame, and predicting poses for a batch of frames. 
    """
    
    def __init__(self):
        """
        
        """
        pass
    
    def load_model(self):
        """
        Load the PoseEstimator model.
        
        Parameters:
        """
        pass
    
    def predict_frame(self, frame: np.ndarray):
        """
        Predict the pose for a single numpy frame. 
        
        Parameters:
        frame: np.ndarray
        
        Returns: 
            a pose prediction for a single numpy frame. 
        """
        pass
    
    def predict_batch(self, batch: list[np.ndarray]):
        """
        Predict poses for a batch of numpy frames.
        
        Parameters:
            batch: a list of multiple numpy frames
            
        Returns: 
            predictions for each numpy frame in the batch.
        """
        pass