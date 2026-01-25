"""
Form classifier for exercise technique evaluation.
Contains the class FormClassifier with method predict()
"""

class FormClassifier:
    """
    Rule-based classifier for evaluating exercise form based on joint angles.
    """
    
    def __init__(self):
        """Initialize the classifier with angle thresholds.
        Need to define all angle thresholds, based on angle_utils functions."""
        self.thresholds = {
            "squat_knee_min": 90, #temporary, need to talk to Jedd about defining
            "squat_knee_max": 90,
            "deadlift_back_min": 90
        }
    
    def predict(self, knee_angle: float) -> str:
        """
        Simple rule-based prediction for form quality. Sample code to start.
            
        Returns:
            "good form" if knee angle >= 90°, "bad form" otherwise
        """
        if knee_angle < self.thresholds["squat_knee_min"] or knee_angle > self.thresholds["squat_knee_max"]:
            return "bad form"
        else:
            return "good form"
