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
            "squat": {
                "knee": {"min": 60, "max": 180, "optimal": 70},
                "hip": {"min": 60, "max": 180, "optimal": 75},
                "spine": {"min": 100, "max": 180, "optimal": 115},
                "elbow": {"min": 45, "max": 90, "optimal": 65}
            },
            "benchpress": {
                "knee": {"min": 80, "max": 100, "optimal": 90},
                "hip": {"min": 100, "max": 140, "optimal": 120},
                "spine": {"min": 120, "max": 160, "optimal": 140},
                "elbow": {"min": 75, "max": 180, "optimal": 90}
            },
            "deadlift": {
                "knee": {"min": 110, "max": 180, "optimal": 115},
                "hip": {"min": 50, "max": 180, "optimal": 60},
                "spine": {"min": 75, "max": 180, "optimal": 85},
                "elbow": {"min": 170, "max": 180, "optimal": 180}
            }
        }
    
    def predict(self, exercise: str, current_angles: dict) -> str:
        """
        Simple rule-based prediction for form quality.
        Args:
            exercise: The name of the exercise
            current_angles: A dictionary of the calculated angles from angle_utils.py
            Expected keys like "left_knee", "right_hip", "spine".
            
        Returns:
            "good form" if angle is within the dict dfinition, "bad form" otherwise
        """
        exercise = exercise.lower()
        if exercise not in self.thresholds:
            return f"Error: {exercise} is not a recognized exercise."

        for joint_name, angle in current_angles.items():
            if angle is None:
                continue
                

            base_joint = joint_name.replace("left_", "").replace("right_", "")
            
            if base_joint in self.thresholds[exercise]:
                min_thresh = self.thresholds[exercise][base_joint]["min"]
                max_thresh = self.thresholds[exercise][base_joint]["max"]
                
                # Check if the angle is within the threshold definition 
                if angle < min_thresh or angle > max_thresh:
                    return f"bad form: {joint_name} angle ({angle:.1f}°) is out of bounds for {exercise}."
                    
        return "good form"
