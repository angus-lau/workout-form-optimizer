
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.features.angle_utils as angle_utils
from src.features.classifier import FormClassifier

def test_pose_pipeline():

    classifier = FormClassifier()
    
    print("Running Test 1: Standing Straight")
    # All joints in a straight vertical line along the Y-axis
    standing_joints = {
        "LEFT_SHOULDER": (0.0, 2.0),
        "RIGHT_SHOULDER": (0.0, 2.0),
        "LEFT_HIP": (0.0, 1.0),
        "RIGHT_HIP": (0.0, 1.0),
        "LEFT_KNEE": (0.0, 0.0),
        "RIGHT_KNEE": (0.0, 0.0),
        "LEFT_ANKLE": (0.0, -1.0),
        "RIGHT_ANKLE": (0.0, -1.0),
    }
    
    standing_angles = {
        "left_knee": angle_utils.compute_left_knee_angle(standing_joints),
        "right_knee": angle_utils.compute_right_knee_angle(standing_joints),
        "back": angle_utils.compute_back_angle(standing_joints)
    }
    
    print(f"Calculated Angles: {standing_angles}")
    result1 = classifier.predict("squat", standing_angles)
    print(f"Classification: {result1}\n")


    print("Running Test 2: Bad Squat (Knee Angle Too Acute)")
    # Knee is at origin (0,0). Ankle is along X-axis (1,0). Hip is at (1,1).
    # This creates a 45-degree angle at the knee, which is below the 60-degree minimum.
    bad_squat_joints = {
        "LEFT_SHOULDER": (1.0, 2.0),
        "RIGHT_SHOULDER": (1.0, 2.0),
        "LEFT_HIP": (1.0, 1.0),
        "RIGHT_HIP": (1.0, 1.0),
        "LEFT_KNEE": (0.0, 0.0),
        "RIGHT_KNEE": (0.0, 0.0),
        "LEFT_ANKLE": (1.0, 0.0),
        "RIGHT_ANKLE": (1.0, 0.0),
    }
    
    bad_squat_angles = {
        "left_knee": angle_utils.compute_left_knee_angle(bad_squat_joints),
        "right_knee": angle_utils.compute_right_knee_angle(bad_squat_joints),
        "back": angle_utils.compute_back_angle(bad_squat_joints)
    }
    
    print(f"Calculated Angles: {bad_squat_angles}")
    result2 = classifier.predict("squat", bad_squat_angles)
    print(f"Classification: {result2}")

if __name__ == "__main__":
    test_pose_pipeline()