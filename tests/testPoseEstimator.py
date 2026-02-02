import os
import cv2
import pytest

from src.features.pose_estimator import PoseEstimator  # adjust import if needed

TEST_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(TEST_DIR, "testImages")

GOOD_IMG = os.path.join(IMG_DIR, "goodFormDeadlift.png")
BAD_IMG = os.path.join(IMG_DIR, "badFormDeadlift.png")

def run_pose(image_path):
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    frame = cv2.imread(image_path)
    assert frame is not None, f"Failed to load image: {image_path}"

    h, w, _ = frame.shape

    pe = PoseEstimator()
    result = pe.predict_frame(frame)

    assert "joints" in result
    assert "visibility" in result

    joints = result["joints"]
    visibility = result["visibility"]

    assert len(joints) > 0, "No joints detected"

    for name, (x, y) in joints.items():
        assert 0 <= x <= w
        assert 0 <= y <= h
        assert 0.0 <= visibility[name] <= 1.0

    return joints, visibility

@pytest.mark.parametrize("img_path", [GOOD_IMG, BAD_IMG])
def test_single_image_pose(img_path):
    joints, visibility = run_pose(img_path)

    print(f"\n--- {os.path.basename(img_path)} ---")
    for k in sorted(joints):
        x, y = joints[k]
        print(f"{k}: ({x:.1f}, {y:.1f}) vis={visibility[k]:.2f}")