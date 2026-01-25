import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2 as cv
import numpy as np
from typing import Dict, Tuple, Optional

from src.features.backends.mediapipe_backend import MediaPipeBackend
from src.features.angle_utils import (
    compute_left_knee_angle, compute_right_knee_angle,
    compute_left_hip_angle, compute_right_hip_angle,
    compute_back_angle
)
from src.ui.overlays import Overlays


def _convert_joints_to_pixels(
    joints_dict: Dict[str, Tuple[float, float]],
    visibility_dict: Dict[str, float],
    frame_width: int,
    frame_height: int
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[float, float]]]:
    """
    Convert MediaPipe joints to pixel coordinates for drawing and normalized coordinates for angle calculation.
    
    Args:
        joints_dict: Dictionary with MediaPipe joint names and coordinates
        visibility_dict: Dictionary with joint visibility scores
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
        
    Returns:
        Tuple of (joints_pixels, joints_normalized) dictionaries
    """
    joints_pixels: Dict[str, Tuple[int, int]] = {}
    joints_normalized: Dict[str, Tuple[float, float]] = {}
    
    for joint_name, coords in joints_dict.items():
        if not isinstance(coords, (tuple, list)) or len(coords) < 2:
            continue
        
        x_raw, y_raw = float(coords[0]), float(coords[1])
        
        # Check visibility - skip joints with very low visibility
        visibility = visibility_dict.get(joint_name, 1.0)
        if visibility < 0.1:  # Threshold for visibility
            continue
        
        # Validate coordinates are not (0,0) or invalid
        # MediaPipe returns normalized coords (0-1), so (0,0) is suspicious unless it's actually at top-left
        # Better check: ensure coords are within reasonable bounds
        if x_raw < 0 or x_raw > 1 or y_raw < 0 or y_raw > 1:
            continue
        
        # Skip if coordinates are exactly (0,0) - likely undetected
        # (Allow if visibility is high, but we already checked that)
        if abs(x_raw) < 0.001 and abs(y_raw) < 0.001:
            continue
        
        # Determine if coordinates are normalized (0-1) or pixel coordinates
        if 0 <= x_raw <= 1 and 0 <= y_raw <= 1:
            # Normalized coordinates - convert to pixels for drawing
            x_pixel = int(x_raw * frame_width)
            y_pixel = int(y_raw * frame_height)
            
            # Validate pixel coordinates are within frame bounds
            if 0 <= x_pixel < frame_width and 0 <= y_pixel < frame_height:
                joints_pixels[joint_name] = (x_pixel, y_pixel)
                joints_normalized[joint_name] = (x_raw, y_raw)
        else:
            # Pixel coordinates - validate and convert to normalized
            x_pixel = int(x_raw)
            y_pixel = int(y_raw)
            
            # Validate pixel coordinates are within frame bounds
            if 0 <= x_pixel < frame_width and 0 <= y_pixel < frame_height:
                joints_pixels[joint_name] = (x_pixel, y_pixel)
                x_norm = x_raw / frame_width
                y_norm = y_raw / frame_height
                joints_normalized[joint_name] = (x_norm, y_norm)
    
    return joints_pixels, joints_normalized


def _process_frame_with_angles(
    frame: np.ndarray,
    joints_pixels: Dict[str, Tuple[int, int]],
    joints_normalized: Dict[str, Tuple[float, float]],
    overlays: Overlays
) -> None:
    """
    Calculate angles and draw skeleton with angle overlays on the frame.
    
    Args:
        frame: The video frame to draw on
        joints_pixels: Dictionary mapping joint names to pixel coordinates
        joints_normalized: Dictionary mapping joint names to normalized coordinates
        overlays: Overlays instance for drawing
    """
    if not joints_pixels:
        return
    
    # Calculate angles separately for left and right
    angles_dict: Dict[str, float] = {}
    
    # Calculate left and right knee angles
    left_knee_angle = compute_left_knee_angle(joints_normalized)
    right_knee_angle = compute_right_knee_angle(joints_normalized)
    
    # Calculate left and right hip angles
    left_hip_angle = compute_left_hip_angle(joints_normalized)
    right_hip_angle = compute_right_hip_angle(joints_normalized)
    
    # Calculate back angle (averages joints internally)
    back_angle = compute_back_angle(joints_normalized)
    
    # Map angles to joint names for display on skeleton
    if left_knee_angle is not None:
        angles_dict["LEFT_KNEE"] = left_knee_angle
    if right_knee_angle is not None:
        angles_dict["RIGHT_KNEE"] = right_knee_angle
    if left_hip_angle is not None:
        angles_dict["LEFT_HIP"] = left_hip_angle
    if right_hip_angle is not None:
        angles_dict["RIGHT_HIP"] = right_hip_angle
    if back_angle is not None:
        # Display back angle at left shoulder (whole-body measurement)
        angles_dict["LEFT_SHOULDER"] = back_angle
    
    # Draw skeleton and angles (only draws valid joints)
    overlays.draw_skeleton(frame, joints_pixels, angles_dict)
    
    # Display angle text at top of frame (compact format)
    y_offset = 20
    font_scale = 0.4
    thickness = 1
    if left_knee_angle is not None:
        cv.putText(frame, f"L Knee: {int(left_knee_angle)}", 
                  (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 
                  font_scale, (0, 255, 0), thickness)
        y_offset += 18
    if right_knee_angle is not None:
        cv.putText(frame, f"R Knee: {int(right_knee_angle)}", 
                  (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 
                  font_scale, (0, 255, 0), thickness)
        y_offset += 18
    if left_hip_angle is not None:
        cv.putText(frame, f"L Hip: {int(left_hip_angle)}", 
                  (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 
                  font_scale, (0, 255, 0), thickness)
        y_offset += 18
    if right_hip_angle is not None:
        cv.putText(frame, f"R Hip: {int(right_hip_angle)}", 
                  (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 
                  font_scale, (0, 255, 0), thickness)
        y_offset += 18
    if back_angle is not None:
        cv.putText(frame, f"Back: {int(back_angle)}", 
                  (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 
                  font_scale, (0, 255, 0), thickness)


def opencv() -> None:
    """
    Opens the first working webcam (cycles from 0-4) and displays its feed with pose detection.
    Shows skeleton overlay and angle measurements for squat analysis.
    Press 'q' to quit.
    """
    # Initialize pose backend and load model
    backend = MediaPipeBackend(model_complexity=1, enable_segmentation=False)
    try:
        backend.load()
    except Exception as e:
        print(f"Error loading pose backend: {e}")
        return
    
    # Initialize overlays for drawing
    overlays = Overlays()
    
    # Try to find a working camera
    for i in range(5):
        capture = cv.VideoCapture(i)
        
        if not capture.isOpened():
            capture.release()
            continue
        
        # Verify camera is actually working by reading a test frame
        ret, test_frame = capture.read()
        if not ret or test_frame is None:
            print(f"Camera {i} opened but cannot read frames. Trying next camera...")
            capture.release()
            continue
        
        print(f"Camera {i} opened successfully. Press 'q' to quit.")
        print(f"Frame size: {test_frame.shape}")
        
        # Create window before entering loop
        cv.namedWindow("Webcam - Squat Analysis", cv.WINDOW_NORMAL)
        cv.resizeWindow("Webcam - Squat Analysis", 800, 600)
        
        # Show first frame to ensure window appears
        cv.imshow("Webcam - Squat Analysis", test_frame)
        cv.waitKey(1)
        
        try:
            while True:
                is_true, frame = capture.read()
                if not is_true or frame is None:
                    print("Failed to read frame")
                    break
                
                try:
                    # Get pose predictions
                    result = backend.predict_frame(frame)
                    joints_dict = result.get("joints", {})
                    visibility_dict = result.get("visibility", {})
                    
                    # Only process if we have valid landmarks
                    if not joints_dict or len(joints_dict) == 0:
                        # No detection - skip drawing this frame
                        pass
                    else:
                        frame_height, frame_width = frame.shape[:2]
                        
                        # Convert MediaPipe joints to pixel coordinates for drawing
                        joints_pixels, joints_normalized = _convert_joints_to_pixels(
                            joints_dict, visibility_dict, frame_width, frame_height
                        )
                        
                        # Calculate angles and draw skeleton with overlays
                        _process_frame_with_angles(frame, joints_pixels, joints_normalized, overlays)
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue showing frame even if pose detection fails
                
                # Show frame
                cv.imshow("Webcam - Squat Analysis", frame)
                
                # Press 'q' to exit
                key = cv.waitKey(20) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == 27:  # ESC key
                    print("Quitting...")
                    break
        
        except Exception as e:
            print(f"Error in camera loop: {e}")
        finally:
            capture.release()
            cv.destroyAllWindows()
            cv.waitKey(1)
            break  # Exit after first working camera


def process_video(video_path: str, exercise_type: Optional[str] = None, output_path: Optional[str] = None) -> None:
    """
    Process a video file and display it with pose detection and angle analysis.
    
    Args:
        video_path: Path to the input video file
        exercise_type: Optional exercise type (currently unused, kept for compatibility)
        output_path: Optional path to save output video with analysis overlay
    """
    from pathlib import Path
    
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Initialize pose backend and load model
    backend = MediaPipeBackend(model_complexity=1, enable_segmentation=False)
    try:
        backend.load()
    except Exception as e:
        print(f"Error loading pose backend: {e}")
        return
    
    # Initialize overlays for drawing
    overlays = Overlays()
    
    # Open video file
    capture = cv.VideoCapture(str(video_file))
    
    if not capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(capture.get(cv.CAP_PROP_FPS))
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"Video opened: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print("Press 'q' to quit, SPACE to pause/resume")
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")
    
    # Create window
    cv.namedWindow("Video - Squat Analysis", cv.WINDOW_NORMAL)
    cv.resizeWindow("Video - Squat Analysis", 800, 600)
    
    paused = False
    frame_delay = max(1, int(1000 / fps)) if fps > 0 else 30
    
    try:
        while True:
            if not paused:
                ret, frame = capture.read()
                if not ret or frame is None:
                    print("End of video or failed to read frame")
                    break
            else:
                # When paused, just wait for key press
                key = cv.waitKey(30) & 0xFF
                if key == ord(' '):
                    paused = False
                elif key == ord('q') or key == 27:
                    break
                continue
            
            try:
                    # Get pose predictions
                    result = backend.predict_frame(frame)
                    joints_dict = result.get("joints", {})
                    visibility_dict = result.get("visibility", {})
                    
                    # Only process if we have valid landmarks
                    if not joints_dict or len(joints_dict) == 0:
                        # No detection - skip drawing this frame
                        pass
                    else:
                        frame_height, frame_width = frame.shape[:2]
                        
                        # Convert MediaPipe joints to pixel coordinates for drawing
                        joints_pixels, joints_normalized = _convert_joints_to_pixels(
                            joints_dict, visibility_dict, frame_width, frame_height
                        )
                        
                        # Calculate angles and draw skeleton with overlays
                        _process_frame_with_angles(frame, joints_pixels, joints_normalized, overlays)
            
            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                # Continue showing frame even if pose detection fails
            
            # Show frame
            cv.imshow("Video - Squat Analysis", frame)
            
            # Write frame if output is specified
            if writer is not None:
                writer.write(frame)
            
            # Handle key presses
            key = cv.waitKey(frame_delay) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Quitting...")
                break
            elif key == ord(' '):  # SPACE to pause/resume
                paused = not paused
                if paused:
                    print("Paused - Press SPACE to resume, 'q' to quit")
                else:
                    print("Resumed")
    
    except Exception as e:
        print(f"Error in video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"Output video saved to: {output_path}")
        cv.destroyAllWindows()
        cv.waitKey(1)


if __name__ == "__main__":
    opencv()