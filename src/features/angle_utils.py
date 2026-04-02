"""Angle calculation utilities for joint angle measurements from pose landmarks."""

import numpy as np
from typing import Optional, Tuple, Dict


def compute_angle(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
) -> Optional[float]:
    """Compute the angle at point b formed by points a, b, and c.

    Calculates the angle in degrees between vectors (a - b) and (c - b),
    where b is the vertex of the angle.

    Args:
        a: First point coordinates (x, y, z).
        b: Vertex point coordinates (x, y, z) - the point where the angle is measured.
        c: Third point coordinates (x, y, z).

    Returns:
        Angle in degrees (0-180) between the two vectors, or None if input is invalid
        or points are identical (zero-length vectors).
    """
    if a is None or b is None or c is None:
        return None

    if not isinstance(a, tuple) or not isinstance(b, tuple) or not isinstance(c, tuple):
        return None

    if len(a) < 2 or len(b) < 2 or len(c) < 2:
        return None

    try:
        point_a = np.array([float(a[0]), float(a[1])])
        point_b = np.array([float(b[0]), float(b[1])])
        point_c = np.array([float(c[0]), float(c[1])])
    except (ValueError, TypeError, IndexError):
        return None

    vector1 = point_a - point_b
    vector2 = point_c - point_b

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0.0 or norm2 == 0.0:
        return None

    cosine_angle = np.dot(vector1, vector2) / (norm1 * norm2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)

    return angle_degrees


def compute_back_angle(joints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Compute back/spinal alignment angle from MediaPipe joint coordinates.
    
    Calculates the angle at the hip formed by shoulder-hip-ankle using averaged positions.
    This measures spinal alignment and back posture, where angles close
    to 180° indicate a straight/neutral spine, and deviations indicate
    forward lean or rounding.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
                'LEFT_ANKLE', 'RIGHT_ANKLE'
            
    Returns:
        Back angle in degrees (0-180), or None if required joints are missing.
        180° = straight vertical alignment, smaller angles = forward lean.
    """
    if joints is None:
        return None
    
    # Get left and right joints
    left_shoulder = joints.get("LEFT_SHOULDER")
    right_shoulder = joints.get("RIGHT_SHOULDER")
    left_hip = joints.get("LEFT_HIP")
    right_hip = joints.get("RIGHT_HIP")
    left_ankle = joints.get("LEFT_ANKLE")
    right_ankle = joints.get("RIGHT_ANKLE")
    
    # Average left and right to get center positions
    shoulder = None
    hip = None
    ankle = None
    
    if left_shoulder and right_shoulder:
        shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2.0, 
                   (left_shoulder[1] + right_shoulder[1]) / 2.0, 0.0)
    elif left_shoulder:
        shoulder = (left_shoulder[0], left_shoulder[1], 0.0)
    elif right_shoulder:
        shoulder = (right_shoulder[0], right_shoulder[1], 0.0)
    
    if left_hip and right_hip:
        hip = ((left_hip[0] + right_hip[0]) / 2.0, 
              (left_hip[1] + right_hip[1]) / 2.0, 0.0)
    elif left_hip:
        hip = (left_hip[0], left_hip[1], 0.0)
    elif right_hip:
        hip = (right_hip[0], right_hip[1], 0.0)
    
    if left_ankle and right_ankle:
        ankle = ((left_ankle[0] + right_ankle[0]) / 2.0, 
                (left_ankle[1] + right_ankle[1]) / 2.0, 0.0)
    elif left_ankle:
        ankle = (left_ankle[0], left_ankle[1], 0.0)
    elif right_ankle:
        ankle = (right_ankle[0], right_ankle[1], 0.0)
    
    if shoulder is None or hip is None or ankle is None:
        return None
    
    return compute_angle(shoulder, hip, ankle)


def compute_knee_angle(joints: Dict[str, Tuple[float, float]], 
                       side: str = 'LEFT') -> Optional[float]:
    """Compute knee flexion angle from MediaPipe joint coordinates.
    
    Calculates the angle at the knee joint formed by hip-knee-ankle.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: '{side}_HIP', '{side}_KNEE', '{side}_ANKLE'
        side: Body side ('LEFT' or 'RIGHT'). Default is 'LEFT'.
            
    Returns:
        Knee flexion angle in degrees (0-180), or None if required joints are missing.
    """
    if joints is None:
        return None
    
    side = side.upper()
    hip = joints.get(f"{side}_HIP")
    knee = joints.get(f"{side}_KNEE")
    ankle = joints.get(f"{side}_ANKLE")
    
    if not all([hip, knee, ankle]):
        return None
    
    # Convert to (x, y, z) format for compute_angle
    hip_3d = (hip[0], hip[1], 0.0)
    knee_3d = (knee[0], knee[1], 0.0)
    ankle_3d = (ankle[0], ankle[1], 0.0)
    
    return compute_angle(hip_3d, knee_3d, ankle_3d)


def compute_left_knee_angle(joints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Compute left knee flexion angle from MediaPipe joint coordinates.
    
    Calculates the angle at the left knee joint formed by left_hip-left_knee-left_ankle.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'
            
    Returns:
        Left knee flexion angle in degrees (0-180), or None if required joints are missing.
    """
    return compute_knee_angle(joints, side='LEFT')


def compute_right_knee_angle(joints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Compute right knee flexion angle from MediaPipe joint coordinates.
    
    Calculates the angle at the right knee joint formed by right_hip-right_knee-right_ankle.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'
            
    Returns:
        Right knee flexion angle in degrees (0-180), or None if required joints are missing.
    """
    return compute_knee_angle(joints, side='RIGHT')


def compute_hip_angle(joints: Dict[str, Tuple[float, float]], 
                      side: str = 'LEFT') -> Optional[float]:
    """Compute hip angle from MediaPipe joint coordinates.
    
    Calculates the angle at the hip joint formed by shoulder-hip-knee.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: '{side}_SHOULDER', '{side}_HIP', '{side}_KNEE'
        side: Body side ('LEFT' or 'RIGHT'). Default is 'LEFT'.
            
    Returns:
        Hip angle in degrees (0-180), or None if required joints are missing.
    """
    if joints is None:
        return None
    
    side = side.upper()
    shoulder = joints.get(f"{side}_SHOULDER")
    hip = joints.get(f"{side}_HIP")
    knee = joints.get(f"{side}_KNEE")
    
    if not all([shoulder, hip, knee]):
        return None
    
    # Convert to (x, y, z) format for compute_angle
    shoulder_3d = (shoulder[0], shoulder[1], 0.0)
    hip_3d = (hip[0], hip[1], 0.0)
    knee_3d = (knee[0], knee[1], 0.0)
    
    return compute_angle(shoulder_3d, hip_3d, knee_3d)


def compute_left_hip_angle(joints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Compute left hip angle from MediaPipe joint coordinates.
    
    Calculates the angle at the left hip joint formed by left_shoulder-left_hip-left_knee.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: 'LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'
            
    Returns:
        Left hip angle in degrees (0-180), or None if required joints are missing.
    """
    return compute_hip_angle(joints, side='LEFT')


def compute_right_hip_angle(joints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Compute right hip angle from MediaPipe joint coordinates.
    
    Calculates the angle at the right hip joint formed by right_shoulder-right_hip-right_knee.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'
            
    Returns:
        Right hip angle in degrees (0-180), or None if required joints are missing.
    """
    return compute_hip_angle(joints, side='RIGHT')


def compute_elbow_angle(joints: Dict[str, Tuple[float, float]], 
                        side: str = 'LEFT') -> Optional[float]:
    """Compute elbow flexion angle from MediaPipe joint coordinates.
    
    Calculates the angle at the elbow joint formed by shoulder-elbow-wrist.
    This measures elbow flexion/extension, where smaller angles indicate
    more flexion (bent arm) and larger angles indicate more extension.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: '{side}_SHOULDER', '{side}_ELBOW', '{side}_WRIST'
        side: Body side ('LEFT' or 'RIGHT'). Default is 'LEFT'.
            
    Returns:
        Elbow flexion angle in degrees (0-180), or None if required joints
        are missing or invalid. 180° = fully extended, smaller values = more flexion.
    """
    if joints is None:
        return None
    
    side = side.upper()
    shoulder = joints.get(f"{side}_SHOULDER")
    elbow = joints.get(f"{side}_ELBOW")
    wrist = joints.get(f"{side}_WRIST")
    
    if not all([shoulder, elbow, wrist]):
        return None
    
    if not isinstance(shoulder, tuple) or not isinstance(elbow, tuple) or not isinstance(wrist, tuple):
        return None
    
    if len(shoulder) < 2 or len(elbow) < 2 or len(wrist) < 2:
        return None
    
    # Convert to (x, y, z) format for compute_angle
    shoulder_3d = (shoulder[0], shoulder[1], 0.0)
    elbow_3d = (elbow[0], elbow[1], 0.0)
    wrist_3d = (wrist[0], wrist[1], 0.0)
    
    return compute_angle(shoulder_3d, elbow_3d, wrist_3d)


def compute_left_elbow_angle(joints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Compute left elbow flexion angle from MediaPipe joint coordinates.
    
    Calculates the angle at the left elbow joint formed by left_shoulder-left_elbow-left_wrist.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'
            
    Returns:
        Left elbow flexion angle in degrees (0-180), or None if required joints are missing.
        180° = fully extended, smaller values = more flexion.
    """
    return compute_elbow_angle(joints, side='LEFT')


def compute_right_elbow_angle(joints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Compute right elbow flexion angle from MediaPipe joint coordinates.
    
    Calculates the angle at the right elbow joint formed by right_shoulder-right_elbow-right_wrist.
    
    Args:
        joints: Dictionary with MediaPipe joint names as keys and normalized (x, y) coordinates.
                Expected keys: 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'
            
    Returns:
        Right elbow flexion angle in degrees (0-180), or None if required joints are missing.
        180° = fully extended, smaller values = more flexion.
    """
    return compute_elbow_angle(joints, side='RIGHT')
