"""Angle calculation utilities for joint angle measurements from pose landmarks."""

import numpy as np
from typing import Optional, Tuple, Dict


def compute_angle(a: Tuple[float, float, float],
                  b: Tuple[float, float, float],
                  c: Tuple[float, float, float]) -> Optional[float]:
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


def compute_knee_angle(joints: Dict[str, Tuple[float, float, float]]) -> Optional[float]:
    """Compute knee flexion angle from joint coordinates.
    
    Calculates the angle at the knee joint formed by hip-knee-ankle.
    This measures knee flexion/extension, where smaller angles indicate
    more flexion (bent knee) and larger angles indicate more extension.
    
    Args:
        joints: Dictionary containing joint coordinates. Expected keys:
            - 'hip': Hip joint coordinates (x, y, z)
            - 'knee': Knee joint coordinates (x, y, z)
            - 'ankle': Ankle joint coordinates (x, y, z)
            
    Returns:
        Knee flexion angle in degrees (0-180), or None if required joints
        are missing or invalid. 180° = fully extended, smaller values = more flexion.
    """
    if joints is None:
        return None
    
    required_keys = ['hip', 'knee', 'ankle']
    if not all(key in joints for key in required_keys):
        return None
    
    hip = joints['hip']
    knee = joints['knee']
    ankle = joints['ankle']
    
    if hip is None or knee is None or ankle is None:
        return None
    
    if not isinstance(hip, tuple) or not isinstance(knee, tuple) or not isinstance(ankle, tuple):
        return None
    
    if len(hip) < 2 or len(knee) < 2 or len(ankle) < 2:
        return None
    
    return compute_angle(hip, knee, ankle)


def compute_hip_angle(joints: Dict[str, Tuple[float, float, float]]) -> Optional[float]:
    """Compute hip angle from joint coordinates.
    
    Calculates the angle at the hip joint formed by shoulder-hip-knee.
    This measures hip flexion/extension, useful for assessing squat depth,
    deadlift form, and overall hip mobility.
    
    Args:
        joints: Dictionary containing joint coordinates. Expected keys:
            - 'shoulder': Shoulder joint coordinates (x, y, z)
            - 'hip': Hip joint coordinates (x, y, z)
            - 'knee': Knee joint coordinates (x, y, z)
            
    Returns:
        Hip angle in degrees (0-180), or None if required joints are missing
        or invalid. Larger angles indicate more hip extension, smaller angles
        indicate more hip flexion.
    """
    if joints is None:
        return None
    
    required_keys = ['shoulder', 'hip', 'knee']
    if not all(key in joints for key in required_keys):
        return None
    
    shoulder = joints['shoulder']
    hip = joints['hip']
    knee = joints['knee']
    
    if shoulder is None or hip is None or knee is None:
        return None
    
    if not isinstance(shoulder, tuple) or not isinstance(hip, tuple) or not isinstance(knee, tuple):
        return None
    
    if len(shoulder) < 2 or len(hip) < 2 or len(knee) < 2:
        return None
    
    return compute_angle(shoulder, hip, knee)


def compute_back_angle(joints: Dict[str, Tuple[float, float, float]]) -> Optional[float]:
    """Compute back/spinal alignment angle from joint coordinates.
    
    Calculates the angle at the hip formed by shoulder-hip-ankle.
    This measures spinal alignment and back posture, where angles close
    to 180° indicate a straight/neutral spine, and deviations indicate
    forward lean or rounding.
    
    Args:
        joints: Dictionary containing joint coordinates. Expected keys:
            - 'shoulder': Shoulder joint coordinates (x, y, z)
            - 'hip': Hip joint coordinates (x, y, z)
            - 'ankle': Ankle joint coordinates (x, y, z)
            
    Returns:
        Back angle in degrees (0-180), or None if required joints are missing
        or invalid. 180° = straight vertical alignment, smaller angles = forward lean.
    """
    if joints is None:
        return None
    
    required_keys = ['shoulder', 'hip', 'ankle']
    if not all(key in joints for key in required_keys):
        return None
    
    shoulder = joints['shoulder']
    hip = joints['hip']
    ankle = joints['ankle']
    
    if shoulder is None or hip is None or ankle is None:
        return None
    
    if not isinstance(shoulder, tuple) or not isinstance(hip, tuple) or not isinstance(ankle, tuple):
        return None
    
    if len(shoulder) < 2 or len(hip) < 2 or len(ankle) < 2:
        return None
    
    return compute_angle(shoulder, hip, ankle)
