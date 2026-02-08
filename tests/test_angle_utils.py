"""Comprehensive test script for angle_utils.py.

Tests all angle calculation functions including:
- compute_angle (base function)
- compute_back_angle
- compute_knee_angle (with side parameter)
- compute_left_knee_angle, compute_right_knee_angle
- compute_hip_angle (with side parameter)
- compute_left_hip_angle, compute_right_hip_angle
- compute_elbow_angle (with side parameter)

Tests cover:
- Valid inputs with known angles
- Edge cases (collinear points, zero vectors)
- Invalid inputs (None, missing keys, wrong types)
- Side parameter functionality
- MediaPipe coordinate format compatibility
"""

import numpy as np
from src.features.angle_utils import (
    compute_angle,
    compute_back_angle,
    compute_knee_angle,
    compute_left_knee_angle,
    compute_right_knee_angle,
    compute_hip_angle,
    compute_left_hip_angle,
    compute_right_hip_angle,
    compute_elbow_angle,
)

def test_compute_angle_basic():
    """Test compute_angle with known geometric configurations."""

    print("Test Batch 1: Basic compute_angle Tests")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Right angle (90 degrees)
    tests_total += 1
    a1 = (1, 0, 0)
    b1 = (0, 0, 0)
    c1 = (0, 1, 0)
    angle1 = compute_angle(a1, b1, c1)
    passed = abs(angle1 - 90) < 1
    tests_passed += passed
    print(f"\nTest 1: Right angle (90 degrees): {'PASS' if passed else 'FAIL'}")
    print(f" Points: A={a1}, B={b1}, C={c1}")
    print(f" Expected: ~90°, Got: {angle1:.2f}°")
    
    # Test 2: Straight line (180 degrees)
    tests_total += 1
    a2 = (0, 0, 0)
    b2 = (1, 0, 0)
    c2 = (2, 0, 0)
    angle2 = compute_angle(a2, b2, c2)
    passed = abs(angle2 - 180) < 1
    tests_passed += passed
    print(f"\nTest 2: Straight line (180 degrees): {'PASS' if passed else 'FAIL'}")
    print(f"  Points: A={a2}, B={b2}, C={c2}")
    print(f"  Expected: ~180°, Got: {angle2:.2f}°")
    
    # Test 3: Acute angle (60 degrees)
    tests_total += 1
    a3 = (0, 0, 0)
    b3 = (1, 0, 0)
    c3 = (0.5, np.sqrt(3)/2, 0)
    angle3 = compute_angle(a3, b3, c3)
    passed = abs(angle3 - 60) < 1
    tests_passed += passed
    print(f"\nTest 3 - Acute angle (60 degrees): {'PASS' if passed else 'FAIL'}")
    print(f"  Points: A={a3}, B={b3}, C={c3}")
    print(f"  Expected: ~60°, Got: {angle3:.2f}°")
    
    # Test 4: 45 degree angle
    tests_total += 1
    a4 = (0, 1, 0)
    b4 = (0, 0, 0)
    c4 = (1, 1, 0)
    angle4 = compute_angle(a4, b4, c4)
    passed = abs(angle4 - 45) < 1
    tests_passed += passed
    print(f"\nTest 4 - 45 degree angle: {'PASS' if passed else 'FAIL'}")
    print(f"  Points: A={a4}, B={b4}, C={c4}")
    print(f"  Expected: ~45°, Got: {angle4:.2f}°")
    
    # Test 5: Fully extended knee (180 degrees)
    tests_total += 1
    hip = (0, 0, 0)
    knee = (0, 1, 0)
    ankle_extended = (0, 2, 0)
    angle_extended = compute_angle(hip, knee, ankle_extended)
    passed = abs(angle_extended - 180) < 1
    tests_passed += passed
    print(f"\nTest 5 - Fully extended knee (180°): {'PASS' if passed else 'FAIL'}")
    print(f"  Hip={hip}, Knee={knee}, Ankle={ankle_extended}")
    print(f"  Expected: ~180°, Got: {angle_extended:.2f}°")
    
    # Test 6: Bent knee
    tests_total += 1
    ankle_bent = (1, 1.5, 0)
    angle_bent = compute_angle(hip, knee, ankle_bent)
    passed = angle_bent < 180 and angle_bent > 0
    tests_passed += passed
    print(f"\nTest 6 - Bent knee: {'PASS' if passed else 'FAIL'}")
    print(f"  Hip={hip}, Knee={knee}, Ankle={ankle_bent}")
    print(f"  Expected: <180° (bent), Got: {angle_bent:.2f}°")
    
    # Test 7: Normalized coordinates (MediaPipe format)
    tests_total += 1
    a7 = (0.5, 0.3, 0.0)
    b7 = (0.5, 0.5, 0.0)
    c7 = (0.7, 0.5, 0.0)
    angle7 = compute_angle(a7, b7, c7)
    passed = abs(angle7 - 90) < 1
    tests_passed += passed
    print(f"\nTest 7 - Normalized coordinates (0-1 range): {'PASS' if passed else 'FAIL'}")
    print(f"  Points: A={a7}, B={b7}, C={c7}")
    print(f"  Expected: ~90°, Got: {angle7:.2f}°")
    
    print(f"\n{'─' * 70}")
    print(f"Section 1 Results: {tests_passed}/{tests_total} tests passed")
    assert tests_passed == tests_total, f"Failed! Only {tests_passed}/{tests_total} passed."

def test_compute_angle_edge_cases():
    """Test compute_angle with edge cases and error conditions."""
    
    print("Test Batch 2: Edge Cases and Error Handling")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: None inputs
    tests_total += 1
    angle = compute_angle(None, (0, 0, 0), (1, 0, 0))
    passed = angle is None
    tests_passed += passed
    print(f"\nTest 1 - None as first point: {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: None, Got: {angle}")
    
    tests_total += 1
    angle = compute_angle((0, 1, 0), None, (1, 0, 0))
    passed = angle is None
    tests_passed += passed
    print(f"\nTest 2 - None as vertex point: {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: None, Got: {angle}")
    
    tests_total += 1
    angle = compute_angle((0, 1, 0), (0, 0, 0), None)
    passed = angle is None
    tests_passed += passed
    print(f"\nTest 3 - None as third point: {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: None, Got: {angle}")
    
    # Test 4: Invalid types (list instead of tuple)
    tests_total += 1
    angle = compute_angle([0, 1], (0, 0, 0), (1, 0, 0))
    passed = angle is None
    tests_passed += passed
    print(f"\nTest 4 - List instead of tuple: {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: None, Got: {angle}")
    
    # Test 5: Too short tuple
    tests_total += 1
    angle = compute_angle((0,), (0, 0, 0), (1, 0, 0))
    passed = angle is None
    tests_passed += passed
    print(f"\nTest 5 - Tuple too short: {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: None, Got: {angle}")
    
    # Test 6: Zero-length vector (identical points a and b)
    tests_total += 1
    angle = compute_angle((0, 0, 0), (0, 0, 0), (1, 0, 0))
    passed = angle is None
    tests_passed += passed
    print(f"\nTest 6 - Zero-length vector (a=b): {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: None, Got: {angle}")
    
    # Test 7: Zero-length vector (identical points b and c)
    tests_total += 1
    angle = compute_angle((0, 1, 0), (0, 0, 0), (0, 0, 0))
    passed = angle is None
    tests_passed += passed
    print(f"\nTest 7 - Zero-length vector (b=c): {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: None, Got: {angle}")
    
    # Test 8: Collinear points (180 degrees)
    tests_total += 1
    angle = compute_angle((2, 0, 0), (1, 0, 0), (0, 0, 0))
    passed = angle is not None and abs(angle - 180) < 1
    tests_passed += passed
    print(f"\nTest 8 - Collinear points (0°): {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: ~0°, Got: {angle:.2f}°" if angle else f"  Got: {angle}")
    
    # Test 9: Non-numeric coordinates
    tests_total += 1
    try:
        angle = compute_angle(("a", "b", 0), (0, 0, 0), (1, 0, 0))
        passed = angle is None
    except:
        passed = False
    tests_passed += passed
    print(f"\nTest 9 - Non-numeric coordinates: {'PASS' if passed else 'FAIL'}")
    print(f"  Expected: None (or exception handled), Got: {angle if 'angle' in locals() else 'Exception'}")
    
    print(f"\n{'─' * 70}")
    print(f"Section 2 Results: {tests_passed}/{tests_total} tests passed")
    assert tests_passed == tests_total, f"Failed! Only {tests_passed}/{tests_total} passed."