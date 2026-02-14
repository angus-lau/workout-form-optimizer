def analyze_workout(frames):
    """
    Analyzes workout frames. 
    Expects frames to have keys matching the utility output: 
    'knee_angle' and 'hip_angle'.
    """
    
    def summarize(vals):
        # Filter out None values in case the pose estimator missed a frame
        clean_vals = [v for v in vals if v is not None]
        
        if not clean_vals:
            return {"avg": 0, "min": 0, "max": 0}
            
        return {
            "avg": round(sum(clean_vals) / len(clean_vals), 2),
            "min": round(min(clean_vals), 2),
            "max": round(max(clean_vals), 2),
        }

    # Extracting values based on angle_utils output
    knee = [f.get("knee_angle") for f in frames]
    hip = [f.get("hip_angle") for f in frames]

    knee_summary = summarize(knee)
    hip_summary = summarize(hip)

    # Logic for pass/fail
    knee_pass = all(120 <= v <= 150 for v in knee if v is not None)
    hip_pass = all(80 <= v <= 100 for v in hip if v is not None)

    return {
        "frames": frames,
        "summary": {
            "knee_angle": knee_summary,
            "hip_angle": hip_summary,
        },
        "status": {
            "knee_angle": "PASS" if knee_pass else "FAIL",
            "hip_angle": "PASS" if hip_pass else "FAIL",
            "overall": "PASS" if (knee_pass and hip_pass) else "FAIL"
        },
    }

if __name__ == "__main__":
    # Create a small test list to pass into the function
    test_frames = [
        {"frame": 0, "knee_angle": 132.4, "hip_angle": 88.1},
        {"frame": 1, "knee_angle": 134.0, "hip_angle": 90.2},
        {"frame": 2, "knee_angle": 129.6, "hip_angle": 85.9},
    ]
    # Pass test_frames into the function
    print(analyze_workout(test_frames))