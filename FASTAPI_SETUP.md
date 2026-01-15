# FastAPI Setup Guide - Workout Form Optimizer

## Overview

This guide explains how to set up and use the FastAPI application for analyzing and optimizing workout form through pose estimation.

## Quick Start

### 1. Install Dependencies

```bash
# From the project root directory
pip install -r requirements.txt
```

Required packages:
- **fastapi**: Modern web framework for building APIs
- **uvicorn**: ASGI server to run FastAPI
- **pydantic**: Data validation using Python type hints
- **python-multipart**: File upload support

### 2. Run the API Server

#### Option A: Using the setup script
```bash
chmod +x run_api.sh
./run_api.sh
```

#### Option B: Manual startup
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

## API Endpoints

### Health & Status

**GET** `/health`
- Check API health and model status
- Returns: `{"status": "healthy", "model_loaded": boolean}`

### Pose Estimation

**POST** `/api/pose/estimate`
- Estimate pose from a single image
- Request: Upload image file
- Returns: Dictionary with pose landmarks and coordinates

**POST** `/api/pose/batch`
- Estimate poses from multiple images
- Request: Upload multiple image files
- Returns: List of pose estimations with error handling

### Form Analysis

**POST** `/api/form/analyze`
- Analyze workout form based on pose data
- Request JSON:
  ```json
  {
    "pose": {
      "shoulder": [x, y, z],
      "hip": [x, y, z],
      "knee": [x, y, z],
      "ankle": [x, y, z]
    },
    "exercise_type": "squat|deadlift|benchpress"
  }
  ```
- Returns: Form score, joint angles, and detailed feedback

**POST** `/api/form/video`
- Analyze form throughout a video
- Request: Upload video file + exercise_type parameter
- Returns: Frame-by-frame analysis with aggregate metrics

### Exercise-Specific Endpoints

**POST** `/api/exercises/squat`
- Specialized squat form analysis
- Request: Upload image file

**POST** `/api/exercises/deadlift`
- Specialized deadlift form analysis
- Request: Upload image file

**POST** `/api/exercises/benchpress`
- Specialized bench press form analysis
- Request: Upload image file

## Interactive Documentation

Once the server is running, access:

- **Swagger UI**: `http://localhost:8000/docs`

This interface allow you to:
- View all available endpoints
- See request/response schemas
- Test endpoints directly from the browser
- View example requests and responses

## API Response Structure

### Success Response
```json
{
  "exercise_type": "squat",
  "form_score": 85.5,
  "joint_angles": {
    "knee_angle": 95.2,
    "hip_angle": 78.5,
    "back_angle": 175.3
  },
  "feedback": [
    {
      "area": "knees",
      "severity": "warning",
      "message": "Knee angle too acute..."
    }
  ],
  "status": "success"
}
```

### Error Response
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Environment Variables

Configure the API using environment variables:

```bash
# Server configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_RELOAD=True

# Model configuration
export MODEL_PATH=/path/to/model
export CONFIDENCE_THRESHOLD=0.5
```

## Testing the API

### Using cURL

```bash
# Test health endpoint
curl http://localhost:8000/health

# Upload and analyze an image
curl -X POST "http://localhost:8000/api/exercises/squat" \
  -F "file=@path/to/image.jpg"

# Analyze form with pose data
curl -X POST "http://localhost:8000/api/form/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "pose": {
      "shoulder": [0.5, 0.3, 0.0],
      "hip": [0.5, 0.5, 0.0],
      "knee": [0.5, 0.7, 0.0],
      "ankle": [0.5, 0.9, 0.0]
    },
    "exercise_type": "squat"
  }'
```

### Using Python Requests

```python
import requests

# Test health
response = requests.get("http://localhost:8000/health")
print(response.json())

# Upload image for analysis
with open("workout_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/v1/exercises/squat",
        files=files
    )
    print(response.json())
```

## Supported Exercises

The API currently supports form analysis for:
1. Squat
2. Deadlift
3. Bench Press

## Troubleshooting

### Port Already in Use
```bash
# Change the port
python -m uvicorn src.api.main:app --port 8001
```

### Model Not Loading
- Ensure MediaPipe is properly installed: `pip install mediapipe`
- Check that the pose estimator initialization doesn't raise errors

### File Upload Issues
- Maximum file size: 500MB for videos
- Supported formats: JPG, PNG, MP4, MOV

### CORS Issues
The API has CORS enabled for all origins. To restrict this, modify the CORS middleware in `src/api/main.py`.

## Development

### Running with Auto-reload
```bash
python -m uvicorn src.api.main:app --reload
```