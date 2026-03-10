# Folder Structure


## Overview
This document describes the current repository folder structure.

## Top-Level Folders
### `data/`
Contains curated datasets used to train the model, organized by exercise:
- `data/benchpress/`
- `data/squat/`
- `data/deadlift/`

### `src/`
Core library code.

#### `src/data/`
- `ingest.py` - downloads and merges dataset into main dataset folder
- `preprocess.py` - scans for videos in raw/ directory, resizes, extracts frames, creates new class folder, moves frames into processed directory
- `metadata.py`- scans raw video files, attaches labels (exercise + form) and saves it into a CSV file

#### `src/features/`
- `backend\mediapipe_backend.py` - provides a thin wrapper so PoseEstimator can swap backends without changing callers
- `angle_utils.py` - angle calculation utilities for joint angle measurements from pose landmarks
- `pose_estimator.py` - defines the PoseEstimator class for estimating poses in images using MediaPipe. 

#### `src/api/`
- `main.py` - FastAPI application entry point defining endpoints for pose estimation, form analysis, video processing, and exercise metadata
- `schemas.py` - Pydantic models used for request/response validation
- `config.py` - configuration constants and settings for the API

#### `src/ui/`
- `overlays.py` - defines a Class for drawing pose overlays on video frame
- `opencv_demo.py`- opens the first working webcam and displays the feed

### `tests/`
- `test_stubs.py` - unit tests for stubs

# Virtual environments
- `workout/` - local virtual environment


