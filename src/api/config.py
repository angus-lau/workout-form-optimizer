"""
Configuration settings for the FastAPI application.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "src" / "data" / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "src" / "data" / "processed_data"

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"

# Model Settings
MODEL_PATH = os.getenv("MODEL_PATH", None)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Video Processing Settings
MAX_VIDEO_SIZE_MB = 500
TEMP_VIDEO_DIR = "/tmp/workout_optimizer_videos"

# Ensure temp directory exists
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
