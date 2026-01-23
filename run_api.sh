#!/bin/bash
# FastAPI Setup and Run Script

echo "🏋️  Workout Form Optimizer - FastAPI Setup"
echo "=============================================="

# Check if venv exists and activate it
if [ -d "workout/bin" ]; then
    echo "✓ Virtual environment found"
    source workout/bin/activate
else
    echo "⚠ Virtual environment not found. Creating..."
    python3 -m venv workout
    source workout/bin/activate
fi

# Install/upgrade dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "📖 Interactive API Documentation:"
echo "    http://localhost:8000/docs (Swagger UI)"
echo ""
echo "🚀 Starting API server..."
echo ""

# Run FastAPI with uvicorn
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload