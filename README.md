# workout-optimizer

## setup
1. `git clone git@github.com:angus-lau/workout-form-optimizer.git`
2. `cd workout-form-optimizer`
3. Create a virtual environment "workout"
4. Activate virtual environment:
   - **Bash/Linux/Mac**: `source workout/bin/activate`
   - **PowerShell/Windows**: `.\workout\Scripts\Activate.ps1`
5. `pip install -r requirements.txt`

## running from command line
- Using webcam: `python main.py`
- Using video file ingestion: `python main.py path/to/video`
- Example: `python main.py data/squat/squat_000003.mp4`

## running the web frontend (video files only)
1. Start the API server with virtual environment active: `python -m uvicorn api.server:app --reload --host localhost --port 8000`
2. Open the frontend: open `frontend/index.html` in your browser, or serve it with `python -m http.server 8080 --directory frontend` in a different terminal and visit `http://localhost:8080`
3. Choose a video from the list (data/ and dataset/ folders) or upload your own. The video is processed once and not saved.

## naming
- snake_case for variables and functions
- PascalCase for classes
- UPPER_CASE for constants

## structure
- functions short and single purpose
- imports at top of file
- each module should handle one responsibility

## docstring + type hints
- add docstring for all functions and classes
```py
def example_function(path: str, output_dir: str) -> list:
    """Preprocess a raw video into standardized frames."""
```

## clean code
- avoid unused variables
- avoid commented out code left behind
- avoid giant functions

## git
- present tense e.g 'add function a'
- branch off for each feat/fix and add pull request
- delete branch locally after pull request accepted
- never push to main

