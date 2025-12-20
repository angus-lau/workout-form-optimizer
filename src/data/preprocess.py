import cv2
import os
import glob
from pathlib import Path


INPUT_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/processed"
IMAGE_SIZE = (224, 224) 
FRAME_SKIP = 10

def process_videos():
    # 1. Find all mp4 files in the input folder (recursively)
    # This looks for files like: data/raw/pushups/video1.mp4
    video_paths = list(Path(INPUT_FOLDER).rglob("*.mp4"))
    
    print(f"Found {len(video_paths)} videos to process...")

    for video_path in video_paths:
        print(f"Processing: {video_path.name}")
        
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        
        # Get the class name (e.g., 'pushups') from the parent folder
        class_name = video_path.parent.name
        video_name = video_path.stem # 'video1' without .mp4
        
        # Create the specific output folder for this video
        # path becomes: data/processed/pushups/video1/
        save_path = os.path.join(OUTPUT_FOLDER, class_name, video_name)
        os.makedirs(save_path, exist_ok=True)
        
        count = 0
        saved_count = 0
        
        while True:
            # Read one frame
            success, frame = cap.read()
            
            # If no frame is returned, the video is finished
            if not success:
                break
            
            # 2. Skip frames to save space
            # Only run this code if count is divisible by FRAME_SKIP
            if count % FRAME_SKIP == 0:
                
                # 3. Resize the image to be square
                resized_frame = cv2.resize(frame, IMAGE_SIZE)
                
                # 4. Save the image
                filename = f"frame_{saved_count}.jpg"
                full_file_path = os.path.join(save_path, filename)
                
                cv2.imwrite(full_file_path, resized_frame)
                saved_count += 1
                
            count += 1
            
        cap.release()
        print(f"  -> Saved {saved_count} frames to {save_path}")

if __name__ == "__main__":
    process_videos()