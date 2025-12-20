'''
Preprocessor for videos in data/raw

Scans for videos in raw/ directory, resizes, extracts frames for every FRAME_SKIP, creates new class folder,
moves frames into processed directory
'''


import cv2
import os
import glob
from pathlib import Path


INPUT_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/processed"
IMAGE_SIZE = (224, 224) #this resolution size is commonly used for pretrained models, can go lower
FRAME_SKIP = 10 #more or less frames?
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

def process_videos():
    '''
    Extracts and resizes frames from videos found in the input directory.

    This function scans `INPUT_FOLDER` recursively for video files. For each video found,
    it creates a corresponding subdirectory in `OUTPUT_FOLDER` preserving the class name.
    It iterates through the video, skipping frames based on `FRAME_SKIP`, resizes the 
    kept frames to `IMAGE_SIZE`, and saves them as .jpg files.

    Directory Structure:
        Input:  data/raw/{class_name}/{video_name}.mp4
        Output: data/processed/{class_name}/{video_name}/frame_{i}.jpg

    Global Constants Used:
        INPUT_FOLDER (str): Root path to raw video data.
        OUTPUT_FOLDER (str): Root path where processed frames will be saved.
        IMAGE_SIZE (tuple): Target (width, height) for resized images.
        FRAME_SKIP (int): Interval for sampling (e.g., 10 saves every 10th frame).

    Returns:
        None: Files are written directly to the disk.
    '''
    video_paths = [
        p for p in Path(INPUT_FOLDER).rglob("*") 
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    print(f"Found {len(video_paths)} videos to process...")

    for video_path in video_paths:
        print(f"Processing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        video_name = video_path.stem # 'video1' without .mp4
        
        # Create the specific output folder for this video
        save_path = os.path.join(OUTPUT_FOLDER, video_name)
        os.makedirs(save_path, exist_ok=True)
        
        count = 0
        saved_count = 0
        
        while True:
    
            success, frame = cap.read()
            

            if not success:
                break
            

            if count % FRAME_SKIP == 0:
                
                resized_frame = cv2.resize(frame, IMAGE_SIZE)
                
                #saving as .jpg rather than as a np array is better for storage and dataset scaleability
                filename = f"frame_{saved_count}.jpg"
                full_file_path = os.path.join(save_path, filename)
                
                cv2.imwrite(full_file_path, resized_frame)
                saved_count += 1
                
            count += 1
            
        cap.release()
        print(f"  -> Saved {saved_count} frames to {save_path}")

if __name__ == "__main__":
    process_videos()