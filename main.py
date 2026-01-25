"""Main entry point for workout form optimizer application."""

import argparse
import sys
from pathlib import Path

from src.ui.opencv_demo import process_video


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Workout Form Optimizer - Analyze exercise form from video"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "-e", "--exercise",
        type=str,
        default="squat",
        help="Type of exercise to analyze (default: squat). Currently informational only."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path to save output video with analysis overlay"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the workout form optimizer."""
    args = parse_arguments()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"Processing video: {video_path}")
    print(f"Exercise type: {args.exercise}")
    if args.output:
        print(f"Output video: {args.output}")
    print("Press 'q' to quit, SPACE to pause/resume during playback")
    print("-" * 50)
    
    process_video(str(video_path), args.exercise, args.output)


if __name__ == "__main__":
    main()

