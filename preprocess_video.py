import argparse
import cv2
import os
from pathlib import Path
import math

def enhance_resolution(frame, scale_factor=2):
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    enhanced = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return enhanced

def process_video(video_path, output_dir, time_interval):
    # Create output directory and all intermediate directories if they don't exist
    current_path = Path()
    for part in Path(output_dir).parts:
        current_path = current_path / part
        current_path.mkdir(exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_interval = round(fps * time_interval)  # Round to nearest integer for better accuracy
    
    print(f"Video FPS: {fps}")
    print(f"Total frames reported: {total_frames}")
    print(f"Calculated duration: {duration:.2f} seconds")
    print(f"Frame interval: {frame_interval} frames (for {time_interval} sec interval)")
    print(f"Expected saved frames: {int(total_frames / frame_interval) + 1 if frame_interval > 0 else 0}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frames at specified interval
        if frame_count % frame_interval == 0:
            # Enhance resolution
            enhanced_frame = enhance_resolution(frame)
            
            # Save the frame
            output_path = os.path.join(output_dir, f"{saved_count}.jpg")
            cv2.imwrite(output_path, enhanced_frame)
            saved_count += 1
        
        frame_count += 1
    
    # Release video capture
    cap.release()
    print(f"Processed {frame_count} frames, saved {saved_count} enhanced frames to {output_dir}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Enhance video resolution and split into frames")
    parser.add_argument("--video_path", type=str, help="Path to the input MP4 video file")
    parser.add_argument("--output_dir", type=str, default="output_frames", 
                        help="Directory to save enhanced frames (default: output_frames)")
    parser.add_argument("--interval", type=float, default=0.1, 
                        help="Time interval in seconds between saved frames (default: 0.1)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the video
    process_video(args.video_path, args.output_dir, args.interval)

if __name__ == "__main__":
    main()