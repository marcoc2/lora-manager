import cv2
import os
from pathlib import Path

def create_video_from_images(image_folder, output_video_path, fps=12):
    """
    Creates a video from images in a folder.
    
    Args:
        image_folder (str or Path): Path to the folder containing images.
        output_video_path (str or Path): Path to the output video file.
        fps (int): Frames per second for the video.
    """
    image_folder = Path(image_folder)
    output_video_path = Path(output_video_path)
    
    images = sorted([img for img in image_folder.glob("*") if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]], key=lambda x: x.stat().st_mtime)
    
    if not images:
        print(f"No images found in {image_folder}")
        return

    # Find first valid image to get dimensions
    frame = None
    for img_path in images:
        temp_frame = cv2.imread(str(img_path))
        if temp_frame is not None:
            frame = temp_frame
            break
            
    if frame is None:
        print(f"No valid images could be read in {image_folder}")
        return

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    # mp4v is a good option for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    count = 0
    for image in images:
        img = cv2.imread(str(image))
        if img is not None:
            # Resize if dimensions don't match first frame
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            video.write(img)
            count += 1
        else:
            print(f"Warning: Could not read image {image}")

    video.release()
    print(f"Video saved to {output_video_path} ({count} frames)")
