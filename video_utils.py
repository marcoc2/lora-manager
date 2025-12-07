"""
Video utility functions.
Wrapper around video_generator for backwards compatibility.
"""
from pathlib import Path
from typing import Union, Optional

# Import from robust generator
from video_generator import create_video_from_folder, create_video_from_arrays


def create_video_from_images(
    image_folder: Union[str, Path],
    output_video_path: Union[str, Path],
    fps: int = 12
) -> Optional[Path]:
    """
    Creates a video from images in a folder.

    This is a wrapper around the robust video_generator for backwards compatibility.
    Uses imageio-ffmpeg (bundled FFmpeg) which works reliably on Windows.

    Args:
        image_folder: Path to the folder containing images.
        output_video_path: Path to the output video file.
        fps: Frames per second for the video.

    Returns:
        Path to the created video file, or None if failed.
    """
    return create_video_from_folder(
        image_folder=image_folder,
        output_path=output_video_path,
        fps=fps,
        sort_by_mtime=True,
        create_gif_fallback=True
    )


# For direct imports
__all__ = ['create_video_from_images', 'create_video_from_folder', 'create_video_from_arrays']
