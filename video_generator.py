"""
Robust video generator that works reliably on Windows.
Uses imageio-ffmpeg (bundled FFmpeg) as primary method.
Falls back to GIF or direct FFmpeg if available.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

def ensure_dependencies():
    """Ensure required packages are installed."""
    required = ['imageio', 'imageio-ffmpeg', 'pillow']
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# Ensure dependencies on import
ensure_dependencies()

import imageio
from PIL import Image


def get_image_files(folder: Union[str, Path], sort_by_mtime: bool = True) -> List[Path]:
    """Get sorted list of image files from folder."""
    folder = Path(folder)
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if sort_by_mtime:
        files = sorted(files, key=lambda x: x.stat().st_mtime)
    else:
        files = sorted(files, key=lambda x: x.name)

    return files


def load_image_as_array(path: Path) -> Optional[np.ndarray]:
    """Load image as numpy array (RGB)."""
    try:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def create_video_imageio(
    images: List[np.ndarray],
    output_path: Path,
    fps: int = 12,
    codec: str = 'libx264',
    quality: int = 8
) -> bool:
    """
    Create video using imageio-ffmpeg (bundled FFmpeg).
    This is the most reliable method on Windows.
    """
    try:
        # imageio-ffmpeg downloads its own FFmpeg binary
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec=codec,
            quality=quality,  # 0-10, higher is better
            pixelformat='yuv420p',  # Compatibility with most players
            macro_block_size=8,  # Ensure dimensions are compatible
        )

        for frame in images:
            writer.append_data(frame)

        writer.close()

        # Verify file was created and has content
        if output_path.exists() and output_path.stat().st_size > 1000:
            return True
        else:
            print("Warning: imageio created empty or very small file")
            return False

    except Exception as e:
        print(f"imageio-ffmpeg failed: {e}")
        return False


def create_video_ffmpeg_direct(
    image_folder: Path,
    output_path: Path,
    fps: int = 12,
    pattern: str = "*.png"
) -> bool:
    """
    Try to use system FFmpeg directly if available.
    Fallback method if imageio-ffmpeg fails.
    """
    try:
        # Check if ffmpeg is available
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("System FFmpeg not found")
        return False

    try:
        # Create temporary file list for ffmpeg
        images = get_image_files(image_folder)
        if not images:
            return False

        # Use concat demuxer for arbitrary filenames
        list_file = output_path.parent / f"{output_path.stem}_filelist.txt"
        with open(list_file, 'w') as f:
            for img in images:
                # FFmpeg needs forward slashes and escaped single quotes
                escaped_path = str(img.absolute()).replace('\\', '/').replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
                f.write(f"duration {1/fps}\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=300)

        # Clean up temp file
        list_file.unlink(missing_ok=True)

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            return True
        else:
            print(f"FFmpeg error: {result.stderr.decode()}")
            return False

    except Exception as e:
        print(f"Direct FFmpeg failed: {e}")
        return False


def create_gif(
    images: List[np.ndarray],
    output_path: Path,
    fps: int = 12,
    optimize: bool = True
) -> bool:
    """
    Create animated GIF as ultimate fallback.
    Always works but larger file size.
    """
    try:
        gif_path = output_path.with_suffix('.gif')

        pil_images = [Image.fromarray(img) for img in images]

        # Calculate duration in milliseconds
        duration = int(1000 / fps)

        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0,
            optimize=optimize
        )

        if gif_path.exists() and gif_path.stat().st_size > 1000:
            print(f"Created GIF at {gif_path}")
            return True
        return False

    except Exception as e:
        print(f"GIF creation failed: {e}")
        return False


def create_video_from_folder(
    image_folder: Union[str, Path],
    output_path: Union[str, Path],
    fps: int = 12,
    sort_by_mtime: bool = True,
    create_gif_fallback: bool = True
) -> Optional[Path]:
    """
    Main function to create video from images folder.

    Tries multiple methods in order:
    1. imageio-ffmpeg (bundled FFmpeg) - most reliable
    2. System FFmpeg (if available)
    3. Animated GIF fallback

    Args:
        image_folder: Path to folder containing images
        output_path: Path to output video file (will be .mp4)
        fps: Frames per second
        sort_by_mtime: Sort images by modification time (True) or name (False)
        create_gif_fallback: Create GIF if video fails

    Returns:
        Path to created video/gif, or None if all methods failed
    """
    image_folder = Path(image_folder)
    output_path = Path(output_path)

    # Ensure output is .mp4
    if output_path.suffix.lower() != '.mp4':
        output_path = output_path.with_suffix('.mp4')

    # Get images
    image_files = get_image_files(image_folder, sort_by_mtime)

    if not image_files:
        print(f"No images found in {image_folder}")
        return None

    print(f"Found {len(image_files)} images in {image_folder}")

    # Load images
    print("Loading images...")
    images = []
    target_size = None

    for img_path in image_files:
        img = load_image_as_array(img_path)
        if img is not None:
            if target_size is None:
                # Use first image size as target
                target_size = (img.shape[1], img.shape[0])  # width, height
            else:
                # Resize if needed
                if (img.shape[1], img.shape[0]) != target_size:
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
                    img = np.array(pil_img)
            images.append(img)

    if not images:
        print("No valid images could be loaded")
        return None

    print(f"Loaded {len(images)} images at {target_size[0]}x{target_size[1]}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try Method 1: imageio-ffmpeg
    print("Trying imageio-ffmpeg...")
    if create_video_imageio(images, output_path, fps):
        print(f"Video created successfully: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        return output_path

    # Try Method 2: System FFmpeg
    print("Trying system FFmpeg...")
    if create_video_ffmpeg_direct(image_folder, output_path, fps):
        print(f"Video created with system FFmpeg: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        return output_path

    # Method 3: GIF fallback
    if create_gif_fallback:
        print("Creating GIF fallback...")
        gif_path = output_path.with_suffix('.gif')
        if create_gif(images, gif_path, fps):
            print(f"File size: {gif_path.stat().st_size / 1024:.1f} KB")
            return gif_path

    print("All video creation methods failed!")
    return None


def create_video_from_arrays(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 12
) -> Optional[Path]:
    """
    Create video directly from numpy arrays (RGB format).
    Useful for RIFE interpolation output.

    Args:
        frames: List of numpy arrays in RGB format (H, W, C)
        output_path: Path to output video
        fps: Frames per second

    Returns:
        Path to created video, or None if failed
    """
    output_path = Path(output_path)

    if not frames:
        print("No frames provided")
        return None

    print(f"Creating video from {len(frames)} frames...")

    # Ensure output is .mp4
    if output_path.suffix.lower() != '.mp4':
        output_path = output_path.with_suffix('.mp4')

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try imageio-ffmpeg
    if create_video_imageio(frames, output_path, fps):
        print(f"Video created: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        return output_path

    # Fallback to GIF
    print("Trying GIF fallback...")
    gif_path = output_path.with_suffix('.gif')
    if create_gif(frames, gif_path, fps):
        print(f"File size: {gif_path.stat().st_size / 1024:.1f} KB")
        return gif_path

    return None


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create video from images folder")
    parser.add_argument("--input", "-i", required=True, help="Input folder with images")
    parser.add_argument("--output", "-o", default="output_video.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second")
    parser.add_argument("--sort-by-name", action="store_true", help="Sort by filename instead of modification time")
    parser.add_argument("--no-gif-fallback", action="store_true", help="Don't create GIF if video fails")

    args = parser.parse_args()

    result = create_video_from_folder(
        args.input,
        args.output,
        args.fps,
        sort_by_mtime=not args.sort_by_name,
        create_gif_fallback=not args.no_gif_fallback
    )

    if result:
        print(f"\nSuccess! Created: {result}")
    else:
        print("\nFailed to create video")
        sys.exit(1)
