import sys
import os
import torch
import numpy as np
from pathlib import Path
import types

# Add current directory to path for video_generator import
sys.path.insert(0, str(Path(__file__).parent))

# Add reference directory to path
REF_DIR = Path(__file__).parent / "reference" / "ComfyUI-Frame-Interpolation"
sys.path.append(str(REF_DIR))

# Mock comfy.model_management
class MockModelManagement:
    @staticmethod
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def soft_empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Inject mock into sys.modules
mock_comfy = types.ModuleType("comfy")
mock_comfy.model_management = MockModelManagement()
sys.modules["comfy"] = mock_comfy
sys.modules["comfy.model_management"] = MockModelManagement()

# Verify dependencies
try:
    import einops
except ImportError:
    print("Installing einops...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "einops"])

# Import RIFE
try:
    from vfi_models.rife import RIFE_VFI
except ImportError as e:
    print(f"Error importing RIFE: {e}")
    print(f"Sys path: {sys.path}")
    sys.exit(1)

def load_images(folder_path):
    """Load image files sorted by modification time."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    files = sorted([
        f for f in Path(folder_path).glob('*')
        if f.suffix.lower() in extensions
    ], key=lambda x: x.stat().st_mtime)
    return files


def load_image_pil(path):
    """Load image using PIL (more reliable than cv2 on Windows)."""
    from PIL import Image
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def interpolate_video(input_folder, output_file, multiplier=8, fps=24):
    """
    Interpolate frames using RIFE and create video.
    Uses robust video_generator for reliable output on Windows.
    """
    print(f"Initializing RIFE (Multiplier: {multiplier}x)...")

    # Ensure checkpoints directory exists
    ckpt_dir = REF_DIR / "ckpts" / "rife"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rife = RIFE_VFI()

    images = load_images(input_folder)
    if not images:
        print("No images found!")
        return False

    print(f"Found {len(images)} images.")

    # Load images into tensor using PIL (more reliable than cv2)
    print("Loading images...")
    frames_list = []
    for img_path in images:
        try:
            img = load_image_pil(img_path)
            # Normalize to 0-1 float (RIFE expects this)
            img = img.astype(np.float32) / 255.0
            frames_list.append(img)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")

    if not frames_list:
        print("No valid images could be loaded!")
        return False

    frames_tensor = torch.from_numpy(np.stack(frames_list))
    print(f"Loaded {len(frames_list)} frames, shape: {frames_tensor.shape}")

    print("Starting interpolation...")
    try:
        # RIFE_VFI.vfi returns (out_tensor,)
        with torch.no_grad():
            out_tuple = rife.vfi(
                ckpt_name="rife47.pth",
                frames=frames_tensor,
                multiplier=multiplier,
                clear_cache_after_n_frames=10,
                fast_mode=True,
                ensemble=True
            )
            out_frames = out_tuple[0]  # [N, H, W, C] tensor

        print(f"Generated {len(out_frames)} interpolated frames.")

        # Convert tensor to list of numpy arrays (RGB, uint8)
        print("Converting frames...")
        frame_arrays = []
        for i in range(len(out_frames)):
            frame = out_frames[i].numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame_arrays.append(frame)

        # Use robust video generator instead of cv2
        print("Creating video with robust generator...")
        from video_generator import create_video_from_arrays

        output_path = Path(output_file)
        result = create_video_from_arrays(frame_arrays, output_path, fps)

        if result:
            print(f"Video saved to {result}")
            return True
        else:
            print("Video creation failed!")
            return False

    except Exception as e:
        print(f"Error during interpolation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interpolate video frames using RIFE AI")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing images")
    parser.add_argument("--output", "-o", default="video_rife_torch.mp4", help="Output video path")
    parser.add_argument("--multiplier", "-m", type=int, default=8, help="Frame multiplier (2, 4, 8, 16)")
    parser.add_argument("--fps", type=int, default=24, help="Output FPS")

    args = parser.parse_args()

    success = interpolate_video(args.input, args.output, args.multiplier, args.fps)
    sys.exit(0 if success else 1)
