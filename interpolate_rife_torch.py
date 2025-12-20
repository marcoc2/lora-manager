import sys
import os
import traceback
from pathlib import Path
import types

print("[RIFE] Script starting...")

# Add current directory to path for video_generator import
sys.path.insert(0, str(Path(__file__).parent))

# Add reference directory to path
REF_DIR = Path(__file__).parent / "reference" / "ComfyUI-Frame-Interpolation"
sys.path.append(str(REF_DIR))

# Verify reference directory exists
if not REF_DIR.exists():
    print(f"[RIFE] ERROR: Reference directory not found: {REF_DIR}")
    sys.exit(1)

print(f"[RIFE] Reference dir: {REF_DIR}")

# Check for required dependencies first
print("[RIFE] Checking dependencies...")
try:
    import torch
    import numpy as np
    print(f"[RIFE] PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"[RIFE] ERROR: PyTorch not available: {e}")
    sys.exit(1)

try:
    import einops
    print(f"[RIFE] einops: {einops.__version__}")
except ImportError:
    print("[RIFE] Installing einops...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "einops"])
    import einops

try:
    import packaging
    print(f"[RIFE] packaging: available")
except ImportError:
    print("[RIFE] Installing packaging...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
    import packaging

# Mock comfy.model_management BEFORE importing RIFE
print("[RIFE] Setting up comfy mock...")

class MockModelManagement:
    @staticmethod
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def soft_empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Create mock module structure
mock_comfy = types.ModuleType("comfy")
mock_model_management = types.ModuleType("comfy.model_management")

# Add functions to the mock module
mock_model_management.get_torch_device = MockModelManagement.get_torch_device
mock_model_management.soft_empty_cache = MockModelManagement.soft_empty_cache

mock_comfy.model_management = mock_model_management

# Inject mocks into sys.modules
sys.modules["comfy"] = mock_comfy
sys.modules["comfy.model_management"] = mock_model_management

print("[RIFE] Mock modules injected")

# Verify checkpoint exists
ckpt_path = REF_DIR / "ckpts" / "rife" / "rife47.pth"
if not ckpt_path.exists():
    print(f"[RIFE] WARNING: Checkpoint not found at {ckpt_path}")
    print("[RIFE] Will be downloaded on first use")

# Import RIFE
print("[RIFE] Importing RIFE_VFI...")
try:
    from vfi_models.rife import RIFE_VFI
    print("[RIFE] RIFE_VFI imported successfully")
except ImportError as e:
    print(f"[RIFE] ERROR importing RIFE: {e}")
    print(f"[RIFE] Sys path: {sys.path}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"[RIFE] ERROR during RIFE import: {e}")
    traceback.print_exc()
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
        print(f"DEBUG: Frame array count: {len(frame_arrays)}")
        if frame_arrays:
            print(f"DEBUG: First frame shape: {frame_arrays[0].shape}, dtype: {frame_arrays[0].dtype}")

        from video_generator import create_video_from_arrays

        output_path = Path(output_file)
        print(f"DEBUG: Calling create_video_from_arrays with output: {output_path}, fps: {fps}")
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
    import traceback

    parser = argparse.ArgumentParser(description="Interpolate video frames using RIFE AI")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing images")
    parser.add_argument("--output", "-o", default="video_rife_torch.mp4", help="Output video path")
    parser.add_argument("--multiplier", "-m", type=int, default=8, help="Frame multiplier (2, 4, 8, 16)")
    parser.add_argument("--fps", type=int, default=24, help="Output FPS")

    args = parser.parse_args()

    try:
        print(f"[RIFE] Starting interpolation")
        print(f"[RIFE] Input: {args.input}")
        print(f"[RIFE] Output: {args.output}")
        print(f"[RIFE] Multiplier: {args.multiplier}x, FPS: {args.fps}")
        success = interpolate_video(args.input, args.output, args.multiplier, args.fps)
        if success:
            print(f"[RIFE] Interpolation completed successfully")
        else:
            print(f"[RIFE] Interpolation failed")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[RIFE] FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
