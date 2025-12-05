import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
import types

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
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    files = sorted([
        f for f in Path(folder_path).glob('*') 
        if f.suffix.lower() in extensions
    ], key=lambda x: x.stat().st_mtime)
    return files

def interpolate_video(input_folder, output_file, multiplier=8, fps=24):
    print(f"Initializing RIFE (Multiplier: {multiplier}x)...")
    
    # Ensure checkpoints directory exists
    ckpt_dir = REF_DIR / "ckpts" / "rife"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    rife = RIFE_VFI()
    
    images = load_images(input_folder)
    if not images:
        print("No images found!")
        return

    print(f"Found {len(images)} images.")
    
    # Load images into tensor
    # RIFE expects [N, H, W, C] or [N, C, H, W]?
    # vfi_utils.preprocess_frames does: n h w c -> n c h w
    # So we should provide n h w c (standard cv2/numpy)
    
    frames_list = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to 0-1? Comfy usually expects 0-1 float tensors
        img = img.astype(np.float32) / 255.0
        frames_list.append(img)
        
    frames_tensor = torch.from_numpy(np.stack(frames_list))
    
    print("Starting interpolation...")
    try:
        # RIFE_VFI.vfi returns (out_tensor,)
        # It handles batching and caching internally
        with torch.no_grad():
            out_tuple = rife.vfi(
                ckpt_name="rife47.pth",
                frames=frames_tensor,
                multiplier=multiplier,
                clear_cache_after_n_frames=10,
                fast_mode=True,
                ensemble=True
            )
            out_frames = out_tuple[0] # [N, H, W, C] (postprocess_frames returns cpu tensor)
            
        print(f"Generated {len(out_frames)} frames.")
        
        # Save video
        height, width = out_frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        for i in range(len(out_frames)):
            frame = out_frames[i].numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
            
        video.release()
        print(f"Video saved to {output_file}")
        
    except Exception as e:
        print(f"Error during interpolation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".", help="Input folder containing images")
    parser.add_argument("--output", default="video_rife_torch.mp4", help="Output video path")
    parser.add_argument("--multiplier", type=int, default=16, help="Frame multiplier")
    parser.add_argument("--fps", type=int, default=16, help="Output FPS")
    
    args = parser.parse_args()
    
    interpolate_video(args.input, args.output, args.multiplier, args.fps)
