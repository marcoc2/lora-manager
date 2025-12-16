"""
Qwen2.5-VL Caption Generator using HuggingFace Transformers

Uses Qwen/Qwen2.5-VL-7B-Instruct for high-quality image captioning.
"""
import os
import math
import torch
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable
import gc


# Target ~0.25 MP for faster processing (512x512 = 262,144 pixels)
TARGET_MEGAPIXELS = 0.25
TARGET_PIXELS = int(TARGET_MEGAPIXELS * 1_000_000)  # 250,000 pixels


def resize_for_captioning(image: Image.Image, target_pixels: int = TARGET_PIXELS) -> Image.Image:
    """
    Resize image to approximately target_pixels while maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        target_pixels: Target total pixel count (default ~0.25 MP)

    Returns:
        Resized PIL Image (or original if already smaller)
    """
    width, height = image.size
    current_pixels = width * height

    if current_pixels <= target_pixels:
        # Image is already small enough
        return image

    # Calculate scale factor to reach target pixels
    # new_width * new_height = target_pixels
    # (width * scale) * (height * scale) = target_pixels
    # scale^2 = target_pixels / (width * height)
    scale = math.sqrt(target_pixels / current_pixels)

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Use LANCZOS for high quality downscaling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized


class QwenGenerator:
    """Qwen2.5-VL generator for image captioning using HuggingFace Transformers"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        print("Initializing QwenGenerator...")
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Prompts (can be customized by controller)
        self.system_prompt = "You are an expert AI image captioner used to create datasets for image diffusion models."
        self.user_prompt = "Describe this image in detail, including the subject, clothing, pose, action, and background environment. Provide a continuous description without bullet points or numbered lists."

    def _init_model(self):
        """Initialize the Qwen model on demand"""
        if self.model is None:
            try:
                print(f"Loading Qwen2.5-VL model: {self.model_name}")
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

                print("Loading processor...")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                print("Processor loaded successfully")

                print("Loading model...")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                print("Model loaded successfully")

                self.model.eval()
                print("Model initialization complete")

            except ImportError as e:
                raise ImportError(
                    f"Required packages not installed. Run: pip install transformers qwen-vl-utils\n{e}"
                )
            except Exception as e:
                print(f"Error during model initialization: {str(e)}")
                raise

    def generate_caption(self, image_path: Path,
                        progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Generate caption for a single image using Qwen2.5-VL

        Args:
            image_path: Path to the image file
            progress_callback: Optional callback for progress updates

        Returns:
            Generated caption string
        """
        try:
            print(f"\nProcessing image: {image_path}")
            self._init_model()

            # Open and process image
            print("Opening image...")
            image_path = Path(image_path)
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            print(f"Image opened. Original size: {original_size}")

            # Resize to ~0.25 MP for faster processing
            image = resize_for_captioning(image)
            if image.size != original_size:
                print(f"Resized to: {image.size} ({image.size[0] * image.size[1]:,} pixels)")

            # Prepare messages in Qwen2.5-VL format
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.user_prompt}
                    ]
                }
            ]

            # Apply chat template
            print("Preparing inputs...")
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            # Move to device
            inputs = inputs.to(self.model.device)
            print("Inputs prepared successfully")

            # Generate caption
            print("Generating caption...")
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                )

            # Decode output - get only the generated part
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]

            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            print(f"Caption generated: {caption[:100]}...")

            # Clean up GPU memory
            print("Cleaning up memory...")
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            if progress_callback:
                progress_callback(f"Generated caption for {image_path.name}")

            print("Caption generation complete")
            return caption.strip()

        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            if progress_callback:
                progress_callback(f"Error processing {image_path.name}: {str(e)}")
            raise

    def process_directory(self, images_dir: Path, captions_dir: Path,
                         prefix: str = "",
                         progress_callback: Optional[Callable[[str, int], None]] = None) -> Tuple[int, int]:
        """
        Process all images in a directory

        Args:
            images_dir: Directory containing images
            captions_dir: Directory to save captions
            prefix: Optional prefix to add to captions
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (processed_count, failed_count)
        """
        print(f"\nStarting directory processing...")
        print(f"Images directory: {images_dir}")
        print(f"Captions directory: {captions_dir}")
        print(f"Using prefix: '{prefix}'")

        captions_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        failed = 0

        # List all images
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            image_files.extend(images_dir.glob(ext))

        # Deduplicate (Windows case-insensitive)
        seen = set()
        unique_files = []
        for f in image_files:
            if f.name.lower() not in seen:
                seen.add(f.name.lower())
                unique_files.append(f)
        image_files = unique_files

        total_files = len(image_files)
        print(f"Found {total_files} images to process")

        for idx, img_path in enumerate(image_files):
            try:
                print(f"\nProcessing image {idx + 1}/{total_files}: {img_path.name}")

                # Generate caption
                caption = self.generate_caption(img_path)

                # Add prefix if specified
                if prefix:
                    caption = f"{prefix} {caption}"

                # Save caption
                caption_path = captions_dir / f"{img_path.stem}.txt"
                caption_path.write_text(caption, encoding='utf-8')
                print(f"Caption saved to: {caption_path}")

                processed += 1

                if progress_callback:
                    progress_callback(f"Processing {img_path.name}...", int((idx + 1) * 100 / total_files))

            except Exception as e:
                print(f"Failed to process {img_path.name}: {str(e)}")
                if progress_callback:
                    progress_callback(f"Failed to process {img_path.name}: {str(e)}", -1)
                failed += 1

            # Clean up GPU memory periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        print(f"\nDirectory processing complete.")
        print(f"Successfully processed: {processed}")
        print(f"Failed: {failed}")
        return processed, failed
