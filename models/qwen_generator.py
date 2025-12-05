import os
import base64
from pathlib import Path
from typing import Tuple, Optional, Callable
import gc

class QwenGenerator:
    def __init__(self, model_path="models/Qwen3-VL-4B-Instruct-UD-Q8_K_XL.gguf"):
        self.model_path = model_path
        self.llm = None
        self.chat_handler = None
        
    def _init_model(self):
        """Initialize the Qwen model on demand"""
        if self.llm is None:
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import Llava15ChatHandler
            except ImportError:
                raise ImportError("llama-cpp-python is not installed. Please install it with CUDA support.")

            # Check if model exists
            if not os.path.exists(self.model_path):
                # Try relative to current working directory
                if os.path.exists(os.path.join(os.getcwd(), self.model_path)):
                    self.model_path = os.path.join(os.getcwd(), self.model_path)
                else:
                    raise FileNotFoundError(f"Model file not found at {self.model_path}")

            # Initialize model
            # Note: Qwen VL models often use the Llava handler structure in llama-cpp-python
            # If a separate clip model is needed, it should be handled here. 
            # Assuming embedded clip for now based on user request.
            
            # Check for separate mmproj file if needed (not standard for single-file GGUF but possible)
            # For now, we assume the main GGUF contains everything or auto-detects.
            
            try:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            except ImportError:
                # Fallback or error if too old
                raise ImportError("Your llama-cpp-python version does not support Qwen2.5-VL. Please upgrade.")

            # Qwen25VLChatHandler also needs clip_model_path, usually the same file for GGUF
            self.chat_handler = Qwen25VLChatHandler(clip_model_path=self.model_path)
            
            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=self.chat_handler,
                n_gpu_layers=0, # Force CPU to avoid access violation crashes
                n_ctx=4096, # Context window
                verbose=False
            )
            
    def image_to_base64(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def generate_caption(self, image_path: Path, 
                        progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Generate caption for a single image using Qwen3-VL
        """
        try:
            self._init_model()
            
            image_b64 = self.image_to_base64(image_path)
            image_url = f"data:image/jpeg;base64,{image_b64}"
            
            system_prompt = "You are an expert AI image captioner used to create datasets for image diffusion models."
            user_prompt = "Describe this image in detail focusing on the subject, clothes, action, and background environment."
            
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            caption = response["choices"][0]["message"]["content"]
            
            if progress_callback:
                progress_callback(f"Generated caption for {image_path.name}")
                
            return caption
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error processing {image_path.name}: {str(e)}")
            raise

    def process_directory(self, images_dir: Path, captions_dir: Path,
                         prefix: str = "",
                         progress_callback: Optional[Callable[[str, int], None]] = None) -> Tuple[int, int]:
        """
        Process all images in a directory
        """
        captions_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        failed = 0
        
        # List all images
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            image_files.extend(images_dir.glob(ext))
        
        total_files = len(image_files)
        
        for idx, img_path in enumerate(image_files):
            try:
                # Generate caption
                caption = self.generate_caption(img_path)
                
                # Add prefix if specified
                if prefix:
                    caption = f"{prefix} {caption}"
                
                # Save caption
                caption_path = captions_dir / f"{img_path.stem}.txt"
                caption_path.write_text(caption, encoding='utf-8')
                
                processed += 1
                
                if progress_callback:
                    progress_callback(f"Processing {img_path.name}...", int((idx + 1) * 100 / total_files))
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Failed to process {img_path.name}: {str(e)}", -1)
                failed += 1
                
        return processed, failed
