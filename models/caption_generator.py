import os
import math
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
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
    """
    width, height = image.size
    current_pixels = width * height

    if current_pixels <= target_pixels:
        return image

    scale = math.sqrt(target_pixels / current_pixels)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import warnings
import transformers
transformers.utils.TRUST_REMOTE_CODE = True


# Monkey patch a função de verificação do transformers para sempre retornar True
def _always_true(*args, **kwargs):
    return True

transformers.utils.hub._is_true = _always_true
transformers.utils.hub.is_remote_url = _always_true
transformers.utils.hub.has_file = _always_true

# Substituir a função original
transformers.utils.hub._is_true = _always_true

# Fix for Florence-2 SDPA error: Monkey patch PreTrainedModel.get_correct_attn_implementation
# This bypasses the check entirely if it fails
from transformers.modeling_utils import PreTrainedModel

if not hasattr(PreTrainedModel, "_patched_attn_impl"):
    original_get_attn = PreTrainedModel.get_correct_attn_implementation
    
    def patched_get_attn(self, *args, **kwargs):
        try:
            return original_get_attn(self, *args, **kwargs)
        except AttributeError:
            # Fallback if _supports_sdpa is missing
            return "eager"
            
    PreTrainedModel.get_correct_attn_implementation = patched_get_attn
    PreTrainedModel._patched_attn_impl = True

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if os.path.basename(filename) != "modeling_florence2.py":
        return get_imports(filename)
    imports = get_imports(filename)
    # Remove flash_attn only if it's in the list (safe remove)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

class CaptionGenerator:
    def __init__(self, model_version="base"):
        self.processor = None
        self.model = None
        self.model_version = model_version
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _init_model(self):
        """Inicializa o modelo Florence-2 sob demanda"""
        if self.processor is None:
            identifier = f"microsoft/Florence-2-{self.model_version}"
            
            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                self.model = AutoModelForCausalLM.from_pretrained(
                    identifier,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    force_download=False,
                    resume_download=True,
                    local_files_only=False
                ).to(self.device)
                
                # Fix for AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'
                # Try patching via sys.modules
                import sys
                if self.model.__module__ in sys.modules:
                    mod = sys.modules[self.model.__module__]
                    if hasattr(mod, "Florence2ForConditionalGeneration"):
                        mod.Florence2ForConditionalGeneration._supports_sdpa = False
                        print("Patched Florence2ForConditionalGeneration class in module")
                
                # Also patch instance
                self.model._supports_sdpa = False
                
                print(f"DEBUG: Model type: {type(self.model)}", flush=True)
                print(f"DEBUG: Has _supports_sdpa: {hasattr(self.model, '_supports_sdpa')}", flush=True)
                print(f"DEBUG: _supports_sdpa value: {getattr(self.model, '_supports_sdpa', 'MISSING')}", flush=True)
                
                self.processor = AutoProcessor.from_pretrained(
                    identifier,
                    trust_remote_code=True,
                    force_download=False,
                    resume_download=True,
                    local_files_only=False
                )
            
            self.model.eval()
    
    def generate_caption(self, image_path: Path, 
                        progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Gera caption para uma única imagem usando Florence-2
        
        Args:
            image_path: Caminho da imagem
            progress_callback: Função opcional para reportar progresso
            
        Returns:
            str: Caption gerado
        """
        try:
            self._init_model()

            # Abre e processa a imagem
            image = Image.open(image_path).convert('RGB')

            # Redimensiona para ~0.25 MP para processamento mais rápido
            image = resize_for_captioning(image)

            # Prepara inputs
            task_prompt = '<MORE_DETAILED_CAPTION>'
            inputs = self.processor(
                text=task_prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move para GPU com tipos corretos
            inputs['input_ids'] = inputs['input_ids'].to(self.device, dtype=torch.long)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device, dtype=torch.long)
            inputs['pixel_values'] = inputs['pixel_values'].to(self.device, dtype=torch.float16)
            
            # Gera caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pixel_values=inputs['pixel_values'],
                    max_new_tokens=512,
                    num_beams=5,
                    do_sample=False,
                    length_penalty=1.0,
                    repetition_penalty=1.5,
                    use_cache=False # Fix for AttributeError: 'NoneType' object has no attribute 'shape'
                )
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(image.width, image.height)
                )
                caption = parsed_answer[task_prompt]
            
            # Limpa memória GPU
            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
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
        Processa todas as imagens em um diretório
        
        Args:
            images_dir: Diretório com as imagens
            captions_dir: Diretório para salvar os captions
            prefix: Prefixo a ser adicionado no início de cada caption
            progress_callback: Função para reportar progresso (mensagem, valor)
        """
        captions_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        failed = 0
        
        # Lista todas as imagens
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(images_dir.glob(ext))
        
        total_files = len(image_files)
        
        for idx, img_path in enumerate(image_files):
            try:
                # Gera caption
                caption = self.generate_caption(img_path)
                
                # Adiciona prefixo se especificado
                if prefix:
                    caption = f"{prefix} {caption}"
                
                # Salva caption
                caption_path = captions_dir / f"{img_path.stem}.txt"
                caption_path.write_text(caption)
                
                processed += 1
                
                if progress_callback:
                    progress_callback(f"Processing {img_path.name}...", int((idx + 1) * 100 / total_files))
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Failed to process {img_path.name}: {str(e)}", -1)
                failed += 1
            
            # Limpa memória GPU periodicamente
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return processed, failed
