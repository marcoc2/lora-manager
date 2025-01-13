import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable
import gc
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

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if os.path.basename(filename) != "modeling_florence2.py":
        return get_imports(filename)
    imports = get_imports(filename)
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
            
            # Redimensiona se necessário
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
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
                    repetition_penalty=1.5
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
