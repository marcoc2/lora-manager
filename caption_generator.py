import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable

class CaptionGenerator:
    def __init__(self):
        self.processor = None
        self.model = None
        
    def _init_model(self):
        """Inicializa o modelo BLIP sob demanda"""
        if self.processor is None:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    def generate_caption(self, image_path: Path, 
                        progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Gera caption para uma única imagem
        
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
            inputs = self.processor(image, return_tensors="pt")
            
            # Gera caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    temperature=0.9,
                    top_p=0.9,
                )
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
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
        
        return processed, failed