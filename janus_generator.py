import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import VLChatProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable
import gc
import warnings
import transformers
transformers.utils.TRUST_REMOTE_CODE = True

class JanusGenerator:
    def __init__(self):
        print("Initializing JanusGenerator...")
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.default_prompt = "Please describe the image in a continuous paragraph, without using line breaks, bullet points, or numbered lists. "\
        "Provide a detailed and coherent description of the scene, objects, and any relevant details in a single block of text."
        self.custom_prompt = None
        
    def set_prompt(self, prompt: str):
        """Define um prompt personalizado completo"""
        print(f"Setting custom prompt: {prompt}")
        self.custom_prompt = prompt
        
    def add_context(self, context: str):
        """Adiciona contexto ao prompt padrão"""
        print(f"Adding context to default prompt: {context}")
        self.custom_prompt = f"{self.default_prompt} {context}"
    
    def _init_model(self):
        """Inicializa o modelo Janus sob demanda"""
        if self.processor is None:
            try:
                print("Starting model initialization...")
                model_path = "deepseek-ai/Janus-Pro-7B"
                print(f"Loading model from: {model_path}")
                
                config = AutoConfig.from_pretrained(model_path)
                print("Loaded config")
                language_config = config.language_config
                language_config._attn_implementation = 'eager'
                
                print("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    language_config=language_config,
                    trust_remote_code=True
                )
                print("Model loaded successfully")
                
                if torch.cuda.is_available():
                    print("Moving model to GPU...")
                    self.model = self.model.to(torch.bfloat16).cuda()
                else:
                    print("Moving model to CPU...")
                    self.model = self.model.to(torch.float16)
                
                print("Loading processor...")
                self.processor = VLChatProcessor.from_pretrained(model_path)
                self.tokenizer = self.processor.tokenizer
                print("Processor loaded successfully")
                
                self.model.eval()
                print("Model initialization complete")
            except Exception as e:
                print(f"Error during model initialization: {str(e)}")
                raise
    
    def generate_caption(self, image_path: Path, 
                        progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Gera caption para uma única imagem usando Janus
        """
        try:
            print(f"\nProcessing image: {image_path}")
            self._init_model()
            
            # Abre e processa a imagem
            print("Opening image...")
            image = Image.open(image_path).convert('RGB')
            print(f"Image opened successfully. Size: {image.size}")
            
            # Redimensiona se necessário
            max_size = 768  # Janus trabalha melhor com imagens 768x768
            if max(image.size) > max_size:
                print(f"Resizing image from {image.size}", end="")
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                print(f" to {image.size}")
            
            # Prepara a conversação
            prompt = self.custom_prompt if self.custom_prompt else self.default_prompt
            print(f"Using prompt: {prompt}")
            
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # Prepara inputs como no Gradio
            pil_images = [Image.fromarray(np.array(image))]
            print("Preparing inputs...")
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
            print("Inputs prepared successfully")
            
            print("Preparing input embeddings...")
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            print("Input embeddings prepared")
            
            # Gera caption usando os mesmos parâmetros do Gradio
            print("Generating caption...")
            with torch.no_grad():
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=True,
                    use_cache=True,
                    temperature=0.1,
                    top_p=0.95,
                )
                
                caption = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                print(f"Caption generated: {caption[:100]}...")
            
            # Limpa memória GPU
            print("Cleaning up memory...")
            del prepare_inputs, outputs, inputs_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            if progress_callback:
                progress_callback(f"Generated caption for {image_path.name}")
            
            print("Caption generation complete")
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            if progress_callback:
                progress_callback(f"Error processing {image_path.name}: {str(e)}")
            raise
    
    def process_directory(self, images_dir: Path, captions_dir: Path,
                         prefix: str = "",
                         progress_callback: Optional[Callable[[str, int], None]] = None) -> Tuple[int, int]:
        """
        Processa todas as imagens em um diretório
        """
        print(f"\nStarting directory processing...")
        print(f"Images directory: {images_dir}")
        print(f"Captions directory: {captions_dir}")
        print(f"Using prefix: '{prefix}'")
        
        captions_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        failed = 0
        
        # Lista todas as imagens
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(images_dir.glob(ext))
        
        total_files = len(image_files)
        print(f"Found {total_files} images to process")
        
        for idx, img_path in enumerate(image_files):
            try:
                print(f"\nProcessing image {idx + 1}/{total_files}: {img_path.name}")
                # Gera caption
                caption = self.generate_caption(img_path)
                
                # Adiciona prefixo se especificado
                if prefix:
                    caption = f"{prefix} {caption}"
                
                # Salva caption
                caption_path = captions_dir / f"{img_path.stem}.txt"
                caption_path.write_text(caption)
                print(f"Caption saved to: {caption_path}")
                
                processed += 1
                
                if progress_callback:
                    progress_callback(f"Processing {img_path.name}...", int((idx + 1) * 100 / total_files))
                
            except Exception as e:
                print(f"Failed to process {img_path.name}: {str(e)}")
                if progress_callback:
                    progress_callback(f"Failed to process {img_path.name}: {str(e)}", -1)
                failed += 1
            
            # Limpa memória GPU periodicamente
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"\nDirectory processing complete.")
        print(f"Successfully processed: {processed}")
        print(f"Failed: {failed}")
        return processed, failed