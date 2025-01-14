import torch
import timm
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, List
from PIL import Image
from torch.nn import functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from timm.data import create_transform, resolve_data_config

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]

class DanbooruGenerator:
    MODEL_REPOS = {
        "vit": "SmilingWolf/wd-vit-tagger-v3",
        "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
        "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    }
    
    def __init__(self, model_type="vit", general_threshold=0.35, character_threshold=0.75):
        """
        Inicializa o WD14 Tagger
        
        Args:
            model_type: Tipo de modelo ('vit', 'swinv2' ou 'convnext')
            general_threshold: Limiar para tags gerais
            character_threshold: Limiar para tags de personagens
        """
        if model_type not in self.MODEL_REPOS:
            raise ValueError(f"Modelo deve ser um de: {list(self.MODEL_REPOS.keys())}")
            
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.labels = None
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_labels(self) -> LabelData:
        """Carrega e organiza as tags do modelo"""
        repo_id = self.MODEL_REPOS[self.model_type]
        csv_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
        
        df = pd.read_csv(csv_path, usecols=["name", "category"])
        return LabelData(
            names=df["name"].tolist(),
            rating=list(np.where(df["category"] == 9)[0]),
            general=list(np.where(df["category"] == 0)[0]),
            character=list(np.where(df["category"] == 4)[0])
        )
    
    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Garante que a imagem está em RGB"""
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        return image
    
    def _pad_square(self, image: Image.Image) -> Image.Image:
        """Pad a imagem para um quadrado"""
        w, h = image.size
        px = max(image.size)
        canvas = Image.new("RGB", (px, px), (255, 255, 255))
        canvas.paste(image, ((px - w) // 2, (px - h) // 2))
        return canvas
    
    def _init_model(self):
        """Inicializa o modelo sob demanda"""
        if self.model is None:
            repo_id = self.MODEL_REPOS[self.model_type]
            
            # Carrega o modelo
            self.model = timm.create_model("hf-hub:" + repo_id).eval()
            state_dict = timm.models.load_state_dict_from_hf(repo_id)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            
            # Cria o transform
            self.transform = create_transform(**resolve_data_config(
                self.model.pretrained_cfg, model=self.model
            ))
            
            # Carrega as labels
            self.labels = self._load_labels()
    
    def _process_tags(self, probs: torch.Tensor) -> Tuple[str, Dict, Dict, Dict]:
        """
        Processa as probabilidades em tags organizadas
        
        Returns:
            Tuple contendo:
            - caption: string com todas as tags
            - ratings: dict com ratings
            - char_tags: dict com tags de personagens
            - general_tags: dict com tags gerais
        """
        probs = list(zip(self.labels.names, probs.numpy()))
        
        # Processa ratings
        rating_tags = dict([probs[i] for i in self.labels.rating])
        
        # Processa tags gerais
        general_tags = [probs[i] for i in self.labels.general]
        general_tags = dict([x for x in general_tags if x[1] > self.general_threshold])
        general_tags = dict(sorted(general_tags.items(), key=lambda x: x[1], reverse=True))
        
        # Processa tags de personagens
        char_tags = [probs[i] for i in self.labels.character]
        char_tags = dict([x for x in char_tags if x[1] > self.character_threshold])
        char_tags = dict(sorted(char_tags.items(), key=lambda x: x[1], reverse=True))
        
        # Combina as tags num caption
        combined_names = list(general_tags.keys())
        combined_names.extend(list(char_tags.keys()))
        caption = ", ".join(combined_names)
        
        return caption, rating_tags, char_tags, general_tags
    
    def generate_tags(self, image_path: str | Path,
                     progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Gera tags no estilo Danbooru para uma imagem
        
        Args:
            image_path: Caminho da imagem
            progress_callback: Função opcional para reportar progresso
            
        Returns:
            str: Tags no formato Danbooru
        """
        try:
            self._init_model()
            
            # Carrega e pré-processa a imagem
            image = Image.open(image_path)
            image = self._ensure_rgb(image)
            image = self._pad_square(image)
            
            # Aplica as transformações do modelo
            inputs = self.transform(image).unsqueeze(0)
            inputs = inputs[:, [2, 1, 0]]  # RGB para BGR
            inputs = inputs.to(self.device)
            
            # Faz a inferência
            with torch.inference_mode():
                outputs = self.model(inputs)
                outputs = F.sigmoid(outputs)
                outputs = outputs.cpu()
            
            # Processa as tags
            caption, ratings, char_tags, general_tags = self._process_tags(outputs.squeeze(0))
            
            if progress_callback:
                progress_callback(f"Generated tags for {image_path}")
                
            return caption
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error processing {image_path}: {str(e)}")
            raise
    
    def process_directory(self, images_dir: Path, tags_dir: Path,
                         prefix: str = "",
                         progress_callback: Optional[Callable[[str, int], None]] = None) -> Tuple[int, int]:
        """
        Processa todas as imagens em um diretório
        
        Args:
            images_dir: Diretório com as imagens
            tags_dir: Diretório para salvar os arquivos de tags
            prefix: Prefixo opcional para adicionar às tags
            progress_callback: Função para reportar progresso
        """
        tags_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        failed = 0
        
        # Lista todas as imagens
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(Path(images_dir).glob(ext))
        
        total_files = len(image_files)
        
        for idx, img_path in enumerate(image_files):
            try:
                # Gera tags
                tags = self.generate_tags(img_path)
                
                # Adiciona prefixo se especificado
                if prefix:
                    tags = f"{prefix} {tags}"
                
                # Salva tags
                tags_path = tags_dir / f"{img_path.stem}.txt"
                tags_path.write_text(tags)
                
                processed += 1
                
                if progress_callback:
                    progress_callback(f"Processing {img_path.name}...", 
                                   int((idx + 1) * 100 / total_files))
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Failed to process {img_path.name}: {str(e)}", -1)
                failed += 1
        
        return processed, failed