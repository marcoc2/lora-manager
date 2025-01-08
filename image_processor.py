import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Callable

class ImageProcessor:
    def __init__(self, use_face_detection: bool = True):
        self.use_face_detection = use_face_detection
        self._face_cascade = None
        
    def _init_face_cascade(self):
        """Inicializa o detector facial sob demanda"""
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, image: np.ndarray) -> list:
        """Detecta rostos em uma imagem usando OpenCV"""
        self._init_face_cascade()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                                  minNeighbors=5, minSize=(30, 30))
        return faces

    def process_image(self, image_path: Path, output_path: Path, 
                    target_size: Tuple[int, int],
                    progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Processa uma única imagem com redimensionamento inteligente e/ou crop.
        - Se a imagem for quadrada e maior que o target, faz apenas resize
        - Se a imagem não for quadrada, redimensiona mantendo a menor dimensão igual ao target
        e depois faz crop do excesso da dimensão maior
        """
        try:
            # Abre imagem com PIL
            with Image.open(image_path) as pil_image:
                pil_image = pil_image.convert("RGB")
                
                width, height = pil_image.size
                target_width, target_height = target_size
                
                # Se a imagem for quadrada (ou quase quadrada, com margem de 5%)
                if abs(width - height) <= min(width, height) * 0.05:
                    # Faz apenas resize se for maior que o target
                    if width > target_width:
                        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
                else:
                    # Imagem não é quadrada
                    # Calcula a proporção baseada na menor dimensão
                    min_dim = min(width, height)
                    target_min = min(target_width, target_height)
                    scale = target_min / min_dim
                    
                    # Redimensiona mantendo proporção
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Se após o resize ainda precisar de crop
                    if new_width != target_width or new_height != target_height:
                        # Centraliza o crop
                        left = (new_width - target_width) // 2 if new_width > target_width else 0
                        top = (new_height - target_height) // 2 if new_height > target_height else 0
                        right = left + target_width
                        bottom = top + target_height
                        
                        # Faz o crop
                        pil_image = pil_image.crop((left, top, right, bottom))
                        
                        # Se ainda precisar de padding (caso raro, mas possível)
                        if pil_image.size != target_size:
                            new_image = Image.new("RGB", target_size, (0, 0, 0))
                            paste_x = (target_width - pil_image.width) // 2
                            paste_y = (target_height - pil_image.height) // 2
                            new_image.paste(pil_image, (paste_x, paste_y))
                            pil_image = new_image

                # Salva resultado
                pil_image.save(output_path, "PNG")

                if progress_callback:
                    progress_callback(f"Processed {image_path.name}")
                    
                return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error processing {image_path.name}: {str(e)}")
            return False


    def process_directory(self, input_dir: Path, output_dir: Path, 
                        target_size: Tuple[int, int],
                        progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[int, int]:
        """
        Processa todas as imagens em um diretório.

        Args:
            input_dir: Diretório com as imagens de entrada.
            output_dir: Diretório para salvar as imagens processadas.
            target_size: Dimensão final desejada (largura, altura).
            progress_callback: Função para reportar progresso.

        Returns:
            Tuple[int, int]: Número de imagens processadas e falhas.
        """
        # Cria diretório de saída se não existir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Lista todos os arquivos de imagem suportados
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_files.extend(input_dir.glob(ext))

        if not image_files:
            return 0, 0

        total_processed = 0
        total_failed = 0

        for image_path in image_files:
            output_path = output_dir / f"{image_path.stem}.png"
            success = self.process_image(image_path, output_path, target_size, progress_callback)
            if success:
                total_processed += 1
            else:
                total_failed += 1

        return total_processed, total_failed


