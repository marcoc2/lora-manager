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
        Processa uma única imagem
        
        Args:
            image_path: Caminho da imagem de entrada
            output_path: Caminho para salvar a imagem processada
            target_size: Tamanho desejado (width, height)
            progress_callback: Função para reportar progresso
            
        Returns:
            bool: True se processou com sucesso, False caso contrário
        """
        try:
            # Abre imagem com PIL
            with Image.open(image_path) as pil_image:
                pil_image = pil_image.convert("RGB")
                
                # Verifica se precisa fazer upscaling
                if pil_image.width < target_size[0] or pil_image.height < target_size[1]:
                    ratio = max(target_size[0] / pil_image.width, 
                              target_size[1] / pil_image.height)
                    new_size = (int(pil_image.width * ratio), 
                              int(pil_image.height * ratio))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                # Se usar detecção facial
                if self.use_face_detection:
                    # Converte para OpenCV
                    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    faces = self.detect_faces(opencv_image)
                    
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        center_x = x + w // 2
                        center_y = y + h // 2
                    else:
                        center_x = pil_image.width // 2
                        center_y = pil_image.height // 2
                else:
                    center_x = pil_image.width // 2
                    center_y = pil_image.height // 2

                # Calcula coordenadas do crop
                left = max(center_x - target_size[0] // 2, 0)
                top = max(center_y - target_size[1] // 2, 0)
                right = min(left + target_size[0], pil_image.width)
                bottom = min(top + target_size[1], pil_image.height)

                # Faz o crop
                cropped_image = pil_image.crop((left, top, right, bottom))

                # Se o crop for menor que o tamanho desejado, centraliza em fundo preto
                if cropped_image.size != target_size:
                    new_image = Image.new("RGB", target_size, (0, 0, 0))
                    paste_x = (target_size[0] - cropped_image.width) // 2
                    paste_y = (target_size[1] - cropped_image.height) // 2
                    new_image.paste(cropped_image, (paste_x, paste_y))
                    cropped_image = new_image

                # Salva resultado
                cropped_image.save(output_path, "PNG")

                if progress_callback:
                    progress_callback(f"Processed {image_path.name}")
                    
                return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error processing {image_path.name}: {str(e)}")
            return False

    def process_directory(self, input_dir: Path, output_dir: Path, 
                         target_size: Tuple[int, int],
                         progress_callback: Optional[Callable[[str], None]] = None) -> tuple:
        """
        Processa todas as imagens em um diretório
        
        Returns:
            tuple: (total_processed, total_failed)
        """
        # Cria diretório de saída se não existir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lista todas as imagens
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            image_files.extend(input_dir.glob(ext))
        
        total_processed = 0
        total_failed = 0
        
        for img_path in image_files:
            output_path = output_dir / f"{img_path.stem}.png"
            
            if self.process_image(img_path, output_path, target_size, progress_callback):
                total_processed += 1
            else:
                total_failed += 1
        
        return total_processed, total_failed
