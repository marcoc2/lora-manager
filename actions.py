from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QDialog
from PyQt6.QtGui import QStandardItem
from PyQt6.QtCore import Qt
from pathlib import Path
import toml
import traceback

from gui_components import SuffixInputDialog, TomlConfigDialog, CaptionConfigDialog
from caption_generator import CaptionGenerator
from danbooru_generator import DanbooruGenerator

class DatasetActionsMixin:
    def toggle_face_detection(self):
        if self.face_detection.isChecked():
            self.face_detection.setText("Face Detection: ON")
        else:
            self.face_detection.setText("Face Detection: OFF")
    
    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path = Path(folder).absolute()
            self.populate_tree_view(self.dataset_path)
            self.update_status()
    
    def populate_tree_view(self, path):
        self.tree_model.clear()
        self.tree_model.setHorizontalHeaderLabels(['Dataset Structure'])
        
        root_item = QStandardItem(str(path))
        self.tree_model.appendRow(root_item)
        
        def add_directory_contents(parent_item, dir_path):
            try:
                for item_path in sorted(Path(dir_path).iterdir()):
                    item = QStandardItem(item_path.name)
                    parent_item.appendRow(item)
                    
                    if item_path.is_dir():
                        add_directory_contents(item, item_path)
            except Exception as e:
                print(f"Error accessing {dir_path}: {e}")
        
        add_directory_contents(root_item, path)
        self.tree_view.expandAll()
    
    def update_status(self):
        if self.dataset_path:
            self.status_label.setText(f"Dataset selected: {self.dataset_path}")
        else:
            self.status_label.setText("No dataset selected")
    
    def get_image_files(self, directory):
        """Obter todos os arquivos de imagem suportados em um diretório,
        incluindo diferentes casos de extensão e formatos adicionais."""
        # Lista para armazenar todos os arquivos de imagem encontrados
        image_files = []
        
        # Extensões de imagem comuns em diferentes casos
        extensions = [
            "*.jpg", "*.jpeg", "*.png", "*.webp",  # minúsculas
            "*.JPG", "*.JPEG", "*.PNG", "*.WEBP",  # maiúsculas
            "*.Jpg", "*.Jpeg", "*.Png", "*.Webp"   # misto
        ]
        
        # Procurar por cada tipo de extensão
        for ext in extensions:
            image_files.extend(directory.glob(ext))
        
        return image_files
    
    def process_images(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            input_dir = self.dataset_path
            output_dir = self.dataset_path / "cropped_images"
            
            # Encontrar todas as imagens com extensões diversas usando nosso método
            our_image_files = self.get_image_files(input_dir)
            n_files = len(our_image_files)
            
            # Verificar quais arquivos o método original encontraria
            original_glob_files = list(input_dir.glob("*.[jp][pn][g]"))
            n_original_files = len(original_glob_files)
            
            # Criar listas de nomes para comparação
            our_filenames = [f.name for f in our_image_files]
            original_filenames = [f.name for f in original_glob_files]
            
            # Encontrar diferenças
            missing_files = [f for f in our_filenames if f not in original_filenames]
            
            # Mostrar diagnóstico
            diagnostic_message = (
                f"Diagnóstico:\n"
                f"- Nosso método encontrou: {n_files} imagens\n"
                f"- Método original encontraria: {n_original_files} imagens\n\n"
                f"Arquivos que seriam ignorados pelo método original:\n"
                f"{', '.join(missing_files)}"
            )
            
            QMessageBox.information(self, "Diagnóstico", diagnostic_message)
            
            if n_files == 0:
                QMessageBox.warning(self, "Warning", "No images found in the input directory!")
                return
            
            # SOLUÇÃO TEMPORÁRIA: Modificar o comportamento do ImageProcessor
            # Vamos sobrecarregar o método process_directory para processar nossa lista de arquivos
            
            # Guardar método original para restaurar depois
            original_process_dir = self.image_processor.process_directory
            
            # Definir uma função wrapper customizada
            def custom_process_directory(input_dir, output_dir, target_size):
                processed = 0
                failed = 0
                
                # Garantir que o diretório de saída existe
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Processar cada arquivo que encontramos
                for image_path in our_image_files:
                    try:
                        # Chamar o método de processamento de arquivo único do ImageProcessor
                        # (Observe que você pode precisar ajustar isto com base na implementação real)
                        output_path = output_dir / f"{image_path.stem}.png"
                        if self.image_processor.process_image(image_path, output_path, target_size):
                            processed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        failed += 1
                
                return processed, failed
            
            # Substituir o método temporariamente
            try:
                # Esta é uma solução rápida - pode não funcionar dependendo da implementação do ImageProcessor
                self.image_processor.process_directory = custom_process_directory
                
                target_size = (self.crop_width.value(), self.crop_height.value())
                self.image_processor.use_face_detection = self.face_detection.isChecked()
                
                processed, failed = self.image_processor.process_directory(
                    input_dir, output_dir, target_size
                )
                
                QMessageBox.information(self, "Success", 
                    f"Processing complete!\n\nSuccessfully processed: {processed}\nFailed: {failed}")
            finally:
                # Restaurar o método original
                self.image_processor.process_directory = original_process_dir
            
            self.tree_model.clear()
            self.populate_tree_view(self.dataset_path)
            self.tree_view.expandAll()
            self.update_status()
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing images: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
    
    def generate_captions(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
            
        try:
            cropped_dir = self.dataset_path / "cropped_images"
            if not cropped_dir.exists():
                QMessageBox.warning(self, "Warning", "Please process images first!")
                return
            
            # Usar o método melhorado para encontrar imagens
            image_files = self.get_image_files(cropped_dir)
            n_files = len(image_files)
            
            if n_files == 0:
                QMessageBox.warning(self, "Warning", "No images found in cropped_images folder!")
                return
            
            config_dialog = CaptionConfigDialog(self)
            if config_dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            config = config_dialog.get_values()
            
            progress = QProgressDialog("Generating captions...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.show()
            
            def update_progress(message: str, value: int):
                if value >= 0:
                    progress.setLabelText(message)
                    progress.setValue(value)
            
            captions_dir = cropped_dir / "captions"
            
            # Escolhe o gerador apropriado
            generator = None
            if config['method'] == "Florence-2":
                generator = CaptionGenerator()
            elif config['method'] == "Danbooru":
                model_type = config.get('model_type', 'vit')
                generator = DanbooruGenerator(model_type=model_type)
            else:  # Janus-7B
                from janus_generator import JanusGenerator
                generator = JanusGenerator()
                if config['janus_context']:
                    if config['replace_prompt']:
                        generator.set_prompt(config['janus_context'])
                    else:
                        generator.add_context(config['janus_context'])
            
            processed, failed = generator.process_directory(
                cropped_dir, 
                captions_dir,
                prefix=config['prefix'],
                progress_callback=update_progress
            )
            
            progress.close()
            
            QMessageBox.information(self, "Success", 
                f"Caption generation complete!\n\nSuccessfully processed: {processed}\nFailed: {failed}")
            
            self.tree_model.clear()
            self.populate_tree_view(self.dataset_path)
            self.tree_view.expandAll()
            self.update_status()
            
        except Exception as e:
            error_msg = f"Error generating captions:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
    
    def generate_toml(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return
        
        try:
            dialog = TomlConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                config = dialog.get_values()
                
                cropped_dir = self.dataset_path / "cropped_images"
                cropped_dir.mkdir(parents=True, exist_ok=True)
                
                toml_data = {
                    "general": {
                        "shuffle_caption": False,
                        "caption_extension": ".txt",
                        "keep_tokens": 1
                    },
                    "datasets": [{
                        "resolution": config['resolution'],
                        "batch_size": 1,
                        "keep_tokens": 1,
                        "subsets": [{
                            "image_dir": str(cropped_dir.resolve()),
                            "class_tokens": config['class_tokens'],
                            "num_repeats": config['num_repeats']
                        }]
                    }]
                }
                
                toml_path = cropped_dir / "dataset.toml"
                with open(toml_path, "w", encoding="utf-8") as f:
                    toml.dump(toml_data, f)
                
                QMessageBox.information(self, "Success", "dataset.toml generated successfully!")
                self.tree_model.clear()
                self.populate_tree_view(self.dataset_path)
                self.update_status()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating dataset.toml: {str(e)}")
    
    def rename_and_convert_images(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset folder first!")
            return

        dialog = SuffixInputDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        suffix = dialog.get_suffix()
        if not suffix:
            QMessageBox.warning(self, "Warning", "Suffix cannot be empty!")
            return

        image_dir = self.dataset_path / "cropped_images"
        if not image_dir.exists():
            QMessageBox.warning(self, "Warning", "Cropped images directory does not exist!")
            return

        # Usar o método melhorado para encontrar imagens
        image_files = self.get_image_files(image_dir)
        
        if not image_files:
            QMessageBox.warning(self, "Warning", "No images found for renaming and conversion!")
            return

        converted_count = 0
        from PIL import Image

        for idx, image_path in enumerate(sorted(image_files), 1):
            try:
                new_name = f"{image_path.stem}{suffix}_{str(idx).zfill(3)}.png"
                new_path = image_dir / new_name

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img.save(new_path, "PNG")

                if image_path.suffix.lower() != ".png":
                    image_path.unlink()  # Remove o arquivo original

                converted_count += 1
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to process {image_path.name}: {e}")

        QMessageBox.information(self, "Success", 
            f"Renamed and converted {converted_count} images with suffix '{suffix}' successfully!")
        
        self.tree_model.clear()
        self.populate_tree_view(self.dataset_path)
        self.update_status()
    
    def analyze_dataset(self):
        # Implementação dummy para análise do dataset.
        QMessageBox.information(self, "Analyze Dataset", "Dataset analysis not implemented yet.")