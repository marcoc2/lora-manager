"""
Project Manager - Implementa o padrão "Manifesto Local" (lora_project.json)

Cada pasta de dataset pode conter um arquivo lora_project.json que persiste:
- Configurações de treinamento por modelo
- Trigger word global
- Histórico de treinamentos
- Artifact selecionado
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import json

from pydantic import BaseModel, Field, ValidationError
from PyQt6.QtCore import QObject, pyqtSignal, QTimer


# ============================================================================
# Pydantic Models
# ============================================================================

class ModelType(str, Enum):
    """Tipos de modelo suportados para treinamento"""
    SDXL = "sdxl"
    FLUX = "flux"
    QWEN = "qwen"
    ZIMAGE = "zimage"
    WAN = "wan"


class ProjectAssets(BaseModel):
    """Caminhos relativos dos assets do projeto"""
    source_images: str = "."
    captions: str = "./captions"
    processed: Optional[str] = None  # Ex: "./cropped_images_512"


class TrainingRun(BaseModel):
    """Registro de uma sessão de treinamento"""
    timestamp: datetime
    model_type: str  # Usando str para flexibilidade
    steps: int
    output_name: str
    output_path: Optional[str] = None
    config_snapshot: Dict[str, Any] = {}  # Snapshot dos parâmetros usados


class LoraProject(BaseModel):
    """Schema principal do manifesto de projeto"""
    version: str = "1.0"
    project_name: str
    description: Optional[str] = None
    trigger_word: Optional[str] = None
    preferred_model: str = "flux"

    # Configurações por modelo (persistem entre sessões)
    # Estrutura: {"flux": {...}, "qwen": {...}, "zimage": {...}}
    model_configs: Dict[str, Dict[str, Any]] = {}

    # Assets e estrutura
    assets: ProjectAssets = Field(default_factory=ProjectAssets)
    active_artifact: Optional[str] = None  # Último artifact selecionado

    # Histórico de treinamentos
    training_history: List[TrainingRun] = []

    # Metadados
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)

    class Config:
        # Permite serialização de datetime para JSON
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Project Manager
# ============================================================================

class ProjectManager(QObject):
    """
    Gerencia projetos de LoRA training.

    Responsabilidades:
    - Carregar/salvar lora_project.json
    - Auto-save com debounce
    - Notificar mudanças via sinais Qt
    """

    # Sinais
    project_loaded = pyqtSignal(object)  # Emite LoraProject ou None
    project_saved = pyqtSignal()
    project_modified = pyqtSignal()  # Emite quando há mudanças não salvas
    save_status_changed = pyqtSignal(str)  # "saving", "saved", "error"

    MANIFEST_FILENAME = "lora_project.json"
    AUTOSAVE_DELAY_MS = 500  # Debounce de 500ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: Optional[LoraProject] = None
        self._project_path: Optional[Path] = None
        self._has_unsaved_changes = False

        # Timer para auto-save com debounce
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._do_autosave)

    @property
    def project(self) -> Optional[LoraProject]:
        """Retorna o projeto atual"""
        return self._project

    @property
    def project_path(self) -> Optional[Path]:
        """Retorna o path da pasta do projeto"""
        return self._project_path

    @property
    def manifest_path(self) -> Optional[Path]:
        """Retorna o path completo do arquivo de manifesto"""
        if self._project_path:
            return self._project_path / self.MANIFEST_FILENAME
        return None

    @property
    def has_project(self) -> bool:
        """Verifica se há um projeto carregado"""
        return self._project is not None

    @property
    def has_unsaved_changes(self) -> bool:
        """Verifica se há mudanças não salvas"""
        return self._has_unsaved_changes

    # -------------------------------------------------------------------------
    # Métodos de Carregamento/Salvamento
    # -------------------------------------------------------------------------

    def is_project(self, path: Path) -> bool:
        """Verifica se uma pasta contém um projeto (tem lora_project.json)"""
        manifest = path / self.MANIFEST_FILENAME
        return manifest.exists()

    def load_project(self, path: Path) -> Optional[LoraProject]:
        """
        Carrega um projeto de uma pasta.

        Args:
            path: Caminho da pasta do dataset

        Returns:
            LoraProject se encontrado e válido, None caso contrário
        """
        self._project_path = path.absolute()
        manifest_path = self._project_path / self.MANIFEST_FILENAME

        if not manifest_path.exists():
            self._project = None
            self.project_loaded.emit(None)
            return None

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._project = LoraProject.model_validate(data)
            self._has_unsaved_changes = False
            self.project_loaded.emit(self._project)
            print(f"[ProjectManager] Loaded project: {self._project.project_name}")
            return self._project

        except ValidationError as e:
            print(f"[ProjectManager] Validation error loading project: {e}")
            self._project = None
            self.project_loaded.emit(None)
            return None
        except json.JSONDecodeError as e:
            print(f"[ProjectManager] JSON decode error: {e}")
            self._project = None
            self.project_loaded.emit(None)
            return None
        except Exception as e:
            print(f"[ProjectManager] Error loading project: {e}")
            self._project = None
            self.project_loaded.emit(None)
            return None

    def create_project(self, path: Path, project_name: Optional[str] = None) -> LoraProject:
        """
        Cria um novo projeto em uma pasta.

        Args:
            path: Caminho da pasta do dataset
            project_name: Nome do projeto (default: nome da pasta)

        Returns:
            LoraProject criado
        """
        self._project_path = path.absolute()

        if project_name is None:
            project_name = path.name

        self._project = LoraProject(
            project_name=project_name,
            created_at=datetime.now(),
            last_modified=datetime.now()
        )

        # Salva imediatamente
        self.save_project()
        self.project_loaded.emit(self._project)
        print(f"[ProjectManager] Created new project: {project_name}")
        return self._project

    def save_project(self) -> bool:
        """
        Salva o projeto atual no disco.

        Returns:
            True se salvou com sucesso, False caso contrário
        """
        if not self._project or not self._project_path:
            return False

        try:
            self.save_status_changed.emit("saving")

            # Atualiza timestamp de modificação
            self._project.last_modified = datetime.now()

            manifest_path = self._project_path / self.MANIFEST_FILENAME

            # Serializa para JSON
            data = self._project.model_dump(mode='json')

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            self._has_unsaved_changes = False
            self.save_status_changed.emit("saved")
            self.project_saved.emit()
            print(f"[ProjectManager] Project saved: {manifest_path}")
            return True

        except Exception as e:
            print(f"[ProjectManager] Error saving project: {e}")
            self.save_status_changed.emit("error")
            return False

    def close_project(self):
        """Fecha o projeto atual"""
        if self._has_unsaved_changes:
            self.save_project()

        self._project = None
        self._project_path = None
        self._has_unsaved_changes = False
        self.project_loaded.emit(None)

    # -------------------------------------------------------------------------
    # Auto-save com Debounce
    # -------------------------------------------------------------------------

    def schedule_autosave(self):
        """
        Agenda um auto-save com debounce.
        Chamado a cada alteração de parâmetro.
        """
        if not self._project:
            return

        self._has_unsaved_changes = True
        self.project_modified.emit()

        # Reinicia o timer (debounce)
        self._autosave_timer.stop()
        self._autosave_timer.start(self.AUTOSAVE_DELAY_MS)

    def _do_autosave(self):
        """Executa o auto-save após o debounce"""
        if self._has_unsaved_changes:
            self.save_project()

    # -------------------------------------------------------------------------
    # Métodos de Acesso/Modificação do Projeto
    # -------------------------------------------------------------------------

    def set_trigger_word(self, trigger_word: str):
        """Define a trigger word global do projeto"""
        if self._project:
            self._project.trigger_word = trigger_word if trigger_word else None
            self.schedule_autosave()

    def set_active_artifact(self, artifact_name: Optional[str]):
        """Define o artifact ativo (ex: 'cropped_images_512')"""
        if self._project:
            self._project.active_artifact = artifact_name
            self.schedule_autosave()

    def set_preferred_model(self, model_type: str):
        """Define o modelo preferido do projeto"""
        if self._project:
            self._project.preferred_model = model_type
            self.schedule_autosave()

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Retorna as configurações salvas para um tipo de modelo.

        Args:
            model_type: "flux", "qwen", "zimage", "sdxl", "wan"

        Returns:
            Dict com configurações ou {} se não houver
        """
        if self._project:
            return self._project.model_configs.get(model_type, {})
        return {}

    def set_model_config(self, model_type: str, config: Dict[str, Any]):
        """
        Salva as configurações de um modelo no projeto.

        Args:
            model_type: "flux", "qwen", "zimage", "sdxl", "wan"
            config: Dict com configurações do modelo
        """
        if self._project:
            self._project.model_configs[model_type] = config
            self.schedule_autosave()

    def add_training_run(self, model_type: str, steps: int, output_name: str,
                         output_path: Optional[str] = None,
                         config_snapshot: Optional[Dict] = None):
        """
        Adiciona um registro de treinamento ao histórico.

        Args:
            model_type: Tipo do modelo usado
            steps: Número de steps do treinamento
            output_name: Nome do arquivo de saída
            output_path: Caminho do arquivo de saída (opcional)
            config_snapshot: Snapshot dos parâmetros usados (opcional)
        """
        if self._project:
            run = TrainingRun(
                timestamp=datetime.now(),
                model_type=model_type,
                steps=steps,
                output_name=output_name,
                output_path=output_path,
                config_snapshot=config_snapshot or {}
            )
            self._project.training_history.append(run)
            self.schedule_autosave()

    def update_project_info(self, name: Optional[str] = None,
                           description: Optional[str] = None):
        """Atualiza informações básicas do projeto"""
        if self._project:
            if name is not None:
                self._project.project_name = name
            if description is not None:
                self._project.description = description
            self.schedule_autosave()

    # -------------------------------------------------------------------------
    # Utilitários
    # -------------------------------------------------------------------------

    def get_project_summary(self) -> Dict[str, Any]:
        """Retorna um resumo do projeto para exibição"""
        if not self._project:
            return {}

        return {
            "name": self._project.project_name,
            "trigger_word": self._project.trigger_word,
            "preferred_model": self._project.preferred_model,
            "active_artifact": self._project.active_artifact,
            "training_count": len(self._project.training_history),
            "created_at": self._project.created_at.strftime("%Y-%m-%d %H:%M"),
            "last_modified": self._project.last_modified.strftime("%Y-%m-%d %H:%M"),
        }
