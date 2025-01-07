# Lora Training Manager

Uma ferramenta gráfica para gerenciar datasets e treinamentos de modelos LoRA para Stable Diffusion, com integração aos scripts Kohya.

## Características

- Interface gráfica multiplataforma usando PyQt6
- Visualização e gerenciamento da estrutura do dataset em árvore
- Processamento automatizado de imagens:
  - Detecção facial e crop inteligente
  - Redimensionamento com preservação de proporção
  - Upscaling quando necessário
- Geração automática de captions usando BLIP
- Geração automática de dataset.toml para treinamento
- Integração com scripts Kohya

## Pré-requisitos

### Python
- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)

### Dependências do Sistema (Linux)
```bash
sudo apt-get install python3-pyqt6
sudo apt-get install libxcb-cursor0
sudo apt-get install xcb
sudo apt-get install python3-pyqt6.qtsvg
sudo apt-get install libxcb-xinerama0
sudo apt-get install libxcb-randr0
sudo apt-get install libxcb-xtest0
sudo apt-get install libxcb-shape0
sudo apt-get install libxcb-xkb1
```

### Dependências Python
```bash
pip install PyQt6 Pillow opencv-python transformers torch toml
```

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/lora-training-manager.git
cd lora-training-manager
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Execute o programa:
```bash
python main-gui.py
```

2. Selecione a pasta do seu dataset usando o botão "Select Dataset Folder"

3. Use o menu de contexto (botão direito) na TreeView para acessar as operações:
   - Process Images: Processa as imagens do dataset
   - Generate Captions: Gera captions usando BLIP
   - Generate dataset.toml: Cria o arquivo de configuração para treinamento
   - Analyze Dataset: Mostra estatísticas do dataset atual

## Estrutura do Projeto

```
lora-training-manager/
├── main-gui.py              # Interface gráfica principal
├── dataset_manager.py       # Módulo de gerenciamento de dataset
├── image_processor.py       # Processamento de imagens
├── caption_generator.py     # Geração de captions com BLIP
├── requirements.txt         # Dependências do projeto
└── README.md               # Este arquivo
```

## Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para enviar pull requests.

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a GNU General Public License v3.0 - veja o arquivo [LICENSE](LICENSE) para detalhes.

Copyright (C) 2024 

Este programa é software livre: você pode redistribuí-lo e/ou modificá-lo
sob os termos da GNU General Public License conforme publicada pela
Free Software Foundation, seja a versão 3 da Licença, ou
(a seu critério) qualquer versão posterior.

Este programa é distribuído na esperança de que seja útil,
mas SEM QUALQUER GARANTIA; sem mesmo a garantia implícita de
COMERCIALIZAÇÃO ou ADEQUAÇÃO A UM DETERMINADO FIM. Veja a
GNU General Public License para mais detalhes.

Você deve ter recebido uma cópia da GNU General Public License
junto com este programa. Se não, veja <https://www.gnu.org/licenses/>.