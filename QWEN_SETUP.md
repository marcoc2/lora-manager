# Qwen-Image LoRA Training Setup

Esta GUI agora suporta treinamento de LoRAs para Qwen-Image usando o Musubi Tuner.

## Requisitos

1. **Python 3.10+** com PyTorch 2.5.1+ (CUDA 12.x recomendado)
2. **Musubi Tuner** instalado
3. **Modelos base do Qwen-Image**

## Instalação do Musubi Tuner

```bash
# Instalar PyTorch com CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Clonar e instalar Musubi Tuner
git clone https://github.com/kohya-ss/musubi-tuner
cd musubi-tuner
pip install -e .

# Configurar accelerate
accelerate config
```

## Modelos Necessários

Baixe os seguintes arquivos do Qwen-Image:

### 1. DiT (Diffusion Transformer)
- `qwen_image_bf16.safetensors` ou `qwen_image_fp8_e4m3fn.safetensors`
- Local: [Hugging Face - Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)

### 2. Text Encoder (Qwen2.5-VL 7B)
- `qwen_2.5_vl_7b_fp8_scaled.safetensors` ou versão bf16
- Local: [Hugging Face - Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)

### 3. VAE
- `qwen_image_vae.safetensors`
- Local: [Hugging Face - Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)

## Como Usar na GUI

### 1. Preparar Dataset
1. Selecione sua pasta de dataset
2. Processe as imagens (redimensionar/croppar)
3. Gere captions usando Florence-2, Danbooru ou Janus-7B
4. Clique em **"Generate Qwen-Image dataset.toml"** para criar o arquivo de configuração específico

### 2. Configurar Treinamento
1. Vá para a aba **"Qwen-Image Training"**
2. Configure os caminhos dos modelos:
   - **DiT Model**: Caminho para o modelo qwen_image_*.safetensors
   - **Text Encoder**: Caminho para qwen_2.5_vl_7b_*.safetensors
   - **VAE Model**: Caminho para qwen_image_vae.safetensors
   - **Musubi Directory**: Pasta onde você clonou o musubi-tuner

### 3. Ajustar Parâmetros
- **Network Dimension**: 32 (recomendado)
- **Learning Rate**: 2e-4 (padrão)
- **Epochs**: 32 (ajuste conforme necessário)
- **Batch Size**: 1 (para economizar VRAM)
- **Blocks to Swap**: Use valores > 0 para economizar VRAM (ex: 36)

### 4. Opções Avançadas
- **FP8 for Text Encoder**: Marque para economizar VRAM
- **Mixed Precision**: bf16 (recomendado)
- **Gradient Checkpointing**: Marque para economizar VRAM
- **SDPA**: Marque para melhor performance

### 5. Cache (Opcional mas Recomendado)
Antes do treinamento, execute:
1. **Cache Latents**: Pré-processa as imagens
2. **Cache Text Encoder**: Pré-processa as legendas

### 6. Iniciar Treinamento
Clique em **"Start Training"** para adicionar à fila de treinamento.

## Economia de VRAM

Para GPUs com 12-16GB de VRAM:
- Marque **FP8 for Text Encoder**
- Configure **Blocks to Swap** para 36 ou menos
- Use **Batch Size** = 1
- **Network Dimension** = 16-32
- Marque **Gradient Checkpointing**

## Conversão de LoRA

Após o treinamento, use o botão **"Convert LoRA"** para converter para uso em:
- ComfyUI
- Diffusers
- Outros frameworks compatíveis

## Estrutura de Arquivos Gerada

```
dataset_folder/
├── cropped_images/
│   ├── cache_imgs/           # Cache de latents
│   ├── dataset.toml          # Configuração Qwen-específica
│   ├── image_001.png
│   ├── image_001.txt
│   └── ...
└── ...
```

## Formato do dataset.toml para Qwen

```toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
resolution = [768, 768]

[[datasets]]
is_image = true
image_directory = "/caminho/para/cropped_images"
cache_directory = "/caminho/para/cache_imgs"
num_repeats = 1
```

## Troubleshooting

### Erro de dtype mismatch
- Verifique se todos os modelos estão no mesmo dtype (bf16 ou fp8)
- Use `--device cpu` ao cachear text encoder com FP8

### Out of Memory (OOM)
- Aumente **Blocks to Swap**
- Ative **FP8 for Text Encoder**
- Reduza **Network Dimension**
- Use **Batch Size** = 1

### Performance
- Use **Mixed Precision**: bf16
- Ative **SDPA**
- Pre-cache latents e text encoder
- Use **Gradient Checkpointing**

## Recursos Adicionais

- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
- [Musubi Tuner GitHub](https://github.com/kohya-ss/musubi-tuner)
- [ComfyUI Qwen-Image Guide](https://docs.comfy.org/tutorials/image/qwen/qwen-image)