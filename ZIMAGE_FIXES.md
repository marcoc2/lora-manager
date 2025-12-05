# Correções necessárias para Z-Image Training no Lora Manager

## Resumo dos Problemas

O wrapper Qt não estava treinando corretamente por 3 motivos principais:

1. **Model path incorreto** - apontando para arquivo `.safetensors` único ao invés do repositório HuggingFace completo
2. **Adapter path incompleto** - faltando o caminho completo do HuggingFace
3. **Falta `disable_sampling`** - causando erro `enable_gqa` durante o treinamento
4. **Caminho do toolkit errado** - `reference/ai-toolkit` ao invés de `reference/ai-toolkit-original`

---

## 1. Correções em `zimage_training_config.json`

### ANTES:
```json
{
    "model_path": "E:/flux_models/z_image_turbo_bf16.safetensors",
    "adapter_path": "./models/zimage_turbo_training_adapter_v1.safetensors",
    ...
}
```

### DEPOIS:
```json
{
    "model_path": "Tongyi-MAI/Z-Image-Turbo",
    "adapter_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors",
    ...
}
```

**Arquivo corrigido criado**: `zimage_training_config_fixed.json`

---

## 2. Correções em `controllers/training_controller.py`

### Linha 395 - Adicionar `disable_sampling`

**ANTES:**
```python
                    "dtype": "bf16"
                },
```

**DEPOIS:**
```python
                    "dtype": "bf16",
                    "disable_sampling": True  # FIXED: Disable sampling to avoid enable_gqa error
                },
```

### Linha 435 - Corrigir caminho do toolkit

**ANTES:**
```python
toolkit_path = Path("reference/ai-toolkit").absolute()
```

**DEPOIS:**
```python
toolkit_path = Path("reference/ai-toolkit-original").absolute()
```

### Linha 441 - Adicionar definição do run_script

**ANTES:**
```python
command = f'"{sys.executable}" {run_script} {yaml_path}'
```

**DEPOIS:**
```python
run_script = toolkit_path / "run.py"
command = f'"{sys.executable}" {run_script} {yaml_path}'
```

---

## 3. Correções em `debug_zimage.py`

Todas as mesmas correções acima se aplicam. Arquivo corrigido criado: `debug_zimage_fixed.py`

---

## 4. Correções em `zimage_widgets_ui.py`

### Valores padrão no `__init__` (linha 52 e 64):

**ANTES:**
```python
self.model_path.setText("Tongyi-MAI/Z-Image-Turbo")  # OK ✓
self.adapter_path.setText("./models/zimage_turbo_training_adapter_v1.safetensors")  # ERRADO ✗
```

**DEPOIS:**
```python
self.model_path.setText("Tongyi-MAI/Z-Image-Turbo")  # OK ✓
self.adapter_path.setText("ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors")  # CORRETO ✓
```

---

## 5. Correção adicional: diffusers library

O erro `enable_gqa` requer patch na biblioteca diffusers:

```bash
python -c "
import sys
file_path = r'C:\Apps\sd-scripts\venv\lib\site-packages\diffusers\models\attention_dispatch.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'enable_gqa=enable_gqa,' not in line:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Successfully removed enable_gqa parameter')
"
```

**Nota**: Este patch já foi aplicado no seu sistema durante nossos testes.

---

## Como testar as correções

1. Atualize `zimage_training_config.json` com os valores corretos
2. Aplique as correções em `controllers/training_controller.py`
3. Aplique as correções em `zimage_widgets_ui.py`
4. Teste executando um treinamento pela UI do Qt

---

## Exemplo de configuração funcional

Use como referência o arquivo que funcionou:
`F:/AppsCrucial/lora-manager/reference/ai-toolkit-original/config/examples/zimage_training_example.yaml`

Principais diferenças:
- ✅ `name_or_path: "Tongyi-MAI/Z-Image-Turbo"` (HF repo completo)
- ✅ `assistant_lora_path: "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors"` (caminho HF completo)
- ✅ `disable_sampling: true` na seção `train`
- ✅ Caminho correto para o toolkit: `reference/ai-toolkit-original`

---

## Arquivos de referência criados

1. `debug_zimage_fixed.py` - Versão corrigida do debug script
2. `zimage_training_config_fixed.json` - Config corrigido
3. `ZIMAGE_FIXES.md` - Este documento

---

## Resumo das mudanças necessárias

| Arquivo | Linha | Mudança |
|---------|-------|---------|
| `zimage_training_config.json` | 2 | `model_path`: usar HF repo completo |
| `zimage_training_config.json` | 3 | `adapter_path`: usar caminho HF completo |
| `controllers/training_controller.py` | 395 | Adicionar `"disable_sampling": True` |
| `controllers/training_controller.py` | 435 | Corrigir para `reference/ai-toolkit-original` |
| `controllers/training_controller.py` | 441 | Adicionar definição `run_script` |
| `zimage_widgets_ui.py` | 64 | Alterar adapter_path padrão para HF completo |
| `diffusers` library | N/A | Remover parâmetro `enable_gqa` (já aplicado) |
