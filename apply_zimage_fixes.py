#!/usr/bin/env python3
"""
Script para aplicar as correções no training_controller.py
Execute: python apply_zimage_fixes.py
"""

import re
from pathlib import Path

def fix_training_controller():
    """Aplica correções no training_controller.py"""

    file_path = Path("controllers/training_controller.py")

    if not file_path.exists():
        print(f"❌ Arquivo não encontrado: {file_path}")
        return False

    # Ler conteúdo
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    fixes_applied = []

    # Fix 1: Adicionar disable_sampling na linha ~395
    if '"dtype": "bf16"\n                    },\n                    "model":' in content:
        content = content.replace(
            '"dtype": "bf16"\n                    },\n                    "model":',
            '"dtype": "bf16",\n                        "disable_sampling": True  # FIXED: Disable sampling to avoid enable_gqa error\n                    },\n                    "model":'
        )
        fixes_applied.append("✅ Adicionado 'disable_sampling: True' na seção train")

    # Fix 2: Corrigir caminho do toolkit
    if 'Path("reference/ai-toolkit")' in content:
        content = content.replace(
            'Path("reference/ai-toolkit")',
            'Path("reference/ai-toolkit-original")'
        )
        fixes_applied.append("✅ Corrigido caminho do toolkit para 'reference/ai-toolkit-original'")

    # Fix 3: Adicionar definição do run_script e usar venv correto
    if 'command = f\'"{{sys.executable}}" {{run_script}} {{yaml_path}}\'' in content:
        old_block = '''        # Build command
        # Assuming ai-toolkit is in reference/ai-toolkit
        # We need to run run.py from that directory
        toolkit_path = Path("reference/ai-toolkit-original").absolute()

        # We need to run this in a python environment that has ai-toolkit dependencies
        # Using sys.executable ensures we use the same python that is running this app
        # which is likely the one where the user installed the requirements.

        command = f'"{sys.executable}" {run_script} {yaml_path}\''''

        new_block = '''        # Build command
        # Assuming ai-toolkit is in reference/ai-toolkit-original
        # We need to run run.py from that directory
        toolkit_path = Path("reference/ai-toolkit-original").absolute()
        run_script = toolkit_path / "run.py"

        # Use the specific venv python if possible
        venv_python = Path("C:/Apps/sd-scripts/venv/Scripts/python.exe")
        if venv_python.exists():
            python_exe = str(venv_python)
        else:
            python_exe = sys.executable

        command = f'"{python_exe}" "{run_script}" "{yaml_path}"\''''

        content = content.replace(old_block, new_block)
        fixes_applied.append("✅ Adicionada definição de 'run_script' e uso do venv correto")

    # Verificar se houve mudanças
    if content == original_content:
        print("⚠️  Nenhuma correção aplicada. Verifique se as correções já foram aplicadas ou se o formato do arquivo mudou.")
        return False

    # Fazer backup
    backup_path = file_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"💾 Backup criado: {backup_path}")

    # Salvar arquivo corrigido
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n🎉 Correções aplicadas com sucesso!")
    for fix in fixes_applied:
        print(f"   {fix}")

    return True

def fix_zimage_widgets():
    """Corrige o valor padrão do adapter_path em zimage_widgets_ui.py"""

    file_path = Path("zimage_widgets_ui.py")

    if not file_path.exists():
        print(f"❌ Arquivo não encontrado: {file_path}")
        return False

    # Ler conteúdo
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Fix: Adapter path padrão
    if './models/zimage_turbo_training_adapter_v1.safetensors' in content:
        content = content.replace(
            'self.adapter_path.setText("./models/zimage_turbo_training_adapter_v1.safetensors")',
            'self.adapter_path.setText("ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors")'
        )

        # Fazer backup
        backup_path = file_path.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"💾 Backup criado: {backup_path}")

        # Salvar arquivo corrigido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print("✅ Corrigido adapter_path padrão em zimage_widgets_ui.py")
        return True
    else:
        print("⚠️  Adapter path já está correto ou formato mudou")
        return False

def main():
    print("=" * 60)
    print("  Script de Correções para Z-Image Training")
    print("=" * 60)
    print()

    success = True

    print("📝 Aplicando correções em training_controller.py...")
    if not fix_training_controller():
        success = False

    print()
    print("📝 Aplicando correções em zimage_widgets_ui.py...")
    if not fix_zimage_widgets():
        success = False

    print()
    print("=" * 60)
    if success:
        print("✅ Todas as correções foram aplicadas com sucesso!")
        print()
        print("Próximos passos:")
        print("1. Reinicie o aplicativo Qt")
        print("2. Verifique se o zimage_training_config.json tem os valores corretos:")
        print("   - model_path: 'Tongyi-MAI/Z-Image-Turbo'")
        print("   - adapter_path: 'ostris/zimage_turbo_training_adapter/...'")
        print("3. Tente treinar novamente")
    else:
        print("⚠️  Algumas correções não foram aplicadas.")
        print("Verifique os arquivos manualmente seguindo ZIMAGE_FIXES.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
