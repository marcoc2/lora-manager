#!/usr/bin/env python3
"""
Teste para verificar se conseguimos executar scripts em venvs específicos
"""
import subprocess
import sys
from pathlib import Path

def test_venv_execution():
    print("=== Testing venv execution from GUI ===")
    
    # Teste 1: Python direto do venv sd-scripts (o que você estava usando)
    sdscripts_python = Path("C:/Apps/sd-scripts/venv/Scripts/python.exe")
    
    print("Testing different venv paths...")
    test_paths = [
        "C:/Apps/sd-scripts/venv/Scripts/python.exe",
        # Se você tem outros venvs, adicione aqui
    ]
    
    if sdscripts_python.exists():
        print(f"\n1. Testing direct python path: {sdscripts_python}")
        try:
            result = subprocess.run([
                str(sdscripts_python),
                "-c", "import sys; print('Python path:', sys.executable); print('Packages:', [p for p in sys.modules.keys() if 'torch' in p or 'accelerate' in p][:5])"
            ], capture_output=True, text=True, timeout=10)
            
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Return code:", result.returncode)
            
        except Exception as e:
            print("ERROR:", e)
    else:
        print(f"1. sd-scripts python not found at {sdscripts_python}")
    
    # Teste 2: Verificar se accelerate funciona
    if sdscripts_python.exists():
        print(f"\n2. Testing accelerate command:")
        try:
            result = subprocess.run([
                str(sdscripts_python),
                "-m", "accelerate", "--help"
            ], capture_output=True, text=True, timeout=10)
            
            print("Accelerate available:", "usage: accelerate" in result.stdout.lower())
            print("Return code:", result.returncode)
            
        except Exception as e:
            print("ERROR:", e)
    
    # Teste 3: Verificar sistema atual
    print(f"\n3. Current system:")
    print("Current Python:", sys.executable)
    try:
        import torch
        print("PyTorch version:", torch.__version__)
    except:
        print("PyTorch: Not available")
    
    try:
        import accelerate
        print("Accelerate version:", accelerate.__version__)
    except:
        print("Accelerate: Not available")

if __name__ == "__main__":
    test_venv_execution()