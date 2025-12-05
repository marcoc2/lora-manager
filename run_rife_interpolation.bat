@echo off
set "VENV_PYTHON=C:\Apps\sd-scripts\venv\Scripts\python.exe"
set "SCRIPT_NAME=interpolate_rife_torch.py"
set "INSTALL_DIR=f:\AppsCrucial\lora-manager"

echo ===================================================
echo      Interpolador RIFE (via PyTorch)
echo ===================================================

:: 1. Check if script is in current folder
if exist "%~dp0%SCRIPT_NAME%" (
    set "SCRIPT_PATH=%~dp0%SCRIPT_NAME%"
) else (
    :: 2. Check default installation folder
    if exist "%INSTALL_DIR%\%SCRIPT_NAME%" (
        set "SCRIPT_PATH=%INSTALL_DIR%\%SCRIPT_NAME%"
    ) else (
        echo [ERRO] Nao encontrei o script %SCRIPT_NAME%
        echo Nem na pasta atual, nem em %INSTALL_DIR%
        pause
        exit /b
    )
)

echo Usando script: %SCRIPT_PATH%
echo.

"%VENV_PYTHON%" "%SCRIPT_PATH%"

pause
