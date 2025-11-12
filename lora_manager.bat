@echo off
echo Iniciando Lora Manager...

REM Navegar para o diretório do sd-scripts e ativar o ambiente virtual
cd /d C:\Apps\sd-scripts
call .\venv\Scripts\activate.bat

REM Navegar para o diretório do lora-manager
cd /d F:\AppsCrucial\lora-manager\

REM Executar o aplicativo
python .\main.py

REM Manter a janela aberta em caso de erro
pause
