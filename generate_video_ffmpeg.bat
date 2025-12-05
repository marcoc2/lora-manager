@echo off
setlocal EnableDelayedExpansion

set "OUTPUT_FILE=video_preview.mp4"
set "FPS=3"
set "DURATION=0.3333333333"

echo ===================================================
echo      Gerador de Video via FFMPEG (3 FPS)
echo ===================================================

:: Check FFMPEG
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO CRITICO] FFMPEG nao encontrado no sistema!
    echo Por favor, instale o FFMPEG e adicione ao PATH.
    pause
    exit /b
)

if exist list.txt del list.txt

echo.
echo 1. Buscando imagens (ordenadas por data)...

set count=0
for /f "delims=" %%f in ('dir /b /od *.png *.jpg *.jpeg *.webp 2^>nul') do (
    echo file '%%f' >> list.txt
    echo duration !DURATION! >> list.txt
    set /a count+=1
    echo Found: %%f
)

echo.
echo ---------------------------------------------------
echo Total de imagens encontradas: !count!
echo ---------------------------------------------------

if !count! equ 0 (
    echo [ERRO] Nenhuma imagem encontrada nesta pasta!
    pause
    exit /b
)

echo.
echo 2. Iniciando FFMPEG...
echo Comando: ffmpeg -f concat -safe 0 -i list.txt -vsync vfr -pix_fmt yuv420p "%OUTPUT_FILE%" -y
echo.

ffmpeg -f concat -safe 0 -i list.txt -vsync vfr -pix_fmt yuv420p "%OUTPUT_FILE%" -y

if exist "%OUTPUT_FILE%" (
    echo.
    echo ===================================================
    echo [SUCESSO] Video criado: %OUTPUT_FILE%
    echo Tamanho: 
    for %%I in ("%OUTPUT_FILE%") do echo %%~zI bytes
    echo ===================================================
    del list.txt
) else (
    echo.
    echo [ERRO] O arquivo de video nao foi criado.
    echo Verifique se as imagens sao validas.
)

pause
