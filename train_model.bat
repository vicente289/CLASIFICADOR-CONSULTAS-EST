@echo off
cd /d "%~dp0clasificador_consultas_estudiantiles"
py -3.10 train_model.py
if errorlevel 1 (
    echo.
    echo No se pudo entrenar con py -3.10. Intentando con python...
    python train_model.py
)
pause
