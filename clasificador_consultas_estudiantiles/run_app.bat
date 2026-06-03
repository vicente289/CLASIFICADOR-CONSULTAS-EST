@echo off
cd /d "%~dp0"
py -3.10 -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo No se pudo iniciar con py -3.10. Intentando con python...
    python -m streamlit run app.py
)
pause
