"""Punto de entrada desde la raiz del repositorio.

Permite ejecutar:
    streamlit run app.py

sin tener que entrar manualmente a la carpeta del proyecto.
"""

from pathlib import Path
import runpy
import sys


PROJECT_DIR = Path(__file__).resolve().parent / "clasificador_consultas_estudiantiles"
PROJECT_APP = PROJECT_DIR / "app.py"

if not PROJECT_APP.exists():
    raise FileNotFoundError(f"No se encontro la aplicacion principal: {PROJECT_APP}")

sys.path.insert(0, str(PROJECT_DIR))
runpy.run_path(str(PROJECT_APP), run_name="__main__")
