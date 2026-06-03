"""Punto de entrada de entrenamiento desde la raiz del repositorio."""

from pathlib import Path
import runpy
import sys


PROJECT_DIR = Path(__file__).resolve().parent / "clasificador_consultas_estudiantiles"
TRAIN_SCRIPT = PROJECT_DIR / "train_model.py"

if not TRAIN_SCRIPT.exists():
    raise FileNotFoundError(f"No se encontro el script de entrenamiento: {TRAIN_SCRIPT}")

sys.path.insert(0, str(PROJECT_DIR))
runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
