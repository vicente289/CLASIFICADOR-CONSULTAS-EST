"""Registro local de retroalimentacion para mejorar el prototipo."""

from csv import DictReader, DictWriter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = BASE_DIR / "feedback_consultas.csv"

FEEDBACK_COLUMNS = [
    "fecha",
    "consulta",
    "texto_limpio",
    "categoria_modelo",
    "categoria_final",
    "confianza",
    "requiere_revision",
    "top_3",
    "clasificacion_correcta",
    "categoria_corregida",
    "observacion",
]


def save_feedback(
    consulta,
    result,
    clasificacion_correcta,
    categoria_corregida="",
    observacion="",
    output_path=None,
):
    """Guarda una fila de retroalimentacion en CSV."""
    output = Path(output_path) if output_path else FEEDBACK_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output.exists()

    row = {
        "fecha": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "consulta": str(consulta).strip(),
        "texto_limpio": result.get("texto_limpio", ""),
        "categoria_modelo": result.get("categoria_modelo", result.get("categoria", "")),
        "categoria_final": result.get("categoria_final", result.get("categoria", "")),
        "confianza": result.get("confianza", ""),
        "requiere_revision": result.get("requiere_revision", False),
        "top_3": "; ".join(
            f"{item.get('categoria')}: {item.get('confianza')}%"
            for item in result.get("top_3", [])
        ),
        "clasificacion_correcta": bool(clasificacion_correcta),
        "categoria_corregida": categoria_corregida,
        "observacion": str(observacion).strip(),
    }

    with output.open("a", newline="", encoding="utf-8") as file:
        writer = DictWriter(file, fieldnames=FEEDBACK_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return output


def load_feedback_summary(output_path=None):
    """Resume la retroalimentacion guardada hasta el momento."""
    output = Path(output_path) if output_path else FEEDBACK_PATH
    if not output.exists():
        return {
            "total": 0,
            "correctas": 0,
            "correcciones": 0,
            "revision_manual": 0,
        }

    with output.open("r", newline="", encoding="utf-8") as file:
        rows = list(DictReader(file))

    correctas = sum(row.get("clasificacion_correcta") == "True" for row in rows)
    revision_manual = sum(row.get("requiere_revision") == "True" for row in rows)

    return {
        "total": len(rows),
        "correctas": correctas,
        "correcciones": len(rows) - correctas,
        "revision_manual": revision_manual,
    }
