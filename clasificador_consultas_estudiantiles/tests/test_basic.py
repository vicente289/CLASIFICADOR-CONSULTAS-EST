from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.preprocess import clean_text


EXPECTED_CATEGORIES = {
    "Inscripciones",
    "Horarios",
    "Pagos",
    "Notas",
    "Trámites",
    "Plataforma virtual",
    "Becas",
    "Exámenes",
    "Otros",
}


def test_dataset_exists():
    assert (ROOT_DIR / "dataset_consultas.csv").exists()


def test_dataset_has_required_columns():
    dataset = pd.read_csv(ROOT_DIR / "dataset_consultas.csv", encoding="utf-8")
    assert list(dataset.columns) == ["consulta", "categoria"]


def test_dataset_has_expected_categories():
    dataset = pd.read_csv(ROOT_DIR / "dataset_consultas.csv", encoding="utf-8")
    assert set(dataset["categoria"].unique()) == EXPECTED_CATEGORIES


def test_clean_text_returns_valid_text():
    cleaned = clean_text(" Hola!!! ¿NO puedo pagar la inscripción?   ")
    assert isinstance(cleaned, str)
    assert cleaned == cleaned.lower()
    assert "no" in cleaned
    assert "pagar" in cleaned
    assert "inscripcion" in cleaned
