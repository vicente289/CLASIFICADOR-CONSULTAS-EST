"""Funciones de preprocesamiento para consultas estudiantiles en español."""

import re


ACCENT_TRANSLATION = str.maketrans(
    {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "Á": "a",
        "É": "e",
        "Í": "i",
        "Ó": "o",
        "Ú": "u",
        "Ü": "u",
    }
)


def to_lowercase(text: str) -> str:
    """Convierte el texto a minusculas de forma segura."""
    return str(text).lower()


def normalize_basic_text(text: str) -> str:
    """Normaliza acentos frecuentes y espacios extremos."""
    text = str(text).translate(ACCENT_TRANSLATION)
    return text.strip()


def remove_unnecessary_signs(text: str) -> str:
    """Elimina signos y caracteres que no aportan a la clasificacion."""
    return re.sub(r"[^a-zA-Z0-9ñÑ\s]", " ", str(text))


def remove_duplicate_spaces(text: str) -> str:
    """Reduce espacios multiples a un solo espacio."""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_text(text: str) -> str:
    """Limpia una consulta conservando palabras importantes.

    No elimina stopwords para mantener terminos como "no", "pago",
    "nota", "beca", "plataforma" o "inscripcion", que son utiles
    para clasificar consultas estudiantiles.
    """
    text = to_lowercase(text)
    text = normalize_basic_text(text)
    text = remove_unnecessary_signs(text)
    text = remove_duplicate_spaces(text)
    return text
