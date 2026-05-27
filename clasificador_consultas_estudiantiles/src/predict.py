"""Funciones de prediccion para usar el clasificador entrenado."""

from pathlib import Path

import joblib
import numpy as np

from src.preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "modelo_consultas.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizador_tfidf.pkl"

_model = None
_vectorizer = None


def load_artifacts():
    """Carga el modelo y el vectorizador una sola vez."""
    global _model, _vectorizer

    if _model is None or _vectorizer is None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "No se encontraron los artefactos del modelo. "
                "Ejecuta primero: python train_model.py"
            )
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)

    return _model, _vectorizer


def _probabilities_from_model(model, features):
    """Obtiene probabilidades incluso si algun modelo no expone predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[0]

    scores = model.decision_function(features)
    scores = np.asarray(scores).reshape(-1)
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def predict_category(texto: str, top_n: int = 3):
    """Predice categoria, confianza y top categorias probables."""
    if not texto or not str(texto).strip():
        raise ValueError("La consulta no puede estar vacia.")

    model, vectorizer = load_artifacts()
    clean_query = clean_text(texto)
    features = vectorizer.transform([clean_query])

    predicted_category = model.predict(features)[0]
    probabilities = _probabilities_from_model(model, features)
    classes = model.classes_

    category_probability = dict(zip(classes, probabilities))
    confidence = float(category_probability[predicted_category] * 100)

    top_indexes = np.argsort(probabilities)[::-1][:top_n]
    top_categories = [
        {
            "categoria": str(classes[index]),
            "confianza": round(float(probabilities[index] * 100), 2),
        }
        for index in top_indexes
    ]

    return {
        "categoria": str(predicted_category),
        "confianza": round(confidence, 2),
        "top_3": top_categories,
        "texto_limpio": clean_query,
    }
