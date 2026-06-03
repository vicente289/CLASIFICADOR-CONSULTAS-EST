"""Explicabilidad ligera para el clasificador TF-IDF."""

from src.predict import load_artifacts
from src.preprocess import clean_text


def _get_class_index(model, category):
    classes = list(model.classes_)
    if category in classes:
        return classes.index(category)
    return 0


def explain_prediction(texto, categoria_modelo=None, top_n=8):
    """Devuelve palabras que mas aportan a la categoria sugerida.

    La explicacion usa el mismo vector TF-IDF de entrenamiento. Si el modelo
    final es Regresion Logistica, combina el peso TF-IDF de la palabra con el
    coeficiente aprendido para la categoria. Si se usa Naive Bayes, utiliza
    sus probabilidades logaritmicas como aproximacion.
    """
    model, vectorizer = load_artifacts()
    texto_limpio = clean_text(texto)
    features = vectorizer.transform([texto_limpio])
    feature_names = vectorizer.get_feature_names_out()
    active_indexes = features.nonzero()[1]

    if len(active_indexes) == 0:
        return {
            "texto_limpio": texto_limpio,
            "categoria_modelo": categoria_modelo,
            "palabras_clave": [],
            "metodo": "TF-IDF",
        }

    if categoria_modelo is None:
        categoria_modelo = str(model.predict(features)[0])

    class_index = _get_class_index(model, categoria_modelo)
    tfidf_values = features.toarray()[0]
    contributions = []

    for index in active_indexes:
        tfidf_weight = float(tfidf_values[index])

        if hasattr(model, "coef_"):
            model_weight = float(model.coef_[class_index][index])
            contribution = tfidf_weight * model_weight
            method = "TF-IDF x coeficiente de Regresión Logística"
        elif hasattr(model, "feature_log_prob_"):
            model_weight = float(model.feature_log_prob_[class_index][index])
            contribution = tfidf_weight * model_weight
            method = "TF-IDF x probabilidad de Naive Bayes"
        else:
            model_weight = tfidf_weight
            contribution = tfidf_weight
            method = "Peso TF-IDF"

        contributions.append(
            {
                "palabra": str(feature_names[index]),
                "peso_tfidf": round(tfidf_weight, 4),
                "aporte": round(contribution, 4),
                "peso_modelo": round(model_weight, 4),
            }
        )

    positive_contributions = [item for item in contributions if item["aporte"] > 0]
    ranked_items = positive_contributions or contributions
    ranked_items = sorted(ranked_items, key=lambda item: item["aporte"], reverse=True)

    return {
        "texto_limpio": texto_limpio,
        "categoria_modelo": categoria_modelo,
        "palabras_clave": ranked_items[:top_n],
        "metodo": method,
    }
