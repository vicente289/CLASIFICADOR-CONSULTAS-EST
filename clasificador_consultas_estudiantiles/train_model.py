"""Entrena y evalua modelos clasicos para consultas estudiantiles."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

from src.evaluate import save_confusion_matrix, save_metrics_report
from src.preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_consultas.csv"
MODEL_PATH = BASE_DIR / "modelo_consultas.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizador_tfidf.pkl"
REPORT_PATH = BASE_DIR / "metrics_report.txt"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
RANDOM_STATE = 29

CATEGORIES = [
    "Inscripciones",
    "Horarios",
    "Pagos",
    "Notas",
    "Trámites",
    "Plataforma virtual",
    "Becas",
    "Exámenes",
    "Otros",
]


def load_dataset():
    """Carga y valida el dataset del proyecto."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH, encoding="utf-8")
    expected_columns = {"consulta", "categoria"}
    if set(dataset.columns) != expected_columns:
        raise ValueError("El dataset debe tener exactamente las columnas consulta,categoria")

    missing_categories = set(CATEGORIES) - set(dataset["categoria"].unique())
    if missing_categories:
        raise ValueError(f"Faltan categorias en el dataset: {missing_categories}")

    dataset = dataset.dropna(subset=["consulta", "categoria"]).copy()
    dataset["consulta_limpia"] = dataset["consulta"].apply(clean_text)
    return dataset


def train_and_compare_models(x_train, y_train, x_test, y_test):
    """Entrena Regresion Logistica y Naive Bayes, luego compara resultados."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    models = {
        "Regresion Logistica Multiclase": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=8,
            random_state=RANDOM_STATE,
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    results = []
    trained_models = {}
    for model_name, model in models.items():
        model.fit(x_train_tfidf, y_train)
        predictions = model.predict(x_test_tfidf)
        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)

        trained_models[model_name] = model
        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "predictions": predictions,
            }
        )

    best_result = max(results, key=lambda item: (item["f1_macro"], item["accuracy"]))
    best_model_name = best_result["model"]
    best_model = trained_models[best_model_name]

    return best_model_name, best_model, vectorizer, best_result, results


def main():
    dataset = load_dataset()
    x = dataset["consulta_limpia"]
    y = dataset["categoria"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_model_name, best_model, vectorizer, best_result, results = train_and_compare_models(
        x_train,
        y_train,
        x_test,
        y_test,
    )

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    report_results = [
        {
            "model": result["model"],
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
        }
        for result in results
    ]
    labels = [category for category in CATEGORIES if category in y_test.unique()]

    save_metrics_report(
        y_test,
        best_result["predictions"],
        labels=labels,
        output_path=REPORT_PATH,
        model_name=best_model_name,
        comparison_results=report_results,
    )
    save_confusion_matrix(
        y_test,
        best_result["predictions"],
        labels=labels,
        output_path=CONFUSION_MATRIX_PATH,
    )

    print("Entrenamiento completado")
    print(f"Consultas del dataset: {len(dataset)}")
    print(f"Modelo seleccionado: {best_model_name}")
    for result in report_results:
        print(
            f"- {result['model']}: "
            f"accuracy={result['accuracy']:.4f}, f1_macro={result['f1_macro']:.4f}"
        )
    print(f"Modelo guardado en: {MODEL_PATH.name}")
    print(f"Vectorizador guardado en: {VECTORIZER_PATH.name}")
    print(f"Reporte guardado en: {REPORT_PATH.name}")
    print(f"Matriz de confusion guardada en: {CONFUSION_MATRIX_PATH.name}")


if __name__ == "__main__":
    main()
