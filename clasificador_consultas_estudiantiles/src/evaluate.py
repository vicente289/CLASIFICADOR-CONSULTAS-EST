"""Utilidades reutilizables para evaluar modelos de clasificacion."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(y_true, y_pred):
    """Calcula metricas principales con promedio macro."""
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_score,
    }


def save_metrics_report(
    y_true,
    y_pred,
    labels,
    output_path,
    model_name,
    comparison_results=None,
):
    """Guarda un reporte textual con metricas y classification_report."""
    output_path = Path(output_path)
    metrics = calculate_metrics(y_true, y_pred)

    lines = [
        "REPORTE DE METRICAS - CLASIFICADOR DE CONSULTAS ESTUDIANTILES",
        "=" * 68,
        f"Modelo final seleccionado: {model_name}",
        "",
        "Metricas generales del modelo final",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision macro: {metrics['precision_macro']:.4f}",
        f"Recall macro: {metrics['recall_macro']:.4f}",
        f"F1-score macro: {metrics['f1_macro']:.4f}",
        "",
    ]

    if comparison_results:
        lines.extend(
            [
                "Comparacion de modelos",
                "-" * 24,
            ]
        )
        for result in comparison_results:
            lines.append(
                f"{result['model']}: accuracy={result['accuracy']:.4f}, "
                f"f1_macro={result['f1_macro']:.4f}"
            )
        lines.append("")

    lines.extend(
        [
            "Classification report",
            "-" * 24,
            classification_report(y_true, y_pred, labels=labels, zero_division=0),
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return metrics


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    """Genera y guarda una matriz de confusion como imagen PNG."""
    output_path = Path(output_path)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Matriz de confusion - Clasificador de consultas")
    plt.xlabel("Categoria predicha")
    plt.ylabel("Categoria real")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return matrix
