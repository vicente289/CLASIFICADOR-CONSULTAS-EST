"""Aplicacion Streamlit para clasificar consultas estudiantiles."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_category


BASE_DIR = Path(__file__).resolve().parent

CATEGORY_DESCRIPTIONS = {
    "Inscripciones": "Consultas sobre registro, reinscripcion, materias y fechas de inscripcion.",
    "Horarios": "Preguntas sobre horarios de clases, aulas, turnos y cambios de grupo.",
    "Pagos": "Dudas sobre cuotas, mensualidades, deudas, recibos y pagos pendientes.",
    "Notas": "Consultas sobre calificaciones, promedios, kardex y revision de notas.",
    "Trámites": "Solicitudes de certificados, constancias, historial academico y documentos.",
    "Plataforma virtual": "Problemas de acceso, contrasena, tareas, aulas virtuales y sistema en linea.",
    "Becas": "Preguntas sobre postulacion, requisitos, renovacion y resultados de becas.",
    "Exámenes": "Dudas sobre parciales, finales, recuperatorios, fechas y modalidades de examen.",
    "Otros": "Consultas generales que no encajan claramente en las categorias anteriores.",
}

EXAMPLE_QUERIES = [
    "No puedo ingresar a la plataforma",
    "¿Cuánto debo pagar?",
    "¿Cuándo es el parcial?",
    "Necesito una constancia",
    "¿Dónde veo mi horario?",
]


def render_sidebar():
    st.sidebar.header("Ejemplos de prueba")
    for example in EXAMPLE_QUERIES:
        if st.sidebar.button(example, use_container_width=True):
            st.session_state["consulta"] = example

    st.sidebar.header("Categorias")
    for category, description in CATEGORY_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**{category}:** {description}")


def render_prediction_result(result):
    st.subheader("Resultado")
    col_category, col_confidence = st.columns(2)
    col_category.metric("Categoria predicha", result["categoria"])
    col_confidence.metric("Confianza", f"{result['confianza']:.2f}%")

    st.markdown("**Top 3 categorias probables**")
    top_df = pd.DataFrame(result["top_3"])
    st.dataframe(top_df, hide_index=True, use_container_width=True)
    st.bar_chart(top_df.set_index("categoria"))

    with st.expander("Texto procesado"):
        st.write(result["texto_limpio"])


def main():
    st.set_page_config(
        page_title="Clasificador de Consultas Estudiantiles",
        layout="wide",
    )
    render_sidebar()

    st.title("Clasificador Automático de Consultas Estudiantiles")
    st.write(
        "Prototipo académico de Aprendizaje Supervisado y Procesamiento del "
        "Lenguaje Natural. Usa TF-IDF y un modelo clásico de clasificación "
        "multiclase para identificar la categoría de una consulta escrita en español."
    )

    if "consulta" not in st.session_state:
        st.session_state["consulta"] = ""

    consulta = st.text_area(
        "Escribe una consulta estudiantil",
        key="consulta",
        height=130,
        placeholder="Ejemplo: No puedo entrar a la plataforma virtual",
    )

    classify = st.button("Clasificar consulta", type="primary")
    if classify:
        if not consulta.strip():
            st.warning("Ingresa una consulta antes de clasificar.")
            return

        try:
            result = predict_category(consulta)
            render_prediction_result(result)
        except FileNotFoundError as error:
            st.error(str(error))
            st.info("Entrena el modelo ejecutando: python train_model.py")
        except ValueError as error:
            st.warning(str(error))

    st.divider()
    st.subheader("Evidencia del entrenamiento")
    report_path = BASE_DIR / "metrics_report.txt"
    matrix_path = BASE_DIR / "confusion_matrix.png"

    if report_path.exists():
        with st.expander("Ver reporte de metricas"):
            st.text(report_path.read_text(encoding="utf-8"))
    else:
        st.info("El reporte aparecera despues de ejecutar python train_model.py.")

    if matrix_path.exists():
        st.image(str(matrix_path), caption="Matriz de confusion")


if __name__ == "__main__":
    main()
