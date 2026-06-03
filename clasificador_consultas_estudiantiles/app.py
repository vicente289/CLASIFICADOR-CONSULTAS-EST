"""Aplicacion Streamlit para clasificar consultas estudiantiles."""

from pathlib import Path
from html import escape

import pandas as pd
import streamlit as st

from src.predict import predict_category


BASE_DIR = Path(__file__).resolve().parent

CATEGORY_DESCRIPTIONS = {
    "Inscripciones": "Registro, reinscripcion, materias y fechas de inscripcion.",
    "Horarios": "Horarios de clases, aulas, turnos y cambios de grupo.",
    "Pagos": "Cuotas, mensualidades, deudas, recibos y pagos pendientes.",
    "Notas": "Calificaciones, promedios, kardex y revision de notas.",
    "Trámites": "Certificados, constancias, historial academico y documentos.",
    "Plataforma virtual": "Acceso, contrasena, tareas, aulas virtuales y sistema en linea.",
    "Becas": "Postulacion, requisitos, renovacion y resultados de becas.",
    "Exámenes": "Parciales, finales, recuperatorios, fechas y modalidad de examen.",
    "Otros": "Consultas generales o ambiguas.",
}

EXAMPLE_QUERIES = [
    "No puedo ingresar a la plataforma",
    "¿Cuánto debo pagar?",
    "¿Cuándo es el parcial?",
    "Necesito una constancia",
    "¿Dónde veo mi horario?",
]

SUMMARY_METRICS = [
    {
        "label": "Exactitud",
        "value": "87.04%",
        "caption": "Predicciones correctas",
        "accent": "accent-cyan",
    },
    {
        "label": "Precisión macro",
        "value": "88.10%",
        "caption": "Promedio por categoría",
        "accent": "accent-blue",
    },
    {
        "label": "Recall macro",
        "value": "87.04%",
        "caption": "Casos detectados",
        "accent": "accent-violet",
    },
    {
        "label": "F1-score macro",
        "value": "86.47%",
        "caption": "Balance precisión/recall",
        "accent": "accent-cyan",
    },
]

CATEGORY_METRICS = [
    ["Inscripciones", "100.00%", "100.00%", "100.00%", 6],
    ["Horarios", "100.00%", "100.00%", "100.00%", 6],
    ["Pagos", "100.00%", "50.00%", "67.00%", 6],
    ["Notas", "83.00%", "83.00%", "83.00%", 6],
    ["Trámites", "86.00%", "100.00%", "92.00%", 6],
    ["Plataforma virtual", "83.00%", "83.00%", "83.00%", 6],
    ["Becas", "86.00%", "100.00%", "92.00%", 6],
    ["Exámenes", "83.00%", "83.00%", "83.00%", 6],
    ["Otros", "71.00%", "83.00%", "77.00%", 6],
]


def apply_custom_styles():
    """Aplica estilos visuales sin alterar la logica de la aplicacion."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #07111f;
                --panel: #0c1b2e;
                --panel-soft: #10243d;
                --line: rgba(125, 211, 252, 0.18);
                --text: #e5f0ff;
                --muted: #9fb3c8;
                --cyan: #22d3ee;
                --blue: #3b82f6;
                --violet: #a78bfa;
                --warning-bg: rgba(245, 158, 11, 0.14);
                --warning-line: rgba(245, 158, 11, 0.55);
                --success-bg: rgba(34, 197, 94, 0.14);
                --success-line: rgba(34, 197, 94, 0.42);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(34, 211, 238, 0.13), transparent 30rem),
                    radial-gradient(circle at top right, rgba(167, 139, 250, 0.10), transparent 28rem),
                    var(--bg);
                color: var(--text);
            }

            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2.5rem;
                max-width: 1180px;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #081526 0%, #0b1728 100%);
                border-right: 1px solid var(--line);
            }

            section[data-testid="stSidebar"] details {
                border: 1px solid rgba(125, 211, 252, 0.20);
                border-radius: 8px;
                background: rgba(8, 21, 38, 0.70);
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.20);
                overflow: hidden;
            }

            section[data-testid="stSidebar"] details summary {
                background: rgba(16, 36, 61, 0.82);
                border-bottom: 1px solid rgba(125, 211, 252, 0.14);
            }

            .sidebar-category-list {
                display: flex;
                flex-direction: column;
                gap: 0.42rem;
                padding: 0.35rem 0.05rem 0.15rem;
            }

            .sidebar-category-list details {
                border: 0 !important;
                background: transparent !important;
                box-shadow: none !important;
            }

            .sidebar-category-list details summary {
                border: 0 !important;
                background: transparent !important;
                min-height: 32px;
                padding: 0.12rem 0.18rem !important;
                list-style: none;
                cursor: pointer;
            }

            .sidebar-category-list details summary::-webkit-details-marker {
                display: none;
            }

            .sidebar-category-list details summary::marker {
                content: "";
            }

            .sidebar-category-list details[open] summary {
                background: rgba(34, 211, 238, 0.08) !important;
                border-radius: 8px;
            }

            .sidebar-category-description {
                margin: 0.1rem 0.35rem 0.55rem 2rem;
                color: #b9c7d6;
                font-size: 0.84rem;
                line-height: 1.45;
            }

            .category-row {
                display: grid;
                grid-template-columns: 20px 1fr;
                align-items: center;
                gap: 0.55rem;
                min-height: 32px;
                padding: 0.2rem 0.38rem;
                border-radius: 8px;
                color: #dbeafe;
            }

            .category-row:hover {
                background: rgba(34, 211, 238, 0.08);
            }

            .category-icon {
                position: relative;
                width: 15px;
                height: 13px;
                border: 1.5px solid var(--cyan);
                border-radius: 3px;
                box-shadow: 0 0 12px rgba(34, 211, 238, 0.18);
            }

            .category-icon::before {
                content: "";
                position: absolute;
                left: 2px;
                top: -4px;
                width: 7px;
                height: 4px;
                border: 1.5px solid var(--cyan);
                border-bottom: 0;
                border-radius: 3px 3px 0 0;
                background: rgba(8, 21, 38, 0.95);
            }

            .category-name {
                color: #e5f0ff;
                font-size: 0.93rem;
                font-weight: 700;
                line-height: 1.2;
            }

            .sidebar-total-card {
                display: grid;
                grid-template-columns: 36px 1fr;
                gap: 0.7rem;
                align-items: center;
                margin-top: 1rem;
                padding: 0.95rem;
                border: 1px solid rgba(34, 211, 238, 0.26);
                border-radius: 8px;
                background: linear-gradient(135deg, rgba(8, 47, 73, 0.72), rgba(15, 23, 42, 0.88));
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.20);
            }

            .sidebar-total-icon {
                width: 26px;
                height: 26px;
                border: 2px solid var(--cyan);
                border-radius: 50%;
                color: var(--cyan);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 900;
            }

            .sidebar-total-label {
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 800;
            }

            .sidebar-total-number {
                color: var(--cyan);
                font-size: 1.5rem;
                font-weight: 900;
                line-height: 1.1;
            }

            .sidebar-total-caption {
                color: #b7c7d8;
                font-size: 0.82rem;
            }

            h1, h2, h3 {
                color: var(--text);
                letter-spacing: 0;
            }

            .hero {
                padding: 1.45rem 1.8rem 1.35rem;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: linear-gradient(135deg, rgba(12, 27, 46, 0.95), rgba(16, 36, 61, 0.82));
                box-shadow: 0 18px 55px rgba(0, 0, 0, 0.26);
                margin-bottom: 1.1rem;
                text-align: center;
            }

            .hero-title {
                margin: 0 auto 0.75rem;
                max-width: 900px;
                font-size: 2.35rem;
                font-weight: 850;
                line-height: 1.12;
                color: #f8fbff;
            }

            .hero-subtitle {
                max-width: 820px;
                color: var(--muted);
                font-size: 1.02rem;
                line-height: 1.65;
                margin: 0 auto;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-color: var(--line);
                border-radius: 8px;
                background: rgba(12, 27, 46, 0.62);
                box-shadow: 0 16px 42px rgba(0, 0, 0, 0.18);
            }

            .section-kicker {
                margin: 0 0 0.4rem 0;
                color: var(--cyan);
                font-size: 0.76rem;
                text-transform: uppercase;
                font-weight: 800;
                letter-spacing: 0.08em;
            }

            .section-title {
                margin: 0 0 0.9rem 0;
                color: #f8fbff;
                font-size: 1.28rem;
                font-weight: 780;
            }

            .result-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.85rem;
                margin: 0.85rem 0 1rem 0;
            }

            .metric-card,
            .top-card,
            .model-card {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: linear-gradient(180deg, rgba(16, 36, 61, 0.94), rgba(8, 21, 38, 0.92));
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
            }

            .metric-card {
                min-height: 138px;
                padding: 1rem;
                text-align: center;
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                font-weight: 800;
                letter-spacing: 0.07em;
            }

            .metric-value {
                margin-top: 0.45rem;
                color: #f8fbff;
                font-size: 1.75rem;
                line-height: 1.1;
                font-weight: 850;
                word-break: break-word;
            }

            .metric-caption {
                margin-top: 0.55rem;
                color: var(--muted);
                font-size: 0.88rem;
            }

            .accent-cyan {
                border-top: 3px solid var(--cyan);
            }

            .accent-blue {
                border-top: 3px solid var(--blue);
            }

            .accent-violet {
                border-top: 3px solid var(--violet);
            }

            .status-success,
            .status-warning {
                border-radius: 8px;
                padding: 0.82rem 0.95rem;
                margin: 0.65rem 0 0.9rem 0;
                font-weight: 700;
            }

            .status-success {
                border: 1px solid var(--success-line);
                background: var(--success-bg);
                color: #bbf7d0;
            }

            .status-warning {
                border: 1px solid var(--warning-line);
                background: var(--warning-bg);
                color: #fde68a;
            }

            .top-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin-top: 0.55rem;
            }

            .top-card {
                padding: 0.82rem;
            }

            .top-rank {
                color: var(--cyan);
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
            }

            .top-category {
                color: #f8fbff;
                font-size: 1.05rem;
                font-weight: 760;
                margin-top: 0.22rem;
            }

            .top-confidence {
                color: var(--muted);
                margin-top: 0.12rem;
                font-size: 0.92rem;
            }

            .model-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.8rem;
            }

            .model-card {
                padding: 0.95rem;
            }

            .model-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
            }

            .model-value {
                color: #f8fbff;
                margin-top: 0.32rem;
                font-size: 1.02rem;
                font-weight: 760;
            }

            div.stButton > button {
                border: 1px solid rgba(34, 211, 238, 0.55);
                background: linear-gradient(90deg, #0891b2 0%, #2563eb 100%);
                color: #ffffff;
                border-radius: 8px;
                min-height: 42px;
                font-weight: 800;
                box-shadow: 0 10px 26px rgba(37, 99, 235, 0.22);
            }

            div.stButton > button:hover {
                border-color: rgba(125, 211, 252, 0.9);
                color: #ffffff;
                transform: translateY(-1px);
            }

            textarea {
                border-radius: 8px !important;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stImage"],
            div[data-testid="stExpander"] {
                border-radius: 8px;
            }

            div[data-testid="stTabs"] button[data-baseweb="tab"] {
                color: var(--muted);
            }

            div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
                color: var(--cyan);
            }

            div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
                color: var(--cyan);
            }

            div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {
                color: var(--cyan);
                font-weight: 800;
            }

            div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
                background-color: var(--cyan);
            }

            @media (max-width: 820px) {
                .result-grid,
                .top-grid,
                .model-grid {
                    grid-template-columns: 1fr;
                }

                .hero {
                    padding: 1.1rem;
                }

                .hero-title {
                    font-size: 1.85rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <section class="hero">
            <h1 class="hero-title">Clasificador Automático de Consultas Estudiantiles</h1>
            <p class="hero-subtitle">
                Sistema de IA para clasificar consultas académicas mediante
                Aprendizaje Supervisado y Procesamiento del Lenguaje Natural.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("### Consultas de prueba")
    for example in EXAMPLE_QUERIES:
        if st.sidebar.button(example, width="stretch"):
            st.session_state["consulta"] = example

    with st.sidebar.expander("Ver categorías del sistema"):
        category_items = "\n".join(
            f"""
            <details>
                <summary>
                    <div class="category-row">
                        <span class="category-icon"></span>
                        <span class="category-name">{escape(category)}</span>
                    </div>
                </summary>
                <div class="sidebar-category-description">{escape(description)}</div>
            </details>
            """
            for category, description in CATEGORY_DESCRIPTIONS.items()
        )
        st.markdown(
            f"""
            <div class="sidebar-category-list">
                {category_items}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-total-card">
            <div class="sidebar-total-icon">i</div>
            <div>
                <div class="sidebar-total-label">Total categorías</div>
                <div class="sidebar-total-number">{len(CATEGORY_DESCRIPTIONS)}</div>
                <div class="sidebar-total-caption">Clases disponibles</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, caption, accent):
    st.markdown(
        f"""
        <div class="metric-card {accent}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_message(requires_review):
    if requires_review:
        st.markdown(
            """
            <div class="status-warning">
                La consulta es ambigua o tiene baja confianza. Se recomienda revisión manual.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-success">
                Consulta clasificada correctamente.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_top_categories(top_categories):
    st.markdown("#### Top 3 categorías probables")
    columns = st.columns(3)
    for index, item in enumerate(top_categories, start=1):
        with columns[index - 1]:
            st.markdown(
                f"""
                <div class="top-card">
                    <div class="top-rank">Opción {index}</div>
                    <div class="top-category">{item["categoria"]}</div>
                    <div class="top-confidence">{item["confianza"]:.2f}% de confianza</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    top_df = pd.DataFrame(top_categories)
    st.dataframe(top_df, hide_index=True, width="stretch")


def render_summary_metrics():
    metric_columns = st.columns(4)
    for column, metric in zip(metric_columns, SUMMARY_METRICS):
        with column:
            render_metric_card(
                metric["label"],
                metric["value"],
                metric["caption"],
                metric["accent"],
            )


def render_category_metrics_table():
    metrics_df = pd.DataFrame(
        CATEGORY_METRICS,
        columns=["Categoría", "Precisión", "Recall", "F1-score", "Soporte"],
    )
    st.dataframe(metrics_df, hide_index=True, width="stretch")


def render_prediction_result(result):
    final_category = result.get("categoria_final", result["categoria"])
    model_category = result.get("categoria_modelo", final_category)
    confidence = float(result["confianza"])
    requires_review = bool(result.get("requiere_revision", False))
    status_text = "Revisión manual" if requires_review else "Clasificada"
    status_caption = "Baja confianza detectada" if requires_review else "Confianza suficiente"

    with st.container(border=True):
        st.markdown('<p class="section-kicker">Resultado del clasificador</p>', unsafe_allow_html=True)
        render_status_message(requires_review)

        col_category, col_confidence, col_status = st.columns(3)
        with col_category:
            render_metric_card("Categoría final", final_category, f"Modelo: {model_category}", "accent-cyan")
        with col_confidence:
            render_metric_card("Confianza", f"{confidence:.2f}%", "Probabilidad máxima", "accent-blue")
        with col_status:
            render_metric_card("Estado", status_text, status_caption, "accent-violet")

        render_top_categories(result["top_3"])

        with st.expander("Ver texto procesado"):
            st.write(result["texto_limpio"])


def render_input_section():
    if "consulta" not in st.session_state:
        st.session_state["consulta"] = ""

    with st.container(border=True):
        st.markdown('<p class="section-kicker">Consulta estudiantil</p>', unsafe_allow_html=True)
        st.markdown(
            '<h2 class="section-title">Ingrese la consulta para clasificar</h2>',
            unsafe_allow_html=True,
        )

        consulta = st.text_area(
            "Texto de la consulta",
            key="consulta",
            height=105,
            placeholder="Ejemplo: No puedo entrar a la plataforma virtual",
            label_visibility="collapsed",
        )

        classify = st.button("Clasificar consulta", type="primary", width="stretch")

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


def render_model_info():
    st.markdown(
        """
        <div class="model-grid">
            <div class="model-card">
                <div class="model-label">Modelo final</div>
                <div class="model-value">Regresión Logística Multiclase</div>
            </div>
            <div class="model-card">
                <div class="model-label">Representación</div>
                <div class="model-value">TF-IDF</div>
            </div>
            <div class="model-card">
                <div class="model-label">Dataset</div>
                <div class="model-value">270 consultas</div>
            </div>
            <div class="model-card">
                <div class="model-label">Categorías</div>
                <div class="model-value">9 clases</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_training_evidence():
    with st.container(border=True):
        st.markdown('<p class="section-kicker">Evidencia del entrenamiento</p>', unsafe_allow_html=True)
        st.caption(
            "Esta sección resume la evaluación del modelo con el conjunto de prueba. "
            "No cambia con cada consulta individual ingresada en la aplicación."
        )

        report_path = BASE_DIR / "metrics_report.txt"
        matrix_path = BASE_DIR / "confusion_matrix.png"
        metrics_tab, matrix_tab, info_tab = st.tabs(
            ["Métricas", "Matriz de confusión", "Información del modelo"]
        )

        with metrics_tab:
            st.markdown("#### Resumen de desempeño")
            render_summary_metrics()
            st.markdown("#### Métricas por categoría")
            render_category_metrics_table()

            if report_path.exists():
                with st.expander("Ver reporte técnico completo", expanded=False):
                    st.text(report_path.read_text(encoding="utf-8"))
            else:
                st.info("El reporte aparecera despues de ejecutar python train_model.py.")

        with matrix_tab:
            st.info(
                "La diagonal principal representa las clasificaciones correctas. "
                "Los valores fuera de la diagonal indican confusiones entre categorías. "
                "Esta matriz corresponde al entrenamiento/evaluación del modelo, no a una consulta individual."
            )
            if matrix_path.exists():
                st.image(str(matrix_path), caption="Matriz de confusión")
            else:
                st.info("La matriz aparecera despues de ejecutar python train_model.py.")

        with info_tab:
            render_model_info()


def main():
    st.set_page_config(
        page_title="Clasificador de Consultas Estudiantiles",
        layout="wide",
    )
    apply_custom_styles()
    render_sidebar()
    render_header()
    render_input_section()
    render_training_evidence()


if __name__ == "__main__":
    main()
