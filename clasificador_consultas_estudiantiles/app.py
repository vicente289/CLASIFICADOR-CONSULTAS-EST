"""Aplicacion Streamlit para clasificar consultas estudiantiles."""

from pathlib import Path

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

            h1, h2, h3 {
                color: var(--text);
                letter-spacing: 0;
            }

            .hero {
                padding: 1.45rem 1.6rem;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: linear-gradient(135deg, rgba(12, 27, 46, 0.95), rgba(16, 36, 61, 0.82));
                box-shadow: 0 18px 55px rgba(0, 0, 0, 0.26);
                margin-bottom: 1.1rem;
            }

            .hero-title {
                margin: 0 0 0.35rem 0;
                font-size: clamp(2rem, 4vw, 3.15rem);
                font-weight: 800;
                line-height: 1.05;
                color: #f8fbff;
            }

            .hero-subtitle {
                max-width: 850px;
                color: var(--muted);
                font-size: 1.02rem;
                margin: 0.2rem 0 1rem 0;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                min-height: 32px;
                padding: 0.32rem 0.72rem;
                border-radius: 8px;
                border: 1px solid rgba(34, 211, 238, 0.34);
                background: rgba(34, 211, 238, 0.09);
                color: #c9f7ff;
                font-weight: 700;
                font-size: 0.83rem;
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

            .footer-note {
                margin-top: 1.2rem;
                padding: 0.95rem 1rem;
                border-left: 3px solid var(--cyan);
                background: rgba(34, 211, 238, 0.08);
                border-radius: 8px;
                color: #c7d9eb;
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

            @media (max-width: 820px) {
                .result-grid,
                .top-grid,
                .model-grid {
                    grid-template-columns: 1fr;
                }

                .hero {
                    padding: 1.1rem;
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
            <div class="badge-row">
                <span class="badge">Python</span>
                <span class="badge">TF-IDF</span>
                <span class="badge">Regresión Logística</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("### Consultas de prueba")
    for example in EXAMPLE_QUERIES:
        if st.sidebar.button(example, use_container_width=True):
            st.session_state["consulta"] = example

    with st.sidebar.expander("Ver categorías del sistema"):
        for category, description in CATEGORY_DESCRIPTIONS.items():
            st.markdown(f"**{category}**  \n{description}")


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
    st.dataframe(top_df, hide_index=True, use_container_width=True)


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

        classify = st.button("Clasificar consulta", type="primary", use_container_width=True)

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

        report_path = BASE_DIR / "metrics_report.txt"
        matrix_path = BASE_DIR / "confusion_matrix.png"
        metrics_tab, matrix_tab, info_tab = st.tabs(
            ["Métricas", "Matriz de confusión", "Información del modelo"]
        )

        with metrics_tab:
            if report_path.exists():
                with st.expander("Ver reporte de métricas", expanded=False):
                    st.text(report_path.read_text(encoding="utf-8"))
            else:
                st.info("El reporte aparecera despues de ejecutar python train_model.py.")

        with matrix_tab:
            if matrix_path.exists():
                st.image(str(matrix_path), caption="Matriz de confusión")
            else:
                st.info("La matriz aparecera despues de ejecutar python train_model.py.")

        with info_tab:
            render_model_info()


def render_footer():
    st.markdown(
        """
        <div class="footer-note">
            Este prototipo corresponde a una versión alfa del sistema y puede mejorar con más datos reales.
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    render_footer()


if __name__ == "__main__":
    main()
