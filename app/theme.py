"""Estilo de la interfaz: consola de administracion con barra lateral oscura y tarjetas blancas por seccion."""

import streamlit as st

ACCENT = "#0033E2"

CSS = """
:root {
  --accent: #0033E2;
  --nav-bg: #0F2F52;
  --nav-active: #1B4B7C;
  --nav-text: #C9D8E8;
  --line: #E2E6EB;
  --muted: #6B7785;
  --ink: #1F2933;
}
.stApp, .stApp button, .stApp input, .stApp textarea { font-family: Roboto, "Segoe UI", system-ui, -apple-system, sans-serif; }
/* los iconos de Streamlit son una fuente: no deben heredar la tipografia del texto */
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
.stApp { background: #F2F4F7; }

/* barra superior blanca, sin el menu de despliegue */
[data-testid="stHeader"] { background: #fff; border-bottom: 1px solid var(--line); height: 3rem; }
[data-testid="stToolbar"] { right: 1rem; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4.2rem !important; padding-bottom: 3rem; max-width: 1180px; }

/* titulo de pagina */
.page-title { font-size: 20px; font-weight: 600; color: var(--ink); margin: 0.2rem 0 0; }
.page-sub { color: var(--muted); font-size: 13px; margin: 2px 0 14px; }

/* tarjetas: contenedores con borde */
[class*="st-key-card_"] {
  background: #fff; border: 1px solid var(--line); border-radius: 6px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, .04); padding: 6px 10px 10px;
}
.card-title { font-size: 14px; font-weight: 600; color: var(--ink); margin: 4px 0 6px; }
.card-hint { color: var(--muted); font-size: 12px; margin: -2px 0 8px; }

/* metricas */
[data-testid="stMetric"] { padding: 2px 0; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 12.5px; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 26px; font-weight: 600; color: var(--ink); }

/* controles */
.stButton > button, [data-testid="stBaseButton-secondary"] {
  border: 1px solid #CBD2D9; background: #fff; color: var(--ink); border-radius: 4px; font-weight: 500;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
[data-testid="stSelectbox"] label p, [data-testid="stSlider"] label p, [data-testid="stCheckbox"] label p,
[data-testid="stWidgetLabel"] p { font-size: 12.5px; color: #3E4C59; font-weight: 500; }

/* conversacion */
[data-testid="stChatMessage"] { background: transparent; border-bottom: 1px solid #F0F2F5; border-radius: 0; padding: 12px 4px; }
[data-testid="stChatInput"] { border-radius: 6px; }

/* ---------- barra lateral ---------- */
[data-testid="stSidebar"] { background: var(--nav-bg); border-right: none; }
[data-testid="stSidebar"] * { color: var(--nav-text); }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); margin: 14px 0; }
[data-testid="stSidebarHeader"] { padding: 6px 10px 0; height: 2.2rem; min-height: 0; }
[data-testid="stSidebarContent"] { padding-top: 0; }

.brand { display: flex; align-items: center; gap: 10px; padding: 4px 6px 14px; }
.brand-logo { width: 34px; height: 34px; border-radius: 8px; background: var(--accent); color: #fff !important;
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; letter-spacing: .5px; }
.brand-name { color: #fff !important; font-weight: 600; font-size: 15px; line-height: 1.2; }
.brand-sub { color: #8FA8C2 !important; font-size: 11.5px; }
.nav-label { color: #7F9AB6 !important; font-size: 11px; letter-spacing: .08em; font-weight: 600; margin: 10px 8px 4px; }

/* el radio de navegacion se ve como menu */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
  padding: 10px 12px; border-radius: 4px; width: 100%; border-left: 3px solid transparent; cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: rgba(255,255,255,.07); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
  background: var(--nav-active); border-left-color: #5B9DFF;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) * { color: #fff; font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] p { font-size: 14px; }

.status { font-size: 12px; color: #A9BDD2 !important; margin: 5px 8px; display: flex; align-items: center; gap: 8px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: #4ADE80; }
.dot.off { background: #F87171; }
"""


def inject() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        '<div class="brand"><div class="brand-logo">MA</div><div><div class="brand-name">Mercado Automotriz</div>'
        '<div class="brand-sub">Ecuador · SRI y AEADE</div></div></div>', unsafe_allow_html=True)


def nav_label(text: str) -> None:
    st.markdown(f'<div class="nav-label">{text}</div>', unsafe_allow_html=True)


def status(text: str, ok: bool = True) -> None:
    st.markdown(f'<div class="status"><span class="dot {"" if ok else "off"}"></span>{text}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def card_title(text: str, hint: str = "") -> None:
    st.markdown(f'<div class="card-title">{text}</div>' + (f'<div class="card-hint">{hint}</div>' if hint else ""),
                unsafe_allow_html=True)


def card(key: str):
    """Tarjeta blanca con borde; `key` debe ser unico (la clase CSS sale de `st-key-card_<key>`)."""
    return st.container(border=False, key=f"card_{key}")
