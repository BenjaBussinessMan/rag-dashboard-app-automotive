"""Interfaz Streamlit: asistente RAG + dashboard del mercado. Consume la API (RAG_API_URL)."""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")
MESES_ABREV = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
# Paleta categorica en orden fijo (azul, naranja, aqua, amarillo...): el color sigue a la entidad, no al ranking.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BAR_COLOR = SERIES_COLORS[0]
EJEMPLOS = [
    "¿Cuántos vehículos registró Toyota en 2026?",
    "¿Qué provincia concentra más registros en 2025?",
    "¿Cómo estuvo el mercado automotriz en marzo de 2026?",
    "¿Cuánto crecieron las ventas entre enero y septiembre de 2026 según AEADE?",
]

st.set_page_config(page_title="Mercado automotriz Ecuador", layout="wide")


class ApiError(Exception):
    pass


def api_get(path: str, **params):
    params = {k: v for k, v in params.items() if v not in (None, "Todas")}
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise ApiError(f"No se pudo consultar la API en {API_URL}: {exc}") from exc


def api_ask(question: str, model: str | None, k: int) -> dict:
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question, "model": model, "k": k}, timeout=900)
    except requests.RequestException as exc:
        raise ApiError(f"No se pudo conectar con la API en {API_URL}: {exc}") from exc
    if r.status_code != 200:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise ApiError(f"La API respondió {r.status_code}: {detail}")
    return r.json()


@st.cache_data(ttl=120, show_spinner=False)
def cached_get(path: str, **params):
    return api_get(path, **params)


def fmt(n) -> str:
    return f"{int(n):,}".replace(",", ".")


# ---------------------------------------------------------------- asistente

def render_response(resp: dict) -> None:
    if resp["refused"]:
        st.info(resp["answer"])
    elif resp["grounded"]:
        st.markdown(resp["answer"])
        st.caption("Respuesta verificada: cada cifra aparece en el fragmento citado.")
    else:
        st.warning(resp["answer"])
    with st.expander(f"Fuentes consultadas ({len(resp['sources'])})"):
        for s in resp["sources"]:
            pagina = f", página {s['pagina']}" if s.get("pagina") else ""
            st.markdown(f"**[{s['n']}]** `{s['doc_id']}` · {s.get('fuente') or 'fuente desconocida'}{pagina}")
            st.caption(s["text"])
    filtros = ", ".join(f"{k}={v}" for k, v in resp["filters"].items()) or "ninguno"
    st.caption(f"Modelo {resp['model']} · {resp['seconds']} s · filtros detectados: {filtros}"
               + (" · se reintentó una vez" if resp["retried"] else ""))


def tab_asistente(model: str | None, k: int) -> None:
    st.session_state.setdefault("history", [])
    pending = st.session_state.pop("pending", None)

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_response(turn["response"])

    if not st.session_state.history and pending is None:
        st.write("Pregunta sobre registros de vehículos nuevos del SRI y los boletines de AEADE. Por ejemplo:")
        cols = st.columns(2)
        for i, ejemplo in enumerate(EJEMPLOS):
            if cols[i % 2].button(ejemplo, key=f"ej{i}", width="stretch"):
                st.session_state.pending = ejemplo
                st.rerun()

    question = pending or st.chat_input("Escribe tu pregunta")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando en las fuentes y redactando la respuesta (puede tardar unos segundos)..."):
                    resp = api_ask(question, model, k)
            except ApiError as exc:
                st.error(str(exc))
                return
            render_response(resp)
        st.session_state.history.append({"question": question, "response": resp})


# ---------------------------------------------------------------- dashboard

# Vega formatea los miles con coma; se reemplaza por punto para coincidir con el resto de la interfaz.
MILES_ES = alt.Axis(labelExpr="replace(format(datum.value, ',.0f'), regexp(',', 'g'), '.')", gridOpacity=0.15)


def line_chart(df: pd.DataFrame, anios: list[int]) -> alt.Chart:
    df = df.assign(anio=df["anio"].astype(str), mes_txt=df["mes"].map(lambda m: MESES_ABREV[m - 1]),
                   registros_txt=df["registros"].map(fmt))
    domain = [str(a) for a in anios]
    scale = alt.Scale(domain=domain, range=SERIES_COLORS[:len(domain)])
    base = alt.Chart(df).encode(
        x=alt.X("mes_txt:N", sort=MESES_ABREV, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("registros:Q", title="Registros", axis=MILES_ES),
        color=alt.Color("anio:N", scale=scale, legend=alt.Legend(title="Año", orient="top")))
    lines = base.mark_line(strokeWidth=2).encode(
        tooltip=[alt.Tooltip("anio:N", title="Año"), alt.Tooltip("mes_txt:N", title="Mes"),
                 alt.Tooltip("registros_txt:N", title="Registros")])
    points = base.mark_point(size=60, filled=True, stroke="white", strokeWidth=1.5).encode(
        tooltip=[alt.Tooltip("anio:N", title="Año"), alt.Tooltip("mes_txt:N", title="Mes"),
                 alt.Tooltip("registros_txt:N", title="Registros")])
    ultimo = df.sort_values("mes").groupby("anio").tail(1)
    labels = alt.Chart(ultimo).mark_text(dy=-12, fontWeight="bold").encode(
        x=alt.X("mes_txt:N", sort=MESES_ABREV), y="registros:Q", text="anio:N",
        color=alt.Color("anio:N", scale=scale, legend=None))
    return (lines + points + labels).properties(height=340)


def bar_chart(df: pd.DataFrame) -> alt.Chart:
    df = df.assign(registros_txt=df["registros"].map(fmt), participacion_txt=df["participacion"].map(
        lambda v: f"{v:.1f} %".replace(".", ",")))
    base = alt.Chart(df).encode(y=alt.Y("nombre:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)))
    bars = base.mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, size=16).encode(
        x=alt.X("registros:Q", title="Registros", axis=MILES_ES),
        tooltip=[alt.Tooltip("nombre:N", title=""), alt.Tooltip("registros_txt:N", title="Registros"),
                 alt.Tooltip("participacion_txt:N", title="Participación")])
    texto = base.mark_text(align="left", dx=5).encode(x="registros:Q", text="participacion_txt:N")
    return (bars + texto).properties(height=max(180, 30 * len(df)))


def tab_dashboard() -> None:
    try:
        meta = cached_get("/stats/meta")
        kpis = cached_get("/stats/kpis")
    except ApiError as exc:
        st.error(str(exc))
        return

    act, ant = kpis["actual"], kpis.get("anterior")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Registros {act['anio']} ({kpis['periodo']})", fmt(act["registros"]),
              None if kpis["variacion_pct"] is None else f"{kpis['variacion_pct']:+.1f} % vs {ant['anio']}".replace(".", ","))
    c2.metric("Eléctricos e híbridos", f"{act['electrificados_pct']:.1f} %".replace(".", ","),
              None if not ant else f"{act['electrificados_pct'] - ant['electrificados_pct']:+.1f} pp vs {ant['anio']}".replace(".", ","))
    c3.metric("Marca líder", act["marca_lider"] or "-")
    st.caption("Segmento: automóvil, camión y camioneta. Fuente: SRI Ecuador, vehículos nuevos (fecha de proceso). "
               "La comparación usa el mismo rango de meses porque el último año está incompleto "
               f"(datos hasta {meta['cobertura'][str(act['anio'])]}).")

    st.subheader("Registros por mes")
    f1, f2, f3, f4 = st.columns(4)
    marca = f1.selectbox("Marca", ["Todas"] + meta["marcas"])
    combustible = f2.selectbox("Combustible", ["Todas"] + meta["combustibles"])
    provincia = f3.selectbox("Provincia", ["Todas"] + meta["provincias"])
    clase = f4.selectbox("Clase", ["Todas"] + meta["clases"])
    mensual = pd.DataFrame(cached_get("/stats/mensual", marca=marca, combustible=combustible,
                                      provincia=provincia, clase=clase))
    if mensual.empty:
        st.info("No hay registros para esa combinación de filtros.")
    else:
        st.altair_chart(line_chart(mensual, meta["anios"]), width="stretch")
        with st.expander("Ver tabla de datos"):
            tabla = mensual.pivot(index="mes", columns="anio", values="registros").rename(index=lambda m: MESES_ABREV[m - 1])
            st.dataframe(tabla, width="stretch")

    st.subheader("Ranking")
    r1, r2, r3, r4 = st.columns(4)
    dimension = r1.selectbox("Agrupar por", ["marca", "provincia", "combustible", "clase"])
    anio = r2.selectbox("Año", list(reversed(meta["anios"])))
    limite = r3.slider("Elementos", 5, 20, 10)
    mismo = r4.checkbox(f"Solo {kpis['periodo']}", value=True,
                        help="Compara años con el mismo rango de meses; el último año está incompleto.")
    top = pd.DataFrame(cached_get("/stats/top", dimension=dimension, anio=anio, limit=limite,
                                  mes_hasta=kpis["mes_corte"] if mismo else None))
    if not top.empty:
        st.altair_chart(bar_chart(top), width="stretch")
        st.caption("El valor a la derecha de cada barra es la participación dentro del total filtrado.")


def tab_acerca() -> None:
    st.markdown("""
**Cómo funciona.** La pregunta se analiza para detectar año, mes, marca, provincia y combustible. Se recuperan
fragmentos con búsqueda híbrida (BM25 y embeddings BGE-M3, fusionados con RRF). Un LLM local (Ollama)
redacta la respuesta usando solo esos fragmentos, y un guardrail verifica que cada cifra aparezca en el
fragmento que se cita; si no, la respuesta se rechaza.

**Fuentes.** Registros de vehículos nuevos del SRI (datos abiertos) y boletines mensuales de AEADE.
Las cifras de ambas fuentes **no coinciden** (cobertura y definición de venta distintas): la respuesta
siempre indica de cuál viene cada cifra.

**Límites.** El año 2026 está incompleto. El guardrail comprueba que las cifras existan en la fuente citada,
no que esa fuente sea la más adecuada: con modelos pequeños conviene revisar las fuentes.
""")


def main() -> None:
    st.title("Asistente del mercado automotriz de Ecuador")
    with st.sidebar:
        st.header("Configuración")
        try:
            health = cached_get("/health")
        except ApiError as exc:
            st.error(str(exc))
            health = None
        model = None
        if health:
            models = health["models"] or [health["default_model"]]
            model = st.selectbox("Modelo local", models,
                                 index=models.index(health["default_model"]) if health["default_model"] in models else 0,
                                 help="Los modelos de 7B son más fiables pero más lentos.")
            if not health["ollama"]:
                st.warning("Ollama no responde: el asistente no podrá generar respuestas.")
            st.caption(f"{health['corpus_docs']} documentos indexados")
        k = st.slider("Fragmentos a recuperar", 3, 8, 5)
        if st.button("Borrar conversación"):
            st.session_state.history = []
            st.rerun()

    t1, t2, t3 = st.tabs(["Asistente", "Dashboard", "Acerca de"])
    with t1:
        tab_asistente(model, k)
    with t2:
        tab_dashboard()
    with t3:
        tab_acerca()


main()
