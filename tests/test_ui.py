"""Prueba de humo de la interfaz Streamlit con la API simulada (sin red ni modelos)."""

from pathlib import Path

import pytest
import requests

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

ANSWER = {"question": "q", "answer": "Toyota registró 2.433 vehículos en 2026. [1]", "grounded": True, "refused": False,
          "retried": False, "problems": [], "filters": {"anio": [2026]}, "model": "qwen2.5:3b-instruct",
          "seconds": 1.2, "route": "sql", "sql": "SELECT count(*) AS registros FROM vehiculos",
          "table": [{"registros": 2433}],
          "sources": [{"n": 1, "doc_id": "sql:resultado", "fuente": "SRI datos abiertos (consulta SQL)",
                       "pagina": None, "doc_type": "sql", "score": 1.0, "text": "registros: 2.433"}]}

GET = {
    "/health": {"status": "ok", "corpus_docs": 1335, "ollama": True, "models": ["qwen2.5:3b-instruct"],
                "default_model": "qwen2.5:3b-instruct"},
    "/stats/meta": {"anios": [2025, 2026], "cobertura": {"2025": "2025-12-31", "2026": "2026-09-30"},
                    "marcas": ["KIA", "TOYOTA"], "combustibles": ["GASOLINA"], "provincias": ["GUAYAS"], "clases": ["AUTOMOVIL"]},
    "/stats/kpis": {"periodo": "enero a septiembre", "mes_corte": 9, "variacion_pct": 40.9,
                    "actual": {"anio": 2026, "registros": 78219, "electrificados_pct": 12.3, "marca_lider": "KIA"},
                    "anterior": {"anio": 2025, "registros": 55512, "electrificados_pct": 7.8, "marca_lider": "CHEVROLET"}},
    "/stats/mensual": [{"anio": 2025, "mes": 1, "registros": 100}, {"anio": 2026, "mes": 1, "registros": 150}],
    "/stats/top": [{"nombre": "KIA", "registros": 12198, "participacion": 15.6}],
}


class Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.headers, self.text = payload, status, {"content-type": "application/json"}, ""

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    import streamlit as st
    st.cache_data.clear()  # la cache de Streamlit es global al proceso: evita respuestas de otra prueba
    monkeypatch.setattr(requests, "get", lambda url, params=None, timeout=None: Resp(GET[url.split("8000")[-1].split("?")[0]]))
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: Resp(ANSWER))


def run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_navegacion_y_estado():
    at = run()
    assert [o for o in at.sidebar.radio[0].options] == [
        ":material/forum:  Asistente", ":material/monitoring:  Dashboard", ":material/info:  Acerca de"]
    estado = " ".join(m.value for m in at.sidebar.markdown)
    assert "API conectada" in estado and "1335 documentos indexados" in estado


def test_asistente_responde_con_sql_y_fuentes():
    at = run()
    at.chat_input[0].set_value("¿Cuántos vehículos registró Toyota en 2026?").run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown)
    assert "Toyota registró 2.433 vehículos" in textos and "Cifra calculada con una consulta SQL" in " ".join(c.value for c in at.caption)
    assert [e.label for e in at.expander] == ["Consulta SQL y resultado", "Fuentes consultadas (1)"]


def test_dashboard_muestra_indicadores():
    at = run()
    at.sidebar.radio[0].set_value(":material/monitoring:  Dashboard").run()
    assert not at.exception
    valores = {m.label: m.value for m in at.metric}
    assert valores["Registros 2026 (enero a septiembre)"] == "78.219" and valores["Marca líder"] == "KIA"
    assert [m.delta for m in at.metric][:2] == ["+40,9 % vs 2025", "+4,5 pp vs 2025"]


def test_api_caida_muestra_error_y_no_revienta(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("sin api")
    monkeypatch.setattr(requests, "get", boom)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("No se pudo consultar la API" in e.value for e in at.error)
