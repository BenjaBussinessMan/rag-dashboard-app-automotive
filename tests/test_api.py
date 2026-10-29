import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.generation.ollama import OllamaUnavailable
from src.retrieval.bm25 import tokenize
from src.retrieval.hybrid import HybridRetriever

FRAG = ["Marca KIA, año 2026. Registros: 15.649 (8,1 % del segmento).",
        "Marca TOYOTA, año 2026. Registros: 2.433 (3,1 % del segmento)."]


class FakeDense:
    def scores(self, q):
        import numpy as np
        qs = set(tokenize(q))
        return np.array([len(qs & set(tokenize(t))) for t in FRAG], dtype=np.float32)


class FakeLLM:
    def __init__(self, model="fake", replies=None, installed=("fake", "otro")):
        self.model, self.replies, self._installed = model, list(replies or []), list(installed)

    def installed_models(self):
        return self._installed

    def chat(self, messages, temperature=0.0):
        return self.replies.pop(0) if self.replies else "Kia registró 15.649 vehículos en 2026."


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("""CREATE TABLE vehiculos AS SELECT * FROM (VALUES
        (2025, 1, 'KIA', 'GUAYAS', 'GASOLINA', 'AUTOMOVIL'),
        (2025, 2, 'KIA', 'GUAYAS', 'ELECTRICO', 'AUTOMOVIL'),
        (2025, 2, 'TOYOTA', 'PICHINCHA', 'GASOLINA', 'CAMIONETA'),
        (2026, 1, 'KIA', 'GUAYAS', 'GASOLINA', 'AUTOMOVIL'),
        (2026, 1, 'KIA', 'PICHINCHA', 'ELECTRICO', 'AUTOMOVIL'),
        (2026, 2, 'TOYOTA', 'PICHINCHA', 'HIBRIDO_GASOLINA_BATERIAS', 'CAMIONETA'),
        (2026, 2, 'KIA', 'GUAYAS', 'GASOLINA', 'AUTOMOVIL')
    ) t(anio, mes, marca, provincia, combustible, clase)""")
    c.execute("ALTER TABLE vehiculos ADD COLUMN fecha_proceso TIMESTAMP DEFAULT TIMESTAMP '2026-02-20'")
    return c


@pytest.fixture
def client(con):
    corpus = pd.DataFrame({"doc_id": ["a", "b"], "doc_type": ["marca_anual"] * 2, "anio": [2026, 2026],
                           "marca": ["KIA", "TOYOTA"], "fuente": ["SRI datos abiertos (procesado)"] * 2,
                           "pagina": [None, None], "text": FRAG})
    retriever = HybridRetriever(corpus, dense=FakeDense())
    app = create_app(retriever=retriever, con=con, llm_factory=lambda m: FakeLLM(m), default_model="fake")
    return TestClient(app)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["corpus_docs"] == 2 and r["models"] == ["fake", "otro"]


def test_search(client):
    r = client.post("/search", json={"query": "registros de Kia 2026", "k": 1}).json()
    assert r["hits"][0]["doc_id"] == "a" and r["parsed_filters"]["marca"] == ["KIA"]


def test_ask_respuesta_verificada(client):
    r = client.post("/ask", json={"question": "¿Cuántos vehículos registró Kia en 2026?", "k": 2}).json()
    assert r["grounded"] and not r["refused"] and "[1]" in r["answer"] and r["model"] == "fake"
    assert r["sources"][0]["n"] == 1 and r["sources"][0]["fuente"].startswith("SRI")


def test_ask_validaciones(client):
    assert client.post("/ask", json={"question": "ab"}).status_code == 422
    assert client.post("/ask", json={"question": "¿Cuántos Kia?", "k": 99}).status_code == 422
    r = client.post("/ask", json={"question": "¿Cuántos Kia?", "model": "inexistente"})
    assert r.status_code == 400 and "no instalado" in r.json()["detail"]


def test_ask_ollama_caido(con):
    class Caido(FakeLLM):
        def installed_models(self):
            raise OllamaUnavailable("Ollama no responde")

        def chat(self, *a, **k):
            raise OllamaUnavailable("Ollama no responde")
    corpus = pd.DataFrame({"doc_id": ["a", "b"], "doc_type": ["x", "x"], "text": FRAG})
    app = create_app(retriever=HybridRetriever(corpus, dense=FakeDense()), con=con, llm_factory=lambda m: Caido(m),
                     default_model="fake")
    r = TestClient(app).post("/ask", json={"question": "¿Cuántos Kia en 2026?"})
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]


def test_stats(client):
    meta = client.get("/stats/meta").json()
    assert meta["anios"] == [2025, 2026] and meta["marcas"][0] == "KIA" and "GUAYAS" in meta["provincias"]
    mensual = client.get("/stats/mensual", params={"marca": "KIA"}).json()
    assert {(m["anio"], m["mes"]): m["registros"] for m in mensual}[(2026, 1)] == 2
    top = client.get("/stats/top", params={"dimension": "marca", "anio": 2026}).json()
    assert top[0] == {"nombre": "KIA", "registros": 3, "participacion": 75.0}
    k = client.get("/stats/kpis").json()
    # mismo rango de meses (ene-feb): 2026 = 4 registros vs 2025 = 3
    assert k["actual"]["registros"] == 4 and k["anterior"]["registros"] == 3 and k["variacion_pct"] == 33.3
    assert k["periodo"] == "enero a febrero" and k["actual"]["electricos"] == 1


def test_stats_rechaza_dimension_no_permitida(client):
    r = client.get("/stats/top", params={"dimension": "marca; DROP TABLE vehiculos", "anio": 2026})
    assert r.status_code == 400
    assert client.get("/stats/meta").status_code == 200  # la tabla sigue ahi
