import numpy as np
import pandas as pd

from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.hybrid import HybridRetriever, rrf


class FakeDense:
    """Indice semantico de juguete: puntaje por superposicion de palabras de la consulta."""
    def __init__(self, texts):
        self.texts = texts

    def scores(self, query):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) for t in self.texts], dtype=np.float32)


CORPUS = pd.DataFrame({
    "doc_id": ["a", "b", "c", "d"],
    "doc_type": ["marca_anual", "marca_anual", "boletin_aeade", "marca_anual"],
    "anio": [2025, 2026, 2026, 2026],
    "text": ["Marca TOYOTA año 2025 registros 2.433 Hilux",
             "Marca KIA año 2026 registros 15.649 Soluto",
             "Boletín AEADE septiembre 2026 récord 15.649 vehículos vendidos",
             "Marca CHEVROLET año 2026 registros 10.977 D-Max"],
})


def test_tokenize_normaliza_numeros_y_tildes():
    assert tokenize("Récord de 15.649 vehículos") == ["record", "15649", "vehiculos"]
    assert tokenize("CAMIÓN 1.5") == ["camion", "1", "5"]


def test_bm25_prioriza_coincidencia_exacta():
    idx = BM25Index(CORPUS["text"].tolist())
    s = idx.scores("Hilux")
    assert s.argmax() == 0 and s[1] == 0
    assert idx.scores("15.649")[[1, 2]].min() > 0
    assert idx.scores("palabra inexistente").sum() == 0


def test_rrf_favorece_consenso():
    fused = rrf([[1, 2, 3], [2, 9, 8]])
    assert [d for d, _ in fused][:2] == [2, 1]  # 2 aparece en ambos rankings
    assert dict(fused)[2] > dict(fused)[1] > dict(fused)[9]


def test_hybrid_y_filtros():
    r = HybridRetriever(CORPUS, dense=FakeDense(CORPUS["text"].tolist()))
    hits = r.search("registros de Kia 2026", k=2)
    assert hits[0].doc_id == "b" and set(hits[0].ranks) == {"bm25", "dense"}
    solo_pdf = r.search("15.649", k=5, filters={"doc_type": "boletin_aeade"})
    assert [h.doc_id for h in solo_pdf] == ["c"]
    anios = r.search("registros", k=5, filters={"anio": [2025]})
    assert {h.doc_id for h in anios} == {"a"}


def test_modo_dense_sin_indice_falla_claro():
    r = HybridRetriever(CORPUS)
    try:
        r.search("kia", mode="dense")
    except ValueError as e:
        assert "semantico" in str(e)
    else:
        raise AssertionError("debia fallar")


def test_query_parser():
    from src.retrieval.query import QueryParser
    corpus = pd.DataFrame({
        "marca": ["TOYOTA", "GREAT WALL", "KIA", None, None],
        "provincia": [None, None, None, "MANABI", None],
        "combustible": [None, None, None, None, "ELECTRICO"],
        "anio": [2025, 2026, 2026, 2026, 2026],
    })
    p = QueryParser(corpus)
    assert p.parse("ventas de Toyota durante 2025") == {
        "anio": [2025], "marca": ["TOYOTA"], "doc_type": ["marca_anual", "provincia_anual", "combustible_anual"]}
    f = p.parse("registros de Kia en julio de 2026")
    assert f["mes"] == [7] and f["marca"] == ["KIA"] and "doc_type" not in f
    assert p.parse("GWM en 2026")["marca"] == ["GREAT WALL"]
    assert p.parse("autos eléctricos en Manabí 2026")["combustible"] == ["ELECTRICO"]
    assert p.parse("Kia frente a 2025 y 2026")["doc_type"] == ["comparativo_periodo"]
    assert p.parse("hola qué tal") == {}
    assert p.parse("qué provincia lidera en 2026")["doc_type"] == ["ranking"]


def test_query_parser_mes_sin_entidad_y_ranking_amplio():
    from src.retrieval.query import QueryParser
    corpus = pd.DataFrame({"marca": ["KIA"], "anio": [2025]})
    p = QueryParser(corpus)
    assert p.parse("total de vehículos registrados en enero de 2025")["doc_type"] == ["mercado_mensual"]
    assert p.parse("registros de Kia en enero de 2025").get("doc_type") is None
    assert p.parse("qué combustible es el más usado en 2025")["doc_type"] == ["ranking"]
