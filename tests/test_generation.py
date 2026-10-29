import pandas as pd
import pytest
import requests

from src.generation import guardrails as g
from src.generation.ollama import OllamaClient, OllamaUnavailable
from src.generation.rag import REFUSAL, RAGPipeline
from src.retrieval.bm25 import tokenize
from src.retrieval.hybrid import HybridRetriever

FRAG = ["Marca KIA, año 2026. Registros: 15.649 (8,1 % del segmento), puesto 1.",
        "AEADE: crecimiento 40.3% entre Ene-Sep. Total 123.927."]


def test_canon_unifica_formatos():
    assert g.canon("15.649") == "15649"
    assert g.canon("3,1") == g.canon("3.1") == "3.1"
    assert g.canon("1.234,5") == "1234.5"
    assert g.numbers("Total $29.999 y 40,3 %") == {"29999", "40.3"}


def test_respuesta_respaldada():
    v = g.validate("Kia registró 15.649 vehículos [1], el 8,1 % [1].", FRAG, "¿cuántos Kia en 2026?")
    assert v.grounded and not v.problems


def test_cifra_calculada_se_marca():
    v = g.validate("Kia registró 15.649 [1], un 12 % más que el año previo [1].", FRAG)
    assert not v.grounded and v.unsupported_numbers == ["12"]


def test_citas_invalidas_o_ausentes():
    assert g.validate("Kia registró 15.649 [4].", FRAG).invalid_citations == [4]
    v = g.validate("Kia registró 15.649 vehículos.", FRAG)
    assert not v.grounded and not v.has_citations
    assert g.validate(REFUSAL, FRAG, refusal=REFUSAL).grounded


def test_anio_de_la_pregunta_es_valido():
    assert g.validate("En 2026 no hay datos de esa marca [1].", FRAG, "marcas en 2026").grounded


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, temperature=0.0):
        self.calls.append(messages)
        return self.replies.pop(0)


class FakeDense:
    def __init__(self, texts):
        self.texts = texts

    def scores(self, q):
        import numpy as np
        qs = set(tokenize(q))
        return np.array([len(qs & set(tokenize(t))) for t in self.texts], dtype=np.float32)


@pytest.fixture
def retriever():
    corpus = pd.DataFrame({"doc_id": ["a", "b"], "doc_type": ["marca_anual", "boletin_aeade"],
                           "fuente": ["SRI datos abiertos (procesado)", "AEADE (datos del SRI)"],
                           "text": FRAG, "anio": [2026, 2026], "marca": ["KIA", None]})
    return HybridRetriever(corpus, dense=FakeDense(FRAG))


def test_pipeline_respuesta_correcta(retriever):
    llm = FakeLLM("Kia registró 15.649 vehículos en 2026 [1].")
    ans = RAGPipeline(retriever, llm, k=2).answer("¿Cuántos vehículos registró Kia en 2026?")
    assert ans.verdict.grounded and not ans.retried and len(llm.calls) == 1
    assert "Fragmentos" in llm.calls[0][1]["content"] and "[1] (fuente:" in llm.calls[0][1]["content"]


def test_pipeline_reintenta_y_se_corrige(retriever):
    llm = FakeLLM("Kia creció 99 % [1].", "Kia registró 15.649 vehículos [1].")
    ans = RAGPipeline(retriever, llm, k=2).answer("¿Cuántos vehículos registró Kia en 2026?")
    assert ans.retried and ans.verdict.grounded and "15.649" in ans.text
    assert "cifras sin respaldo: 99" in llm.calls[1][-1]["content"]


def test_pipeline_rechaza_si_sigue_sin_respaldo(retriever):
    llm = FakeLLM("Kia creció 99 % [1].", "Kia creció 98 % [1].")
    ans = RAGPipeline(retriever, llm, k=2).answer("¿Cuánto creció Kia?")
    assert not ans.verdict.grounded and ans.text.startswith(REFUSAL) and "98" in ans.text


def test_ollama_no_disponible(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("sin servidor")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(OllamaUnavailable, match="ollama pull"):
        OllamaClient("qwen2.5:3b-instruct").check()
