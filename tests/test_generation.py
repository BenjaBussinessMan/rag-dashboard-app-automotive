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


def test_cifra_de_otro_fragmento_no_vale():
    # 123.927 existe, pero en el fragmento [2]; la oracion cita [1]
    v = g.validate("Kia creció 123.927 [1].", FRAG)
    assert not v.grounded and v.unsupported_numbers == ["123927"]
    assert g.validate("Kia registró 15.649 [1]. El total fue 123.927 [2].", FRAG).grounded


def test_cifra_sin_cita_en_su_oracion():
    v = g.validate("Kia registró 15.649 vehículos. El total fue 123.927 [2].", FRAG)
    assert not v.grounded and v.unsupported_numbers == ["15649"]
    assert g.validate("Kia registró 15.649 vehículos. [1]", FRAG).grounded  # cita suelta se une a la oracion


def test_ids_y_ceros_a_la_izquierda():
    frag = ["[1] (fuente: SRI; id: marca_mensual:TOYOTA:2026-07)\nRegistros: 343."]
    assert g.validate("Toyota tuvo 343 registros en 2026-07 [1].", frag).grounded
    assert g.canon("07") == g.canon("7") == "7"


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
    assert "sin respaldo en un único fragmento: 99" in llm.calls[1][-1]["content"]


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


def test_atribucion_de_citas_por_el_sistema():
    texto, v = g.attach_citations("Kia registró 15.649 vehículos. El total fue 123.927.", FRAG)
    assert v.grounded
    assert texto.endswith("[2]") and "[1]" in texto


def test_atribucion_ignora_citas_y_encabezados_del_modelo():
    texto, v = g.attach_citations("[1] (fuente: SRI; id: x:2026-07) Kia registró 15.649 [9].", FRAG)
    assert v.grounded and "[9]" not in texto and "fuente" not in texto and texto.endswith("[1]")


def test_atribucion_rechaza_cifras_mezcladas_o_calculadas():
    _, mezcla = g.attach_citations("Kia registró 15.649 de un total de 123.927.", FRAG)  # fragmentos distintos
    assert not mezcla.grounded
    _, calculada = g.attach_citations("Kia creció 99 %.", FRAG)
    assert not calculada.grounded and calculada.unsupported_numbers == ["99"]


def test_atribucion_oracion_sin_cifras():
    frag = ["[1] (fuente: SRI; id: r)\nPICHINCHA lidera el ranking de provincias con más registros."]
    ok, v = g.attach_citations("Pichincha lidera el ranking de provincias.", frag)
    assert v.grounded and ok.endswith("[1]")
    _, v2 = g.attach_citations("Quito tiene el mejor clima automotriz del continente.", frag)
    assert not v2.grounded
    assert g.attach_citations(REFUSAL, frag, refusal=REFUSAL)[1].grounded


def test_limpia_restos_de_encabezados():
    texto, v = g.attach_citations(
        "Según el fragmento , Kia registró 15.649 vehículos (fuente: SRI datos abiertos (procesado); [1]", FRAG)
    assert v.grounded and "fuente" not in texto and "fragmento" not in texto and texto.endswith("[1]")
    texto2, _ = g.attach_citations("Kia registró 15.649 (fuente: ) [1]", FRAG)
    assert "fuente" not in texto2
