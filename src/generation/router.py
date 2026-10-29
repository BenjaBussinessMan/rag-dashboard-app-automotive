"""Decide si una pregunta se resuelve con SQL (cifras exactas del SRI) o con recuperacion de documentos."""

import re

from src.generation.rag import REFUSAL, Answer
from src.generation.guardrails import Verdict
from src.retrieval.query import norm

NUMERIC = re.compile(
    r"\b(cuant[oa]s?|total|suma|sumar|promedio|media|mediana|porcentaje|proporcion|cuota|participacion|"
    r"numero de|cantidad de|maximo|minimo|mas caro|mas barato|crecimiento|crecio|crecieron|variacion|diferencia)\b")
# Fuentes que solo existen como texto: el SQL no las cubre.
DOCUMENT_ONLY = re.compile(r"\b(aeade|boletin|boletines|pdf|diccionario|variable|significa|definicion)\b")


def wants_sql(question: str) -> bool:
    q = norm(question).lower()
    return bool(NUMERIC.search(q)) and not DOCUMENT_ONLY.search(q)


class Assistant:
    """route: 'auto' (heuristica, con caida al RAG), 'sql' o 'rag'."""

    def __init__(self, rag, sql=None):
        self.rag, self.sql = rag, sql

    def answer(self, question: str, route: str = "auto") -> Answer:
        if route in ("auto", "sql") and self.sql is not None and (route == "sql" or wants_sql(question)):
            ans = self.sql.answer(question)
            if ans is not None:
                return ans
            if route == "sql":
                return Answer(question, f"{REFUSAL} (No pude construir una consulta SQL válida para esta pregunta.)",
                              Verdict(grounded=True), route="sql")
        return self.rag.answer(question)
