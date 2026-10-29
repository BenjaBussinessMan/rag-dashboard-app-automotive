"""Pipeline RAG: recuperar -> generar con citas -> validar (guardrail) -> reintentar una vez."""

from dataclasses import dataclass, field

from src.generation.guardrails import Verdict, attach_citations

REFUSAL = "No encuentro esa información en los documentos disponibles."

SYSTEM_PROMPT = f"""Eres un analista del mercado automotriz de Ecuador. Respondes en español, de forma breve y precisa.
Reglas estrictas:
1. Usa SOLO la información de los fragmentos numerados que se te entregan. No uses conocimiento externo.
2. Copia las cifras tal como aparecen en los fragmentos. NO calcules, redondees ni estimes cifras nuevas.
3. No escribas citas ni copies los encabezados "(fuente: ...; id: ...)": el sistema agrega las citas. Cada oración
   debe apoyarse en UN solo fragmento: no combines cifras de fragmentos distintos en una misma oración.
4. Los fragmentos vienen de dos fuentes: "SRI datos abiertos (procesado)" y "AEADE (datos del SRI)". Sus cifras no
   coinciden entre sí; nunca las mezcles ni las sumes. Di de qué fuente viene cada cifra.
5. Si un fragmento indica "Año incompleto", mencionalo cuando hables de ese año.
6. Si los fragmentos no contienen la respuesta, responde exactamente: "{REFUSAL}"
"""


@dataclass
class Answer:
    question: str
    text: str
    verdict: Verdict
    hits: list = field(default_factory=list)
    retried: bool = False
    filters: dict = field(default_factory=dict)
    route: str = "rag"          # "rag" (documentos) o "sql" (consulta exacta)
    sql: str | None = None
    table: list | None = None


def format_fragments(hits) -> list[str]:
    """Texto exacto que ve el modelo para cada fragmento (el guardrail valida contra este mismo texto)."""
    return [f"[{i}] (fuente: {h.metadata.get('fuente') or 'desconocida'}; id: {h.doc_id})\n{h.text}"
            for i, h in enumerate(hits, start=1)]


def format_context(hits) -> str:
    return "\n\n".join(format_fragments(hits))


class RAGPipeline:
    def __init__(self, retriever, llm, k: int = 5, rerank: bool = False):
        self.retriever, self.llm, self.k, self.rerank = retriever, llm, k, rerank

    def _messages(self, question: str, context: str) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Fragmentos:\n\n{context}\n\nPregunta: {question}"}]

    def answer(self, question: str) -> Answer:
        hits = self.retriever.search(question, k=self.k, mode="hybrid", understand=True, rerank=self.rerank)
        filters = self.retriever.parser.parse(question)
        if not hits:
            return Answer(question, REFUSAL, Verdict(grounded=True), [], False, filters)

        fragments = format_fragments(hits)
        messages = self._messages(question, format_context(hits))
        raw = self.llm.chat(messages)
        text, verdict = attach_citations(raw, fragments, question, REFUSAL)

        retried = False
        if not verdict.grounded:
            retried = True
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "Tu respuesta tiene problemas: " + "; ".join(verdict.problems) + ". Reescríbela usando únicamente "
                    "cifras copiadas literalmente de los fragmentos, sin calcular nada, una idea por oración. "
                    f"Si no puedes, responde exactamente: \"{REFUSAL}\"")}]
            text = self.llm.chat(messages)
            text, verdict = attach_citations(text, fragments, question, REFUSAL)

        if not verdict.grounded:
            text = (f"{REFUSAL} (La respuesta generada no pudo verificarse contra las fuentes: "
                    f"{'; '.join(verdict.problems)}.)")
        return Answer(question, text, verdict, hits, retried, filters)
