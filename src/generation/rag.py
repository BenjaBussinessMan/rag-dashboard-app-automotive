"""Pipeline RAG: recuperar -> generar con citas -> validar (guardrail) -> reintentar una vez."""

from dataclasses import dataclass, field

from src.generation.guardrails import Verdict, validate

REFUSAL = "No encuentro esa información en los documentos disponibles."

SYSTEM_PROMPT = f"""Eres un analista del mercado automotriz de Ecuador. Respondes en español, de forma breve y precisa.
Reglas estrictas:
1. Usa SOLO la información de los fragmentos numerados que se te entregan. No uses conocimiento externo.
2. Copia las cifras tal como aparecen en los fragmentos. NO calcules, redondees ni estimes cifras nuevas.
3. Después de cada dato escribe la cita del fragmento, por ejemplo [1] o [2].
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


def format_context(hits) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        fuente = h.metadata.get("fuente") or "desconocida"
        parts.append(f"[{i}] (fuente: {fuente}; id: {h.doc_id})\n{h.text}")
    return "\n\n".join(parts)


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

        fragments = [h.text for h in hits]
        messages = self._messages(question, format_context(hits))
        text = self.llm.chat(messages)
        verdict = validate(text, fragments, question, REFUSAL)

        retried = False
        if not verdict.grounded:
            retried = True
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    "Tu respuesta tiene problemas: " + "; ".join(verdict.problems) + ". Reescríbela usando únicamente "
                    "cifras copiadas literalmente de los fragmentos, sin calcular nada, y cita [n] tras cada dato. "
                    f"Si no puedes, responde exactamente: \"{REFUSAL}\"")}]
            text = self.llm.chat(messages)
            verdict = validate(text, fragments, question, REFUSAL)

        if not verdict.grounded:
            text = (f"{REFUSAL} (La respuesta generada no pudo verificarse contra las fuentes: "
                    f"{'; '.join(verdict.problems)}.)")
        return Answer(question, text, verdict, hits, retried, filters)
