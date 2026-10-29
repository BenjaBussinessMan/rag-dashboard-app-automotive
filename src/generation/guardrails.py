"""Validacion de la respuesta: cada cifra debe estar respaldada por el fragmento que la cita en esa misma oracion."""

import re
from dataclasses import dataclass, field

NUMBER = re.compile(r"\d[\d.,]*\d|\d")
CITATION = re.compile(r"\[(\d+)\]")
THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def canon(token: str) -> str:
    """'15.649' -> '15649'; '3,1' y '3.1' -> '3.1'; '07' -> '7'."""
    token = token.strip(".,")
    if THOUSANDS.fullmatch(token):
        return token.replace(".", "")
    token = token.replace(".", "").replace(",", ".") if "," in token and "." in token else token.replace(",", ".")
    return token.lstrip("0") or "0" if token.isdigit() else token


def numbers(text: str, drop_citations: bool = False) -> set[str]:
    if drop_citations:
        text = CITATION.sub(" ", text)
    return {canon(t) for t in NUMBER.findall(text)}


def sentences(answer: str) -> list[str]:
    """Divide en oraciones; una cita suelta tras el punto ('... 2.433. [1]') se une a la oracion previa."""
    out: list[str] = []
    for part in (p.strip() for p in SENTENCE_SPLIT.split(answer)):
        if not part:
            continue
        if out and not CITATION.sub("", part).strip(" .,;:"):
            out[-1] += " " + part
        else:
            out.append(part)
    return out


@dataclass
class Verdict:
    grounded: bool
    unsupported_numbers: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    has_citations: bool = True

    @property
    def problems(self) -> list[str]:
        out = []
        if self.unsupported_numbers:
            out.append("cifras o afirmaciones sin respaldo en un único fragmento: " + ", ".join(self.unsupported_numbers))
        if self.invalid_citations:
            out.append("citas inexistentes: " + ", ".join(f"[{c}]" for c in self.invalid_citations))
        if not self.has_citations:
            out.append("la respuesta no cita ningún fragmento")
        return out


def validate(answer: str, fragments: list[str], question: str = "", refusal: str = "") -> Verdict:
    """Regla por oracion: toda cifra debe figurar en alguno de los fragmentos [n] citados en esa oracion.

    Las cifras de la pregunta (p. ej. el año consultado) se aceptan sin cita. `fragments` debe incluir el
    mismo encabezado (fuente, id) que ve el modelo, para que copiar un id no cuente como cifra inventada.
    """
    asked = numbers(question)
    frag_numbers = [numbers(f) for f in fragments]
    unsupported: list[str] = []
    invalid: set[int] = set()
    all_cited: list[int] = []

    for sent in sentences(answer):
        cited = [int(c) for c in CITATION.findall(sent)]
        all_cited += cited
        invalid |= {c for c in cited if not 1 <= c <= len(fragments)}
        allowed = set(asked)
        for c in cited:
            if 1 <= c <= len(fragments):
                allowed |= frag_numbers[c - 1]
        unsupported += [n for n in numbers(sent, drop_citations=True) if n not in allowed]

    unsupported = sorted(set(unsupported), key=lambda x: (len(x), x))
    declines = bool(refusal) and refusal.lower() in answer.lower()
    has_citations = bool(all_cited) or declines
    return Verdict(grounded=not unsupported and not invalid and has_citations,
                   unsupported_numbers=unsupported, invalid_citations=sorted(invalid), has_citations=has_citations)


# --- atribucion de citas por el sistema -------------------------------------------------------------

HEADER_ECHO = re.compile(
    r"\[\d+\]\s*\(fuente:[^)]*\)"          # encabezado completo copiado
    r"|\(\s*fuente:[^()]*(?:\([^()]*\)[^()]*)*\)"  # encabezado cerrado (admite un nivel de parentesis)
    r"|\(\s*fuente:[^\n.]*"                # '(fuente: ...' sin cerrar: hasta el fin de la oracion
    r"|\bid:\s*\S+"
    r"|\bseg[uú]n (?:el|los) fragmentos?\s*\d*\s*,?",  # 'Según el fragmento ,' queda huerfano al quitar el encabezado
    re.IGNORECASE)
MIN_OVERLAP = 0.6


def _content_words(text: str) -> set[str]:
    from src.retrieval.bm25 import tokenize
    return {t for t in tokenize(text) if len(t) >= 4 and not t.isdigit()}


def attach_citations(answer: str, fragments: list[str], question: str = "", refusal: str = "") -> tuple[str, Verdict]:
    """El sistema, no el modelo, decide las citas.

    - Oracion con cifras: todas deben aparecer en UN mismo fragmento (se cita el de mejor posicion).
    - Oracion sin cifras: al menos el 60 % de sus palabras de contenido debe estar en un fragmento.
    - Se descartan las citas y los encabezados que el modelo haya escrito.
    """
    if refusal and refusal.lower() in answer.lower():
        return answer, Verdict(grounded=True)

    asked = numbers(question)
    frag_numbers = [numbers(f) for f in fragments]
    frag_words = [_content_words(f) for f in fragments]
    cleaned = HEADER_ECHO.sub(" ", CITATION.sub(" ", answer))
    out: list[str] = []
    unsupported: list[str] = []
    cited_any = False

    for sent in sentences(cleaned):
        sent = re.sub(r"\s+", " ", sent).strip()
        if not sent.strip(" .,;:-"):
            continue
        nums = numbers(sent) - asked
        if nums:
            ok = next((i for i, fn in enumerate(frag_numbers) if nums <= fn), None)
            if ok is None:
                best = max(frag_numbers, key=lambda fn: len(nums & fn), default=set())
                unsupported += list(nums - best)
                out.append(sent)
            else:
                out.append(f"{sent} [{ok + 1}]")
                cited_any = True
        else:
            words = _content_words(sent)
            scores = [len(words & fw) / len(words) if words else 0 for fw in frag_words]
            best = max(range(len(scores)), key=scores.__getitem__) if scores else None
            if words and best is not None and scores[best] >= MIN_OVERLAP:
                out.append(f"{sent} [{best + 1}]")
                cited_any = True
            else:
                unsupported.append("«" + sent[:40] + "»")
                out.append(sent)

    unsupported = sorted(set(unsupported), key=lambda x: (len(x), x))
    return " ".join(out), Verdict(grounded=not unsupported and cited_any, unsupported_numbers=unsupported,
                                  has_citations=cited_any)
