"""Validacion de la respuesta: toda cifra y toda cita deben estar respaldadas por los fragmentos recuperados."""

import re
from dataclasses import dataclass, field

NUMBER = re.compile(r"\d[\d.,]*\d|\d")
CITATION = re.compile(r"\[(\d+)\]")
THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")


def canon(token: str) -> str:
    """'15.649' -> '15649'; '3,1' y '3.1' -> '3.1'; '$29.999' -> '29999'."""
    token = token.strip(".,")
    if THOUSANDS.fullmatch(token):
        return token.replace(".", "")
    token = token.replace(".", "").replace(",", ".") if "," in token and "." in token else token.replace(",", ".")
    return token


def numbers(text: str, drop_citations: bool = False) -> set[str]:
    if drop_citations:
        text = CITATION.sub(" ", text)
    return {canon(t) for t in NUMBER.findall(text)}


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
            out.append("cifras sin respaldo: " + ", ".join(self.unsupported_numbers))
        if self.invalid_citations:
            out.append("citas inexistentes: " + ", ".join(f"[{c}]" for c in self.invalid_citations))
        if not self.has_citations:
            out.append("la respuesta no cita ningún fragmento")
        return out


def validate(answer: str, fragments: list[str], question: str = "", refusal: str = "") -> Verdict:
    """Cifras permitidas: las presentes en los fragmentos o en la pregunta (p. ej. el año consultado)."""
    allowed = numbers(question)
    for frag in fragments:
        allowed |= numbers(frag)
    found = numbers(answer, drop_citations=True)
    unsupported = sorted(found - allowed, key=lambda x: (len(x), x))

    cited = [int(c) for c in CITATION.findall(answer)]
    invalid = sorted({c for c in cited if not 1 <= c <= len(fragments)})
    declines = bool(refusal) and refusal.lower() in answer.lower()
    has_citations = bool(cited) or declines
    return Verdict(grounded=not unsupported and not invalid and has_citations,
                   unsupported_numbers=unsupported, invalid_citations=invalid, has_citations=has_citations)
