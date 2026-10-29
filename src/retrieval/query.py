"""Comprension de la consulta: extrae filtros de metadatos (año, mes, marca, provincia, combustible, tipo)."""

import re
import unicodedata

import pandas as pd

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
         "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
MARCA_ALIAS = {"GWM": "GREAT WALL", "GREAT WALL MOTOR": "GREAT WALL", "MERCEDES": "MERCEDES BENZ"}
COMBUSTIBLE_CUES = {
    "ELECTRICO": ["ELECTRICO"], "ELECTRICOS": ["ELECTRICO"], "BEV": ["ELECTRICO"],
    "DIESEL": ["DIESEL"], "GASOLINA": ["GASOLINA"],
    "HIBRIDO": ["HIBRIDO_GASOLINA_BATERIAS", "HIBRIDO_DIESEL_BATERIAS"],
    "HIBRIDOS": ["HIBRIDO_GASOLINA_BATERIAS", "HIBRIDO_DIESEL_BATERIAS"],
}
COMPARACION = re.compile(r"\b(frente a|comparad[oa]|comparacion|crecimiento|crecio|variacion|versus|vs)\b")
ANUAL_TYPES = ["marca_anual", "provincia_anual", "combustible_anual"]


def norm(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().upper()


class QueryParser:
    def __init__(self, corpus: pd.DataFrame):
        def values(col):
            return corpus[col].dropna().unique() if col in corpus else []

        marcas = {m: m for m in values("marca")}
        self.marcas = {norm(m): m for m in marcas}
        self.marcas.update({k: v for k, v in MARCA_ALIAS.items() if v in marcas.values()})
        self.provincias = {norm(p): p for p in values("provincia")}
        self.anios = {int(a) for a in values("anio")}
        self.combustibles = set(values("combustible"))

    @staticmethod
    def _find(text: str, names) -> list[str]:
        found = [n for n in names if re.search(rf"(?<![A-Z0-9]){re.escape(n)}(?![A-Z0-9])", text)]
        # una marca contenida en otra mas larga ya encontrada (p. ej. 'GREAT' en 'GREAT WALL') se descarta
        return [n for n in found if not any(n != o and n in o for o in found)]

    def parse(self, query: str) -> dict:
        """Devuelve filtros {campo: [valores]} para la fusion; vacio si no se reconoce nada."""
        q = norm(query)
        filters: dict[str, list] = {}

        anios = sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", q)} & self.anios)
        if anios:
            filters["anio"] = anios
        meses = sorted({n for m, n in MESES.items() if re.search(rf"\b{norm(m)}\b", q)})
        if meses:
            filters["mes"] = meses
        marcas = sorted({self.marcas[n] for n in self._find(q, self.marcas)})
        if marcas:
            filters["marca"] = marcas
        provincias = sorted({self.provincias[n] for n in self._find(q, self.provincias)})
        if provincias:
            filters["provincia"] = provincias
        combustibles = sorted({c for tok in re.findall(r"[A-Z]+", q) for c in COMBUSTIBLE_CUES.get(tok, [])}
                              & self.combustibles)
        if combustibles:
            filters["combustible"] = combustibles

        # intencion por tipo de documento
        if COMPARACION.search(norm(query).lower()) and len(anios) >= 1 and not meses:
            filters["doc_type"] = ["comparativo_periodo"]
        elif anios and not meses and (marcas or provincias or combustibles):
            filters["doc_type"] = ANUAL_TYPES
        return filters
