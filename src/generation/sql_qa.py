"""Preguntas de cifras exactas: el LLM escribe SQL, DuckDB calcula y la respuesta se redacta solo desde el resultado."""

import re

import duckdb
import pandas as pd

from src.analytics.documents import MESES, n, pct
from src.analytics.query import UnsafeQuery, run_select
from src.generation.guardrails import Verdict, attach_citations
from src.generation.rag import REFUSAL, Answer
from src.retrieval.hybrid import Hit

SOURCE = "SRI datos abiertos (consulta SQL)"
MAX_ROWS = 30

SQL_SYSTEM = """Eres un experto en DuckDB. Escribe UNA consulta SQL que responda la pregunta sobre la tabla `vehiculos`.
Cada fila es un vehículo nuevo registrado en el SRI de Ecuador.

Columnas: anio (año de la fecha de proceso), mes (1-12), marca, modelo, pais (de fabricación), anio_modelo, clase,
subclase, tipo (LIVIANO o PESADO), avaluo (USD), fecha_proceso, tipo_servicio, cilindraje, combustible, color_1, color_2,
tipo_persona (NATURAL o JURIDICA), tipo_transaccion (COMPRA LOCAL o IMPORTACION DIRECTA), canton_nombre, provincia.

Reglas:
- Solo SELECT (puede usar WITH). Una sola sentencia, sin comentarios, solo la tabla `vehiculos`.
- Texto en MAYÚSCULAS y sin tildes: 'PICHINCHA', 'ELECTRICO', 'GREAT WALL', 'MANABI'.
- combustible: GASOLINA, DIESEL, ELECTRICO, HIBRIDO_GASOLINA_BATERIAS, HIBRIDO_DIESEL_BATERIAS (híbridos: combustible LIKE 'HIBRIDO%').
- clase: AUTOMOVIL, CAMIONETA, CAMION.
- Conteos: count(*) con alias `registros`. Porcentajes: round(100.0 * parte / total, 1) con alias `porcentaje`. Promedio/mediana de avalúo: round(...) con alias `avaluo_promedio` / `avaluo_mediano`.
- Si la pregunta no menciona un año, no filtres por año: agrupa por anio.
- Al comparar años, usa el mismo rango de meses en ambos.
- Responde solo con el SQL dentro de un bloque ```sql.

Ejemplos:
Pregunta: ¿Cuántos Kia se registraron en 2025?
```sql
SELECT count(*) AS registros FROM vehiculos WHERE marca = 'KIA' AND anio = 2025
```
Pregunta: ¿Qué porcentaje de los registros de 2026 fueron eléctricos?
```sql
SELECT round(100.0 * count(*) FILTER (WHERE combustible = 'ELECTRICO') / count(*), 1) AS porcentaje FROM vehiculos WHERE anio = 2026
```
Pregunta: ¿Cuáles son las 3 marcas con más registros en Guayas en 2025?
```sql
SELECT marca, count(*) AS registros FROM vehiculos WHERE provincia = 'GUAYAS' AND anio = 2025 GROUP BY marca ORDER BY registros DESC LIMIT 3
```
"""

ANSWER_SYSTEM = f"""Eres un analista del mercado automotriz de Ecuador. Respondes en español, de forma breve.
Se te entrega el RESULTADO de una consulta SQL ya ejecutada. Reglas:
1. Usa SOLO las cifras del resultado, tal como aparecen. No calcules, redondees ni estimes nada.
2. Una idea por oración. No escribas citas ni copies encabezados: el sistema agrega la fuente.
3. Si el alcance indica un año incompleto y respondes sobre ese año, menciónalo.
4. La consulta se escribió para esta pregunta y el resultado ya está calculado: responde con él, indicando
   el filtro aplicado (año, marca, provincia...) tal como figura en la consulta. No digas que no hay información.
"""


def extract_sql(text: str) -> str | None:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = m.group(1) if m else text
    m2 = re.search(r"\b(with|select)\b.*", candidate, re.DOTALL | re.IGNORECASE)
    return m2.group(0).strip() if m2 else None


def _cell(v) -> str:
    v = v.item() if hasattr(v, "item") else v
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int):
        return str(v) if 1900 <= v <= 2100 else n(v)  # anios sin separador de miles
    if isinstance(v, float):
        if v == int(v):
            return n(v)
        decimales = 1 if round(v, 1) == round(v, 2) else 2  # 8,1 pero 24.295,77: no se pierde precision
        entero, frac = f"{abs(v):.{decimales}f}".split(".")
        return f"{'-' if v < 0 else ''}{n(int(entero))},{frac}"
    return str(v)


def render_rows(df: pd.DataFrame) -> str:
    rows = df.astype(object).to_dict("records")  # iterrows mezclaria enteros y decimales en una sola fila
    return "\n".join("; ".join(f"{c}: {_cell(v)}" for c, v in row.items()) for row in rows)


class SQLPipeline:
    def __init__(self, con: duckdb.DuckDBPyConnection, llm, parser, clases=("AUTOMOVIL", "CAMION", "CAMIONETA")):
        self.con, self.llm, self.parser, self.clases = con, llm, parser, clases

    def scope(self) -> str:
        rows = self.con.execute("SELECT anio, max(fecha_proceso) FROM vehiculos GROUP BY anio ORDER BY anio").fetchall()
        partes = []
        for anio, mx in rows:
            partes.append(f"{anio}" + ("" if mx.month == 12 else f" (incompleto, hasta {mx.day:02d}/{mx.month:02d}/{mx.year})"))
        return f"años {', '.join(partes)}; clases {', '.join(c.lower() for c in self.clases)}"

    def _hints(self, question: str) -> str:
        f = self.parser.parse(question)
        parts = []
        if "anio" in f:
            parts.append("anio IN (" + ", ".join(map(str, f["anio"])) + ")")
        if "mes" in f:
            parts.append("mes IN (" + ", ".join(map(str, f["mes"])) + ")")
        for col in ("marca", "provincia", "combustible"):
            if col in f:
                parts.append(f"{col} IN (" + ", ".join(f"'{v}'" for v in f[col]) + ")")
        return ("\nValores reconocidos en la pregunta (úsalos exactos): " + "; ".join(parts)) if parts else ""

    def generate(self, question: str):
        """Devuelve (sql, DataFrame) o None si no se logra una consulta valida y ejecutable."""
        messages = [{"role": "system", "content": SQL_SYSTEM + f"\nAlcance de los datos: {self.scope()}."},
                    {"role": "user", "content": f"Pregunta: {question}{self._hints(question)}"}]
        for attempt in range(2):
            raw = self.llm.chat(messages)
            sql = extract_sql(raw)
            try:
                if not sql:
                    raise ValueError("no se encontró una consulta SQL en la respuesta")
                return sql, run_select(self.con, sql, MAX_ROWS)
            except (UnsafeQuery, duckdb.Error, ValueError) as exc:
                if attempt == 1:
                    return None
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Esa consulta falló: {exc}. Corrígela y responde solo con el SQL."}]
        return None

    def _loaded_years(self) -> set[int]:
        return {r[0] for r in self.con.execute("SELECT DISTINCT anio FROM vehiculos").fetchall()}

    def answer(self, question: str) -> Answer | None:
        """None = no se pudo resolver con SQL (el llamador cae al RAG por documentos)."""
        # Un año fuera de la base daria "0 registros", que seria enganoso: se aclara el alcance.
        missing = sorted({int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)} - self._loaded_years())
        if missing:
            msg = (f"{REFUSAL} (La base cargada no incluye {', '.join(map(str, missing))}; cubre {self.scope()}.)")
            return Answer(question, msg, Verdict(grounded=True), [], False, self.parser.parse(question), "sql")
        generated = self.generate(question)
        if generated is None:
            return None
        sql, df = generated
        table = df.astype(object).where(df.notna(), None).to_dict("records")
        filters = self.parser.parse(question)

        if df.empty:
            msg = f"{REFUSAL} (La consulta no devolvió filas; la base cargada cubre {self.scope()}.)"
            return Answer(question, msg, Verdict(grounded=True), [], False, filters, "sql", sql, table)

        one_line = " ".join(sql.split())
        fragment = (f"[1] (fuente: {SOURCE}; id: sql:resultado)\n"
                    f"Resultado de una consulta SQL sobre los registros del SRI. Alcance: {self.scope()}.\n"
                    f"Consulta ejecutada: {one_line}\nResultado:\n{render_rows(df)}")
        hit = Hit(doc_id="sql:resultado", text=fragment.split("\n", 1)[1], score=1.0,
                  metadata={"fuente": SOURCE, "doc_type": "sql"})
        messages = [{"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": f"{fragment}\n\nPregunta: {question}"}]

        def compose(raw: str):
            # con un resultado no vacio, un rechazo del modelo no es una respuesta valida
            if REFUSAL.lower() in raw.lower():
                return raw, Verdict(grounded=False, unsupported_numbers=["el modelo rechazó responder con un resultado disponible"])
            return attach_citations(raw, [fragment], question, REFUSAL)

        raw = self.llm.chat(messages)
        text, verdict = compose(raw)

        retried = False
        if not verdict.grounded:
            retried = True
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": "Tu respuesta tiene problemas: " + "; ".join(verdict.problems) +
                          ". El resultado SÍ responde la pregunta. Reescríbela usando únicamente las cifras del "
                          "resultado, sin calcular nada."}]
            raw = self.llm.chat(messages)
            text, verdict = compose(raw)

        if not verdict.grounded:
            # el resultado de la base es confiable aunque el modelo no lo redacte bien: se muestra tal cual
            text = "Resultado de la consulta: " + render_rows(df.head(5)).replace("\n", " | ") + " [1]"
            verdict = Verdict(grounded=True)
        return Answer(question, text, verdict, [hit], retried, filters, "sql", sql, table)
