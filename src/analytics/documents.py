"""Genera documentos de texto (con metadatos) a partir de agregados SQL para indexar en el RAG.

El texto contiene solo cifras calculadas por DuckDB; el LLM nunca las inventa.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
MIN_REGISTROS_MARCA = 200  # marcas menores no generan documentos propios


def n(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def money(v) -> str:
    return "$" + n(round(v))


def pct(v) -> str:
    return f"{v:.1f}".replace(".", ",") + " %"


def variacion(antes, ahora) -> str:
    if not antes:
        return "sin base de comparación"
    d = (ahora - antes) / antes * 100
    return f"{'+' if d >= 0 else ''}{pct(d)}"


def _items(pairs, total=None) -> str:
    if total:
        return ", ".join(f"{k} ({n(c)}, {pct(c / total * 100)})" for k, c in pairs)
    return ", ".join(f"{k} ({n(c)})" for k, c in pairs)


def _top(con, keys, item, k) -> dict:
    """{(claves...): [(item, n), ...]} con los k mas frecuentes por grupo."""
    cols = ", ".join(keys)
    rows = con.execute(f"""
        WITH g AS (SELECT {cols}, {item} AS item, count(*) AS c FROM vehiculos
                   WHERE {item} IS NOT NULL GROUP BY {cols}, {item})
        SELECT {cols}, item, c FROM g
        QUALIFY row_number() OVER (PARTITION BY {cols} ORDER BY c DESC, item) <= {k}
        ORDER BY {cols}, c DESC, item
    """).fetchall()
    out: dict = {}
    for *key, it, c in rows:
        out.setdefault(tuple(key), []).append((it, c))
    return out


def _doc(doc_type, doc_id, text, anio=None, mes=None, marca=None, provincia=None, combustible=None):
    return {"doc_id": doc_id, "doc_type": doc_type, "anio": anio, "mes": mes, "marca": marca,
            "provincia": provincia, "combustible": combustible,
            "fuente": "SRI datos abiertos (procesado)", "pagina": None, "text": text}


class CorpusBuilder:
    def __init__(self, con: duckdb.DuckDBPyConnection, clases, dictionary_path: Path | None = None):
        self.con = con
        clases = [c.lower() for c in clases]
        self.segmento = clases[0] if len(clases) == 1 else ", ".join(clases[:-1]) + f" y {clases[-1]}"
        self.dictionary_path = dictionary_path
        rows = con.execute("""SELECT anio, min(fecha_proceso), max(fecha_proceso), count(*)
                              FROM vehiculos GROUP BY anio ORDER BY anio""").fetchall()
        self.cobertura = {a: (mn, mx, c) for a, mn, mx, c in rows}
        self.anios = sorted(self.cobertura)

    def _nota(self, anio) -> str:
        mx = self.cobertura[anio][1]
        if mx.month < 12:
            return f" Año incompleto: datos hasta el {mx.day:02d}/{mx.month:02d}/{mx.year}."
        return ""

    def _ctx(self) -> str:
        return f"Segmento: {self.segmento}. Fuente: SRI Ecuador, vehículos nuevos (fecha de proceso)."

    def mercado_mensual(self) -> list[dict]:
        base = self.con.execute("""
            SELECT anio, mes, count(*), median(avaluo),
                   sum((combustible = 'ELECTRICO')::int), sum((combustible LIKE 'HIBRIDO%')::int)
            FROM vehiculos GROUP BY ALL ORDER BY 1, 2""").fetchall()
        marcas = _top(self.con, ["anio", "mes"], "marca", 5)
        provs = _top(self.con, ["anio", "mes"], "provincia", 3)
        clases = _top(self.con, ["anio", "mes"], "clase", 5)
        combs = _top(self.con, ["anio", "mes"], "combustible", 4)
        docs, prev = [], {}
        for anio, mes, total, av, ele, hib in base:
            ant = prev.get((anio, mes - 1)) if mes > 1 else prev.get((anio - 1, 12))
            var = f" Variación vs mes anterior: {variacion(ant, total)}." if ant is not None else ""
            prev[(anio, mes)] = total
            text = (
                f"Mercado automotriz de Ecuador, {MESES[mes - 1]} de {anio}. {self._ctx()} "
                f"Registros de vehículos nuevos: {n(total)}.{var} Avalúo mediano: {money(av)}. "
                f"Eléctricos: {n(ele)} ({pct(ele / total * 100)}); híbridos: {n(hib)} ({pct(hib / total * 100)}). "
                f"Clases: {_items(clases[(anio, mes)], total)}. "
                f"Combustibles: {_items(combs[(anio, mes)], total)}. "
                f"Marcas líderes: {_items(marcas[(anio, mes)], total)}. "
                f"Provincias líderes: {_items(provs[(anio, mes)], total)}.")
            docs.append(_doc("mercado_mensual", f"mercado_mensual:{anio}-{mes:02d}", text, anio, mes))
        return docs

    def marca_anual(self) -> list[dict]:
        base = self.con.execute(f"""
            WITH t AS (SELECT anio, count(*) total FROM vehiculos GROUP BY anio),
            m AS (SELECT marca, anio, count(*) c, median(avaluo) av FROM vehiculos
                  WHERE marca IS NOT NULL GROUP BY marca, anio)
            SELECT marca, anio, c, av, total,
                   row_number() OVER (PARTITION BY anio ORDER BY c DESC) AS puesto
            FROM m JOIN t USING (anio)
            QUALIFY sum(c) OVER (PARTITION BY marca) >= {MIN_REGISTROS_MARCA}
            ORDER BY anio, puesto""").fetchall()
        modelos = _top(self.con, ["marca", "anio"], "modelo", 3)
        combs = _top(self.con, ["marca", "anio"], "combustible", 3)
        provs = _top(self.con, ["marca", "anio"], "provincia", 3)
        clases = _top(self.con, ["marca", "anio"], "clase", 3)
        docs = []
        for marca, anio, c, av, total, puesto in base:
            k = (marca, anio)
            text = (
                f"Marca {marca}, año {anio}. {self._ctx()} Registros: {n(c)} "
                f"({pct(c / total * 100)} del segmento), puesto {puesto} del ranking de marcas. "
                f"Avalúo mediano: {money(av)}. Clases: {_items(clases[k], c)}. "
                f"Modelos más registrados: {_items(modelos[k])}. "
                f"Combustibles: {_items(combs[k], c)}. Provincias principales: {_items(provs[k], c)}."
                f"{self._nota(anio)}")
            docs.append(_doc("marca_anual", f"marca_anual:{marca}:{anio}", text, anio, marca=marca))
        return docs

    def marca_mensual(self) -> list[dict]:
        rows = self.con.execute(f"""
            WITH m AS (SELECT marca, anio, mes, count(*) c, median(avaluo) av FROM vehiculos
                       WHERE marca IS NOT NULL GROUP BY ALL)
            SELECT marca, anio, mes, c, av FROM m
            QUALIFY sum(c) OVER (PARTITION BY marca) >= {MIN_REGISTROS_MARCA}
            ORDER BY marca, anio, mes""").fetchall()
        docs, prev = [], {}
        for marca, anio, mes, c, av in rows:
            idx = anio * 12 + mes
            ant = prev.get((marca, idx - 1))
            var = f" Variación vs mes anterior: {variacion(ant, c)}." if ant is not None else ""
            prev[(marca, idx)] = c
            text = (f"Marca {marca}, {MESES[mes - 1]} de {anio}. {self._ctx()} "
                    f"Registros: {n(c)}. Avalúo mediano: {money(av)}.{var}")
            docs.append(_doc("marca_mensual", f"marca_mensual:{marca}:{anio}-{mes:02d}", text, anio, mes, marca))
        return docs

    def provincia_anual(self) -> list[dict]:
        base = self.con.execute("""
            WITH t AS (SELECT anio, count(*) total FROM vehiculos GROUP BY anio),
            p AS (SELECT provincia, anio, count(*) c, median(avaluo) av,
                         avg((tipo_persona = 'JURIDICA')::int) * 100 jur,
                         avg((tipo_transaccion LIKE 'IMPORT%')::int) * 100 imp
                  FROM vehiculos WHERE provincia IS NOT NULL GROUP BY provincia, anio)
            SELECT provincia, anio, c, av, total, jur, imp,
                   row_number() OVER (PARTITION BY anio ORDER BY c DESC)
            FROM p JOIN t USING (anio) ORDER BY anio, c DESC""").fetchall()
        marcas = _top(self.con, ["provincia", "anio"], "marca", 5)
        combs = _top(self.con, ["provincia", "anio"], "combustible", 3)
        clases = _top(self.con, ["provincia", "anio"], "clase", 3)
        docs = []
        for prov, anio, c, av, total, jur, imp, puesto in base:
            k = (prov, anio)
            text = (
                f"Provincia {prov}, año {anio}. {self._ctx()} Registros: {n(c)} "
                f"({pct(c / total * 100)} del segmento), puesto {puesto} entre provincias. "
                f"Avalúo mediano: {money(av)}. Compras de personas jurídicas: {pct(jur)}; "
                f"importación directa: {pct(imp)}. Clases: {_items(clases[k], c)}. "
                f"Marcas líderes: {_items(marcas[k], c)}. Combustibles: {_items(combs[k], c)}."
                f"{self._nota(anio)}")
            docs.append(_doc("provincia_anual", f"provincia_anual:{prov}:{anio}", text, anio, provincia=prov))
        return docs

    def combustible_anual(self) -> list[dict]:
        base = self.con.execute("""
            WITH t AS (SELECT anio, count(*) total FROM vehiculos GROUP BY anio),
            c AS (SELECT combustible, anio, count(*) c, median(avaluo) av FROM vehiculos
                  WHERE combustible IS NOT NULL GROUP BY combustible, anio)
            SELECT combustible, anio, c, av, total FROM c JOIN t USING (anio)
            ORDER BY anio, c DESC""").fetchall()
        marcas = _top(self.con, ["combustible", "anio"], "marca", 5)
        provs = _top(self.con, ["combustible", "anio"], "provincia", 3)
        docs = []
        for comb, anio, c, av, total in base:
            k = (comb, anio)
            text = (
                f"Combustible {comb}, año {anio}. {self._ctx()} Registros: {n(c)} "
                f"({pct(c / total * 100)} del segmento). Avalúo mediano: {money(av)}. "
                f"Marcas líderes: {_items(marcas[k], c)}. Provincias principales: {_items(provs[k], c)}."
                f"{self._nota(anio)}")
            docs.append(_doc("combustible_anual", f"combustible_anual:{comb}:{anio}", text, anio, combustible=comb))
        return docs

    def comparativo_periodo(self) -> list[dict]:
        """Compara el mismo rango de meses entre años consecutivos (evita comparar contra un año incompleto)."""
        docs = []
        for prev, cur in zip(self.anios, self.anios[1:]):
            if cur != prev + 1:
                continue
            corte = self.cobertura[cur][1].month
            periodo = f"enero a {MESES[corte - 1]}"
            rows = self.con.execute(f"""
                SELECT marca, sum((anio = {prev})::int) a, sum((anio = {cur})::int) b
                FROM vehiculos WHERE mes <= {corte} AND anio IN ({prev}, {cur}) AND marca IS NOT NULL
                GROUP BY marca HAVING a + b >= 100 ORDER BY b DESC""").fetchall()
            tot_a, tot_b = self.con.execute(f"""
                SELECT sum((anio = {prev})::int), sum((anio = {cur})::int) FROM vehiculos
                WHERE mes <= {corte} AND anio IN ({prev}, {cur})""").fetchone()
            docs.append(_doc("comparativo_periodo", f"comparativo:total:{prev}-{cur}", (
                f"Comparación del mismo periodo ({periodo}) entre {prev} y {cur}. {self._ctx()} "
                f"Registros {prev}: {n(tot_a)}; {cur}: {n(tot_b)}; variación {variacion(tot_a, tot_b)}."), cur))
            for marca, a, b in rows:
                docs.append(_doc("comparativo_periodo", f"comparativo:{marca}:{prev}-{cur}", (
                    f"Marca {marca}: comparación del mismo periodo ({periodo}) entre {prev} y {cur}. "
                    f"{self._ctx()} Registros {prev}: {n(a)}; {cur}: {n(b)}; variación {variacion(a, b)}."),
                    cur, marca=marca))
        return docs

    def diccionario(self) -> list[dict]:
        if not self.dictionary_path or not Path(self.dictionary_path).exists():
            return []
        items = json.loads(Path(self.dictionary_path).read_text(encoding="utf-8"))
        return [_doc("diccionario", f"diccionario:{i['variable']}", (
            f"Variable {i['variable']} del dataset SRI de vehículos nuevos: {i['definicion']}. "
            f"Tipo: {i['tipo']}; formato: {i['formato']}.")) for i in items]

    def build(self) -> pd.DataFrame:
        docs = (self.mercado_mensual() + self.marca_anual() + self.marca_mensual()
                + self.provincia_anual() + self.combustible_anual() + self.comparativo_periodo()
                + self.diccionario())
        df = pd.DataFrame(docs)
        assert df["doc_id"].is_unique, "doc_id duplicados"
        return df


def save_corpus(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "documents.parquet", index=False)
    with (out_dir / "documents.jsonl").open("w", encoding="utf-8") as f:
        for rec in json.loads(df.to_json(orient="records", force_ascii=False)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
