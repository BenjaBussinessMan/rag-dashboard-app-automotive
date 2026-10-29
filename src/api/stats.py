"""Consultas agregadas para el dashboard (todas parametrizadas; las columnas salen de listas blancas)."""

import duckdb

DIMENSIONS = ("marca", "provincia", "combustible", "clase")
FILTERS = ("marca", "provincia", "combustible", "clase")
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def _where(filters: dict) -> tuple[str, list]:
    clauses, params = [], []
    for col, val in filters.items():
        if val is None:
            continue
        if col not in FILTERS:
            raise ValueError(f"filtro no permitido: {col}")
        clauses.append(f"{col} = ?")
        params.append(val)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def meta(con: duckdb.DuckDBPyConnection) -> dict:
    def distinct(col):
        return [r[0] for r in con.execute(
            f"SELECT {col} FROM vehiculos WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY count(*) DESC").fetchall()]
    anios = [r[0] for r in con.execute("SELECT DISTINCT anio FROM vehiculos ORDER BY anio").fetchall()]
    cobertura = {a: str(mx.date()) for a, mx in con.execute(
        "SELECT anio, max(fecha_proceso) FROM vehiculos GROUP BY anio").fetchall()}
    return {"anios": anios, "cobertura": cobertura, **{d + "s" if d != "clase" else "clases": distinct(d)
                                                      for d in DIMENSIONS}}


def monthly(con: duckdb.DuckDBPyConnection, **filters) -> list[dict]:
    where, params = _where(filters)
    rows = con.execute(f"SELECT anio, mes, count(*) FROM vehiculos{where} GROUP BY anio, mes ORDER BY anio, mes",
                       params).fetchall()
    return [{"anio": a, "mes": m, "registros": n} for a, m, n in rows]


def top(con: duckdb.DuckDBPyConnection, dimension: str, anio: int, limit: int = 10,
        mes_hasta: int | None = None, **filters) -> list[dict]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension no permitida: {dimension}")
    where, params = _where(filters)
    cond = f"{where} {'AND' if where else 'WHERE'} anio = ? AND {dimension} IS NOT NULL"
    params += [anio]
    if mes_hasta:
        cond += " AND mes <= ?"
        params.append(mes_hasta)
    rows = con.execute(f"""
        WITH g AS (SELECT {dimension} AS nombre, count(*) AS registros FROM vehiculos{cond} GROUP BY {dimension})
        SELECT nombre, registros, registros * 100.0 / sum(registros) OVER () FROM g
        ORDER BY registros DESC LIMIT ?""", params + [int(limit)]).fetchall()
    return [{"nombre": n, "registros": r, "participacion": round(p, 1)} for n, r, p in rows]


def kpis(con: duckdb.DuckDBPyConnection) -> dict:
    """Compara el ultimo año con el anterior en el MISMO rango de meses (el ultimo año suele estar incompleto)."""
    anios = [r[0] for r in con.execute("SELECT DISTINCT anio FROM vehiculos ORDER BY anio").fetchall()]
    if not anios:
        return {}
    actual = anios[-1]
    corte = con.execute("SELECT max(mes) FROM vehiculos WHERE anio = ?", [actual]).fetchone()[0]
    anterior = actual - 1 if (actual - 1) in anios else None

    def resumen(anio):
        total, elec, hib = con.execute("""
            SELECT count(*), sum((combustible = 'ELECTRICO')::int), sum((combustible LIKE 'HIBRIDO%')::int)
            FROM vehiculos WHERE anio = ? AND mes <= ?""", [anio, corte]).fetchone()
        lider = con.execute("""SELECT marca FROM vehiculos WHERE anio = ? AND mes <= ? AND marca IS NOT NULL
                               GROUP BY marca ORDER BY count(*) DESC LIMIT 1""", [anio, corte]).fetchone()
        return {"anio": anio, "registros": total, "electricos": elec or 0, "hibridos": hib or 0,
                "electrificados_pct": round(((elec or 0) + (hib or 0)) * 100 / total, 1) if total else 0,
                "marca_lider": lider[0] if lider else None}

    out = {"periodo": f"enero a {MESES[corte - 1]}", "mes_corte": corte, "actual": resumen(actual),
           "anterior": resumen(anterior) if anterior else None, "variacion_pct": None}
    if out["anterior"] and out["anterior"]["registros"]:
        out["variacion_pct"] = round((out["actual"]["registros"] / out["anterior"]["registros"] - 1) * 100, 1)
    return out
