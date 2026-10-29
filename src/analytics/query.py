"""Herramienta SQL de solo lectura para respuestas numericas exactas (text-to-SQL del RAG).

El SQL lo escribe un LLM, asi que se trata como entrada no confiable: una sola sentencia SELECT/WITH,
solo la vista `vehiculos` (o CTEs propios), sin funciones de archivos ni del sistema.
"""

import re

import duckdb
import pandas as pd

ALLOWED_TABLE = "vehiculos"

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|"
    r"call|set|read_\w+|glob|getenv|current_setting|duckdb_\w+|pg_\w+|information_schema|"
    r"sniff_csv|parquet_\w+|query_table|checkpoint|summarize|describe|show)\b", re.IGNORECASE)
_RELATIONS = re.compile(r"\b(?:from|join)\s+([A-Za-z_][\w.]*)", re.IGNORECASE)
_CTE_NAMES = re.compile(r"\b([A-Za-z_]\w*)\s+as\s*\(", re.IGNORECASE)


class UnsafeQuery(ValueError):
    pass


def validate_select(sql: str) -> str:
    stmt = sql.strip().rstrip(";").strip()
    if ";" in stmt:
        raise UnsafeQuery("solo se permite una sentencia")
    if "--" in stmt or "/*" in stmt:
        raise UnsafeQuery("no se permiten comentarios en la consulta")
    if not re.match(r"^(select|with)\b", stmt, re.IGNORECASE):
        raise UnsafeQuery("solo se permiten consultas SELECT")
    if _FORBIDDEN.search(stmt):
        raise UnsafeQuery("la consulta contiene operaciones no permitidas")
    ctes = {c.lower() for c in _CTE_NAMES.findall(stmt)}
    for rel in _RELATIONS.findall(stmt):
        if rel.lower() != ALLOWED_TABLE and rel.lower() not in ctes:
            raise UnsafeQuery(f"solo se puede consultar la tabla `{ALLOWED_TABLE}` (no `{rel}`)")
    return stmt


def run_select(con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = 200) -> pd.DataFrame:
    stmt = validate_select(sql)
    return con.execute(f"SELECT * FROM ({stmt}) LIMIT {int(max_rows)}").df()
