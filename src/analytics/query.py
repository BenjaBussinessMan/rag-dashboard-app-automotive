"""Herramienta SQL de solo lectura para respuestas numericas exactas (text-to-SQL del RAG)."""

import re

import duckdb
import pandas as pd

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|"
    r"call|set|read_parquet|read_csv|read_json|glob)\b", re.IGNORECASE)


class UnsafeQuery(ValueError):
    pass


def validate_select(sql: str) -> str:
    stmt = sql.strip().rstrip(";").strip()
    if ";" in stmt:
        raise UnsafeQuery("solo se permite una sentencia")
    if not re.match(r"^(select|with)\b", stmt, re.IGNORECASE):
        raise UnsafeQuery("solo se permiten consultas SELECT")
    if _FORBIDDEN.search(stmt):
        raise UnsafeQuery("la consulta contiene operaciones no permitidas")
    return stmt


def run_select(con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = 200) -> pd.DataFrame:
    stmt = validate_select(sql)
    return con.execute(f"SELECT * FROM ({stmt}) LIMIT {int(max_rows)}").df()
