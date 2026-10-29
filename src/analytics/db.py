"""Capa analitica: DuckDB sobre el Parquet procesado, con filtro de años y clases."""

from pathlib import Path

import duckdb

DEFAULT_CLASES = ("AUTOMOVIL", "CAMION", "CAMIONETA")


def _sql_list(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" if isinstance(v, str) else str(int(v))
                     for v in values)


def connect(processed_dir: Path, years=None, clases=DEFAULT_CLASES) -> duckdb.DuckDBPyConnection:
    """Conexion en memoria con la vista `vehiculos` (anio/mes salen de fecha_proceso)."""
    con = duckdb.connect()
    where = []
    if years:
        where.append(f"anio_archivo IN ({_sql_list(years)})")
    if clases:
        where.append(f"clase IN ({_sql_list(clases)})")
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    pattern = (Path(processed_dir) / "vehiculos_nuevos" / "*" / "*.parquet").as_posix()
    con.execute(f"""
        CREATE VIEW vehiculos AS
        SELECT *, year(fecha_proceso) AS anio, month(fecha_proceso) AS mes
        FROM read_parquet('{pattern}', hive_partitioning = true)
        {clause}
    """)
    return con
