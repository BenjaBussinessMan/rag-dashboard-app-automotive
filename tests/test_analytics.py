from datetime import date

import pandas as pd
import pytest

from src.analytics import documents as d
from src.analytics.db import connect
from src.analytics.query import UnsafeQuery, run_select, validate_select


def _write_dataset(root):
    rows = []
    for year, fechas in ((2025, ["2025-01-10", "2025-02-10", "2025-12-20"]),
                         (2026, ["2026-01-10", "2026-01-15", "2026-02-10"])):
        for i, f in enumerate(fechas):
            rows.append({"marca": "KIA" if i % 2 == 0 else "TOYOTA", "modelo": f"M{i}",
                         "clase": "AUTOMOVIL", "combustible": "GASOLINA", "provincia": "GUAYAS",
                         "tipo_persona": "NATURAL", "tipo_transaccion": "COMPRA LOCAL",
                         "avaluo": 1000.0 * (i + 1), "fecha_proceso": pd.Timestamp(f),
                         "anio_archivo": year})
        rows.append({**rows[-1], "clase": "MOTOCICLETA", "fecha_proceso": pd.Timestamp(fechas[0])})
    df = pd.DataFrame(rows)
    for y, g in df.groupby("anio_archivo"):
        part = root / "vehiculos_nuevos" / f"anio_archivo={y}"
        part.mkdir(parents=True)
        g.drop(columns="anio_archivo").to_parquet(part / "part.parquet", index=False)


def test_formatos():
    assert d.n(12345) == "12.345"
    assert d.pct(12.34) == "12,3 %"
    assert d.variacion(100, 150) == "+50,0 %"
    assert d.variacion(0, 5) == "sin base de comparación"


def test_filtro_de_clases_y_corpus(tmp_path):
    _write_dataset(tmp_path)
    con = connect(tmp_path, years=[2025, 2026], clases=("AUTOMOVIL",))
    assert con.execute("SELECT count(*) FROM vehiculos WHERE clase <> 'AUTOMOVIL'").fetchone()[0] == 0
    b = d.CorpusBuilder(con, ["AUTOMOVIL"])
    assert b.cobertura[2026][1].date() == date(2026, 2, 10)
    docs = b.build()
    assert docs["doc_id"].is_unique
    ene = docs[docs.doc_id == "mercado_mensual:2026-01"].text.iloc[0]
    assert "Registros de vehículos nuevos: 2" in ene
    # el año 2026 es incompleto: el comparativo usa solo enero-febrero en ambos años
    comp = docs[docs.doc_id == "comparativo:total:2025-2026"].text.iloc[0]
    assert "enero a febrero" in comp and "Registros 2025: 2; 2026: 3" in comp
    assert "Año incompleto" in docs[docs.doc_id == "combustible_anual:GASOLINA:2026"].text.iloc[0]


def test_sql_solo_lectura():
    assert validate_select("SELECT 1;") == "SELECT 1"
    for bad in ("DROP TABLE x", "SELECT 1; DELETE FROM x", "COPY vehiculos TO 'a.csv'",
                "SELECT * FROM read_csv('x.csv')", "WITH a AS (SELECT 1) INSERT INTO t SELECT 1"):
        with pytest.raises(UnsafeQuery):
            validate_select(bad)


def test_run_select(tmp_path):
    _write_dataset(tmp_path)
    con = connect(tmp_path, clases=("AUTOMOVIL",))
    out = run_select(con, "SELECT marca, count(*) AS n FROM vehiculos GROUP BY marca ORDER BY marca")
    assert out["n"].sum() == 6


def test_sql_solo_tabla_vehiculos_y_sin_vectores_de_fuga():
    ok = ("WITH t AS (SELECT marca, count(*) n FROM vehiculos GROUP BY marca) "
          "SELECT * FROM t ORDER BY n DESC")
    assert validate_select(ok)
    for bad in ("SELECT * FROM otra_tabla", "SELECT * FROM vehiculos v JOIN secretos s ON 1=1",
                "SELECT * FROM information_schema.tables", "SELECT * FROM duckdb_settings()",
                "SELECT * FROM vehiculos -- comentario", "SELECT getenv('PATH')",
                "SELECT * FROM read_parquet('x')", "SELECT * FROM parquet_scan('x')"):
        with pytest.raises(UnsafeQuery):
            validate_select(bad)
