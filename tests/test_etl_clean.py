import pandas as pd

from src.etl import clean
from src.utils.dates import parse_fecha

HEADER_2021 = ("CATEGORÍA;CÓDIGO DE VEHÍCULO;TIPO TRANSACCIÓN;MARCA;MODELO;PAIS;AÑO MODELO;CLASE;"
               "SUB CLASE;TIPO;AVALÚO;FECHA PROCESO (DD/MM/AA);TIPO SERVICIO;CILINDRAJE;"
               "TIPO COMBUSTIBLE;FECHA COMPRA (DD/MM/AA);CANTON;COLOR 1;COLOR 2;"
               "PERSONA NATURAL - SOCIEDAD;")
HEADER_2025 = ("CATEGORÍA;CÓDIGO DE VEHÍCULO;TIPO TRANSACCIÓN;MARCA;MODELO;PAIS;AÑO MODELO;CLASE;"
               "SUB CLASE;TIPO;AVALUO;FECHA PROCESO (DD/MM/AAAA);TIPO SERVICIO;CILINDRAJE;"
               "TIPO COMBUSTIBLE;FECHA COMPRA (DD/MM/AAAA);CANTÓN;COLOR 1;COLOR 2;"
               "PERSONA NATURAL - JURIDICA")


def test_parse_fecha_formatos():
    s = pd.Series(["2-dic-21", "9/12/2024", "20/11/2025", "basura"])
    out = parse_fecha(s)
    assert out.iloc[0] == pd.Timestamp("2021-12-02")
    assert out.iloc[1] == pd.Timestamp("2024-12-09")
    assert out.iloc[2] == pd.Timestamp("2025-11-20")
    assert pd.isna(out.iloc[3])


def test_esquemas_distintos_dan_mismas_columnas(tmp_path):
    f21 = tmp_path / "a.csv"
    f21.write_bytes((HEADER_2021 + "\n1;2;COMPRA LOCAL;MACK;GR;ESPAÑA;2022;CAMION;VOLQUETA;PESADO;"
                     "1430,83;2-dic-21;PAR;12800;DIESEL;15-dic-21;10927;BLA;;SOCIEDAD;\n").encode("latin-1"))
    f25 = tmp_path / "b.csv"
    f25.write_bytes((HEADER_2025 + "\n1;2;IMPORTACI?N DIRECTA;MACK;GR;ESPA?A;2022;CAMION;"
                     "CAMION PEQUE?O;PESADO;240000;20/11/2025;PAR;12742;DIESEL;29/10/2025;21701;NEG;;"
                     "NATURAL").encode("latin-1"))
    a, b = clean.clean_year(f21, 2021), clean.clean_year(f25, 2025)
    assert list(a.columns) == list(b.columns)
    assert a["avaluo"].iloc[0] == 1430.83
    assert a["pais"].iloc[0] == b["pais"].iloc[0] == "ESPANA"
    assert b["tipo_transaccion"].iloc[0] == "IMPORTACION DIRECTA"
    assert b["subclase"].iloc[0] == "CAMION PEQUENO"
    assert clean.quality_report(b, 2025)["signos_interrogacion"] == 0


def test_utf8_con_bom(tmp_path):
    f = tmp_path / "c.csv"
    f.write_bytes(("﻿" + HEADER_2025 + "\n1;2;COMPRA LOCAL;M;X;ECUADOR;2022;C;S;LIVIANO;10,00;"
                   "18/09/2026;PAR;100;GASOLINA;02/09/2026;QUITO;BLA;;NATURAL").encode("utf-8"))
    df = clean.clean_year(f, 2026)
    assert df["categoria"].iloc[0] == 1
    assert df["canton_nombre"].iloc[0] == "QUITO"
    assert pd.isna(df["canton_codigo"].iloc[0])
