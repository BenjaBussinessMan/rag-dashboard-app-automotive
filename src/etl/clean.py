"""Limpieza y unificacion de esquema de los CSV del SRI a un dataset Parquet."""

import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.utils.dates import parse_fecha

logger = logging.getLogger(__name__)

CSV_SEP = ";"

# prefijo del encabezado normalizado -> nombre canonico (el orden importa: el primero que coincide gana)
COLUMN_MAP = [
    ("CATEGORIA", "categoria"),
    ("CODIGO DE VEHICULO", "codigo_vehiculo"),
    ("TIPO TRANSACCION", "tipo_transaccion"),
    ("MARCA", "marca"),
    ("MODELO", "modelo"),
    ("PAIS", "pais"),
    ("ANO MODELO", "anio_modelo"),
    ("SUB CLASE", "subclase"),
    ("CLASE", "clase"),
    ("TIPO SERVICIO", "tipo_servicio"),
    ("TIPO COMBUSTIBLE", "combustible"),
    ("TIPO", "tipo"),
    ("AVALUO", "avaluo"),
    ("FECHA PROCESO", "fecha_proceso"),
    ("FECHA COMPRA", "fecha_compra"),
    ("CILINDRAJE", "cilindraje"),
    ("CANTON", "canton"),
    ("COLOR 1", "color_1"),
    ("COLOR 2", "color_2"),
    ("PERSONA NATURAL", "tipo_persona"),
]

# El CSV 2025 trae '?' donde debia ir la enie / tilde.
TEXT_REPAIRS = {"ESPA?A": "ESPANA", "PEQUE?O": "PEQUENO", "IMPORTACI?N": "IMPORTACION"}

TEXT_COLS = ["tipo_transaccion", "marca", "modelo", "pais", "clase", "subclase", "tipo",
             "tipo_servicio", "combustible", "canton", "color_1", "color_2", "tipo_persona"]


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Los CSV mezclan UTF-8 con BOM (2026) y latin-1 (resto); se detecta por archivo."""
    raw = path.read_bytes()
    try:
        raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        encoding = "latin-1"
    df = pd.read_csv(path, sep=CSV_SEP, encoding=encoding, dtype=str, index_col=False)
    return df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        norm = strip_accents(str(col)).upper().strip()
        for prefix, canonical in COLUMN_MAP:
            if norm.startswith(prefix):
                mapping[col] = canonical
                break
        else:
            raise ValueError(f"columna desconocida: {col!r}")
    out = df.rename(columns=mapping)
    if out.columns.duplicated().any():
        raise ValueError(f"columnas duplicadas tras normalizar: {list(out.columns)}")
    return out


def _clean_text(serie: pd.Series) -> pd.Series:
    s = serie.astype("string").str.strip().str.upper()
    for bad, good in TEXT_REPAIRS.items():
        s = s.str.replace(bad, good, regex=False)
    s = s.map(lambda v: strip_accents(v) if isinstance(v, str) else v)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.replace("", pd.NA).astype("category")


def _to_number(serie: pd.Series) -> pd.Series:
    s = serie.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def clean_year(path: Path, year: int) -> pd.DataFrame:
    df = rename_columns(read_csv_auto(path))
    for col in TEXT_COLS:
        df[col] = _clean_text(df[col])
    df["categoria"] = pd.to_numeric(df["categoria"], errors="coerce").astype("Int64")
    df["codigo_vehiculo"] = pd.to_numeric(df["codigo_vehiculo"], errors="coerce").astype("Int64")
    df["anio_modelo"] = pd.to_numeric(df["anio_modelo"], errors="coerce").astype("Int16")
    df["cilindraje"] = _to_number(df["cilindraje"]).astype("float32")
    df["avaluo"] = _to_number(df["avaluo"]).astype("float64")
    df["fecha_proceso"] = parse_fecha(df["fecha_proceso"])
    df["fecha_compra"] = parse_fecha(df["fecha_compra"])
    df["tipo_persona"] = df["tipo_persona"].astype("string").replace({"SOCIEDAD": "JURIDICA"}).astype("category")
    # 2021-2025 traen el codigo de canton (SRI), 2026 el nombre; no hay tabla de equivalencia
    canton = df["canton"].astype("string")
    es_codigo = canton.str.fullmatch(r"\d+").fillna(False)
    df["canton_codigo"] = canton.where(es_codigo).astype("category")
    df["canton_nombre"] = canton.where(~es_codigo).astype("category")
    df = df.drop(columns="canton")
    df["anio_archivo"] = pd.Series(year, index=df.index, dtype="int16")
    return df


def quality_report(df: pd.DataFrame, year: int) -> dict:
    return {
        "year": year,
        "rows": int(len(df)),
        "nulls": {c: int(n) for c, n in df.isna().sum().items() if n},
        "avaluo_le_0": int((df["avaluo"] <= 0).sum()),
        "fecha_proceso": [str(df["fecha_proceso"].min().date()), str(df["fecha_proceso"].max().date())],
        "signos_interrogacion": int(sum(df[c].astype("string").str.contains(r"\?", regex=True).sum()
                                        for c in TEXT_COLS if c in df.columns)),
    }


def run_etl(raw_dir: Path, out_dir: Path, years=None) -> list[dict]:
    """CSV crudos -> data/processed/vehiculos_nuevos/anio_archivo=YYYY/*.parquet + etl_report.json"""
    files = sorted(raw_dir.glob("SRI_Vehiculos_Nuevos_*.csv"))
    reports = []
    dataset = out_dir / "vehiculos_nuevos"
    for path in files:
        year = int(re.search(r"(\d{4})", path.stem).group(1))
        if years and year not in years:
            continue
        logger.info("%s: procesando", path.name)
        df = clean_year(path, year)
        part = dataset / f"anio_archivo={year}"
        part.mkdir(parents=True, exist_ok=True)
        for old in part.glob("*.parquet"):
            old.unlink()
        # category -> string: los indices del diccionario cambian por archivo y rompen la lectura conjunta
        out = df.drop(columns="anio_archivo")
        cat_cols = out.select_dtypes("category").columns
        out[cat_cols] = out[cat_cols].astype("string")
        out.to_parquet(part / "part.parquet", index=False, compression="zstd")
        report = quality_report(df, year)
        reports.append(report)
        logger.info("%s: %s filas, %s nulos, %s '?'", path.name, report["rows"],
                    sum(report["nulls"].values()), report["signos_interrogacion"])
    (out_dir / "etl_report.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    return reports
