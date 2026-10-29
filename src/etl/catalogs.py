"""Catalogos del diccionario de datos del SRI (SRI_Vehiculos_DD.xlsx): cantones, colores y definiciones."""

import re
from pathlib import Path

import pandas as pd

from src.etl.clean import strip_accents

DICTIONARY_FILE = "SRI_Vehiculos_DD.xlsx"
CANTON_ALIASES = {"ORELLANA": "FRANCISCO DE ORELLANA"}  # 2026 usa el nombre de la provincia


def _norm(text) -> str:
    return re.sub(r"\s+", " ", strip_accents(str(text)).upper()).strip()


def load_dim_canton(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="Catálogo_Cantones", header=None, skiprows=2, dtype=str)
    df.columns = ["canton_codigo", "canton_nombre", "provincia_codigo", "provincia"]
    df = df.dropna(subset=["canton_codigo"]).copy()
    df["canton_codigo"] = df["canton_codigo"].str.strip()
    for col in ("canton_nombre", "provincia"):
        df[col] = df[col].map(_norm)
    return df.reset_index(drop=True)


def load_dim_color(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="Catálogo_Colores", dtype=str)
    df.columns = ["color_codigo", "color_nombre"]
    df = df.dropna()
    df["color_codigo"] = df["color_codigo"].str.strip().str.upper()
    df["color_nombre"] = df["color_nombre"].map(_norm)
    return df.reset_index(drop=True)


def load_data_dictionary(xlsx: Path) -> list[dict]:
    """Definiciones de las variables (texto util para indexar en el RAG)."""
    df = pd.read_excel(xlsx, sheet_name="Diccionario", header=None, skiprows=6, dtype=str)
    out = []
    for _, r in df.iterrows():
        out.append({"variable": r[1], "definicion": r[2], "tipo": r[3], "formato": r[4]})
    return out


def _name_keys(name: str) -> list[str]:
    """'PLAYAS (GENERAL VILLAMIL)' -> nombre completo, base y alterno."""
    keys = [name]
    m = re.fullmatch(r"(.*?)\s*\((.*)\)", name)
    if m:
        keys += [m.group(1).strip(), m.group(2).strip()]
    return keys


def resolve_canton(df: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    """Completa canton_codigo / canton_nombre / provincia desde el catalogo.

    Nombres ambiguos (BOLIVAR, OLMEDO) o fuera de catalogo quedan sin resolver.
    La provincia se deduce del prefijo de 3 digitos del codigo.
    """
    by_name: dict[str, set] = {}
    for code, name in zip(dim["canton_codigo"], dim["canton_nombre"]):
        for key in _name_keys(name):
            by_name.setdefault(key, set()).add(code)
    unique = {k: next(iter(v)) for k, v in by_name.items() if len(v) == 1}
    for alias, target in CANTON_ALIASES.items():
        unique[alias] = unique[target]

    codigo = df["canton_codigo"].astype("string")
    nombre = df["canton_nombre"].astype("string").map(lambda v: _norm(v) if isinstance(v, str) else v)
    desde_nombre = nombre.map(unique).astype("string")
    codigo = codigo.fillna(desde_nombre)

    nombre_cat = dim.set_index("canton_codigo")["canton_nombre"]
    prov_prefijo = dim.drop_duplicates("provincia_codigo").set_index("provincia_codigo")["provincia"]

    out = df.copy()
    out["canton_codigo"] = codigo.astype("category")
    out["canton_nombre"] = codigo.map(nombre_cat).astype("string").fillna(nombre).astype("category")
    out["provincia"] = codigo.str[:3].map(prov_prefijo).astype("category")
    return out
