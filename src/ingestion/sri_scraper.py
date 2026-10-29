"""Descarga de los CSV anuales de vehiculos nuevos del SRI (datos abiertos)."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from src.utils.dates import parse_fecha

logger = logging.getLogger(__name__)

URL_TEMPLATE = "https://descargas.sri.gob.ec/download/datosAbiertos/SRI_Vehiculos_Nuevos_{year}.csv"
DEFAULT_YEARS = tuple(range(2021, 2027))
DATE_COLUMN_PREFIX = "FECHA PROCESO"
CSV_SEP = ";"
CSV_ENCODING = "latin-1"
CHUNK_SIZE = 1024 * 256
TIMEOUT = (10, 120)
MANIFEST_NAME = "manifest.json"


def build_url(year: int) -> str:
    return URL_TEMPLATE.format(year=year)


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_size(session: requests.Session, url: str) -> int | None:
    try:
        resp = session.head(
            url, timeout=TIMEOUT, allow_redirects=True, headers={"Accept-Encoding": "identity"}
        )
        resp.raise_for_status()
        return int(resp.headers["Content-Length"])
    except (requests.RequestException, KeyError, ValueError):
        return None


def _is_up_to_date(session: requests.Session, url: str, dest: Path, year: int) -> bool:
    if not dest.exists() or dest.stat().st_size == 0:
        return False
    if year >= datetime.now().year:
        return False  # el año en curso crece, siempre se refresca
    return _remote_size(session, url) == dest.stat().st_size


def _download(session: requests.Session, url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def _describe(path: Path) -> dict:
    """Valida lectura y resume el archivo. La normalizacion de esquema es de otra etapa."""
    df = pd.read_csv(path, sep=CSV_SEP, encoding=CSV_ENCODING, low_memory=False, index_col=False)
    col = next((c for c in df.columns if c.upper().startswith(DATE_COLUMN_PREFIX)), None)
    if df.empty or col is None:
        raise ValueError(f"{path.name}: sin filas o sin columna '{DATE_COLUMN_PREFIX}*'")
    fechas = parse_fecha(df[col])
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "fecha_min": None if fechas.isna().all() else fechas.min().strftime("%Y-%m-%d"),
        "fecha_max": None if fechas.isna().all() else fechas.max().strftime("%Y-%m-%d"),
    }


def download_year(session: requests.Session, year: int, out_dir: Path, force: bool = False) -> dict:
    url = build_url(year)
    dest = out_dir / Path(url).name

    if not force and _is_up_to_date(session, url, dest, year):
        logger.info("%s: sin cambios, se omite", dest.name)
        return {"year": year, "status": "skipped"}

    logger.info("%s: descargando", dest.name)
    _download(session, url, dest)
    if dest.stat().st_size == 0:
        dest.unlink()
        raise ValueError(f"{dest.name}: archivo vacio")

    info = _describe(dest)
    entry = {
        "year": year,
        "status": "downloaded",
        "url": url,
        "file": dest.name,
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        **info,
    }
    logger.info("%s: %s filas, fecha max %s", dest.name, info["rows"], info["fecha_max"])
    return entry


def _update_manifest(out_dir: Path, entries: list[dict]) -> None:
    path = out_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for entry in entries:
        if entry["status"] == "downloaded":
            manifest[str(entry["year"])] = {k: v for k, v in entry.items() if k != "status"}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def download_all(years=DEFAULT_YEARS, out_dir: Path = Path("data/raw"), force: bool = False) -> list[dict]:
    """Descarga cada año; un fallo no detiene los demas. Devuelve el resultado por año."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()
    results = []
    for year in years:
        try:
            results.append(download_year(session, year, out_dir, force))
        except Exception as exc:
            logger.error("%s: fallo (%s)", year, exc)
            results.append({"year": year, "status": "failed", "error": str(exc)})
    _update_manifest(out_dir, results)
    return results
