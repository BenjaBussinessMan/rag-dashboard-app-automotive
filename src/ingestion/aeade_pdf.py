"""Ingesta de los boletines mensuales de AEADE (PDF exportados de Power BI) como documentos del corpus.

Particularidades de estos PDF:
- Las tablas traen dos encabezados: uno del mes anterior (oculto) y el correcto al final de la tabla.
  Los datos son siempre del mes del boletin (el Total coincide con la ultima barra del grafico).
- Los graficos de barras se extraen como etiquetas + lista de valores: se reconstruye la serie.
- Los graficos circulares pierden la asociacion etiqueta-porcentaje: se marcan como poco confiables.
"""

import re
from pathlib import Path

from pypdf import PdfReader

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
ABREV = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
HEADER_STARTS = ("Marca", "Segmento", "Energía", "Provincias", "Origen", "Part.", "Var.", "Var ")
FLECHAS = {"🡱": "(sube)", "🡳": "(baja)"}
MONTH_LINE = re.compile(r"^(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)\s*$")
FUENTE = "AEADE (datos del SRI)"


def boletin_month(path: Path) -> tuple[int, int]:
    """'Boletin-de-Prensa-Agosto-2026.pdf' -> (2026, 8)"""
    m = re.search(r"-([A-Za-z]+)-(\d{4})", path.stem)
    return int(m.group(2)), MESES.index(m.group(1).lower()) + 1


def _is_stale_header(line: str, prev_abrev: str) -> bool:
    return line.startswith(HEADER_STARTS) and re.search(rf"\b{prev_abrev}\b", line) is not None


def extract_series(lines: list[str], mes: int) -> tuple[list[int], list[int]] | None:
    """Valores tras la linea '2025 2026': 12 de 2025 y `mes` de 2026. None si no cuadra."""
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "2025 2026")
    except StopIteration:
        return None
    vals = []
    for ln in lines[start + 1:]:
        if re.fullmatch(r"[\d.]+", ln):
            vals.append(int(ln.replace(".", "")))
        else:
            break
    if len(vals) != 12 + mes:
        return None
    return vals[:12], vals[12:]


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def clean_page(text: str, mes: int) -> tuple[str, dict]:
    """Devuelve (texto limpio, info) con info = {seccion, grafico, serie_ok}."""
    for flecha, palabra in FLECHAS.items():
        text = text.replace(flecha, palabra)
    prev = ABREV[mes - 2] if mes > 1 else "Dic"
    lines = [ln.rstrip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln.strip() and ln.strip() != "Power BI Desktop"]
    serie = extract_series(lines, mes)

    out = []
    for ln in lines:
        s = ln.strip()
        if re.fullmatch(r"\d{1,2}", s) or _is_stale_header(s, prev):
            continue  # numero de pagina suelto o encabezado del mes anterior
        if serie:
            raw_numeric = re.fullmatch(r"[\d.]+", ln)  # sin espacios iniciales: solo graficos
            if raw_numeric or MONTH_LINE.match(s) or s == "2025 2026":
                continue
            ln = ln.replace("Número de unidades", "")
            if re.fullmatch(r"[\d.]+", ln.strip()) and not ln.startswith(" "):
                continue  # etiqueta del eje que venia pegada a "Número de unidades"
        out.append(ln.strip())

    seccion = next((ln for ln in out if len(ln) > 12 and ln.upper() == ln and "FUENTE" not in ln), "")
    texto = "\n".join(out)
    if serie:
        a, b = serie
        a_txt = ", ".join(f"{MESES[i]} {_fmt(v)}" for i, v in enumerate(a))
        b_txt = ", ".join(f"{MESES[i]} {_fmt(v)}" for i, v in enumerate(b))
        texto += (f"\nSerie mensual en unidades. 2025: {a_txt}. "
                  f"2026 (enero a {MESES[mes - 1]}): {b_txt}.")
    grafico = bool(re.search(r"UNIDADES POR ORIGEN|PARTICIPACIÓN POR TIPO", texto))
    return texto, {"seccion": seccion, "grafico": grafico, "serie_ok": serie is not None}


def build_documents(pdf_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(Path(pdf_dir).glob("Boletin-de-Prensa-*.pdf")):
        anio, mes = boletin_month(path)
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            texto, info = clean_page(page.extract_text() or "", mes)
            if len(texto) < 80:
                continue  # portada, paginas en blanco
            cabecera = (f"{FUENTE}. Boletín de prensa de {MESES[mes - 1]} de {anio}, página {i}. "
                        f"Las columnas de las tablas son del mes {ABREV[mes - 1]} {str(anio)[2:]} "
                        f"y acumulado Ene-{ABREV[mes - 1]}. ")
            if info["grafico"]:
                cabecera += ("Atención: el texto de los gráficos circulares puede tener porcentajes "
                             "desalineados de sus etiquetas; usar solo las tablas. ")
            docs.append({
                "doc_id": f"aeade:{anio}-{mes:02d}:p{i}", "doc_type": "boletin_aeade",
                "anio": anio, "mes": mes, "marca": None, "provincia": None, "combustible": None,
                "fuente": FUENTE, "pagina": i, "text": cabecera + texto.replace("\n", " | "),
            })
    return docs
