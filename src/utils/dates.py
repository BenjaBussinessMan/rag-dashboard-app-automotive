import pandas as pd

_MESES = {"ene": "01", "feb": "02", "mar": "03", "abr": "04", "may": "05", "jun": "06",
          "jul": "07", "ago": "08", "sep": "09", "oct": "10", "nov": "11", "dic": "12"}


def parse_fecha(serie: pd.Series) -> pd.Series:
    """Los formatos varian por año: '2-dic-21', '9/12/2024', '20/11/2025'."""
    texto = serie.astype("string").str.strip().str.lower()
    for nombre, num in _MESES.items():
        texto = texto.str.replace(f"-{nombre}-", f"/{num}/", regex=False)
    largo = pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce")
    corto = pd.to_datetime(texto, format="%d/%m/%y", errors="coerce")
    return largo.fillna(corto)
