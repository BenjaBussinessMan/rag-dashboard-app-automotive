from pathlib import Path

import pytest

from src.ingestion import aeade_pdf as a

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

PAGINA = """Power BI Desktop
TOP 20 – MARCAS
Marca Ago 25 Ago 26 Ene-Ago 25 Ene-Ago 26
KIA 1.559 2.623 14.136 20.005
  11.219 15.649 88.335 123.927
Marca Sep 25 Sep 26 Ene-Sep 25 Ene-Sep 26
4
"""

SERIE = "VENTAS MENSUALES\n0\n5.000\nENERO\n2025 2026\n" + "\n".join(str(1000 + i) for i in range(12 + 9)) + "\nTITULO"


def test_boletin_month():
    assert a.boletin_month(Path("Boletin-de-Prensa-Septiembre-2026.pdf")) == (2026, 9)


def test_encabezado_del_mes_anterior_se_elimina():
    texto, _ = a.clean_page(PAGINA, 9)
    assert "Ago 25" not in texto and "Marca Sep 25 Sep 26" in texto
    assert "Power BI" not in texto and "KIA 1.559" in texto


def test_serie_mensual_reconstruida():
    texto, info = a.clean_page(SERIE, 9)
    assert info["serie_ok"]
    assert "enero 1.000" in texto and "septiembre 1.020" in texto
    assert "2026 (enero a septiembre)" in texto


def test_serie_con_cantidad_inesperada_no_se_inventa():
    assert a.extract_series(["2025 2026", "1", "2", "3"], 9) is None


@pytest.mark.skipif(not (RAW / "Boletin-de-Prensa-Septiembre-2026.pdf").exists(), reason="PDF no disponible")
def test_boletin_real_septiembre():
    docs = [d for d in a.build_documents(RAW) if d["mes"] == 9]
    serie = next(d for d in docs if "Serie mensual" in d["text"] and "VENTAS MENSUALES DE VEHÍCULOS |" in d["text"])
    assert "septiembre 11.219" in serie["text"] and "septiembre 15.649" in serie["text"]
    assert all("Ago 25" not in d["text"] for d in docs)
