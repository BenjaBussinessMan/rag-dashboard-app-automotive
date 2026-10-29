from datetime import datetime

from src.ingestion import sri_scraper as s


def test_build_url():
    assert s.build_url(2024).endswith("/SRI_Vehiculos_Nuevos_2024.csv")


def test_closed_year_skipped_when_size_matches(tmp_path, monkeypatch):
    dest = tmp_path / "SRI_Vehiculos_Nuevos_2021.csv"
    dest.write_bytes(b"abc")
    monkeypatch.setattr(s, "_remote_size", lambda *a: 3)
    assert s._is_up_to_date(None, "u", dest, 2021)


def test_size_mismatch_redownloads(tmp_path, monkeypatch):
    dest = tmp_path / "x.csv"
    dest.write_bytes(b"abc")
    monkeypatch.setattr(s, "_remote_size", lambda *a: 99)
    assert not s._is_up_to_date(None, "u", dest, 2021)


def test_current_year_always_refreshed(tmp_path, monkeypatch):
    dest = tmp_path / "x.csv"
    dest.write_bytes(b"abc")
    monkeypatch.setattr(s, "_remote_size", lambda *a: 3)
    assert not s._is_up_to_date(None, "u", dest, datetime.now().year)
