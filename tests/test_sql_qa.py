import duckdb
import pandas as pd
import pytest

from src.generation.router import Assistant, wants_sql
from src.generation.sql_qa import SQLPipeline, extract_sql, render_rows
from src.retrieval.query import QueryParser


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("""CREATE TABLE vehiculos AS SELECT * FROM (VALUES
        (2025, 1, 'KIA', 'GUAYAS', 'GASOLINA', TIMESTAMP '2025-01-10'),
        (2026, 1, 'KIA', 'GUAYAS', 'GASOLINA', TIMESTAMP '2026-01-10'),
        (2026, 2, 'KIA', 'PICHINCHA', 'ELECTRICO', TIMESTAMP '2026-02-10'),
        (2026, 2, 'TOYOTA', 'PICHINCHA', 'GASOLINA', TIMESTAMP '2026-02-20')
    ) t(anio, mes, marca, provincia, combustible, fecha_proceso)""")
    return c


@pytest.fixture
def parser():
    return QueryParser(pd.DataFrame({"marca": ["KIA", "TOYOTA"], "anio": [2025, 2026]}))


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, temperature=0.0):
        self.calls.append(messages)
        return self.replies.pop(0)


KIA_SQL = "```sql\nSELECT count(*) AS registros FROM vehiculos WHERE marca = 'KIA' AND anio = 2026\n```"


def test_extract_sql():
    assert extract_sql(KIA_SQL).startswith("SELECT count(*)")
    assert extract_sql("Claro: SELECT 1 AS x") == "SELECT 1 AS x"
    assert extract_sql("no hay consulta") is None


def test_render_rows_formatea_cifras():
    df = pd.DataFrame({"anio": [2026], "registros": [15649], "porcentaje": [8.1]})
    assert render_rows(df) == "anio: 2026; registros: 15.649; porcentaje: 8,1"


def test_respuesta_sql_verificada(con, parser):
    llm = FakeLLM(KIA_SQL, "Kia registró 2 vehículos en 2026.")
    ans = SQLPipeline(con, llm, parser).answer("¿Cuántos Kia se registraron en 2026?")
    assert ans.route == "sql" and ans.verdict.grounded and "[1]" in ans.text
    assert ans.table == [{"registros": 2}] and "marca = 'KIA'" in ans.sql
    assert ans.hits[0].metadata["fuente"].startswith("SRI datos abiertos (consulta SQL)")
    assert "anio IN (2026)" in llm.calls[0][1]["content"] and "marca IN ('KIA')" in llm.calls[0][1]["content"]


def test_cifra_inventada_cae_al_resultado_de_la_base(con, parser):
    llm = FakeLLM(KIA_SQL, "Kia registró 99 vehículos.", "Kia registró 98 vehículos.")
    ans = SQLPipeline(con, llm, parser).answer("¿Cuántos Kia se registraron en 2026?")
    assert ans.retried and ans.verdict.grounded
    assert ans.text.startswith("Resultado de la consulta: registros: 2") and "99" not in ans.text


def test_sql_invalido_se_corrige_una_vez(con, parser):
    llm = FakeLLM("```sql\nSELECT count(*) FROM tabla_inexistente\n```", KIA_SQL, "Kia registró 2 vehículos.")
    ans = SQLPipeline(con, llm, parser).answer("¿Cuántos Kia se registraron en 2026?")
    assert ans.route == "sql" and ans.table == [{"registros": 2}]
    assert "falló" in llm.calls[1][-1]["content"]


def test_sql_peligroso_o_irrecuperable_devuelve_none(con, parser):
    llm = FakeLLM("```sql\nDROP TABLE vehiculos\n```", "```sql\nDELETE FROM vehiculos\n```")
    assert SQLPipeline(con, llm, parser).answer("borra todo") is None
    assert con.execute("SELECT count(*) FROM vehiculos").fetchone()[0] == 4


def test_anio_fuera_de_la_base_no_responde_cero(con, parser):
    llm = FakeLLM()  # no debe llamarse
    ans = SQLPipeline(con, llm, parser).answer("¿Cuántos Kia se registraron en 2024?")
    assert ans.route == "sql" and "No encuentro" in ans.text and "no incluye 2024" in ans.text and not llm.calls


def test_consulta_sin_filas(con, parser):
    llm = FakeLLM("```sql\nSELECT marca FROM vehiculos WHERE marca = 'NOEXISTE'\n```")
    ans = SQLPipeline(con, llm, parser).answer("¿Qué marca llamada Noexiste hay?")
    assert "No encuentro" in ans.text and "no devolvió filas" in ans.text and ans.table == []


def test_router():
    assert wants_sql("¿Cuántos Kia se registraron en 2026?")
    assert wants_sql("¿Qué porcentaje fueron eléctricos?")
    assert not wants_sql("¿Cuánto crecieron las ventas según AEADE?")
    assert not wants_sql("¿Qué significa la variable avalúo?")
    assert not wants_sql("¿Cómo estuvo el mercado en marzo?")


def test_router_cae_al_rag_si_sql_falla(con, parser):
    class Rag:
        def answer(self, q):
            return "RAG"
    sql = SQLPipeline(con, FakeLLM("sin sql", "sin sql"), parser)
    assert Assistant(Rag(), sql).answer("¿Cuántos Kia en 2026?") == "RAG"
    assert Assistant(Rag(), sql).answer("¿Cómo estuvo marzo?") == "RAG"
    forzado = Assistant(Rag(), SQLPipeline(con, FakeLLM("x", "x"), parser)).answer("¿Cuántos Kia?", route="sql")
    assert forzado.route == "sql" and "No encuentro" in forzado.text


def test_rechazo_del_modelo_con_resultado_valido_no_gana(con, parser):
    refusal = "No encuentro esa información en los documentos disponibles."
    llm = FakeLLM(KIA_SQL, refusal, refusal)
    ans = SQLPipeline(con, llm, parser).answer("¿Cuántos Kia se registraron en 2026?")
    assert ans.retried and ans.verdict.grounded
    assert ans.text.startswith("Resultado de la consulta: registros: 2") and "No encuentro" not in ans.text
    # el fragmento que ve el modelo incluye la consulta ejecutada
    assert "Consulta ejecutada: SELECT count(*)" in llm.calls[1][1]["content"]


def test_cifras_decimales_conservan_precision():
    df = pd.DataFrame({"a": [24295.77], "b": [8.1], "c": [-3.5], "d": [1234567.0]})
    assert render_rows(df) == "a: 24.295,77; b: 8,1; c: -3,5; d: 1.234.567"


def test_router_rankings_y_cual_es():
    for q in ("¿Cuáles son las 3 marcas con más registros en 2026?", "¿En qué mes de 2026 hubo más registros?",
              "¿En qué provincia se registraron más vehículos BYD en 2026?", "¿Cuál es la marca con más registros en Guayas?",
              "¿Cuáles son las marcas más vendidas en 2026?"):
        assert wants_sql(q), q
    for q in ("¿Cuál es la capital de Francia?", "¿Qué significa la variable avalúo?",
              "¿Cuáles son las marcas más vendidas según AEADE?", "¿Cómo estuvo el mercado en marzo de 2026?"):
        assert not wants_sql(q), q


def test_sin_datos_rechaza_en_lugar_de_inventar(con, parser):
    llm = FakeLLM("SIN_DATOS")
    pipe = SQLPipeline(con, llm, parser)
    ans = pipe.answer("¿Cuántos vehículos usados se vendieron en 2026?")
    assert ans.route == "sql" and "No encuentro" in ans.text and "solo vehículos nuevos" in ans.text and not ans.table
    # un SQL valido que ademas menciona SIN_DATOS en un comentario de texto no cuenta como rechazo
    llm2 = FakeLLM("SIN_DATOS\n" + KIA_SQL, "Kia registró 2 vehículos.")
    assert SQLPipeline(con, llm2, parser).answer("¿Cuántos Kia se registraron en 2026?").table == [{"registros": 2}]
