"""Compila los resultados de las evaluaciones (data/processed/*.json) en docs/EVALUACION.md."""

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "EVALUACION.md"
HALLAZGOS = ROOT / "docs" / "hallazgos.md"  # texto escrito a mano tras revisar los resultados


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def retrieval_section() -> str:
    data = load(PROCESSED / "retrieval_eval.json")
    if not data:
        return "_Sin resultados: ejecuta `python scripts/eval_retrieval.py`._"
    k = data["k"]
    res = data["results"]
    rows = [[name, f"{m['overall']['hit@1']:.2f}", f"{m['overall'][f'hit@{k}']:.2f}", f"{m['overall']['mrr']:.3f}"]
            for name, m in res.items()]
    cats = list(next(iter(res.values()))["by_category"])
    n_cat = {c: next(iter(res.values()))["by_category"][c]["n"] for c in cats}
    cat_rows = [[name] + [f"{m['by_category'][c][f'hit@{k}']:.2f}" for c in cats] for name, m in res.items()]
    return (f"{data['n']} consultas. `hit@1`: el documento correcto es el primero; `hit@{k}`: está entre los {k} primeros; "
            f"`mrr`: media del inverso de la posición.\n\n"
            + table(["Configuración", "hit@1", f"hit@{k}", "MRR"], rows)
            + f"\n\n**hit@{k} por categoría**\n\n"
            + table(["Configuración"] + [f"{c} ({n_cat[c]})" for c in cats], cat_rows))


def sql_section() -> str:
    blocks = []
    for path in sorted(PROCESSED.glob("sql_eval_*.json")):
        d = load(path)
        if not (d and "results" in d):
            continue
        cats = ", ".join(f"{c} {v['ok']}/{v['n']}" for c, v in d["by_category"].items())
        strict = f" (comparación estricta, que exige columnas y redondeo idénticos: {d['ok_strict']}/{d['n']})" \
            if "ok_strict" in d else ""
        blocks.append(f"**{d['model']}**: {d['ok']}/{d['n']} correctas ({pct(d['ok'] / d['n'])}){strict}; "
                      f"{d['mean_seconds']} s por consulta en promedio. Por categoría: {cats}.")
        fails = [r for r in d["results"] if not r["ok"]]
        if fails:
            blocks.append("\n".join(f"- Falla: *{r['q']}* — esperado `{r['gold']}`, obtenido `{r['got']}`" for r in fails))
    return "\n\n".join(blocks) or "_Sin resultados: ejecuta `python scripts/eval_sql.py`._"


def generation_section() -> str:
    runs = [d for d in (load(p) for p in sorted(PROCESSED.glob("generation_eval_*.json"))) if d and "results" in d]
    if not runs:
        return "_Sin resultados: ejecuta `python scripts/eval_generation.py`._"
    cats = list(runs[0]["by_category"])
    rows = []
    for d in runs:
        rows.append([d["model"], f"{d['ok']}/{d['n']} ({pct(d['ok'] / d['n'])})"]
                    + [f"{d['by_category'][c]['ok']}/{d['by_category'][c]['n']}" for c in cats]
                    + [f"{d['route_ok']}/{d['n']}", d["mean_seconds"]])
    out = [table(["Modelo", "Correctas"] + cats + ["Ruta esperada", "s por pregunta"], rows)]
    for d in runs:
        fails = [r for r in d["results"] if not r["ok"]]
        if fails:
            out.append(f"\n**Fallos de {d['model']}**\n")
            out += [f"- *{r['q']}* (ruta {r['route']}): {'; '.join(r['problems'])}" for r in fails]
    return "\n".join(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.parent.mkdir(exist_ok=True)
    hallazgos = HALLAZGOS.read_text(encoding="utf-8") if HALLAZGOS.exists() else ""
    text = f"""# Evaluación del asistente

Generado el {date.today().isoformat()} con `python scripts/make_report.py` a partir de los resultados medidos
(`data/processed/*.json`). Corpus: SRI 2025-2026 (automóvil, camión y camioneta) y boletines AEADE de julio a
septiembre de 2026. Todo corre en local: BGE-M3 para embeddings y modelos de Ollama para generación.

## 1. Recuperación

{retrieval_section()}

## 2. Text-to-SQL (cifras exactas)

Exactitud de ejecución: se compara el resultado de la consulta que escribe el modelo contra el de una consulta de
referencia escrita a mano. Un SQL distinto pero con el mismo resultado cuenta como correcto. La comparación tolera
columnas adicionales y diferencias de redondeo (por ejemplo `21880,3` frente a `21880`), pero no diferencias en los
valores; también se informa la comparación estricta.

{sql_section()}

## 3. Asistente completo

Preguntas de extremo a extremo (router, SQL o documentos, guardrail). Una respuesta es correcta si contiene las
cifras y textos esperados, no contiene valores prohibidos y quedó verificada; las preguntas fuera de alcance deben
rechazarse.

{generation_section()}

{hallazgos}
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"escrito {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
