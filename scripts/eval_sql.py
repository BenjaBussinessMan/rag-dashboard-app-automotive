"""Exactitud de ejecucion del text-to-SQL: compara el resultado del SQL generado con el de una consulta de referencia."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.analytics.db import connect  # noqa: E402
from src.generation.ollama import DEFAULT_MODEL, OllamaClient  # noqa: E402
from src.generation.sql_qa import SQLPipeline  # noqa: E402
from src.retrieval.query import QueryParser  # noqa: E402
from src.retrieval.service import PROCESSED, load_corpus  # noqa: E402

QUESTIONS = ROOT / "tests" / "eval" / "sql_questions.json"


def normalize(df: pd.DataFrame) -> list[tuple]:
    """Resultado como lista de tuplas sin nombres de columna; numeros redondeados, texto en mayusculas."""
    rows = []
    for rec in df.astype(object).to_dict("records"):
        rows.append(tuple(round(float(v), 1) if isinstance(v, (int, float)) and not isinstance(v, bool)
                          else str(v).upper() for v in rec.values()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-gpu", type=int, default=None)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    con = connect(PROCESSED, years=[2025, 2026])
    llm = OllamaClient(args.model, num_gpu=args.num_gpu)
    llm.check()
    pipeline = SQLPipeline(con, llm, QueryParser(load_corpus()))

    results = []
    for item in json.loads(args.questions.read_text(encoding="utf-8")):
        t0 = time.time()
        gold = normalize(con.execute(item["gold"]).df())
        generated = pipeline.generate(item["q"])
        sql, got = (None, None) if generated is None else (generated[0], normalize(generated[1]))
        ok = got == gold
        results.append({"q": item["q"], "ok": ok, "sql": sql, "got": got, "gold": gold,
                        "seconds": round(time.time() - t0, 1)})
        print(f"[{'OK' if ok else 'FALLA'}] {item['q']}  ({results[-1]['seconds']} s)")
        if not ok:
            print(f"    esperado: {gold}\n    obtenido: {got}\n    sql: {sql}")

    n_ok = sum(r["ok"] for r in results)
    print(f"\nExactitud de ejecucion ({args.model}): {n_ok}/{len(results)}")
    out = ROOT / "data" / "processed" / f"sql_eval_{args.model.replace(':', '_')}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
