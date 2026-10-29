"""Evalua el asistente completo (router, SQL, RAG, guardrails) con criterios de correccion automaticos.

Una respuesta es correcta si, segun el caso:
- debe rechazarse (`refuse`): el asistente rechaza;
- en otro caso: no rechaza, contiene todas las cifras (`expect_numbers`, o las del resultado de `gold_sql`) y los
  textos (`expect_text`) esperados, no contiene nada de `forbid`, y quedo verificada por el guardrail.
"""

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.db import connect  # noqa: E402
from src.generation.guardrails import numbers  # noqa: E402
from src.generation.ollama import DEFAULT_MODEL, OllamaClient  # noqa: E402
from src.generation.rag import REFUSAL, RAGPipeline  # noqa: E402
from src.generation.router import Assistant  # noqa: E402
from src.generation.sql_qa import SQLPipeline, _cell  # noqa: E402
from src.retrieval.service import PROCESSED, build_retriever  # noqa: E402

QUESTIONS = ROOT / "tests" / "eval" / "generation_questions.json"


def plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def expectations(item: dict, con) -> tuple[set[str], list[str]]:
    """Cifras y textos esperados: los explicitos mas los que salen de ejecutar `gold_sql`."""
    nums = set(numbers(" ".join(item.get("expect_numbers", []))))
    texts = list(item.get("expect_text", []))
    if item.get("gold_sql"):
        df = con.execute(item["gold_sql"]).df().astype(object)
        for rec in df.to_dict("records")[:5]:
            for v in rec.values():
                cell = _cell(v)
                if isinstance(v, str):
                    texts.append(v)
                else:
                    nums |= numbers(cell)
    return nums, texts


def score(item: dict, ans, con) -> tuple[bool, list[str]]:
    refused = ans.text.startswith(REFUSAL)
    if item.get("refuse"):
        return refused, [] if refused else ["debia rechazar y respondio"]
    nums, texts = expectations(item, con)
    problems = []
    if refused:
        problems.append("rechazo una pregunta contestable")
    got = numbers(ans.text, drop_citations=True)
    problems += [f"falta la cifra {n}" for n in sorted(nums - got)]
    problems += [f"falta el texto {t!r}" for t in texts if plain(t) not in plain(ans.text)]
    problems += [f"contiene lo prohibido {f!r}" for f in item.get("forbid", []) if f in ans.text or
                 numbers(f) <= got and numbers(f)]
    if not ans.verdict.grounded:
        problems.append("no verificada por el guardrail")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-gpu", type=int, default=None, help="0 = solo CPU")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    llm = OllamaClient(args.model, num_gpu=args.num_gpu)
    llm.check()
    retriever = build_retriever(rerank=False)
    con = connect(PROCESSED, years=[2025, 2026])
    assistant = Assistant(RAGPipeline(retriever, llm, k=5), SQLPipeline(con, llm, retriever.parser))

    rows = []
    for item in json.loads(args.questions.read_text(encoding="utf-8")):
        t0 = time.time()
        ans = assistant.answer(item["q"])
        ok, problems = score(item, ans, con)
        route_ok = item.get("route") is None or ans.route == item["route"]
        rows.append({"q": item["q"], "category": item["category"], "ok": ok, "problems": problems,
                     "route": ans.route, "route_expected": item.get("route"), "route_ok": route_ok,
                     "refused": ans.text.startswith(REFUSAL), "grounded": ans.verdict.grounded,
                     "retried": ans.retried, "answer": ans.text, "sql": ans.sql,
                     "seconds": round(time.time() - t0, 1)})
        r = rows[-1]
        print(f"\n[{'OK' if ok else 'FALLA'}] ({r['category']}, ruta {r['route']}{'' if route_ok else ' INESPERADA'}) "
              f"{r['q']}  ({r['seconds']} s)\n{r['answer'][:400]}" + (f"\n    -> {problems}" if problems else ""))

    by_cat: dict = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"ok": 0, "n": 0})
        c["n"] += 1
        c["ok"] += r["ok"]
    n_ok = sum(r["ok"] for r in rows)
    print(f"\nCorrectas ({args.model}): {n_ok}/{len(rows)}")
    for cat, c in by_cat.items():
        print(f"  {cat:<12}{c['ok']}/{c['n']}")
    summary = {"model": args.model, "ok": n_ok, "n": len(rows), "by_category": by_cat,
               "route_ok": sum(r["route_ok"] for r in rows),
               "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1), "results": rows}
    out = ROOT / "data" / "processed" / f"generation_eval_{args.model.replace(':', '_')}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
