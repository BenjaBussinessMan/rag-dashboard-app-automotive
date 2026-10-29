"""Compara BM25, semantico, hibrido (RRF) e hibrido + reranker sobre un set de consultas con respuesta conocida."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.service import build_retriever  # noqa: E402

CONFIGS = [
    ("bm25", {"mode": "bm25"}),
    ("dense", {"mode": "dense"}),
    ("dense+meta", {"mode": "dense", "understand": True}),
    ("hybrid_rrf", {"mode": "hybrid"}),
    ("hybrid+meta", {"mode": "hybrid", "understand": True}),
    ("hybrid+meta bm25=.5", {"mode": "hybrid", "understand": True, "bm25_weight": 0.5}),
    ("hybrid+meta+rerank", {"mode": "hybrid", "understand": True, "rerank": True}),
]


def metrics(ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    return {"hit@1": sum(r == 1 for r in ranks) / n,
            f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=ROOT / "tests" / "eval" / "retrieval_queries.json")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    retriever = build_retriever()
    known = set(retriever.corpus["doc_id"])
    for q in queries:
        missing = [e for e in q["expected"] if e not in known]
        assert not missing, f"ids esperados inexistentes: {missing}"

    results = {}
    for name, kw in CONFIGS:
        ranks = []
        for q in queries:
            ids = [h.doc_id for h in retriever.search(q["q"], k=10, **kw)]
            pos = [ids.index(e) + 1 for e in q["expected"] if e in ids]
            ranks.append(min(pos) if pos else None)
            if args.verbose and not pos:
                print(f"[{name}] FALLA: {q['q']!r} -> {ids[:3]}")
        results[name] = metrics(ranks, args.k)

    print(f"{'config':<22}{'hit@1':>8}{f'hit@{args.k}':>8}{'mrr':>8}")
    for name, m in results.items():
        print(f"{name:<22}{m['hit@1']:>8.2f}{m[f'hit@{args.k}']:>8.2f}{m['mrr']:>8.3f}")
    (ROOT / "data" / "processed" / "retrieval_eval.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
