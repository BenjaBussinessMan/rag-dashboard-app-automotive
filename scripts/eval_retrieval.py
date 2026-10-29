"""Compara BM25, semantico, hibrido (RRF) y reranker sobre consultas con respuesta conocida, por categoria.

Cada consulta define `expected` (ids de documentos validos) o `must_contain` (textos que debe contener un
documento relevante, para consultas de termino exacto donde hay muchos documentos validos).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import tokenize  # noqa: E402
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


def norm(text: str) -> str:
    return " ".join(tokenize(text))


def metrics(ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    return {"n": n, "hit@1": sum(r == 1 for r in ranks) / n,
            f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n}


def first_relevant_rank(query: dict, hits) -> int | None:
    for pos, h in enumerate(hits, start=1):
        if h.doc_id in query.get("expected", []):
            return pos
        needles = query.get("must_contain")
        if needles and all(norm(n) in norm(h.text) for n in needles):
            return pos
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=ROOT / "tests" / "eval" / "retrieval_queries.json")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    retriever = build_retriever()
    known = set(retriever.corpus["doc_id"])
    norm_texts = [norm(t) for t in retriever.corpus["text"]]
    for q in queries:
        assert q.get("expected") or q.get("must_contain"), f"consulta sin criterio: {q['q']}"
        missing = [e for e in q.get("expected", []) if e not in known]
        assert not missing, f"ids esperados inexistentes: {missing}"
        if q.get("must_contain"):
            assert any(all(norm(n) in t for n in q["must_contain"]) for t in norm_texts), \
                f"ningun documento contiene {q['must_contain']}"

    results: dict = {}
    for name, kw in CONFIGS:
        ranks, by_cat = [], defaultdict(list)
        for q in queries:
            hits = retriever.search(q["q"], k=10, **kw)
            rank = first_relevant_rank(q, hits)
            ranks.append(rank)
            by_cat[q["category"]].append(rank)
            if args.verbose and (rank is None or rank > args.k):
                print(f"[{name}] FALLA ({q['category']}): {q['q']!r} -> {[h.doc_id for h in hits[:3]]}")
        results[name] = {"overall": metrics(ranks, args.k),
                         "by_category": {c: metrics(r, args.k) for c, r in by_cat.items()}}

    k = args.k
    print(f"\n{len(queries)} consultas")
    print(f"{'config':<22}{'hit@1':>8}{f'hit@{k}':>8}{'mrr':>8}")
    for name, m in results.items():
        o = m["overall"]
        print(f"{name:<22}{o['hit@1']:>8.2f}{o[f'hit@{k}']:>8.2f}{o['mrr']:>8.3f}")
    cats = list(next(iter(results.values()))["by_category"])
    print(f"\nhit@{k} por categoria")
    print(f"{'config':<22}" + "".join(f"{c[:11]:>12}" for c in cats))
    for name, m in results.items():
        print(f"{name:<22}" + "".join(f"{m['by_category'][c][f'hit@{k}']:>12.2f}" for c in cats))
    (ROOT / "data" / "processed" / "retrieval_eval.json").write_text(
        json.dumps({"k": k, "n": len(queries), "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
