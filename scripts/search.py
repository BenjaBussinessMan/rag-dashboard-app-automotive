import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.service import build_retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Busqueda hibrida sobre el corpus")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--filter", action="append", default=[], metavar="CAMPO=VALOR",
                        help="p. ej. --filter anio=2026 --filter marca=KIA")
    args = parser.parse_args()

    filters = {}
    for item in args.filter:
        key, val = item.split("=", 1)
        filters[key] = int(val) if val.isdigit() else val

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    retriever = build_retriever(dense=args.mode != "bm25", rerank=not args.no_rerank)
    for i, h in enumerate(retriever.search(args.query, args.k, args.mode, filters=filters,
                                           rerank=not args.no_rerank), start=1):
        rr = f" rerank={h.rerank_score:.2f}" if h.rerank_score is not None else ""
        print(f"{i}. {h.doc_id}  rrf={h.score:.4f}{rr}  ranks={h.ranks}")
        print(f"   {h.text[:220]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
