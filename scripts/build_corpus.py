import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.db import DEFAULT_CLASES, connect  # noqa: E402
from src.analytics.documents import CorpusBuilder, save_corpus  # noqa: E402
from src.ingestion.aeade_pdf import build_documents  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera el corpus de documentos para el RAG")
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--clases", nargs="+", default=list(DEFAULT_CLASES))
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()

    con = connect(args.processed_dir, args.years, args.clases)
    builder = CorpusBuilder(con, args.clases, args.processed_dir / "data_dictionary.json")
    df = builder.build()
    pdf_docs = build_documents(args.pdf_dir)
    if pdf_docs:
        df = pd.concat([df, pd.DataFrame(pdf_docs)], ignore_index=True)
        assert df["doc_id"].is_unique, "doc_id duplicados"
    save_corpus(df, args.processed_dir / "corpus")
    print(df["doc_type"].value_counts().to_string())
    print(f"total: {len(df)} documentos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
