"""Carga el corpus y arma el recuperador listo para usar (CLI, API, evaluacion)."""

from pathlib import Path

import pandas as pd

from src.retrieval.dense import DEFAULT_MODEL, DenseIndex
from src.retrieval.hybrid import RERANKER_MODEL, HybridRetriever, Reranker

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def load_corpus(processed_dir: Path = PROCESSED) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / "corpus" / "documents.parquet")


def build_retriever(processed_dir: Path = PROCESSED, dense: bool = True, rerank: bool = True,
                    embed_model: str = DEFAULT_MODEL, rerank_model: str = RERANKER_MODEL) -> HybridRetriever:
    corpus = load_corpus(processed_dir)
    dense_index = None
    if dense:
        dense_index = DenseIndex.build(corpus["text"].tolist(), processed_dir / "index", embed_model)
    reranker = Reranker(rerank_model) if rerank else None
    return HybridRetriever(corpus, dense_index, reranker)
