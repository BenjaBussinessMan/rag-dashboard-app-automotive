"""Recuperacion hibrida: BM25 + semantica fusionadas con RRF, y re-ranking opcional con cross-encoder."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.retrieval.bm25 import BM25Index
from src.retrieval.query import QueryParser

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
META_COLS = ["doc_type", "anio", "mes", "marca", "provincia", "combustible", "fuente", "pagina"]


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), rank desde 1."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float
    metadata: dict
    ranks: dict = field(default_factory=dict)  # posicion en cada recuperador (1 = mejor)
    rerank_score: float | None = None


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, max_length=512, device=device,
                                  model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32})

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=8,
                                             show_progress_bar=False), dtype=np.float32)


class HybridRetriever:
    def __init__(self, corpus: pd.DataFrame, dense=None, reranker: Reranker | None = None):
        self.corpus = corpus.reset_index(drop=True)
        self.texts = self.corpus["text"].tolist()
        self.bm25 = BM25Index(self.texts)
        self.dense = dense
        self.reranker = reranker
        self.parser = QueryParser(self.corpus)

    def _mask(self, filters: dict | None) -> np.ndarray:
        mask = np.ones(len(self.corpus), dtype=bool)
        for col, val in (filters or {}).items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            mask &= self.corpus[col].isin(vals).to_numpy()
        return mask

    @staticmethod
    def _ranking(scores: np.ndarray, mask: np.ndarray, n: int) -> list[int]:
        masked = np.where(mask, scores, -np.inf)
        order = np.argsort(-masked)[:n]
        return [int(i) for i in order if np.isfinite(masked[i])]

    def search(self, query: str, k: int = 5, mode: str = "hybrid", candidates: int = 30,
               filters: dict | None = None, rerank: bool = False, understand: bool = False,
               bm25_weight: float = 1.0) -> list[Hit]:
        """mode: 'bm25' | 'dense' | 'hybrid'. rerank reordena los `candidates` mejores.

        understand: extrae filtros de la consulta (año, marca, ...) y agrega recuperadores restringidos
        a esos metadatos dentro del RRF; es una señal suave, no excluye documentos sin esos campos.
        """
        mask = self._mask(filters)
        parsed = self.parser.parse(query) if understand else {}
        meta_mask = mask & self._mask(parsed) if parsed else None
        if meta_mask is not None and not meta_mask.any():
            meta_mask = None

        scorers = {}
        if mode in ("bm25", "hybrid"):
            scorers["bm25"] = (self.bm25.scores(query), bm25_weight)
        if mode in ("dense", "hybrid"):
            if self.dense is None:
                raise ValueError("no hay indice semantico cargado")
            scorers["dense"] = (self.dense.scores(query), 1.0)

        rankings: dict[str, list[int]] = {}
        weights: dict[str, float] = {}
        for name, (sc, w) in scorers.items():
            rankings[name], weights[name] = self._ranking(sc, mask, candidates), w
            if meta_mask is not None:
                rankings[f"{name}_meta"], weights[f"{name}_meta"] = self._ranking(sc, meta_mask, candidates), w

        ordered = rrf(list(rankings.values()), weights=list(weights.values()))[:candidates]

        hits = []
        for doc, score in ordered:
            row = self.corpus.iloc[doc]
            hits.append(Hit(
                doc_id=row["doc_id"], text=row["text"], score=float(score),
                metadata={c: (None if pd.isna(row[c]) else row[c]) for c in META_COLS if c in row},
                ranks={name: r.index(doc) + 1 for name, r in rankings.items() if doc in r}))

        if rerank:
            if self.reranker is None:
                raise ValueError("no hay reranker cargado")
            scores = self.reranker.score(query, [h.text for h in hits])
            for h, s in zip(hits, scores):
                h.rerank_score = float(s)
            hits.sort(key=lambda h: -h.rerank_score)
        return hits[:k]
