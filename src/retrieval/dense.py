"""Indice semantico con embeddings locales (BGE-M3 por defecto), cacheado en disco."""

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer
    device = _device()
    model = SentenceTransformer(name, device=device)
    if device == "cuda":
        model.half()
    model.max_seq_length = 512
    return model


def corpus_fingerprint(model_name: str, texts: list[str]) -> str:
    h = hashlib.sha256(model_name.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, model, model_name: str = DEFAULT_MODEL):
        self.embeddings = embeddings
        self.model = model
        self.model_name = model_name

    @classmethod
    def build(cls, texts: list[str], index_dir: Path, model_name: str = DEFAULT_MODEL, model=None,
              batch_size: int = 16) -> "DenseIndex":
        """Reutiliza embeddings en disco si el corpus y el modelo no cambiaron."""
        index_dir.mkdir(parents=True, exist_ok=True)
        emb_path, meta_path = index_dir / "embeddings.npy", index_dir / "embeddings_meta.json"
        fingerprint = corpus_fingerprint(model_name, texts)
        model = model or load_model(model_name)
        if emb_path.exists() and meta_path.exists():
            if json.loads(meta_path.read_text())["fingerprint"] == fingerprint:
                return cls(np.load(emb_path), model, model_name)
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(emb_path, emb)
        meta_path.write_text(json.dumps({"fingerprint": fingerprint, "model": model_name,
                                         "n": len(texts), "dim": int(emb.shape[1])}))
        return cls(emb, model, model_name)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.embeddings @ q.astype(np.float32)
