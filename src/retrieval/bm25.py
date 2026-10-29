"""BM25 (Okapi) propio sobre matriz dispersa; sin dependencias externas."""

import re
import unicodedata

import numpy as np
import scipy.sparse as sp

STOPWORDS = frozenset("""
a al algo ante con como cual cuales cuando de del el ella en entre es esta este esto la las lo los
me mi mis no o para por que se si sin su sus un una unas uno unos y
""".split())


def tokenize(text: str) -> list[str]:
    """Minusculas, sin tildes; '15.649' -> '15649' para que el numero sea un solo token."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    text = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", text)
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(texts))
        for i, text in enumerate(texts):
            toks = tokenize(text)
            lengths[i] = len(toks)
            counts: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            rows += [i] * len(counts)
            cols += counts.keys()
            vals += counts.values()
        self.vocab = vocab
        self.tf = sp.csc_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)), dtype=np.float32)
        df = np.diff(self.tf.indptr)
        n = len(texts)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1e-9)).astype(np.float32)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.tf.shape[0], dtype=np.float32)
        for term in set(tokenize(query)):
            j = self.vocab.get(term)
            if j is None:
                continue
            lo, hi = self.tf.indptr[j], self.tf.indptr[j + 1]
            docs, tf = self.tf.indices[lo:hi], self.tf.data[lo:hi]
            out[docs] += self.idf[j] * tf * (self.k1 + 1) / (tf + self.norm[docs])
        return out
