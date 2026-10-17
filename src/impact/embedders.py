"""Embedding backends: a deterministic hashing embedder and sentence-transformers (CPU)."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (len(texts), dim)."""


class HashingEmbedder:
    """Signed feature hashing of word unigrams and bigrams.

    Needs no model download and is fully deterministic, so tests and offline runs are stable.
    It is lexical, not semantic: use sentence-transformers when quality matters.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            for feat in words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]:
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                out[r, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1.0, norm)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return np.asarray(v, dtype=np.float32)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device="cpu")

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return [
            float(s) for s in self._model.predict([(query, p) for p in passages], batch_size=32)
        ]


class CachedEmbedder:
    """Memoises vectors by text, so re-indexing unchanged clauses or repeated questions is free."""

    def __init__(self, inner: Embedder, max_items: int = 20000) -> None:
        self.inner, self.name, self._max = inner, inner.name, max_items
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            if len(self._cache) + len(missing) > self._max:
                self._cache.clear()
            for t, v in zip(missing, self.inner.encode(missing), strict=True):
                self._cache[t] = v
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])


def make_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hashing":
        return CachedEmbedder(HashingEmbedder())
    if backend == "sentence-transformers":
        return CachedEmbedder(SentenceTransformerEmbedder(model_name))
    raise ValueError(f"unknown embedding backend {backend!r} (hashing | sentence-transformers)")
