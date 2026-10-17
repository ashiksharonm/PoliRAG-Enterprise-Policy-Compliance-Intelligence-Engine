"""Hybrid (BM25 + dense) retrieval over clauses, with optional cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from src.eval.retrieval_metrics import fuse

from .clauses import Clause, PolicyDoc
from .embedders import Embedder


class Reranker(Protocol):
    def scores(self, query: str, passages: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class Hit:
    clause: Clause
    rank: int
    score: float


class ClauseIndex:
    """In-memory index over the clauses of a corpus version. Cheap to rebuild per request."""

    def __init__(
        self,
        docs: list[PolicyDoc],
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 20,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> None:
        self.clauses = [c for d in docs for c in d.clauses]
        self.embedder, self.reranker = embedder, reranker
        self.candidates, self.sw, self.bw = candidates, semantic_weight, bm25_weight
        texts = [c.text for c in self.clauses]
        self._bm25 = BM25Okapi([t.lower().split() for t in texts]) if texts else None
        self._vec = embedder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float32)

    def search(self, query: str, k: int = 3, qvec: np.ndarray | None = None) -> list[Hit]:
        if not self.clauses or self._bm25 is None:
            return []
        q = qvec if qvec is not None else self.embedder.encode([query])[0]
        dense = self._vec @ q
        lex = self._bm25.get_scores(query.lower().split())
        n = min(2 * self.candidates, len(self.clauses))
        d_order = np.argsort(-dense, kind="stable")[:n]
        b_order = np.argsort(-lex, kind="stable")[:n]
        order = fuse(
            {int(i): float(dense[i]) for i in d_order},
            {int(i): float(lex[i]) for i in b_order},
            self.sw,
            self.bw,
        )
        order = order[: self.candidates]
        if self.reranker is not None and order:
            rs = np.asarray(self.reranker.scores(query, [self.clauses[i].text for i in order]))
            perm = np.argsort(-rs, kind="stable")
            order = [order[j] for j in perm]
            ranked = [float(rs[j]) for j in perm]
        else:
            ranked = [float(dense[i]) for i in order]
        return [Hit(self.clauses[i], r + 1, ranked[r]) for r, i in enumerate(order[:k])]
