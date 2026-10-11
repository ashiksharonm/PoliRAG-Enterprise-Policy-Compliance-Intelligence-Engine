"""Pure-Python retrieval metrics, span-based relevance and bootstrap CIs.

Relevance is defined on character spans of the source documents rather than
on chunk ids, so the same gold set can score any chunking strategy: a retrieved
chunk is relevant if it overlaps a gold evidence span in the same document.
"""

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

Span = Tuple[str, int, int]  # (doc, start_char, end_char)


def resolve_quote(text: str, quote: str) -> Tuple[int, int]:
    """Locate ``quote`` in ``text`` ignoring whitespace differences.

    Raises ValueError unless the quote occurs exactly once.
    """
    pattern = r"\s+".join(re.escape(tok) for tok in quote.split())
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"quote matched {len(matches)} times: {quote!r}")
    return matches[0].start(), matches[0].end()


def overlaps(a: Span, b: Span) -> bool:
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def relevance_flags(ranked: Sequence[Span], gold: Sequence[Span]) -> List[int]:
    """1 if the ranked chunk overlaps any gold span, else 0."""
    return [int(any(overlaps(c, g) for g in gold)) for c in ranked]


def recall_at_k(ranked: Sequence[Span], gold: Sequence[Span], k: int) -> float:
    """Fraction of gold evidence spans covered by at least one top-k chunk.

    For multi-hop questions this requires both supporting passages.
    """
    if not gold:
        raise ValueError("recall is undefined without gold evidence")
    top = ranked[:k]
    return sum(any(overlaps(c, g) for c in top) for g in gold) / len(gold)


def reciprocal_rank(ranked: Sequence[Span], gold: Sequence[Span]) -> float:
    for rank, rel in enumerate(relevance_flags(ranked, gold), 1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[Span], gold: Sequence[Span], k: int,
              n_relevant_total: int) -> float:
    """Binary-gain nDCG@k; the ideal list places min(k, n_relevant_total) hits first."""
    flags = relevance_flags(ranked[:k], gold)
    dcg = sum(f / np.log2(i + 2) for i, f in enumerate(flags))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, n_relevant_total)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap CI, resampling queries with replacement."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def minmax(scores: Sequence[float]) -> List[float]:
    """Min-max normalisation, identical to HybridRetriever._normalize_scores."""
    if not len(scores):
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def fuse(semantic: Dict[int, float], bm25: Dict[int, float],
         semantic_weight: float = 0.7, bm25_weight: float = 0.3) -> List[int]:
    """Weighted sum of min-max normalised scores, as in HybridRetriever._hybrid_search.

    Inputs map chunk index -> raw score over each retriever's candidate list.
    Returns chunk indices sorted by fused score (descending, stable).
    """
    combined: Dict[int, float] = {}
    for scores, w in ((semantic, semantic_weight), (bm25, bm25_weight)):
        ids = list(scores)
        for i, s in zip(ids, minmax([scores[i] for i in ids])):
            combined[i] = combined.get(i, 0.0) + s * w
    return sorted(combined, key=lambda i: combined[i], reverse=True)
