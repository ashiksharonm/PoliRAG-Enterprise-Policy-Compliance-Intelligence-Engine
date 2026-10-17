"""Tests for span-based retrieval metrics and the gold set itself."""

import json
import math
from pathlib import Path

import pytest

from src.eval.retrieval_metrics import (
    bootstrap_ci, fuse, minmax, ndcg_at_k, overlaps, recall_at_k,
    reciprocal_rank, relevance_flags, resolve_quote,
)

ROOT = Path(__file__).resolve().parents[1]
A = ("d", 0, 10)
B = ("d", 10, 20)
C = ("d", 20, 30)
OTHER = ("x", 0, 10)


def test_overlap_is_half_open_and_doc_specific():
    assert overlaps(A, ("d", 5, 15))
    assert not overlaps(A, B)  # touching, not overlapping
    assert not overlaps(A, OTHER)


def test_recall_multihop_needs_both_spans():
    gold = [("d", 2, 4), ("d", 22, 24)]
    assert recall_at_k([A, B, C], gold, 1) == 0.5
    assert recall_at_k([A, B, C], gold, 3) == 1.0
    assert recall_at_k([OTHER, B], gold, 2) == 0.0


def test_recall_requires_gold():
    with pytest.raises(ValueError):
        recall_at_k([A], [], 1)


def test_reciprocal_rank():
    gold = [("d", 25, 26)]
    assert reciprocal_rank([A, B, C], gold) == pytest.approx(1 / 3)
    assert reciprocal_rank([A, B], gold) == 0.0


def test_ndcg_known_values():
    gold = [("d", 12, 13)]
    # single relevant chunk at rank 2: DCG = 1/log2(3), ideal = 1
    assert ndcg_at_k([A, B, C], gold, 10, 1) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([B, A, C], gold, 10, 1) == pytest.approx(1.0)
    assert ndcg_at_k([A, C], gold, 10, 1) == 0.0
    assert relevance_flags([A, B], gold) == [0, 1]


def test_ndcg_two_relevant_ideal_normalisation():
    gold = [("d", 2, 3), ("d", 12, 13)]
    perfect = ndcg_at_k([A, B, C], gold, 10, 2)
    assert perfect == pytest.approx(1.0)
    assert ndcg_at_k([A, C, B], gold, 10, 2) < 1.0


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.0, 1.0] * 20
    m, lo, hi = bootstrap_ci(vals, seed=1)
    assert m == 0.5 and lo < m < hi
    assert (m, lo, hi) == bootstrap_ci(vals, seed=1)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)


def test_resolve_quote_whitespace_and_uniqueness():
    text = "alpha beta\ngamma  delta. alpha"
    s, e = resolve_quote(text, "beta gamma delta")
    assert text[s:e] == "beta\ngamma  delta"
    with pytest.raises(ValueError):
        resolve_quote(text, "alpha")
    with pytest.raises(ValueError):
        resolve_quote(text, "missing")


def test_minmax_and_fuse():
    assert minmax([1, 3, 2]) == [0.0, 1.0, 0.5]
    assert minmax([2, 2]) == [1.0, 1.0]
    assert fuse({0: 1.0, 1: 0.0}, {1: 5.0, 2: 0.0}) == [0, 1, 2]
    assert fuse({0: 1.0, 1: 0.0}, {1: 5.0, 2: 0.0}, 0.3, 0.7)[0] == 1


def test_gold_set_is_well_formed():
    gold = json.loads((ROOT / "data/eval/retrieval_gold.json").read_text())
    qs = gold["questions"]
    assert len(qs) >= 60
    assert len({q["id"] for q in qs}) == len(qs)
    assert {q["type"] for q in qs} == {
        "factual", "numeric", "multi_hop", "negation", "paraphrase", "not_covered"}
    for q in qs:
        assert q["question"].strip() and q["gold_answer"].strip()
        assert (len(q["evidence"]) == 0) == (q["type"] == "not_covered")
        for ev in q["evidence"]:
            text = (ROOT / gold["corpus_dir"] / ev["doc"]).read_text()
            resolve_quote(text, ev["quote"])  # raises unless unique
