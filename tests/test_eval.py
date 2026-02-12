"""Tests for evaluation: metrics calculations, golden dataset management."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.eval.golden_dataset import GoldenDatasetManager
from src.eval.metrics import EvaluationMetrics


# ---------------------------------------------------------------------------
# Recall@K tests
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def setup_method(self):
        self.metrics = EvaluationMetrics()

    def test_perfect_recall(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]
        assert self.metrics.calculate_recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_no_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        assert self.metrics.calculate_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "b", "y", "z"]
        relevant = ["a", "b", "c"]
        recall = self.metrics.calculate_recall_at_k(retrieved, relevant, k=5)
        assert abs(recall - 2 / 3) < 1e-9

    def test_empty_relevant(self):
        assert self.metrics.calculate_recall_at_k(["a"], [], k=1) == 0.0


# ---------------------------------------------------------------------------
# MRR tests
# ---------------------------------------------------------------------------

class TestMRR:
    def setup_method(self):
        self.metrics = EvaluationMetrics()

    def test_first_position(self):
        assert self.metrics.calculate_mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_third_position(self):
        mrr = self.metrics.calculate_mrr(["x", "y", "a"], ["a"])
        assert abs(mrr - 1 / 3) < 1e-9

    def test_not_found(self):
        assert self.metrics.calculate_mrr(["x", "y"], ["a"]) == 0.0


# ---------------------------------------------------------------------------
# Precision@K tests
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def setup_method(self):
        self.metrics = EvaluationMetrics()

    def test_perfect_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert self.metrics.calculate_precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_half_precision(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = ["a", "b"]
        precision = self.metrics.calculate_precision_at_k(retrieved, relevant, k=4)
        assert abs(precision - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Golden dataset tests
# ---------------------------------------------------------------------------

class TestGoldenDatasetManager:
    def test_create_and_add(self, mock_settings):
        manager = GoldenDatasetManager()
        ds = manager.create_dataset("test_set")
        manager.add_qa_pair(ds, question="What is GDPR?", category="privacy")
        assert len(ds.qa_pairs) == 1
        assert ds.qa_pairs[0].question == "What is GDPR?"

    def test_save_and_load_roundtrip(self, mock_settings, tmp_path):
        manager = GoldenDatasetManager()
        ds = manager.create_dataset("roundtrip", description="test")
        manager.add_qa_pair(ds, question="Q1", category="cat1")
        manager.add_qa_pair(ds, question="Q2", difficulty="hard")

        path = tmp_path / "test_golden.json"
        manager.save_dataset(ds, path)
        loaded = manager.load_dataset(path)

        assert loaded.name == "roundtrip"
        assert len(loaded.qa_pairs) == 2

    def test_filter_by_category(self, mock_settings):
        manager = GoldenDatasetManager()
        ds = manager.create_dataset("filter_test")
        manager.add_qa_pair(ds, question="Q1", category="a")
        manager.add_qa_pair(ds, question="Q2", category="b")
        manager.add_qa_pair(ds, question="Q3", category="a")

        filtered = manager.get_by_category(ds, "a")
        assert len(filtered) == 2

    def test_sample_dataset(self, mock_settings):
        manager = GoldenDatasetManager()
        ds = manager.create_sample_dataset()
        assert len(ds.qa_pairs) == 5
