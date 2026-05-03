"""Tests for model_card module."""

# ruff: noqa: PLR2004

from __future__ import annotations

import json

import pytest

from src.training.model_card import build_model_card

REQUIRED_SECTIONS = [
    "model_details",
    "intended_use",
    "factors",
    "metrics",
    "evaluation_data",
    "training_data",
    "quantitative_analyses",
    "ethical_considerations",
    "caveats_and_recommendations",
    "failure_scenarios",
]


@pytest.mark.fast
class TestBuildModelCard:
    """Tests for build_model_card function."""

    @staticmethod
    def test_all_sections_present_with_minimal_values() -> None:
        """All 10 sections should be present even with no extra values."""
        card = build_model_card("mlp")
        for section in REQUIRED_SECTIONS:
            assert section in card, f"Missing section: {section}"

    @staticmethod
    def test_model_details_populated() -> None:
        """Model details should reflect model_type and passed values."""
        card = build_model_card(
            "mlp",
            random_seed=99,
            dataset_version="abc123def",
        )
        md = card["model_details"]
        assert md["model_type"] == "mlp"
        assert md["seed"] == 99
        assert md["dataset_version"] == "abc123def"
        assert "PyTorch" in md["framework"]

    @staticmethod
    def test_different_model_types() -> None:
        """Each model_type should have distinct info."""
        dummy = build_model_card("dummy")
        logistic = build_model_card("logistic")
        mlp = build_model_card("mlp")

        assert "DummyClassifier" in dummy["model_details"]["model_name"]
        assert "Logistic Regression" in logistic["model_details"]["model_name"]
        assert "MLP" in mlp["model_details"]["model_name"]

    @staticmethod
    def test_metrics_populated_from_values() -> None:
        """Passed metric values should appear in the metrics section."""
        card = build_model_card(
            "mlp",
            accuracy=0.85,
            precision=0.72,
            recall=0.68,
            f1_score=0.70,
            roc_auc=0.92,
            pr_auc=0.81,
            brier_score=0.12,
            ece=0.05,
        )
        entries = card["metrics"]["primary_metrics"]
        keys_found = {e["key"] for e in entries}
        assert "accuracy" in keys_found
        assert "roc_auc" in keys_found
        assert "ece" in keys_found
        assert all(isinstance(e["value"], float) for e in entries)

    @staticmethod
    def test_confusion_matrix_when_present() -> None:
        """Confusion matrix should be included when tn/fp/fn/tp provided."""
        card = build_model_card("mlp", tn=800, fp=50, fn=30, tp=120)
        cm = card["metrics"]["confusion_matrix"]
        assert cm == {"tn": 800, "fp": 50, "fn": 30, "tp": 120}

    @staticmethod
    def test_confusion_matrix_omitted_when_missing() -> None:
        """Confusion matrix should be absent when not provided."""
        card = build_model_card("mlp", accuracy=0.9)
        assert "confusion_matrix" not in card["metrics"]

    @staticmethod
    def test_cost_analysis_populated() -> None:
        """Cost analysis should include provided cost values."""
        card = build_model_card(
            "mlp",
            cost_fn=500.0,
            cost_fp=50.0,
            total_cost=35000.0,
        )
        qa = card["quantitative_analyses"]
        assert qa["cost_analysis"]["cost_fn"] == 500.0
        assert qa["cost_analysis"]["cost_fp"] == 50.0
        assert qa["cost_analysis"]["total_cost"] == 35000.0

    @staticmethod
    def test_optimal_threshold_populated() -> None:
        """Optimal threshold should appear when provided."""
        card = build_model_card(
            "mlp", optimal_threshold=0.42, optimal_total_cost=28000.0
        )
        qa = card["quantitative_analyses"]
        assert qa["optimal_threshold"] == 0.42
        assert qa["optimal_total_cost"] == 28000.0

    @staticmethod
    def test_risk_band_metrics_populated() -> None:
        """Risk band metrics should be populated when provided."""
        card = build_model_card(
            "mlp",
            pct_low=0.40,
            pct_medium=0.45,
            pct_high=0.15,
            churn_rate_low=0.05,
            churn_rate_medium=0.30,
            churn_rate_high=0.70,
            capture_high=0.60,
        )
        rb = card["quantitative_analyses"]["risk_bands"]
        assert "definitions" in rb
        assert rb["metrics"]["pct_low"] == 0.40
        assert rb["metrics"]["churn_rate_high"] == 0.70

    @staticmethod
    def test_precision_recall_at_k_populated() -> None:
        """Precision@k and recall@k should be included when provided."""
        card = build_model_card(
            "mlp",
            precision_at_100=0.90,
            recall_at_100=0.45,
            precision_at_500=0.75,
            recall_at_500=0.80,
        )
        pk = card["quantitative_analyses"]["precision_recall_at_k"]
        assert pk["precision_at_100"] == 0.90
        assert pk["recall_at_500"] == 0.80

    @staticmethod
    def test_calibration_included() -> None:
        """Calibration metrics should appear when provided."""
        card = build_model_card("mlp", brier_score=0.10, ece=0.04)
        cal = card["quantitative_analyses"]["calibration"]
        assert cal["brier_score"] == 0.10
        assert cal["ece"] == 0.04

    @staticmethod
    def test_static_sections_present() -> None:
        """Static sections (intended_use, factors, ethical, failure) must
        always be present with content."""
        card = build_model_card("mlp")

        assert "primary" in card["intended_use"]
        assert "class_imbalance" in card["factors"]
        assert len(card["ethical_considerations"]["biases"]) >= 3
        assert len(card["caveats_and_recommendations"]["limitations"]) >= 4
        assert len(card["failure_scenarios"]["data_failures"]) >= 2

    @staticmethod
    def test_output_is_json_serializable() -> None:
        """Output must be JSON-serializable for mlflow.log_dict."""
        card = build_model_card(
            "mlp",
            accuracy=0.85,
            f1_score=0.70,
            total_cost=35000.0,
            random_seed=42,
            dataset_version="abc123",
        )
        json_str = json.dumps(card, indent=2)
        loaded = json.loads(json_str)
        assert loaded["model_details"]["model_type"] == "mlp"
        assert loaded["model_details"]["seed"] == 42
