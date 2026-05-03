from __future__ import annotations

from src.training.compare_models import (
    ModelResult,
    build_comparison_table,
    build_threshold_comparison,
    generate_markdown_report,
    plot_confusion_matrices,
    plot_cost_comparison,
    plot_metrics_radar,
    plot_pr_comparison,
    plot_roc_comparison,
    plot_threshold_tradeoff,
    train_and_evaluate_dummy,
    train_and_evaluate_logistic,
    train_and_evaluate_mlp,
)
from src.training.dummy_trainer import (
    DummyTrainingConfig,
    run_all_strategies,
    train_dummy_strategy,
)
from src.training.logistic_trainer import (
    LogisticTrainingConfig,
    cross_validate_logistic,
    train_logistic_classifier,
)
from src.training.metrics import compute_binary_classification_metrics
from src.training.mlp import MLP, MLPForTraining, MLPTrainer
from src.training.model_card import build_model_card

__all__ = [
    "MLP",
    "DummyTrainingConfig",
    "LogisticTrainingConfig",
    "MLPForTraining",
    "MLPTrainer",
    "ModelResult",
    "build_comparison_table",
    "build_model_card",
    "build_threshold_comparison",
    "compute_binary_classification_metrics",
    "cross_validate_logistic",
    "generate_markdown_report",
    "plot_confusion_matrices",
    "plot_cost_comparison",
    "plot_metrics_radar",
    "plot_pr_comparison",
    "plot_roc_comparison",
    "plot_threshold_tradeoff",
    "run_all_strategies",
    "train_and_evaluate_dummy",
    "train_and_evaluate_logistic",
    "train_and_evaluate_mlp",
    "train_dummy_strategy",
    "train_logistic_classifier",
]
