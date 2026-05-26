"""Evaluation import boundary for RXNGraphormer."""

from .evaluator import (
    ClassificationEvaluation,
    ClassificationMetrics,
    RegressionEvaluation,
    RegressionMetrics,
    classification_batch_input,
    classification_metrics,
    evaluate_classification,
    evaluate_regression,
    regression_batch_input,
    regression_metrics,
)
from .sequence_metrics import sequence_exact_match

_WORKFLOW_EXPORTS = {
    "ClassificationEvaluationSettings": "classification_workflow",
    "ClassificationSplitResult": "classification_workflow",
    "RegressionEvaluationSettings": "regression_workflow",
    "RegressionSplitResult": "regression_workflow",
    "evaluate_classification_split": "classification_workflow",
    "evaluate_classification_splits": "classification_workflow",
    "evaluate_regression_prediction": "regression_workflow",
    "evaluate_regression_split": "regression_workflow",
    "evaluate_regression_splits": "regression_workflow",
    "write_classification_csv": "classification_workflow",
    "write_classification_json": "classification_workflow",
    "write_regression_csv": "regression_workflow",
    "write_regression_json": "regression_workflow",
}

__all__ = [
    "ClassificationEvaluation",
    "ClassificationEvaluationSettings",
    "ClassificationMetrics",
    "ClassificationSplitResult",
    "RegressionEvaluation",
    "RegressionEvaluationSettings",
    "RegressionMetrics",
    "RegressionSplitResult",
    "classification_batch_input",
    "classification_metrics",
    "evaluate_classification",
    "evaluate_classification_split",
    "evaluate_classification_splits",
    "evaluate_regression",
    "evaluate_regression_prediction",
    "evaluate_regression_split",
    "evaluate_regression_splits",
    "regression_batch_input",
    "regression_metrics",
    "sequence_exact_match",
    "write_classification_csv",
    "write_classification_json",
    "write_regression_csv",
    "write_regression_json",
]


def __getattr__(name: str) -> object:
    module_name = _WORKFLOW_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if module_name == "classification_workflow":
        from . import classification_workflow

        return getattr(classification_workflow, name)
    from . import regression_workflow

    return getattr(regression_workflow, name)
