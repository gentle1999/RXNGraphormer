__all__ = [
    "CheckpointAdapter",
    "CheckpointLoadReport",
    "DataLoaderSettings",
    "RXNGraphormerDataModule",
    "RXNGraphormerLitModule",
    "RegressionEvaluation",
    "RegressionMetrics",
    "RXNGraphormerPredictor",
    "evaluate_regression",
    "export_embeddings_csv",
    "export_predictions_csv",
    "load_checkpoint",
    "regression_metrics",
]


def __getattr__(name: str) -> object:
    if name in {"CheckpointAdapter", "CheckpointLoadReport", "load_checkpoint"}:
        from . import compatibility

        return getattr(compatibility, name)
    if name in {"DataLoaderSettings", "RXNGraphormerDataModule"}:
        from . import training

        return getattr(training, name)
    if name in {"RegressionEvaluation", "RegressionMetrics", "evaluate_regression", "regression_metrics"}:
        from . import evaluation

        return getattr(evaluation, name)
    if name == "RXNGraphormerLitModule":
        from . import training

        return getattr(training, name)
    if name in {"RXNGraphormerPredictor", "export_predictions_csv", "export_embeddings_csv"}:
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
