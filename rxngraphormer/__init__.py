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


def __getattr__(name):
    if name in {"CheckpointAdapter", "CheckpointLoadReport", "load_checkpoint"}:
        from . import checkpointing

        return getattr(checkpointing, name)
    if name in {"DataLoaderSettings", "RXNGraphormerDataModule"}:
        from . import datamodule

        return getattr(datamodule, name)
    if name in {"RegressionEvaluation", "RegressionMetrics", "evaluate_regression", "regression_metrics"}:
        from . import evaluator

        return getattr(evaluator, name)
    if name == "RXNGraphormerLitModule":
        from . import lightning_module

        return getattr(lightning_module, name)
    if name in {"RXNGraphormerPredictor", "export_predictions_csv", "export_embeddings_csv"}:
        from . import predictor

        return getattr(predictor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
