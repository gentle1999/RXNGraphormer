# RXNGraphormer Lightning Components

This package contains the Lightning-native training surface:

- `module.py`: `RXNGraphormerLitModule`, the Lightning wrapper around the legacy torch model.
- `datamodule.py`: `RXNGraphormerDataModule`, split and dataloader construction.
- `callbacks.py`: runtime metric callbacks such as epoch time and CUDA peak memory.
- `trainer.py`: Trainer/callback/logger construction helpers used by CLI entry points.
- `workflow.py`: standard fit workflow that wires config overrides, data, module,
  checkpoint initialization, Trainer construction, and `trainer.fit`.
- `experiment.py` (parent package): manifest/report helpers used after fit.

The standard run layout is:

- `<default_root_dir>/<tag>/version_<n>/checkpoints/`: Lightning checkpoints.
- `<default_root_dir>/<tag>/version_<n>/reports/`: optional post-fit eval reports.
- `<default_root_dir>/<tag>/version_<n>/manifest.json`: full run manifest.
- `<default_root_dir>/manifest.json`: latest run manifest index copy.

Use `rxngraphormer.lightning` or `rxngraphormer.training` for public imports.
The old flat wrappers have been removed. CLI entry points should stay thin and
delegate training orchestration to `fit_config`/`build_fit_artifacts`.
