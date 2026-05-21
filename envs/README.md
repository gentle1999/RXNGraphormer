# Environment Profiles

Only one package is published: `rxngraphormer`.

- repository root: model environment, or full `rxngraphormer[all]` environment
  on hardware that supports single-environment end-to-end execution.
- `envs/preprocess/.venv`: optional split preprocessing environment that
  installs the same package with `rxngraphormer[preprocess]`.

`envs/preprocess` does not pin a CUDA version. Torch, DGL, and mapping
dependencies are resolved by the installer and any user-provided uv arguments.
Use `scripts/rxngraphormer_pipeline.sh --layout auto` to deploy the appropriate
layout automatically.
