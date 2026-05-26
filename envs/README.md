# Environment Profiles

Only one package is published: `rxngraphormer`.

- repository root: routed model environment. The router installs
  `rxngraphormer[all]` here for model, evaluation, prediction, compatibility,
  and sequence-generation commands.
- `envs/preprocess/.venv`: optional split preprocessing environment that
  installs the same package with `rxngraphormer[preprocess]`.

`envs/preprocess` does not pin a CUDA version. Torch, DGL, and mapping
dependencies are resolved by the installer and any user-provided uv arguments.
Use `scripts/rxngraphormer_pipeline.sh --layout auto <command> [args...]` to
route a CLI command to the appropriate environment. The router creates the
selected environment automatically when its virtualenv or required modules are
missing.
