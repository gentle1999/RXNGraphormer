# RXNGraphormer Environments

This branch publishes one Python package, `rxngraphormer`. The package contains
model code, preprocessing code, configuration loading, and the dataset protocol.
Environment layout is an installation choice, not a package split.

## Extras

- default install: core model training, evaluation, inference, checkpoint
  loading, and preprocessed `.pt` dataset consumption.
- `.[preprocess]`: preprocessing and atom-mapping dependencies.
- `.[all]`: full routed CLI runtime: core model dependencies, preprocessing
  and atom-mapping dependencies, and OpenNMT sequence-generation dependencies.

The package metadata keeps `torch`, `torch-geometric`, and DGL unbound to a
specific CUDA build. CUDA wheels are selected by `uv`, the configured package
indexes, or explicit install arguments.

## Hardware Boundary

The highest supported single-environment end-to-end target is NVIDIA
Hopper-class hardware (`sm_90`) or older. On these machines, the root
environment can install `rxngraphormer[all]` and run preprocessing, training,
and evaluation in one environment.

Blackwell-class or newer hardware should use split layout. The model
environment still installs `rxngraphormer[all]` with the newer Torch/CUDA wheel
required by the GPU, while preprocessing commands run in a separate environment
that installs `rxngraphormer[preprocess]`. The preprocessing code path uses CPU
for mapping, so DGL and Torch are normal package dependencies rather than
CUDA-suffixed package names.

## CLI Router

Use `scripts/rxngraphormer_pipeline.sh` as the user-facing environment router
for the published CLI entry points. Router options come before the command.
Arguments after the command are passed unchanged to the selected RXNGraphormer
CLI.

```bash
scripts/rxngraphormer_pipeline.sh preprocess --config config/pretrain_parameters.json
scripts/rxngraphormer_pipeline.sh train --config config_toml/buchwald_hartwig_parameters.toml
scripts/rxngraphormer_pipeline.sh eval --config config/buchwald_hartwig_eval.json
```

The router also accepts the full console script names, but the routed form is
still preferred because it keeps hardware/environment selection in one place:

```bash
scripts/rxngraphormer_pipeline.sh rxngraphormer-preprocess --config config/pretrain_parameters.json
scripts/rxngraphormer_pipeline.sh rxngraphormer-train --config config_toml/buchwald_hartwig_parameters.toml
scripts/rxngraphormer_pipeline.sh rxngraphormer-eval --config config/buchwald_hartwig_eval.json
```

The default `--layout auto` mode reads GPU compute capability with
`nvidia-smi`. It selects `single` for `sm_90` or older and `split` for newer
architectures.

Force a layout when needed:

```bash
scripts/rxngraphormer_pipeline.sh --layout single rxngraphormer --help
scripts/rxngraphormer_pipeline.sh --layout split rxngraphormer --help
```

By default, the router initializes only the selected runtime environment if its
virtualenv or required modules are missing. Use `--sync` to force a refresh, or
`--no-sync` to disable automatic initialization.

In `single` layout all commands run in the root `rxngraphormer[all]`
environment. In `split` layout model commands still run in the root
`rxngraphormer[all]` environment, while preprocessing commands run in
`envs/preprocess/.venv` with `rxngraphormer[preprocess]`.

Hide CUDA during preprocessing if a local GPU stack interferes with mapping or
DGL import:

```bash
scripts/rxngraphormer_pipeline.sh --layout split --preprocess-cpu preprocess --config config/pretrain_parameters.json
```

Pass extra uv installation options without changing package metadata:

```bash
scripts/rxngraphormer_pipeline.sh \
  --layout split \
  --sync \
  --model-sync-arg --upgrade \
  train --config config_toml/buchwald_hartwig_parameters.toml

scripts/rxngraphormer_pipeline.sh \
  --layout split \
  --sync \
  --preprocess-install-arg --torch-backend \
  --preprocess-install-arg cpu \
  preprocess --config config/pretrain_parameters.json
```

Inspect a route without creating or running an environment:

```bash
scripts/rxngraphormer_pipeline.sh --layout split --dry-run preprocess --help
```

## Environment Checks

Use the router for environment initialization and compatibility checks. It
creates the selected environment when needed before running the check.

Model environment:

```bash
scripts/rxngraphormer_pipeline.sh --check-env rxngraphormer --help
```

Single end-to-end preprocessing profile:

```bash
scripts/rxngraphormer_pipeline.sh --layout single --check-env preprocess --help
```

Split preprocessing environment:

```bash
scripts/rxngraphormer_pipeline.sh --layout split --check-env preprocess --help
```
