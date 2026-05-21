# RXNGraphormer Environments

This branch publishes one Python package, `rxngraphormer`. The package contains
model code, preprocessing code, configuration loading, and the dataset protocol.
Environment layout is an installation choice, not a package split.

## Extras

- default install: model training, evaluation, inference, checkpoint loading,
  and preprocessed `.pt` dataset consumption.
- `.[preprocess]`: preprocessing and atom-mapping dependencies.
- `.[all]`: default model dependencies plus preprocessing and sequence extras
  for a single end-to-end environment.

The package metadata keeps `torch`, `torch-geometric`, and DGL unbound to a
specific CUDA build. CUDA wheels are selected by `uv`, the configured package
indexes, or explicit install arguments.

## Hardware Boundary

The highest supported single-environment end-to-end target is NVIDIA
Hopper-class hardware (`sm_90`) or older. On these machines, the root
environment can install `rxngraphormer[all]` and run preprocessing, training,
and evaluation in one environment.

Blackwell-class or newer hardware should use split layout. The model environment
uses the newer Torch/CUDA wheel required by the GPU, while preprocessing runs
in a separate environment that installs `rxngraphormer[preprocess]`. The
preprocessing code path uses CPU for mapping, so DGL and Torch are normal
package dependencies rather than CUDA-suffixed package names.

## Unified Pipeline

Use one command entry point for both layouts:

```bash
scripts/rxngraphormer_pipeline.sh \
  --preprocess-config config/pretrain_parameters.json \
  --train-config config_toml/buchwald_hartwig_parameters.toml \
  --eval-config config/buchwald_hartwig_eval.json \
  --stages preprocess,train,eval
```

The default `--layout auto` mode reads GPU compute capability with
`nvidia-smi`. It selects `single` for `sm_90` or older and `split` for newer
architectures.

Force a layout when needed:

```bash
scripts/rxngraphormer_pipeline.sh --layout single
scripts/rxngraphormer_pipeline.sh --layout split
```

In `single` layout the root environment is synced and then receives
`rxngraphormer[all]`. In `split` layout the root environment is synced with
default model dependencies only, and `envs/preprocess/.venv` receives
`rxngraphormer[preprocess]`.

Hide CUDA during preprocessing if a local GPU stack interferes with mapping or
DGL import:

```bash
scripts/rxngraphormer_pipeline.sh --layout split --preprocess-cpu --stages preprocess
```

Pass extra uv installation options without changing package metadata:

```bash
scripts/rxngraphormer_pipeline.sh \
  --layout split \
  --model-sync-arg --upgrade \
  --preprocess-install-arg --torch-backend \
  --preprocess-install-arg cpu
```

## Direct Commands

Model-only environment:

```bash
uv sync
uv run python scripts/reproduce/check_environment.py model
```

Single end-to-end environment:

```bash
uv sync
uv pip install -e '.[all]'
uv run python scripts/reproduce/check_environment.py preprocess
```

Split preprocessing environment:

```bash
uv venv envs/preprocess/.venv --allow-existing
uv pip install --python envs/preprocess/.venv/bin/python -e '.[preprocess]'
envs/preprocess/.venv/bin/python scripts/reproduce/check_environment.py preprocess
```
