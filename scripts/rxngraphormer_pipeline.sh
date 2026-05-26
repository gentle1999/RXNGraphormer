#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ENV_DIR="$ROOT_DIR"
PREPROCESS_PROJECT_DIR="$ROOT_DIR/envs/preprocess"
PREPROCESS_VENV_DIR="$PREPROCESS_PROJECT_DIR/.venv"

LAYOUT="auto"
SYNC_MODE="auto"
CHECK_ENV=0
PREPROCESS_CPU=0
DRY_RUN=0

MODEL_SYNC_ARGS=()
PREPROCESS_INSTALL_ARGS=()
CLI_ARGS=()
COMMAND=""
ENTRY=""
ROUTE_TARGET=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/rxngraphormer_pipeline.sh [router-options] <command> [cli-args...]

Environment router for the RXNGraphormer CLI entry points. Router options must
appear before <command>; every argument after <command> is passed unchanged to
the selected RXNGraphormer CLI.

Commands:
  rxngraphormer
  rxngraphormer-train
  rxngraphormer-train-legacy
  rxngraphormer-train-lit
  rxngraphormer-eval
  rxngraphormer-eval-legacy
  rxngraphormer-compat
  rxngraphormer-predict
  rxngraphormer-predict-sequence
  rxngraphormer-preprocess

Aliases:
  train              -> rxngraphormer-train
  train-legacy       -> rxngraphormer-train-legacy
  train-lit          -> rxngraphormer-train-lit
  eval               -> rxngraphormer-eval
  eval-legacy        -> rxngraphormer-eval-legacy
  compat             -> rxngraphormer-compat
  predict            -> rxngraphormer-predict
  predict-sequence   -> rxngraphormer-predict-sequence
  preprocess         -> rxngraphormer-preprocess

Router options:
  --layout NAME              Environment layout: auto,single,split
                             Default: auto
  --model-env DIR            Model environment project directory
                             Default: repository root
  --preprocess-project DIR   Preprocessing project directory
                             Default: envs/preprocess
  --preprocess-venv DIR      Virtualenv used by split preprocessing
                             Default: envs/preprocess/.venv
  --sync                     Sync the selected environment before running
  --no-sync                  Never sync before running
  --check-env                Run the selected environment compatibility check
                             before executing the command
  --no-check-env             Disable compatibility checks
                             Default
  --preprocess-cpu           Hide CUDA devices for split preprocessing commands
                             and preprocessing compatibility checks
  --model-sync-arg ARG       Extra argument passed to uv sync for the model env
  --preprocess-install-arg ARG
                             Extra argument passed to uv pip install in the
                             split preprocessing environment
  --dry-run                  Print the resolved route without executing it
  -h, --help                 Show this help

Routing:
  auto layout selects single for machines without NVIDIA GPUs or with max GPU
  compute capability <= 9.0. It selects split for newer GPUs.

  By default, the router auto-initializes the selected environment if its
  virtualenv or required modules are missing. Use --no-sync to disable this.

  single layout sends every command to the root model environment.
  split layout sends rxngraphormer-preprocess, preprocess, and
  "rxngraphormer preprocess ..." to envs/preprocess/.venv. Other commands run
  in the root model environment.

Examples:
  scripts/rxngraphormer_pipeline.sh train --config config_toml/bh_scratch_reproduce.toml
  scripts/rxngraphormer_pipeline.sh preprocess --config config/pretrain_parameters.json
  scripts/rxngraphormer_pipeline.sh --layout split --sync preprocess --config config/pretrain_parameters.json
  scripts/rxngraphormer_pipeline.sh --dry-run rxngraphormer-preprocess --help
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT_DIR/$path"
  fi
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "directory not found: $path"
}

require_uv() {
  command -v uv >/dev/null 2>&1 || die "uv is required for this route but was not found on PATH"
}

uv_in_project() {
  local project_dir="$1"
  shift
  (cd "$project_dir" && uv "$@")
}

python_has_modules() {
  local python="$1"
  shift
  [[ -x "$python" ]] || return 1

  "$python" - "$@" >/dev/null 2>&1 <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

detect_max_compute_capability() {
  local caps max cap
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '0\n'
    return
  fi

  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  max="0"
  while IFS= read -r cap; do
    cap="${cap//[[:space:]]/}"
    [[ -n "$cap" ]] || continue
    if awk -v lhs="$cap" -v rhs="$max" 'BEGIN { exit (lhs + 0 > rhs + 0) ? 0 : 1 }'; then
      max="$cap"
    fi
  done <<< "$caps"
  printf '%s\n' "$max"
}

resolve_layout() {
  case "$LAYOUT" in
    single|split)
      ;;
    auto)
      local max_cc
      max_cc="$(detect_max_compute_capability)"
      if awk -v cap="$max_cc" 'BEGIN { exit (cap + 0 <= 9.0) ? 0 : 1 }'; then
        LAYOUT="single"
      else
        LAYOUT="split"
      fi
      echo "==> Auto layout: max GPU compute capability $max_cc; selected $LAYOUT"
      ;;
    *)
      die "--layout must be one of: auto, single, split"
      ;;
  esac
}

resolve_entry() {
  local command="$1"
  case "$command" in
    rxngraphormer|rxngraphormer-train|rxngraphormer-train-legacy|rxngraphormer-train-lit|\
rxngraphormer-eval|rxngraphormer-eval-legacy|rxngraphormer-compat|rxngraphormer-predict|\
rxngraphormer-predict-sequence|rxngraphormer-preprocess)
      printf '%s\n' "$command"
      ;;
    train)
      printf '%s\n' "rxngraphormer-train"
      ;;
    train-legacy)
      printf '%s\n' "rxngraphormer-train-legacy"
      ;;
    train-lit)
      printf '%s\n' "rxngraphormer-train-lit"
      ;;
    eval)
      printf '%s\n' "rxngraphormer-eval"
      ;;
    eval-legacy)
      printf '%s\n' "rxngraphormer-eval-legacy"
      ;;
    compat)
      printf '%s\n' "rxngraphormer-compat"
      ;;
    predict)
      printf '%s\n' "rxngraphormer-predict"
      ;;
    predict-sequence)
      printf '%s\n' "rxngraphormer-predict-sequence"
      ;;
    preprocess)
      printf '%s\n' "rxngraphormer-preprocess"
      ;;
    *)
      die "unsupported command: $command"
      ;;
  esac
}

resolve_route_target() {
  local entry="$1"
  if [[ "$entry" == "rxngraphormer-preprocess" ]]; then
    printf '%s\n' "preprocess"
    return
  fi

  if [[ "$entry" == "rxngraphormer" && "${CLI_ARGS[0]:-}" == "preprocess" ]]; then
    printf '%s\n' "preprocess"
    return
  fi

  printf '%s\n' "model"
}

selected_runtime_env() {
  if [[ "$LAYOUT" == "split" && "$ROUTE_TARGET" == "preprocess" ]]; then
    printf '%s\n' "preprocess"
  else
    printf '%s\n' "model"
  fi
}

selected_bin_path() {
  local runtime_env="$1"
  if [[ "$runtime_env" == "preprocess" ]]; then
    printf '%s/bin/%s\n' "$PREPROCESS_VENV_DIR" "$ENTRY"
  else
    printf '%s/.venv/bin/%s\n' "$MODEL_ENV_DIR" "$ENTRY"
  fi
}

sync_model_env() {
  require_uv
  require_dir "$MODEL_ENV_DIR"

  local cmd=(sync)
  cmd+=("${MODEL_SYNC_ARGS[@]}")
  echo "==> Sync model environment: $MODEL_ENV_DIR"
  uv_in_project "$MODEL_ENV_DIR" "${cmd[@]}"

  echo "==> Install rxngraphormer[all] into model environment"
  uv_in_project "$MODEL_ENV_DIR" pip install -e "$ROOT_DIR[all]"
}

sync_preprocess_env() {
  require_uv
  mkdir -p "$PREPROCESS_PROJECT_DIR"

  echo "==> Create preprocessing virtualenv: $PREPROCESS_VENV_DIR"
  uv venv "$PREPROCESS_VENV_DIR" --allow-existing

  local install_cmd=(pip install --python "$PREPROCESS_VENV_DIR/bin/python" -e "$ROOT_DIR[preprocess]")
  install_cmd+=("${PREPROCESS_INSTALL_ARGS[@]}")
  echo "==> Install rxngraphormer[preprocess] into preprocessing environment"
  uv "${install_cmd[@]}"
}

sync_selected_env() {
  local runtime_env
  runtime_env="$(selected_runtime_env)"
  if [[ "$runtime_env" == "preprocess" ]]; then
    sync_preprocess_env
  else
    sync_model_env
  fi
}

should_auto_sync() {
  local runtime_env bin_path
  runtime_env="$(selected_runtime_env)"
  bin_path="$(selected_bin_path "$runtime_env")"

  if [[ "$runtime_env" == "preprocess" ]]; then
    if [[ ! -x "$bin_path" ]]; then
      return 0
    fi
    if ! python_has_modules "$PREPROCESS_VENV_DIR/bin/python" \
      rxngraphormer.preprocessing torch torch_geometric pandas dgl dgllife rdkit rxnmapper localmapper; then
      return 0
    fi
    return 1
  fi

  local model_python="$MODEL_ENV_DIR/.venv/bin/python"
  if [[ ! -x "$model_python" || ! -x "$bin_path" ]]; then
    return 0
  fi

  if ! python_has_modules "$model_python" \
    rxngraphormer rxngraphormer.preprocessing torch torch_geometric pandas rdkit sklearn safetensors \
    dgl dgllife rxnmapper localmapper onmt; then
    return 0
  fi

  return 1
}

sync_if_needed() {
  case "$SYNC_MODE" in
    never)
      return
      ;;
    always)
      sync_selected_env
      ;;
    auto)
      if should_auto_sync; then
        sync_selected_env
      fi
      ;;
    *)
      die "internal error: invalid sync mode: $SYNC_MODE"
      ;;
  esac
}

check_model_env() {
  local profile="$1"
  require_uv
  require_dir "$MODEL_ENV_DIR"
  echo "==> Check model environment profile: $profile"
  uv_in_project "$MODEL_ENV_DIR" run --no-sync python scripts/reproduce/check_environment.py "$profile"
}

check_preprocess_env() {
  [[ -x "$PREPROCESS_VENV_DIR/bin/python" ]] || die "preprocessing environment is missing: $PREPROCESS_VENV_DIR"
  echo "==> Check preprocessing environment"
  if [[ "$PREPROCESS_CPU" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="" "$PREPROCESS_VENV_DIR/bin/python" "$ROOT_DIR/scripts/reproduce/check_environment.py" preprocess
  else
    "$PREPROCESS_VENV_DIR/bin/python" "$ROOT_DIR/scripts/reproduce/check_environment.py" preprocess
  fi
}

check_selected_env() {
  [[ "$CHECK_ENV" -eq 1 ]] || return

  if [[ "$LAYOUT" == "split" && "$ROUTE_TARGET" == "preprocess" ]]; then
    check_preprocess_env
    return
  fi

  if [[ "$ROUTE_TARGET" == "preprocess" ]]; then
    check_model_env preprocess
  else
    check_model_env model
  fi
}

print_route() {
  local runtime_env bin_path
  runtime_env="$(selected_runtime_env)"
  bin_path="$(selected_bin_path "$runtime_env")"

  printf 'layout=%s\n' "$LAYOUT"
  printf 'runtime_env=%s\n' "$runtime_env"
  printf 'route_target=%s\n' "$ROUTE_TARGET"
  printf 'entry=%s\n' "$ENTRY"
  printf 'bin=%s\n' "$bin_path"
  printf 'args='
  printf '%q ' "${CLI_ARGS[@]}"
  printf '\n'
}

run_selected_entry() {
  local runtime_env bin_path
  runtime_env="$(selected_runtime_env)"
  bin_path="$(selected_bin_path "$runtime_env")"

  if [[ "$runtime_env" == "preprocess" ]]; then
    [[ -x "$bin_path" ]] || die "missing preprocessing command: $bin_path; rerun with --sync"
    if [[ "$PREPROCESS_CPU" -eq 1 ]]; then
      exec env CUDA_VISIBLE_DEVICES="" "$bin_path" "${CLI_ARGS[@]}"
    fi
    exec "$bin_path" "${CLI_ARGS[@]}"
  fi

  require_uv
  require_dir "$MODEL_ENV_DIR"
  cd "$MODEL_ENV_DIR"
  exec uv run --no-sync "$ENTRY" "${CLI_ARGS[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layout)
      LAYOUT="${2:?}"
      shift 2
      ;;
    --model-env)
      MODEL_ENV_DIR="$(abs_path "${2:?}")"
      shift 2
      ;;
    --preprocess-project)
      PREPROCESS_PROJECT_DIR="$(abs_path "${2:?}")"
      shift 2
      ;;
    --preprocess-venv)
      PREPROCESS_VENV_DIR="$(abs_path "${2:?}")"
      shift 2
      ;;
    --sync)
      SYNC_MODE="always"
      shift
      ;;
    --no-sync)
      SYNC_MODE="never"
      shift
      ;;
    --check-env)
      CHECK_ENV=1
      shift
      ;;
    --no-check-env)
      CHECK_ENV=0
      shift
      ;;
    --preprocess-cpu)
      PREPROCESS_CPU=1
      shift
      ;;
    --model-sync-arg)
      MODEL_SYNC_ARGS+=("${2:?}")
      shift 2
      ;;
    --preprocess-install-arg)
      PREPROCESS_INSTALL_ARGS+=("${2:?}")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      SYNC_MODE="never"
      CHECK_ENV=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      [[ $# -gt 0 ]] || die "missing command after --"
      COMMAND="$1"
      shift
      CLI_ARGS=("$@")
      break
      ;;
    --*)
      die "unknown router option: $1"
      ;;
    *)
      COMMAND="$1"
      shift
      CLI_ARGS=("$@")
      break
      ;;
  esac
done

if [[ -z "$COMMAND" ]]; then
  usage
  exit 0
fi

ENTRY="$(resolve_entry "$COMMAND")"
ROUTE_TARGET="$(resolve_route_target "$ENTRY")"

resolve_layout

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_route
  exit 0
fi

sync_if_needed
check_selected_env
run_selected_entry
