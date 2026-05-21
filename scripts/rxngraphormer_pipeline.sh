#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ENV_DIR="$ROOT_DIR"
PREPROCESS_PROJECT_DIR="$ROOT_DIR/envs/preprocess"
PREPROCESS_VENV_DIR="$PREPROCESS_PROJECT_DIR/.venv"

LAYOUT="auto"
STAGES="sync"
SYNC_ENVS="all"
SEQUENCE=0
CHECK_ENV=1
PREPROCESS_CPU=0

PREPROCESS_CONFIG=""
TRAIN_CONFIG=""
EVAL_CONFIG=""

PREPROCESS_ARGS=()
TRAIN_ARGS=()
EVAL_ARGS=()
MODEL_SYNC_ARGS=()
PREPROCESS_INSTALL_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/rxngraphormer_pipeline.sh [options]

Default behavior:
  Auto-detect the hardware and choose a single end-to-end environment when
  possible. If the hardware can support it, the root environment is used with
  rxngraphormer[all]. Otherwise the root environment stays model-only and a
  separate preprocessing environment is created under envs/preprocess/.venv
  with rxngraphormer[preprocess].

Options:
  --layout NAME              Environment layout: auto,single,split
                             Default: auto
  --stages LIST              Comma list: sync,preprocess,train,eval
                             Default: sync
  --env NAME                 Environments to sync: all,model,preprocess,none
                             Default: all
  --model-env DIR            Main model environment directory
                             Default: repository root
  --preprocess-project DIR    Preprocessing project directory
                             Default: envs/preprocess
  --preprocess-venv DIR      Virtualenv used by split preprocessing
                             Default: envs/preprocess/.venv
  --preprocess-cpu           Hide CUDA devices during preprocessing
  --no-check-env             Skip compatibility checks after sync
  --sequence                 Include the sequence extra for model sync/run
  --model-sync-arg ARG       Extra argument passed to uv sync for the model env
  --preprocess-install-arg ARG
                             Extra argument passed to uv pip install in split preprocessing
  --skip-sync                Do not sync environments before running stages
  --preprocess-config PATH   Config used by rxngraphormer-preprocess
  --train-config PATH        Config used by rxngraphormer-train
  --eval-config PATH         Config used by rxngraphormer-eval
  --preprocess-arg ARG       Extra argument passed to rxngraphormer-preprocess
  --train-arg ARG            Extra argument passed to rxngraphormer-train
  --eval-arg ARG             Extra argument passed to rxngraphormer-eval
  -h, --help                 Show this help

Examples:
  scripts/rxngraphormer_pipeline.sh
  scripts/rxngraphormer_pipeline.sh --layout single --stages preprocess,train,eval
  scripts/rxngraphormer_pipeline.sh --layout split --stages preprocess,train,eval
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

has_stage() {
  local needle="$1"
  case ",$STAGES," in
    *",$needle,"*) return 0 ;;
    *) return 1 ;;
  esac
}

env_selected() {
  local needle="$1"
  case "$SYNC_ENVS" in
    all) return 0 ;;
    "$needle") return 0 ;;
    none) return 1 ;;
    model|preprocess) return 1 ;;
    *) die "--env must be one of: all, model, preprocess, none" ;;
  esac
}

require_file() {
  local path="$1"
  [[ -n "$path" ]] || die "required config path is empty"
  [[ -f "$path" ]] || die "file not found: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "directory not found: $path"
}

validate_stages() {
  local old_ifs="$IFS"
  IFS=","
  read -ra stage_list <<< "$STAGES"
  IFS="$old_ifs"

  local stage
  for stage in "${stage_list[@]}"; do
    case "$stage" in
      sync|preprocess|train|eval) ;;
      *) die "--stages contains unsupported stage: $stage" ;;
    esac
  done
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

uv_in_project() {
  local project_dir="$1"
  shift
  (cd "$project_dir" && uv "$@")
}

sync_model_env() {
  require_dir "$MODEL_ENV_DIR"
  local cmd=(sync)
  cmd+=("${MODEL_SYNC_ARGS[@]}")
  echo "==> Sync model environment: $MODEL_ENV_DIR"
  uv_in_project "$MODEL_ENV_DIR" "${cmd[@]}"

  if [[ "$LAYOUT" == "single" ]]; then
    local install_cmd=(pip install -e "$ROOT_DIR[all]")
    echo "==> Install rxngraphormer[all] into single end-to-end environment"
    uv_in_project "$MODEL_ENV_DIR" "${install_cmd[@]}"
    return
  fi

  if [[ "$SEQUENCE" -eq 1 ]]; then
    echo "==> Install sequence extra into model environment"
    uv_in_project "$MODEL_ENV_DIR" pip install -e "$ROOT_DIR[sequence]"
  fi
}

sync_preprocess_env() {
  if [[ "$LAYOUT" == "single" ]]; then
    return
  fi
  require_dir "$PREPROCESS_PROJECT_DIR"
  echo "==> Create preprocessing virtualenv: $PREPROCESS_VENV_DIR"
  uv venv "$PREPROCESS_VENV_DIR" --allow-existing
  local install_cmd=(pip install --python "$PREPROCESS_VENV_DIR/bin/python" -e "$ROOT_DIR[preprocess]")
  install_cmd+=("${PREPROCESS_INSTALL_ARGS[@]}")
  echo "==> Install preprocessing extras"
  uv "${install_cmd[@]}"
}

check_model_env() {
  [[ "$CHECK_ENV" -eq 1 ]] || return
  echo "==> Check model environment"
  uv_in_project "$MODEL_ENV_DIR" run --no-sync python scripts/reproduce/check_environment.py model
}

check_preprocess_env() {
  [[ "$CHECK_ENV" -eq 1 ]] || return
  echo "==> Check preprocessing environment"
  if [[ "$LAYOUT" == "single" ]]; then
    uv_in_project "$MODEL_ENV_DIR" run --no-sync python scripts/reproduce/check_environment.py preprocess
  else
    if [[ "$PREPROCESS_CPU" -eq 1 ]]; then
      CUDA_VISIBLE_DEVICES="" "$PREPROCESS_VENV_DIR/bin/python" "$ROOT_DIR/scripts/reproduce/check_environment.py" preprocess
    else
      "$PREPROCESS_VENV_DIR/bin/python" "$ROOT_DIR/scripts/reproduce/check_environment.py" preprocess
    fi
  fi
}

run_preprocess() {
  echo "==> Preprocess data"
  [[ -n "$PREPROCESS_CONFIG" ]] || die "--preprocess-config is required for the preprocess stage"
  require_file "$PREPROCESS_CONFIG"
  if [[ "$LAYOUT" == "single" ]]; then
    uv_in_project "$MODEL_ENV_DIR" run --no-sync rxngraphormer-preprocess --config_json "$PREPROCESS_CONFIG" "${PREPROCESS_ARGS[@]}"
  else
    if [[ "$PREPROCESS_CPU" -eq 1 ]]; then
      CUDA_VISIBLE_DEVICES="" "$PREPROCESS_VENV_DIR/bin/rxngraphormer-preprocess" --config_json "$PREPROCESS_CONFIG" "${PREPROCESS_ARGS[@]}"
    else
      "$PREPROCESS_VENV_DIR/bin/rxngraphormer-preprocess" --config_json "$PREPROCESS_CONFIG" "${PREPROCESS_ARGS[@]}"
    fi
  fi
}

run_train() {
  echo "==> Train model"
  [[ -n "$TRAIN_CONFIG" ]] || die "--train-config is required for the train stage"
  require_file "$TRAIN_CONFIG"
  local cmd=(run)
  if [[ "$LAYOUT" == "single" ]]; then
    cmd+=(--no-sync)
  fi
  if [[ "$SEQUENCE" -eq 1 ]]; then
    cmd+=(--extra sequence)
  fi
  uv_in_project "$MODEL_ENV_DIR" "${cmd[@]}" rxngraphormer-train --config "$TRAIN_CONFIG" "${TRAIN_ARGS[@]}"
}

run_eval() {
  echo "==> Evaluate / run inference"
  [[ -n "$EVAL_CONFIG" ]] || die "--eval-config is required for the eval stage"
  require_file "$EVAL_CONFIG"
  local cmd=(run)
  if [[ "$LAYOUT" == "single" ]]; then
    cmd+=(--no-sync)
  fi
  if [[ "$SEQUENCE" -eq 1 ]]; then
    cmd+=(--extra sequence)
  fi
  uv_in_project "$MODEL_ENV_DIR" "${cmd[@]}" rxngraphormer-eval --config_json "$EVAL_CONFIG" "${EVAL_ARGS[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layout)
      LAYOUT="${2:?}"
      shift 2
      ;;
    --stages)
      STAGES="${2:?}"
      shift 2
      ;;
    --env)
      SYNC_ENVS="${2:?}"
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
    --preprocess-cpu)
      PREPROCESS_CPU=1
      shift
      ;;
    --no-check-env)
      CHECK_ENV=0
      shift
      ;;
    --sequence)
      SEQUENCE=1
      shift
      ;;
    --preprocess-config)
      PREPROCESS_CONFIG="$(abs_path "${2:?}")"
      shift 2
      ;;
    --train-config)
      TRAIN_CONFIG="$(abs_path "${2:?}")"
      shift 2
      ;;
    --eval-config)
      EVAL_CONFIG="$(abs_path "${2:?}")"
      shift 2
      ;;
    --model-sync-arg)
      MODEL_SYNC_ARGS+=("${2:?}")
      shift 2
      ;;
    --preprocess-install-arg)
      PREPROCESS_INSTALL_ARGS+=("${2:?}")
      shift 2
      ;;
    --skip-sync)
      SYNC_ENVS="none"
      shift
      ;;
    --preprocess-arg)
      PREPROCESS_ARGS+=("${2:?}")
      shift 2
      ;;
    --train-arg)
      TRAIN_ARGS+=("${2:?}")
      shift 2
      ;;
    --eval-arg)
      EVAL_ARGS+=("${2:?}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

command -v uv >/dev/null 2>&1 || die "uv is required but was not found on PATH"
require_dir "$MODEL_ENV_DIR"
resolve_layout
validate_stages

if has_stage sync || [[ "$SYNC_ENVS" != "none" ]]; then
  if [[ "$LAYOUT" == "single" ]]; then
    if env_selected model || env_selected preprocess; then
      sync_model_env
      check_model_env
      check_preprocess_env
    fi
  else
    if env_selected model; then
      sync_model_env
      check_model_env
    fi
    if env_selected preprocess; then
      sync_preprocess_env
      check_preprocess_env
    fi
  fi
fi

if has_stage preprocess; then
  run_preprocess
fi

if has_stage train; then
  run_train
fi

if has_stage eval; then
  run_eval
fi
