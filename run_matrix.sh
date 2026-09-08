#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_matrix.sh <matrix.env> [--dry-run] [--force]

The matrix configuration selects a base experiment configuration, models,
budgets, and optional model-specific batch sizes.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_PATH="$1"
shift
DRY_RUN=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [[ ! -f "${MATRIX_PATH}" ]]; then
  if [[ -f "${SCRIPT_DIR}/${MATRIX_PATH}" ]]; then
    MATRIX_PATH="${SCRIPT_DIR}/${MATRIX_PATH}"
  else
    echo "Matrix configuration not found: ${MATRIX_PATH}" >&2
    exit 2
  fi
fi
MATRIX_PATH="$(realpath "${MATRIX_PATH}")"
# shellcheck disable=SC1090
source "${MATRIX_PATH}"

required=(BASE_CONFIG MODELS BUDGET_TYPE BUDGETS)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required matrix variable: ${variable}" >&2
    exit 2
  fi
done
case "${BUDGET_TYPE}" in
  target_relus) budget_option=--target-relus ;;
  keep_ratio) budget_option=--keep-ratio ;;
  *) echo "BUDGET_TYPE must be target_relus or keep_ratio" >&2; exit 2 ;;
esac

if [[ "${BASE_CONFIG}" != /* ]]; then
  BASE_CONFIG="${SCRIPT_DIR}/${BASE_CONFIG}"
fi
STAGE="${STAGE:-all}"
DATA_ROOT_OVERRIDE="${DATA_ROOT_OVERRIDE:-}"
WORKERS_OVERRIDE="${WORKERS_OVERRIDE:-}"
MODEL_BATCH_SIZES="${MODEL_BATCH_SIZES:-}"

batch_for_model() {
  local wanted="$1"
  local entry
  for entry in ${MODEL_BATCH_SIZES}; do
    if [[ "${entry%%:*}" == "${wanted}" ]]; then
      printf '%s' "${entry#*:}"
      return
    fi
  done
}

echo "ReLUPruner experiment matrix"
echo "  matrix:  ${MATRIX_PATH}"
echo "  models:  ${MODELS}"
echo "  budgets: ${BUDGETS} (${BUDGET_TYPE})"
echo "  stage:   ${STAGE}"

for model in ${MODELS}; do
  batch_size="$(batch_for_model "${model}")"
  for budget in ${BUDGETS}; do
    command=(
      bash "${SCRIPT_DIR}/run_experiment.sh" "${BASE_CONFIG}" "${STAGE}"
      --model "${model}" "${budget_option}" "${budget}"
    )
    [[ -z "${batch_size}" ]] || command+=(--batch-size "${batch_size}")
    [[ -z "${DATA_ROOT_OVERRIDE}" ]] || command+=(--data-root "${DATA_ROOT_OVERRIDE}")
    [[ -z "${WORKERS_OVERRIDE}" ]] || command+=(--workers "${WORKERS_OVERRIDE}")
    [[ "${FORCE}" == "0" ]] || command+=(--force)

    printf '\n[%s / %s] ' "${model}" "${budget}"
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == "0" ]]; then
      "${command[@]}"
    fi
  done
done
