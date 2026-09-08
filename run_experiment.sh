#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_experiment.sh <config.env> [stage] [options]

Stages:
  all (default), info, teacher, prune, finetune, evaluate

Options:
  --model NAME          Override MODEL from the configuration.
  --target-relus N      Use an exact, model-independent ReLU budget.
  --keep-ratio R        Use a fraction of this model's total ReLU positions.
  --data-root PATH      Override DATA_ROOT from the configuration.
  --batch-size N        Override all train/evaluation batch sizes.
  --workers N           Override data-loader worker count.
  --force               Rerun and overwrite completed stages.

Examples:
  bash run_experiment.sh configs/cifar10_resnet18.env info --model resnet34
  bash run_experiment.sh configs/cifar10_resnet18.env
  bash run_experiment.sh configs/cifar10_resnet18.env all --model resnet34 --target-relus 49152
  bash run_experiment.sh configs/cifar10_resnet18.env prune --keep-ratio 0.05
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$1"
shift

STAGE="all"
if [[ $# -gt 0 && "$1" != --* ]]; then
  STAGE="$1"
  shift
fi

MODEL_OVERRIDE=""
TARGET_RELUS_OVERRIDE=""
KEEP_RATIO_OVERRIDE=""
DATA_ROOT_OVERRIDE=""
BATCH_SIZE_OVERRIDE=""
WORKERS_OVERRIDE=""
FORCE_COMPLETED_STAGES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 2; }
      MODEL_OVERRIDE="$2"
      shift 2
      ;;
    --target-relus)
      [[ $# -ge 2 ]] || { echo "--target-relus requires a value" >&2; exit 2; }
      TARGET_RELUS_OVERRIDE="$2"
      shift 2
      ;;
    --keep-ratio)
      [[ $# -ge 2 ]] || { echo "--keep-ratio requires a value" >&2; exit 2; }
      KEEP_RATIO_OVERRIDE="$2"
      shift 2
      ;;
    --data-root)
      [[ $# -ge 2 ]] || { echo "--data-root requires a value" >&2; exit 2; }
      DATA_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "--batch-size requires a value" >&2; exit 2; }
      BATCH_SIZE_OVERRIDE="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "--workers requires a value" >&2; exit 2; }
      WORKERS_OVERRIDE="$2"
      shift 2
      ;;
    --force)
      FORCE_COMPLETED_STAGES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -n "${TARGET_RELUS_OVERRIDE}" && -n "${KEEP_RATIO_OVERRIDE}" ]]; then
  echo "Choose only one of --target-relus and --keep-ratio." >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  if [[ -f "${SCRIPT_DIR}/${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_PATH}"
  else
    echo "Configuration file not found: ${CONFIG_PATH}" >&2
    exit 2
  fi
fi
CONFIG_PATH="$(realpath "${CONFIG_PATH}")"

# Config files are trusted shell assignments so paths and numbers remain easy to edit.
# shellcheck disable=SC1090
source "${CONFIG_PATH}"

OVERRIDDEN=0
if [[ -n "${MODEL_OVERRIDE}" ]]; then
  MODEL="${MODEL_OVERRIDE}"
  # Model-specific checkpoints from the base config must never leak across architectures.
  TEACHER_CHECKPOINT=""
  FINAL_CHECKPOINT=""
  OVERRIDDEN=1
fi
if [[ -n "${TARGET_RELUS_OVERRIDE}" ]]; then
  TARGET_RELUS="${TARGET_RELUS_OVERRIDE}"
  KEEP_RATIO=""
  OVERRIDDEN=1
elif [[ -n "${KEEP_RATIO_OVERRIDE}" ]]; then
  KEEP_RATIO="${KEEP_RATIO_OVERRIDE}"
  TARGET_RELUS=""
  OVERRIDDEN=1
fi
if [[ -n "${DATA_ROOT_OVERRIDE}" ]]; then
  DATA_ROOT="${DATA_ROOT_OVERRIDE}"
fi
if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
  if [[ ! "${BATCH_SIZE_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--batch-size must be a positive integer: ${BATCH_SIZE_OVERRIDE}" >&2
    exit 2
  fi
  TEACHER_BATCH_SIZE="${BATCH_SIZE_OVERRIDE}"
  PRUNER_BATCH_SIZE="${BATCH_SIZE_OVERRIDE}"
  FINETUNE_BATCH_SIZE="${BATCH_SIZE_OVERRIDE}"
  EVAL_BATCH_SIZE="${BATCH_SIZE_OVERRIDE}"
fi
if [[ -n "${WORKERS_OVERRIDE}" ]]; then
  if [[ ! "${WORKERS_OVERRIDE}" =~ ^[0-9]+$ ]]; then
    echo "--workers must be a non-negative integer: ${WORKERS_OVERRIDE}" >&2
    exit 2
  fi
  WORKERS="${WORKERS_OVERRIDE}"
fi
if [[ "${FORCE_COMPLETED_STAGES}" == "1" ]]; then
  SKIP_COMPLETED_STAGES=0
fi

required=(
  PYTHON_BIN DATASET DATA_ROOT MODEL DEVICE OUTPUT_ROOT LOG_ROOT
  TEACHER_EPOCHS TEACHER_BATCH_SIZE TEACHER_LR
  PRUNER_EPOCHS PRUNER_BATCH_SIZE PRUNER_LR
  FINETUNE_EPOCHS FINETUNE_BATCH_SIZE FINETUNE_LR
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required configuration variable: ${variable}" >&2
    exit 2
  fi
done

TARGET_RELUS="${TARGET_RELUS:-}"
KEEP_RATIO="${KEEP_RATIO:-}"
if [[ -z "${TARGET_RELUS}" && -z "${KEEP_RATIO}" ]]; then
  echo "Set exactly one budget: TARGET_RELUS or KEEP_RATIO." >&2
  exit 2
fi
if [[ -n "${TARGET_RELUS}" && -n "${KEEP_RATIO}" ]]; then
  echo "TARGET_RELUS and KEEP_RATIO cannot both be set." >&2
  exit 2
fi
if [[ -n "${TARGET_RELUS}" && ! "${TARGET_RELUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TARGET_RELUS must be a positive integer: ${TARGET_RELUS}" >&2
  exit 2
fi

case "${STAGE}" in
  all|info|teacher|prune|finetune|evaluate) ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    usage
    exit 2
    ;;
esac

if [[ "${PYTHON_BIN}" == */* ]]; then
  if [[ "${PYTHON_BIN}" != /* ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/${PYTHON_BIN}"
  fi
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable is not available: ${PYTHON_BIN}" >&2
    exit 2
  fi
else
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Python executable is not available on PATH" >&2
    exit 2
  fi
fi
if [[ "${DATA_ROOT}" != /* ]]; then
  DATA_ROOT="${SCRIPT_DIR}/${DATA_ROOT}"
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATA_ROOT}" >&2
  exit 2
fi

WORKERS="${WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-0}"
SEED="${SEED:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SKIP_COMPLETED_STAGES="${SKIP_COMPLETED_STAGES:-1}"
INIT_FROM_TEACHER="${INIT_FROM_TEACHER:-1}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
LR_MILESTONES="${LR_MILESTONES:-150,180,210}"
LR_DECAY="${LR_DECAY:-0.1}"
KD_TEMPERATURE="${KD_TEMPERATURE:-4}"
PRUNER_GAMMA="${PRUNER_GAMMA:-0.5}"
PRUNER_ALPHA="${PRUNER_ALPHA:-0.5}"
PRUNER_WARMUP_EPOCHS="${PRUNER_WARMUP_EPOCHS:-5}"
PRUNER_TARGET_EPOCH_RATIO="${PRUNER_TARGET_EPOCH_RATIO:-0.6}"
PRUNER_UPDATE_INTERVAL="${PRUNER_UPDATE_INTERVAL:-100}"
FINETUNE_GAMMA="${FINETUNE_GAMMA:-0.5}"
FINETUNE_ALPHA="${FINETUNE_ALPHA:-0.5}"
FINETUNE_BETA="${FINETUNE_BETA:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
if [[ "${OVERRIDDEN}" == "1" ]]; then
  EXPERIMENT_NAME=""
fi
if [[ -n "${TARGET_RELUS}" ]]; then
  BUDGET_LABEL="relus_${TARGET_RELUS}"
else
  BUDGET_LABEL="ratio_${KEEP_RATIO}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL}_${BUDGET_LABEL}}"

cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

DATASET_OUTPUT="${OUTPUT_ROOT}/${DATASET}"
EXPERIMENT_OUTPUT="${DATASET_OUTPUT}/${EXPERIMENT_NAME}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DATASET_OUTPUT}/${MODEL}_teacher.pt}"
PRUNING_DIR="${EXPERIMENT_OUTPUT}/pruning"
PRUNED_CHECKPOINT="${PRUNING_DIR}/best_at_budget.pt"
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-${EXPERIMENT_OUTPUT}/final.pt}"
EXPERIMENT_LOG_DIR="${LOG_ROOT}/${DATASET}/${EXPERIMENT_NAME}"
TEACHER_LOG_DIR="${LOG_ROOT}/${DATASET}/teachers"

mkdir -p \
  "${DATASET_OUTPUT}" \
  "${EXPERIMENT_OUTPUT}" \
  "${EXPERIMENT_LOG_DIR}" \
  "${TEACHER_LOG_DIR}"
cp -- "${CONFIG_PATH}" "${EXPERIMENT_OUTPUT}/config.source.env"
{
  printf 'DATASET=%q\n' "${DATASET}"
  printf 'MODEL=%q\n' "${MODEL}"
  printf 'TARGET_RELUS=%q\n' "${TARGET_RELUS}"
  printf 'KEEP_RATIO=%q\n' "${KEEP_RATIO}"
  printf 'EXPERIMENT_NAME=%q\n' "${EXPERIMENT_NAME}"
  printf 'TEACHER_CHECKPOINT=%q\n' "${TEACHER_CHECKPOINT}"
  printf 'FINAL_CHECKPOINT=%q\n' "${FINAL_CHECKPOINT}"
  printf 'SEED=%q\n' "${SEED}"
  printf 'DATA_ROOT=%q\n' "${DATA_ROOT}"
  printf 'TEACHER_BATCH_SIZE=%q\n' "${TEACHER_BATCH_SIZE}"
  printf 'PRUNER_BATCH_SIZE=%q\n' "${PRUNER_BATCH_SIZE}"
  printf 'FINETUNE_BATCH_SIZE=%q\n' "${FINETUNE_BATCH_SIZE}"
  printf 'EVAL_BATCH_SIZE=%q\n' "${EVAL_BATCH_SIZE}"
  printf 'WORKERS=%q\n' "${WORKERS}"
} > "${EXPERIMENT_OUTPUT}/config.resolved.env"

common_args=(
  --model "${MODEL}"
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --workers "${WORKERS}"
  --seed "${SEED}"
)
if [[ "${PIN_MEMORY}" == "1" ]]; then
  common_args+=(--pin-memory)
fi

append_limit_args() {
  local train_limit="$1"
  local validation_limit="$2"
  LIMIT_ARGS=()
  if [[ -n "${train_limit}" ]]; then
    LIMIT_ARGS+=(--max-train-batches "${train_limit}")
  fi
  if [[ -n "${validation_limit}" ]]; then
    LIMIT_ARGS+=(--max-val-batches "${validation_limit}")
  fi
}

run_logged() {
  local log_file="$1"
  shift
  echo
  echo "[$(date '+%F %T')] Running: $*"
  "$@" 2>&1 | tee "${log_file}"
}

run_info() {
  "${PYTHON_BIN}" -u -m scripts.model_info \
    --dataset "${DATASET}" \
    --model "${MODEL}"
}

run_teacher() {
  if [[ "${SKIP_COMPLETED_STAGES}" == "1" && -f "${TEACHER_CHECKPOINT}" ]]; then
    echo "Teacher checkpoint already exists; skipping: ${TEACHER_CHECKPOINT}"
    return
  fi
  append_limit_args "${TEACHER_MAX_TRAIN_BATCHES:-}" "${MAX_VAL_BATCHES:-}"
  run_logged "${TEACHER_LOG_DIR}/${MODEL}.log" \
    "${PYTHON_BIN}" -u -m scripts.train_teacher \
    "${common_args[@]}" \
    --batch-size "${TEACHER_BATCH_SIZE}" \
    --epochs "${TEACHER_EPOCHS}" \
    --learning-rate "${TEACHER_LR}" \
    --momentum "${MOMENTUM}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --lr-milestones "${LR_MILESTONES}" \
    --lr-decay "${LR_DECAY}" \
    --output "${TEACHER_CHECKPOINT}" \
    "${LIMIT_ARGS[@]}"
}

run_pruner() {
  if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
    echo "Teacher checkpoint is required: ${TEACHER_CHECKPOINT}" >&2
    exit 3
  fi
  if [[ "${SKIP_COMPLETED_STAGES}" == "1" && -f "${PRUNED_CHECKPOINT}" ]]; then
    echo "Pruned checkpoint already exists; skipping: ${PRUNED_CHECKPOINT}"
    return
  fi
  append_limit_args "${PRUNER_MAX_TRAIN_BATCHES:-}" "${MAX_VAL_BATCHES:-}"
  init_args=()
  if [[ "${INIT_FROM_TEACHER}" == "1" ]]; then
    init_args+=(--init-from-teacher)
  fi
  budget_args=()
  if [[ -n "${TARGET_RELUS}" ]]; then
    budget_args+=(--target-relus "${TARGET_RELUS}")
  else
    budget_args+=(--global-keep-ratio "${KEEP_RATIO}")
  fi
  run_logged "${EXPERIMENT_LOG_DIR}/pruning.log" \
    "${PYTHON_BIN}" -u -m scripts.train_pruner \
    "${common_args[@]}" \
    --batch-size "${PRUNER_BATCH_SIZE}" \
    --epochs "${PRUNER_EPOCHS}" \
    --learning-rate "${PRUNER_LR}" \
    --momentum "${MOMENTUM}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --lr-milestones "${LR_MILESTONES}" \
    --lr-decay "${LR_DECAY}" \
    --teacher "${TEACHER_CHECKPOINT}" \
    "${budget_args[@]}" \
    --warmup-epochs "${PRUNER_WARMUP_EPOCHS}" \
    --target-epoch-ratio "${PRUNER_TARGET_EPOCH_RATIO}" \
    --update-interval "${PRUNER_UPDATE_INTERVAL}" \
    --gamma "${PRUNER_GAMMA}" \
    --alpha "${PRUNER_ALPHA}" \
    --temperature "${KD_TEMPERATURE}" \
    --output-dir "${PRUNING_DIR}" \
    "${init_args[@]}" \
    "${LIMIT_ARGS[@]}"
}

run_finetune() {
  if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
    echo "Teacher checkpoint is required: ${TEACHER_CHECKPOINT}" >&2
    exit 3
  fi
  if [[ ! -f "${PRUNED_CHECKPOINT}" ]]; then
    echo "At-budget pruned checkpoint is required: ${PRUNED_CHECKPOINT}" >&2
    exit 3
  fi
  if [[ "${SKIP_COMPLETED_STAGES}" == "1" && -f "${FINAL_CHECKPOINT}" ]]; then
    echo "Final checkpoint already exists; skipping: ${FINAL_CHECKPOINT}"
    return
  fi
  append_limit_args "${FINETUNE_MAX_TRAIN_BATCHES:-}" "${MAX_VAL_BATCHES:-}"
  run_logged "${EXPERIMENT_LOG_DIR}/finetune.log" \
    "${PYTHON_BIN}" -u -m scripts.finetune \
    "${common_args[@]}" \
    --batch-size "${FINETUNE_BATCH_SIZE}" \
    --epochs "${FINETUNE_EPOCHS}" \
    --learning-rate "${FINETUNE_LR}" \
    --momentum "${MOMENTUM}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --lr-milestones "${LR_MILESTONES}" \
    --lr-decay "${LR_DECAY}" \
    --teacher "${TEACHER_CHECKPOINT}" \
    --stage-one "${PRUNED_CHECKPOINT}" \
    --gamma "${FINETUNE_GAMMA}" \
    --alpha "${FINETUNE_ALPHA}" \
    --beta "${FINETUNE_BETA}" \
    --temperature "${KD_TEMPERATURE}" \
    --output "${FINAL_CHECKPOINT}" \
    "${LIMIT_ARGS[@]}"
}

run_evaluate() {
  if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
    echo "Final checkpoint is required: ${FINAL_CHECKPOINT}" >&2
    exit 3
  fi
  append_limit_args "" "${MAX_VAL_BATCHES:-}"
  run_logged "${EXPERIMENT_LOG_DIR}/evaluate.log" \
    "${PYTHON_BIN}" -u -m scripts.evaluate \
    "${common_args[@]}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --checkpoint "${FINAL_CHECKPOINT}" \
    "${LIMIT_ARGS[@]}"
}

echo "ReLUPruner experiment"
echo "  config:      ${CONFIG_PATH}"
echo "  stage:       ${STAGE}"
echo "  dataset:     ${DATASET}"
echo "  model:       ${MODEL}"
if [[ -n "${TARGET_RELUS}" ]]; then
  echo "  ReLU budget: ${TARGET_RELUS} (exact count)"
else
  echo "  keep ratio:  ${KEEP_RATIO} (model-relative fraction)"
fi
echo "  teacher:     ${TEACHER_CHECKPOINT}"
echo "  output:      ${EXPERIMENT_OUTPUT}"
echo "  python:      ${PYTHON_BIN}"

START_SECONDS=${SECONDS}
if [[ "${STAGE}" != "info" ]]; then
  # Validate the architecture/dataset pair before an expensive dataset scan.
  run_info >/dev/null
fi
case "${STAGE}" in
  info) run_info ;;
  teacher) run_teacher ;;
  prune) run_pruner ;;
  finetune) run_finetune ;;
  evaluate) run_evaluate ;;
  all)
    run_teacher
    run_pruner
    run_finetune
    run_evaluate
    ;;
esac

echo
echo "Completed stage '${STAGE}' in $((SECONDS - START_SECONDS)) seconds."
if [[ "${STAGE}" != "info" ]]; then
  echo "Final checkpoint: ${FINAL_CHECKPOINT}"
fi
