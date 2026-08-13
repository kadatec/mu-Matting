#!/usr/bin/env bash
set -euo pipefail

GPUS=${1:?Usage: train.sh GPUS WORK_DIR}
WORK_DIR=${2:?Usage: train.sh GPUS WORK_DIR}

if [[ ! "${GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS must be a positive integer" >&2
    exit 2
fi

: "${SAPIENS_ROOT:?Set SAPIENS_ROOT to the official Sapiens checkout}"
: "${SAPIENS_PRETRAINED_CHECKPOINT:?Set SAPIENS_PRETRAINED_CHECKPOINT}"
SAPIENS_ROOT=$(cd -- "${SAPIENS_ROOT}" && pwd)
export SAPIENS_ROOT

EXPECTED_CHECKPOINT=sapiens_0.3b_epoch_1600_clean.pth
if [[ "${SAPIENS_PRETRAINED_CHECKPOINT##*/}" != "${EXPECTED_CHECKPOINT}" ]]; then
    echo "Expected backbone checkpoint: ${EXPECTED_CHECKPOINT}" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CONFIG="${PROJECT_ROOT}/configs/sapiens_0.3b_alpha_train.py"
UPSTREAM_TRAIN="${SAPIENS_ROOT}/seg/tools/train.py"

if [[ ! -f "${UPSTREAM_TRAIN}" ]]; then
    echo "Official training entry not found: ${UPSTREAM_TRAIN}" >&2
    exit 2
fi

UPSTREAM_PATHS="${SAPIENS_ROOT}/seg:${SAPIENS_ROOT}/pretrain"
UPSTREAM_PATHS="${UPSTREAM_PATHS}:${SAPIENS_ROOT}/engine:${SAPIENS_ROOT}/cv"
export PYTHONPATH="${PROJECT_ROOT}/src:${UPSTREAM_PATHS}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
exec torchrun \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT:-29500}" \
    "${UPSTREAM_TRAIN}" \
    "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch
