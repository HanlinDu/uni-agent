#!/usr/bin/env bash
set -xeuo pipefail

# Multi-node launcher for the VERL v1 synchronous colocated trainer.
#
# Train and rollout reuse the same global GPU pool. For example, with two
# 8-GPU nodes, TRAIN_TP=4 gives train DP=4, while GEN_TP=4 creates four
# 4-GPU rollout replicas over the same 16 physical GPUs.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

export NNODES_TRAIN=${NNODES_TRAIN:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

export TRAIN_TP=${TRAIN_TP:-4}
export TRAIN_PP=${TRAIN_PP:-1}
export TRAIN_CP=${TRAIN_CP:-4}
export TRAIN_EP=${TRAIN_EP:-8}
export TRAIN_ETP=${TRAIN_ETP:-1}

export GEN_TP=${GEN_TP:-4}
export GEN_DP=${GEN_DP:-4}
export GEN_PP=${GEN_PP:-1}

export PROJECT_NAME=${PROJECT_NAME:-'verl-uni-agent'}
export EXP_NAME=${EXP_NAME:-'V1-Qwen3-30B-Multi-Node-Sync'}
export MODEL_NAME='Qwen3-Coder-30B-A3B-Instruct'

# The generated YAML must be part of Ray's working_dir package so workers on
# every node can resolve it. The single-node launcher defaults this to /tmp,
# which is only suitable when all workers are local.
export AGENT_CONFIG_TMPDIR="${REPO_ROOT}/examples/agent_interaction"
export AGENT_CONFIG_REPO_ROOT="${REPO_ROOT}"

total_gpus=$((NNODES_TRAIN * NGPUS_PER_NODE))
train_model_parallel=$((TRAIN_TP * TRAIN_PP * TRAIN_CP))
rollout_replica_size=$((GEN_TP * GEN_DP * GEN_PP))

if ((total_gpus % train_model_parallel != 0)); then
    echo "Invalid train parallelism: total_gpus=${total_gpus} is not divisible by TRAIN_TP*TRAIN_PP*TRAIN_CP=${train_model_parallel}" >&2
    exit 1
fi

if ((total_gpus % rollout_replica_size != 0)); then
    echo "Invalid rollout parallelism: total_gpus=${total_gpus} is not divisible by GEN_TP*GEN_DP*GEN_PP=${rollout_replica_size}" >&2
    exit 1
fi

exec "${SCRIPT_DIR}/single_node_v1_sync.sh"
