#!/usr/bin/env bash
set -xeuo pipefail

# Multi-node launcher for the VERL v1 colocate-async trainer.
#
# This is colocated async, not separate async: train and rollout each map onto
# the same global GPU pool, so their logical world sizes are not added together.
# With two 8-GPU nodes and GEN_TP=4, rollout has four 4-GPU replicas sharing
# the same 16 GPUs used by Megatron training.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

export NNODES_TRAIN=${NNODES_TRAIN:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

export TRAIN_TP=${TRAIN_TP:-4}
export TRAIN_CP=${TRAIN_CP:-2}
export TRAIN_PP=${TRAIN_PP:-2}
export TRAIN_EP=${TRAIN_EP:-8}
export TRAIN_ETP=${TRAIN_ETP:-1}

# one whole dp replica per node, avoid inner_dp
# gen_pp must be 1
export GEN_TP=${GEN_TP:-8}
export GEN_DP=${GEN_DP:-1}
export GEN_PP=${GEN_PP:-1}

export PROJECT_NAME='verl-uni-agent'
export EXP_NAME='V1-Qwen3-30B-Multi-Node-Colocate-Async'
# export MODEL_NAME='Qwen3-4B-Instruct-2507'
export MODEL_NAME='Qwen3-Coder-30B-A3B-Instruct'

# Ray packages this generated YAML with working_dir, making the same path
# available to AgentLoop workers scheduled on every cluster node.
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

exec "${SCRIPT_DIR}/single_node_v1_colocate_async.sh"
