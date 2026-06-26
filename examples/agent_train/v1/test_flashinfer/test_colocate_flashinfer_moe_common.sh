#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

: "${FLASHINFER_MOE_FP16:?Set FLASHINFER_MOE_FP16 to 0 or 1}"
if [[ "${FLASHINFER_MOE_FP16}" != "0" && "${FLASHINFER_MOE_FP16}" != "1" ]]; then
    echo "FLASHINFER_MOE_FP16 must be 0 or 1" >&2
    exit 1
fi

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}"}
SOURCE_TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_vefaas.parquet"}
RUNTIME_ENV_SOURCE=${RUNTIME_ENV_SOURCE:-"${REPO_ROOT}/examples/agent_interaction/runtime_env.yaml"}
TEST_DATA_FILE=${FLASHINFER_TEST_DATA_FILE:-"/file_system/dhl/tmp/uni_agent_flashinfer_moe_ab.parquet"}
TEST_SAMPLES=${FLASHINFER_TEST_SAMPLES:-8}

if ! [[ "${TEST_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FLASHINFER_TEST_SAMPLES must be a positive integer" >&2
    exit 1
fi

export TEST_SAMPLES
if [[ "${FLASHINFER_MOE_FP16}" == "1" ]]; then
    mode=on
else
    mode=off
fi

export PROJECT_NAME=${PROJECT_NAME:-verl-uni-agent}
export EXP_NAME=${EXP_NAME:-test-colocate-flashinfer-moe-${mode}}
export MODEL_NAME=${MODEL_NAME:-Qwen3-Coder-30B-A3B-Instruct}

export NNODES_TRAIN=${NNODES_TRAIN:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export TRAIN_TP=${TRAIN_TP:-4}
export TRAIN_CP=${TRAIN_CP:-2}
export TRAIN_PP=${TRAIN_PP:-2}
export TRAIN_EP=${TRAIN_EP:-8}
export TRAIN_ETP=${TRAIN_ETP:-1}
export GEN_TP=${GEN_TP:-8}
export GEN_DP=${GEN_DP:-1}
export GEN_PP=${GEN_PP:-1}

export AGENT_CONCURRENCY=${AGENT_CONCURRENCY:-${TEST_SAMPLES}}
export AGENT_CONFIG_TMPDIR="${REPO_ROOT}/examples/agent_interaction"
export AGENT_CONFIG_REPO_ROOT="${REPO_ROOT}"

export TRAIN_FILE="${TEST_DATA_FILE}"
export TEST_FILE="${TEST_DATA_FILE}"
export RUNTIME_ENV="${SCRIPT_DIR}/test_runtime_env_flashinfer_moe_${mode}.yaml"
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"/file_system/dhl/tmp/${EXP_NAME}-rollout"}

generated_launcher=$(mktemp "${SCRIPT_DIR}/test_colocate_flashinfer_moe_${mode}.XXXXXX.sh")
cleanup() {
    rm -f "${generated_launcher}" "${RUNTIME_ENV}"
    if [[ -z "${FLASHINFER_TEST_DATA_FILE:-}" ]]; then
        rm -f "${TEST_DATA_FILE}"
    fi
}
trap cleanup EXIT

cd "${REPO_ROOT}"

SOURCE_TEST_FILE="${SOURCE_TEST_FILE}" TEST_DATA_FILE="${TEST_DATA_FILE}" \
TEST_SAMPLES="${TEST_SAMPLES}" python3 -c '
import os
from datasets import load_dataset

source = os.path.expanduser(os.environ["SOURCE_TEST_FILE"])
output = os.environ["TEST_DATA_FILE"]
sample_count = int(os.environ["TEST_SAMPLES"])
dataset = load_dataset("parquet", data_files=source, split="train")
if len(dataset) < sample_count:
    raise ValueError(
        f"Need at least {sample_count} samples in {source}, found {len(dataset)}"
    )
dataset.select(range(sample_count)).to_parquet(output)
print(f"FlashInfer MoE A/B dataset: {output} ({sample_count} fixed samples)")
'

sed -E \
    "s/^([[:space:]]*)VLLM_USE_FLASHINFER_MOE_FP16:.*/\1VLLM_USE_FLASHINFER_MOE_FP16: \"${FLASHINFER_MOE_FP16}\"/" \
    "${RUNTIME_ENV_SOURCE}" > "${RUNTIME_ENV}"

# Derive a one-step launcher from the production colocate script. Stop before
# its trailing experimental backend overrides so this A/B changes only the
# FlashInfer MoE environment variable.
awk '
    /^train_prompt_bsz=/ { print "train_prompt_bsz=${TEST_SAMPLES}"; next }
    /^n_resp_per_prompt=/ { print "n_resp_per_prompt=1"; next }
    /^train_prompt_mini_bsz=/ { print "train_prompt_mini_bsz=${TEST_SAMPLES}"; next }
    /^total_training_steps=/ { print "total_training_steps=1"; next }
    /^test_freq=/ { print "test_freq=-1"; next }
    /trainer.logger=/ { print "    trainer.logger=[\047console\047] \\"; next }
    /trainer.log_val_generations=/ {
        print
        print "    trainer.rollout_data_dir=\"${ROLLOUT_DATA_DIR}\" \\"
        next
    }
    /actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=/ {
        print "    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}"
        exit
    }
    { print }
' "${SCRIPT_DIR}/single_node_v1_colocate_async.sh" > "${generated_launcher}"
chmod +x "${generated_launcher}"

echo "FlashInfer MoE FP16: ${FLASHINFER_MOE_FP16}"
echo "Runtime env: ${RUNTIME_ENV}"
echo "Rollout dump: ${ROLLOUT_DATA_DIR}"

"${generated_launcher}"
