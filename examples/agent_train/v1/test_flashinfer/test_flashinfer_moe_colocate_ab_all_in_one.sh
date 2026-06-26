#!/usr/bin/env bash
set -xeuo pipefail

MODE=${1:-}
if [[ "${MODE}" != "off" && "${MODE}" != "on" ]]; then
    echo "Usage: $0 <off|on>" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}"}
SOURCE_TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_vefaas.parquet"}
MODEL_PATH=${MODEL_PATH:-"/file_system/common-models/Qwen/Qwen3-Coder-30B-A3B-Instruct"}
RUNTIME_ENV_SOURCE=${RUNTIME_ENV_SOURCE:-"${REPO_ROOT}/examples/agent_interaction/runtime_env.yaml"}
AGENT_CONFIG_SOURCE=${AGENT_CONFIG_SOURCE:-"${REPO_ROOT}/examples/agent_interaction/agent_config_vefaas.yaml"}
TEST_SAMPLES=${FLASHINFER_TEST_SAMPLES:-8}

SHARED_ROOT=${FLASHINFER_TEST_SHARED_ROOT:-"/file_system/dhl/tmp"}
TEST_DATA_FILE=${FLASHINFER_TEST_DATA_FILE:-"${SHARED_ROOT}/uni_agent_flashinfer_moe_ab.parquet"}
EXP_NAME=${EXP_NAME:-"test-colocate-flashinfer-moe-${MODE}"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${SHARED_ROOT}/${EXP_NAME}-rollout"}
CKPTS_DIR=${CKPTS_DIR:-"/file_system/dhl/save_ckpt/uni_agent/ckpts/verl-uni-agent/${EXP_NAME}"}

NNODES_TRAIN=${NNODES_TRAIN:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_TP=${TRAIN_TP:-4}
TRAIN_CP=${TRAIN_CP:-2}
TRAIN_PP=${TRAIN_PP:-2}
TRAIN_EP=${TRAIN_EP:-8}
TRAIN_ETP=${TRAIN_ETP:-1}
GEN_TP=${GEN_TP:-8}
GEN_DP=${GEN_DP:-1}
GEN_PP=${GEN_PP:-1}
AGENT_CONCURRENCY=${AGENT_CONCURRENCY:-${TEST_SAMPLES}}

if [[ "${MODE}" == "on" ]]; then
    FLASHINFER_MOE_FP16=1
else
    FLASHINFER_MOE_FP16=0
fi

RUNTIME_ENV=$(mktemp "${SCRIPT_DIR}/test_runtime_env_flashinfer_moe_${MODE}.XXXXXX.yaml")
AGENT_CONFIG=$(mktemp "${REPO_ROOT}/examples/agent_interaction/test_agent_flashinfer_moe_${MODE}.XXXXXX.yaml")
AGENT_CONFIG_PATH="examples/agent_interaction/$(basename "${AGENT_CONFIG}")"
cleanup() {
    rm -f "${RUNTIME_ENV}" "${AGENT_CONFIG}"
}
trap cleanup EXIT

SOURCE_TEST_FILE="${SOURCE_TEST_FILE}" TEST_DATA_FILE="${TEST_DATA_FILE}" \
TEST_SAMPLES="${TEST_SAMPLES}" python3 -c '
import os
from datasets import load_dataset

source = os.path.expanduser(os.environ["SOURCE_TEST_FILE"])
output = os.environ["TEST_DATA_FILE"]
count = int(os.environ["TEST_SAMPLES"])
dataset = load_dataset("parquet", data_files=source, split="train")
if len(dataset) < count:
    raise ValueError(f"Need {count} samples in {source}, found {len(dataset)}")
dataset.select(range(count)).to_parquet(output)
print(f"Wrote {count} fixed samples to {output}")
'

sed -E \
    "s/^([[:space:]]*)VLLM_USE_FLASHINFER_MOE_FP16:.*/\1VLLM_USE_FLASHINFER_MOE_FP16: \"${FLASHINFER_MOE_FP16}\"/" \
    "${RUNTIME_ENV_SOURCE}" > "${RUNTIME_ENV}"

sed -E \
    "s/^([[:space:]]*)concurrency:[[:space:]]*[0-9]+/\1concurrency: ${AGENT_CONCURRENCY}/" \
    "${AGENT_CONFIG_SOURCE}" > "${AGENT_CONFIG}"

rm -rf "${ROLLOUT_DATA_DIR}"

max_prompt_length=4096
max_response_length=65536
max_model_length=$((max_prompt_length + max_response_length))
ppo_max_token_len=$((max_model_length / TRAIN_CP))

echo "MODE=${MODE}"
echo "VLLM_USE_FLASHINFER_MOE_FP16=${FLASHINFER_MOE_FP16}"
echo "TEST_DATA_FILE=${TEST_DATA_FILE}"
echo "ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR}"

ray job submit --runtime-env "${RUNTIME_ENV}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    model_engine=megatron \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=0 \
    transfer_queue.enable=True \
    transfer_queue.backend.storage_backend=SimpleStorage \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
    transfer_queue.backend.SimpleStorage.total_storage_size=512 \
    trainer.v1.sampler.max_off_policy_threshold=2 \
    trainer.v1.sampler.max_off_policy_strategy=drop \
    trainer.v1.sampler.custom_sampler.path=null \
    trainer.v1.sampler.custom_sampler.name=null \
    data.train_files="${TEST_DATA_FILE}" \
    data.val_files="${TEST_DATA_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${TEST_SAMPLES} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG_PATH}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=swe_agent \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.n=1 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=${max_model_length} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=4e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TEST_SAMPLES} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAIN_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${TRAIN_ETP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.data_parallel_size=${GEN_DP} \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_length} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${TRAIN_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${TRAIN_ETP} \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=False \
    trainer.logger=['console'] \
    trainer.project_name=verl-uni-agent \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes=${NNODES_TRAIN} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}

python3 -c '
import json
import pathlib
import statistics
import sys

path = pathlib.Path(sys.argv[1]) / "1.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
scores = [row["score"] for row in rows]
lengths = [len(row["output"]) for row in rows]
print({
    "samples": len(rows),
    "scores": scores,
    "mean_score": statistics.mean(scores),
    "empty_outputs": sum(not row["output"].strip() for row in rows),
    "output_chars": lengths,
})
' "${ROLLOUT_DATA_DIR}"
