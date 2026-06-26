#!/usr/bin/env bash
set -xeuo pipefail

# VERL v1 synchronous bring-up for uni-agent.
#
# Intended validation path:
# dataset -> AgentLoopManagerTQ -> UniAgentLoop -> TransferQueue
#         -> reward_score from UniAgentLoop -> GRPO -> actor update
#
# Run this script before trying colocate_async. It deliberately uses a small
# rollout.n and small agent concurrency so failures are easy to inspect.

project_name=${PROJECT_NAME:-'verl-uni-agent'}
model_name=${MODEL_NAME:-'Qwen3-4B-Instruct-2507'}
exp_name=${EXP_NAME:-'V1-Qwen3-4B-Sync'}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}"}
MODEL_PATH=${MODEL_PATH:-"/file_system/common-models/Qwen/${model_name}"}
CKPTS_DIR=${CKPTS_DIR:-"/file_system/dhl/save_ckpt/uni_agent/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_vefaas.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-"${RAY_DATA_HOME}/code/uni-agent/examples/agent_interaction/runtime_env.yaml"}
AGENT_CONFIG_SOURCE=${AGENT_CONFIG_SOURCE:-"${RAY_DATA_HOME}/code/uni-agent/examples/agent_interaction/agent_config_vefaas.yaml"}
AGENT_CONFIG_TMPDIR=${AGENT_CONFIG_TMPDIR:-"${RAY_DATA_HOME}/code/uni-agent/examples/agent_interaction"}

# UniAgentLoop reads concurrency from its own YAML rather than Hydra. Generate a
# temporary copy so this v1 smoke test does not launch hundreds of sandboxes.
AGENT_CONCURRENCY=${AGENT_CONCURRENCY:-16}
AGENT_CONFIG_LOCAL_PATH=$(mktemp "${AGENT_CONFIG_TMPDIR%/}/uni-agent-v1-sync-agent.XXXXXX.yaml")
AGENT_CONFIG_PATH="examples/agent_interaction/$(basename "${AGENT_CONFIG_LOCAL_PATH}")"
cleanup() {
    rm -f "${AGENT_CONFIG_LOCAL_PATH}"
}
trap cleanup EXIT
sed -E \
    "s/^([[:space:]]*)concurrency:[[:space:]]*[0-9]+/\1concurrency: ${AGENT_CONCURRENCY}/" \
    "${AGENT_CONFIG_SOURCE}" > "${AGENT_CONFIG_LOCAL_PATH}"

# Fail before submitting a Ray job if TransferQueue is installed into a
# different Python environment.
python3 -c "import transfer_queue; print('TransferQueue:', transfer_queue.__file__)"

rollout_name="vllm"
rollout_mode="async"
adv_estimator="grpo"

use_kl_in_reward=False
use_kl_loss=False

max_prompt_length=$((1024 * 4))
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 64))}

temperature=1.0
top_p=1.0
top_k=-1
val_temperature=1.0
val_top_p=0.95
val_top_k=-1

use_dynamic_bsz=True
offload=True
gen_tp=${GEN_TP:-4} 
gen_dp=${GEN_DP:-1}
gen_pp=${GEN_PP:-1}
train_tp=${TRAIN_TP:-4}
train_pp=${TRAIN_PP:-1}
train_cp=${TRAIN_CP:-1}
train_ep=${TRAIN_EP:-8}
train_etp=${TRAIN_ETP:-1}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

optimizer_offload_fraction=1.0
USE_MBRIDGE=True
USE_DIST_CKPT=False

NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Small v1 bring-up batch. One prompt produces rollout.n trajectories.
train_prompt_bsz=2
n_resp_per_prompt=1
train_prompt_mini_bsz=2
total_training_steps=10

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

echo "AGENT_CONFIG_LOCAL_PATH=${AGENT_CONFIG_LOCAL_PATH}"
echo "AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH}"
test -f "${AGENT_CONFIG_LOCAL_PATH}"

# v1 architecture switches below:
# - trainer.use_v1 selects TaskRunnerV1 and AgentLoopManagerTQ.
# - TransferQueue stores and schedules trajectory fields between stages.
# - sync waits for a complete sampled batch before actor update.
# - UniAgentLoop supplies reward_score, so the colocated reward model is disabled.
# - max_off_policy_strategy=drop controls staleness only, not failed groups.
ray job submit --runtime-env "${RUNTIME_ENV}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    model_engine=megatron \
    \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=sync \
    \
    transfer_queue.enable=True \
    transfer_queue.backend.storage_backend=SimpleStorage \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
    transfer_queue.backend.SimpleStorage.total_storage_size=2048 \
    \
    trainer.v1.sampler.max_off_policy_threshold=8 \
    trainer.v1.sampler.max_off_policy_strategy=drop \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG_PATH}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=swe_agent \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=4e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
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
    \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp} \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=${gen_pp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.hybrid_engine=True \
    \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=False \
    \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes=${NNODES_TRAIN} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
