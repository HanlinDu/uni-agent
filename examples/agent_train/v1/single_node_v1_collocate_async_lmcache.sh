#!/usr/bin/env bash
set -xeuo pipefail

# LMCache validation variant for VERL v1 colocate-async uni-agent training.
#
# This script keeps the original single_node_v1_collocate_async.sh untouched.
# It starts a local LMCache MP server, enables vLLM's local prefix cache, and
# attaches LMCache as an external KV cache through vLLM's kv_transfer_config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMCACHE_ENV_FILE=${LMCACHE_ENV_FILE:-"${SCRIPT_DIR}/lmcache_mp.local.env"}

if [[ ! -f "${LMCACHE_ENV_FILE}" ]]; then
    echo "LMCache env file not found: ${LMCACHE_ENV_FILE}" >&2
    exit 1
fi

set -a
# shellcheck source=/dev/null
source "${LMCACHE_ENV_FILE}"
set +a

if [[ -n "${LMCACHE_FORCE_SKIP_SAVE:-}" ]]; then
    echo "LMCACHE_FORCE_SKIP_SAVE must be unset or empty; LMCache uses any non-empty value as skip-save." >&2
    exit 1
fi

python3 - <<'PY'
import importlib

for name in (
    "lmcache",
    "lmcache.integration.vllm.lmcache_mp_connector",
    "vllm",
    "transfer_queue",
):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__file__', '<builtin>')}")
PY

project_name=${PROJECT_NAME:-'verl-uni-agent'}
model_name=${MODEL_NAME:-'Qwen3-4B-Instruct-2507'}
exp_name=${EXP_NAME:-'V1-Qwen3-4B-Colocate-Async-LMCache'}
safe_exp_name=${exp_name//[^A-Za-z0-9_.-]/_}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}"}
MODEL_PATH=${MODEL_PATH:-"/file_system/common-models/Qwen/${model_name}"}
CKPTS_DIR=${CKPTS_DIR:-"/file_system/dhl/save_ckpt/uni_agent/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_vefaas.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-"${RAY_DATA_HOME}/code/uni-agent/examples/agent_interaction/runtime_env.yaml"}
AGENT_CONFIG_SOURCE=${AGENT_CONFIG_SOURCE:-"${RAY_DATA_HOME}/code/uni-agent/examples/agent_interaction/agent_config_vefaas.yaml"}
AGENT_CONFIG_TMPDIR=${AGENT_CONFIG_TMPDIR:-"${RAY_DATA_HOME}/code/uni-agent/examples/agent_interaction"}

RAY_PROFILING_PATH=${RAY_PROFILING_PATH:-""}
SANITIZE_VEFAAS_ROUTE=${SANITIZE_VEFAAS_ROUTE:-True}

# UniAgentLoop's sandbox concurrency belongs to its YAML, not the VERL Hydra
# tree. Keep it intentionally small while validating abort/resume behavior.
AGENT_CONCURRENCY=${AGENT_CONCURRENCY:-64}
AGENT_CONFIG_LOCAL_PATH=$(mktemp "${AGENT_CONFIG_TMPDIR%/}/uni-agent-v1-lmcache-agent.XXXXXX.yaml")
AGENT_CONFIG_PATH="examples/agent_interaction/$(basename "${AGENT_CONFIG_LOCAL_PATH}")"

lmcache_pid=""
LMCACHE_SERVER_LOG="${LMCACHE_LOG_DIR%/}/${safe_exp_name}.lmcache-server.log"
RAY_JOB_LOG="${LMCACHE_LOG_DIR%/}/${safe_exp_name}.ray-job.log"
RUNTIME_ENV_LOCAL_PATH=""

cleanup() {
    local status=$?
    rm -f "${AGENT_CONFIG_LOCAL_PATH}"
    if [[ -n "${RUNTIME_ENV_LOCAL_PATH}" ]]; then
        rm -f "${RUNTIME_ENV_LOCAL_PATH}"
    fi
    if [[ -n "${lmcache_pid}" ]] && kill -0 "${lmcache_pid}" 2>/dev/null; then
        kill "${lmcache_pid}" 2>/dev/null || true
        wait "${lmcache_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT

sed -E \
    "s/^([[:space:]]*)concurrency:[[:space:]]*[0-9]+/\1concurrency: ${AGENT_CONCURRENCY}/" \
    "${AGENT_CONFIG_SOURCE}" > "${AGENT_CONFIG_LOCAL_PATH}"

mkdir -p "${LMCACHE_LOG_DIR}"

RUNTIME_ENV_EFFECTIVE="${RUNTIME_ENV}"
if [[ "${SANITIZE_VEFAAS_ROUTE}" == "True" || "${SANITIZE_VEFAAS_ROUTE}" == "true" || "${SANITIZE_VEFAAS_ROUTE}" == "1" ]]; then
    RUNTIME_ENV_LOCAL_PATH=$(mktemp "${LMCACHE_LOG_DIR%/}/${safe_exp_name}.runtime-env.XXXXXX.yaml")
    python3 - "${RUNTIME_ENV}" "${RUNTIME_ENV_LOCAL_PATH}" <<'PY_RUNTIME_ENV'
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

env_vars = config.get("env_vars") or {}
route = env_vars.get("VEFAAS_FUNCTION_ROUTE")
if isinstance(route, str):
    env_vars["VEFAAS_FUNCTION_ROUTE"] = route.rstrip("/")
# Validation needs vLLM INFO stats such as Prefix cache hit rate.
env_vars["VLLM_LOGGING_LEVEL"] = "INFO"

with dst.open("w", encoding="utf-8") as f:
    f.write("# LMCache validation runtime env copy.\n")
    f.write("# Original training runtime_env.yaml is not modified.\n")
    f.write("# VEFAAS_FUNCTION_ROUTE is normalized without a trailing slash to avoid //create_session.\n")
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
PY_RUNTIME_ENV
    RUNTIME_ENV_EFFECTIVE="${RUNTIME_ENV_LOCAL_PATH}"
fi

lmcache server \
    --host "${LMCACHE_MP_BIND_HOST}" \
    --port "${LMCACHE_MP_PORT}" \
    --http-host "${LMCACHE_HTTP_HOST}" \
    --http-port "${LMCACHE_HTTP_PORT}" \
    --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
    --eviction-policy LRU \
    --chunk-size "${LMCACHE_CHUNK_SIZE}" \
    > "${LMCACHE_SERVER_LOG}" 2>&1 &
lmcache_pid=$!

python3 - "${LMCACHE_MP_BIND_HOST}" "${LMCACHE_MP_PORT}" "${lmcache_pid}" "${LMCACHE_WAIT_TIMEOUT_SECONDS}" <<'PY'
import os
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
pid = int(sys.argv[3])
timeout = float(sys.argv[4])
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            print(f"LMCache MP server is reachable at {host}:{port}")
            sys.exit(0)
    except OSError as exc:
        last_error = exc

    try:
        os.kill(pid, 0)
    except OSError:
        print(f"LMCache MP server process {pid} exited before becoming reachable", file=sys.stderr)
        sys.exit(2)

    time.sleep(1.0)

print(f"Timed out waiting for LMCache MP server at {host}:{port}: {last_error}", file=sys.stderr)
sys.exit(1)
PY

kv_transfer_config=$(printf '{kv_connector:LMCacheMPConnector,kv_connector_module_path:lmcache.integration.vllm.lmcache_mp_connector,kv_role:kv_both,kv_connector_extra_config:{lmcache.mp.host:%s,lmcache.mp.port:%s,lmcache.mp.mp_transfer_mode:%s}}' \
    "${LMCACHE_MP_CONNECTOR_HOST}" \
    "${LMCACHE_MP_PORT}" \
    "${LMCACHE_MP_TRANSFER_MODE}")

rollout_name="vllm"
rollout_mode="async"
adv_estimator="grpo"

use_kl_in_reward=False
use_kl_loss=False

max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-65536}

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:--1}

enforce_eager=${ENFORCE_EAGER:-False}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
offload=${OFFLOAD:-True}
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

optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}

NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Keep both the producer backlog and each GRPO group small for validation.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-${train_prompt_bsz}}
total_training_steps=${TOTAL_TRAINING_STEPS:-5}
test_freq=${TEST_FREQ:-10}
val_max_samples=${VAL_MAX_SAMPLES:-128}
val_batch_size=${VAL_BATCH_SIZE:-${val_max_samples}}
tq_num_data_storage_units=${TQ_NUM_DATA_STORAGE_UNITS:-8}
tq_total_storage_size=${TQ_TOTAL_STORAGE_SIZE:-8192}

cd "${RAY_DATA_HOME}/code/uni-agent"

ray_job_cmd=(
    ray job submit --runtime-env "${RUNTIME_ENV_EFFECTIVE}"
    --
    python3 -m verl.trainer.main_ppo
    --config-name=ppo_trainer
    model_engine=megatron

    trainer.use_v1=True
    trainer.v1.trainer_mode=colocate_async
    trainer.v1.colocate_async.num_warmup_batches=0

    transfer_queue.enable=True
    transfer_queue.backend.storage_backend=SimpleStorage
    "transfer_queue.backend.SimpleStorage.num_data_storage_units=${tq_num_data_storage_units}"
    "transfer_queue.backend.SimpleStorage.total_storage_size=${tq_total_storage_size}"

    trainer.v1.sampler.max_off_policy_threshold=2
    trainer.v1.sampler.max_off_policy_strategy=drop

    trainer.v1.sampler.custom_sampler.path=null
    trainer.v1.sampler.custom_sampler.name=null

    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    "data.val_max_samples=${val_max_samples}"
    "data.val_batch_size=${val_batch_size}"
    data.prompt_key=prompt
    data.filter_overlong_prompts=True
    data.truncation=error
    "data.max_prompt_length=${max_prompt_length}"
    "data.max_response_length=${max_response_length}"
    "data.train_batch_size=${train_prompt_bsz}"
    data.return_raw_chat=True

    "actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG_PATH}"
    actor_rollout_ref.rollout.agent.default_agent_loop=swe_agent
    actor_rollout_ref.rollout.agent.num_workers=2
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    "actor_rollout_ref.rollout.n=${n_resp_per_prompt}"

    "algorithm.adv_estimator=${adv_estimator}"
    "algorithm.use_kl_in_reward=${use_kl_in_reward}"
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.bypass_mode=True
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}"
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low=4e-4
    actor_rollout_ref.actor.clip_ratio_high=4e-4
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    "+actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length))"
    actor_rollout_ref.model.use_fused_kernels=False
    "actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=0.1
    "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    "actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE}"
    "actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT}"
    "actor_rollout_ref.actor.megatron.param_offload=${offload}"
    "actor_rollout_ref.actor.megatron.grad_offload=${offload}"
    "actor_rollout_ref.actor.megatron.optimizer_offload=${offload}"
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}"
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}"
    "actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}"
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean

    "actor_rollout_ref.rollout.name=${rollout_name}"
    "actor_rollout_ref.rollout.mode=${rollout_mode}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}"
    "actor_rollout_ref.rollout.data_parallel_size=${gen_dp}"
    "actor_rollout_ref.rollout.pipeline_model_parallel_size=${gen_pp}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))"
    "actor_rollout_ref.rollout.temperature=${temperature}"
    "actor_rollout_ref.rollout.top_p=${top_p}"
    "actor_rollout_ref.rollout.top_k=${top_k}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    "actor_rollout_ref.rollout.enforce_eager=${enforce_eager}"
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.hybrid_engine=True

    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.disable_log_stats=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_hybrid_kv_cache_manager=True
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config=${kv_transfer_config}"

    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"
    "actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT}"
    "actor_rollout_ref.ref.megatron.param_offload=${offload}"
    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}"
    "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}"
    "actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}"

    reward.reward_model.enable=False
    reward.reward_model.enable_resource_pool=False

    "trainer.logger=['console']"
    "trainer.project_name=${project_name}"
    "trainer.experiment_name=${exp_name}"
    trainer.val_before_train=False
    "trainer.test_freq=${test_freq}"
    trainer.save_freq=-1
    trainer.total_epochs=1
    "trainer.total_training_steps=${total_training_steps}"
    trainer.resume_mode=disable
    trainer.log_val_generations=0
    "trainer.default_local_dir=${CKPTS_DIR}"
    "trainer.nnodes=${NNODES_TRAIN}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "ray_kwargs.timeline_json_file=${RAY_PROFILING_PATH}"

    # "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}"
    # "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}"
    # "actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}"
    # "actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}"
)

set +e
"${ray_job_cmd[@]}" 2>&1 | tee "${RAY_JOB_LOG}"
ray_status=${PIPESTATUS[0]}
set -e

if (( ray_status != 0 )); then
    echo "Ray job failed with status ${ray_status}; skipping cache-hit validation." >&2
    exit "${ray_status}"
fi

log_search_paths=("${RAY_JOB_LOG}" "${LMCACHE_SERVER_LOG}")
if [[ -d /tmp/ray/session_latest/logs ]]; then
    log_search_paths+=(/tmp/ray/session_latest/logs)
fi

validation_failed=0

if grep -R -E "LMCache hit tokens: [1-9][0-9]*|Stored [1-9][0-9]* tokens" "${log_search_paths[@]}" >/dev/null 2>&1; then
    echo "Validation: found non-zero LMCache external store/write."
else
    echo "Validation failed: no non-zero LMCache store/write entry found." >&2
    validation_failed=1
fi

if grep -R -E "need to load: [1-9][0-9]*|Retrieved [1-9][0-9]* tokens" "${log_search_paths[@]}" >/dev/null 2>&1; then
    echo "Validation: found at least one real LMCache retrieve/read."
else
    echo "Validation failed: no non-zero LMCache retrieve/read entry found." >&2
    validation_failed=1
fi

if grep -R -P -i "(?<!External )Prefix cache hit rate: (0\.[0-9]*[1-9]|[1-9][0-9]*(\.[0-9]+)?)%" "${log_search_paths[@]}" >/dev/null 2>&1; then
    echo "Validation: found non-zero vLLM local prefix-cache hit-rate log."
else
    echo "Validation failed: no non-zero vLLM local prefix-cache hit-rate log found." >&2
    validation_failed=1
fi

if grep -R -E -i "External prefix cache hit rate: (0\.[0-9]*[1-9]|[1-9][0-9]*(\.[0-9]+)?)%" "${log_search_paths[@]}" >/dev/null 2>&1; then
    echo "Validation: found non-zero vLLM external prefix-cache hit-rate log."
fi

if grep -R -E "training/num_turns/mean:np.float64\(0\.0\)|training/num_turns/mean:0\.0" "${RAY_JOB_LOG}" >/dev/null 2>&1; then
    echo "Validation failed: rollout produced zero agent turns, so cache-hit validation is not meaningful." >&2
    if [[ -d /file_system/dhl/tmp/swebench_qwen3_coder ]]; then
        grep -R -n -E "Agent loop failed before producing interaction result|create_session|//create_session|Prompt Tokens|Completion Tokens" \
            /file_system/dhl/tmp/swebench_qwen3_coder 2>/dev/null | tail -20 >&2 || true
    fi
    validation_failed=1
fi

if (( validation_failed != 0 )); then
    echo "Cache validation failed. Logs:" >&2
    echo "  LMCache server: ${LMCACHE_SERVER_LOG}" >&2
    echo "  Ray job:        ${RAY_JOB_LOG}" >&2
    exit 2
fi

echo "Cache validation passed. Logs:"
echo "  LMCache server: ${LMCACHE_SERVER_LOG}"
echo "  Ray job:        ${RAY_JOB_LOG}"
