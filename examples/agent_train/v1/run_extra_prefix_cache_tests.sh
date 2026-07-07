#!/usr/bin/env bash
set -euo pipefail

# Functional and performance test harness for uni-agent extra prefix cache.
#
# Default load:
#   TRAIN_PROMPT_BSZ=8, N_RESP_PER_PROMPT=4, AGENT_CONCURRENCY=64,
#   TOTAL_TRAINING_STEPS=10
#
# Cases:
#   functional    LMCache extra cache, external_reuse validation, must prove reuse.
#   perf-extra    LMCache extra cache under the same load.
#   perf-baseline No LMCache extra cache, same load, original colocate-async path.
#   perf          perf-extra + perf-baseline.
#   all           functional + perf-extra + perf-baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CASE="${1:-all}"
LOG_DIR="${LOG_DIR:-/tmp/uni-agent-lmcache/extra-prefix-tests}"
mkdir -p "${LOG_DIR}"

TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-8}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-${TRAIN_PROMPT_BSZ}}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-64}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
TEST_FREQ="${TEST_FREQ:--1}"

COMMON_ENV=(
    "TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}"
    "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}"
    "TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ}"
    "AGENT_CONCURRENCY=${AGENT_CONCURRENCY}"
    "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
    "TEST_FREQ=${TEST_FREQ}"
)

suffix="bsz${TRAIN_PROMPT_BSZ}_n${N_RESP_PER_PROMPT}_c${AGENT_CONCURRENCY}_s${TOTAL_TRAINING_STEPS}"

run_with_log() {
    local name="$1"
    local log_file="$2"
    shift 2

    echo "==== ${name} ====" | tee "${log_file}"
    echo "log_file=${log_file}" | tee -a "${log_file}"
    echo "started_at=$(date -Iseconds)" | tee -a "${log_file}"

    local start_ts
    start_ts=$(date +%s)
    set +e
    "$@" 2>&1 | tee -a "${log_file}"
    local status=${PIPESTATUS[0]}
    set -e
    local end_ts
    end_ts=$(date +%s)

    echo "finished_at=$(date -Iseconds)" | tee -a "${log_file}"
    echo "elapsed_seconds=$((end_ts - start_ts))" | tee -a "${log_file}"
    echo "exit_status=${status}" | tee -a "${log_file}"
    return "${status}"
}

run_functional() {
    local log_file="${LOG_DIR}/functional_extra_reuse_${suffix}.log"
    run_with_log "functional extra prefix cache reuse" "${log_file}" \
        env \
        "${COMMON_ENV[@]}" \
        "EXP_NAME=V1-Qwen3-4B-ExtraPrefix-Functional-${suffix}" \
        "LMCACHE_VALIDATION_MODE=external_reuse" \
        "LMCACHE_ADVANCE_EPOCH_ON_WEIGHT_UPDATE=False" \
        bash "${SCRIPT_DIR}/single_node_v1_collocate_async_lmcache.sh"
}

run_perf_extra() {
    local log_file="${LOG_DIR}/perf_extra_reuse_${suffix}.log"
    run_with_log "performance with extra prefix cache" "${log_file}" \
        env \
        "${COMMON_ENV[@]}" \
        "EXP_NAME=V1-Qwen3-4B-ExtraPrefix-Perf-${suffix}" \
        "LMCACHE_VALIDATION_MODE=external_reuse" \
        "LMCACHE_ADVANCE_EPOCH_ON_WEIGHT_UPDATE=False" \
        bash "${SCRIPT_DIR}/single_node_v1_collocate_async_lmcache.sh"
}

run_perf_baseline() {
    local log_file="${LOG_DIR}/perf_no_extra_${suffix}.log"
    run_with_log "performance without extra prefix cache" "${log_file}" \
        env \
        "${COMMON_ENV[@]}" \
        "EXP_NAME=V1-Qwen3-4B-NoExtra-Perf-${suffix}" \
        "ENABLE_PREFIX_CACHING=True" \
        "DISABLE_LOG_STATS=False" \
        bash "${SCRIPT_DIR}/single_node_v1_collocate_async.sh"
}

case "${CASE}" in
    functional)
        run_functional
        ;;
    perf-extra)
        run_perf_extra
        ;;
    perf-baseline|perf-no-extra|baseline)
        run_perf_baseline
        ;;
    perf)
        run_perf_extra
        run_perf_baseline
        ;;
    all)
        run_functional
        run_perf_extra
        run_perf_baseline
        ;;
    *)
        echo "Usage: $0 [functional|perf-extra|perf-baseline|perf|all]" >&2
        exit 64
        ;;
esac

cat <<EOF

Test logs are under: ${LOG_DIR}
Useful checks:
  rg -n "Cache validation passed|Validation failed|new_external_tokens=[1-9][0-9]* will_load=True|External prefix cache hit rate: (0\\.[0-9]*[1-9]|[1-9][0-9]*(\\.[0-9]+)?)%|elapsed_seconds=|exit_status=" ${LOG_DIR}/*_${suffix}.log
EOF
