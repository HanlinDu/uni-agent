#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export FLASHINFER_MOE_FP16=1
export EXP_NAME=${EXP_NAME:-test-colocate-flashinfer-moe-on}

exec "${SCRIPT_DIR}/test_colocate_flashinfer_moe_common.sh"
