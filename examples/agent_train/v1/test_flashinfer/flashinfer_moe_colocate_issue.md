# [Bug] FlashInfer FP16 MoE produces corrupted rollouts with Qwen3-Coder-30B-A3B in multi-node colocate-async

## Summary

Enabling `VLLM_USE_FLASHINFER_MOE_FP16=1` causes systematically corrupted
rollouts when running Qwen3-Coder-30B-A3B-Instruct with the VERL v1
`colocate_async` trainer.

With the variable disabled, the same model, prompts, parallelism, and trainer
configuration produce normal multi-turn agent trajectories and valid SWE-bench
Verified rewards. With the variable enabled, all eight tested trajectories
degenerate into repetitive multilingual fragments, fail to emit valid tool
calls, and consume almost the entire 65K response budget in only one or two
assistant turns.

The job itself does not crash. The corrupted trajectories are accepted by the
trainer and result in all-zero rewards.

## Environment

### Machines

- 2 nodes
- 8 x NVIDIA H20 per node
- 96 GB GPU memory per GPU (`97871 MiB`)
- Compute capability: 9.0
- NVIDIA driver: `535.161.08`
- Driver-reported CUDA compatibility: `12.9`

### Container image

The test ran in a private Kubernetes CUDA container. The registry and image tag
are not exposed inside the user container, so an exact pullable image reference
is unavailable.

Observable image configuration:

- Ubuntu `22.04.5 LTS`
- Image build serial: `20250714`
- CUDA toolkit/runtime: `12.9.1`
- cuDNN: `9.16.0.29`
- NCCL: `2.27.3`
- Python: `3.12.13`
- PyTorch: `2.11.0+cu129`
- vLLM: `0.20.2+cu129`
- FlashInfer: `0.6.8.post1`
- FlashAttention: `2.8.3`
- Triton: `3.6.0`
- Ray: `2.54.1`
- Transformers: `5.12.1`
- Megatron-Core: `0.16.0`
- mbridge: `0.15.1`

### Model

`Qwen3-Coder-30B-A3B-Instruct`

Local checkpoint path used in the test:

```text
/file_system/common-models/Qwen/Qwen3-Coder-30B-A3B-Instruct
```

### Source revisions

Uni-Agent:

```text
Repository: https://github.com/HanlinDu/uni-agent
Commit: 0754facf6537aee0a929ec4df3fa59f2495ed038
Subject: [core] fix: replace per-run log sinks with single dispatch sink to fix leaks and HDFS stalls (#63)
```

VERL:

```text
Repository: https://github.com/volcengine/verl
Commit: 8a694930275061f52ebd538c906ef8819af56dbd
Version description: v0.8.0-86-g8a694930
Subject: [fsdp, model] feat: per-unit LoRA summon, FSDP1/2 compatibility, and strip-modules support (#6512)
```

Both worktrees were dirty. Relevant local VERL changes were:

1. Run `process_weights_after_loading()` under
   `set_current_vllm_config(self.model_runner.vllm_config)` after colocated
   weight loading.
2. Disable the multi-node vLLM CLI arguments in `vllm_async_server.py` for this
   colocated deployment.
3. Add support for the negative
   `--no-enable-prefix-caching` BooleanOptionalAction flag.
4. Add diagnostic logging around abort, sleep, weight update, and generation
   resume.

The Uni-Agent runtime environment also contained the multi-node NCCL/GLOO
interface settings and shared CUDA/Triton cache paths. Infrastructure
credentials are intentionally omitted.

## Trainer configuration

- Trainer mode: `colocate_async`
- Training world: 2 nodes x 8 GPUs
- Training parallelism:
  - TP = 4
  - PP = 2
  - CP = 2
  - EP = 8
  - ETP = 1
- Rollout parallelism:
  - TP = 8
  - DP = 1
  - PP = 1
- vLLM mode: async
- Rollouts per prompt: 1
- Fixed prompts: first 8 rows of the same SWE-bench Verified parquet
- Prompt batch size: 8
- PPO mini-batch size: 8
- Maximum prompt length: 4096
- Maximum response length: 65536
- Sampling: temperature 1.0, top-p 1.0
- One training step
- No checkpoint save

The two runs used byte-identical input prompts. The only intended A/B
difference was:

```yaml
# Baseline
VLLM_USE_FLASHINFER_MOE_FP16: "0"

# Failing run
VLLM_USE_FLASHINFER_MOE_FP16: "1"
```

## Reproduction

Use the standalone script attached with this report:

```text
examples/agent_train/v1/test_flashinfer_moe_colocate_ab_all_in_one.sh
```

This script contains the complete Ray/Hydra launch command. It does not
dynamically rewrite or invoke the existing training launcher. It only creates:

1. A fixed eight-row parquet on a shared filesystem.
2. A temporary Ray runtime environment with the selected
   `VLLM_USE_FLASHINFER_MOE_FP16` value.
3. A temporary agent YAML with reduced concurrency.

Required environment-specific inputs:

```bash
export TEST_FILE=/path/to/swe_bench_verified_vefaas.parquet
export MODEL_PATH=/path/to/Qwen3-Coder-30B-A3B-Instruct
export RUNTIME_ENV_SOURCE=/path/to/runtime_env.yaml
export AGENT_CONFIG_SOURCE=/path/to/agent_config_vefaas.yaml
export FLASHINFER_TEST_SHARED_ROOT=/shared/path/visible/to/all/ray/nodes
```

Run the baseline:

```bash
bash examples/agent_train/v1/test_flashinfer_moe_colocate_ab_all_in_one.sh off
```

Run FlashInfer FP16 MoE:

```bash
bash examples/agent_train/v1/test_flashinfer_moe_colocate_ab_all_in_one.sh on
```

The runtime environment must contain the deployment credentials required by
the SWE agent sandbox. These credentials are unrelated to the A/B variable and
should not be included in public logs.

## Results

| Metric | FP16 MoE off | FP16 MoE on |
|---|---:|---:|
| Rollout records written | 8/8 | 8/8 |
| Non-empty serialized outputs | 8/8 | 8/8 |
| SWE-bench Verified reward | 2/8 | 0/8 |
| Mean number of agent turns | 44.625 | 1.875 |
| Trajectories ending at token limit | 1/8 | 8/8 |
| Mean response tokens | 39,860 | 65,514 |
| Response length clip ratio | 0.125 | 0.875 |
| Rollout/actor probability Pearson correlation | 0.99916 | 0.37015 |
| Rollout correlation KL | 0.00239 | 5.83441 |
| Training PPL | 2.81 | 2182.82 |
| Mean reward passed to training | 0.25 | 0.0 |

### Baseline behavior

Seven of eight trajectories ended with `finished`. They contained valid
`<tool_call>` blocks, interacted with the repository, submitted patches, and
two samples received reward 1.

Example beginning:

```text
I'll work through this issue systematically...

<tool_call>
<function=str_replace_editor>
...
```

### Behavior with `VLLM_USE_FLASHINFER_MOE_FP16=1`

All eight trajectories ended with `token_limit` after only one or two turns.
They contained repetitive and corrupted-looking multilingual fragments and no
valid tool-call structure.

Example beginning:

```text
I'll fix the same gapestdtehukotusWith a concept that is already in the order.
I will use the entire codetexts to in the number the options...
```

The on-run also logged FlashInfer/TRT-LLM fused MoE autotuning:

```text
[AutoTuner]: Tuning trtllm::fused_moe::gemm1
[AutoTuner]: Tuning trtllm::fused_moe::gemm2
```

The training-side diagnostics strongly indicate that rollout model outputs no
longer match the training model:

```text
training/rollout_actor_probs_pearson_corr: 0.370153
rollout_corr/kl: 5.834410
rollout_corr/training_ppl: 2182.823
```

The baseline values were:

```text
training/rollout_actor_probs_pearson_corr: 0.999158
rollout_corr/kl: 0.002392
rollout_corr/training_ppl: 2.811
```

## Expected behavior

Enabling the FlashInfer FP16 MoE implementation may change performance and
small numerical details, but it should preserve coherent text generation,
valid agent tool calls, and close agreement between rollout and training-side
probabilities.

## Actual behavior

The FlashInfer FP16 MoE path silently produces unusable generations. The
rollout pipeline and trainer complete successfully, so the invalid trajectories
are passed into training as apparently valid samples with zero reward.

## Conclusion

`VLLM_USE_FLASHINFER_MOE_FP16=1` is not usable for this configuration:

- NVIDIA H20
- Qwen3-Coder-30B-A3B-Instruct
- vLLM 0.20.2
- FlashInfer 0.6.8.post1
- TP=8 rollout
- VERL v1 multi-node colocate-async

Keeping `VLLM_USE_FLASHINFER_MOE_FP16=0` restores normal agent rollouts and
rollout/training probability agreement.

The failure appears to be in the FlashInfer/TRT-LLM FP16 fused MoE execution
path or its interaction with colocated weight loading, rather than in the
agent, verifier, or dataset pipeline.

## Additional note

Both runs emitted a `StatefulDataLoader` worker cleanup warning after the
training step had completed and the rollout JSONL had been written. This
warning occurred in both configurations and does not explain the A/B
difference.
