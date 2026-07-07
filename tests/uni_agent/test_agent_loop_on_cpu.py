from __future__ import annotations

from types import SimpleNamespace

import pytest

from uni_agent.agent_loop import UniAgentLoop


class _FakeChatModel:
    def __init__(self):
        self.tools_schemas = None

    def set_tools_schemas(self, tools_schemas):
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(self, messages):
        return {
            "prompt_ids": [11, 12, 13],
            "extra_fields": {},
        }


@pytest.mark.asyncio
async def test_build_empty_agent_output_keeps_minimal_valid_response_mask() -> None:
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.chat_model = _FakeChatModel()
    loop.tools_manager = SimpleNamespace(tools_schemas=[])
    loop.interaction = SimpleNamespace(messages=[{"role": "user", "content": "hello"}])
    loop.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(prompt_length=16, response_length=32),
        )
    )
    loop.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    loop._synth_failed_routed_experts = lambda length: None

    output = await loop._build_empty_agent_output(exit_reason="agent_loop_failed")

    assert output.response_ids == [0]
    assert output.response_mask == [1]
    assert output.reward_score == 0
    assert output.extra_fields["traj_masked"] == 1
    assert output.extra_fields["traj_exit_reason"] == "agent_loop_failed"
