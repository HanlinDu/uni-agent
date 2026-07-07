from __future__ import annotations

import importlib.util

import pytest

from uni_agent.cache import (
    build_cache_salt,
    build_epoch,
    build_request_id,
    compute_store_prefix_len,
    compute_system_prefix_len,
    get_namespace,
    resolve_model_cache_epoch,
    with_model_cache_epoch,
)


def test_build_cache_salt_is_stable_and_epoch_scoped() -> None:
    cfg = {
        "namespace": "uni-agent/system prefix",
        "model_cache_epoch": "epoch0",
        "tokenizer_namespace": "qwen3",
    }
    system_messages = [{"role": "system", "content": "You are a coding agent."}]
    tools = [{"type": "function", "function": {"name": "bash"}}]

    salt = build_cache_salt(
        cfg,
        model_path="/models/qwen",
        tools_schemas=tools,
        system_messages=system_messages,
    )
    assert salt == build_cache_salt(
        dict(cfg),
        model_path="/models/qwen",
        tools_schemas=list(tools),
        system_messages=list(system_messages),
    )
    assert " " not in salt
    assert "/" not in salt
    assert get_namespace(cfg) in salt

    next_epoch_salt = build_cache_salt(
        {**cfg, "model_cache_epoch": "step-1"},
        model_path="/models/qwen",
        tools_schemas=tools,
        system_messages=system_messages,
    )
    assert next_epoch_salt != salt


def test_runtime_epoch_overrides_static_epoch_without_mutating_input() -> None:
    cfg = {"model_cache_epoch": "epoch0"}
    runtime_cfg = with_model_cache_epoch(cfg, "step-3")

    assert cfg == {"model_cache_epoch": "epoch0"}
    assert resolve_model_cache_epoch(cfg) == "epoch0"
    assert resolve_model_cache_epoch(runtime_cfg) == "step-3"
    assert build_epoch(3) == "step-3"
    assert build_epoch("7") == "step-7"
    assert build_epoch("bad") == "step-0"


@pytest.mark.parametrize(
    ("prompt_ids", "variant_prompt_ids", "expected"),
    [
        ([1, 2, 3], [1, 2, 3], 3),
        ([1, 2, 3, 4], [1, 2, 9, 4], 2),
        ([1, 2, 3], [9, 2, 3], 0),
        ([], [1, 2], 0),
        ([1, 2], [], 0),
    ],
)
def test_compute_system_prefix_len(prompt_ids: list[int], variant_prompt_ids: list[int], expected: int) -> None:
    assert compute_system_prefix_len(prompt_ids, variant_prompt_ids) == expected


@pytest.mark.parametrize(
    ("system_prefix_len", "chunk_size", "expected"),
    [
        (884, 64, 832),
        (884, 0, 884),
        (63, 64, 0),
        (64, 64, 64),
        (0, 64, 0),
        (-10, 64, 0),
    ],
)
def test_compute_store_prefix_len(system_prefix_len: int, chunk_size: int, expected: int) -> None:
    assert compute_store_prefix_len(system_prefix_len, chunk_size) == expected


def test_build_request_id_encodes_policy() -> None:
    request_id = build_request_id(read=True, write=False, store_token_limit=832, base_request_id="abc")
    assert request_id == "uaxpc__r1__w0__l832__abc"

    request_id = build_request_id(read=False, write=True, store_token_limit=-1, base_request_id="abc")
    assert request_id == "uaxpc__r0__w1__l0__abc"


def test_extract_extra_prefix_cache_config_from_omegaconf() -> None:
    from omegaconf import OmegaConf

    from uni_agent.cache.epoch import _extract_extra_prefix_cache_config

    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "extra_prefix_cache": {
                        "enable": True,
                        "advance_epoch_on_weight_update": True,
                        "namespace": "uni-agent-system-prefix",
                    }
                }
            }
        }
    )

    assert _extract_extra_prefix_cache_config(cfg) == {
        "enable": True,
        "advance_epoch_on_weight_update": True,
        "namespace": "uni-agent-system-prefix",
    }


@pytest.mark.asyncio
async def test_epoch_advance_and_resolve_via_internal_kv(monkeypatch: pytest.MonkeyPatch) -> None:
    from uni_agent.cache import maybe_advance_extra_prefix_cache_epoch
    from uni_agent.cache.epoch import resolve_runtime_model_cache_epoch

    values: dict[str, str] = {}

    monkeypatch.setattr("uni_agent.cache.epoch._internal_kv_set_epoch", lambda namespace, epoch: values.setdefault(namespace, epoch) == epoch)
    monkeypatch.setattr("uni_agent.cache.epoch._internal_kv_get_epoch", lambda namespace: values.get(namespace))

    cfg = {
        "actor_rollout_ref": {
            "rollout": {
                "extra_prefix_cache": {
                    "enable": True,
                    "namespace": "unit-test",
                    "model_cache_epoch": "epoch0",
                }
            }
        }
    }

    assert maybe_advance_extra_prefix_cache_epoch(cfg, 2) == "step-2"
    assert await resolve_runtime_model_cache_epoch(
        {"enable": True, "namespace": "unit-test", "model_cache_epoch": "epoch0"}
    ) == "step-2"


@pytest.mark.asyncio
async def test_runtime_epoch_ignores_registry_when_advance_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from uni_agent.cache.epoch import resolve_runtime_model_cache_epoch

    monkeypatch.setattr("uni_agent.cache.epoch._internal_kv_get_epoch", lambda namespace: "step-3")

    assert await resolve_runtime_model_cache_epoch(
        {
            "enable": True,
            "namespace": "unit-test",
            "model_cache_epoch": "epoch0",
            "advance_epoch_on_weight_update": False,
        }
    ) == "epoch0"


@pytest.mark.skipif(importlib.util.find_spec("lmcache") is None, reason="LMCache is not installed")
def test_lmcache_system_prefix_connector_policy_parser() -> None:
    from uni_agent.cache.lmcache_system_prefix_connector import LMCacheSystemPrefixConnector

    connector = LMCacheSystemPrefixConnector.__new__(LMCacheSystemPrefixConnector)

    assert connector._policy_from_request_id("uaxpc__r1__w0__l832__abc") == {
        "read": True,
        "write": False,
        "store_token_limit": 832,
        "tagged": True,
    }
    assert connector._policy_from_request_id("untagged-request") == {
        "read": True,
        "write": True,
        "store_token_limit": 0,
        "tagged": False,
    }
