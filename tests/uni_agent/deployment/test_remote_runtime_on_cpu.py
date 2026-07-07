from __future__ import annotations

from uni_agent.deployment.remote_runtime import RemoteRuntime


def test_remote_runtime_strips_trailing_slash_from_base_url() -> None:
    runtime = RemoteRuntime(
        run_id="test",
        auth_token="token",
        base_url="https://example.com/runtime/",
    )

    assert runtime._api_url == "https://example.com/runtime"