import sys
import types

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.weight_update_hooks import maybe_run_weight_update_hooks


def test_weight_update_hooks_noop_without_config() -> None:
    config = OmegaConf.create({"trainer": {}})

    assert maybe_run_weight_update_hooks(config, 3) == []


def test_weight_update_hooks_dynamic_callable(monkeypatch) -> None:
    module = types.ModuleType("verl_test_weight_update_hook")
    calls = []

    def hook(config, global_step, *, log=None, suffix=""):
        calls.append((global_step, suffix, log is not None))
        return f"step-{global_step}{suffix}"

    module.hook = hook
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = OmegaConf.create(
        {
            "trainer": {
                "weight_update_hooks": [
                    {
                        "path": "verl_test_weight_update_hook.hook",
                        "kwargs": {"suffix": "-ok"},
                    }
                ]
            }
        }
    )

    assert maybe_run_weight_update_hooks(config, 7) == ["step-7-ok"]
    assert calls == [(7, "-ok", True)]
