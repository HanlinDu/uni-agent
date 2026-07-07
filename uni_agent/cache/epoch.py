import logging
from typing import Any

from .extra_prefix_cache import (
    build_epoch,
    enabled,
    get_namespace,
    normalize_config,
    resolve_model_cache_epoch,
)

logger = logging.getLogger(__name__)

_ACTOR_NAME = "uni_agent_extra_prefix_cache_epoch_registry"
_INTERNAL_KV_NAMESPACE = "uni_agent_extra_prefix_cache"
_INTERNAL_KV_PREFIX = "epoch:"


class _ExtraPrefixCacheEpochRegistry:
    def __init__(self) -> None:
        self._epochs: dict[str, str] = {}

    def get(self, namespace: str, default: str) -> str:
        return self._epochs.get(namespace, default)

    def set(self, namespace: str, epoch: str) -> str:
        self._epochs[namespace] = epoch
        return epoch

    def advance(self, namespace: str, global_step: int | str | None) -> str:
        epoch = build_epoch(global_step)
        self._epochs[namespace] = epoch
        return epoch

    def snapshot(self) -> dict[str, str]:
        return dict(self._epochs)


def _try_get_ray():
    try:
        import ray
    except Exception:
        return None
    try:
        if not ray.is_initialized():
            return None
    except Exception:
        return None
    return ray


def _internal_kv_key(namespace: str) -> str:
    return f"{_INTERNAL_KV_PREFIX}{namespace}"


def _internal_kv_available() -> bool:
    if _try_get_ray() is None:
        return False
    try:
        from ray.experimental.internal_kv import _internal_kv_initialized

        return bool(_internal_kv_initialized())
    except Exception:
        return False


def _internal_kv_get_epoch(namespace: str) -> str | None:
    if not _internal_kv_available():
        return None
    try:
        from ray.experimental.internal_kv import _internal_kv_get

        value = _internal_kv_get(_internal_kv_key(namespace), namespace=_INTERNAL_KV_NAMESPACE)
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _internal_kv_set_epoch(namespace: str, epoch: str) -> bool:
    if not _internal_kv_available():
        return False
    try:
        from ray.experimental.internal_kv import _internal_kv_put

        _internal_kv_put(
            _internal_kv_key(namespace),
            str(epoch),
            overwrite=True,
            namespace=_INTERNAL_KV_NAMESPACE,
        )
        return True
    except Exception:
        return False


def _get_epoch_registry(*, create: bool):
    ray = _try_get_ray()
    if ray is None:
        return None

    try:
        return ray.get_actor(_ACTOR_NAME)
    except Exception:
        if not create:
            return None

    actor_cls = ray.remote(num_cpus=0)(_ExtraPrefixCacheEpochRegistry)
    try:
        return actor_cls.options(name=_ACTOR_NAME, get_if_exists=True).remote()
    except TypeError:
        try:
            return actor_cls.options(name=_ACTOR_NAME).remote()
        except Exception:
            try:
                return ray.get_actor(_ACTOR_NAME)
            except Exception:
                return None
    except Exception:
        try:
            return ray.get_actor(_ACTOR_NAME)
        except Exception:
            return None


def _extract_extra_prefix_cache_config(config: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            extra_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.extra_prefix_cache")
            if extra_cfg is not None:
                return normalize_config(extra_cfg)
    except Exception:
        pass

    try:
        rollout_cfg = config.actor_rollout_ref.rollout
    except Exception:
        rollout_cfg = None

    if rollout_cfg is None and isinstance(config, dict):
        rollout_cfg = (config.get("actor_rollout_ref") or {}).get("rollout")

    if rollout_cfg is None:
        return normalize_config(config)

    try:
        extra_cfg = rollout_cfg.get("extra_prefix_cache", None)
    except Exception:
        extra_cfg = getattr(rollout_cfg, "extra_prefix_cache", None)
    return normalize_config(extra_cfg)


def _emit_epoch_status(run_logger: logging.Logger, level: str, message: str, *args: Any) -> None:
    rendered = message % args if args else message
    getattr(run_logger, level)(rendered)
    # Ray sometimes suppresses module logger output from trainer hooks. Keep an
    # explicit stdout breadcrumb for validation and diagnosis.
    print(rendered, flush=True)


def maybe_advance_extra_prefix_cache_epoch(
    config: Any,
    global_step: int | str | None,
    *,
    log: logging.Logger | None = None,
) -> str | None:
    """Advance the runtime cache epoch after rollout weights are updated.

    The external KV cache is invalidated logically: new requests use a new
    namespace epoch in ``cache_salt`` and therefore stop reading old KV entries.
    Physical deletion is intentionally left to the backend retention policy.
    """

    run_logger = log or logger
    cfg = _extract_extra_prefix_cache_config(config)
    if not enabled(cfg):
        return None
    if not bool(cfg.get("advance_epoch_on_weight_update", True)):
        _emit_epoch_status(
            run_logger,
            "info",
            "ExtraPrefixCache epoch advance skipped reason=policy_off config=%s",
            cfg,
        )
        return None

    namespace = get_namespace(cfg)
    epoch = build_epoch(global_step)

    if _internal_kv_set_epoch(namespace, epoch):
        _emit_epoch_status(
            run_logger,
            "info",
            "ExtraPrefixCache epoch advanced namespace=%s epoch=%s global_step=%s backend=ray_internal_kv",
            namespace,
            epoch,
            global_step,
        )
        return epoch

    actor = _get_epoch_registry(create=True)
    ray = _try_get_ray()
    if actor is None or ray is None:
        _emit_epoch_status(
            run_logger,
            "warning",
            "ExtraPrefixCache epoch advance skipped namespace=%s epoch=%s reason=registry_unavailable",
            namespace,
            epoch,
        )
        return None

    try:
        epoch = ray.get(actor.set.remote(namespace, epoch))
    except Exception as exc:
        _emit_epoch_status(
            run_logger,
            "warning",
            "ExtraPrefixCache epoch advance failed namespace=%s global_step=%s error=%s",
            namespace,
            global_step,
            exc,
        )
        return None

    _emit_epoch_status(
        run_logger,
        "info",
        "ExtraPrefixCache epoch advanced namespace=%s epoch=%s global_step=%s backend=ray_actor",
        namespace,
        epoch,
        global_step,
    )
    return str(epoch)


async def resolve_runtime_model_cache_epoch(
    config: Any,
    *,
    log: logging.Logger | None = None,
) -> str:
    cfg = normalize_config(config)
    default_epoch = resolve_model_cache_epoch(cfg)
    if not enabled(cfg):
        return default_epoch
    if not bool(cfg.get("advance_epoch_on_weight_update", True)):
        return default_epoch

    namespace = get_namespace(cfg)
    internal_epoch = _internal_kv_get_epoch(namespace)
    if internal_epoch:
        return internal_epoch

    actor = _get_epoch_registry(create=False)
    if actor is None:
        return default_epoch

    try:
        return str(await actor.get.remote(namespace, default_epoch))
    except Exception as exc:
        run_logger = log or logger
        run_logger.warning(
            "ExtraPrefixCache epoch resolve failed namespace=%s default_epoch=%s error=%s",
            namespace,
            default_epoch,
            exc,
        )
        return default_epoch
