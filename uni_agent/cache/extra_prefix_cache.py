import hashlib
import json
import re
import uuid
from typing import Any


REQUEST_ID_PREFIX = "uaxpc"
DEFAULT_NAMESPACE = "uni-agent-system-prefix"
DEFAULT_MODEL_CACHE_EPOCH = "epoch0"
_SAFE_SALT_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def normalize_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True) or {}
    except Exception:
        pass
    if hasattr(config, "items"):
        return dict(config.items())
    return {}


def enabled(config: Any) -> bool:
    cfg = normalize_config(config)
    return bool(cfg.get("enable", cfg.get("enabled", False)))


def get_namespace(config: Any) -> str:
    cfg = normalize_config(config)
    namespace = str(cfg.get("namespace", DEFAULT_NAMESPACE))
    return _sanitize_salt(namespace)


def build_epoch(global_step: int | str | None) -> str:
    try:
        step = int(global_step or 0)
    except (TypeError, ValueError):
        step = 0
    return f"step-{max(step, 0)}"


def resolve_model_cache_epoch(config: Any) -> str:
    cfg = normalize_config(config)
    return str(
        cfg.get("runtime_model_cache_epoch")
        or cfg.get("model_cache_epoch")
        or cfg.get("epoch")
        or DEFAULT_MODEL_CACHE_EPOCH
    )


def with_model_cache_epoch(config: Any, epoch: str | None) -> dict[str, Any]:
    cfg = normalize_config(config)
    if epoch:
        cfg["runtime_model_cache_epoch"] = str(epoch)
    return cfg


def _short_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _sanitize_salt(value: str) -> str:
    value = value.replace("@", ":").replace("/", ":").replace("\\", ":").replace("\x00", "")
    value = _SAFE_SALT_RE.sub("_", value).strip("_.:-")
    if not value:
        value = "default"
    if len(value) <= 192:
        return value
    return f"{value[:160]}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def build_cache_salt(
    config: Any,
    *,
    model_path: str | None,
    tools_schemas: list[dict] | None,
    system_messages: list[dict[str, Any]],
) -> str:
    cfg = normalize_config(config)
    namespace = get_namespace(cfg)
    model_namespace = str(cfg.get("model_namespace") or model_path or "model")
    epoch = resolve_model_cache_epoch(cfg)
    system_hash = _short_hash(
        {
            "tools": tools_schemas or [],
            "system_messages": system_messages,
            "tokenizer": cfg.get("tokenizer_namespace", ""),
        }
    )
    model_hash = _short_hash(model_namespace, length=10)
    return _sanitize_salt(f"{namespace}:{model_hash}:{epoch}:sys{system_hash}")


def compute_system_prefix_len(prompt_ids: list[int], variant_prompt_ids: list[int]) -> int:
    limit = min(len(prompt_ids), len(variant_prompt_ids))
    idx = 0
    while idx < limit and prompt_ids[idx] == variant_prompt_ids[idx]:
        idx += 1
    return idx


def get_positive_int(config: Any, key: str, default: int = 0) -> int:
    cfg = normalize_config(config)
    try:
        value = int(cfg.get(key, default) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def compute_store_prefix_len(system_prefix_len: int, chunk_size: int = 0) -> int:
    system_prefix_len = max(int(system_prefix_len or 0), 0)
    chunk_size = max(int(chunk_size or 0), 0)
    if system_prefix_len <= 0:
        return 0
    if chunk_size <= 0:
        return system_prefix_len
    return (system_prefix_len // chunk_size) * chunk_size


def build_request_id(
    *,
    read: bool,
    write: bool,
    store_token_limit: int = 0,
    base_request_id: str | None = None,
) -> str:
    base = base_request_id or uuid.uuid4().hex
    read_flag = 1 if read else 0
    write_flag = 1 if write else 0
    limit = max(int(store_token_limit or 0), 0)
    return f"{REQUEST_ID_PREFIX}__r{read_flag}__w{write_flag}__l{limit}__{base}"
