import asyncio
import hashlib
import json
import uuid
from functools import cached_property
from typing import Any

from uni_agent.utils import get_event_loop, simple_timer


class MaxTokenExceededError(Exception):
    pass


class AgentChatModel:
    client: Any
    """AsyncLLM server manager"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_model_len: int
    """Max model context length"""

    sampling_params: dict[str, Any]
    """Sampling parameters for the model"""

    tools_schemas: list[dict] = None

    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
        self.loop = asyncio.get_running_loop()

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        from verl.utils.tokenizer import normalize_token_ids

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                tools=self.tools_schemas,
            ),
        )
        prompt_ids = normalize_token_ids(prompt_ids)
        extra_prefix_cache = await self._build_extra_prefix_cache_metadata(messages, prompt_ids)
        return {
            "request_id": str(uuid.uuid4()),
            "prompt_ids": prompt_ids,
            "response_mask": [],
            "response_logprobs": [],
            "routed_experts": None,
            "metrics": {},
            "extra_fields": {},
            "extra_prefix_cache": extra_prefix_cache,
        }

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
    ):
        """Append newly added user/tool messages to the rollout cache."""

        valid_roles = {"user", "tool"}
        invalid_roles = [message["role"] for message in new_messages if message["role"] not in valid_roles]
        assert not invalid_roles, f"New messages must be user or tool, but got invalid roles: {invalid_roles}"

        # encode tool response
        tool_response_ids = await self._get_new_message_ids(new_messages)

        # append tool response to prompt
        rollout_cache["prompt_ids"] += tool_response_ids
        rollout_cache["response_mask"] += [0] * len(tool_response_ids)
        if rollout_cache["response_logprobs"]:
            rollout_cache["response_logprobs"] += [0.0] * len(tool_response_ids)

        return rollout_cache

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
        **kwargs,
    ) -> tuple[str, list[dict], dict[str, Any], dict[str, int]]:
        """Run one model call. Returns ``(text, tool_calls, rollout_cache,
        generation_info)``. ``tool_calls`` is always ``[]`` on the training
        path -- verl returns token ids, so callers must parse ``text``.
        """
        request_id = rollout_cache["request_id"]
        prompt_ids = rollout_cache["prompt_ids"]
        metrics = rollout_cache["metrics"]

        if len(prompt_ids) >= self.max_model_len:
            raise MaxTokenExceededError(
                f"prompt_ids length {len(rollout_cache['prompt_ids'])} exceeds max_model_len {self.max_model_len}\n"
                f"Last tool response: {messages[-1]['content']}"
            )

        sampling_params = kwargs.get("sampling_params", self.sampling_params)
        extra_prefix_cache_kwargs = self._build_extra_prefix_cache_generate_kwargs(rollout_cache)

        with simple_timer("generate_sequences", metrics):
            token_output = await self.client.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                **extra_prefix_cache_kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        else:
            metrics["num_preempted"] += token_output.num_preempted if token_output.num_preempted is not None else 0
        generation_info = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_output.token_ids),
        }
        response_ids = token_output.token_ids
        rollout_cache["prompt_ids"] += response_ids
        rollout_cache["response_mask"] += [1] * len(response_ids)
        if token_output.log_probs is not None:
            rollout_cache["response_logprobs"] += token_output.log_probs
        if token_output.routed_experts is not None:
            rollout_cache["routed_experts"] = token_output.routed_experts
        if not rollout_cache["extra_fields"]:
            rollout_cache["extra_fields"].update(token_output.extra_fields)
        else:
            max_global_steps = token_output.extra_fields.get("max_global_steps", None)
            if max_global_steps is not None:
                rollout_cache["extra_fields"]["max_global_steps"] = max_global_steps
        response_str = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(response_ids))

        if len(rollout_cache["prompt_ids"]) >= self.max_model_len:
            raise MaxTokenExceededError(
                f"prompt_ids length {len(rollout_cache['prompt_ids'])} exceeds max_model_len {self.max_model_len}\n"
                f"Generated response:\n{response_str}"
            )

        return response_str, [], rollout_cache, generation_info

    async def _build_extra_prefix_cache_metadata(
        self,
        messages: list[dict[str, Any]],
        prompt_ids: list[int],
    ) -> dict[str, Any]:
        config = getattr(self, "extra_prefix_cache", None) or {}
        if not _extra_prefix_cache_enabled(config):
            return {}

        from verl.utils.tokenizer import normalize_token_ids

        system_messages = [message for message in messages if message.get("role") == "system"]
        variant_messages = self._build_system_prefix_variant_messages(messages)
        variant_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                variant_messages,
                add_generation_prompt=True,
                tokenize=True,
                tools=self.tools_schemas,
            ),
        )
        variant_prompt_ids = normalize_token_ids(variant_prompt_ids)
        stable_prefix_token_len = _compute_common_prefix_len(prompt_ids, variant_prompt_ids)
        if stable_prefix_token_len <= 0:
            return {}

        stable_prefix_fingerprint = _short_hash(
            {
                "prefix_source": "uni-agent-system-prefix",
                "system_messages": system_messages,
                "tools": self.tools_schemas or [],
                "tokenizer": config.get("tokenizer_fingerprint", ""),
                "template": config.get("template_fingerprint", ""),
            }
        )
        return {
            "stable_prefix_token_len": stable_prefix_token_len,
            "stable_prefix_fingerprint": stable_prefix_fingerprint,
            "prefix_source": "uni-agent-system-prefix",
            "tokenizer_fingerprint": config.get("tokenizer_fingerprint"),
            "template_fingerprint": config.get("template_fingerprint"),
        }

    def _build_extra_prefix_cache_generate_kwargs(self, rollout_cache: dict[str, Any]) -> dict[str, Any]:
        metadata = rollout_cache.get("extra_prefix_cache") or {}
        if not metadata:
            return {}
        return {"extra_prefix_cache_metadata": metadata}

    def _build_system_prefix_variant_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        variant_messages: list[dict[str, Any]] = []
        replaced_first_user = False
        for message in messages:
            new_message = dict(message)
            if not replaced_first_user and new_message.get("role") == "user":
                new_message["content"] = ""
                replaced_first_user = True
            variant_messages.append(new_message)
        return variant_messages

    async def _get_new_message_ids(self, new_messages: list[dict[str, Any]]) -> list[int]:
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import normalize_token_ids

        tokenized_prompt = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.tokenizer,
                new_messages,
                add_generation_prompt=True,
                tokenize=True,
            ),
        )
        return self.message_boundary_tokens + normalize_token_ids(tokenized_prompt)

    @cached_property
    def message_boundary_tokens(self) -> list[int]:
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import normalize_token_ids

        dummy_history = [
            {"role": "user", "content": "dummy user"},
            {"role": "assistant", "content": "dummy assistant"},
        ]
        dummy_next_message = {"role": "user", "content": "dummy user"}

        try:
            standalone_ids = normalize_token_ids(
                apply_chat_template(
                    self.tokenizer,
                    [dummy_next_message],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            with_boundary_ids = normalize_token_ids(
                apply_chat_template(
                    self.tokenizer,
                    dummy_history + [dummy_next_message],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
        except Exception:
            return []

        if not standalone_ids or with_boundary_ids[-len(standalone_ids) :] != standalone_ids:
            return []

        text_before_message_ids = with_boundary_ids[: -len(standalone_ids)]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            return []

        for i in range(len(text_before_message_ids) - 1, -1, -1):
            if text_before_message_ids[i] == eos_id:
                return text_before_message_ids[i + 1 :]

        return []


def _extra_prefix_cache_enabled(config: Any) -> bool:
    if config is None:
        return False
    if isinstance(config, dict):
        return bool(config.get("enable", config.get("enabled", False)))
    try:
        return bool(config.get("enable", config.get("enabled", False)))
    except Exception:
        return bool(getattr(config, "enable", getattr(config, "enabled", False)))


def _compute_common_prefix_len(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _short_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


# this class is only used for Inference-Only Scenario
class OpenAICompatibleChatModel:
    base_url: str
    """OpenAI-compatible API base URL, for example http://127.0.0.1:8000/v1"""

    api_key: str
    """API key for the chat completion endpoint"""

    model_name: str
    """Model name sent to the OpenAI-compatible endpoint"""

    sampling_params: dict[str, Any]
    """Default sampling parameters passed to the endpoint"""

    timeout: int | float
    """HTTP timeout in seconds"""

    tools_schemas: list[dict] = None

    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
        if not hasattr(self, "sampling_params"):
            self.sampling_params = {}
        if not hasattr(self, "timeout"):
            self.timeout = 300
        self.base_url = self.base_url.rstrip("/")
        self.loop = get_event_loop()

        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Stateless: caller owns ``messages`` and re-passes them every
        :meth:`query`. Cache holds only metrics.
        """
        return {"metrics": {}}

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, Any]],
        rollout_cache: dict[str, Any] | None,
    ):
        """No-op; kept so :class:`AgentInteraction` can dispatch uniformly
        across training and inference paths.
        """
        return rollout_cache

    def _normalize_messages_for_api(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip locally-added fields the OpenAI API doesn't accept.
        Tool messages missing ``tool_call_id`` (format-error fallbacks)
        pass through as-is.
        """
        normalized_messages = []
        for message in messages:
            normalized_message = {"role": message["role"]}
            if message.get("content") is not None:
                normalized_message["content"] = message["content"]
            if message["role"] == "assistant" and message.get("tool_calls"):
                normalized_message["tool_calls"] = message["tool_calls"]
            if message["role"] == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id is not None:
                    normalized_message["tool_call_id"] = tool_call_id
                if message.get("name") is not None:
                    normalized_message["name"] = message["name"]
            normalized_messages.append(normalized_message)
        return normalized_messages

    # OpenAI ChatCompletion top-level sampling fields.
    _OPENAI_TOP_LEVEL_SAMPLING_FIELDS: frozenset[str] = frozenset(
        {
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "n",
            "seed",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "user",
        }
    )

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
        **kwargs,
    ) -> tuple[str, list[dict], dict[str, Any], dict[str, int]]:
        """Run one chat-completion call. Returns ``(text, tool_calls,
        rollout_cache, generation_info)``. ``tool_calls`` is the OpenAI
        ``{"id", "type", "function": {"name", "arguments"}}`` shape (one
        entry per parallel call; ``[]`` if the model returned plain text).
        """
        sampling_params = kwargs.get("sampling_params", self.sampling_params) or {}
        api_messages = self._normalize_messages_for_api(messages)

        top_level = {k: v for k, v in sampling_params.items() if k in self._OPENAI_TOP_LEVEL_SAMPLING_FIELDS}
        extra_body = {k: v for k, v in sampling_params.items() if k not in self._OPENAI_TOP_LEVEL_SAMPLING_FIELDS}

        with simple_timer("generate_sequences", rollout_cache["metrics"]):
            chat_completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                tools=self.tools_schemas,
                extra_body=extra_body or None,
                **top_level,
            )

        response_message = chat_completion.choices[0].message
        response_content = response_message.content or ""
        response_tool_calls = list(response_message.tool_calls or [])

        serialized_tool_calls: list[dict] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in response_tool_calls
        ]

        usage = chat_completion.usage
        completion_tokens = usage.completion_tokens if usage is not None else max(len(response_content.split()), 1)
        prompt_tokens = usage.prompt_tokens if usage is not None else 0
        return (
            response_content,
            serialized_tool_calls,
            rollout_cache,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
