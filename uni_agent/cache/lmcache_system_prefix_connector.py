import re
from typing import Any

from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector,
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)
from lmcache.utils import init_logger as lmcache_init_logger

logger = lmcache_init_logger(__name__)

_POLICY_RE = re.compile(r"^uaxpc__r(?P<read>[01])__w(?P<write>[01])__l(?P<limit>\d+)__")


class LMCacheSystemPrefixConnector(LMCacheMPConnector):
    """LMCache connector wrapper for uni-agent's system-prefix external cache.

    The wrapper keeps LMCache namespacing in vLLM's native ``cache_salt`` and
    carries per-request read/write policy in the vLLM request id. This keeps the
    policy in uni-agent while reusing LMCache's MP transport unchanged.
    """

    def _policy_from_request_id(self, request_id: str) -> dict[str, Any]:
        match = _POLICY_RE.match(request_id)
        if match is None:
            return {"read": True, "write": True, "store_token_limit": 0, "tagged": False}
        return {
            "read": match.group("read") == "1",
            "write": match.group("write") == "1",
            "store_token_limit": int(match.group("limit")),
            "tagged": True,
        }

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        policy = self._policy_from_request_id(request.request_id)
        if not policy["read"]:
            tracker = self._get_or_create_request_tracker(request)
            logger.info(
                "ExtraPrefixCache read deny request_id=%s cache_salt=%s",
                request.request_id,
                tracker.cache_salt,
            )
            return 0, False

        ret, will_load = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if ret is not None and policy["tagged"]:
            tracker = self._get_or_create_request_tracker(request)
            logger.info(
                "ExtraPrefixCache read allow request_id=%s cache_salt=%s new_external_tokens=%s will_load=%s",
                request.request_id,
                tracker.cache_salt,
                ret,
                will_load,
            )
        return ret, will_load

    def _process_new_requests(self, scheduler_output: "SchedulerOutput", metadata: LMCacheMPConnectorMetadata) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        for new_request in scheduler_output.scheduled_new_reqs:
            request_tracker = self._get_request_tracker(new_request.req_id)
            num_new_tokens = scheduler_output.num_scheduled_tokens[new_request.req_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)
            self._maybe_add_store_metadata(request_tracker, lmcache_tokens_per_chunk, metadata)

    def _process_cached_requests(self, scheduler_output: "SchedulerOutput", metadata: LMCacheMPConnectorMetadata) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._get_request_tracker(request_id)

            new_block_ids = cached_reqs.new_block_ids[idx] or ()
            if request_id not in cached_reqs.resumed_req_ids:
                request_tracker.append_block_ids(new_block_ids)

            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)
            self._maybe_add_store_metadata(request_tracker, lmcache_tokens_per_chunk, metadata)

    def _maybe_add_store_metadata(
        self,
        request_tracker: "LMCacheMPRequestTracker",
        lmcache_tokens_per_chunk: int,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
            request_tracker,
            lmcache_tokens_per_chunk,
            self._group_tokens_per_block,
        )
        policy = self._policy_from_request_id(request_tracker.request_id)
        if r_meta is None:
            if policy["tagged"] and policy["write"]:
                logger.info(
                    "ExtraPrefixCache write wait request_id=%s cache_salt=%s scheduled_tokens=%d stored_tokens=%d chunk_tokens=%d",
                    request_tracker.request_id,
                    request_tracker.cache_salt,
                    request_tracker.num_scheduled_tokens,
                    request_tracker.num_stored_tokens,
                    lmcache_tokens_per_chunk,
                )
            return

        if not policy["write"]:
            logger.info(
                "ExtraPrefixCache write deny request_id=%s cache_salt=%s range=%d-%d reason=policy",
                request_tracker.request_id,
                request_tracker.cache_salt,
                r_meta.op.start,
                r_meta.op.end,
            )
            return

        limit = policy["store_token_limit"]
        if limit > 0 and r_meta.op.end > limit:
            logger.info(
                "ExtraPrefixCache write deny request_id=%s cache_salt=%s range=%d-%d limit=%d",
                request_tracker.request_id,
                request_tracker.cache_salt,
                r_meta.op.start,
                r_meta.op.end,
                limit,
            )
            return

        metadata.add_request_metadata(r_meta)
        logger.info(
            "ExtraPrefixCache write allow request_id=%s cache_salt=%s range=%d-%d",
            request_tracker.request_id,
            request_tracker.cache_salt,
            r_meta.op.start,
            r_meta.op.end,
        )
