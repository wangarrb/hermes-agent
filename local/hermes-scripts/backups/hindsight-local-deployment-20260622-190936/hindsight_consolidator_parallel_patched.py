"""Consolidation engine for automatic observation creation from memories.

The consolidation engine runs as a background job after retain operations complete.
It processes new memories and either:
- Creates new observations from novel facts
- Updates existing observations when new evidence supports/contradicts/refines them

Observations are stored in memory_units with fact_type='observation' and include:
- proof_count: Number of supporting memories
- source_memory_ids: Array of memory UUIDs that contribute to this observation
- history: JSONB tracking changes over time

NOTE: Observations are distinct from mental models (pinned reflections).
- Observations: auto-generated bottom-up by this engine from raw facts (memory_units table, fact_type='observation')
- Mental models: user-defined queries stored in the mental_models table, refreshed on demand via reflect
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

from ...config import get_config
from ..db_utils import acquire_with_retry
from ..llm_wrapper import sanitize_llm_output
from ..memory_engine import Budget, fq_table
from ..retain import embedding_utils
from .prompts import build_batch_consolidation_prompt

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine
    from ..response_models import MemoryFact, RecallResult

logger = logging.getLogger(__name__)


def _read_positive_int_env(name: str, default: int, *, min_value: int = 1, max_value: int | None = None) -> int:
    """Read a positive integer env knob with conservative clamping."""
    raw = os.getenv(name)
    try:
        value = int(raw) if raw not in (None, "") else int(default)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        value = int(default)
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _read_non_negative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return max(0, int(raw) if raw not in (None, "") else int(default))
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return max(0, int(default))


def _is_rate_limit_error(exc: BaseException | None) -> bool:
    """Return True for provider 429/rate-limit exceptions, including wrapped errors."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status_code = getattr(cur, "status_code", None) or getattr(getattr(cur, "response", None), "status_code", None)
        if status_code == 429 or str(status_code) == "429":
            return True
        body = getattr(cur, "body", None)
        haystack = " ".join(str(x) for x in [type(cur).__name__, cur, body] if x is not None).lower()
        if any(
            marker in haystack
            for marker in [
                "http 429",
                "status_code=429",
                "status code: 429",
                "rate limit",
                "rate_limit",
                "too many requests",
                "throttling",
                "concurrency allocated quota exceeded",
            ]
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class AdaptiveLLMConcurrencyLimiter:
    """Dynamic async limiter for consolidation LLM calls.

    On the first 429 in a cooldown window, halve the live LLM concurrency and
    put new acquisitions to sleep.  If another 429 happens after the sleep
    window, halve again, down to 1.  In-flight calls are allowed to finish;
    future calls observe the lower limit.
    """

    def __init__(self, initial_limit: int, *, backoff_s: int = 300, name: str = "consolidation-llm") -> None:
        self.initial_limit = max(1, int(initial_limit))
        self.current_limit = self.initial_limit
        self.backoff_s = max(1, int(backoff_s))
        self.name = name
        self.active = 0
        self.rate_limit_events = 0
        self.cooldown_until = 0.0
        self._condition = asyncio.Condition()

    async def __aenter__(self) -> "AdaptiveLLMConcurrencyLimiter":
        async with self._condition:
            while True:
                now = time.time()
                cooldown_remaining = self.cooldown_until - now
                if cooldown_remaining > 0:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=cooldown_remaining)
                    except asyncio.TimeoutError:
                        pass
                    continue
                if self.active < self.current_limit:
                    self.active += 1
                    return self
                await self._condition.wait()

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        async with self._condition:
            self.active = max(0, self.active - 1)
            self._condition.notify_all()

    async def handle_rate_limit(self, *, bank_id: str, batch_label: str, exc: BaseException) -> None:
        async with self._condition:
            now = time.time()
            already_in_cooldown = self.cooldown_until > now
            old_limit = self.current_limit
            if not already_in_cooldown:
                self.current_limit = max(1, old_limit // 2 if old_limit > 1 else 1)
            self.cooldown_until = max(self.cooldown_until, now + self.backoff_s)
            self.rate_limit_events += 1
            cooldown_until = self.cooldown_until
            new_limit = self.current_limit
            self._condition.notify_all()

        if already_in_cooldown:
            logger.warning(
                "[CONSOLIDATION] bank=%s 429/rate-limit for %s while cooldown is active; "
                "keeping llm concurrency=%s, sleeping %.0fs. error=%s",
                bank_id,
                batch_label,
                new_limit,
                max(0.0, cooldown_until - time.time()),
                exc,
            )
        else:
            logger.warning(
                "[CONSOLIDATION] bank=%s 429/rate-limit for %s; sleeping %ss and reducing "
                "LLM concurrency %s -> %s. error=%s",
                bank_id,
                batch_label,
                self.backoff_s,
                old_limit,
                new_limit,
                exc,
            )
        await asyncio.sleep(max(0.0, cooldown_until - time.time()))



def _consolidation_parallel_limits(config: Any) -> tuple[int, int, int]:
    """Return (recall_limit, llm_limit, batch_parallel_limit).

    Recall/search fanout and LLM provider concurrency are intentionally separate:
    a small LLM batch size can keep recall fanout bounded while still allowing the
    provider to run more structured calls concurrently.
    """
    llm_default = getattr(config, "consolidation_llm_max_concurrent", None) or 1
    llm_limit = _read_positive_int_env(
        "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT",
        int(llm_default),
        min_value=1,
    )
    recall_limit = _read_positive_int_env(
        "HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT",
        60,
        min_value=1,
    )
    batch_parallel_limit = _read_positive_int_env(
        "HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES",
        llm_limit,
        min_value=1,
    )
    return recall_limit, llm_limit, batch_parallel_limit


async def _filter_live_source_memories(
    conn: "Connection",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    """Return only the source memory ids that still exist in the bank.

    Uses FOR SHARE to block concurrent deletes from removing a row between the
    check and the subsequent insert/update. Combined with the delete path running
    its stale-observation sweep *after* deleting the source row, this closes the
    race window where consolidation would otherwise produce an orphan observation.

    Oracle note: Oracle doesn't support FOR SHARE, so the SQL rewriter promotes
    it to FOR UPDATE. Oracle's MVCC consistent-read semantics make FOR SHARE
    unnecessary (the sweep runs AFTER deletion), but FOR UPDATE is more
    conservative and still correct.
    """
    if not source_memory_ids:
        return []
    rows = await conn.fetch(
        f"""
        SELECT id
        FROM {fq_table("memory_units")}
        WHERE id = ANY($1::uuid[]) AND bank_id = $2
        FOR SHARE
        """,
        source_memory_ids,
        bank_id,
    )
    live = {row["id"] for row in rows}
    return [mid for mid in source_memory_ids if mid in live]


class _CreateAction(BaseModel):
    text: str
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""


class _UpdateAction(BaseModel):
    text: str
    observation_id: str  # UUID of the existing observation to update
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""


class _DeleteAction(BaseModel):
    observation_id: str  # UUID of the observation to remove


class _ConsolidationBatchResponse(BaseModel):
    creates: list[_CreateAction] = []
    updates: list[_UpdateAction] = []
    deletes: list[_DeleteAction] = []


@dataclass
class _BatchLLMResult:
    creates: list[_CreateAction] = field(default_factory=list)
    updates: list[_UpdateAction] = field(default_factory=list)
    deletes: list[_DeleteAction] = field(default_factory=list)
    obs_count: int = 0
    prompt_chars: int = 0
    failed: bool = False


@dataclass
class _SourceAggregation:
    """Fields inherited by an observation from its source memories."""

    event_date: datetime | None
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None
    tags: list[str]


def _aggregate_source_fields(source_mems: list[dict[str, Any]], tags: list[str] | None = None) -> _SourceAggregation:
    """Compute the observation fields inherited from a set of source memories.

    Temporal aggregation rules:
    - ``event_date``    — earliest across sources (min)
    - ``occurred_start`` — earliest across sources (min)
    - ``occurred_end``   — latest across sources (max)
    - ``mentioned_at``   — latest across sources (max)

    Fields remain ``None`` when no source memory carries that information, so
    observations are never stamped with an artificial timestamp.

    ``tags`` defaults to those of the first source memory when not explicitly
    provided (all memories in a consolidation batch share the same tag set).
    """
    effective_tags = tags if tags is not None else (source_mems[0].get("tags") or [] if source_mems else [])
    return _SourceAggregation(
        event_date=_min_date(m.get("event_date") for m in source_mems),
        occurred_start=_min_date(m.get("occurred_start") for m in source_mems),
        occurred_end=_max_date(m.get("occurred_end") for m in source_mems),
        mentioned_at=_max_date(m.get("mentioned_at") for m in source_mems),
        tags=effective_tags,
    )


async def _count_observations_for_scope(
    conn: "Connection",
    bank_id: str,
    tags: list[str],
) -> int:
    """Count existing observations matching the given tag scope.

    Returns the count of observations whose tags contain all specified tags.
    Observations with no tags are not counted (the limit does not apply to them).
    """
    return await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
        f"WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
        bank_id,
        tags,
    )


def _build_response_model(max_creates: int | None = None) -> type[_ConsolidationBatchResponse]:
    """Build a response model, optionally constraining max creates via JSON schema."""
    if max_creates is None or max_creates < 0:
        return _ConsolidationBatchResponse

    from pydantic import Field as PydanticField

    clamped = max(max_creates, 0)

    class _ConstrainedConsolidationBatchResponse(_ConsolidationBatchResponse):
        creates: list[_CreateAction] = PydanticField(default=[], max_length=clamped)

    return _ConstrainedConsolidationBatchResponse


class ConsolidationPerfLog:
    """Performance logging for consolidation operations."""

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        self.start_time = time.time()
        self.lines: list[str] = []
        self.timings: dict[str, float] = {}
        self.llm_calls: int = 0
        self.total_obs_in_context: int = 0
        self.total_prompt_chars: int = 0

    def log(self, message: str) -> None:
        """Add a log line."""
        self.lines.append(message)

    def record_timing(self, key: str, duration: float) -> None:
        """Record a timing measurement."""
        if key in self.timings:
            self.timings[key] += duration
        else:
            self.timings[key] = duration

    def record_llm_call(self, obs_count: int, prompt_chars: int) -> None:
        """Record stats for a single LLM call."""
        self.llm_calls += 1
        self.total_obs_in_context += obs_count
        self.total_prompt_chars += prompt_chars

    def flush(self) -> None:
        """Flush all log lines to the logger."""
        total_time = time.time() - self.start_time
        header = f"\n{'=' * 60}\nCONSOLIDATION for bank {self.bank_id}"
        footer = f"{'=' * 60}\nCONSOLIDATION COMPLETE: {total_time:.3f}s total\n{'=' * 60}"

        log_output = header + "\n" + "\n".join(self.lines) + "\n" + footer
        logger.info(log_output)


async def run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    operation_id: str | None = None,
    observation_scopes: str | list | None = None,
) -> dict[str, Any]:
    """
    Run consolidation job for a bank.

    This is called after retain operations to consolidate new memories into mental models.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication

    Returns:
        Dict with consolidation results
    """
    # Resolve bank-specific config with hierarchical overrides
    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # Build a configured LLM wrapper that applies per-bank settings (e.g. safety settings)
    # to every call without leaking across operations.
    llm_config = memory_engine._consolidation_llm_config.with_config(config)

    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size
    max_memories_per_round = config.consolidation_max_memories_per_round
    llm_batch_size = max(1, config.consolidation_llm_batch_size)
    recall_limit, llm_limit, batch_parallel_limit = _consolidation_parallel_limits(config)
    rate_limit_backoff_s = _read_positive_int_env(
        "HINDSIGHT_API_CONSOLIDATION_429_BACKOFF_SECONDS",
        _read_positive_int_env("HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS", 300, min_value=1),
        min_value=1,
    )
    recall_semaphore = asyncio.Semaphore(recall_limit)
    llm_semaphore = AdaptiveLLMConcurrencyLimiter(
        llm_limit,
        backoff_s=rate_limit_backoff_s,
        name=f"consolidation-llm:{bank_id}",
    )
    write_lock = asyncio.Lock()
    perf_lock = asyncio.Lock()

    # Check if consolidation is enabled
    if not config.enable_observations:
        logger.debug(f"Consolidation disabled for bank {bank_id}")
        return {"status": "disabled", "bank_id": bank_id}

    pool = memory_engine._backend

    # Get bank profile
    async with acquire_with_retry(pool) as conn:
        t0 = time.time()
        bank_row = await conn.fetchrow(
            f"""
            SELECT bank_id, name
            FROM {fq_table("banks")}
            WHERE bank_id = $1
            """,
            bank_id,
        )

        if not bank_row:
            logger.warning(f"Bank {bank_id} not found for consolidation")
            return {"status": "bank_not_found", "bank_id": bank_id}

        perf.record_timing("fetch_bank", time.time() - t0)

        # Count total unconsolidated memories for progress logging
        total_count = await conn.fetchval(
            f"""
            SELECT COUNT(*)
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND consolidated_at IS NULL
              AND consolidation_failed_at IS NULL
              AND fact_type IN ('experience', 'world')
            """,
            bank_id,
        )

    if total_count == 0:
        logger.debug(f"No new memories to consolidate for bank {bank_id}")
        return {"status": "no_new_memories", "bank_id": bank_id, "memories_processed": 0}

    logger.info(
        f"[CONSOLIDATION] bank={bank_id} total_unconsolidated={total_count} "
        f"parallel_batches={batch_parallel_limit} llm_limit={llm_limit} recall_limit={recall_limit} "
        f"llm_batch_size={llm_batch_size} rate_limit_backoff={rate_limit_backoff_s}s"
    )
    perf.log(f"[1] Found {total_count} pending memories to consolidate")

    # Process each memory with individual commits for crash recovery
    stats: dict[str, int] = {
        "memories_processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "observations_deleted": 0,
        "actions_executed": 0,
        "skipped": 0,
        "memories_failed": 0,
    }

    # Track all unique tags from consolidated memories for mental model refresh filtering
    consolidated_tags: set[str] = set()

    round_limit_enabled = max_memories_per_round > 0
    round_remaining = max_memories_per_round if round_limit_enabled else float("inf")
    hit_round_limit = False

    async def _prepare_llm_batches(
        pool: Any,
        fetch_limit: int,
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        """Fetch unconsolidated memories and group them into LLM-safe batches.

        Returns (memories, batches).  memories is empty when there are no more
        unconsolidated rows.
        """
        async with acquire_with_retry(pool) as conn:
            t0 = time.time()
            memories = await conn.fetch(
                f"""
                SELECT id, text, fact_type, occurred_start, occurred_end, event_date, tags, mentioned_at,
                       observation_scopes
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND consolidated_at IS NULL
                  AND consolidation_failed_at IS NULL
                  AND fact_type IN ('experience', 'world')
                ORDER BY created_at ASC
                LIMIT $2
                """,
                bank_id,
                fetch_limit,
            )
            perf.record_timing("fetch_memories", time.time() - t0)

        if not memories:
            return [], []

        tag_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for m in memories:
            tag_key = tuple(sorted(m.get("tags") or []))
            tag_groups.setdefault(tag_key, []).append(dict(m))

        batches: list[list[dict[str, Any]]] = []
        for group in tag_groups.values():
            for i in range(0, len(group), llm_batch_size):
                batches.append(group[i : i + llm_batch_size])
        return memories, batches

    llm_batch_num = 0
    # Pre-fetch pipeline: while the current round's LLM batches are running,
    # fetch the next round's memories in the background so they are ready when
    # the current round finishes.
    next_prefetch: asyncio.Task | None = None
    while True:
        # Cap fetch size by remaining round budget
        fetch_limit = (
            min(max_memories_per_batch, int(round_remaining)) if round_limit_enabled else max_memories_per_batch
        )

        # If a background fetch is pending, collect its result now.
        if next_prefetch is not None:
            memories, llm_batches = await next_prefetch
            next_prefetch = None
        else:
            memories, llm_batches = await _prepare_llm_batches(pool, fetch_limit)

        if not memories:
            break  # No more unconsolidated memories

        async def _process_one_llm_batch(llm_batch_num_for_log: int, llm_batch: list[dict[str, Any]]) -> dict[str, Any]:
            llm_batch_start = time.time()
            batch_perf = ConsolidationPerfLog(bank_id)

            all_results: list[dict[str, Any]] = []
            all_deleted = 0
            succeeded_ids: list[Any] = []
            failed_ids: list[Any] = []

            pending: list[list[dict[str, Any]]] = [llm_batch]
            while pending:
                sub_batch = pending.pop(0)

                # Determine observation_scopes for this sub-batch. All memories share
                # the same tags (enforced by tag_groups), so we only check the first memory.
                # asyncpg returns JSONB columns as raw JSON strings, so parse if needed.
                _obs_raw = sub_batch[0].get("observation_scopes") if sub_batch else None
                _obs_parsed = json.loads(_obs_raw) if isinstance(_obs_raw, str) else _obs_raw

                # Resolve the scope spec into a concrete list[list[str]] (or None for combined).
                if _obs_parsed == "per_tag":
                    _memory_tags = sub_batch[0].get("tags") or []
                    obs_tags_list = [[tag] for tag in _memory_tags] if _memory_tags else None
                elif _obs_parsed == "all_combinations":
                    _memory_tags = sub_batch[0].get("tags") or []
                    obs_tags_list = (
                        [
                            list(combo)
                            for r in range(1, len(_memory_tags) + 1)
                            for combo in combinations(_memory_tags, r)
                        ]
                        if _memory_tags
                        else None
                    )
                elif _obs_parsed == "combined" or _obs_parsed is None:
                    obs_tags_list = None  # single combined pass (default behaviour)
                else:
                    # explicit list[list[str]]
                    obs_tags_list = _obs_parsed

                sub_deleted: int = 0
                sub_llm_failed = False
                if obs_tags_list:
                    # Multi-pass: run one observation consolidation pass per tag set.
                    # Each pass gets its own connection. Recall and LLM are bounded by
                    # semaphores; writes are serialized by write_lock inside
                    # _process_memory_batch.
                    sub_results: list[dict[str, Any]] = []
                    for obs_tags in obs_tags_list:
                        async with acquire_with_retry(pool) as conn:
                            pass_results, pass_deleted, pass_failed = await _process_memory_batch(
                                conn=conn,
                                memory_engine=memory_engine,
                                llm_config=llm_config,
                                bank_id=bank_id,
                                memories=sub_batch,
                                request_context=request_context,
                                perf=batch_perf,
                                config=config,
                                obs_tags_override=obs_tags,
                                recall_semaphore=recall_semaphore,
                                llm_semaphore=llm_semaphore,
                                write_lock=write_lock,
                            )
                        sub_deleted += pass_deleted
                        sub_llm_failed = sub_llm_failed or pass_failed
                        # Merge results: prefer non-skipped actions
                        if not sub_results:
                            sub_results = pass_results
                        else:
                            for i, (existing, new) in enumerate(zip(sub_results, pass_results)):
                                if existing.get("action") == "skipped" and new.get("action") != "skipped":
                                    sub_results[i] = new
                                elif existing.get("action") != "skipped" and new.get("action") != "skipped":
                                    # Both did something — combine into "multiple"
                                    existing_created = existing.get(
                                        "created", 1 if existing.get("action") == "created" else 0
                                    )
                                    existing_updated = existing.get(
                                        "updated", 1 if existing.get("action") == "updated" else 0
                                    )
                                    new_created = new.get("created", 1 if new.get("action") == "created" else 0)
                                    new_updated = new.get("updated", 1 if new.get("action") == "updated" else 0)
                                    total = existing_created + existing_updated + new_created + new_updated
                                    sub_results[i] = {
                                        "action": "multiple",
                                        "created": existing_created + new_created,
                                        "updated": existing_updated + new_updated,
                                        "merged": 0,
                                        "total_actions": total,
                                    }
                else:
                    # Normal single pass using the memory's own tags.
                    async with acquire_with_retry(pool) as conn:
                        sub_results, sub_deleted, sub_llm_failed = await _process_memory_batch(
                            conn=conn,
                            memory_engine=memory_engine,
                            llm_config=llm_config,
                            bank_id=bank_id,
                            memories=sub_batch,
                            request_context=request_context,
                            perf=batch_perf,
                            config=config,
                            recall_semaphore=recall_semaphore,
                            llm_semaphore=llm_semaphore,
                            write_lock=write_lock,
                        )

                all_deleted += sub_deleted

                if sub_llm_failed and len(sub_batch) > 1:
                    # Split and retry with smaller batches
                    mid = len(sub_batch) // 2
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} LLM failed for sub-batch of {len(sub_batch)},"
                        f" splitting into {mid}/{len(sub_batch) - mid}"
                    )
                    pending[0:0] = [sub_batch[:mid], sub_batch[mid:]]
                elif sub_llm_failed:
                    # batch_size=1 and still failing — mark as permanently failed for now
                    failed_ids.append(sub_batch[0]["id"])
                    all_results.append({"action": "failed"})
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} LLM failed for single memory"
                        f" {sub_batch[0]['id']}, marking consolidation_failed_at"
                    )
                else:
                    succeeded_ids.extend(m["id"] for m in sub_batch)
                    all_results.extend(sub_results)

            # Commit consolidated_at / consolidation_failed_at in a single DB round-trip.
            # This is a write path, so serialize it with observation writes.
            async with write_lock:
                async with acquire_with_retry(pool) as conn:
                    if succeeded_ids:
                        await conn.executemany(
                            f"UPDATE {fq_table('memory_units')} SET consolidated_at = NOW() WHERE id = $1",
                            [(mem_id,) for mem_id in succeeded_ids],
                        )
                    if failed_ids:
                        await conn.executemany(
                            f"UPDATE {fq_table('memory_units')} SET consolidation_failed_at = NOW() WHERE id = $1",
                            [(mem_id,) for mem_id in failed_ids],
                        )

            # Aggregate perf into the job-level perf after the batch finishes so
            # concurrent batches do not corrupt per-batch delta logging.
            async with perf_lock:
                for key, duration in batch_perf.timings.items():
                    perf.record_timing(key, duration)
                perf.llm_calls += batch_perf.llm_calls
                perf.total_obs_in_context += batch_perf.total_obs_in_context
                perf.total_prompt_chars += batch_perf.total_prompt_chars

            # Checkpoint: abort if the operation (and thus the bank) was deleted mid-run.
            if operation_id and not await memory_engine._check_op_alive(operation_id):
                logger.info(
                    f"[CONSOLIDATION] bank={bank_id} operation {operation_id} cancelled (bank deleted), stopping early"
                )
                return {"cancelled": True, "llm_batch_num": llm_batch_num_for_log}

            batch_stats = {
                "observations_deleted": all_deleted,
                "observations_created": 0,
                "observations_updated": 0,
                "observations_merged": 0,
                "actions_executed": 0,
                "skipped": 0,
                "memories_failed": 0,
                "memories_processed": 0,
            }

            for result in all_results:
                batch_stats["memories_processed"] += 1
                action = result.get("action")
                if action == "created":
                    batch_stats["observations_created"] += 1
                    batch_stats["actions_executed"] += 1
                elif action == "updated":
                    batch_stats["observations_updated"] += 1
                    batch_stats["actions_executed"] += 1
                elif action == "merged":
                    batch_stats["observations_merged"] += 1
                    batch_stats["actions_executed"] += 1
                elif action == "multiple":
                    batch_stats["observations_created"] += result.get("created", 0)
                    batch_stats["observations_updated"] += result.get("updated", 0)
                    batch_stats["observations_merged"] += result.get("merged", 0)
                    batch_stats["actions_executed"] += result.get("total_actions", 0)
                elif action == "skipped":
                    batch_stats["skipped"] += 1
                elif action == "failed":
                    batch_stats["memories_failed"] += 1

            return {
                "cancelled": False,
                "llm_batch_num": llm_batch_num_for_log,
                "memories_in_batch": len(llm_batch),
                "llm_calls": batch_perf.llm_calls,
                "timings": batch_perf.timings,
                "input_tokens": int(batch_perf.total_prompt_chars / 4),
                "elapsed": time.time() - llm_batch_start,
                "stats": batch_stats,
            }

        async def _run_llm_batch_wave(tasks: list[asyncio.Task[dict[str, Any]]]) -> None:
            nonlocal stats
            for batch_result in await asyncio.gather(*tasks):
                if batch_result.get("cancelled"):
                    stats["_cancelled"] = 1
                    continue
                batch_stats = batch_result["stats"]
                for key, value in batch_stats.items():
                    stats[key] += value
                timing_parts = [
                    f"{key}={batch_result['timings'][key]:.3f}s"
                    for key in ["recall", "llm", "embedding", "db_write"]
                    if key in batch_result["timings"]
                ]
                logger.info(
                    f"[CONSOLIDATION] bank={bank_id} llm_batch #{batch_result['llm_batch_num']}"
                    f" ({batch_result['memories_in_batch']} memories, {batch_result['llm_calls']} llm calls)"
                    f" | {stats['memories_processed']}/{total_count} processed"
                    f" | {', '.join(timing_parts)}"
                    f" | created={batch_stats['observations_created']} updated={batch_stats['observations_updated']}"
                    f" skipped={batch_stats['skipped']}"
                    + (f" failed={batch_stats['memories_failed']}" if batch_stats["memories_failed"] else "")
                    + f" | input_tokens=~{batch_result['input_tokens']}"
                    f" | elapsed={batch_result['elapsed']:.3f}s"
                    f" | limits: batch_parallel={batch_parallel_limit} llm={getattr(llm_semaphore, 'current_limit', llm_limit)}/{llm_limit} recall={recall_limit}"
                )

        running_tasks: list[asyncio.Task[dict[str, Any]]] = []
        for llm_batch in llm_batches:
            llm_batch_num += 1
            running_tasks.append(asyncio.create_task(_process_one_llm_batch(llm_batch_num, llm_batch)))
            if len(running_tasks) >= batch_parallel_limit:
                # Kick off next round's DB fetch in the background while the
                # current wave of LLM batches is running.  Because the fetch uses
                # a separate connection from the pool, it overlaps cleanly with
                # LLM work.
                if round_limit_enabled and round_remaining > len(memories) and next_prefetch is None:
                    next_fetch_limit = (
                        min(max_memories_per_batch, int(round_remaining - len(memories)))
                        if round_limit_enabled
                        else max_memories_per_batch
                    )
                    next_prefetch = asyncio.create_task(_prepare_llm_batches(pool, next_fetch_limit))
                await _run_llm_batch_wave(running_tasks)
                running_tasks = []
                if stats.get("_cancelled"):
                    if next_prefetch:
                        next_prefetch.cancel()
                    return {"status": "cancelled", "bank_id": bank_id, **stats}

        if running_tasks:
            # Trailing wave — may still be followed by more rounds, so kick off
            # a background pre-fetch if the round budget permits.
            if round_limit_enabled and round_remaining > len(memories) and next_prefetch is None:
                next_fetch_limit = (
                    min(max_memories_per_batch, int(round_remaining - len(memories)))
                    if round_limit_enabled
                    else max_memories_per_batch
                )
                next_prefetch = asyncio.create_task(_prepare_llm_batches(pool, next_fetch_limit))
            await _run_llm_batch_wave(running_tasks)
            if stats.get("_cancelled"):
                if next_prefetch:
                    next_prefetch.cancel()
                return {"status": "cancelled", "bank_id": bank_id, **stats}

        # Update round budget after processing this DB fetch batch
        if round_limit_enabled:
            round_remaining -= len(memories)
            if round_remaining <= 0:
                hit_round_limit = True
                if next_prefetch:
                    next_prefetch.cancel()
                break

    # Re-submit consolidation if we hit the round limit and there's likely more work
    if hit_round_limit:
        remaining = total_count - stats["memories_processed"]
        logger.info(
            f"[CONSOLIDATION] bank={bank_id} hit round limit of {max_memories_per_round} memories,"
            f" ~{remaining} remaining. Re-queuing consolidation."
        )
        try:
            await memory_engine.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] bank={bank_id} failed to re-queue consolidation: {e}")

    # Build summary
    perf.log(
        f"[3] Results: {stats['memories_processed']} memories -> "
        f"{stats['actions_executed']} actions "
        f"({stats['observations_created']} created, "
        f"{stats['observations_updated']} updated, "
        f"{stats['observations_merged']} merged, "
        f"{stats['skipped']} skipped)"
    )

    # Add timing breakdown
    timing_parts = []
    if "recall" in perf.timings:
        timing_parts.append(f"recall={perf.timings['recall']:.3f}s")
    if "llm" in perf.timings:
        timing_parts.append(f"llm={perf.timings['llm']:.3f}s")
    if "embedding" in perf.timings:
        timing_parts.append(f"embedding={perf.timings['embedding']:.3f}s")
    if "db_write" in perf.timings:
        timing_parts.append(f"db_write={perf.timings['db_write']:.3f}s")

    if perf.llm_calls > 0:
        timing_parts.append(f"avg_obs={perf.total_obs_in_context / perf.llm_calls:.1f}")
        timing_parts.append(f"avg_prompt_tokens=~{perf.total_prompt_chars / perf.llm_calls / 4:.0f}")

    if timing_parts:
        perf.log(f"[4] Timing breakdown: {', '.join(timing_parts)}")

    # Trigger mental model refreshes only on the final round (when all memories are processed).
    # If we hit the round limit and re-queued, skip MM refresh — the next round will handle it.
    if hit_round_limit:
        stats["mental_models_refreshed"] = 0
        logger.info(f"[CONSOLIDATION] bank={bank_id} skipping mental model refresh (round limit hit, re-queued)")
    else:
        # SECURITY: Only refresh mental models with matching tags (or all if no tags were consolidated)
        mental_models_refreshed = await _trigger_mental_model_refreshes(
            memory_engine=memory_engine,
            bank_id=bank_id,
            request_context=request_context,
            consolidated_tags=list(consolidated_tags) if consolidated_tags else None,
            perf=perf,
        )
        stats["mental_models_refreshed"] = mental_models_refreshed

    perf.flush()

    return {"status": "completed", "bank_id": bank_id, **stats}


async def _trigger_mental_model_refreshes(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    consolidated_tags: list[str] | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> int:
    """
    Trigger refreshes for mental models with refresh_after_consolidation=true.

    SECURITY: Only triggers refresh for mental models whose tags overlap with the
    consolidated memory tags, preventing unnecessary refreshes across security boundaries.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        consolidated_tags: Tags from memories that were consolidated (None = refresh all)
        perf: Performance logging

    Returns:
        Number of mental models scheduled for refresh
    """
    pool = memory_engine._backend

    # Find mental models with refresh_after_consolidation=true that are actually stale.
    # The tag filter on the SELECT enforces the security boundary (never look outside the
    # relevant tag scope); compute_mental_model_is_stale then verifies that new memories
    # in the MM's scope really were ingested since its last refresh.
    async with acquire_with_retry(pool) as conn:
        if consolidated_tags:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (
                    (tags IS NOT NULL AND tags != '{{}}' AND tags && $2::varchar[])
                    OR (tags IS NULL OR tags = '{{}}')
                  )
                """,
                bank_id,
                consolidated_tags,
            )
        else:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (tags IS NULL OR tags = '{{}}')
                """,
                bank_id,
            )

        rows = []
        for candidate in candidates:
            if await memory_engine.compute_mental_model_is_stale(conn, bank_id, candidate):
                rows.append(candidate)

    if not rows:
        return 0

    if perf:
        if consolidated_tags:
            perf.log(
                f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true "
                f"(filtered by tags: {consolidated_tags})"
            )
        else:
            perf.log(f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true")

    # Submit refresh tasks for each mental model
    refreshed_count = 0
    for row in rows:
        mental_model_id = row["id"]
        try:
            await memory_engine.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            refreshed_count += 1
            logger.info(
                f"[CONSOLIDATION] Triggered refresh for mental model {mental_model_id} "
                f"(name: {row['name']}) in bank {bank_id}"
            )
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] Failed to trigger refresh for mental model {mental_model_id}: {e}")

    return refreshed_count


async def _process_memory_batch(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    llm_config: Any,
    bank_id: str,
    memories: list[dict[str, Any]],
    request_context: "RequestContext",
    perf: ConsolidationPerfLog | None = None,
    config: Any = None,
    obs_tags_override: list[str] | None = None,
    recall_semaphore: asyncio.Semaphore | None = None,
    llm_semaphore: asyncio.Semaphore | None = None,
    write_lock: asyncio.Lock | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """
    Process a batch of memories in a single LLM call.

    Steps:
    1. Parallel recalls — one per fact (read-only; safe to parallelise)
    2. Union of retrieved observations across the batch (deduped by id)
    3. Single LLM call with all N facts + unioned observations
    4. Sequential action execution (writes remain serial for consistency)
    5. Returns one result dict per memory, in the same order as `memories`

    Per-fact security: action execution validates each learning_id against the
    observations that were recalled specifically for that fact, so cross-tag
    updates cannot occur.

    Args:
        obs_tags_override: When set, use these tags for observation recall and
            create/update instead of the memory's own tags. This enables multi-pass
            consolidation where a single memory can contribute to observations
            scoped at different tag levels (e.g., user-level vs session-level).
    """
    import asyncio

    # 1. Parallel recalls — one per fact
    # When obs_tags_override is set, use it as the observation scope for all facts.
    t0 = time.time()
    observation_scope_tags = obs_tags_override if obs_tags_override is not None else None
    async def _bounded_recall(m: dict[str, Any]) -> "RecallResult":
        if recall_semaphore is None:
            return await _find_related_observations(
                memory_engine=memory_engine,
                bank_id=bank_id,
                query=m["text"],
                request_context=request_context,
                tags=observation_scope_tags if observation_scope_tags is not None else (m.get("tags") or []),
            )
        async with recall_semaphore:
            return await _find_related_observations(
                memory_engine=memory_engine,
                bank_id=bank_id,
                query=m["text"],
                request_context=request_context,
                tags=observation_scope_tags if observation_scope_tags is not None else (m.get("tags") or []),
            )

    recall_tasks = [_bounded_recall(m) for m in memories]
    per_fact_recalls = await asyncio.gather(*recall_tasks)
    if perf:
        perf.record_timing("recall", time.time() - t0)

    # 2. Build per-fact observation sets (keyed by memory ID string) for secure action validation
    per_fact_obs_ids: dict[str, set[str]] = {
        str(memories[i]["id"]): {str(obs.id) for obs in r.results} for i, r in enumerate(per_fact_recalls)
    }

    # Union all observations (deduped by id)
    seen_ids: set[str] = set()
    union_observations: list["MemoryFact"] = []
    union_source_facts: dict[str, "MemoryFact"] = {}
    for recall_result in per_fact_recalls:
        for obs in recall_result.results:
            obs_id = str(obs.id)
            if obs_id not in seen_ids:
                seen_ids.add(obs_id)
                union_observations.append(obs)
        if recall_result.source_facts:
            union_source_facts.update(recall_result.source_facts)

    # Determine effective tag scope for observations.
    # When obs_tags_override is set, use it; otherwise use the memory's own tags.
    if obs_tags_override is not None:
        fact_tags = obs_tags_override
    else:
        # All memories in the batch share the same tag set (enforced by batching)
        fact_tags = memories[0].get("tags") or [] if memories else []

    # 2b. Compute remaining observation slots for this scope (if limit configured)
    max_obs = config.max_observations_per_scope if config is not None else -1
    remaining_observation_slots: int | None = None
    if max_obs > 0 and fact_tags:
        current_count = await _count_observations_for_scope(conn, bank_id, fact_tags)
        remaining_observation_slots = max(max_obs - current_count, 0)
        if remaining_observation_slots == 0:
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} scope={fact_tags} at observation limit "
                f"({current_count}/{max_obs}), only updates/deletes allowed"
            )

    # 3. Single LLM call.  The adaptive limiter is passed down so 429 handling
    # can release the active slot before sleeping/backing off.
    t0 = time.time()
    llm_result = await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=memories,
        union_observations=union_observations,
        union_source_facts=union_source_facts,
        config=config,
        remaining_observation_slots=remaining_observation_slots,
        max_observations_per_scope=max_obs,
        llm_semaphore=llm_semaphore,
        bank_id=bank_id,
    )
    if perf:
        perf.record_timing("llm", time.time() - t0)
        perf.record_llm_call(llm_result.obs_count, llm_result.prompt_chars)

    async def _apply_llm_actions() -> tuple[list[dict[str, Any]], int]:
        # 4. Sequential execution of deletes / updates / creates
        # Deletes run first to free observation slots before creates consume them.
        # Track which memory indices participated so we can build per-memory results for stats
        per_memory_created: set[str] = set()
        per_memory_updated: set[str] = set()

        mem_by_id = {str(m["id"]): m for m in memories}

        # Execute deletes first to free observation slots before creates consume them
        deleted_count = 0
        for delete in llm_result.deletes:
            # Security: the observation must be present in the unioned recall
            if not any(str(obs.id) == delete.observation_id for obs in union_observations):
                logger.debug(
                    f"Batch consolidation: rejected delete — observation {delete.observation_id} not in unioned recall"
                )
                continue
            await _execute_delete_action(conn=conn, bank_id=bank_id, observation_id=delete.observation_id)
            deleted_count += 1

        for update in llm_result.updates:
            source_mems = [mem_by_id[fid] for fid in update.source_fact_ids if fid in mem_by_id]
            if not source_mems:
                continue
            # Security: the observation must have been recalled for at least one of the source facts
            if not any(update.observation_id in per_fact_obs_ids.get(str(m["id"]), set()) for m in source_mems):
                logger.debug(
                    f"Batch consolidation: rejected update — observation {update.observation_id} "
                    f"not in any source fact's recall"
                )
                continue
            agg = _aggregate_source_fields(source_mems, tags=fact_tags)
            await _execute_update_action(
                conn=conn,
                memory_engine=memory_engine,
                bank_id=bank_id,
                source_memory_ids=[m["id"] for m in source_mems],
                observation_id=update.observation_id,
                new_text=update.text,
                observations=union_observations,
                source_fact_tags=agg.tags,
                source_occurred_start=agg.occurred_start,
                source_occurred_end=agg.occurred_end,
                source_mentioned_at=agg.mentioned_at,
                perf=perf,
            )
            for m in source_mems:
                per_memory_updated.add(str(m["id"]))

        for create in llm_result.creates:
            source_mems = [mem_by_id[fid] for fid in create.source_fact_ids if fid in mem_by_id]
            if not source_mems:
                continue
            agg = _aggregate_source_fields(source_mems, tags=fact_tags)
            await _execute_create_action(
                conn=conn,
                memory_engine=memory_engine,
                bank_id=bank_id,
                source_memory_ids=[m["id"] for m in source_mems],
                text=create.text,
                source_fact_tags=agg.tags,
                event_date=agg.event_date,
                occurred_start=agg.occurred_start,
                occurred_end=agg.occurred_end,
                mentioned_at=agg.mentioned_at,
                perf=perf,
            )
            for m in source_mems:
                per_memory_created.add(str(m["id"]))

        # Build per-memory result dicts for the stats tracker in the outer loop
        results: list[dict[str, Any]] = []
        for m in memories:
            mid = str(m["id"])
            created = mid in per_memory_created
            updated = mid in per_memory_updated
            if created and updated:
                results.append({"action": "multiple", "created": 1, "updated": 1, "merged": 0, "total_actions": 2})
            elif created:
                results.append({"action": "created"})
            elif updated:
                results.append({"action": "updated"})
            else:
                results.append({"action": "skipped", "reason": "no_durable_knowledge"})

        return results, deleted_count

    if write_lock is None:
        results, deleted_count = await _apply_llm_actions()
    else:
        async with write_lock:
            results, deleted_count = await _apply_llm_actions()

    return results, deleted_count, llm_result.failed


def _min_date(dates: "Any") -> "datetime | None":
    """Return the minimum non-None datetime from an iterable."""
    return min((d for d in dates if d is not None), default=None)


def _max_date(dates: "Any") -> "datetime | None":
    """Return the maximum non-None datetime from an iterable."""
    return max((d for d in dates if d is not None), default=None)


async def _execute_update_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_id: str,
    new_text: str,
    observations: list["MemoryFact"],
    source_fact_tags: list[str] | None = None,
    source_occurred_start: datetime | None = None,
    source_occurred_end: datetime | None = None,
    source_mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> None:
    """
    Update an existing observation.

    Extends source_memory_ids with all contributing memories, updates temporal fields
    (LEAST for occurred_start, GREATEST for occurred_end / mentioned_at), and merges tags.
    """
    model = next((m for m in observations if str(m.id) == observation_id), None)
    if not model:
        logger.debug(f"Update skipped: observation {observation_id} not found in recall results")
        return

    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(
            f"Update skipped: all {len(source_memory_ids)} source memories for observation "
            f"{observation_id} were deleted concurrently"
        )
        return
    source_memory_ids = live_source_memory_ids

    from ...config import get_config

    history_entry = {
        "previous_text": model.text,
        "previous_tags": list(model.tags or []),
        "previous_occurred_start": model.occurred_start,
        "previous_occurred_end": model.occurred_end,
        "previous_mentioned_at": model.mentioned_at,
        "changed_at": datetime.now(timezone.utc).isoformat(),
        "new_source_memory_ids": [str(mid) for mid in source_memory_ids],
    }

    source_ids = list(model.source_fact_ids or []) + source_memory_ids

    # SECURITY: Merge source fact's tags into existing observation tags so all contributors can see it
    existing_tags = set(model.tags or [])
    source_tags = set(source_fact_tags or [])
    merged_tags = list(existing_tags | source_tags)

    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [new_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    config = get_config()
    history_clause = ""  # history column does not exist in current schema

    t0 = time.time()
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET text = $1,
            embedding = $2::vector,
            source_memory_ids = $3,
            proof_count = $4,
            tags = $9,
            updated_at = now(),
            occurred_start = LEAST(occurred_start, COALESCE($6, occurred_start)),
            occurred_end = GREATEST(occurred_end, COALESCE($7, occurred_end)),
            mentioned_at = GREATEST(mentioned_at, COALESCE($8, mentioned_at))
        WHERE id = $5
        """,
        new_text,
        embedding_str,
        source_ids,
        len(source_ids),
        uuid.UUID(observation_id),
        source_occurred_start,
        source_occurred_end,
        source_mentioned_at,
        merged_tags,
    )

    # Sync observation_sources junction table (Oracle only — PG uses native array ops).
    if memory_engine._backend.ops.uses_observation_sources_table:
        obs_uuid = uuid.UUID(observation_id)
        await conn.execute(
            f"DELETE FROM {fq_table('observation_sources')} WHERE observation_id = $1",
            obs_uuid,
        )
        if source_ids:
            await conn.executemany(
                f"""
                INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
                VALUES ($1, $2)
                ON CONFLICT (observation_id, source_id) DO NOTHING
                """,
                [(obs_uuid, sid) for sid in dict.fromkeys(source_ids)],
            )

    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Updated observation {observation_id} from {len(source_memory_ids)} source memories")


async def _execute_create_action(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    text: str,
    source_fact_tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> None:
    """
    Create a new observation from one or more source memories.

    Tags are inherited from the source facts (determined algorithmically, not by LLM)
    to maintain visibility scope.
    """
    await _create_observation_directly(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        source_memory_ids=source_memory_ids,
        observation_text=text,
        tags=source_fact_tags or [],
        event_date=event_date,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
        perf=perf,
    )
    logger.debug(f"Created observation from {len(source_memory_ids)} source memories")


async def _execute_delete_action(
    conn: "Connection",
    bank_id: str,
    observation_id: str,
) -> None:
    """Delete a superseded or contradicted observation."""
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 AND fact_type = 'observation'",
        uuid.UUID(observation_id),
        bank_id,
    )
    logger.debug(f"Deleted observation {observation_id}")


async def _create_memory_links(
    conn: "Connection",
    memory_id: uuid.UUID,
    observation_id: uuid.UUID,
) -> None:
    """
    Placeholder for observation link creation.

    Observations do NOT get any memory_links copied from their source facts.
    Instead, retrieval uses source_memory_ids to traverse:
    - Entity connections: observation → source_memory_ids → unit_entities
    - Semantic similarity: observations have their own embeddings
    - Temporal proximity: observations have their own temporal fields

    This avoids data duplication and ensures observations are always
    connected via their source facts' relationships.

    The memory_id and observation_id parameters are kept for interface
    compatibility but no links are created.
    """
    # No links are created - observations rely on source_memory_ids for traversal
    pass


async def _find_related_observations(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    tags: list[str] | None = None,
) -> "RecallResult":
    """
    Find observations related to the given query using optimized recall.

    SECURITY: Filters by tags using all_strict matching to prevent cross-tenant/cross-user
    information leakage. Observations are only consolidated within the same tag scope.

    Uses max_tokens to naturally limit observations (no artificial count limit).
    Includes source memories with dates for LLM context.

    Args:
        tags: Optional tags to filter observations (uses all_strict matching for security)

    Returns:
        List of related observations with their tags, source memories, and dates
    """
    # Use recall to find related observations with token budget
    # max_tokens naturally limits how many observations are returned
    from ...tracing import get_tracer, is_tracing_enabled

    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # SECURITY: Use all_strict matching if tags provided to prevent cross-scope consolidation
    tags_match = "all_strict" if tags else "any"

    # Create span for recall operation within consolidation
    tracer = get_tracer()
    if is_tracing_enabled():
        recall_span = tracer.start_span("hindsight.consolidation_recall")
        recall_span.set_attribute("hindsight.bank_id", bank_id)
        recall_span.set_attribute("hindsight.query", query[:100])  # Truncate for brevity
        recall_span.set_attribute("hindsight.fact_type", "observation")
    else:
        recall_span = None

    # Resolve budget: consolidation doesn't need deep recall, default to LOW to reduce memory fan-out
    recall_budget = Budget(config.consolidation_recall_budget)

    try:
        recall_result = await memory_engine.recall_async(
            bank_id=bank_id,
            query=query,
            budget=recall_budget,
            max_tokens=config.consolidation_max_tokens,  # Token budget for observations (configurable)
            fact_type=["observation"],  # Only retrieve observations
            request_context=request_context,
            tags=tags,  # Filter by source memory's tags
            tags_match=tags_match,  # Use strict matching for security
            include_source_facts=True,  # Embed source facts so we avoid a separate DB fetch
            max_source_facts_tokens=config.consolidation_source_facts_max_tokens,
            max_source_facts_tokens_per_observation=config.consolidation_source_facts_max_tokens_per_observation,
            _quiet=True,  # Suppress logging
        )
    finally:
        if recall_span:
            recall_span.end()

    return recall_result


def _build_observations_for_llm(
    observations: "list[MemoryFact]",
    source_facts: "dict[str, MemoryFact]",
) -> list[dict[str, Any]]:
    """Serialize MemoryFact observations into dicts for the consolidation LLM prompt."""
    obs_list = []
    for obs in observations:
        obs_data: dict[str, Any] = {
            "id": obs.id,
            "text": obs.text,
            "proof_count": len(obs.source_fact_ids or []) or 1,
        }
        if obs.occurred_start:
            obs_data["occurred_start"] = obs.occurred_start
        if obs.occurred_end:
            obs_data["occurred_end"] = obs.occurred_end
        if obs.mentioned_at:
            obs_data["mentioned_at"] = obs.mentioned_at
        source_memories = []
        for sid in obs.source_fact_ids or []:
            sf = source_facts.get(sid)
            if sf is None:
                continue
            sf_data: dict[str, Any] = {"text": sf.text}
            if sf.context:
                sf_data["context"] = sf.context
            if sf.occurred_start:
                sf_data["occurred_start"] = sf.occurred_start
            if sf.occurred_end:
                sf_data["occurred_end"] = sf.occurred_end
            if sf.mentioned_at:
                sf_data["mentioned_at"] = sf.mentioned_at
            source_memories.append(sf_data)
        if source_memories:
            obs_data["source_memories"] = source_memories
        obs_list.append(obs_data)
    return obs_list


async def _consolidate_batch_with_llm(
    llm_config: Any,
    memories: list[dict[str, Any]],
    union_observations: "list[MemoryFact]",
    union_source_facts: "dict[str, MemoryFact]",
    config: Any,
    remaining_observation_slots: int | None = None,
    max_observations_per_scope: int = -1,
    llm_semaphore: AdaptiveLLMConcurrencyLimiter | None = None,
    bank_id: str = "",
) -> _BatchLLMResult:
    """Single LLM call for a batch of facts against a pooled set of observations."""
    if config is None:
        raise ValueError("config is required for _consolidate_batch_with_llm")
    if union_observations:
        obs_list = _build_observations_for_llm(union_observations, union_source_facts)
        observations_text = json.dumps(obs_list, indent=2, ensure_ascii=False)
    else:
        observations_text = "[]"

    def _fact_line(m: dict[str, Any]) -> str:
        text = f"[{m['id']}] {m['text']}"
        temporal_parts = []
        if m.get("occurred_start"):
            temporal_parts.append(f"occurred_start={m['occurred_start']}")
        if m.get("occurred_end"):
            temporal_parts.append(f"occurred_end={m['occurred_end']}")
        if m.get("mentioned_at"):
            temporal_parts.append(f"mentioned_at={m['mentioned_at']}")
        if temporal_parts:
            text += f" ({', '.join(temporal_parts)})"
        return text

    facts_lines = "\n".join(_fact_line(m) for m in memories)

    # Build capacity note for the prompt when observation limit is configured
    observation_capacity_note: str | None = None
    if remaining_observation_slots is not None and max_observations_per_scope > 0:
        if remaining_observation_slots == 0:
            observation_capacity_note = (
                f"OBSERVATION LIMIT REACHED ({max_observations_per_scope}/{max_observations_per_scope}). "
                "Only UPDATE or DELETE existing observations. Do NOT create new ones — "
                "merge new knowledge into existing observations via UPDATE."
            )
        elif remaining_observation_slots <= len(memories):
            observation_capacity_note = (
                f"This scope has {remaining_observation_slots} observation slot(s) remaining "
                f"(out of {max_observations_per_scope}). Prefer UPDATE over CREATE when possible."
            )

    prompt_template = build_batch_consolidation_prompt(config.observations_mission, observation_capacity_note)
    prompt = prompt_template.format(
        facts_text=facts_lines,
        observations_text=observations_text,
    )

    # Use a constrained response model when observation limit is active
    response_model = _build_response_model(max_creates=remaining_observation_slots)

    max_attempts = config.consolidation_max_attempts
    inner_max_retries = config.consolidation_llm_max_retries
    max_rate_limit_retries = _read_non_negative_int_env("HINDSIGHT_API_CONSOLIDATION_429_MAX_RETRIES", 0)
    last_exc: Exception | None = None
    # Pre-compute a stable identifier set for the batch so failure logs name the
    # exact memories whose consolidation is failing — without this, an opaque
    # "LLM batch call failed" line gives operators no way to find the offending
    # input until adaptive bisection narrows the batch down to a single memory.
    memory_ids = [str(m.get("id")) for m in memories]
    if len(memory_ids) <= 5:
        ids_label = ", ".join(memory_ids)
    else:
        ids_label = f"{', '.join(memory_ids[:3])}, ... +{len(memory_ids) - 3} more"
    batch_label = f"{len(memory_ids)} memories [{ids_label}]"
    attempt = 1
    rate_limit_retries = 0
    while attempt <= max_attempts:
        try:
            call_kwargs: dict[str, Any] = {
                "messages": [{"role": "user", "content": prompt}],
                "response_format": response_model,
                "scope": "consolidation",
            }
            if inner_max_retries is not None:
                call_kwargs["max_retries"] = inner_max_retries
            if llm_semaphore is None:
                response: _ConsolidationBatchResponse = await llm_config.call(**call_kwargs)
            else:
                async with llm_semaphore:
                    response = await llm_config.call(**call_kwargs)
            # Defensive truncation: some LLM providers may not enforce JSON schema max_length
            creates = response.creates
            if remaining_observation_slots is not None and remaining_observation_slots >= 0:
                if len(creates) > remaining_observation_slots:
                    logger.info(
                        f"[CONSOLIDATION] Truncating {len(creates)} creates to {remaining_observation_slots} "
                        f"(max_observations_per_scope={max_observations_per_scope})"
                    )
                    creates = creates[:remaining_observation_slots]
            return _BatchLLMResult(
                creates=creates,
                updates=response.updates,
                deletes=response.deletes,
                obs_count=len(union_observations),
                prompt_chars=len(prompt),
            )
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc) and llm_semaphore is not None:
                rate_limit_retries += 1
                if max_rate_limit_retries and rate_limit_retries > max_rate_limit_retries:
                    logger.error(
                        f"[CONSOLIDATION] 429/rate-limit retries exceeded "
                        f"({rate_limit_retries - 1}/{max_rate_limit_retries}) for {batch_label}: {exc}"
                    )
                    break
                await llm_semaphore.handle_rate_limit(bank_id=bank_id or "<unknown>", batch_label=batch_label, exc=exc)
                # 429 is capacity pressure, not bad input. Do not consume the
                # normal attempt budget or split/mark facts failed; retry the
                # same batch after the global cooldown at the reduced limit.
                continue
            logger.warning(
                f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}) for {batch_label}: {exc}"
            )
            attempt += 1

    logger.error(
        f"[CONSOLIDATION] LLM batch call failed after {max_attempts} attempts for {batch_label}, "
        f"skipping batch. Last error: {last_exc}"
    )
    return _BatchLLMResult(obs_count=len(union_observations), prompt_chars=len(prompt), failed=True)


async def _create_observation_directly(
    conn: "Connection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_text: str,
    tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """Create an observation from one or more source memories with pre-processed text."""
    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(f"Create skipped: all {len(source_memory_ids)} source memories were deleted concurrently")
        return {"action": "skipped", "reason": "sources_deleted"}
    source_memory_ids = live_source_memory_ids

    # Generate embedding for the observation (convert to string for pgvector)
    t0 = time.time()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [observation_text])
    embedding_str = str(embeddings[0]) if embeddings else None
    if perf:
        perf.record_timing("embedding", time.time() - t0)

    # Create the observation as a memory_unit
    now = datetime.now(timezone.utc)
    obs_event_date = event_date or now
    obs_occurred_start = occurred_start
    obs_occurred_end = occurred_end
    obs_mentioned_at = mentioned_at or now
    obs_tags = tags or []

    t0 = time.time()
    observation_id = uuid.uuid4()

    # Query varies based on text search backend
    config = get_config()
    if config.text_search_extension == "vchord":
        # VectorChord: manually tokenize and insert search_vector
        query = f"""
            INSERT INTO {fq_table("memory_units")} (
                id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10,
                    tokenize($3, 'llmlingua2')::bm25_catalog.bm25vector)
            RETURNING id
        """
    else:  # native or pg_textsearch
        # Native PostgreSQL: search_vector is GENERATED ALWAYS, don't include it
        # pg_textsearch: indexes operate on base columns directly, don't populate search_vector
        query = f"""
            INSERT INTO {fq_table("memory_units")} (
                id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                tags, event_date, occurred_start, occurred_end, mentioned_at
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, 1, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """

    row = await conn.fetchrow(
        query,
        observation_id,
        bank_id,
        observation_text,
        embedding_str,
        source_memory_ids,
        obs_tags,
        obs_event_date,
        obs_occurred_start,
        obs_occurred_end,
        obs_mentioned_at,
    )

    # Populate observation_sources junction table (Oracle only — PG uses native array ops).
    if memory_engine._backend.ops.uses_observation_sources_table and source_memory_ids:
        await conn.executemany(
            f"""
            INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
            VALUES ($1, $2)
            ON CONFLICT (observation_id, source_id) DO NOTHING
            """,
            [(observation_id, sid) for sid in dict.fromkeys(source_memory_ids)],
        )

    if perf:
        perf.record_timing("db_write", time.time() - t0)

    logger.debug(f"Created observation {observation_id} from {len(source_memory_ids)} memories (tags: {obs_tags})")

    return {"action": "created", "observation_id": str(row["id"]), "tags": obs_tags}
