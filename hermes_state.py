#!/usr/bin/env python3
"""SQLite state store for Hermes Agent: session metadata, message history, model
config, FTS5 search. WAL mode (concurrent readers + one writer); compression
splits sessions via parent_session_id chains; sessions are source-tagged
('cli', 'telegram', ...). Batch-runner / RL trajectories live elsewhere.
"""

import asyncio
import atexit
import hashlib
import json
import logging
import os
import queue
import random
import re
import sqlite3
import stat
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager, suppress
from pathlib import Path

from hermes_constants import get_hermes_home, mkdir_under_hermes_home
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, cast

from hermes_state_common import (
    SCHEMA_VERSION,
    TITLE_SOURCE_DERIVED as _TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM as _TITLE_SOURCE_LLM,
    TITLE_SOURCE_USER as _TITLE_SOURCE_USER,
    escape_like as _escape_like, _placeholders,
    stat_db_file_identity as _stat_db_file_identity,
)
from hermes_state_holders import read_only_db_uri
from hermes_state_pidns import holder_pid_checkable
from hermes_state_health import (
    STORAGE_CORRUPT, mark_storage_corrupt, note_storage_error, storage_corrupt_reason, storage_state,
)
from hermes_state_errors import (
    _DELETED_WAL_GENERATION_MSG, _DISK_IO_ERROR_MARKER, _STATE_DB_CORRUPT_MSG, _STATE_DB_GENERATION_KEY,
    _STATE_DB_REPLACED_MSG, DeletedWalGenerationError, SessionCompressionInProgressError, StateDbCorruptError,
    StateDbReplacedError, _is_no_more_rows, classify_persistence_error, is_malformed_db_error,
    is_malformed_schema_error, is_sqlite_lock_error,
)
from hermes_state_guard import (
    _STATE_DB_GUARD_BYPASS_ENV, _in_test_context, _is_production_state_db, _real_platform_state_root,
    _register_test_instance, _set_last_init_error, get_last_init_error,
)
from hermes_state_readpool import _READ_POOL_MAX, _proc_fd_targets, _read_budget_for
from hermes_state_sessions import SessionSessionsMixin
from hermes_state_fts import SessionFtsSetupMixin, load_fts5_cjk_extension
from hermes_state_portability import SessionPortabilityMixin
from hermes_state_telegram import SessionTelegramTopicsMixin
from hermes_state_profile_repair import SessionProfileRepairMixin
from hermes_state_schema import SessionSchemaMixin, _MESSAGE_UID_BACKFILL_DONE
from hermes_state_copy_provenance import (
    _CONTINUATION_KEYS as _RECEIPT3_CONTINUATION_KEYS,
    _DOMAIN_SHA256 as _RECEIPT3_DOMAIN, build_epoch_continuation_header, build_journal_page,
    boundary_roster_digest, encode_receipt3_operation_key, encode_receipt3_page_key,
    encode_receipt3_token_key, final_all_owner_readback,
    observe_native_session_contexts, read_operation, read_partial_recovery_state,
    seal_epoch_blocks, validate_boundary_roster, validate_receipt3_head,
    verify_frozen_proof, verify_independent_copy_joint, verify_roster_continuation_inventory,
)
import hermes_state_holders as _state_holders
import hermes_state_lockguard as _lockguard
from hermes_state_lockowners import log_write_lock_holders
from hermes_state_dbfile import (
    _connect_tracked_db, _fd_is_truly_unlinked, _prepare_connection_retirement,
    _read_sqlite_application_id, _stat_sqlite_sidecar_identity,
    _watched_sqlite_sidecar_paths, has_invalid_sqlite_header_preopen, is_zeroed_state_db, quarantine_cross_process_lock,
    quarantine_invalid_state_db,
    RetiredGenerationCaptureError, capture_retired_wal_generation, refuse_deleted_wal_generation,
)
from hermes_state_messages import SessionMessagesMixin
from hermes_state_coverage import SessionCoverageMixin
from hermes_state_rewind import SessionRewindMixin
from hermes_state_wal import (
    _WAL_INCOMPAT_MARKERS, _on_disk_journal_mode, apply_database_pragmas, apply_wal_with_fallback,
)
from hermes_state_repair import _claim_repair_attempt, preflight_db_writability, repair_state_db_schema
from hermes_state_titles import SessionTitlesMixin
from hermes_state_usage import SessionUsageMixin
from hermes_state_maintenance import SessionMaintenanceMixin
from hermes_state_gateway import SessionGatewayMixin
from hermes_state_compression import SessionCompressionMixin
from hermes_state_search import SessionSearchMixin

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MAX_SAFE_MESSAGES = 20_000  # resume/export guard default


def _configured_transcript_limit(key: str, fallback: int = _MAX_SAFE_MESSAGES) -> int:
    """``sessions.<key>`` from config.yaml (lazy import: circular at load), else *fallback*; 0 disables."""
    try:
        from hermes_cli.config import load_config_readonly
        value = (load_config_readonly().get("sessions") or {}).get(key)
        if value is None:
            return fallback
        limit = int(value)
        return limit if limit >= 0 else fallback
    except Exception:
        return fallback


def resolved_max_resume_messages() -> int:
    return _configured_transcript_limit("max_resume_messages")


def resolved_max_export_messages() -> int:
    return _configured_transcript_limit("max_export_messages")


class SessionResumeTooLargeError(ValueError):
    def __init__(
        self, message_count: int, limit: int = _MAX_SAFE_MESSAGES, scope: str = "across its lineage",
    ):
        self.message_count, self.limit = message_count, limit
        self.scope = scope
        super().__init__(
            f"This session is too long to reload safely ({message_count} messages; limit {limit}). "
            "Start a fresh chat and keep a copy with the dashboard Sessions page's Export action or "
            "`hermes sessions export --format md --session-id <id>` (neither is capped), or raise the limit "
            "with `hermes config set sessions.max_resume_messages 0`."
        )


class SessionExportTooLargeError(ValueError):
    def __init__(self, session_id: str, message_count: int, limit: int = _MAX_SAFE_MESSAGES):
        self.session_id, self.message_count, self.limit = session_id, message_count, limit
        # User-facing refusal shared by every in-memory JSON/JSONL export (CLI and console).
        super().__init__(
            f"Session '{session_id}' has more than {limit:,} exportable messages; the JSON/JSONL "
            "backup is built in memory and capped per session. Use the dashboard Sessions page's streaming "
            "Export action, or set sessions.max_export_messages: 0 in config.yaml to disable the guard."
        )


def _compression_lock_holder_process_is_dead(holder: str) -> bool:
    """True only when a ``pid=<n>`` lock holder's local PID is provably gone.
    Reclaim on kernel proof only: unstructured/same-process holders (another
    thread's live lease) and any probe doubt keep the lease until TTL expiry
    (PID reuse must never steal a live lease; a wrongly-kept one self-heals).
    Foreign/unstamped PID namespaces defer to TTL: see ``hermes_state_pidns``."""
    match = re.search(r"(?:^|:)pid=(\d+)(?::|$)", holder or "")
    pid = int(match.group(1)) if match else 0
    if pid <= 0 or pid == os.getpid():
        return False
    if not holder_pid_checkable(holder):
        return False  # foreign / unknown namespace: defer to TTL
    if psutil is not None:
        try:
            return not psutil.pid_exists(pid)  # recycled PIDs read as alive (conservative)
        except Exception:
            return False
    # psutil-less fallback is POSIX-only: on Windows os.kill(pid, 0) maps sig=0 to
    # CTRL_C_EVENT and can kill the target's console group.
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — nt early-returns just above
    except ProcessLookupError:
        return True
    except (OSError, OverflowError):  # PermissionError is an OSError: alive but foreign
        return False
    return False


# Billing buckets that aren't a routable provider identity: a session that persisted only
# one of these (never ran /model) falls back to the config default. Shared by
# session_gateway_runtime and tui_gateway.server so they cannot drift.
_BARE_BILLING_PROVIDERS = frozenset({"auto", "custom"})

T = TypeVar("T")

# Import-time snapshot lets _default_db_path() detect a re-pointed DEFAULT_DB_PATH
# (tests monkeypatch the constant directly).
DEFAULT_DB_PATH = _IMPORT_DEFAULT_DB_PATH = get_hermes_home() / "state.db"

# Back off from read-only opens after one fails: not per query, but short enough that
# transient fd pressure doesn't strand the read pool.
_READ_OPEN_RETRY_SECONDS = 60.0
# Transient SQLITE_IOERR retry budget for READ-ONLY opens (#100436): a WAL writer's checkpoint/
# reset/frame flush surfaces "disk I/O error" to a concurrent mode=ro reader for a millisecond-
# wide window — the ro connection cannot perform WAL recovery because recovery writes the -shm
# index, which mode=ro refuses. The writer closes the window on its own, so a few short retries
# make the open succeed instead of 500-ing the whole /api/sessions poll (or any other ro opener).
# Deliberately NOT for writable opens: a writer owns the transition, so an IOERR there is a real
# storage/fd problem. A persistent IOERR still exhausts the budget and propagates.
_READ_ONLY_IOERR_RETRY_ATTEMPTS, _READ_ONLY_IOERR_RETRY_BACKOFF_S = 3, 0.05
# SQLite busy handler budget for reads. Under DELETE (rollback-journal) mode a reader needs a SHARED
# lock, which every commit from another process blocks across its journal+db fsyncs; the writer
# connection's 1 s timeout exists for writes (they retry at application level) and starved readers
# into "database is locked" (dashboard 503s, `hermes sessions list` crashes) under a busy gateway.
_READ_BUSY_TIMEOUT_S = 5.0


def _default_db_path() -> Path:
    """Default state DB path at CALL time: a re-pointed ``DEFAULT_DB_PATH`` wins, else
    ``get_hermes_home()`` is resolved fresh (a runtime HERMES_HOME redirect works regardless of import)."""
    return DEFAULT_DB_PATH if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH else get_hermes_home() / "state.db"


# Live-DB guard knobs live HERE (not in hermes_state_guard): the hermetic conftest monkeypatches
# ``hermes_state._STATE_DB_GUARD_BYPASS`` (``@pytest.mark.live_system_guard_bypass`` escape hatch)
# and ``_EXTRA_DENY_ROOTS`` (the pre-sandbox root, so custom-HERMES_HOME deployments are covered).
_STATE_DB_GUARD_BYPASS = False
_STATE_DB_GUARD_EXTRA_DENY_ROOTS: tuple[Path, ...] = ()


def _ensure_test_isolation(db_path: Path) -> None:
    """Raise before any connection/mkdir/pragma/byte probe when a pytest-context process
    (env OR ancestry) resolves a production DB.

    Env alone is not enough: a child spawned with a rebuilt environment loses ``PYTEST_*`` and
    ``HERMES_HOME`` together, which is precisely the state in which it writes to production (#82770).
    """
    if _STATE_DB_GUARD_BYPASS or os.environ.get(_STATE_DB_GUARD_BYPASS_ENV) or not _in_test_context():
        return
    try:
        resolved = Path(db_path).expanduser().resolve()
    except Exception:
        return
    roots = [r for r in (_real_platform_state_root(),) if r is not None]
    for extra in _STATE_DB_GUARD_EXTRA_DENY_ROOTS:
        try:
            roots.append(Path(extra).expanduser().resolve())
        except Exception:
            continue
    for root in roots:
        if _is_production_state_db(resolved, root):
            raise RuntimeError(
                "live-system guard: test attempted to open production "
                f"state.db at {resolved} (under real Hermes root {root}). "
                "Tests must run against a temporary HERMES_HOME — pass an "
                "explicit tmp db_path or let the hermetic conftest redirect "
                "HERMES_HOME. If this test genuinely needs the live database, mark it with "
                "@pytest.mark.live_system_guard_bypass — or, for a spawned "
                f"child process, export {_STATE_DB_GUARD_BYPASS_ENV}=1 in "
                "its environment."
            )


def _secure_state_db_files(db_path: Path, *, create_main: bool = False) -> None:
    """Create/tighten a writable state database and its sidecars to 0600.

    SQLite otherwise creates ``state.db``, ``-wal``, and ``-shm`` according to
    the process umask (commonly 0644 under 0022). Read-only SessionDB
    attachments never call this helper and remain observational.

    Existing files are tightened with ``chmod(2)`` on the path: opening the
    file and closing that descriptor would drop every POSIX ``fcntl`` lock the
    process holds on its inode — including the locks of an already-open SQLite
    connection to the same database. A lock-losing close in one process lets a
    sibling's connection take the shared-memory DMS exclusively at its own
    close, checkpoint, and unlink the sidecars while long-lived holders
    (gateway, desktop ``hermes serve``) keep using the deleted inodes.
    """
    if os.name == "nt":
        return

    main_path = db_path
    if create_main:
        # O_EXCL: only a brand-new inode gets a descriptor. Opening an existing
        # file here and closing it would drop this process's POSIX locks on it.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            fd = os.open(main_path, flags, 0o600)
        except FileExistsError:
            pass
        except IsADirectoryError:
            # Not a database file at all; sqlite3.connect() raises the
            # canonical error for this, and a directory leaks no row data.
            return
        else:
            os.close(fd)

    for path in (
        main_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        # fchmod on an fd of a pre-existing file cannot be used here: close(fd)
        # would release this process's POSIX locks on that inode, stripping the
        # locks of any live SQLite connection to the same database. chmod(2)
        # never opens the file, so it leaves the lock state untouched.
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(st.st_mode):
            # Refuse a planted symlink exactly like O_NOFOLLOW would.
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        os.chmod(path, 0o600)


# Openings of the background-review harness prompts (agent/background_review.py).
_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(msg: dict[str, Any]) -> bool:
    """Persisted harness prompt (older builds wrote the forked curator's turns
    into real sessions; replaying them hijacks the session)."""
    if not isinstance(msg, dict) or msg.get("role") not in {"user", "system"}:
        return False
    content = msg.get("content")
    return isinstance(content, str) and content.lstrip().startswith(_REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop harness messages and the curator-mode assistant reply that immediately followed each."""
    if not messages:
        return messages
    out: list[dict[str, Any]] = []
    skip_next_assistant = False
    previous_was_harness = False
    for msg in messages:
        if _is_background_review_harness_message(msg):
            # A consecutive harness prompt occupies the preceding prompt's
            # immediate reply slot, so it must not arm another assistant skip.
            skip_next_assistant = not previous_was_harness
            previous_was_harness = True
            continue
        if skip_next_assistant:
            skip_next_assistant = False
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                previous_was_harness = False
                continue  # the curator-mode reply to the harness prompt
        previous_was_harness = False
        out.append(msg)
    return out


# Matches a bare protocol/tool-name marker such as "[memory]" or "[skill_manage]".
_STALE_TOOL_CALL_MARKER_RE = re.compile(r"^\[[A-Za-z_][A-Za-z0-9_.-]*\]$")


def _is_stale_tool_call_marker_message(msg: dict[str, Any]) -> bool:
    """Assistant tool-call turn whose content is a bare ``[marker]`` (an older
    conversation_loop persisted a local template's marker as the final response)."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant" or not msg.get("tool_calls"):
        return False
    content = msg.get("content")
    return isinstance(content, str) and bool(_STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()))


def _strip_stale_tool_call_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Blank stale ``[marker]`` assistant content (replaying it teaches the model
    to keep emitting it); tool_call/result pairing stays intact."""
    repaired = 0
    for msg in filter(_is_stale_tool_call_marker_message, messages):
        msg["content"] = ""
        repaired += 1
    if repaired:
        logger.info(
            "Cleared %d stale tool-call marker message(s) while restoring session (#78148)", repaired,
        )
    return messages


_SESSION_DB_CONSEQUENCE = "Sessions will not be saved until this is fixed."
_NETWORK_DRIVE_HINT = " If the database lives on a network drive, move it to a local disk."
_NETWORK_DRIVE_GLOSS = "the session database could not be opened; it may be on a network or unsupported drive"
_NETWORK_DRIVE_ACTION = (
    "Move it to a local disk (`hermes {profile_arg}doctor` shows where it is), then start Hermes again."
)


def format_session_db_unavailable(
    prefix: str = "Hermes can't open its session history right now",
    *,
    details: bool = False,
) -> str:
    """User-facing one-liner: ``<prefix>: <gloss>. <consequence> <action>[ network hint]``.

    The cause table lives in ``hermes_state_user_copy`` so CLI, gateway and TUI agree. Chat
    surfaces (gateway, TUI) get the one-liner; ``details=True`` (the CLI banner) appends a
    ``Details: <raw cause>`` line for the raw SQLite text. Network filesystems (NFS/SMB/FUSE/ZFS)
    cannot host SQLite's write-ahead log: when the raw cause carries one of those markers the
    message names the network-drive suspicion, because ``hermes doctor --fix`` cannot repair a
    mount — only moving the file can."""
    from hermes_constants import profile_cli_selector

    profile_arg = profile_cli_selector()
    cause = get_last_init_error()
    if not cause:
        return (
            f"{prefix}. {_SESSION_DB_CONSEQUENCE} Run `hermes {profile_arg}doctor` to check the "
            "storage location."
        )
    from hermes_state_user_copy import describe_storage_failure
    failure = describe_storage_failure(cause)
    gloss, action, hint = failure.gloss, failure.action, ""
    if any(m in cause.lower() for m in _WAL_INCOMPAT_MARKERS):
        if failure.cause == "unknown":
            gloss, action = _NETWORK_DRIVE_GLOSS, _NETWORK_DRIVE_ACTION.replace("{profile_arg}", profile_arg)
        else:
            hint = _NETWORK_DRIVE_HINT
    text = f"{prefix}: {gloss}. {_SESSION_DB_CONSEQUENCE} {action}{hint}"
    if details:
        from hermes_state_user_copy import storage_failure_details
        text += f"\nDetails: {storage_failure_details(cause)}"
    return text


# Auto-repair at most once per DB path per process (no repair loops; serialises concurrent
# web_server / gateway opens on the same malformed file).
_repair_attempted_paths: set[str] = set()
_repair_attempt_lock = threading.Lock()
# Cross-process schema-surgery lock timeout (``_repair_attempt_lock`` covers one interpreter
# only); sized for the slowest legitimate holder (VACUUM, multi-GB DB).
_REPAIR_LOCK_TIMEOUT_SECONDS = 120.0
_IS_WINDOWS = sys.platform == "win32"


def _close_time_checkpoint_configurable() -> bool:
    """Whether this runtime can switch off SQLite's close-time checkpoint (Python 3.12+ ``setconfig``)."""
    return (getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None) is not None
            and hasattr(sqlite3.Connection, "setconfig"))


def divert_session_transcript_jsonl(session_id: str, messages) -> Optional[Path]:
    """Append pending messages to HERMES_HOME/sessions/<id>.jsonl (state.db was replaced under a
    live process). Returns the path, or None if nothing to write."""
    sid = str(session_id or "").strip()
    if not sid or not messages:
        return None
    sessions_dir = get_hermes_home() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{sid}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for msg in messages:
            if msg is not None:
                record = msg if isinstance(msg, dict) else {"content": str(msg)}
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


# Process-wide shared SessionDB registry: long-lived in-process callers share ONE writer
# connection per resolved path via hermes_state_registry.acquire(); one-shots use SessionDB() + close().
def _foreign_state_db_holders(db_path: Path) -> list[tuple[int, str]]:
    """Compatibility delegate to the state-holder authority."""
    return _state_holders.foreign_state_db_holders(db_path)


# ── Process-wide shared SessionDB registry (#90837) ── lives in hermes_state_registry.py (acquire /
# release / close_all / release_or_close). Long-lived in-process callers (gateway, tui_gateway, cron,
# in-process tools) share ONE writer connection per resolved path via hermes_state_registry.acquire(); CLI
# one-shots, recovery flows, and read-only cross-profile opens use SessionDB() directly with their own close().


class SessionDB(
    SessionSessionsMixin, SessionFtsSetupMixin, SessionSearchMixin, SessionSchemaMixin,
    SessionPortabilityMixin, SessionTelegramTopicsMixin, SessionCompressionMixin,
    SessionGatewayMixin, SessionMaintenanceMixin, SessionUsageMixin, SessionTitlesMixin,
    SessionMessagesMixin, SessionCoverageMixin, SessionRewindMixin, SessionProfileRepairMixin,
):
    """SQLite-backed session storage with FTS5 search; many reader threads, one writer (WAL)."""

    # Only these state-owned producers join automatic stale-open reconciliation; messaging/UI
    # sources have their own lifecycle owners; unknown sources fail closed.
    # See #60609.  `recovered` = placeholders `hermes sessions recover` synthesizes for
    # orphaned messages (no live owner, never stamped ended_at); without it they are immortal.
    _AUTO_PRUNE_STALE_OPEN_SOURCES: tuple[str, ...] = (
        "cli", "cron", "kanban", "acp", "api_server", "subagent", "tool", "recovered",
    )

    # ── Write-contention tuning ──
    # SQLite's deterministic busy handler convoys under many hermes processes: keep its
    # timeout short (1s) and retry with random jitter. Patience is TIME-based (a sibling
    # legitimately holds the lock for seconds: checkpoint at close, VACUUM, recovery, FTS
    # optimize); attempt-counted budgets destroyed turns on a healthy store. Transcript
    # writes (failure aborts the turn) get the long budget; observation-only activity
    # writes sit on the response-critical path and get a sub-second one.
    _WRITE_PATIENCE_S, _TRANSCRIPT_WRITE_PATIENCE_S, _ACTIVITY_WRITE_PATIENCE_S = 20.0, 60.0, 0.5
    # A live compression lock gets a short wait (compression publishes in seconds), but the lease
    # is a correctness boundary: a writer still locked out afterwards is refused.
    # Observation-only activity heartbeat/label writes (#76354 review S1): these run on (or adjacent to) the
    # response-critical path and must never wait out the full routine patience under contention. Sub-second
    # budget; a skipped write is retried naturally at the next heartbeat window.
    # A live compression lock gets its own, much shorter budget than the write lock. Compression publishes
    # in a couple of seconds, so a brief wait saves the overwhelming majority of concurrent turns (#75083).
    # It deliberately stays short: the lease is a correctness boundary, not just a busy signal (see
    # test_compression_lease_blocks_non_owner_but_allows_owner_flush), so a writer that is still locked out
    # after this budget must still be refused rather than allowed to land a stale turn in a session whose
    # compression is genuinely long-running or wedged.
    _COMPRESSION_BUSY_WAIT_S = 5.0
    _WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S = 0.020, 0.150  # fast jitter for the first _SLOW_AFTER_S
    _WRITE_RETRY_SLOW_AFTER_S = 2.0
    _WRITE_RETRY_SLOW_MIN_S, _WRITE_RETRY_SLOW_MAX_S = 0.250, 1.000
    # PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 50
    # Bounded FTS ``'merge'`` (ms of lock each) instead of ``'optimize'`` (9-18s per index on a 10GB
    # DB, longer than a writer's patience); up to _COMMANDS_PER_PASS per index, stopping on no-progress.
    _FTS_MERGE_EVERY_N_WRITES, _FTS_MERGE_MAX_PAGES_PER_INDEX, _FTS_MERGE_COMMANDS_PER_PASS = 1000, 500, 4
    # Imports cap lower than exports: an import holds one BEGIN IMMEDIATE.
    _IMPORT_MAX_SESSIONS, _IMPORT_MAX_MESSAGES_PER_SESSION, _IMPORT_MAX_TOTAL_MESSAGES = 500, 10_000, 50_000
    _IMPORT_MAX_SESSION_BYTES, _IMPORT_MAX_TOTAL_BYTES = 5 * 1024 * 1024, 25 * 1024 * 1024
    # Accounting workers retire when idle so a bound-method target can't keep an abandoned SessionDB alive.
    _TOKEN_WRITER_IDLE_SECONDS = 30.0

    @staticmethod
    def _store_system_prompt(conn, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES (?, ?)",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    @staticmethod
    def _delete_unreferenced_system_prompts(conn) -> None:
        conn.execute(
            "DELETE FROM system_prompts WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash) AND NOT EXISTS ("
            "SELECT 1 FROM sessions WHERE sessions.tool_names = system_prompts.hash)"
        )

    @staticmethod
    def _session_row_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for column in ("system_prompt", "tool_names"):
            if f"_{column}_resolved" in data:
                resolved = data.pop(f"_{column}_resolved")
                if column in data:
                    data[column] = resolved
        # Lift /new vs /branch markers out of model_config so list payloads
        # (which strip that heavy field) can still tell a reset sibling from
        # a genuine fork. Keep this a local helper: tests sometimes replace
        # the SessionDB name with a factory lambda.
        if not (data.get("_reset_from") and data.get("_branched_from")):
            raw = data.get("model_config")
            cfg = None
            if isinstance(raw, str) and raw:
                try:
                    cfg = json.loads(raw)
                except (TypeError, ValueError):
                    cfg = None
            elif isinstance(raw, dict):
                cfg = raw
            if isinstance(cfg, dict):
                if not data.get("_reset_from"):
                    value = cfg.get("_reset_from")
                    if isinstance(value, str) and value.strip():
                        data["_reset_from"] = value.strip()
                if not data.get("_branched_from"):
                    value = cfg.get("_branched_from")
                    if isinstance(value, str) and value.strip():
                        data["_branched_from"] = value.strip()
        return data

    @staticmethod
    def _close_connection_quietly(conn: Optional[sqlite3.Connection]) -> None:
        """Close a partially initialized connection without masking its error."""
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            logger.debug("Could not close a SessionDB connection", exc_info=True)

    def _close_conn_logged(self, conn, label: str) -> None:
        """Close *conn*; a failing close leaks a tracked fd: logged at WARNING, never swallowed."""
        try:
            conn.close()
        except Exception as exc:
            logger.warning("%s close failed for %s: %s", label, self.db_path, exc)

    def __init__(self, db_path: Path | None = None, read_only: bool = False):
        self.db_path = db_path or _default_db_path()
        _ensure_test_isolation(self.db_path)  # before any connection/pragma/mkdir
        self.read_only = read_only
        # Keep only the opening call site, never a frame (which pins caller locals).
        self._creation_site = "unknown"
        caller = None
        try:
            caller = sys._getframe(1)
            self._creation_site = (
                f"{caller.f_globals.get('__name__', '?')}.{caller.f_code.co_name}:{caller.f_lineno}"
            )
        except Exception:
            pass  # Diagnostic metadata must not prevent opening the database.
        finally:
            del caller
        self._lock = threading.Lock()
        # Read-path split (WAL only): reads borrow from a BOUNDED read-only pool so they
        # never queue behind writer flushes on self._lock (see _read_ctx); unbounded
        # per-thread connections pinned fds for the process lifetime and hit EMFILE.
        self._read_pool: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue(maxsize=_READ_POOL_MAX)
        # Permits bound PEAK descriptors (the pool bounds only the idle set), shared per
        # DATABASE PATH; acquired non-blocking so a permitless reader degrades to the writer lock.
        # One permit per live read connection, held from before the open in _get_read_conn() until after the
        # close in _close_read_conn(). See _READ_POOL_MAX. Acquired non-blocking on purpose: a reader that
        # cannot get a permit must degrade to the writer lock, not queue here — blocking would convert fd
        # exhaustion into a stall, which is the same outage with a different stack trace. Permits are shared
        # per DATABASE PATH, not per instance: the descriptors they ration belong to the file, and one
        # process holds several SessionDB objects on the same state.db (#98573). See _PathReadBudget.
        self._read_budget = _read_budget_for(self.db_path)
        self._read_permits = self._read_budget.permits
        self._read_conns_lock = threading.Lock()
        # Set when close() begins; an in-flight reader then closes its own connection
        # instead of re-populating a pool nobody will drain again.
        self._read_conns_closed = False
        # Read-open failure backoff is a TIMESTAMP, not a sticky bool: the likeliest trigger
        # is transient EMFILE, and a permanent flag would demote every reader forever.
        self._read_open_failed_at = 0.0
        self._wal_active, self._write_count = False, 0
        # File identity of the opened state.db, compared on every write so an out-of-band
        # replace cannot limp through in-place surgery (inode: mv/new-file; application_id: cp).
        self._db_file_identity: Optional[tuple] = None
        self._db_file_application_id: int = 0
        self._db_sidecar_identity: dict[str, tuple] = {}
        self._db_replaced = self._db_wal_generation_lost = False
        # Durable capture of a lost WAL generation (see _capture_retired_generation): once per handle.
        self._retired_generation_capture: Optional[Path] = None
        self._retired_capture_lock = threading.Lock()
        self._retire_connection: Optional[Callable[[Any], None]] = None
        self._connection_pinned = False  # one unmatched C reference taken at most once per handle
        self._wal_lock_guard: dict = {}  # hermes_state_lockguard.hold() record, see _open_writer
        self._db_corrupt, self._db_corrupt_reason = False, ""  # sticky quarantine (StateDbCorruptError)
        self._fts_usermerge_floor_applied = False  # one-shot usermerge-floor write guard
        self._fts_enabled = self._fts_stale = self._trigram_available = False
        # _fts_cjk_loaded: tokenizer on the writer connection; _fts_cjk_available: messages_fts_cjk
        # is queryable AND not marked stale.
        self._fts_cjk_loaded = self._fts_cjk_available = self._fts_unavailable_warned = False
        self._conn = None
        # Async token accounting; distinct from self._lock so enqueue/flush never contends with writes.
        self._token_queue: deque = deque()
        self._token_queue_cond = threading.Condition(threading.Lock())
        self._token_writer_thread: Optional[threading.Thread] = None
        self._token_writer_stop = self._token_writer_busy = False
        self._token_atexit_hook: Optional[Callable[[], None]] = None
        # Opened via hermes_state_registry.acquire(): close() releases a refcount instead.
        # Set True when this instance is opened via hermes_state_registry.acquire(). Makes close() a no-op so the
        # registry (not individual callers) controls the connection lifecycle (#90837).
        self._shared_registry_owned = False
        initialization_complete = False
        try:
            if read_only:
                self._open_read_only()
            else:
                # Where SQLite's close-time checkpoint cannot be switched off, a lost-generation handle
                # is retired unclosed (see close()). Resolve that capability before opening a writer:
                # late cleanup must not import ctypes or look up it in a cleared module dictionary.
                if not _close_time_checkpoint_configurable():
                    self._retire_connection = _prepare_connection_retirement()
                self._open_writer()
            self._record_db_file_identity()
            initialization_complete = True
        except Exception as exc:
            # Surface WHY via /resume and friends; callers keep their ``_session_db = None`` path.
            _set_last_init_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if not initialization_complete:
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)
            else:
                # Only a successfully opened handle owns a writer connection. Failed
                # construction must not leave a diagnostic member behind.
                self._read_budget.register(self)
                # Test-isolation runs only (gated inside the helper): register
                # for the suite-level leak sweep in tests/conftest.py.
                _register_test_instance(self)

    def _open_writer(self) -> None:
        """Writable open: preflight, zero-byte quarantine, connect + schema (one in-place repair of a
        malformed sqlite_master), generation stamp."""
        # Never materialize a deleted/archived named profile's home: a multiplexer or Desktop backend
        # still holding the profile's route would otherwise re-scaffold it on the next turn (#94590).
        mkdir_under_hermes_home(self.db_path.parent)
        # Read-only file/sidecar preflight BEFORE the first connection: an actionable message
        # instead of an opaque "attempt to write a readonly database" from inside _init_schema.
        preflight_db_writability(self.db_path, db_label="state.db")
        try:
            # Serialize zero-byte check, quarantine, connect and schema commit so concurrent
            # openers don't race the absent-path -> schema-commit window.
            if not self.db_path.exists() or has_invalid_sqlite_header_preopen(self.db_path):
                with quarantine_cross_process_lock(self.db_path) as lock_acquired:
                    if not lock_acquired:
                        logger.warning(
                            "startup quarantine lock for %s not acquired within 5s; proceeding",
                            self.db_path,
                        )
                    self._handle_quarantine_if_invalid(already_locked=lock_acquired)
                    self._connect_and_init_with_lock_patience()
            else:
                self._handle_quarantine_if_invalid(already_locked=False)
                self._connect_and_init_with_lock_patience()
        except sqlite3.DatabaseError as exc:
            # A malformed schema fails on the very first statement (before _init_schema), so the
            # FTS-rebuild layer never sees it: repair sqlite_master in place (backup first), reopen once.
            if not is_malformed_schema_error(exc) or not _claim_repair_attempt(self.db_path):
                raise
            logger.error(
                "state.db schema is malformed (%s) — attempting automatic "
                "repair (a backup copy is made first).", exc,
            )
            self._close_connection_quietly(self._conn)
            if not repair_state_db_schema(self.db_path).get("repaired"):
                raise
            self._connect_and_init_with_lock_patience()
        # FTS optimization is OPT-IN (`hermes db optimize`); no background worker races session lifecycle.
        self._ensure_db_file_generation()
        if self._wal_active:
            # OFD copies of the two POSIX locks that keep a sibling's close from unlinking this WAL
            # generation: any in-process open()/close() of state.db or -shm cancels SQLite's own
            # (howtocorrupt §2.2); these survive it. Lifted in close().
            self._wal_lock_guard = _lockguard.hold(self.db_path)

    def _open_read_only(self) -> None:
        """Read-only attach for cross-profile aggregation: no schema init, NO write
        lock (sidebar polling never contends with that profile's backend); the DB
        must exist. FTS flags are probed with SELECTs only, and the connection is
        closed on ANY probe failure (malformed schema raises DatabaseError) so a
        leaked tracked connection cannot block the forensic backup the writable heal takes next."""
        for attempt in range(_READ_ONLY_IOERR_RETRY_ATTEMPTS + 1):
            try:
                self._conn = conn = self._connect_read_only(timeout=_READ_BUSY_TIMEOUT_S)
                try:
                    apply_database_pragmas(conn, db_label="state.db")
                    cursor = conn.cursor()
                    self._fts_enabled = self._fts_table_probe(cursor, "messages_fts") is True
                    if self._fts_enabled:
                        self._trigram_available = (
                            self._fts_table_probe(cursor, "messages_fts_trigram") is True
                        )
                except BaseException:
                    self._conn = None
                    self._close_connection_quietly(conn)
                    raise
                return
            except sqlite3.OperationalError as ioerr:
                # In-flight WAL checkpoint/reset/frame-flush on the writer side can surface
                # SQLITE_IOERR to a mode=ro reader (it can't do the -shm recovery the read
                # needs). Closes in milliseconds: retry a bounded number of times before
                # classifying the store as failed (#100436; see _READ_ONLY_IOERR_RETRY_ATTEMPTS).
                # A lock is NOT retried here: the connection already waited _READ_BUSY_TIMEOUT_S,
                # and a retry would multiply that wait on blocking callers (TUI, `hermes status`).
                transient = _DISK_IO_ERROR_MARKER in str(ioerr).lower()
                if attempt >= _READ_ONLY_IOERR_RETRY_ATTEMPTS or not transient:
                    raise
                time.sleep(_READ_ONLY_IOERR_RETRY_BACKOFF_S)

    def _connect_read_only(self, timeout: float) -> sqlite3.Connection:
        """``mode=ro`` tracked connection with Row factory. check_same_thread=False: pooled connections
        are borrowed by whichever thread reads next; exclusive ownership is enforced by pool checkout."""
        conn = _connect_tracked_db(
            read_only_db_uri(self.db_path), tracking_path=self.db_path, uri=True,
            check_same_thread=False, timeout=timeout, isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _handle_quarantine_if_invalid(self, already_locked: bool = False) -> None:
        """Quarantine a zero-byte/headerless state.db so a fresh one can open; if quarantine failed,
        raise the clear message instead of opening the zeroed file."""
        if not (self.db_path.exists() and has_invalid_sqlite_header_preopen(self.db_path)):
            return
        try:
            zsize = self.db_path.stat().st_size
        except OSError:
            zsize = -1
        qpath = quarantine_invalid_state_db(self.db_path, already_locked=already_locked)
        where = f"moved aside to {qpath}" if qpath else "left in place (it could not be moved aside)"
        msg = (
            f"state.db was empty or damaged ({zsize} bytes) and has been {where}; Hermes started with a "
            "fresh, empty session database. To bring old sessions back, run "
            f"`hermes sessions recover --source {qpath or self.db_path} --inspect-only`, or restore a "
            "snapshot with `/snapshot list` then `/snapshot restore <id>` (terminal `hermes` chat only)."
        )
        logger.error(msg)
        _set_last_init_error(msg)
        if qpath is None and self.db_path.exists() and has_invalid_sqlite_header_preopen(self.db_path):
            raise sqlite3.DatabaseError(msg)

    def _open_writer_conn(self) -> sqlite3.Connection:
        """Connect + WAL/pragma/tokenizer setup for a writer connection (no schema init). Short timeout:
        jittered application-level retry handles contention, not SQLite's busy handler;
        isolation_level=None: explicit BEGIN IMMEDIATE."""
        conn = _connect_tracked_db(
            str(self.db_path), check_same_thread=False, timeout=1.0, isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            mode = apply_wal_with_fallback(conn, db_label="state.db")
            # "wal" is also the *assumed* mode when the on-disk probe was blocked by a concurrent opener
            # (#86515): the lock-free mode=ro read pool needs a confirmed WAL header, so confirm it here.
            # Unknown -> reads queue on the writer lock (slow but correct) instead of racing SQLITE_BUSY
            # on a file that may really be in rollback-journal mode.
            self._wal_active = mode == "wal" and _on_disk_journal_mode(conn) == "wal"
            # Existing WAL/SHM files may predate the main-file hardening;
            # normalize any sidecars that became visible during WAL setup.
            _secure_state_db_files(self.db_path)
            apply_database_pragmas(conn, db_label="state.db")
            conn.execute("PRAGMA foreign_keys=ON")
            self._fts_cjk_loaded = load_fts5_cjk_extension(conn)
        except BaseException:
            self._close_connection_quietly(conn)
            raise
        return conn

    def _connect_and_init(self) -> None:
        # Refuse before sqlite3.connect (under the startup lock) so we cannot mint
        # a replacement WAL while a live writer still holds a deleted sidecar inode.
        refuse_deleted_wal_generation(self.db_path)
        # Create/tighten the main database before sqlite3.connect() so a
        # permissive process umask can never expose a fresh profile store.
        _secure_state_db_files(self.db_path, create_main=True)
        self._conn = self._open_writer_conn()
        self._init_schema()

    def _connect_and_init_with_lock_patience(self) -> None:
        """Open + init, waiting out a sibling's write lock with jittered patience:
        _init_schema's DDL runs on a 1s-timeout connection, so a sibling's VACUUM
        or checkpoint used to fail the ENTIRE open and callers disabled
        persistence for the whole run. Non-lock errors propagate immediately."""
        # Lock contention during open: _init_schema's DDL/reconcile statements run on a 1s-timeout
        # connection with no retry, so a sibling process holding the write lock (VACUUM, TRUNCATE checkpoint
        # at close, a long FTS pass from an older still-running install) used to fail the ENTIRE open —
        # callers then disable persistence for the whole run ("Failed to initialize SessionDB ... database
        # is locked", #74478). The store is healthy; wait it out with the same jittered patience the write
        # path uses.
        deadline = time.monotonic() + self._WRITE_PATIENCE_S
        while True:
            try:
                self._connect_and_init()
                return
            except sqlite3.OperationalError as exc:
                if not is_sqlite_lock_error(exc):
                    raise
                self._close_connection_quietly(self._conn)
                now = time.monotonic()
                if now >= deadline:
                    log_write_lock_holders(self.db_path, self._WRITE_PATIENCE_S)
                    raise
                jitter = random.uniform(self._WRITE_RETRY_SLOW_MIN_S, self._WRITE_RETRY_SLOW_MAX_S)
                time.sleep(min(jitter, max(deadline - now, 0.001)))

    # ── Read-path split ──

    def _get_read_conn(self) -> Optional[sqlite3.Connection]:
        """Open a fresh read-only connection, or None when unavailable (callers
        return it to self._read_pool). WAL only: WAL readers never block on the
        writer, so reads skip self._lock; under DELETE journal mode (NFS fallback)
        readers hit SQLITE_BUSY storms, so the legacy locked path stays. Autocommit
        reads see everything committed so far (read-your-writes for flush-then-search)."""
        if not self._wal_active or self.read_only:
            return None
        with self._read_conns_lock:
            failed_at = self._read_open_failed_at
            backing_off = failed_at and time.monotonic() - failed_at < _READ_OPEN_RETRY_SECONDS
            if self._read_conns_closed or backing_off:
                return None
        # Permit BEFORE the open: openers race for permits, not descriptors.
        if not self._read_budget.acquire(self):
            logger.debug(
                "read pool at capacity (%d) for %s; serving this read from the "
                "locked writer connection", _READ_POOL_MAX, self.db_path,
            )
            return None
        conn = None  # bound before the try so the handlers can close a half-open one
        try:
            conn = self._connect_read_only(timeout=_READ_BUSY_TIMEOUT_S)
            apply_database_pragmas(conn, db_label="state.db")
            if self._fts_cjk_loaded:  # registers in the connection, not the file: ro is fine
                load_fts5_cjk_extension(conn)
        except BaseException as exc:
            # A half-open connection (open ok, extension load failed) is a live tracked descriptor,
            # the leak shape this pool exists to fix; a stranded permit would shrink the read
            # path by one slot forever. (Not _close_read_conn: callers release their own permit.)
            if conn is not None:
                self._close_conn_logged(conn, "partially-opened read conn")
            self._read_budget.release()
            if not isinstance(exc, sqlite3.Error):
                raise
            with self._read_conns_lock:
                self._read_open_failed_at = time.monotonic()
            logger.debug("read-only connection open failed for %s", self.db_path, exc_info=True)
            return None
        return conn

    def _evict_one_idle_read_conn(self) -> bool:
        """Close one idle pooled connection (a peer on the same file wants its permit); never a live one."""
        try:
            conn = self._read_pool.get_nowait()
        except queue.Empty:
            return False
        self._close_read_conn(conn)
        return True

    def _close_read_conn(self, conn) -> None:
        """Close a pooled read connection and release its permit even when the close fails (withholding
        it would narrow the read path forever). Over-releasing the BoundedSemaphore raises ValueError."""
        try:
            self._close_conn_logged(conn, "read-conn")
        finally:
            self._read_budget.release()

    def _checkout_read_conn(self) -> Optional[sqlite3.Connection]:
        """Borrow a read connection, opening on a miss; None when the read path is unavailable.
        A pool hit costs no permit (the connection already holds one)."""
        if not self._wal_active or self.read_only:
            return None
        try:
            return self._read_pool.get_nowait()
        except queue.Empty:
            return self._get_read_conn()

    @contextmanager
    def _read_ctx(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for read-only statements: a pooled read-only
        connection with NO lock under WAL; otherwise (non-WAL, open failure,
        ceiling reached) the writer connection under self._lock — deliberate
        degradation: slower beats EMFILE, which the supervisor cannot see."""
        conn = self._checkout_read_conn()
        if conn is not None:
            try:
                yield conn
            finally:
                returned = False
                with self._read_conns_lock:
                    if not self._read_conns_closed:
                        try:
                            self._read_pool.put_nowait(conn)
                            returned = True
                        except queue.Full:
                            pass
                if not returned:
                    # close() drained the pool (or queue.Full: unreachable while
                    # permits == maxsize, load-bearing if they drift): surplus.
                    self._close_read_conn(conn)
            return
        with self._lock:
            if self._conn is None:  # close() raced a still-unwinding reader
                self._reopen_after_close_locked(context="read")
            conn = cast(sqlite3.Connection, self._conn)
            if self._wal_active or self.read_only:  # WAL: no SHARED-lock wait; ro: opened with the read budget
                yield conn
                return
            # DELETE-mode writer connection: wait out a sibling's commit like a pooled reader would.
            previous_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            conn.execute(f"PRAGMA busy_timeout={int(_READ_BUSY_TIMEOUT_S * 1000)}")
            try:
                yield conn
            finally:
                with suppress(sqlite3.Error):
                    conn.execute(f"PRAGMA busy_timeout={int(previous_ms)}")

    def _reopen_after_close_locked(self, context: str = "write") -> None:
        """Reopen the writer after ``close()`` raced a live caller (a teardown owner
        set ``_conn = None`` while a worker still had a transcript flush to land).
        Loud (WARNING) and bounded (only after an explicit close()). Caller holds
        ``self._lock``. No _init_schema: no DDL races with siblings during teardown."""
        if self.read_only:
            raise sqlite3.ProgrammingError(
                f"SessionDB for {self.db_path} was closed (read-only handle); "
                f"cannot serve a {context} after close()"
            )
        # A reopen resolves the PATH again: a replaced file would be written through stale WAL/shm
        # assumptions; a quarantined handle must never hand a fresh connection to a damaged file.
        if self._db_corrupt and not (self._db_replaced or self._db_file_was_replaced()):
            raise self._corrupt_error(
                f"state.db connection for {self.db_path} is quarantined after "
                f"structural corruption; refusing to reopen for a {context} "
                "after close(). "
            )
        self._halt_if_db_generation_changed()
        logger.warning(
            "state.db connection for %s was closed while a %s was still in "
            "flight — reopening (teardown/worker race, #94736)", self.db_path, context,
        )
        try:
            self._conn = self._open_writer_conn()
        except Exception as exc:
            raise sqlite3.OperationalError(
                f"state.db connection was closed while a {context} was still "
                f"in flight (a session-teardown path called close() before "
                f"this worker finished — #94736) and the automatic reopen failed: {exc}"
            ) from exc
        if self._wal_active:  # a reopened writer is a live generation holder like the first open
            self._wal_lock_guard = _lockguard.hold(self.db_path)

    def _execute_write(
        self, fn: Callable[[sqlite3.Connection], T], patience_s: Optional[float] = None,
    ) -> T:
        """Run *fn(conn)* inside BEGIN IMMEDIATE with jittered lock retry; commit
        is handled here (callers must not commit). Returns *fn*'s result.
        BEGIN IMMEDIATE takes the WAL write lock up front so contention surfaces
        immediately; on locked/busy the Python lock is released, a jitter slept,
        and the WHOLE callback retried — *fn* must stay idempotent under retry."""
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        deadline = time.monotonic() + patience_s
        compression_deadline: Optional[float] = None  # set on the first compression-busy collision
        # One retry for SQLITE_IOERR raised by BEGIN IMMEDIATE itself (callback not run: nothing
        # replayed). Once fn has started, an IOERR leaves settlement unknown and must propagate.
        # The callback has not run at that point, so there is no durable effect to replay and the retry is
        # exactly-once safe (#99502's contract). Once the callback starts, an IOERR leaves the write's
        # settlement unknown and must propagate — this helper owns non-idempotent transcript/counter
        # mutations, not just idempotent UPSERTs.
        ioerr_begin_retried = False
        while True:
            self._raise_if_db_corrupt(storage=True)
            # NOTE: the replaced/generation live probe runs INSIDE the lock below,
            # not here. close() mutates _conn and _db_sidecar_identity under that
            # same lock, ending the WAL generation (SQLite unlinks the -wal/-shm
            # sidecars on a clean close). A lock-free probe that races close() can
            # observe the mid-teardown state — sidecars already unlinked while
            # _db_sidecar_identity is not yet cleared — and misclassify this
            # process's OWN clean close as an externally deleted generation,
            # raising a sticky DeletedWalGenerationError that permanently refuses
            # later writes (#105567). Inside the lock the probe only ever sees the
            # stable post-close state (identity cleared → adopt / reopen path).
            fn_started = False
            try:
                with self._lock:
                    self._raise_if_db_replaced()
                    if self._conn is None:  # close() raced this writer
                        self._reopen_after_close_locked(context="write")
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        fn_started = True
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                # Success — periodic best-effort checkpoint + FTS merge.
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                if self._write_count % self._FTS_MERGE_EVERY_N_WRITES == 0:
                    self._try_incremental_merge_fts()
                return result
            except SessionCompressionInProgressError:
                # Transient (see _COMPRESSION_BUSY_WAIT_S): a steer landing mid-compression must not abort.
                # A live foreign compression lock is transient: the compressor publishes in a couple of
                # seconds. Without any wait, a steer that lands mid-compression aborts the user's turn as
                # session_persistence_failed and sends the operator hunting disk space that was never the
                # problem (#75083). The budget is _COMPRESSION_BUSY_WAIT_S, not the write-lock patience: the
                # lease is a correctness boundary, so a writer still locked out after a short wait must be
                # refused rather than left to land a stale turn once a long-running or wedged compression
                # finally lets go.
                if compression_deadline is None:
                    compression_deadline = min(time.monotonic() + self._COMPRESSION_BUSY_WAIT_S, deadline)
                if self._sleep_before_write_retry(
                    compression_deadline, self._COMPRESSION_BUSY_WAIT_S
                ):
                    continue
                raise
            except sqlite3.Error as exc:
                # 'no more rows' is a transient engine error on contended WAL appends (some builds
                # raise it as InterfaceError, a sibling of DatabaseError): retry like locked/busy.
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                err_msg = str(exc).lower()
                if isinstance(exc, sqlite3.OperationalError):
                    if is_sqlite_lock_error(exc):
                        if self._sleep_before_write_retry(deadline, patience_s):
                            continue
                        # Say what actually happened, not disk/permission damage. The holder goes to
                        # the log, not the message: classify_persistence_error() buckets by phrase and
                        # a holder's argv (a worktree named fix-corrupt-db) would flip the bucket.
                        log_write_lock_holders(self.db_path, patience_s)
                        raise sqlite3.OperationalError(
                            f"database is locked (another Hermes process held the "
                            f"state.db write lock for over {patience_s:.0f}s — "
                            "likely a long maintenance operation such as VACUUM, "
                            "a large WAL checkpoint, or an older pre-update "
                            "process; the database itself is healthy)"
                        ) from exc
                    if (
                        _DISK_IO_ERROR_MARKER in err_msg and not fn_started and not ioerr_begin_retried
                        and self._sleep_before_write_retry(deadline, patience_s)
                    ):
                        # Retry on the SAME connection: close()+reopen would cancel this process's
                        # POSIX locks for every sibling (howtocorrupt §2.2).
                        ioerr_begin_retried = True
                        continue
                    raise  # non-lock error, callback already ran, or patience exhausted
                if isinstance(exc, sqlite3.DatabaseError):
                    # An out-of-band replace surfaces as this same corruption class; in-file repair
                    # on a NEW generation amplifies the damage.
                    if (
                        "not a database" in err_msg or is_malformed_db_error(exc)
                        or self._is_fts_write_corruption_error(exc)
                    ):
                        with self._lock:
                            self._raise_if_db_replaced()
                    # Corrupt FTS shadow tables fail every write via the sync triggers while canonical
                    # rows are intact: detach the derived indexes atomically and retry (never rebuild here).
                    if self._enter_fts_fail_open(exc, deadline=deadline, patience_s=patience_s):
                        continue
                    # What survives both checks is structural damage: quarantine.
                    if self._is_structural_corruption_error(exc):
                        self._halt_db_corrupt(exc)
                raise

    def _write_sql(
        self, sql: str, params: Any = (), *, many: bool = False, patience_s: Optional[float] = None,
    ) -> None:
        """Run one INSERT/UPDATE/DELETE through ``_execute_write``."""
        def _do(conn):
            (conn.executemany if many else conn.execute)(sql, params)
        self._execute_write(_do, patience_s=patience_s)

    def _write_rowcount(self, sql: str, params: Any = (), *, patience_s: Optional[float] = None) -> int:
        """Run one UPDATE/DELETE through ``_execute_write``; return rows changed
        (``SELECT changes()`` when the driver reports None / negative)."""
        def _do(conn):
            rowcount = conn.execute(sql, params).rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        return self._execute_write(_do, patience_s=patience_s)

    def _read_one(self, sql: str, params: Any = ()) -> Optional[sqlite3.Row]:
        """``fetchone()`` of one read-only statement via ``_read_ctx``."""
        return self._read_retrying_ioerr(lambda conn: conn.execute(sql, params).fetchone())

    def _read_all(self, sql: str, params: Any = ()) -> list[sqlite3.Row]:
        """``fetchall()`` of one read-only statement via ``_read_ctx``."""
        return self._read_retrying_ioerr(lambda conn: conn.execute(sql, params).fetchall())

    def _read_retrying_ioerr(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run an idempotent SELECT through ``_read_ctx``, retrying a transient SQLITE_IOERR.

        A warm ``mode=ro`` pooled reader can hit the same millisecond-wide WAL transition window as a
        read-only OPEN (#100436) when its statement executes or steps: a sibling process's checkpoint /
        WAL reset / frame flush surfaces ``disk I/O error`` because a read-only connection cannot rewrite
        the -shm index (#100871, WSL2 ext4-on-vhdx, multi-process). The window closes on its own, so
        the statement is replayed on the SAME connection within the read-only IOERR budget -- never
        closed and reopened (close() cancels this process's POSIX locks for every sibling connection),
        never quarantined (busy is not broken). A persistent IOERR exhausts the budget and propagates."""
        for attempt in range(_READ_ONLY_IOERR_RETRY_ATTEMPTS + 1):
            try:
                with self._read_ctx() as conn:
                    return fn(conn)
            except sqlite3.OperationalError as exc:
                if attempt >= _READ_ONLY_IOERR_RETRY_ATTEMPTS or _DISK_IO_ERROR_MARKER not in str(exc).lower():
                    note_storage_error(self.db_path, exc)
                    raise
                time.sleep(_READ_ONLY_IOERR_RETRY_BACKOFF_S)
            except sqlite3.DatabaseError as exc:
                # A reader is often the only observer (the sidebar poll on a store nobody is
                # writing to): publish structural damage so the list is not read as empty.
                note_storage_error(self.db_path, exc)
                raise

    def _ensure_db_file_generation(self) -> None:
        """Mint a once-per-file generation stamp (state_meta + application_id). First opener wins (INSERT
        OR IGNORE); application_id is written only while 0 so racers converge. PASSIVE checkpoint only.

        See #45383.
        """
        if self.read_only or self._conn is None:
            return
        token = uuid.uuid4().hex
        try:
            with self._lock:
                # Read first: the stamp is minted once per file, and a no-op INSERT OR IGNORE
                # still takes the write lock — under a sibling's transaction it blocked for the
                # busy timeout and the except below then dropped the token entirely. First
                # opener still wins via INSERT OR IGNORE; racers converge on the re-read.
                row = self._conn.execute(
                    "SELECT value FROM state_meta WHERE key = ?", (_STATE_DB_GENERATION_KEY,),
                ).fetchone()
                if not (row and row[0]):
                    self._conn.execute(
                        "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
                        (_STATE_DB_GENERATION_KEY, token),
                    )
                    row = self._conn.execute(
                        "SELECT value FROM state_meta WHERE key = ?",
                        (_STATE_DB_GENERATION_KEY,),
                    ).fetchone()
                if row and row[0]:
                    token = str(row[0])
                pragma_row = self._conn.execute("PRAGMA application_id").fetchone()
                current = int(pragma_row[0] or 0) if pragma_row else 0
                if current == 0:
                    current = (int(token[:8], 16) & 0x7FFFFFFF) or 1
                    self._conn.execute(f"PRAGMA application_id={current}")
                self._db_file_application_id = current
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass
        except sqlite3.Error as exc:
            logger.debug("state.db generation stamp skipped: %s", exc)

    def _record_db_file_identity(self) -> None:
        """Snapshot inode plus the on-disk generation header when present."""
        self._db_file_identity = _stat_db_file_identity(self.db_path)
        self._db_sidecar_identity = _stat_sqlite_sidecar_identity(self.db_path)
        disk_id = _read_sqlite_application_id(self.db_path)
        if disk_id:
            self._db_file_application_id = disk_id
        elif self._conn is not None and not self._db_file_application_id:
            try:
                pragma_row = self._read_one("PRAGMA application_id")
            except sqlite3.Error:
                pragma_row = None
            if pragma_row and pragma_row[0]:
                self._db_file_application_id = int(pragma_row[0])

    def _db_file_was_replaced(self) -> bool:
        """True when the path no longer names the file this instance opened."""
        recorded = self._db_file_identity
        if recorded is not None and _stat_db_file_identity(self.db_path) != recorded:
            return True
        recorded_app = int(self._db_file_application_id or 0)
        if not recorded_app:
            return False
        # Header 0 = WAL not yet checkpointed, not a replace; a real replacement is nonzero.
        disk_app = _read_sqlite_application_id(self.db_path)
        return bool(disk_app and disk_app != recorded_app)

    def _wal_generation_was_lost(self) -> bool:
        """True when the WAL/SHM generation this handle opened is gone. Recorded
        generation: pure stat (no /proc walk on healthy writes). Empty identity
        (WAL appeared after open, or cleared by a clean close()): probe
        /proc/self/fd for deleted sidecars and adopt the current ones once clean."""
        recorded = self._db_sidecar_identity or {}
        base = os.fspath(self.db_path)
        if recorded:
            return any(
                _stat_db_file_identity(Path(base + suffix)) != ident for suffix, ident in recorded.items()
            )
        if not self._wal_active:  # no sidecar generation to lose; keep /proc off the hot path
            return False
        if sys.platform.startswith("linux"):
            watched = _watched_sqlite_sidecar_paths(self.db_path)
            try:
                for target, fd_path in _proc_fd_targets(os.getpid()):
                    canonical = _state_holders.canonical_sqlite_path(target)
                    if (" (deleted)" in target and canonical in watched
                            and _fd_is_truly_unlinked(fd_path, watched[canonical])):
                        return True
            except OSError:
                return False
        # Probe clean (or unavailable): adopt the current sidecar generation.
        current_identity = _stat_sqlite_sidecar_identity(self.db_path)
        if current_identity:
            self._db_sidecar_identity = current_identity
        return False

    def _halt_if_db_generation_changed(self) -> None:
        """Stop writes (logging once) when the file was replaced or its WAL/SHM generation
        is gone: never run in-file repair on a new generation, never keep committing on a
        split WAL. Both flags are sticky."""
        # A reopen resolves the PATH again — if the file at that path is no longer the one this instance
        # originally opened (out-of-band restore/cp/mv), reconnecting would write into the new generation
        # through stale WAL/shm assumptions (#89332). Refuse instead.
        if self._db_replaced or self._db_file_was_replaced():
            self._db_replaced = True
            self._disable_close_time_checkpoint()
            logger.error(_STATE_DB_REPLACED_MSG)
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._db_wal_generation_lost = True
            self._disable_close_time_checkpoint()
            try:
                self._capture_retired_generation("halt")
            except RetiredGenerationCaptureError as exc:
                logger.error(
                    "Could not capture the retired WAL generation of %s at halt: %s. close() retries "
                    "the capture and refuses to settle without it.", self.db_path, exc,
                )
            logger.error(_DELETED_WAL_GENERATION_MSG)
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)

    def _capture_retired_generation(self, trigger: str) -> Path:
        """Durably capture the lost WAL generation this handle still holds open, once per handle.

        The quarantine keeps the retired frames from being checkpointed under wrong page numbers,
        but they live only in an unlinked inode that dies with this process's last descriptor, and
        the canonical DeletedWalGenerationError remediation is to stop the writers. Capturing at the
        first halt (or at close(), whichever sees the loss first) makes "preserve" outlive the
        process. Raises RetiredGenerationCaptureError; nothing is mutated on failure."""
        with self._retired_capture_lock:
            if self._retired_generation_capture is not None:
                return self._retired_generation_capture
            artifact = capture_retired_wal_generation(
                self.db_path, sidecar_identity=dict(self._db_sidecar_identity or {}), trigger=trigger,
            )
            self._retired_generation_capture = artifact
        logger.warning(
            "Captured the retired WAL generation of %s at %s to %s; read its manifest.json before deciding "
            "whether those frames belong on top of the file now at the path.", self.db_path, trigger, artifact,
        )
        return artifact

    def _raise_if_db_replaced(self) -> None:
        """Sticky-flag fast path (no log spam on every write), then the live probe."""
        if self._db_replaced:
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost:
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)
        self._halt_if_db_generation_changed()

    @classmethod
    def _is_structural_corruption_error(cls, exc: BaseException) -> bool:
        """Bare SQLITE_CORRUPT/NOTADB with no FTS provenance: canonical B-tree/schema/freelist damage,
        never repairable from the live write path."""
        return (
            isinstance(exc, sqlite3.DatabaseError)
            and not isinstance(exc, StateDbCorruptError)
            and not cls._is_fts_write_corruption_error(exc)
            and classify_persistence_error(exc) == "corrupt"
        )

    def _corrupt_error(self, prefix: str = "") -> StateDbCorruptError:
        """Build the quarantine error for this handle (message assembled once)."""
        return StateDbCorruptError(f"{prefix}{_STATE_DB_CORRUPT_MSG} (cause: {self._db_corrupt_reason})")

    def _halt_db_corrupt(self, exc: BaseException) -> None:
        """Quarantine this handle and raise; never run in-file repair here."""
        self._db_corrupt = True
        self._db_corrupt_reason = str(exc)
        # Publish the profile-level state first: readiness, /api/status and the session list
        # endpoints read it, and every other handle in this process refuses writes on it.
        mark_storage_corrupt(self.db_path, exc)
        self._disable_close_time_checkpoint()
        logger.error(
            "state.db %s reported structural corruption outside the FTS "
            "indexes (%s); quarantining this handle: no further writes, no "
            "automatic reopen, no explicit WAL checkpoint at close. Stop the "
            "gateway and run `hermes sessions recover --source %s --inspect-only`.", self.db_path, exc,
            self.db_path,
        )
        err = self._corrupt_error()
        for attr in ("sqlite_errorcode", "sqlite_errorname"):
            if getattr(exc, attr, None) is not None:
                setattr(err, attr, getattr(exc, attr))
        raise err from exc

    def _disable_close_time_checkpoint(self) -> bool:
        """Best-effort SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE (Python 3.12+): sqlite3's
        close() otherwise runs the internal last-connection checkpoint that wrote
        the incident's pages under wrong page numbers (see StateDbCorruptError and
        the generation-loss halts).
        <3.12 has no setconfig, so a lost-generation handle is retired unclosed
        instead (see close()): closing its last descriptor could both run that
        checkpoint and discard committed data present only in an unlinked WAL."""
        flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
        conn = self._conn
        setconfig = getattr(conn, "setconfig", None)
        if flag is None or setconfig is None:
            # Same predicate as _close_time_checkpoint_configurable() plus the per-instance
            # getattr: __init__ binds no retirement capability when either half is missing,
            # and close() must agree with that decision or the lost handle would neither
            # setconfig nor pin.
            return False
        try:
            setconfig(flag, True)
        except Exception:
            # No retention capability is bound on this runtime, so close() will let SQLite run the
            # checkpoint over the newer generation: say so where an operator can see it.
            logger.error(
                "Could not disable SQLite's close-time checkpoint on the quarantined handle for %s; "
                "closing it may checkpoint retired frames over the newer generation.",
                self.db_path, exc_info=True,
            )
            return False
        return True

    def _pin_connection(self, conn) -> None:
        """Retain the exact quarantined connection past GC and interpreter teardown (once per handle).

        Takes the connection as a parameter: callers are lock-held close paths, and the
        writer-conn thread-safety audit flags self._conn in functions outside `with self._lock`."""
        if not self._connection_pinned:
            self._retire_connection(conn)
            self._connection_pinned = True

    def _settle_lost_generation_locked(self) -> bool:
        """Capture the retired generation; return whether the handle must be retired unclosed.

        Where SQLite's close-time checkpoint cannot be switched off (no setconfig, Python < 3.12),
        sqlite3_close would write the retired frames over the newer generation, so the exact
        connection is retired unclosed instead. A failed capture leaves the handle open for a retry
        -- but the pin is taken FIRST on such a runtime: every production caller reaches close()
        through hermes_state_registry.release_or_close, which swallows the error, so an interpreter
        exit before the retry must not be able to checkpoint the stale frames either."""
        self._db_wal_generation_lost = True
        retire_without_close = not self._disable_close_time_checkpoint() and self._retire_connection is not None
        try:
            artifact = self._capture_retired_generation("close")
        except RetiredGenerationCaptureError as exc:
            if retire_without_close:
                self._pin_connection(self._conn)
            logger.error(
                "Could not capture the retired WAL generation of %s at close: %s. The handle stays open "
                "and close() retries the capture; those frames are NOT yet preserved.", self.db_path, exc,
            )
            raise
        logger.warning(
            "Skipping the close-time WAL checkpoint for %s: this handle's WAL/SHM generation "
            "was deleted or replaced; the retired generation is captured at %s. Stop the other "
            "writers before reopening and inspect the capture before deciding its disposition.",
            self.db_path, artifact,
        )
        if retire_without_close:
            logger.warning(
                "Retaining the quarantined connection for %s unclosed: this runtime cannot "
                "switch off SQLite's close-time checkpoint.", self.db_path,
            )
        return retire_without_close

    def _raise_if_db_corrupt(self, *, storage: bool = False) -> None:
        if self._db_corrupt:
            raise self._corrupt_error()
        if storage and storage_state(self.db_path) == STORAGE_CORRUPT:
            # Another handle in this process already saw structural damage on this file.
            # Quarantine this one before it touches SQLite; the error type is the same
            # StateDbCorruptError, so every transcript-diversion owner handles it unchanged.
            self._halt_db_corrupt(sqlite3.DatabaseError(
                "database disk image is malformed (reported earlier in this process: "
                f"{storage_corrupt_reason(self.db_path)})"))

    def _sleep_before_write_retry(self, deadline: float, patience_s: float) -> bool:
        """Sleep one jitter interval if the budget allows; True = retry, False = deadline passed. Small
        jitter for the first _WRITE_RETRY_SLOW_AFTER_S, then slow; never overshoots the deadline."""
        now = time.monotonic()
        if now >= deadline:
            return False
        slow = now - (deadline - patience_s) >= self._WRITE_RETRY_SLOW_AFTER_S
        jitter = random.uniform(*(
            (self._WRITE_RETRY_SLOW_MIN_S, self._WRITE_RETRY_SLOW_MAX_S) if slow
            else (self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
        ))
        time.sleep(min(jitter, max(deadline - now, 0.001)))
        return True

    def _foreign_state_db_holders(self) -> list[tuple[int, str]]:
        """Foreign processes holding this DB or its WAL sidecars (see hermes_state_holders)."""
        return _foreign_state_db_holders(self.db_path)

    def _quarantine_reason(self) -> Optional[str]:
        """Why this handle must not checkpoint or run in-file repair, or None. A corrupted image has
        torn B-trees; a replaced file or a deleted/replaced WAL generation would checkpoint under
        wrong page numbers into the main DB -- the shutdown-time cause of #105670. Precedence note:
        close() evaluates generation loss BEFORE calling this (and skips it entirely when lost —
        a lost generation settles through the capture path, not the quarantine advisory), while
        the halt path checks replaced first."""
        if self._db_corrupt:
            return f"structural corruption ({self._db_corrupt_reason})"
        if self._db_replaced:
            return "a replaced state.db file"
        if self._db_wal_generation_lost:
            return "a deleted WAL generation (split-brain)"
        return None

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint; never raises. PASSIVE never blocks writers;
        TRUNCATE corrupted B-trees on 65K+ page databases under exclusive-lock I/O pressure.

        Previous TRUNCATE strategy caused B-tree corruption on large databases (65K+ pages) due to the
        exclusive-lock I/O pressure from checkpointing thousands of frames at once (issue #45383).
        """
        if self._quarantine_reason() is not None:
            return
        try:
            with self._lock:
                if self._conn is None:
                    return  # closed underneath the timer: nothing to checkpoint, nothing to re-guard
                if self._wal_lock_guard:
                    _lockguard.hold(self.db_path, self._wal_lock_guard)  # a -shm minted after open
                result = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if result and result[1] > 0:
                    logger.debug("WAL checkpoint: %d/%d pages checkpointed", result[2], result[1])
        except Exception as exc:
            logger.warning("WAL checkpoint (PASSIVE) failed: %s", exc)

    def __enter__(self) -> "SessionDB":
        """``with SessionDB(path) as db:`` closes on exit; owners must release deterministically.

        Ownership of a SessionDB should be released explicitly. Historically an instance with a started
        token writer pinned ITSELF (bound-method writer target plus a strong ``atexit`` drain hook), so
        ``__del__`` never ran for exactly the instances that leaked descriptors (#88033). The writer now
        retires after an idle window and the atexit hook holds only a weak reference, so abandoned handles
        are eventually collectible — but "eventually, after the idle window and a GC cycle" is not a release
        policy. Call sites owning a handle are still expected to close it deterministically (see the
        ownership comments in ``run_agent.py`` and ``tui_gateway/methods_session.py``).
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False  # never suppress the caller's exception

    def close(self):
        """Drain queued token deltas, then a PASSIVE checkpoint on writable handles
        (NOT TRUNCATE: a full WAL reset races the gateway's live writer, tearing
        B-tree pages). A registry-shared instance RELEASES one refcount instead.

        Drains queued token deltas first (the background writer needs the connection). Read-only connections
        never request a checkpoint. See #45383.
        When this instance is shared (opened via ``hermes_state_registry.acquire``), ``close()`` RELEASES one
        refcount instead of tearing down the connection: the registry owns the lifecycle and only closes on
        the final release (#90837). This prevents one caller's close from tearing down the writer connection
        that other callers in the same process are still using — while still letting legacy ``close()`` call
        sites return their reference instead of leaking it.
        """
        if self._shared_registry_owned:
            from hermes_state_registry import release
            release(self)
            return
        self._stop_token_writer()
        hook, self._token_atexit_hook = self._token_atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)
        # Closed flag first: an in-flight reader then closes its own connection.
        with self._read_conns_lock:
            self._read_conns_closed = True
        while self._evict_one_idle_read_conn():
            pass
        with self._lock:
            if self._conn:
                generation_lost = not self.read_only and (
                    self._db_wal_generation_lost
                    or (bool(self._db_sidecar_identity) and self._wal_generation_was_lost())
                )
                # Loss is settled here, not at exit: the unlinked WAL inode dies with this process's
                # last descriptor (the capture raises and the handle stays open when it fails).
                retire_without_close = generation_lost and self._settle_lost_generation_locked()
                quarantine_reason = None if generation_lost else self._quarantine_reason()
                if quarantine_reason is not None:
                    logger.warning(
                        "Skipping the close-time WAL checkpoint for %s: this "
                        "handle observed %s. Take a snapshot of state.db, -wal and -shm "
                        "before restarting, then run `hermes sessions recover --source %s --inspect-only`.",
                        self.db_path, quarantine_reason, self.db_path,
                    )
                elif not self.read_only and not generation_lost:  # PASSIVE, not TRUNCATE (see docstring)
                    try:
                        # Every cron run_agent opens+closes a transient SessionDB, so a TRUNCATE here fires
                        # a full WAL reset many times/hour, racing the gateway's long-lived writer on large
                        # WAL databases and tearing hot B-tree pages -- the #45383 corruption this class's
                        # own periodic checkpoint was already made PASSIVE to avoid. TRUNCATE belongs only
                        # on a sole-opener/quiescent connection.
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except Exception as exc:
                        logger.debug("WAL checkpoint (PASSIVE) at close failed: %s", exc)
                _lockguard.release(self._wal_lock_guard)  # before the close: see release()
                if retire_without_close:
                    self._pin_connection(self._conn)
                    self._conn = None
                else:
                    conn, self._conn = self._conn, None
                    self._close_connection_quietly(conn)
                    # Only a clean close ends the generation; retain the recorded
                    # identity when retiring an unsafe handle.
                    self._db_sidecar_identity = {}
        self._read_budget.unregister(self)  # idempotent: a never-registered (failed-init) handle is a no-op

    def __del__(self) -> None:
        """Safety net: close() if the caller forgot. Attribute access stays
        guarded: module teardown order is undefined."""
        if self.__dict__.get("_conn") is not None:
            try:
                self.close()
            except Exception:
                pass

    # ── Async token accounting (SessionUsageMixin) ──
    # queue_token_counts() is a deque append; a single-writer thread applies deltas in
    # order, coalescing consecutive deltas whose route fields are EQUAL (so the merged
    # UPDATE equals applying them sequentially). Exact readers call flush_token_counts().
    _TOKEN_DELTA_SUM_FIELDS = (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
        "api_call_count",
    )
    _TOKEN_DELTA_COST_FIELDS = ("estimated_cost_usd", "actual_cost_usd")
    _TOKEN_DELTA_ROUTE_FIELDS = (
        "model", "cost_status", "cost_source", "pricing_version", "billing_provider", "billing_base_url",
        "billing_mode", "source", "task",
    )

    MAX_TITLE_LENGTH = 100

    # Title provenance, lowest to highest authority: auto-titling may only replace a
    # strictly lower-authority title (``derived`` -> ``llm`` once; never a user-typed name).
    TITLE_SOURCE_DERIVED = _TITLE_SOURCE_DERIVED
    TITLE_SOURCE_LLM = _TITLE_SOURCE_LLM
    TITLE_SOURCE_USER = _TITLE_SOURCE_USER
    _TITLE_SOURCE_RANK = {TITLE_SOURCE_DERIVED: 0, TITLE_SOURCE_LLM: 1, TITLE_SOURCE_USER: 2}

    # Bot Mode's canonical chat is resolved by exact-title lookup: the title IS the identity,
    # so _set_session_title refuses renames of a hidden row holding it.
    # Bot Mode's forever-chat registry: the session titled exactly this, on a bot's profile, IS the bot's
    # canonical chat — resolved by exact-title lookup on every open (no session-id pointer exists). See
    # #92473.
    CANONICAL_BOT_CHAT_TITLE = "Bot Chat"

    # ── Message storage constants (SessionMessagesMixin) ──
    # Prefix marking JSON-encoded structured content; NUL cannot collide with text.
    _CONTENT_JSON_PREFIX = "\x00json:"
    #: Reactions live inside ``display_metadata`` so they survive row rewrites.
    REACTIONS_METADATA_KEY = "reactions"
    # Columns every conversation projection decodes; ``active`` rides along so a display read
    # can split compaction-archived rows without a second query. Contract: must include every
    # ``agent.transcript_repair._OWNED_COLUMNS`` column, because replay stamps
    # ``transcript_row_snapshot(row)`` from these rows (token_count is hashed there, not decoded).
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
        "_compressed_summary, timestamp, token_count, active, api_content, display_kind, display_metadata, message_uid, "
        "absorbed_message_uids, tool_call_uids, tool_call_uid"
    )

    # ── Meta key/value (scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read state_meta[key] on self._lock (not _read_ctx): fts_rebuild_step reads progress before its
        write transaction and a WAL reader would not see it."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str, *, cursor: Optional[sqlite3.Cursor] = None) -> None:
        """Upsert state_meta[key]; with ``cursor`` the write is inline (the caller already holds a
        transaction — nesting BEGIN IMMEDIATE would deadlock).

        The ``receipt3/`` namespace is writer-owned protected evidence (receipt v3,
        schema/3): only the native provenance helper may write it, on this same
        connection inside its own approved transaction. The generic setter — with
        or without ``cursor`` — must never act as that writer."""
        if key == "receipt3" or key.startswith("receipt3/"):
            raise ValueError(
                "receipt3/ is the protected native-provenance namespace; "
                "use the native receipt writer, not set_meta"
            )
        sql = (
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        if cursor is not None:
            cursor.execute(sql, (key, value))
        else:
            self._write_sql(sql, (key, value))

    def retag_kanban_worker_sessions(self, workspaces_root: str) -> int:
        """Retag legacy kanban worker rows from ``cli`` to ``kanban`` by cwd under the board's workspaces
        root; gated once per root via state_meta. Returns rows retagged."""
        prefix = str(workspaces_root).rstrip("/\\")
        if not prefix:
            return 0
        gate = f"kanban_worker_source_retagged:{prefix}"
        if self.get_meta(gate) == "1":
            return 0
        def _do(conn):
            esc = _escape_like(prefix)
            cursor = conn.execute(
                "UPDATE sessions SET source = 'kanban' "
                "WHERE source = 'cli' AND (cwd = ? OR cwd LIKE ? ESCAPE '\\' "
                "OR cwd LIKE ? ESCAPE '\\')",
                (prefix, f"{esc}/%", f"{esc}\\\\%"),
            )
            # rowcount BEFORE set_meta reuses this cursor for its INSERT.
            retagged = cursor.rowcount or 0
            self.set_meta(gate, "1", cursor=cursor)
            return retagged
        return self._execute_write(_do)

    def is_kanban_owned_session(self, session_id: str) -> bool:
        """True when this session — or any segment of the compression lineage a resume would
        materialize — belongs to the Kanban dispatcher (``source``/``created_source`` =
        ``'kanban'``): the transcript of a worker run, not a human conversation (#68779).

        Both columns are checked: ``source`` is live routing state the dispatcher tags
        (``HERMES_SESSION_SOURCE=kanban``) and the legacy retag rewrites, while immutable
        ``created_source`` survives later surface flips. The lineage is the VERIFIED
        compression chain (``_resume_lineage_ids``) — exactly the rows a resume loads — so a
        worker transcript stays kanban-owned across rotations, while a delegate/branch child
        of a worker (a DIFFERENT conversation) is not swept in. Plain classification, not a
        gate: the resume-time guard (``hermes_cli/kanban_resume_guard.py``) owns the decision
        and the dispatcher-owned exemption."""
        lineage = self._resume_lineage_ids(session_id)
        if not lineage:
            return False
        rows = self._read_all(
            f"SELECT 1 FROM sessions WHERE id IN ({_placeholders(lineage)}) "
            "AND (source = 'kanban' OR created_source = 'kanban') LIMIT 1",
            tuple(lineage))
        return bool(rows)

    def list_meta_prefix(self, prefix: str) -> list[tuple[str, str]]:
        """``[(key, value), ...]`` for state_meta keys starting with the literal
        ``prefix`` (LIKE wildcards escaped) — e.g. ``loop:<session_id>`` rows.

        The ``receipt3/`` namespace is refused here: it materializes the whole
        match set and is not the bounded receipt reader (receipt v3, schema/3)."""
        if prefix == "receipt3" or prefix.startswith("receipt3/"):
            raise ValueError(
                "receipt3/ is the protected native-provenance namespace; "
                "the generic prefix scan is not the receipt reader"
            )
        if not prefix:
            return []
        rows = self._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'", (_escape_like(prefix) + "%",),
        )
        return [(row[0], row[1]) for row in rows]

    @classmethod
    def open_quiet_recovery_admission(cls, *, db_path, init_request) -> "QuietRecoveryHandle":
        """Open the finite continuation-only quiet-recovery mode (receipt r4 C02).

        This is the ONLY entry into recovery: a no-create ``mode=rw`` open of an
        existing *settled* store that runs no schema init, no repair, no
        generation mint and no usermerge (see :class:`QuietRecoveryHandle`). It
        refuses here for: a foreign/empty store, or any store that is not the
        settled Hermes schema. Only an approved epoch_continuation request may
        ever reach it; the ordinary constructor and every generic writer must
        not obtain this handle.
        """
        return QuietRecoveryHandle.open(db_path=Path(db_path), init_request=init_request)


# ── quiet recovery admission (receipt v3, r4 C02 -- minimal native slice) ────
# Cadence for the postcommit FTS defer: at most every N approved continuation
# writes marks a deferred FTS due instead of executing usermerge/merge inside
# the recovery handle (native values: every 50th PASSIVE checkpoint / 1000th
# merge in the ordinary writer; this handle only ever defers).
QUIET_RECOVERY_FTS_DEFER_EVERY = 1000

_QUIET_REQUIRED_TABLES = frozenset({"state_meta", "sessions", "messages", "schema_version"})
_QUIET_REQUIRED_REQUEST = frozenset({
    "kind", "token", "owner_session_id", "proof_sha256", "old_epoch", "new_epoch",
    "old_generation", "old_head_sha256", "old_head_token_sha256",
    "boundary_roster_sha256", "alias_inventory_sha256",
})
# Exact stored token-record shape (attempt_token_index): the read-back path
# verifies these bytes, never a relaxed subset.
_QUIET_RECORD_KEYS = frozenset({
    "owner_session_id", "operation_id", "request_sha256", "operation_generation",
    "operation_sha256", "result_sha256",
})
# The epoch-transition bindings of the approved init request. Every committed
# continuation must carry exactly these values; only the token and the
# per-operation oldraw coordinates (old_generation/old_head_sha256/
# old_head_token_sha256) stay free per operation.
_QUIET_BOUND_BINDINGS = (
    "kind", "owner_session_id", "proof_sha256", "old_epoch", "new_epoch",
    "boundary_roster_sha256", "alias_inventory_sha256",
)


class _QuietTokenReplay(Exception):
    """Raised inside the write callback when the token is already committed with
    the same bytes: the transaction rolls back and the stored result is returned
    without counting a second operation."""

    def __init__(self, result):
        super().__init__("epoch continuation token already committed")
        self.result = result


def _validate_quiet_recovery_request(request) -> dict:
    if not isinstance(request, dict) or set(request) != _QUIET_REQUIRED_REQUEST:
        raise ValueError("recovery admission accepts only an approved epoch_continuation request")
    if request["kind"] != "epoch_continuation":
        raise ValueError("recovery admission accepts only an approved epoch_continuation request")
    owner_session_id = request["owner_session_id"]
    if not isinstance(owner_session_id, str) or not owner_session_id:
        raise ValueError("continuation owner_session_id must be a non-empty string")
    token = request["token"]
    if not isinstance(token, str) or not token:
        raise ValueError("continuation token must be a canonical uuid4 string")
    try:
        parsed_token = uuid.UUID(token)
    except (ValueError, AttributeError, TypeError):
        raise ValueError("continuation token must be a canonical uuid4 string") from None
    if (parsed_token.version != 4 or parsed_token.variant != uuid.RFC_4122
            or str(parsed_token) != token):
        raise ValueError("continuation token must be a canonical uuid4 string")
    for key in ("proof_sha256", "old_head_sha256", "old_head_token_sha256",
                "boundary_roster_sha256", "alias_inventory_sha256"):
        if key == "old_head_sha256":
            continue  # gated on old_generation below (null only for a first receipt)
        value = request[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("continuation request fields must be lowercase sha256 hex")
    for key in ("old_epoch", "new_epoch"):
        value = request[key]
        if (not isinstance(value, str) or not value.startswith("sqlite:")
                or len(value) != 71 or any(c not in "0123456789abcdef" for c in value[7:])):
            raise ValueError("continuation epochs must be sqlite:64lowerhex")
    generation = request["old_generation"]
    if type(generation) is not int or generation < 0:
        raise ValueError("old_generation must be a nonnegative int")
    old_head_sha256 = request["old_head_sha256"]
    if generation == 0:
        if old_head_sha256 is not None:
            raise ValueError(
                "a first continuation (old_generation 0) carries a null old head")
    elif (not isinstance(old_head_sha256, str) or len(old_head_sha256) != 64
            or any(c not in "0123456789abcdef" for c in old_head_sha256)):
        raise ValueError("continuation request fields must be lowercase sha256 hex")
    return request


def _quiet_result_from_record(record: dict) -> dict:
    """Exact read-back of one stored token record: exactly the frozen six fields,
    and the stored result digest must recompute from the returned result."""
    if type(record) is not dict or set(record) != _QUIET_RECORD_KEYS:
        raise ValueError("stored continuation token record is unreadable; refusing")
    result = {"token": record["operation_id"],
              "operation_sha256": record["operation_sha256"],
              "generation": record["operation_generation"]}
    if record["result_sha256"] != hashlib.sha256(_quiet_canonical(result)).hexdigest():
        raise ValueError("stored continuation result bytes no longer verify; refusing")
    return result


def _quiet_unique_config_object(pairs):
    """Strict config decode: a duplicate raw key refuses the continuation."""
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate model_config keys refuse the continuation")
        obj[key] = value
    return obj


def _quiet_config_object(raw):
    """Strictly decode the owner's model_config text; returns ``(config, text)``."""
    text = "" if raw is None else raw
    if not isinstance(text, str):
        raise ValueError("owner model_config is not readable; refusing the continuation")
    if not text.strip():
        return {}, ""
    try:
        config = json.loads(text, object_pairs_hook=_quiet_unique_config_object)
    except ValueError:
        raise ValueError(
            "owner model_config is not readable; refusing the continuation") from None
    if not isinstance(config, dict):
        raise ValueError("owner model_config is not an object; refusing the continuation")
    return config, text


def _quiet_protected_head(raw) -> Optional[dict]:
    """The owner's stored protected head (``model_config._message_copy_receipts_v2``)
    or None when absent; unreadable or off-shape heads refuse the continuation."""
    config, _text = _quiet_config_object(raw)
    head = config.get("_message_copy_receipts_v2")
    if head is None:
        return None
    validate_receipt3_head(head)
    return head


def _quiet_install_protected_head(raw, head: dict) -> str:
    """Write the new protected head into the raw model_config text by replacing
    only the head value bytes: every other lexeme of the stored config (the v1
    tokens included) keeps its exact bytes. With no head present the key is
    spliced into the existing object; a duplicate or damaged head region
    refuses."""
    validate_receipt3_head(head)
    head_text = _quiet_canonical(head).decode("utf-8")
    token = '"_message_copy_receipts_v2"'
    text = "" if raw is None else raw
    if not isinstance(text, str) or not text.strip():
        return "{" + token + ":" + head_text + "}"
    occurrences = []
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            break
        occurrences.append(index)
        start = index + 1
    if len(occurrences) > 1:
        raise ValueError("duplicate protected head keys refuse the continuation")
    if occurrences:
        index = occurrences[0]
        if not text[:index].rstrip().endswith(("{", ",")):
            raise ValueError(
                "damaged model_config around the protected head; refusing")
        rest = text[index + len(token):]
        stripped = rest.lstrip()
        if not stripped.startswith(":"):
            raise ValueError(
                "damaged model_config around the protected head; refusing")
        value_start = index + len(token) + (len(rest) - len(stripped)) + 1
        while value_start < len(text) and text[value_start] in " \t\r\n":
            value_start += 1
        try:
            _value, value_end = json.JSONDecoder().raw_decode(text, value_start)
        except ValueError:
            raise ValueError(
                "damaged model_config around the protected head; refusing") from None
        return text[:value_start] + head_text + text[value_end:]
    end = text.rstrip().rfind("}")
    if end < 0:
        raise ValueError("owner model_config is not an object; refusing the continuation")
    body = text[:end]
    separator = "" if body.rstrip().endswith("{") else ","
    return body + separator + token + ":" + head_text + text[end:]


def _quiet_fts_match_probe(conn: sqlite3.Connection, table: str):
    """SELECT-only FTS5 MATCH probe. True: the index answers; False: the table is
    absent; None: present but not verifiable -- fail closed."""
    try:
        conn.execute(f"SELECT count(*) FROM {table} WHERE {table} MATCH ?",
                     ("receiptprobe",)).fetchone()
        return True
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return False
        return None
    except sqlite3.DatabaseError:
        return None


def _quiet_settled_admission_checks(conn: sqlite3.Connection):
    """Fail-closed SELECT-only settled-store checks (receipt r4 C02); returns the
    probed ``(fts_enabled, trigram_available)`` real values, refuses otherwise."""
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'table'")}
    if not _QUIET_REQUIRED_TABLES <= tables:
        raise ValueError(
            "quiet recovery requires an existing settled Hermes store; "
            "no schema init/repair/generation mint is performed here")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None or row[0] != SCHEMA_VERSION:
        raise ValueError(
            "quiet recovery requires the settled schema version; an unsettled "
            "store is refined through the ordinary path first, never here")
    row = conn.execute(
        "SELECT value FROM state_meta WHERE key = ?", (_MESSAGE_UID_BACKFILL_DONE,)).fetchone()
    if row is None or row[0] != "1":
        raise ValueError("quiet recovery requires the finished message_uid backfill")
    row = conn.execute(
        "SELECT value FROM state_meta WHERE key = ?", (_STATE_DB_GENERATION_KEY,)).fetchone()
    if row is None or not str(row[0] or "").strip():
        raise ValueError("quiet recovery requires an existing db_file_generation stamp")
    row = conn.execute(
        "SELECT count(*) FROM state_meta WHERE key LIKE 'fts\\_rebuild\\_%' ESCAPE '\\' "
        "OR key LIKE 'fts\\_v22\\_trash\\_%' ESCAPE '\\'").fetchone()
    if row[0]:
        raise ValueError(
            "quiet recovery refuses a store with pending FTS rebuild/trash markers")
    if SessionDB._db_needs_fts_storage_upgrade(conn.cursor()):
        raise ValueError(
            "quiet recovery refuses a store whose FTS storage layout is not settled")
    fts_enabled = _quiet_fts_match_probe(conn, "messages_fts")
    if fts_enabled is not True:
        raise ValueError("quiet recovery requires a working base FTS index")
    trigram_available = _quiet_fts_match_probe(conn, "messages_fts_trigram")
    if trigram_available is None:
        raise ValueError("quiet recovery cannot verify the trigram FTS index; refusing")
    return fts_enabled, trigram_available


def _quiet_canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8", "strict")


class QuietRecoveryHandle:
    """Continuation-only recovery handle: the single narrow write path.

    Opened with a no-create ``mode=rw`` connection on an existing settled store
    (fail-closed settled checks in :func:`_quiet_settled_admission_checks`); it
    never runs the ordinary constructor's schema init, FTS repair, generation
    mint or usermerge. Generic writers (set_meta/append_message) and FTS
    maintenance entries (optimize_fts/rebuild_fts) refuse. Only
    :meth:`append_epoch_continuation` may write, and it runs on the EXISTING
    ``SessionDB._execute_write`` transaction core (the same function object)
    through the quiet hook set below -- no reopen, no checkpoint, no
    usermerge/merge, no fail-open repair -- binding the approved transition
    and the owner's stored protected head (oldraw CAS). It is token-idempotent
    (same token -> verified readback, no second insert, no count; same key
    with different bytes -> refused). At the cadence it only marks
    ``_fts_deferred_due``; ``_fts_enabled``/``_trigram_available`` keep their
    probed real values.
    """

    _quiet_recovery_admission = True

    # Constants the shared SessionDB transaction core reads on this handle.
    _WRITE_PATIENCE_S = SessionDB._WRITE_PATIENCE_S
    _COMPRESSION_BUSY_WAIT_S = SessionDB._COMPRESSION_BUSY_WAIT_S
    _WRITE_RETRY_SLOW_AFTER_S = SessionDB._WRITE_RETRY_SLOW_AFTER_S
    _WRITE_RETRY_MIN_S = SessionDB._WRITE_RETRY_MIN_S
    _WRITE_RETRY_MAX_S = SessionDB._WRITE_RETRY_MAX_S
    _WRITE_RETRY_SLOW_MIN_S = SessionDB._WRITE_RETRY_SLOW_MIN_S
    _WRITE_RETRY_SLOW_MAX_S = SessionDB._WRITE_RETRY_SLOW_MAX_S
    _CHECKPOINT_EVERY_N_WRITES = SessionDB._CHECKPOINT_EVERY_N_WRITES
    _FTS_MERGE_EVERY_N_WRITES = QUIET_RECOVERY_FTS_DEFER_EVERY
    # The native bounded FTS maintenance pass invoked once at the explicit exit
    # (r4 C02 step 7: 500 pages/index, 4 commands/pass) -- the EXISTING bounded
    # merge implementation, never a private copy.
    _FTS_MERGE_MAX_PAGES_PER_INDEX = SessionDB._FTS_MERGE_MAX_PAGES_PER_INDEX
    _FTS_MERGE_COMMANDS_PER_PASS = SessionDB._FTS_MERGE_COMMANDS_PER_PASS
    _FTS_TABLES = SessionDB._FTS_TABLES
    _present_fts_tables = SessionDB._present_fts_tables
    _fts_table_exists = SessionDB._fts_table_exists
    _merge_fts_incrementally = SessionDB._merge_fts_incrementally
    # The EXISTING transaction core and shared helpers -- never a private copy.
    _execute_write = SessionDB._execute_write
    _sleep_before_write_retry = SessionDB._sleep_before_write_retry
    _is_structural_corruption_error = SessionDB._is_structural_corruption_error
    _is_fts_write_corruption_error = SessionDB._is_fts_write_corruption_error

    def __init__(self, *, db_path, conn, init_request, fts_enabled, trigram_available):
        self.db_path = db_path
        self._conn = conn
        self._init_request = init_request
        self._db_file_identity = _stat_db_file_identity(db_path)
        self._db_replaced = False
        self._db_corrupt = False
        self._db_corrupt_reason = "unspecified"
        self._lock = threading.RLock()
        self._write_count = 0
        self._fts_enabled = fts_enabled
        self._trigram_available = trigram_available
        self._fts_deferred_due = False
        self._fts_usermerge_floor_applied = False
        self._closed = False
        # C02 step 3/4 boundary-roster admission: the single-owner admission
        # approves exactly its frozen owner; a roster admission approves the
        # finite roster (validated module-side against the approved anchors).
        self._approved_owners = (init_request["owner_session_id"],)
        self._roster_requests = None

    @classmethod
    def open(cls, *, db_path: Path, init_request: dict) -> "QuietRecoveryHandle":
        request = _validate_quiet_recovery_request(init_request)
        path = Path(db_path).resolve()
        # ``as_uri()`` percent-encodes the path (?/#/% stay path bytes); mode=rw
        # never creates a database, so a missing store is refused before any
        # connection exists.
        conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            fts_enabled, trigram_available = _quiet_settled_admission_checks(conn)
            return cls(db_path=path, conn=conn, init_request=request,
                       fts_enabled=fts_enabled, trigram_available=trigram_available)
        except BaseException:
            conn.close()
            raise

    @classmethod
    def open_roster(cls, *, db_path: Path, requests, approved_anchors) -> "QuietRecoveryHandle":
        """C02 steps 3/4: enter the finite quiet mode for an approved boundary
        roster (every boundary owner -- g0, v1-only or no-receipt included).

        The same settled no-create admission runs once for the whole roster
        (one store, one finite recovery mode); every request must carry the
        exact same frozen transition frame, the roster member set must equal
        the approved anchor scope exactly (never enlarged by the replacement,
        never narrowed), and the frozen request digest must match the
        canonical roster digest. The partial state of ALL owners is verified
        read-only before any continuation can be written.
        """
        if type(requests) not in (list, tuple) or not requests:
            raise ValueError("a roster admission requires at least one continuation request")
        validated = [_validate_quiet_recovery_request(dict(request)) for request in requests]
        frame = validated[0]
        for request in validated[1:]:
            for bound_field in ("kind", "proof_sha256", "old_epoch", "new_epoch",
                                "boundary_roster_sha256", "alias_inventory_sha256"):
                if request[bound_field] != frame[bound_field]:
                    raise ValueError(
                        "roster continuations do not share one approved transition frame")
        roster_entries = [{"owner_session_id": request["owner_session_id"],
                           "token": request["token"],
                           "old_generation": request["old_generation"],
                           "old_head_sha256": request["old_head_sha256"],
                           "old_head_token_sha256": request["old_head_token_sha256"]}
                          for request in validated]
        roster = validate_boundary_roster(roster_entries,
                                          approved_anchors=list(approved_anchors))
        if roster["roster_sha256"] != frame["boundary_roster_sha256"]:
            raise ValueError(
                "approved boundary roster digest does not match the frozen request")
        handle = cls.open(db_path=db_path, init_request=frame)
        try:
            handle._approved_owners = tuple(roster["owners"])
            handle._roster_requests = {request["owner_session_id"]: request
                                       for request in validated}
            handle.verify_roster_recovery_state()
        except BaseException:
            handle.close()
            raise
        return handle

    # ── refusals: generic writers and maintenance never write here ──
    def set_meta(self, *args, **kwargs):
        raise RuntimeError("generic set_meta is not reachable on the quiet recovery handle")

    def append_message(self, *args, **kwargs):
        raise RuntimeError("generic message writes are not reachable on the quiet recovery handle")

    def optimize_fts(self, *args, **kwargs):
        raise RuntimeError("FTS optimize is deferred on the quiet recovery handle")

    def rebuild_fts(self, *args, **kwargs):
        raise RuntimeError("FTS rebuild is not reachable on the quiet recovery handle")

    # ── quiet hook set for the shared SessionDB._execute_write core ──
    def _raise_if_db_replaced(self) -> None:
        """Quiet inode fence: the path must still name the file this handle opened."""
        if self._db_replaced:
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        recorded = self._db_file_identity
        if recorded is not None and _stat_db_file_identity(self.db_path) != recorded:
            self._db_replaced = True
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)

    def _raise_if_db_corrupt(self, *, storage: bool = False) -> None:
        if self._db_corrupt:
            raise self._corrupt_error()
        if storage and storage_state(self.db_path) == STORAGE_CORRUPT:
            self._halt_db_corrupt(sqlite3.DatabaseError(
                "database disk image is malformed (reported earlier in this process: "
                f"{storage_corrupt_reason(self.db_path)})"))

    def _corrupt_error(self, prefix: str = "") -> StateDbCorruptError:
        return StateDbCorruptError(f"{prefix}{_STATE_DB_CORRUPT_MSG} (cause: {self._db_corrupt_reason})")

    def _halt_db_corrupt(self, exc: BaseException) -> None:
        """Quiet quarantine: sticky local flags and raise; no process-wide publish."""
        self._db_corrupt = True
        self._db_corrupt_reason = str(exc)
        logger.error(
            "quiet recovery handle for %s saw structural corruption (%s); "
            "quarantining this handle: no further continuation writes.", self.db_path, exc,
        )
        err = self._corrupt_error()
        for attr in ("sqlite_errorcode", "sqlite_errorname"):
            if getattr(exc, attr, None) is not None:
                setattr(err, attr, getattr(exc, attr))
        raise err from exc

    def _reopen_after_close_locked(self, context: str = "write") -> None:
        raise RuntimeError(
            "quiet recovery handle is closed; it never reopens after close()")

    def _try_wal_checkpoint(self) -> None:
        """Quiet mode never checkpoints; the post-release pass owns that."""

    def _try_incremental_merge_fts(self) -> None:
        """Quiet mode never merges: mark the postcommit FTS defer instead."""
        self._fts_deferred_due = True

    def _enter_fts_fail_open(self, exc, *, deadline=None, patience_s=None) -> bool:
        return False  # no fail-open repair inside the recoverer

    # ── the single approved write ──
    def append_epoch_continuation(self, request: dict) -> dict:
        request = _validate_quiet_recovery_request(request)
        if self._closed or self._conn is None:
            raise RuntimeError("quiet recovery handle is closed")
        for field in _QUIET_BOUND_BINDINGS:
            if field == "owner_session_id":
                if request[field] not in self._approved_owners:
                    raise ValueError(
                        "continuation owner is not part of the approved boundary roster")
                continue
            if request[field] != self._init_request[field]:
                raise ValueError(
                    "continuation does not match the approved init request transition")
        owner_session_id = request["owner_session_id"]
        token = request["token"]
        key = encode_receipt3_token_key(owner_session_id, token)
        request_sha256 = hashlib.sha256(_quiet_canonical(request)).hexdigest()

        def _do(conn: sqlite3.Connection) -> dict:
            existing = conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
            if existing is not None:
                stored_existing = json.loads(existing[0])
                if existing[0] != _quiet_canonical(stored_existing).decode("utf-8"):
                    raise ValueError(
                        "stored continuation token record is not canonical; refusing")
                if stored_existing.get("request_sha256") != request_sha256:
                    raise ValueError(
                        "continuation token already committed with different bytes; refusing")
                # Exact readback: the stored result is only returned after the
                # record and the journal bytes it names still verify.
                replayed = _quiet_result_from_record(stored_existing)
                journal = read_operation(conn, owner_session_id,
                                         stored_existing["operation_generation"])
                if journal["header"]["operation_sha256"] != stored_existing["operation_sha256"]:
                    raise ValueError(
                        "stored continuation journal no longer matches its token record; refusing")
                raise _QuietTokenReplay(replayed)
            # oldraw / protected-head CAS: the owner row must exist; a stored head
            # must match the old coordinates exactly; absent = a first receipt,
            # and only with old_generation 0 -- a missing head never downgrades
            # an existing journal to "no evidence".
            row = conn.execute(
                "SELECT model_config FROM sessions WHERE id = ?", (owner_session_id,)).fetchone()
            if row is None:
                raise ValueError(
                    "continuation owner session is not part of this settled store")
            head = _quiet_protected_head(row[0])
            if head is not None:
                if (head["owner_session_id"] != owner_session_id
                        or head["active_epoch"] != request["old_epoch"]
                        or head["generation"] != request["old_generation"]
                        or head["head_sha256"] != request["old_head_sha256"]
                        or hashlib.sha256(_quiet_canonical(head)).hexdigest()
                        != request["old_head_token_sha256"]):
                    raise ValueError(
                        "continuation does not match the stored protected head (oldraw CAS)")
                old_ordinal = head["epoch_ordinal"]
                domain_sha256 = head["domain_sha256"]
                v1_binding_sha256 = head["v1_binding_sha256"]
            else:
                if request["old_generation"] != 0:
                    raise ValueError(
                        "continuation claims stored history but the owner carries "
                        "no protected head; refusing")
                journal_lo = "receipt3/o/" + owner_session_id.encode("utf-8").hex() + "/"
                if conn.execute(
                        "SELECT 1 FROM state_meta WHERE key >= ? AND key < ? LIMIT 1",
                        (journal_lo, journal_lo[:-1] + ":")).fetchone() is not None:
                    raise ValueError(
                        "continuation owner already carries journal entries but "
                        "no protected head; refusing")
                old_ordinal = None
                domain_sha256 = _RECEIPT3_DOMAIN
                v1_binding_sha256 = None
            operation_generation = request["old_generation"] + 1
            epoch_ordinal = 1 if old_ordinal is None else old_ordinal + 1
            # Q01 typed context observation on this same connection, inside the
            # transaction (the header carries the observed projection).
            contexts = observe_native_session_contexts(
                conn, owner_session_id=owner_session_id, actor_session_id=owner_session_id,
                kind="epoch_continuation", domain_sha256=domain_sha256)
            page = build_journal_page(
                owner_session_id=owner_session_id, epoch=request["new_epoch"],
                generation=operation_generation, page_ordinal=0,
                items=[{"kind": "native_result", "input_ordinal": None, "row_ref": None,
                        "disposition": "aggregate", "result_count": None}])
            header = build_epoch_continuation_header(
                request=request, request_sha256=request_sha256, domain_sha256=domain_sha256,
                epoch=request["new_epoch"], epoch_ordinal=epoch_ordinal,
                generation=operation_generation,
                previous_head_sha256=request["old_head_sha256"],
                pages=[page], native_session_contexts=contexts)
            operation_sha256 = header["operation_sha256"]
            # Insert-only journal entries under their immutable named keys.
            for named_key, value in (
                    (encode_receipt3_operation_key(owner_session_id, operation_generation),
                     _quiet_canonical(header)),
                    (encode_receipt3_page_key(owner_session_id, operation_generation, 0),
                     _quiet_canonical(page))):
                if conn.execute("SELECT 1 FROM state_meta WHERE key = ?", (named_key,)).fetchone():
                    raise ValueError("receipt3 journal entry already exists; refusing")
                conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                             (named_key, value.decode("utf-8")))
            # Range checkpoints, same transaction: seal the old epoch's exit tail
            # (read back from the stored member bytes), then apply the
            # completed-full-block rule to the new epoch. Insert-or-verified
            # bytes only -- never an overwrite.
            if old_ordinal is not None:
                seal_epoch_blocks(
                    conn, owner_session_id=owner_session_id, epoch=request["old_epoch"],
                    epoch_ordinal=old_ordinal, up_to_generation=request["old_generation"],
                    epoch_exit=True)
            seal_epoch_blocks(
                conn, owner_session_id=owner_session_id, epoch=request["new_epoch"],
                epoch_ordinal=epoch_ordinal, up_to_generation=operation_generation,
                epoch_exit=False)
            new_head = {"schema_version": "hermes-message-copy-receipts/3",
                        "protocol_revision": 3, "domain_sha256": domain_sha256,
                        "owner_session_id": owner_session_id,
                        "active_epoch": request["new_epoch"], "epoch_ordinal": epoch_ordinal,
                        "generation": operation_generation, "head_sha256": operation_sha256,
                        "v1_binding_sha256": v1_binding_sha256,
                        "activation_proof_sha256": request["proof_sha256"]}
            conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?",
                         (_quiet_install_protected_head(row[0], new_head), owner_session_id))
            result = {"token": token, "operation_sha256": operation_sha256,
                      "generation": operation_generation}
            record = {"owner_session_id": owner_session_id, "operation_id": token,
                      "request_sha256": request_sha256,
                      "operation_generation": operation_generation,
                      "operation_sha256": operation_sha256,
                      "result_sha256": hashlib.sha256(_quiet_canonical(result)).hexdigest()}
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?)",
                (key, _quiet_canonical(record).decode("utf-8")))
            return result

        try:
            return self._execute_write(_do)
        except _QuietTokenReplay as replay:
            return replay.result

    # ── C02 step 5: quiet reopen + partial-crash named-only back-projection ──

    @classmethod
    def resume(cls, *, db_path, init_request) -> "QuietRecoveryHandle":
        """Re-enter the finite quiet mode after a crash or a close (r4 C02 step 5).

        The only re-entry is quiet admission again: :meth:`open` re-runs the
        settled no-create ``mode=rw`` checks, and the ordinary constructor,
        the ordinary writer and any automatic reopen stay structurally unable
        to continue. The partial state is then re-read strictly (read-only):
        every committed token of the approved owner must read back exactly --
        canonical record, verified result, verified journal bytes carrying the
        frozen transition -- no orphaned journal bytes may exist, and the
        stored protected head must match the committed continuation or, before
        any commit, still carry the frozen old coordinates. Restarting the
        in-memory counters is not a reset of persisted state: the FTS defer
        flag is deliberately not persisted, and the post-release pass covers
        the missed maintenance (step 7).
        """
        handle = cls.open(db_path=db_path, init_request=init_request)
        try:
            handle.verify_partial_recovery_state()
        except BaseException:
            handle.close()
            raise
        return handle

    def enumerate_committed_continuations(self, owner_session_id=None) -> list:
        """Strict read-only enumeration of the owner's completed continuations
        under this frozen admission (default: the approved request's owner)."""
        if self._closed or self._conn is None:
            raise RuntimeError("quiet recovery handle is closed")
        if owner_session_id is None:
            owner_session_id = self._init_request["owner_session_id"]
        approved = {name: self._init_request[name] for name in _RECEIPT3_CONTINUATION_KEYS}
        return read_partial_recovery_state(
            self._conn, owner_session_id, approved_transition=approved)

    def verify_partial_recovery_state(self) -> dict:
        """Fail-closed read-only verification of the owner's partial recovery
        state; returns ``{"owner_session_id", "committed", "head"}``."""
        if self._closed or self._conn is None:
            raise RuntimeError("quiet recovery handle is closed")
        owner_session_id = self._init_request["owner_session_id"]
        committed = self.enumerate_committed_continuations(owner_session_id)
        if len(committed) > 1:
            raise ValueError(
                "more than one committed continuation for this frozen "
                "transition; refusing the reopen")
        row = self._conn.execute(
            "SELECT model_config FROM sessions WHERE id = ?", (owner_session_id,)).fetchone()
        if row is None:
            raise ValueError("continuation owner session is not part of this settled store")
        head = _quiet_protected_head(row[0])
        if committed:
            last = committed[-1]
            if head is None or (head["owner_session_id"] != owner_session_id
                                or head["active_epoch"] != self._init_request["new_epoch"]
                                or head["generation"] != last["generation"]
                                or head["head_sha256"] != last["operation_sha256"]):
                raise ValueError(
                    "protected head does not match the committed continuation; refusing")
        elif head is None:
            if self._init_request["old_generation"] != 0:
                raise ValueError(
                    "continuation claims stored history but the owner carries "
                    "no protected head; refusing")
        elif (head["owner_session_id"] != owner_session_id
                or head["active_epoch"] != self._init_request["old_epoch"]
                or head["generation"] != self._init_request["old_generation"]
                or head["head_sha256"] != self._init_request["old_head_sha256"]
                or hashlib.sha256(_quiet_canonical(head)).hexdigest()
                != self._init_request["old_head_token_sha256"]):
            raise ValueError(
                "stored protected head no longer carries the frozen old "
                "coordinates; refusing the reopen")
        return {"owner_session_id": owner_session_id, "committed": committed, "head": head}

    def final_readback_binding(self) -> dict:
        """The final readback binding recorded before the explicit exit (r4 C02
        step 7): one canonical digest over the enumerated completed set and the
        protected head; the independent acceptance decides on exactly these
        bytes."""
        state = self.verify_partial_recovery_state()
        message = {"owner_session_id": state["owner_session_id"],
                   "committed": state["committed"], "head": state["head"]}
        return {"owner_session_id": state["owner_session_id"],
                "committed": state["committed"], "head": state["head"],
                "readback_sha256": hashlib.sha256(_quiet_canonical(message)).hexdigest()}

    def exit_quiet(self, *, accepted_readback_sha256: str) -> dict:
        """Explicitly exit the finite quiet mode after the independent
        acceptance (r4 C02 step 7).

        The accepted final readback binding is re-checked first (a mismatch
        refuses the exit and runs nothing). Then exactly one native bounded FTS
        pass runs -- the existing bounded merge (500 pages/index, 4 commands
        per pass) -- AFTER the accepted recovery boundary, as normal
        maintenance: even when a crash lost the in-memory defer flag, the pass
        is scheduled here; its failure is reported and can never replay a
        committed continuation. The handle then closes and the ordinary
        handle/writer path reopens; the counters are not zeroed for anyone and
        the FTS flags keep their probed values.
        """
        if self._closed or self._conn is None:
            raise RuntimeError("quiet recovery handle is closed")
        if (not isinstance(accepted_readback_sha256, str) or len(accepted_readback_sha256) != 64
                or any(c not in "0123456789abcdef" for c in accepted_readback_sha256)):
            raise ValueError("the accepted final readback binding must be lowercase sha256 hex")
        binding = self.final_readback_binding()
        if binding["readback_sha256"] != accepted_readback_sha256:
            raise ValueError(
                "final readback binding does not match the accepted bytes; "
                "quiet mode is not exited")
        fts_pass = {"executed": 0, "failed": None}
        if self._fts_enabled:
            try:
                fts_pass["executed"] = self._merge_fts_incrementally(
                    max_pages=self._FTS_MERGE_MAX_PAGES_PER_INDEX,
                    max_commands=self._FTS_MERGE_COMMANDS_PER_PASS)
            except Exception as exc:
                # The accepted boundary stands; a maintenance failure is
                # reported, never allowed to replay a committed continuation.
                fts_pass["failed"] = f"{type(exc).__name__}: {exc}"[:200]
        report = {"owner_session_id": binding["owner_session_id"],
                  "write_count": self._write_count,
                  "fts_deferred_due_at_exit": self._fts_deferred_due,
                  "fts_enabled": self._fts_enabled,
                  "trigram_available": self._trigram_available,
                  "readback": binding, "fts_pass": fts_pass}
        self.close()
        return report

    # ── C02 steps 1-4 orchestration + final ALL-owner readback ──
    _C02_FENCE_KEYS = frozenset({"kind", "approved", "holders", "fence_id"})

    def append_roster_continuations(self) -> dict:
        """C02 step 4: one epoch continuation per approved boundary owner.

        Each owner's continuation runs on its own token/head CAS (the existing
        single approved write path), in canonical owner order; a partial
        failure leaves the committed set to the read-only back-projection
        (step 5), never a silent repair.
        """
        if self._closed or self._conn is None:
            raise RuntimeError("quiet recovery handle is closed")
        if self._roster_requests is None:
            raise RuntimeError("this admission is not a boundary roster admission")
        results = {}
        for owner in self._approved_owners:
            results[owner] = self.append_epoch_continuation(self._roster_requests[owner])
        return results

    def _roster_transition_blocks(self) -> dict:
        requests = self._roster_requests
        if requests is None:
            requests = {self._init_request["owner_session_id"]: self._init_request}
        return {owner: {name: request[name] for name in _RECEIPT3_CONTINUATION_KEYS}
                for owner, request in requests.items()}

    def verify_roster_recovery_state(self, *, complete: bool = False) -> dict:
        """Fail-closed read-only verification of the roster's recovery state.

        Every approved owner's committed continuation set (0 or 1 while the
        roster is incomplete, exactly 1 for the final ALL readback) is re-read
        strictly against its own frozen transition block, no continuation
        outside the roster may exist, and each protected head must match the
        committed continuation or still carry that owner's frozen old
        coordinates. Returns the per-owner entries and the canonical
        ALL-owner readback digest.
        """
        if self._closed or self._conn is None:
            raise RuntimeError("quiet recovery handle is closed")
        if type(complete) is not bool:
            raise ValueError("complete must be a bool")
        requests = self._roster_requests
        if requests is None:
            requests = {self._init_request["owner_session_id"]: self._init_request}
        inventory = verify_roster_continuation_inventory(
            self._conn, list(self._approved_owners), self._roster_transition_blocks(),
            complete=complete)
        owners = []
        for owner in self._approved_owners:
            request = requests[owner]
            committed = inventory["owners"][owner]
            row = self._conn.execute(
                "SELECT model_config FROM sessions WHERE id = ?", (owner,)).fetchone()
            if row is None:
                raise ValueError(
                    "continuation owner session is not part of this settled store")
            head = _quiet_protected_head(row[0])
            if committed:
                last = committed[-1]
                if head is None or (head["owner_session_id"] != owner
                                    or head["active_epoch"] != request["new_epoch"]
                                    or head["generation"] != last["generation"]
                                    or head["head_sha256"] != last["operation_sha256"]):
                    raise ValueError(
                        "protected head does not match the committed continuation; refusing")
            elif head is None:
                if request["old_generation"] != 0:
                    raise ValueError(
                        "continuation claims stored history but the owner carries "
                        "no protected head; refusing")
            elif (head["owner_session_id"] != owner
                    or head["active_epoch"] != request["old_epoch"]
                    or head["generation"] != request["old_generation"]
                    or head["head_sha256"] != request["old_head_sha256"]
                    or hashlib.sha256(_quiet_canonical(head)).hexdigest()
                    != request["old_head_token_sha256"]):
                raise ValueError(
                    "stored protected head no longer carries the frozen old "
                    "coordinates; refusing")
            owners.append({"owner_session_id": owner, "committed": committed, "head": head})
        return {"owners": owners, "readback_sha256": final_all_owner_readback(owners),
                "complete": all(len(entry["committed"]) == 1 for entry in owners)}

    def final_roster_readback_binding(self) -> dict:
        """The final ALL-owner readback binding recorded before the explicit
        quiet exit (r4 C02 step 7, roster form)."""
        state = self.verify_roster_recovery_state(complete=True)
        return {"owners": state["owners"], "readback_sha256": state["readback_sha256"]}

    def exit_roster_quiet(self, *, accepted_readback_sha256: str) -> dict:
        """Explicitly exit the finite quiet mode after the independent
        acceptance of the ALL-owner readback (r4 C02 step 7, roster form).

        The accepted binding is re-checked first (a mismatch refuses the exit
        and runs nothing); then exactly one native bounded FTS pass runs AFTER
        the accepted boundary, exactly as in the single-owner exit.
        """
        if self._closed or self._conn is None:
            raise RuntimeError("quiet recovery handle is closed")
        if (not isinstance(accepted_readback_sha256, str) or len(accepted_readback_sha256) != 64
                or any(c not in "0123456789abcdef" for c in accepted_readback_sha256)):
            raise ValueError("the accepted final readback binding must be lowercase sha256 hex")
        binding = self.final_roster_readback_binding()
        if binding["readback_sha256"] != accepted_readback_sha256:
            raise ValueError(
                "final all-owner readback binding does not match the accepted bytes; "
                "quiet mode is not exited")
        fts_pass = {"executed": 0, "failed": None}
        if self._fts_enabled:
            try:
                fts_pass["executed"] = self._merge_fts_incrementally(
                    max_pages=self._FTS_MERGE_MAX_PAGES_PER_INDEX,
                    max_commands=self._FTS_MERGE_COMMANDS_PER_PASS)
            except Exception as exc:
                fts_pass["failed"] = f"{type(exc).__name__}: {exc}"[:200]
        report = {"owner_session_ids": [entry["owner_session_id"] for entry in binding["owners"]],
                  "write_count": self._write_count,
                  "fts_deferred_due_at_exit": self._fts_deferred_due,
                  "fts_enabled": self._fts_enabled,
                  "trigram_available": self._trigram_available,
                  "readback": binding, "fts_pass": fts_pass}
        self.close()
        return report

    @classmethod
    def run_c02_recovery_orchestration(cls, *, original_path, replacement_path, fence,
                                       requests, approved_anchors, proof=None):
        """C02 steps 1-4 full orchestration + the final ALL-owner readback.

        Step 1 validates the external holder-drain fence attestation (an
        approved orchestration input -- it is never manufactured here and no
        pane is killed); step 2 runs the independent read-only full-table joint
        between the original latest and the selected byte/data-complete
        replacement; step 3 enters the roster quiet admission on the
        replacement, re-measures the joint read-only AFTER the open and
        re-verifies the frozen proof (whose roster digest must equal the frozen
        request digest); step 4 writes exactly one epoch continuation per
        boundary owner under its own token/head CAS; then the final ALL-owner
        readback binding (complete roster) is read back read-only. Returns
        ``(handle, report)`` -- the caller owns the independent acceptance and
        the explicit exit; the original file is never opened for writing.
        """
        if type(fence) is not dict or set(fence) != set(cls._C02_FENCE_KEYS):
            raise ValueError("external holder-drain fence attestation is required")
        if fence["kind"] != "external_holder_drain_fence" or fence["approved"] is not True:
            raise ValueError(
                "the quiet recovery orchestration requires an approved external fence")
        if type(fence["holders"]) is not list or not fence["holders"] \
                or not all(isinstance(holder, str) and holder for holder in fence["holders"]):
            raise ValueError("the fence attestation must list every drained holder")
        if not isinstance(fence["fence_id"], str) or not fence["fence_id"]:
            raise ValueError("the fence attestation requires a fence_id")
        steps = {"step1_fence": {"fence_id": fence["fence_id"],
                                 "holders": len(fence["holders"]), "approved": True}}
        copy_report = verify_independent_copy_joint(original_path, replacement_path)
        steps["step2"] = {"joint_sha256": copy_report["joint_sha256"],
                          "table_count": copy_report["table_count"]}
        handle = cls.open_roster(db_path=Path(replacement_path), requests=list(requests),
                                 approved_anchors=list(approved_anchors))
        try:
            post_open = verify_independent_copy_joint(original_path, replacement_path)
            if post_open["joint_sha256"] != copy_report["joint_sha256"]:
                raise ValueError(
                    "the quiet admission open changed the replacement joint; refusing")
            if proof is not None:
                verified = verify_frozen_proof(replacement_path, proof)
                if proof.get("roster_sha256") != handle._init_request["boundary_roster_sha256"]:
                    raise ValueError("the frozen proof does not bind this boundary roster")
                steps["step3_proof"] = verified
            pre_commit = handle.verify_roster_recovery_state()
            steps["step3"] = {"post_open_joint_sha256": post_open["joint_sha256"],
                              "owners": len(pre_commit["owners"]),
                              "complete": pre_commit["complete"]}
            results = handle.append_roster_continuations()
            steps["step4"] = {owner: results[owner]["generation"]
                              for owner in sorted(results, key=lambda name: name.encode("utf-8"))}
            readback = handle.verify_roster_recovery_state(complete=True)
            steps["all_readback"] = {"owners": len(readback["owners"]),
                                     "readback_sha256": readback["readback_sha256"],
                                     "complete": readback["complete"]}
        except BaseException:
            handle.close()
            raise
        report = {"steps": steps, "readback": readback,
                  "original_path": str(original_path), "replacement_path": str(replacement_path),
                  "fence_sha256": hashlib.sha256(_quiet_canonical(fence)).hexdigest(),
                  "writes": "continuations-only", "exit": "not_taken"}
        return handle, report

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            conn, self._conn = self._conn, None
            if conn is None:
                return
            try:
                conn.close()
            except Exception:
                pass


class AsyncSessionDB:
    """Async door onto SessionDB: every call runs via asyncio.to_thread so a blocking SQLite call
    never freezes the event loop (no method returns a live cursor)."""

    def __init__(self, db: SessionDB) -> None:
        self._db = db

    def __getattr__(self, name: str):
        attr = getattr(self._db, name)
        if not callable(attr):
            return attr
        async def _offloaded(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)
        return _offloaded
