"""Resolve saved agent sessions within one canonical workspace."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class CodexRolloutCursor:
    thread_id: str
    rollout_path: Path
    byte_offset: int


@dataclass(frozen=True)
class SubmitAck:
    state: Literal["accepted", "unknown"]
    thread_id: str | None
    turn_id: str | None
    message_id: str | None


def canonical_workspace(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_codex_subagent(thread_source: object, source: object) -> bool:
    normalized_thread_source = (
        str(thread_source).strip().casefold() if thread_source is not None else ""
    )
    if normalized_thread_source == "subagent":
        return True
    if normalized_thread_source:
        return False

    source_text = str(source).strip() if source is not None else ""
    if source_text.casefold() == "subagent":
        return True
    try:
        parsed_source = json.loads(source_text)
    except (TypeError, ValueError, RecursionError):
        return False
    return isinstance(parsed_source, dict) and "subagent" in parsed_source


def latest_hermes_session(db_path: str | Path, workspace: str | Path) -> str | None:
    db = Path(db_path)
    if not db.is_file():
        return None
    expected = canonical_workspace(workspace)
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                """
                SELECT id, cwd, COALESCE(last_activity_at, started_at)
                FROM sessions
                WHERE archived = 0 AND cwd IS NOT NULL AND cwd != ''
                ORDER BY COALESCE(last_activity_at, started_at) DESC,
                         started_at DESC
                """
            ).fetchall()
    except (OSError, sqlite3.Error):
        return None
    for session_id, cwd, _updated in rows:
        if canonical_workspace(cwd) == expected:
            return str(session_id)
    return None


def latest_codex_thread(codex_home: str | Path, workspace: str | Path) -> str | None:
    home = Path(codex_home).expanduser()
    expected = canonical_workspace(workspace)
    best: tuple[int, str] | None = None
    for db in home.glob("state_*.sqlite"):
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(threads)").fetchall()
                }
                if not {"id", "cwd", "archived"}.issubset(columns):
                    continue
                updated_expr = (
                    "COALESCE(updated_at_ms, updated_at * 1000)"
                    if "updated_at_ms" in columns and "updated_at" in columns
                    else "updated_at * 1000"
                    if "updated_at" in columns
                    else "created_at * 1000"
                    if "created_at" in columns
                    else "0"
                )
                thread_source_expr = (
                    "thread_source" if "thread_source" in columns else "NULL"
                )
                source_expr = "source" if "source" in columns else "NULL"
                rows = conn.execute(
                    f"""
                    SELECT id, cwd, {updated_expr},
                           {thread_source_expr}, {source_expr}
                    FROM threads
                    WHERE archived = 0 AND cwd IS NOT NULL AND cwd != ''
                    """
                ).fetchall()
        except (OSError, sqlite3.Error):
            continue
        for thread_id, cwd, updated, thread_source, source in rows:
            if _is_codex_subagent(thread_source, source):
                continue
            if canonical_workspace(cwd) != expected:
                continue
            candidate = (int(updated or 0), str(thread_id))
            if best is None or candidate > best:
                best = candidate
    return best[1] if best else None


def codex_rollout_cursor(
    codex_home: str | Path,
    workspace: str | Path,
) -> CodexRolloutCursor | None:
    """Return the current workspace thread and its end-of-rollout cursor."""
    home = Path(codex_home).expanduser()
    expected = canonical_workspace(workspace)
    best: tuple[int, str, Path] | None = None
    for db in home.glob("state_*.sqlite"):
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(threads)").fetchall()
                }
                if not {
                    "id",
                    "cwd",
                    "archived",
                    "rollout_path",
                }.issubset(columns):
                    continue
                updated_expr = (
                    "COALESCE(updated_at_ms, updated_at * 1000)"
                    if "updated_at_ms" in columns and "updated_at" in columns
                    else "updated_at * 1000"
                    if "updated_at" in columns
                    else "created_at * 1000"
                    if "created_at" in columns
                    else "0"
                )
                thread_source_expr = (
                    "thread_source" if "thread_source" in columns else "NULL"
                )
                source_expr = "source" if "source" in columns else "NULL"
                rows = conn.execute(
                    f"""
                    SELECT id, cwd, rollout_path, {updated_expr},
                           {thread_source_expr}, {source_expr}
                    FROM threads
                    WHERE archived = 0 AND cwd IS NOT NULL AND cwd != ''
                          AND rollout_path IS NOT NULL AND rollout_path != ''
                    """
                ).fetchall()
        except (OSError, sqlite3.Error):
            continue
        for thread_id, cwd, rollout_path, updated, thread_source, source in rows:
            if _is_codex_subagent(thread_source, source):
                continue
            if canonical_workspace(cwd) != expected:
                continue
            path = Path(str(rollout_path)).expanduser()
            if not path.is_absolute():
                path = home / path
            candidate = (int(updated or 0), str(thread_id), path.resolve(strict=False))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        return None
    _updated, thread_id, rollout_path = best
    try:
        byte_offset = rollout_path.stat().st_size
    except OSError:
        return None
    return CodexRolloutCursor(
        thread_id=thread_id,
        rollout_path=rollout_path,
        byte_offset=byte_offset,
    )


def _normalized_message(value: object) -> str:
    return " ".join(str(value or "").split())


def _user_message(record: object) -> tuple[str, str | None, str | None] | None:
    if not isinstance(record, dict):
        return None
    record_type = record.get("type")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None

    if record_type == "event_msg" and payload.get("type") == "user_message":
        text = payload.get("message", payload.get("text", ""))
        return (
            str(text or ""),
            str(payload.get("turn_id") or "") or None,
            str(payload.get("message_id") or payload.get("id") or "") or None,
        )

    if (
        record_type == "response_item"
        and payload.get("type") == "message"
        and payload.get("role") == "user"
    ):
        content = payload.get("content")
        if not isinstance(content, list):
            return None
        text = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "input_text"
        )
        return (
            text,
            str(payload.get("turn_id") or "") or None,
            str(payload.get("message_id") or payload.get("id") or "") or None,
        )
    return None


def codex_submit_ack(
    codex_home: str | Path,
    workspace: str | Path,
    before_cursor: CodexRolloutCursor | None,
    marker: str,
) -> SubmitAck:
    """Find an exact new user message and turn after ``before_cursor``."""
    unknown = SubmitAck("unknown", None, None, None)
    if before_cursor is None or not _normalized_message(marker):
        return unknown
    current = codex_rollout_cursor(codex_home, workspace)
    if (
        current is None
        or current.thread_id != before_cursor.thread_id
        or current.rollout_path != before_cursor.rollout_path
        or current.byte_offset < before_cursor.byte_offset
    ):
        return unknown
    try:
        with before_cursor.rollout_path.open("rb") as stream:
            stream.seek(before_cursor.byte_offset)
            appended = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return unknown

    expected = _normalized_message(marker)
    pending_message_id: str | None = None
    for line in appended.splitlines():
        try:
            record = json.loads(line)
        except (TypeError, ValueError, RecursionError):
            continue
        user_message = _user_message(record)
        if user_message is not None:
            text, turn_id, message_id = user_message
            if _normalized_message(text) != expected:
                pending_message_id = None
                continue
            if turn_id:
                return SubmitAck(
                    "accepted", before_cursor.thread_id, turn_id, message_id
                )
            pending_message_id = message_id
            continue
        if pending_message_id is None or not isinstance(record, dict):
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        turn_id = str(payload.get("turn_id") or "") or None
        if record.get("type") == "turn_context" and turn_id:
            return SubmitAck(
                "accepted",
                before_cursor.thread_id,
                turn_id,
                pending_message_id,
            )
    return unknown
