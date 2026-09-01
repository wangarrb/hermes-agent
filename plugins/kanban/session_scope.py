"""Resolve saved agent sessions within one canonical workspace."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def canonical_workspace(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


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
                rows = conn.execute(
                    f"""
                    SELECT id, cwd, {updated_expr}
                    FROM threads
                    WHERE archived = 0 AND cwd IS NOT NULL AND cwd != ''
                    """
                ).fetchall()
        except (OSError, sqlite3.Error):
            continue
        for thread_id, cwd, updated in rows:
            if canonical_workspace(cwd) != expected:
                continue
            candidate = (int(updated or 0), str(thread_id))
            if best is None or candidate > best:
                best = candidate
    return best[1] if best else None
