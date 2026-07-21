from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_worker_runtime as runtime


@pytest.mark.parametrize(
    "status",
    ["returned_for_rework", "reclaimed", "blocked", "completed"],
)
def test_previous_worker_profile_uses_latest_real_run_status(status: str) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE task_runs (task_id TEXT, profile TEXT, status TEXT, started_at INTEGER)"
    )
    conn.execute(
        "INSERT INTO task_runs VALUES (?, ?, ?, ?)",
        ("t1", "implementer", status, 10),
    )

    assert runtime._previous_worker_profile(conn, "t1") == "implementer"


def test_previous_worker_profile_prefers_latest_attempt() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE task_runs (task_id TEXT, profile TEXT, status TEXT, started_at INTEGER)"
    )
    conn.executemany(
        "INSERT INTO task_runs VALUES (?, ?, ?, ?)",
        [
            ("t1", "implementer", "returned_for_rework", 10),
            ("t1", "planner", "reclaimed", 20),
        ],
    )

    assert runtime._previous_worker_profile(conn, "t1") == "planner"


def test_ready_since_uses_rework_hold_release_time() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE task_events (task_id TEXT, kind TEXT, created_at INTEGER)"
    )
    conn.executemany(
        "INSERT INTO task_events VALUES (?, ?, ?)",
        [
            ("t1", "created", 10),
            ("t1", "returned_for_rework", 20),
            ("t1", "rework_hold_released", 30),
        ],
    )

    assert runtime.ready_since(conn, "t1", 1) == 30
