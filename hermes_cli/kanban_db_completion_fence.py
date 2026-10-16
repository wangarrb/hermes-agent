"""Generation snapshot for canonical completion; not a claim or review policy."""
from __future__ import annotations
import sqlite3


def capture_generation(conn: sqlite3.Connection, task_id: str, expected_generation: int | None) -> int | None:
    row = conn.execute("SELECT generation FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None:
        return None
    generation = int(row["generation"])
    if expected_generation is not None and generation != int(expected_generation):
        return None
    return generation


def generation_matches(conn: sqlite3.Connection, task_id: str, generation: int) -> bool:
    return conn.execute("SELECT 1 FROM tasks WHERE id=? AND generation=?", (task_id, generation)).fetchone() is not None
