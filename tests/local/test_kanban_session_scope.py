from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "local/bin/kanban-session-scope.py"


def _hermes_db(path: Path, rows: list[tuple[str, str, float]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                cwd TEXT,
                started_at REAL NOT NULL,
                last_activity_at REAL,
                archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.executemany(
            "INSERT INTO sessions(id, cwd, started_at, last_activity_at) "
            "VALUES (?, ?, ?, ?)",
            [(sid, cwd, updated, updated) for sid, cwd, updated in rows],
        )


def test_hermes_selector_returns_latest_session_for_exact_workspace(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    seqscale = tmp_path / "SeqScale"
    egomotion = tmp_path / "Egomotion4D"
    seqscale.mkdir()
    egomotion.mkdir()
    _hermes_db(
        db,
        [
            ("ego-newer", str(egomotion), 30.0),
            ("seq-older", str(seqscale), 10.0),
            ("seq-latest", str(seqscale / "."), 20.0),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "hermes",
            "--db", str(db),
            "--workspace", str(seqscale),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "seq-latest"


def test_hermes_selector_returns_no_match_for_another_project(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    seqscale = tmp_path / "SeqScale"
    egomotion = tmp_path / "Egomotion4D"
    seqscale.mkdir()
    egomotion.mkdir()
    _hermes_db(db, [("ego-only", str(egomotion), 30.0)])

    result = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "hermes",
            "--db", str(db),
            "--workspace", str(seqscale),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
