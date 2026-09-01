from __future__ import annotations

import sqlite3
from pathlib import Path

from plugins.kanban.codex_listener.codex_kanban_interactive import (
    CodexInteractiveListener,
)


def _codex_db(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                updated_at_ms INTEGER,
                archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.executemany(
            "INSERT INTO threads(id, cwd, updated_at, updated_at_ms) "
            "VALUES (?, ?, ?, ?)",
            [(tid, cwd, updated, updated * 1000) for tid, cwd, updated in rows],
        )


def test_codex_listener_resumes_explicit_latest_thread_for_workspace(
    tmp_path: Path, monkeypatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    seqscale = tmp_path / "SeqScale"
    egomotion = tmp_path / "Egomotion4D"
    seqscale.mkdir()
    egomotion.mkdir()
    _codex_db(
        codex_home / "state_5.sqlite",
        [
            ("ego-newer", str(egomotion), 30),
            ("seq-older", str(seqscale), 10),
            ("seq-latest", str(seqscale / "."), 20),
        ],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    listener = CodexInteractiveListener()

    assert listener.has_saved_sessions(seqscale)
    cmd = listener.build_tui_cmd(seqscale, continue_session=True)

    assert cmd[:3] == ["codex", "resume", "seq-latest"]
    assert "--last" not in cmd
    assert cmd[-1] == str(seqscale)


def test_codex_listener_does_not_resume_unrelated_rollout(
    tmp_path: Path, monkeypatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    rollout_dir = codex_home / "sessions/2026/09/01"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "rollout-ego.jsonl").write_text(
        '{"type":"session_meta","payload":{"cwd":"/tmp/Egomotion4D"}}\n',
        encoding="utf-8",
    )
    seqscale = tmp_path / "SeqScale"
    seqscale.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    listener = CodexInteractiveListener()

    assert not listener.has_saved_sessions(seqscale)
    assert listener.build_tui_cmd(seqscale, continue_session=False)[0] == "codex"
