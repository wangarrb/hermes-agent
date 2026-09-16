from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from plugins.kanban.codex_listener.codex_kanban_interactive import (
    CodexInteractiveListener,
    LISTENER_CAPABILITIES,
)
from plugins.kanban.session_scope import codex_rollout_cursor, codex_submit_ack


def _codex_db(
    path: Path,
    rows: list[tuple[object, ...]],
    *,
    source_columns: tuple[str, ...] = (),
) -> None:
    source_definitions = {
        "thread_source": "TEXT",
        "source": "TEXT",
    }
    extra_definitions = "".join(
        f",\n                {column} {source_definitions[column]}"
        for column in source_columns
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"""
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                updated_at_ms INTEGER,
                archived INTEGER NOT NULL DEFAULT 0{extra_definitions}
            )
            """
        )
        columns = ("id", "cwd", "updated_at", "updated_at_ms", *source_columns)
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(
            f"INSERT INTO threads({', '.join(columns)}) VALUES ({placeholders})",
            [
                (thread_id, cwd, updated, int(updated) * 1000, *metadata)
                for thread_id, cwd, updated, *metadata in rows
            ],
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


def test_codex_listener_skips_newer_explicit_subagent_for_workspace(
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
            ("seq-root", str(seqscale), 10, "root", "cli"),
            ("seq-subagent", str(seqscale), 20, " SUBAGENT ", "cli"),
            ("ego-newer", str(egomotion), 30, "root", "cli"),
        ],
        source_columns=("thread_source", "source"),
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    listener = CodexInteractiveListener()

    assert listener.has_saved_sessions(seqscale)
    cmd = listener.build_tui_cmd(seqscale, continue_session=True)

    assert cmd[:3] == ["codex", "resume", "seq-root"]


@pytest.mark.parametrize(
    "subagent_source",
    [
        "subagent",
        '{"subagent":{"parent_thread_id":"seq-root"}}',
    ],
)
def test_codex_listener_skips_source_only_subagent(
    tmp_path: Path, monkeypatch, subagent_source: str,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    seqscale = tmp_path / "SeqScale"
    seqscale.mkdir()
    _codex_db(
        codex_home / "state_5.sqlite",
        [
            ("seq-root", str(seqscale), 10, "cli"),
            ("seq-subagent", str(seqscale), 20, subagent_source),
        ],
        source_columns=("source",),
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    listener = CodexInteractiveListener()

    assert listener.has_saved_sessions(seqscale)
    cmd = listener.build_tui_cmd(seqscale, continue_session=True)

    assert cmd[:3] == ["codex", "resume", "seq-root"]


@pytest.mark.parametrize("thread_source", [None, ""])
def test_codex_listener_uses_json_source_for_transitional_subagent(
    tmp_path: Path, monkeypatch, thread_source: str | None,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    seqscale = tmp_path / "SeqScale"
    seqscale.mkdir()
    _codex_db(
        codex_home / "state_5.sqlite",
        [
            ("seq-root", str(seqscale), 10, "root", "cli"),
            (
                "seq-subagent",
                str(seqscale),
                20,
                thread_source,
                '{"subagent":{"parent_thread_id":"seq-root"}}',
            ),
        ],
        source_columns=("thread_source", "source"),
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    listener = CodexInteractiveListener()

    assert listener.has_saved_sessions(seqscale)
    cmd = listener.build_tui_cmd(seqscale, continue_session=True)

    assert cmd[:3] == ["codex", "resume", "seq-root"]


def test_codex_listener_prefers_nonempty_root_thread_source_over_json_source(
    tmp_path: Path, monkeypatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    seqscale = tmp_path / "SeqScale"
    seqscale.mkdir()
    _codex_db(
        codex_home / "state_5.sqlite",
        [
            ("seq-older-root", str(seqscale), 10, "root", "cli"),
            (
                "seq-user-root",
                str(seqscale),
                20,
                "user",
                '{"subagent":{"parent_thread_id":"seq-older-root"}}',
            ),
        ],
        source_columns=("thread_source", "source"),
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    listener = CodexInteractiveListener()

    assert listener.has_saved_sessions(seqscale)
    cmd = listener.build_tui_cmd(seqscale, continue_session=True)

    assert cmd[:3] == ["codex", "resume", "seq-user-root"]


@pytest.mark.parametrize("eligible_source", ["cli", "{not-json"])
def test_codex_listener_keeps_plain_and_malformed_root_sources_eligible(
    tmp_path: Path, monkeypatch, eligible_source: str,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    seqscale = tmp_path / "SeqScale"
    seqscale.mkdir()
    _codex_db(
        codex_home / "state_5.sqlite",
        [
            ("seq-older-root", str(seqscale), 10, "root"),
            ("seq-eligible", str(seqscale), 20, eligible_source),
            (
                "seq-subagent",
                str(seqscale),
                30,
                '{"subagent":{"parent_thread_id":"seq-older-root"}}',
            ),
        ],
        source_columns=("source",),
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    listener = CodexInteractiveListener()

    assert listener.has_saved_sessions(seqscale)
    cmd = listener.build_tui_cmd(seqscale, continue_session=True)

    assert cmd[:3] == ["codex", "resume", "seq-eligible"]


def test_codex_listener_keeps_deeply_nested_source_eligible(
    tmp_path: Path, monkeypatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    seqscale = tmp_path / "SeqScale"
    seqscale.mkdir()
    deeply_nested_source = "[" * 100_000 + "0" + "]" * 100_000
    _codex_db(
        codex_home / "state_5.sqlite",
        [
            ("seq-older-root", str(seqscale), 10, "cli"),
            ("seq-deep-root", str(seqscale), 20, deeply_nested_source),
            (
                "seq-subagent",
                str(seqscale),
                30,
                '{"subagent":{"parent_thread_id":"seq-older-root"}}',
            ),
        ],
        source_columns=("source",),
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    listener = CodexInteractiveListener()

    assert listener.has_saved_sessions(seqscale)
    cmd = listener.build_tui_cmd(seqscale, continue_session=True)

    assert cmd[:3] == ["codex", "resume", "seq-deep-root"]


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


def test_codex_submit_ack_reads_exact_user_message_after_saved_cursor(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    workspace = tmp_path / "SeqScale"
    workspace.mkdir()
    rollout = codex_home / "sessions" / "rollout.jsonl"
    rollout.parent.mkdir()
    marker = "execute handoff [task t_1] [by watcher]"
    rollout.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": marker,
                    "turn_id": "turn-old",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with sqlite3.connect(codex_home / "state_5.sqlite") as conn:
        conn.execute(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                rollout_path TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "INSERT INTO threads(id, cwd, rollout_path, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("thread-1", str(workspace), str(rollout), 1),
        )

    before_cursor = codex_rollout_cursor(codex_home, workspace)
    assert before_cursor is not None
    with rollout.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": f"prefix {marker}",
                        "turn_id": "turn-nearby",
                    },
                }
            )
            + "\n"
        )
        stream.write(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "execute   handoff\n[task t_1] [by watcher]",
                        "turn_id": "turn-new",
                        "message_id": "message-new",
                    },
                }
            )
            + "\n"
        )

    ack = codex_submit_ack(codex_home, workspace, before_cursor, marker)

    assert ack.state == "accepted"
    assert ack.thread_id == "thread-1"
    assert ack.turn_id == "turn-new"
    assert ack.message_id == "message-new"


def test_codex_submit_ack_does_not_scan_before_cursor(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    workspace = tmp_path / "SeqScale"
    workspace.mkdir()
    rollout = codex_home / "rollout.jsonl"
    marker = "handoff-marker"
    rollout.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": marker,
                    "turn_id": "turn-old",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with sqlite3.connect(codex_home / "state_5.sqlite") as conn:
        conn.execute(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                rollout_path TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "INSERT INTO threads(id, cwd, rollout_path, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("thread-1", str(workspace), str(rollout), 1),
        )

    before_cursor = codex_rollout_cursor(codex_home, workspace)
    assert before_cursor is not None

    assert codex_submit_ack(
        codex_home, workspace, before_cursor, marker
    ).state == "unknown"


def test_codex_listener_capabilities_advertise_continuity_v3() -> None:
    assert LISTENER_CAPABILITIES == {
        "version": 3,
        "application_submit_ack": True,
        "transport_unknown_non_destructive": True,
        "stable_handoff_id": True,
        "task_title_lifecycle": True,
    }
