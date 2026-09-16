"""Semantic acknowledgement contract for interactive Kanban delivery."""

import json
import sqlite3
import sys
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban import base_listener as bl
from plugins.kanban.codex_listener import codex_kanban_interactive as codex


def _canonical_json(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _write_current_contract(
    root: Path, task_id: str, *, generation: int = 1, version: int = 1,
) -> str:
    record = {
        "schema_version": "seqscale-module-contract-v1",
        "version": version,
        "contract": {"task_id": task_id, "generation": generation},
    }
    content = _canonical_json(record)
    digest = hashlib.sha256(content).hexdigest()
    relative = f"contracts/v{version:03d}.json"
    (root / "contracts").mkdir(parents=True, exist_ok=True)
    (root / relative).write_bytes(content)
    (root / "current-contract.json").write_bytes(
        _canonical_json(
            {
                "schema_version": "seqscale-current-contract-v1",
                "version": version,
                "sha256": digest,
                "path": relative,
            }
        )
    )
    return digest


def _write_workflow_state(
    common_dir: Path,
    task_id: str,
    *,
    contract_sha: str,
    version: int,
    phase: str = "resume_pending",
    write_record: bool = True,
) -> None:
    root = common_dir / "seqscale-workflow" / "tasks" / task_id
    root.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "seqscale-workflow-state-v1",
        "version": version,
        "task_id": task_id,
        "generation": 1,
        "workflow_phase": phase,
        "next_actor": "reviewer" if phase == "review_checkpoint" else "planner",
        "contract_sha256": contract_sha,
    }
    content = _canonical_json(state)
    digest = hashlib.sha256(content).hexdigest()
    relative = f"workflow-state/v{version:03d}.json"
    if write_record:
        (root / "workflow-state").mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(content)
    (root / "workflow-state.json").write_bytes(
        _canonical_json(
            {
                "schema_version": "seqscale-workflow-state-pointer-v1",
                "version": version,
                "sha256": digest,
                "path": relative,
            }
        )
    )


class _Listener(bl.BaseInteractiveListener):
    agent_name = "Test"
    agent_slug = "test"
    idle_markers = ("❯",)

    def build_tui_cmd(self, workspace, **kwargs):
        return []

    def has_saved_sessions(self, workspace):
        return True

    def inject_text(self, *args, **kwargs):
        return "prompt"

    def pane_label(self, task_id=None):
        return f"test-kanban [{task_id}]" if task_id else "test-kanban"

    def on_post_inject(self, args, *, zellij_session, zellij_pane_id,
                       log_path, injected_marker=None,
                       pre_write_composer=None):
        return "known_unsubmitted"


class _StrictListener(_Listener):
    semantic_delivery_required = True

    def on_post_inject(self, *args, **kwargs):
        return None


class _StrictStateListener(_Listener):
    semantic_delivery_required = True

    def __init__(self, state):
        super().__init__()
        self.state = state

    def on_post_inject(self, args, **kwargs):
        return self.state


def test_required_semantic_backend_never_infers_confirmation(tmp_path):
    listener = _StrictListener()
    listener.read_pane_screen = lambda **_: "❯"
    state = listener._post_injection_contract(
        Namespace(), zellij_session="s", zellij_pane_id="0",
        log_path=tmp_path / "watch.log", injected_marker="prompt",
        pre_write_composer=None, correlation="task:t_1",
    )
    assert state == "unknown"


def test_post_inject_receives_marker_and_pre_write_composer(tmp_path):
    seen = {}

    class Capture(_Listener):
        def on_post_inject(self, args, **kwargs):
            seen.update(kwargs)
            return "confirmed"

    Capture()._post_injection_contract(
        Namespace(), zellij_session="s", zellij_pane_id="0",
        log_path=tmp_path / "watch.log", injected_marker="m",
        pre_write_composer="before", correlation="control:3",
    )
    assert seen["injected_marker"] == "m"
    assert seen["pre_write_composer"] == "before"


def test_required_semantic_backend_honors_transport_accepted(tmp_path):
    listener = _StrictStateListener("transport_accepted")
    assert listener._post_injection_contract(
        Namespace(), zellij_session="s", zellij_pane_id="0",
        log_path=tmp_path / "watch.log", injected_marker="prompt",
        pre_write_composer="", correlation="task:t_1",
    ) == "transport_accepted"


class _TaskListener(_Listener):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def on_post_inject(self, args, **kwargs):
        return self.state


class _ConfirmedListener(_Listener):
    def on_post_inject(self, args, **kwargs):
        return "confirmed"


def _task_args(tmp_path):
    return Namespace(profile="reviewer", claim_assignees="reviewer",
                      assist_role=None, zellij_session="s", zellij_pane_id="0",
                      workspace=str(tmp_path), board="default", ttl=900)


def test_known_unsubmitted_task_delivery_requeues_before_worker_start(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("known_unsubmitted")
    args = _task_args(tmp_path)
    listener._init_from_args(args)
    monkeypatch.setattr(listener, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda *a, **k: True)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", assignee="reviewer", workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        claimed, run_id = listener.claim_and_inject_one(args, log_path=tmp_path / "watch.log", conn=conn)
        row = kb.get_task(conn, task_id)
    assert claimed is None and run_id is None
    assert row is not None and row.status != "running"
    log_text = (tmp_path / "watch.log").read_text(encoding="utf-8")
    assert "correlation_kind=task" in log_text
    assert f"correlation_id={task_id}" in log_text
    assert "task_id=" in log_text and "run_id=" in log_text
    assert "generation=" in log_text and "pane_id=0" in log_text
    assert "pane_prefix=test-kanban" in log_text


def test_unknown_task_delivery_retains_fenced_claim(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("unknown")
    args = _task_args(tmp_path)
    listener._init_from_args(args)
    monkeypatch.setattr(listener, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda *a, **k: True)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)
    reclaims = []
    monkeypatch.setattr(
        bl,
        "_reclaim_task_without_signaling_worker",
        lambda *args, **kwargs: reclaims.append((args, kwargs)) or True,
    )
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", assignee="reviewer", workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        claimed, run_id = listener.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )
        row = kb.get_task(conn, task_id)

    assert (claimed, run_id) == (task_id, row.current_run_id)
    assert row.status == "running" and row.claim_lock
    assert reclaims == []
    assert listener._pending_delivery is not None
    log_text = (tmp_path / "watch.log").read_text(encoding="utf-8")
    assert "event=delivery_pending" in log_text
    assert "state=pending" in log_text


def test_pending_delivery_observer_accepts_later_application_ack(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("unknown")
    args = _task_args(tmp_path)
    listener._init_from_args(args)
    monkeypatch.setattr(listener, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda *a, **k: True)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", assignee="reviewer", workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        listener.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )
        listener.state = "transport_accepted"
        state = listener._observe_pending_delivery(
            args, conn, task_id, tmp_path / "watch.log",
        )

    assert state == "transport_accepted"
    assert listener._pending_delivery is None


def test_application_ack_ledger_prevents_duplicate_user_message_on_run_retry(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("transport_accepted")
    args = _task_args(tmp_path)
    listener._init_from_args(args)
    monkeypatch.setattr(listener, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda *a, **k: True)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    injections: list[str] = []
    monkeypatch.setattr(
        bl,
        "zellij_inject",
        lambda **kwargs: injections.append(kwargs["text"]) or True,
    )
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", body="contract_sha256: contract-abc",
            assignee="reviewer", workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        claimed_1, run_1 = listener.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )
        handoff_1 = listener._active_handoff_id
        task_1 = kb.get_task(conn, task_id)
        assert task_1 is not None
        assert bl._reclaim_task_without_signaling_worker(
            conn,
            task_id,
            reason="simulate retry after ACK",
            expected_run_id=task_1.current_run_id,
            expected_generation=task_1.generation,
            expected_claim_lock=task_1.claim_lock,
        )
        listener._clear_active_claim_identity()
        claimed_2, run_2 = listener.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )
        handoff_2 = listener._active_handoff_id

    assert claimed_1 == claimed_2 == task_id
    assert run_1 != run_2
    assert handoff_1 == handoff_2 == bl.stable_handoff_id(
        "default", task_id, 1, "contract-abc", "task", 0,
    )
    assert len(injections) == 1


def test_retry_reuses_handoff_but_new_workflow_resume_reinjects(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("transport_accepted")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    common_dir = tmp_path / "common"
    args = _task_args(workspace)
    listener._init_from_args(args)
    monkeypatch.setattr(listener, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda *a, **k: True)
    monkeypatch.setattr(listener, "_git_common_dir", lambda _workspace: common_dir)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)
    injections: list[str] = []
    monkeypatch.setattr(
        bl,
        "zellij_inject",
        lambda **kwargs: injections.append(kwargs["text"]) or True,
    )
    body = (
        f"artifact_namespace: {artifacts}\n"
        "contract_ref: current-contract.json\n"
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", body=body, assignee="reviewer",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        contract_sha = _write_current_contract(artifacts, task_id)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))

        def claim() -> tuple[int, str]:
            claimed, run_id = listener.claim_and_inject_one(
                args, log_path=tmp_path / "watch.log", conn=conn,
            )
            assert claimed == task_id and run_id is not None
            handoff_id = listener._active_handoff_id
            assert handoff_id is not None
            return run_id, handoff_id

        def retry_ready(reason: str) -> None:
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert bl._reclaim_task_without_signaling_worker(
                conn,
                task_id,
                reason=reason,
                expected_run_id=task.current_run_id,
                expected_generation=task.generation,
                expected_claim_lock=task.claim_lock,
            )
            listener._clear_active_claim_identity()

        run_1, initial_handoff = claim()
        retry_ready("transport retry")
        run_2, retry_handoff = claim()
        retry_ready("legal resume")
        _write_workflow_state(
            common_dir,
            task_id,
            contract_sha=contract_sha,
            version=1,
        )
        run_3, resume_handoff = claim()
        retry_ready("resume transport retry")
        run_4, resume_retry_handoff = claim()

    assert len({run_1, run_2, run_3, run_4}) == 4
    assert retry_handoff == initial_handoff
    assert resume_handoff != initial_handoff
    assert resume_retry_handoff == resume_handoff
    assert len(injections) == 2
    resume_record = listener._read_handoff_record(resume_handoff)
    assert resume_record is not None
    assert resume_record["handoff_kind"] == "resume_pending"
    assert resume_record["sequence"] == 1
    assert resume_record["contract_sha"] == contract_sha


def test_current_contract_version_update_creates_new_logical_handoff(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("transport_accepted")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    common_dir = tmp_path / "common"
    args = _task_args(workspace)
    listener._init_from_args(args)
    monkeypatch.setattr(listener, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda *a, **k: True)
    monkeypatch.setattr(listener, "_git_common_dir", lambda _workspace: common_dir)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)
    injections: list[str] = []
    monkeypatch.setattr(
        bl,
        "zellij_inject",
        lambda **kwargs: injections.append(kwargs["text"]) or True,
    )
    body = (
        f"artifact_namespace: {artifacts}\n"
        "contract_ref: current-contract.json\n"
        "contract_sha256: stale-initial-body-hash\n"
    )
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", body=body, assignee="reviewer",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        sha_v1 = _write_current_contract(artifacts, task_id, version=1)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        claimed, _run_1 = listener.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )
        assert claimed == task_id
        handoff_v1 = listener._active_handoff_id
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert bl._reclaim_task_without_signaling_worker(
            conn,
            task_id,
            reason="contract update",
            expected_run_id=task.current_run_id,
            expected_generation=task.generation,
            expected_claim_lock=task.claim_lock,
        )
        listener._clear_active_claim_identity()
        sha_v2 = _write_current_contract(artifacts, task_id, version=2)
        claimed, _run_2 = listener.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )
        assert claimed == task_id
        handoff_v2 = listener._active_handoff_id

    assert sha_v2 != sha_v1
    assert handoff_v2 != handoff_v1
    assert len(injections) == 2
    record = listener._read_handoff_record(handoff_v2)
    assert record is not None and record["contract_sha"] == sha_v2


def test_handoff_identity_fails_closed_on_missing_contract_pointer(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("transport_accepted")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = tmp_path / "missing-artifacts"
    common_dir = tmp_path / "common"
    monkeypatch.setattr(listener, "_git_common_dir", lambda _workspace: common_dir)
    body = (
        f"artifact_namespace: {artifacts}\n"
        "contract_ref: current-contract.json\n"
    )
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", body=body, assignee="reviewer",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        with pytest.raises(bl.HandoffIdentityError, match="current contract"):
            listener._resolve_handoff_identity(task)


@pytest.mark.parametrize("write_record", [False, True])
def test_handoff_identity_fails_closed_on_invalid_workflow_pointer(
    kanban_home, tmp_path, monkeypatch, write_record,
):
    listener = _TaskListener("transport_accepted")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    common_dir = tmp_path / "common"
    monkeypatch.setattr(listener, "_git_common_dir", lambda _workspace: common_dir)
    body = (
        f"artifact_namespace: {artifacts}\n"
        "contract_ref: current-contract.json\n"
    )
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", body=body, assignee="reviewer",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        contract_sha = _write_current_contract(artifacts, task_id)
        _write_workflow_state(
            common_dir,
            task_id,
            contract_sha=(contract_sha if not write_record else "0" * 64),
            version=1,
            write_record=write_record,
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        expected = "workflow-state record" if not write_record else "contract SHA"
        with pytest.raises(bl.HandoffIdentityError, match=expected):
            listener._resolve_handoff_identity(task)


def test_codex_retry_recovers_ack_after_crash_before_ledger_accept(
    kanban_home, tmp_path, monkeypatch,
):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rollout = codex_home / "rollout.jsonl"
    rollout.write_text("", encoding="utf-8")
    with sqlite3.connect(codex_home / "state_5.sqlite") as state:
        state.execute(
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
        state.execute(
            "INSERT INTO threads(id, cwd, rollout_path, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("thread-1", str(workspace), str(rollout), 1),
        )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(codex.time, "sleep", lambda _: None)
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: "›\n")
    runtime_base = sys.modules.get("base_listener", bl)
    monkeypatch.setattr(runtime_base, "zellij_dump_screen", lambda **_: "›\n")
    monkeypatch.setattr(runtime_base, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(runtime_base, "zellij_rename_pane", lambda **_: True)
    injections: list[str] = []
    monkeypatch.setattr(
        runtime_base,
        "zellij_inject",
        lambda **kwargs: injections.append(kwargs["text"]) or True,
    )
    args = _task_args(workspace)
    first = codex.CodexInteractiveListener()
    first._init_from_args(args)
    monkeypatch.setattr(first, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(first, "on_claim_post_confirm", lambda *a, **k: True)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", body="contract_sha256: contract-abc",
            assignee="reviewer", workspace_kind="dir",
            workspace_path=str(workspace),
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        claimed_1, run_1 = first.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )
        assert claimed_1 == task_id
        assert first._pending_delivery is not None
        assert len(injections) == 1
        with rollout.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "user_message",
                            "message": injections[0],
                            "turn_id": "turn-after-crash",
                        },
                    }
                )
                + "\n"
            )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert bl._reclaim_task_without_signaling_worker(
            conn,
            task_id,
            reason="simulated crash after application ACK",
            expected_run_id=task.current_run_id,
            expected_generation=task.generation,
            expected_claim_lock=task.claim_lock,
        )
        first._clear_active_claim_identity()

        recovered = codex.CodexInteractiveListener()
        recovered._init_from_args(args)
        monkeypatch.setattr(recovered, "on_claim_pre_check", lambda *a, **k: True)
        monkeypatch.setattr(recovered, "on_claim_post_confirm", lambda *a, **k: True)
        claimed_2, run_2 = recovered.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )

    assert claimed_2 == task_id
    assert run_2 != run_1
    assert len(injections) == 1
    stable_prompt = (
        workspace
        / ".codex-kanban"
        / "default"
        / "reviewer"
        / f"handoff-{recovered._active_handoff_id}.md"
    )
    prompt_text = stable_prompt.read_text(encoding="utf-8")
    assert f"run_id={run_2}" in prompt_text
    assert f"--run-id {run_2}" in prompt_text


def test_prepared_handoff_record_survives_crash_before_injection(
    kanban_home, tmp_path,
):
    listener = _TaskListener("unknown")
    listener._board = "default"
    handoff_id = bl.stable_handoff_id(
        "default", "t_crash", 1, "contract-abc", "task", 0,
    )

    listener._write_handoff_record(
        handoff_id,
        {
            "handoff_id": handoff_id,
            "task_id": "t_crash",
            "generation": 1,
            "contract_sha": "contract-abc",
            "handoff_kind": "task",
            "sequence": 0,
            "state": "prepared",
            "run_ids": [1],
        },
    )
    recovered = listener._read_handoff_record(handoff_id)

    assert recovered is not None
    assert recovered["handoff_id"] == handoff_id
    assert recovered["state"] == "prepared"
    assert recovered["run_ids"] == [1]


def test_task_title_is_set_before_injection_and_cleared_on_reclaim(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("known_unsubmitted")
    args = _task_args(tmp_path)
    listener._init_from_args(args)
    monkeypatch.setattr(listener, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda *a, **k: True)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        bl,
        "zellij_rename_pane",
        lambda **kwargs: events.append(("title", kwargs["name"])) or True,
    )
    monkeypatch.setattr(
        bl,
        "zellij_inject",
        lambda **kwargs: events.append(("inject", kwargs["text"])) or True,
    )
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="task", assignee="reviewer", workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        listener.claim_and_inject_one(
            args, log_path=tmp_path / "watch.log", conn=conn,
        )

    assert events[0] == ("title", f"test-kanban [{task_id}]")
    assert next(index for index, event in enumerate(events) if event[0] == "inject") > 0
    assert events[-1] == ("title", "test-kanban")


def test_review_checkpoint_title_uses_review_marker(tmp_path, monkeypatch):
    listener = _TaskListener("unknown")
    listener._log_path = tmp_path / "watch.log"
    titles: list[str] = []
    monkeypatch.setattr(
        bl,
        "zellij_rename_pane",
        lambda **kwargs: titles.append(kwargs["name"]) or True,
    )

    listener._set_active_pane_title(
        session="s", pane_id="0", task_id="t_review", review=True,
    )

    assert titles == ["test-kanban [REVIEW t_review]"]


def test_idle_without_real_progress_is_false_running_after_300_seconds(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("unknown")
    clock = [1_000.0]
    monkeypatch.setattr(bl.time, "time", lambda: clock[0])
    monkeypatch.setattr(bl, "_has_bound_child_process", lambda _pid: False)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="task", assignee="reviewer")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        task = kb.claim_task(conn, task_id, claimer="listener")
        assert task is not None
        assert not listener._is_false_running(
            conn, task, pane_idle=True, workspace=tmp_path,
        )
        clock[0] += 299
        assert kb.heartbeat_worker(
            conn,
            task_id,
            note="listener heartbeat only",
            expected_run_id=task.current_run_id,
            expected_generation=task.generation,
            expected_claim_lock=task.claim_lock,
        )
        assert not listener._is_false_running(
            conn, task, pane_idle=True, workspace=tmp_path,
        )
        clock[0] += 1
        assert listener._is_false_running(
            conn, task, pane_idle=True, workspace=tmp_path,
        )


def test_busy_child_or_progress_ledger_prevents_false_running(
    kanban_home, tmp_path, monkeypatch,
):
    listener = _TaskListener("unknown")
    clock = [1_000.0]
    monkeypatch.setattr(bl.time, "time", lambda: clock[0])
    child_running = [False]
    monkeypatch.setattr(
        bl, "_has_bound_child_process", lambda _pid: child_running[0],
    )
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="task", assignee="reviewer")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        task = kb.claim_task(conn, task_id, claimer="listener")
        assert task is not None
        assert not listener._is_false_running(
            conn, task, pane_idle=True, workspace=tmp_path,
        )
        clock[0] += 300
        assert not listener._is_false_running(
            conn, task, pane_idle=False, workspace=tmp_path,
        )
        clock[0] += 300
        child_running[0] = True
        assert not listener._is_false_running(
            conn, task, pane_idle=True, workspace=tmp_path,
        )
        child_running[0] = False
        listener.record_task_progress(task_id, {"completed": 1})
        clock[0] += 300
        assert not listener._is_false_running(
            conn, task, pane_idle=True, workspace=tmp_path,
        )


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)
    return home


def _args(tmp_path):
    return Namespace(profile="reviewer", claim_assignees="reviewer",
                      assist_role=None, zellij_session="s",
                      zellij_pane_id="0", workspace=str(tmp_path), board="default")


def test_nonconfirmed_control_ack_is_not_marked_delivered(kanban_home, tmp_path, monkeypatch):
    listener = _Listener()
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review", assignee="reviewer")
        kb.claim_task(conn, task_id, claimer="review-pane")
        returned = kb.return_task_for_rework(conn, task_id, actor="reviewer", reason="redo")
        assert listener.pump_control_messages(_args(tmp_path), conn, tmp_path / "watch.log")
        row = kb.list_control_messages(conn)[0]
    assert row.status == "pending"


def test_control_release_cas_race_is_logged(kanban_home, tmp_path, monkeypatch):
    listener = _Listener()
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    monkeypatch.setattr(kb, "release_control_lease", lambda *a, **k: False)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review", assignee="reviewer")
        kb.claim_task(conn, task_id, claimer="review-pane")
        kb.return_task_for_rework(conn, task_id, actor="reviewer", reason="redo")
        listener.pump_control_messages(_args(tmp_path), conn, tmp_path / "watch.log")
    text = (tmp_path / "watch.log").read_text(encoding="utf-8")
    assert "event=delivery_reclaim_race" in text
    assert "correlation_kind=control" in text


def test_task_reclaim_cas_race_is_logged(kanban_home, tmp_path, monkeypatch):
    listener = _TaskListener("known_unsubmitted")
    args = _task_args(tmp_path)
    listener._init_from_args(args)
    monkeypatch.setattr(listener, "on_claim_pre_check", lambda *a, **k: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda *a, **k: True)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)
    monkeypatch.setattr(bl, "_reclaim_task_without_signaling_worker", lambda *a, **k: False)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="task", assignee="reviewer", workspace_kind="dir", workspace_path=str(tmp_path))
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        listener.claim_and_inject_one(args, log_path=tmp_path / "watch.log", conn=conn)
    text = (tmp_path / "watch.log").read_text(encoding="utf-8")
    assert "event=delivery_reclaim_race" in text and "correlation_kind=task" in text


def test_result_release_cas_race_is_logged(kanban_home, tmp_path, monkeypatch):
    listener = _Listener()
    monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    monkeypatch.setattr(kb, "release_result_notification_lease", lambda *a, **k: False)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="result", assignee="reviewer", result_subscriber="reviewer")
        kb.complete_task(conn, task_id, summary="done")
        listener.pump_result_notifications(_args(tmp_path), conn, tmp_path / "watch.log")
    text = (tmp_path / "watch.log").read_text(encoding="utf-8")
    # Since 9909e58eac an uncertain result release logs a single requeued
    # event carrying reclaimed=False (the separate reclaim_race event was
    # consolidated away for the result path).
    assert "event=delivery_requeued" in text and "correlation_kind=result" in text
    assert "reclaimed=False" in text


@pytest.mark.parametrize("kind", ["control", "result"])
def test_confirmed_hook_runs_before_delivery_mark(kanban_home, tmp_path, monkeypatch, kind):
    seen = []
    listener = _ConfirmedListener()
    if kind == "control":
        monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
        monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
        monkeypatch.setattr(kb, "mark_control_delivered", lambda *a, **k: seen.append("mark") or True)
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="review", assignee="reviewer")
            kb.claim_task(conn, task_id, claimer="review-pane")
            kb.return_task_for_rework(conn, task_id, actor="reviewer", reason="redo")
            listener.pump_control_messages(_args(tmp_path), conn, tmp_path / "watch.log")
    else:
        monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: True)
        monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
        monkeypatch.setattr(kb, "mark_result_notifications_delivered", lambda *a, **k: seen.append("mark") or True)
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="result", assignee="reviewer", result_subscriber="reviewer")
            kb.complete_task(conn, task_id, summary="done")
            listener.pump_result_notifications(_args(tmp_path), conn, tmp_path / "watch.log")
    assert seen == ["mark"]


def test_nonconfirmed_result_ack_releases_lease(kanban_home, tmp_path, monkeypatch):
    listener = _Listener()
    monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="result", assignee="reviewer", result_subscriber="reviewer")
        kb.complete_task(conn, task_id, summary="done")
        assert listener.pump_result_notifications(_args(tmp_path), conn, tmp_path / "watch.log")
        row = conn.execute("SELECT status, lease_owner FROM kanban_result_queue").fetchone()
    assert tuple(row) == ("pending", None)
