"""Tests for Codex-backed visible Kanban listener retry policy."""

from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


REPO = Path(__file__).resolve().parents[2]
CODEX_LISTENER_PATH = REPO / "plugins" / "kanban" / "codex_listener" / "codex_kanban_listener.py"
CODEX_INTERACTIVE_PATH = REPO / "plugins" / "kanban" / "codex_listener" / "codex_kanban_interactive.py"
DEEPSEEK_LISTENER_PATH = REPO / "plugins" / "kanban" / "deepseek_listener" / "deepseek_kanban_listener.py"
DEEPSEEK_INTERACTIVE_PATH = REPO / "plugins" / "kanban" / "deepseek_listener" / "deepseek_kanban_interactive.py"


def _load_codex_listener():
    spec = importlib.util.spec_from_file_location("codex_kanban_listener_test", CODEX_LISTENER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_codex_interactive():
    spec = importlib.util.spec_from_file_location("codex_kanban_interactive_test", CODEX_INTERACTIVE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_deepseek_interactive():
    spec = importlib.util.spec_from_file_location("deepseek_kanban_interactive_test", DEEPSEEK_INTERACTIVE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_deepseek_listener():
    spec = importlib.util.spec_from_file_location("deepseek_kanban_listener_test", DEEPSEEK_LISTENER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_codex_provider_failure_requeues_after_shared_cooldown(kanban_home, monkeypatch):
    mod = _load_codex_listener()
    monkeypatch.setattr(mod.listener_policy, "RETRY_COOLDOWN_SECONDS", 0)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="planner provider failure", assignee="planner")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=mod.claim_lock())
        assert claimed is not None
        run_id = claimed.current_run_id

    outcome = mod._codex_provider_failure_retry(
        board="",
        task_id=task_id,
        expected_run_id=run_id,
        reason="HTTP 429 Too Many Requests / quota exceeded",
        ttl_s=3600,
    )

    assert outcome == "ready"
    with kb.connect() as conn:
        row = conn.execute(
            "SELECT status, claim_lock, consecutive_failures, last_failure_error "
            "FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["consecutive_failures"] == 1
        assert "429" in row["last_failure_error"]
        ev = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert ev["kind"] == "codex_provider_failure_retry"


def test_codex_provider_failure_holds_claim_and_heartbeats_during_cooldown(kanban_home, monkeypatch):
    mod = _load_codex_listener()
    monkeypatch.setattr(mod.listener_policy, "RETRY_COOLDOWN_SECONDS", 10)

    clock = {"t": 1000.0}

    def now_fn():
        return clock["t"]

    def sleep_fn(seconds):
        with kb.connect() as conn:
            row = conn.execute(
                "SELECT status, claim_lock FROM tasks WHERE id=?",
                (task_id,),
            ).fetchone()
            assert row["status"] == "running"
            assert row["claim_lock"] == mod.claim_lock()
        clock["t"] += float(seconds)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="planner 503", assignee="planner")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=5, claimer=mod.claim_lock())
        assert claimed is not None
        run_id = claimed.current_run_id

    outcome = mod._codex_provider_failure_retry(
        board="",
        task_id=task_id,
        expected_run_id=run_id,
        reason="HTTP 503 Service Unavailable from provider",
        ttl_s=5,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )

    assert outcome == "ready"
    with kb.connect() as conn:
        heartbeats = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=? AND kind='heartbeat'",
            (task_id,),
        ).fetchone()["n"]
        assert heartbeats >= 1


def test_codex_non_provider_failure_blocks_immediately(kanban_home):
    mod = _load_codex_listener()
    assert mod._is_provider_failure_result(1, {"details": "unit tests failed"}) is False
    assert mod._is_provider_failure_result(1, {"details": "HTTP 503 Service Unavailable"}) is True
    assert mod._is_provider_failure_result(0, {"details": "HTTP 503 Service Unavailable"}) is False


def test_codex_noninteractive_assist_claims_delayed_implementer_task(kanban_home, tmp_path, monkeypatch):
    mod = _load_codex_listener()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    calls = []

    def fake_run_codex_for_task(**kwargs):
        calls.append(kwargs)
        return (
            0,
            {
                "status": "done",
                "summary": "implemented",
                "details": "implemented details",
                "metadata": {},
            },
            tmp_path / "codex.log",
        )

    monkeypatch.setattr(mod, "run_codex_for_task", fake_run_codex_for_task)
    with kb.connect(board="codex-listen-assist-test") as conn:
        task_id = kb.create_task(conn, title="implementer work", assignee="implementer")

    args = argparse.Namespace(
        board="codex-listen-assist-test",
        profile="planner",
        claim_assignees=["planner", "implementer"],
        assist_claim_delay_s=60,
        assist_claim_delay_for=None,
        assist_claim_profile_delay=None,
        ttl=3600,
        sandbox="danger-full-access",
        model=None,
        codex_bin="codex",
        codex_arg=[],
    )

    assert mod.handle_one_task(args) is False
    with kb.connect(board="codex-listen-assist-test") as conn:
        old_ts = int(mod.time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))
        kb.set_workspace_path(conn, task_id, workspace)

    assert mod.handle_one_task(args) is True
    assert calls and calls[0]["task"].id == task_id
    with kb.connect(board="codex-listen-assist-test") as conn:
        row = conn.execute(
            "SELECT status, assignee FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    assert row["status"] == "done"
    assert row["assignee"] == "implementer"


def test_deepseek_noninteractive_completes_claimed_task_with_fake_exec(kanban_home, tmp_path, monkeypatch):
    mod = _load_deepseek_listener()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    seen = {}

    def fake_run_deepseek_for_task(**kwargs):
        seen.update(kwargs)
        return (
            0,
            {
                "status": "done",
                "summary": "deepseek done",
                "details": "deepseek details",
                "metadata": {"checked": True},
            },
            tmp_path / "deepseek.log",
        )

    monkeypatch.setattr(mod, "run_deepseek_for_task", fake_run_deepseek_for_task)
    with kb.connect(board="deepseek-listen-test") as conn:
        task_id = kb.create_task(conn, title="deepseek implementer", assignee="implementer")
        kb.set_workspace_path(conn, task_id, workspace)

    args = argparse.Namespace(
        board="deepseek-listen-test",
        profile="implementer",
        claim_assignees=["implementer"],
        assist_claim_delay_s=0,
        assist_claim_delay_for=None,
        assist_claim_profile_delay=None,
        ttl=3600,
        model="deepseek-v4-flash",
        deepseek_bin="deepseek-tui",
        deepseek_arg=[],
        auto=True,
    )

    assert mod.handle_one_task(args) is True
    assert seen["task"].id == task_id
    assert seen["workspace"] == workspace
    with kb.connect(board="deepseek-listen-test") as conn:
        row = conn.execute(
            "SELECT status, assignee FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    assert row["status"] == "done"
    assert row["assignee"] == "implementer"


def test_codex_interactive_claims_writes_prompt_and_injects(kanban_home, tmp_path, monkeypatch):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    injected = {}
    renamed = []

    def fake_inject(*, session, pane_id, text, log_path):
        injected.update({"session": session, "pane_id": pane_id, "text": text, "log_path": log_path})
        return True

    monkeypatch.setattr(mod, "zellij_inject", fake_inject)
    monkeypatch.setattr(mod, "zellij_rename_pane", lambda **kwargs: renamed.append(kwargs) or True)
    with kb.connect(board="codex-interactive-test") as conn:
        task_id = kb.create_task(
            conn,
            title="interactive planner smoke",
            body="确认 interactive Codex listener 能收到任务。",
            assignee="planner",
        )

    args = argparse.Namespace(
        board="codex-interactive-test",
        workspace=str(workspace),
        profile="planner",
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="7",
    )
    log_path = tmp_path / "interactive.log"
    active_task, run_id = mod.claim_and_inject_one(args, log_path=log_path)

    assert active_task == task_id
    assert isinstance(run_id, int)
    assert injected["session"] == "test-session"
    assert injected["pane_id"] == "7"
    assert renamed[-1]["session"] == "test-session"
    assert renamed[-1]["pane_id"] == "7"
    assert renamed[-1]["name"] == f"planner-codex running {task_id}"
    assert task_id in injected["text"]
    prompt_path = workspace / ".codex-kanban" / "codex-interactive-test" / "planner" / f"{task_id}.md"
    assert prompt_path.exists()
    prompt = prompt_path.read_text(encoding="utf-8")
    assert f"Task: {task_id}" in prompt
    assert f"hermes kanban --board codex-interactive-test complete {task_id}" in prompt

    with kb.connect(board="codex-interactive-test") as conn:
        row = conn.execute(
            "SELECT status, workspace_path, claim_lock FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["workspace_path"] == str(workspace)
        assert "codex-interactive" in row["claim_lock"]


def test_codex_interactive_assist_claims_implementer_with_pane_profile_prompt_dir(kanban_home, tmp_path, monkeypatch):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    injected = {}

    def fake_inject(*, session, pane_id, text, log_path):
        injected.update({"session": session, "pane_id": pane_id, "text": text, "log_path": log_path})
        return True

    monkeypatch.setattr(mod, "zellij_inject", fake_inject)
    with kb.connect(board="codex-assist-test") as conn:
        task_id = kb.create_task(
            conn,
            title="assist implementer smoke",
            body="planner pane 空闲时应能领取 implementer lane 任务。",
            assignee="implementer",
        )

    args = argparse.Namespace(
        board="codex-assist-test",
        workspace=str(workspace),
        profile="planner",
        claim_assignees=["planner", "implementer"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="7",
    )
    log_path = tmp_path / "interactive.log"
    active_task, run_id = mod.claim_and_inject_one(args, log_path=log_path)

    assert active_task == task_id
    assert isinstance(run_id, int)
    prompt_path = workspace / ".codex-kanban" / "codex-assist-test" / "planner" / f"{task_id}.md"
    assert prompt_path.exists()
    prompt = prompt_path.read_text(encoding="utf-8")
    assert "Pane/profile: planner" in prompt
    assert "Task assignee/role: implementer" in prompt
    assert "你现在承担 Hermes Kanban 角色：implementer" in prompt
    assert "虽然当前 TUI pane/profile 是 planner" in prompt
    assert task_id in injected["text"]

    with kb.connect(board="codex-assist-test") as conn:
        row = conn.execute(
            "SELECT status, assignee, workspace_path, claim_lock FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["assignee"] == "implementer"
        assert row["workspace_path"] == str(workspace)
        assert "codex-interactive" in row["claim_lock"]


def test_codex_self_poll_prompt_uses_kanban_next_and_build_cmd_appends_once(tmp_path, monkeypatch):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_SELF_POLL_OWNER", "planner-pane-a")
    monkeypatch.setattr(mod, "has_saved_codex_sessions", lambda workspace: False)

    prompt_path, prompt = mod.write_self_poll_startup_prompt(
        board="egomotion4d",
        profile="planner",
        claim_assignees=["planner", "implementer"],
        workspace=workspace,
        ttl=3600,
        listener_kind="codex-self-poll",
    )
    args = argparse.Namespace(
        workspace=str(workspace),
        continue_session=False,
        codex_bin="codex",
        no_alt_screen=True,
        model=None,
        sandbox="danger-full-access",
        codex_arg=[],
        startup_prompt=prompt,
    )

    cmd = mod.build_codex_cmd(args)

    assert prompt_path.exists()
    assert "hermes kanban --board egomotion4d next" in prompt
    assert "--profile planner" in prompt
    assert "--claim-assignees planner,implementer" in prompt
    assert "--listener-kind codex-self-poll" in prompt
    assert "--owner planner-pane-a" in prompt
    assert "Self-poll owner: planner-pane-a" in prompt
    assert "reset-current" in prompt
    assert cmd[-1] == prompt
    assert "KANBAN_TASK_BOUNDARY" not in prompt


def test_codex_and_deepseek_interactive_default_to_self_poll(tmp_path, monkeypatch):
    codex_mod = _load_codex_interactive()
    deepseek_mod = _load_deepseek_interactive()

    monkeypatch.delenv("HERMES_KANBAN_TASK_DELIVERY", raising=False)
    codex_args = codex_mod.parse_args(["--workspace", str(tmp_path)])
    deepseek_args = deepseek_mod.parse_args(["--workspace", str(tmp_path)])

    assert codex_args.task_delivery == "self-poll"
    assert deepseek_args.task_delivery == "self-poll"

    monkeypatch.setenv("HERMES_KANBAN_TASK_DELIVERY", "inject")
    assert codex_mod.parse_args(["--workspace", str(tmp_path)]).task_delivery == "inject"
    assert deepseek_mod.parse_args(["--workspace", str(tmp_path)]).task_delivery == "inject"


def test_codex_self_poll_launcher_does_not_require_zellij_or_start_watcher(kanban_home, tmp_path, monkeypatch):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    calls = {}

    monkeypatch.delenv("ZELLIJ_SESSION_NAME", raising=False)
    monkeypatch.delenv("ZELLIJ_PANE_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SELF_POLL_OWNER", raising=False)
    monkeypatch.setattr(mod, "has_saved_codex_sessions", lambda workspace: False)
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(AssertionError("self-poll must not wait for Enter")))
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("self-poll must not start watcher")))

    def fake_call(cmd, cwd, env):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["env"] = env
        return 0

    monkeypatch.setattr(mod.subprocess, "call", fake_call)
    args = mod.parse_args(
        [
            "--board",
            "egomotion4d",
            "--workspace",
            str(workspace),
            "--profile",
            "planner",
            "--claim-assignees",
            "planner,implementer",
            "--task-delivery",
            "self-poll",
            "--no-continue",
        ]
    )

    assert mod.launcher_main(args) == 0

    assert calls["cwd"] == str(workspace)
    assert calls["env"]["HERMES_KANBAN_TASK_DELIVERY"] == "self-poll"
    assert calls["env"]["HERMES_KANBAN_PROFILE"] == "planner"
    assert calls["env"]["HERMES_KANBAN_SELF_POLL_OWNER"].startswith("planner-")
    assert "hermes kanban --board egomotion4d next" in calls["cmd"][-1]
    assert "--claim-assignees planner,implementer" in calls["cmd"][-1]
    assert "--owner planner-" in calls["cmd"][-1]


def test_codex_interactive_reset_kanban_reclaims_matching_workspace_claim(kanban_home, tmp_path):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    lock = f"{socket.gethostname()}:99999999:codex-interactive"

    with kb.connect(board="codex-reset-kanban-test") as conn:
        task_id = kb.create_task(conn, title="stuck codex task", assignee="planner")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=lock)
        assert claimed is not None
        kb.set_workspace_path(conn, task_id, workspace)
        kb._set_worker_pid(conn, task_id, 99999999)  # type: ignore[attr-defined]

    reset_ids = mod.reset_kanban_claims(
        board="codex-reset-kanban-test",
        profile="planner",
        claim_assignees=["planner"],
        workspace=workspace,
        reason="operator reset-kanban",
    )

    assert reset_ids == [task_id]
    with kb.connect(board="codex-reset-kanban-test") as conn:
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert row["worker_pid"] is None


def test_codex_interactive_delays_assist_claim_until_ready_age(kanban_home, tmp_path, monkeypatch):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()

    monkeypatch.setattr(mod, "zellij_inject", lambda **kwargs: True)
    monkeypatch.setattr(mod, "zellij_rename_pane", lambda **kwargs: True)
    with kb.connect(board="codex-assist-delay-test") as conn:
        task_id = kb.create_task(conn, title="implementer should get first chance", assignee="implementer")

    args = argparse.Namespace(
        board="codex-assist-delay-test",
        workspace=str(workspace),
        profile="planner",
        claim_assignees=["planner", "implementer"],
        assist_claim_delay_s=60,
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="7",
    )

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "codex.log")
    assert active_task is None
    assert run_id is None

    with kb.connect(board="codex-assist-delay-test") as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert row["status"] == "ready"
        old_ts = int(mod.time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "codex.log")
    assert active_task == task_id
    assert isinstance(run_id, int)


def test_codex_interactive_uses_per_assignee_assist_claim_delay(kanban_home, tmp_path, monkeypatch):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()

    monkeypatch.setattr(mod, "zellij_inject", lambda **kwargs: True)
    monkeypatch.setattr(mod, "zellij_rename_pane", lambda **kwargs: True)
    with kb.connect(board="codex-assist-delay-for-test") as conn:
        task_id = kb.create_task(conn, title="planner waits longer for implementer", assignee="implementer")

    args = argparse.Namespace(
        board="codex-assist-delay-for-test",
        workspace=str(workspace),
        profile="planner",
        claim_assignees=["planner", "implementer"],
        assist_claim_delay_s=0,
        assist_claim_delay_for=["implementer=60"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="7",
    )

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "codex.log")
    assert active_task is None
    assert run_id is None

    with kb.connect(board="codex-assist-delay-for-test") as conn:
        old_ts = int(mod.time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "codex.log")
    assert active_task == task_id
    assert isinstance(run_id, int)


def test_codex_interactive_uses_profile_qualified_assist_claim_delay_for_custom_profile(kanban_home, tmp_path, monkeypatch):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()

    monkeypatch.setattr(mod, "zellij_inject", lambda **kwargs: True)
    monkeypatch.setattr(mod, "zellij_rename_pane", lambda **kwargs: True)
    with kb.connect(board="codex-custom-profile-delay-test") as conn:
        task_id = kb.create_task(conn, title="backup waits for implementer", assignee="implementer")

    args = argparse.Namespace(
        board="codex-custom-profile-delay-test",
        workspace=str(workspace),
        profile="backup_immplementer",
        claim_assignees=["backup_immplementer", "implementer"],
        assist_claim_delay_s=0,
        assist_claim_delay_for=None,
        assist_claim_profile_delay=["backup_immplementer:implementer=60"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="7",
    )

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "codex.log")
    assert active_task is None
    assert run_id is None

    with kb.connect(board="codex-custom-profile-delay-test") as conn:
        old_ts = int(mod.time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "codex.log")
    assert active_task == task_id
    assert isinstance(run_id, int)


def test_codex_startup_reclaims_orphaned_running_task_for_fresh_prompt(kanban_home, tmp_path):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    dead_pid = 99999999
    old_lock = f"{socket.gethostname()}:{dead_pid}:codex-interactive"

    with kb.connect(board="codex-orphan-reclaim-test") as conn:
        task_id = kb.create_task(conn, title="old planner running task", assignee="planner")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=old_lock)
        assert claimed is not None
        kb.set_workspace_path(conn, task_id, workspace)
        kb._set_worker_pid(conn, task_id, dead_pid)  # type: ignore[attr-defined]

    args = argparse.Namespace(
        board="codex-orphan-reclaim-test",
        workspace=str(workspace),
        profile="planner",
        claim_assignees=["planner", "implementer"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="7",
    )

    assert mod.reclaim_orphaned_running_task(args, log_path=tmp_path / "codex.log") is True

    with kb.connect(board="codex-orphan-reclaim-test") as conn:
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None


def test_codex_assist_does_not_reclaim_other_profile_running_task(kanban_home, tmp_path):
    mod = _load_codex_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    dead_pid = 99999999
    old_lock = f"{socket.gethostname()}:{dead_pid}:codex-interactive"

    with kb.connect(board="codex-assist-orphan-test") as conn:
        task_id = kb.create_task(conn, title="old implementer running task", assignee="implementer")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=old_lock)
        assert claimed is not None
        kb.set_workspace_path(conn, task_id, workspace)
        kb._set_worker_pid(conn, task_id, dead_pid)  # type: ignore[attr-defined]

    args = argparse.Namespace(
        board="codex-assist-orphan-test",
        workspace=str(workspace),
        profile="planner",
        claim_assignees=["planner", "implementer"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="7",
    )

    assert mod.reclaim_orphaned_running_task(args, log_path=tmp_path / "codex.log") is False

    with kb.connect(board="codex-assist-orphan-test") as conn:
        row = conn.execute(
            "SELECT status, assignee, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["assignee"] == "implementer"
        assert row["claim_lock"] == old_lock
        assert row["worker_pid"] == dead_pid


def test_deepseek_interactive_assist_claims_implementer_with_pane_profile_prompt_dir(kanban_home, tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    renamed = []

    monkeypatch.setattr(mod, "zellij_inject", lambda **kwargs: True)
    monkeypatch.setattr(mod, "zellij_rename_pane", lambda **kwargs: renamed.append(kwargs) or True)
    monkeypatch.setattr(mod, "_pane_screen", lambda args, log_path: "编写任务或使用 /。")
    with kb.connect(board="deepseek-assist-test") as conn:
        task_id = kb.create_task(
            conn,
            title="critic assists implementer",
            body="critic pane 空闲时应能领取 implementer lane 任务。",
            assignee="implementer",
        )

    args = argparse.Namespace(
        board="deepseek-assist-test",
        workspace=str(workspace),
        profile="critic",
        claim_assignees=["critic", "implementer"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="9",
    )
    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "deepseek.log")

    assert active_task == task_id
    assert isinstance(run_id, int)
    assert renamed[-1]["session"] == "test-session"
    assert renamed[-1]["pane_id"] == "9"
    assert renamed[-1]["name"] == f"critic-deepseek running {task_id}"
    prompt_path = workspace / ".deepseek-kanban" / "deepseek-assist-test" / "critic" / f"{task_id}.md"
    assert prompt_path.exists()
    prompt = prompt_path.read_text(encoding="utf-8")
    assert "Pane/profile: critic" in prompt
    assert "Task assignee/role: implementer" in prompt
    assert "你现在承担 Hermes Kanban 角色：implementer" in prompt


def test_deepseek_interactive_delays_assist_claim_until_ready_age(kanban_home, tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()

    monkeypatch.setattr(mod, "zellij_inject", lambda **kwargs: True)
    monkeypatch.setattr(mod, "zellij_rename_pane", lambda **kwargs: True)
    monkeypatch.setattr(mod, "_pane_screen", lambda args, log_path: "编写任务或使用 /。")
    with kb.connect(board="deepseek-assist-delay-test") as conn:
        task_id = kb.create_task(conn, title="implementer should get first chance", assignee="implementer")

    args = argparse.Namespace(
        board="deepseek-assist-delay-test",
        workspace=str(workspace),
        profile="critic",
        claim_assignees=["critic", "implementer"],
        assist_claim_delay_s=60,
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="9",
    )

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "deepseek.log")
    assert active_task is None
    assert run_id is None

    with kb.connect(board="deepseek-assist-delay-test") as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert row["status"] == "ready"
        old_ts = int(mod.time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "deepseek.log")
    assert active_task == task_id
    assert isinstance(run_id, int)


def test_deepseek_interactive_uses_per_assignee_assist_claim_delay(kanban_home, tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()

    monkeypatch.setattr(mod, "zellij_inject", lambda **kwargs: True)
    monkeypatch.setattr(mod, "zellij_rename_pane", lambda **kwargs: True)
    monkeypatch.setattr(mod, "_pane_screen", lambda args, log_path: "编写任务或使用 /。")
    with kb.connect(board="deepseek-assist-delay-for-test") as conn:
        task_id = kb.create_task(conn, title="critic waits shorter for implementer", assignee="implementer")

    args = argparse.Namespace(
        board="deepseek-assist-delay-for-test",
        workspace=str(workspace),
        profile="critic",
        claim_assignees=["critic", "implementer"],
        assist_claim_delay_s=0,
        assist_claim_delay_for=["implementer=60"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="9",
    )

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "deepseek.log")
    assert active_task is None
    assert run_id is None

    with kb.connect(board="deepseek-assist-delay-for-test") as conn:
        old_ts = int(mod.time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "deepseek.log")
    assert active_task == task_id
    assert isinstance(run_id, int)


def test_deepseek_kanban_defaults_to_continue_session(tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    monkeypatch.setattr(mod, "has_saved_sessions", lambda workspace: True)

    parsed = mod.parse_args(["--workspace", str(tmp_path)])
    assert parsed.continue_session is True

    cmd = mod.build_deepseek_cmd(parsed)
    assert "--continue" in cmd
    assert "--fresh" not in cmd


def test_deepseek_self_poll_prompt_uses_kanban_next_without_task_boundary(tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    monkeypatch.delenv("HERMES_KANBAN_SELF_POLL_OWNER", raising=False)

    prompt_path, prompt = mod.write_self_poll_startup_prompt(
        board="egomotion4d",
        profile="critic",
        claim_assignees=["critic", "implementer"],
        workspace=workspace,
        ttl=3600,
        listener_kind="deepseek-self-poll",
        pane_id="pane/9",
    )

    assert prompt_path.exists()
    assert "hermes kanban --board egomotion4d next" in prompt
    assert "--profile critic" in prompt
    assert "--claim-assignees critic,implementer" in prompt
    assert "--listener-kind deepseek-self-poll" in prompt
    assert "--owner critic-pane_9" in prompt
    assert "Self-poll owner: critic-pane_9" in prompt
    assert "reset-current" in prompt
    assert "KANBAN_TASK_BOUNDARY" not in prompt


def test_deepseek_self_poll_launcher_injects_startup_prompt_once_without_watcher(kanban_home, tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    injected = {}
    popen_cmds = []

    monkeypatch.delenv("HERMES_KANBAN_SELF_POLL_OWNER", raising=False)
    monkeypatch.setattr(mod, "has_saved_sessions", lambda workspace: False)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        mod,
        "zellij_inject",
        lambda *, session, pane_id, text, log_path: injected.update(
            {"session": session, "pane_id": pane_id, "text": text}
        )
        or True,
    )

    class FakeProc:
        pid = 12345

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        popen_cmds.append(cmd)
        assert "--watch-child" not in cmd
        return FakeProc()

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    args = mod.parse_args(
        [
            "--board",
            "egomotion4d",
            "--workspace",
            str(workspace),
            "--profile",
            "critic",
            "--claim-assignees",
            "critic,implementer",
            "--task-delivery",
            "self-poll",
            "--startup-delay-s",
            "0",
            "--zellij-session",
            "test-session",
            "--zellij-pane-id",
            "9",
            "--no-continue",
        ]
    )

    assert mod.launcher_main(args) == 0

    assert len(popen_cmds) == 1
    assert popen_cmds[0][0] == "deepseek-tui"
    assert injected["session"] == "test-session"
    assert injected["pane_id"] == "9"
    assert "hermes kanban --board egomotion4d next" in injected["text"]
    assert "--claim-assignees critic,implementer" in injected["text"]
    assert "--owner critic-9" in injected["text"]
    assert "KANBAN_TASK_BOUNDARY" not in injected["text"]


def test_deepseek_no_continue_uses_fresh_session(tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    monkeypatch.setattr(mod, "has_saved_sessions", lambda workspace: True)

    parsed = mod.parse_args(["--workspace", str(tmp_path), "--no-continue"])
    assert parsed.continue_session is False

    cmd = mod.build_deepseek_cmd(parsed)
    assert "--fresh" in cmd
    assert "--continue" not in cmd


def test_deepseek_injection_contains_hard_task_boundary(kanban_home, tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    injected = {}

    def fake_inject(*, session, pane_id, text, log_path):
        injected.update({"text": text})
        return True

    monkeypatch.setattr(mod, "zellij_inject", fake_inject)
    monkeypatch.setattr(mod, "_pane_screen", lambda args, log_path: "编写任务或使用 /。")
    with kb.connect(board="deepseek-boundary-test") as conn:
        task_id = kb.create_task(conn, title="new boundary task", assignee="implementer")

    args = argparse.Namespace(
        board="deepseek-boundary-test",
        workspace=str(workspace),
        profile="implementer",
        claim_assignees=None,
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="9",
    )
    mod.claim_and_inject_one(args, log_path=tmp_path / "deepseek.log")

    assert "KANBAN_TASK_BOUNDARY" in injected["text"]
    assert "不要延续上一轮输出" in injected["text"]
    assert "File:" in injected["text"]
    assert task_id in injected["text"]


def test_deepseek_claim_delay_after_task_transition():
    mod = _load_deepseek_interactive()

    assert mod._claim_delay_remaining(last_transition_at=100.0, now=103.5, delay_s=8.0) == pytest.approx(4.5)
    assert mod._claim_delay_remaining(last_transition_at=100.0, now=109.0, delay_s=8.0) == pytest.approx(0.0)
    assert mod._claim_delay_remaining(last_transition_at=0.0, now=109.0, delay_s=8.0) == pytest.approx(0.0)


def test_deepseek_does_not_accept_new_task_while_boundary_prompt_visible():
    mod = _load_deepseek_interactive()

    pending = (
        "KANBAN_TASK_BOUNDARY\n"
        "Hermes Kanban 已领取任务 t_demo: unfinished\n"
        "完成后必须运行 hermes kanban --board demo complete t_demo ...\n"
        "工作中"
    )
    queued = "• Pending inputs\n  ↳ hi\n    ↑ edit last queued message\n编写任务或使用 /。"
    active_tool = "╭ ◦ ▷ read running · Searching\n╰ ▏ live: Searching\n编写任务或使用 /。"
    idle = "编写任务或使用 /"

    assert mod._pane_can_accept_new_kanban_task(pending) is False
    assert mod._pane_can_accept_new_kanban_task(queued) is False
    assert mod._pane_can_accept_new_kanban_task(active_tool) is False
    assert mod._pane_can_accept_new_kanban_task(idle) is True


def test_deepseek_claim_does_not_inject_while_pane_has_pending_input(kanban_home, tmp_path, monkeypatch):
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()

    monkeypatch.setattr(
        mod,
        "_pane_screen",
        lambda args, log_path: "• Pending inputs\n  ↳ hi\n编写任务或使用 /。",
    )

    def fail_inject(**kwargs):
        raise AssertionError("listener must not inject when DeepSeek-TUI has queued input")

    monkeypatch.setattr(mod, "zellij_inject", fail_inject)
    with kb.connect(board="deepseek-pending-input-test") as conn:
        task_id = kb.create_task(conn, title="must wait", assignee="implementer")

    args = argparse.Namespace(
        board="deepseek-pending-input-test",
        workspace=str(workspace),
        profile="implementer",
        claim_assignees=None,
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="9",
    )

    active_task, run_id = mod.claim_and_inject_one(args, log_path=tmp_path / "deepseek.log")

    assert (active_task, run_id) == (None, None)
    with kb.connect(board="deepseek-pending-input-test") as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert row["status"] == "ready"


def test_deepseek_interactive_reset_kanban_reclaims_matching_workspace_claim(kanban_home, tmp_path):
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    lock = f"{socket.gethostname()}:99999999:deepseek-interactive"

    with kb.connect(board="deepseek-reset-kanban-test") as conn:
        task_id = kb.create_task(conn, title="stuck deepseek task", assignee="implementer")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=lock)
        assert claimed is not None
        kb.set_workspace_path(conn, task_id, workspace)
        kb._set_worker_pid(conn, task_id, 99999999)  # type: ignore[attr-defined]

    reset_ids = mod.reset_kanban_claims(
        board="deepseek-reset-kanban-test",
        profile="implementer",
        claim_assignees=["implementer"],
        workspace=workspace,
        reason="operator reset-kanban",
    )

    assert reset_ids == [task_id]
    with kb.connect(board="deepseek-reset-kanban-test") as conn:
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert row["worker_pid"] is None


def test_deepseek_launcher_restarts_only_crashed_watchers():
    mod = _load_deepseek_interactive()

    assert mod._should_restart_watcher(0) is False
    assert mod._should_restart_watcher(None) is False
    assert mod._should_restart_watcher(1) is True


def test_deepseek_assist_does_not_silently_adopt_other_profile_running_task(kanban_home, tmp_path):
    """Assist panes must not adopt true implementer tasks without prompt injection."""
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    dead_pid = 99999999
    old_lock = f"{socket.gethostname()}:{dead_pid}:deepseek-interactive"

    with kb.connect(board="deepseek-assist-adopt-test") as conn:
        task_id = kb.create_task(
            conn,
            title="real implementer running task",
            assignee="implementer",
        )
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=old_lock)
        assert claimed is not None
        kb.set_workspace_path(conn, task_id, workspace)
        kb._set_worker_pid(conn, task_id, dead_pid)  # type: ignore[attr-defined]

    args = argparse.Namespace(
        board="deepseek-assist-adopt-test",
        workspace=str(workspace),
        profile="critic",
        claim_assignees=["critic", "implementer"],
        ttl=3600,
    )

    adopted_task, adopted_run = mod.adopt_orphaned_running_task(
        args,
        log_path=tmp_path / "deepseek.log",
    )

    assert adopted_task is None
    assert adopted_run is None
    with kb.connect(board="deepseek-assist-adopt-test") as conn:
        row = conn.execute(
            "SELECT status, assignee, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["assignee"] == "implementer"
        assert row["claim_lock"] == old_lock
        assert row["worker_pid"] == dead_pid


def test_deepseek_startup_reclaims_orphaned_running_task_when_pane_is_idle(kanban_home, tmp_path, monkeypatch):
    """After zellij restart, an idle restored pane must not keep old running tasks alive."""
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    dead_pid = 99999999
    old_lock = f"{socket.gethostname()}:{dead_pid}:deepseek-interactive"

    def fake_run(cmd, **kwargs):
        assert "dump-screen" in cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="Hi，有什么想继续的？\n编写任务或使用 /。\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    with kb.connect(board="deepseek-idle-adopt-test") as conn:
        task_id = kb.create_task(conn, title="old critic running task", assignee="critic")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=old_lock)
        assert claimed is not None
        kb.set_workspace_path(conn, task_id, workspace)
        kb._set_worker_pid(conn, task_id, dead_pid)  # type: ignore[attr-defined]

    args = argparse.Namespace(
        board="deepseek-idle-adopt-test",
        workspace=str(workspace),
        profile="critic",
        claim_assignees=["critic", "implementer"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="9",
    )

    adopted_task, adopted_run = mod.adopt_orphaned_running_task(
        args,
        log_path=tmp_path / "deepseek.log",
    )

    assert adopted_task is None
    assert adopted_run is None
    with kb.connect(board="deepseek-idle-adopt-test") as conn:
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None


def test_deepseek_startup_still_adopts_orphaned_running_task_when_pane_is_busy(kanban_home, tmp_path, monkeypatch):
    """A non-idle pane may still be carrying the old task, so preserve adoption."""
    mod = _load_deepseek_interactive()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    dead_pid = 99999999
    old_lock = f"{socket.gethostname()}:{dead_pid}:deepseek-interactive"

    def fake_run(cmd, **kwargs):
        assert "dump-screen" in cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="正在执行任务，请稍候...\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    with kb.connect(board="deepseek-busy-adopt-test") as conn:
        task_id = kb.create_task(conn, title="active critic running task", assignee="critic")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=old_lock)
        assert claimed is not None
        run_id = claimed.current_run_id
        kb.set_workspace_path(conn, task_id, workspace)
        kb._set_worker_pid(conn, task_id, dead_pid)  # type: ignore[attr-defined]

    args = argparse.Namespace(
        board="deepseek-busy-adopt-test",
        workspace=str(workspace),
        profile="critic",
        claim_assignees=["critic", "implementer"],
        ttl=3600,
        zellij_session="test-session",
        zellij_pane_id="9",
    )

    adopted_task, adopted_run = mod.adopt_orphaned_running_task(
        args,
        log_path=tmp_path / "deepseek.log",
    )

    assert adopted_task == task_id
    assert adopted_run == run_id
    with kb.connect(board="deepseek-busy-adopt-test") as conn:
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["claim_lock"].endswith(":deepseek-interactive")
        assert row["claim_lock"] != old_lock
        assert row["worker_pid"] == mod.os.getpid()


def test_deepseek_internal_reclaim_does_not_signal_watcher_pid(tmp_path, monkeypatch):
    """Interactive recovery must not kill its own watcher while requeueing work."""
    mod = _load_deepseek_interactive()
    captured = {}

    def fake_reclaim(conn, task_id, *, reason=None, signal_fn=None):
        captured.update({"task_id": task_id, "reason": reason, "signal_fn": signal_fn})
        return True

    monkeypatch.setattr(mod.kb, "reclaim_task", fake_reclaim)

    assert mod._reclaim_task_without_signaling_worker(object(), "t_demo", reason="idle pane") is True
    assert captured["task_id"] == "t_demo"
    assert captured["reason"] == "idle pane"
    with pytest.raises(ProcessLookupError):
        captured["signal_fn"](mod.os.getpid(), 15)


def test_deepseek_idle_detector_treats_live_working_status_as_busy():
    mod = _load_deepseek_interactive()

    screen = "● Live  25%\\n编写任务或使用 /。\\n工作中"

    assert mod._looks_like_idle_deepseek_pane(screen) is False


def test_deepseek_stalled_busy_detector_requires_no_progress_and_no_external_child():
    mod = _load_deepseek_interactive()
    watch = mod._PaneProgressWatch()
    screen = "KANBAN_TASK_BOUNDARY\nHermes Kanban 已领取任务 t_demo\n工作中"

    assert mod._observe_pane_progress(
        watch,
        screen=screen,
        now=1000.0,
        latest_session_mtime=900.0,
        has_external_child=False,
        reclaim_s=600.0,
    ) is None
    assert watch.stalled_busy_seen_at == 1000.0

    reason = mod._observe_pane_progress(
        watch,
        screen=screen,
        now=1601.0,
        latest_session_mtime=900.0,
        has_external_child=False,
        reclaim_s=600.0,
    )
    assert reason is not None
    assert "pane looked busy but made no progress" in reason


def test_deepseek_stalled_busy_detector_resets_on_screen_session_or_child_progress():
    mod = _load_deepseek_interactive()
    screen = "KANBAN_TASK_BOUNDARY\nHermes Kanban 已领取任务 t_demo\n工作中"

    watch = mod._PaneProgressWatch()
    assert mod._observe_pane_progress(
        watch,
        screen=screen,
        now=1000.0,
        latest_session_mtime=900.0,
        has_external_child=False,
        reclaim_s=600.0,
    ) is None
    assert mod._observe_pane_progress(
        watch,
        screen=screen + "\nnew output",
        now=1700.0,
        latest_session_mtime=900.0,
        has_external_child=False,
        reclaim_s=600.0,
    ) is None
    assert watch.stalled_busy_seen_at == 1700.0

    watch = mod._PaneProgressWatch()
    assert mod._observe_pane_progress(
        watch,
        screen=screen,
        now=1000.0,
        latest_session_mtime=900.0,
        has_external_child=False,
        reclaim_s=600.0,
    ) is None
    assert mod._observe_pane_progress(
        watch,
        screen=screen,
        now=1700.0,
        latest_session_mtime=1200.0,
        has_external_child=False,
        reclaim_s=600.0,
    ) is None
    assert watch.stalled_busy_seen_at is None

    watch = mod._PaneProgressWatch()
    assert mod._observe_pane_progress(
        watch,
        screen=screen,
        now=1000.0,
        latest_session_mtime=900.0,
        has_external_child=False,
        reclaim_s=600.0,
    ) is None
    assert mod._observe_pane_progress(
        watch,
        screen=screen,
        now=1700.0,
        latest_session_mtime=900.0,
        has_external_child=True,
        reclaim_s=600.0,
    ) is None
    assert watch.stalled_busy_seen_at is None
