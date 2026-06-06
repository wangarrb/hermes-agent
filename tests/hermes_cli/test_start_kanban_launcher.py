from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


START_KANBAN = Path("/home/wyr/bin/start-kanban.sh")


def _write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _line_with(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return line
    raise AssertionError(f"missing line containing {needle!r}\n{text}")


def test_start_kanban_assist_role_dry_run_generates_claim_assignees(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    fake_bin = tmp_path / "bin"
    workspace.mkdir()
    _write_executable(fake_bin / "zellij")
    _write_executable(fake_bin / "hermes")
    _write_executable(home / ".local" / "bin" / "codex-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "codex-kanban-listen")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-listen")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        [
            str(START_KANBAN),
            "-b",
            "egomotion4d",
            "-w",
            str(workspace),
            "-i",
            "deepseek-tui",
            "-p",
            "codex",
            "-c",
            "hermes",
            "--assist-role",
            "planner:implementer",
            "--assist-role",
            "critic:implementer",
            "--idle-pane-reclaim-s",
            "600",
            "--assist-claim-delay-s",
            "90",
            "-n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--profile planner" in out
    assert "--claim-assignees planner,implementer" in out
    assert "--assist-claim-delay-s 90" in out
    assert "--profile implementer" in out
    assert "--claim-assignees implementer" in out
    assert "--idle-pane-reclaim-s 600" in out
    assert "HERMES_KANBAN_ASSIST_CLAIM_DELAY_S=90" in out
    assert "HERMES_KANBAN_CLAIM_ASSIGNEES=critic,implementer" in out
    assert "hermes -p critic --continue" in out
    assert "implementer2" not in out
    assert "implementer3" not in out


def test_start_kanban_assist_role_delay_dry_run_is_profile_specific(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    fake_bin = tmp_path / "bin"
    workspace.mkdir()
    _write_executable(fake_bin / "zellij")
    _write_executable(fake_bin / "hermes")
    _write_executable(home / ".local" / "bin" / "codex-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "codex-kanban-listen")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-listen")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        [
            str(START_KANBAN),
            "-b",
            "egomotion4d",
            "-w",
            str(workspace),
            "-p",
            "codex",
            "-c",
            "hermes",
            "--assist-role",
            "planner:implementer",
            "--assist-role",
            "critic:implementer",
            "--assist-role-delay",
            "planner:implementer:180",
            "--assist-role-delay",
            "critic:implementer:30",
            "--assist-profile-delay",
            "backup_immplementer:implementer:300",
            "-n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--profile planner" in out
    assert "--assist-claim-delay-for implementer=180" in out
    assert "HERMES_KANBAN_CLAIM_ASSIGNEES=critic,implementer" in out
    assert "HERMES_KANBAN_ASSIST_CLAIM_DELAYS=implementer=30" in out
    assert "HERMES_KANBAN_ASSIST_CLAIM_PROFILE_DELAYS=backup_immplementer:implementer=300" in out
    assert "hermes -p critic --continue" in out


def test_start_kanban_assist_delay_shorthand_defaults_to_implementer(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    fake_bin = tmp_path / "bin"
    workspace.mkdir()
    _write_executable(fake_bin / "zellij")
    _write_executable(fake_bin / "hermes")
    _write_executable(home / ".local" / "bin" / "codex-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "codex-kanban-listen")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-listen")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        [
            str(START_KANBAN),
            "-b",
            "egomotion4d",
            "-w",
            str(workspace),
            "--assist-role-delay",
            "backup_immplementer:300",
            "--assist-profile-delay",
            "45",
            "-n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "HERMES_KANBAN_ASSIST_CLAIM_PROFILE_DELAYS=" in out
    assert "backup_immplementer:implementer=300" in out
    assert "implementer:implementer=45" in out


def test_start_kanban_multiple_deepseek_panes_only_primary_continues(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    fake_bin = tmp_path / "bin"
    workspace.mkdir()
    _write_executable(fake_bin / "zellij")
    _write_executable(fake_bin / "hermes")
    _write_executable(home / ".local" / "bin" / "codex-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "codex-kanban-listen")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-listen")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        [
            str(START_KANBAN),
            "-b",
            "egomotion4d",
            "-w",
            str(workspace),
            "-i",
            "deepseek-tui",
            "-c",
            "deepseek-tui",
            "-p",
            "codex",
            "--task-delivery",
            "self-poll",
            "-n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    implementer_line = _line_with(out, "--profile implementer")
    critic_line = _line_with(out, "--profile critic")
    assert "DeepSeek continue policy: auto (primary=implementer, panes=2)" in out
    assert "--claim-assignees implementer" in implementer_line
    assert "--provider openrouter --continue" in implementer_line
    assert "--no-continue" not in implementer_line
    assert "--claim-assignees critic" in critic_line
    assert "--provider openrouter --no-continue" in critic_line
    assert "--provider openrouter --continue" not in critic_line


def test_start_kanban_task_delivery_self_poll_reaches_interactive_bridges(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    fake_bin = tmp_path / "bin"
    workspace.mkdir()
    _write_executable(fake_bin / "zellij")
    _write_executable(fake_bin / "hermes")
    _write_executable(home / ".local" / "bin" / "codex-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "codex-kanban-listen")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-listen")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        [
            str(START_KANBAN),
            "-b",
            "egomotion4d",
            "-w",
            str(workspace),
            "-p",
            "codex",
            "-i",
            "deepseek-tui",
            "--task-delivery",
            "self-poll",
            "-n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Task delivery: self-poll" in out
    assert "--task-delivery self-poll" in _line_with(out, "codex-kanban-interactive")
    assert "--task-delivery self-poll" in _line_with(out, "deepseek-kanban-interactive")


def test_start_kanban_defaults_to_worker_task_delivery(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    fake_bin = tmp_path / "bin"
    workspace.mkdir()
    _write_executable(fake_bin / "zellij")
    _write_executable(fake_bin / "hermes")
    _write_executable(home / ".local" / "bin" / "codex-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "codex-kanban-listen")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-listen")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
            "HERMES_KANBAN_TASK_DELIVERY": "self-poll",
        }
    )
    result = subprocess.run(
        [
            str(START_KANBAN),
            "-b",
            "egomotion4d",
            "-w",
            str(workspace),
            "-p",
            "codex",
            "-i",
            "deepseek-tui",
            "-n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Task delivery: worker" in out
    assert "codex-kanban-listen" in _line_with(out, "codex-kanban-listen")
    assert "--claim-assignees planner" in _line_with(out, "codex-kanban-listen")
    assert "deepseek-kanban-listen" in _line_with(out, "deepseek-kanban-listen")
    assert "--claim-assignees implementer" in _line_with(out, "deepseek-kanban-listen")


def test_start_kanban_explicit_inject_task_delivery_remains_available(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    fake_bin = tmp_path / "bin"
    workspace.mkdir()
    _write_executable(fake_bin / "zellij")
    _write_executable(fake_bin / "hermes")
    _write_executable(home / ".local" / "bin" / "codex-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "codex-kanban-listen")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-interactive")
    _write_executable(home / ".local" / "bin" / "deepseek-kanban-listen")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
        }
    )
    result = subprocess.run(
        [
            str(START_KANBAN),
            "-b",
            "egomotion4d",
            "-w",
            str(workspace),
            "-p",
            "codex",
            "-i",
            "deepseek-tui",
            "--task-delivery",
            "inject",
            "-n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Task delivery: inject" in out
    assert "--task-delivery inject" in _line_with(out, "codex-kanban-interactive")
    assert "--task-delivery inject" in _line_with(out, "deepseek-kanban-interactive")
