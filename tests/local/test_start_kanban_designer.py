from __future__ import annotations

import os
import json
import re
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = REPO_ROOT / "local/bin/start-kanban.sh"
STOP_SCRIPT = REPO_ROOT / "local/bin/stop-kanban.sh"


def _write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _sandboxed_launcher(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    fake_home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    fake_home.mkdir()
    fake_bin.mkdir()

    for command in ("hermes", "zellij"):
        _write_executable(fake_bin / command)
    for wrapper in (
        "codex-kanban-interactive",
        "codewhale-kanban-interactive",
        "reasonix-kanban-interactive",
        "claude-kanban-interactive",
    ):
        _write_executable(fake_home / ".local/bin" / wrapper)

    source = START_SCRIPT.read_text(encoding="utf-8")
    original = 'export REAL_HOME="/home/wyr"'
    assert original in source
    launcher = tmp_path / "local/bin/start-kanban.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        source.replace(original, f'export REAL_HOME="{fake_home}"', 1),
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    test_lib = tmp_path / "local/lib"
    test_lib.mkdir(parents=True)
    for name in ("reviewer_mode.sh", "owner_workspace.sh", "codex_role_home.sh"):
        (test_lib / name).write_text(
            (REPO_ROOT / "local/lib" / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    helper = tmp_path / "local/bin/hermes-kanban-owner-workspace"
    helper.write_text(
        (REPO_ROOT / "local/bin/hermes-kanban-owner-workspace").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    for key in tuple(env):
        if key.startswith("KANBAN_") or key.startswith("CODEWHALE_KANBAN_"):
            env.pop(key)
    return launcher, fake_home, env


def _run_launcher(
    tmp_path: Path,
    *args: str,
    env_updates: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], str, Path, Path]:
    launcher, fake_home, env = _sandboxed_launcher(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if env_updates:
        env.update(env_updates)

    result = subprocess.run(
        [
            str(launcher),
            "--board",
            "test-board",
            "--workspace",
            str(workspace),
            "--dry-run",
            *args,
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    layout_path = fake_home / ".config/zellij/layouts/kanban-launcher.kdl"
    layout = layout_path.read_text(encoding="utf-8") if layout_path.exists() else ""
    designer_workspace = Path(f"{workspace.resolve()}-designer")
    return result, layout, workspace.resolve(), designer_workspace


def _pane(layout: str, role: str) -> str:
    match = re.search(
        rf'pane name="{role}-[^\"]+".*?\n\s*\}}',
        layout,
        flags=re.DOTALL,
    )
    assert match, f"missing {role} pane in:\n{layout}"
    return match.group(0)


def _write_board_binding(fake_home: Path, board: str, workspace: Path) -> None:
    board_dir = fake_home / ".hermes/kanban/boards" / board
    board_dir.mkdir(parents=True)
    (board_dir / "board.json").write_text(
        json.dumps(
            {
                "slug": board,
                "default_workdir": str(workspace.resolve()),
                "archived": False,
            }
        ),
        encoding="utf-8",
    )


def test_board_binding_supplies_workspace_when_w_is_omitted(tmp_path: Path) -> None:
    launcher, fake_home, env = _sandboxed_launcher(tmp_path)
    workspace = tmp_path / "SeqScale"
    workspace.mkdir()
    _write_board_binding(fake_home, "seqscale", workspace)

    result = subprocess.run(
        [str(launcher), "--board", "seqscale", "--dry-run"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"workspace: {workspace.resolve()}" in result.stdout
    assert "/code/Egomotion4D" not in result.stdout


def test_explicit_workspace_must_match_board_binding(tmp_path: Path) -> None:
    launcher, fake_home, env = _sandboxed_launcher(tmp_path)
    seqscale = tmp_path / "SeqScale"
    egomotion = tmp_path / "Egomotion4D"
    seqscale.mkdir()
    egomotion.mkdir()
    _write_board_binding(fake_home, "seqscale", seqscale)

    result = subprocess.run(
        [
            str(launcher),
            "--board", "seqscale",
            "--workspace", str(egomotion),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "workspace" in result.stderr.lower()
    assert "board" in result.stderr.lower()
    assert str(seqscale.resolve()) in result.stderr
    assert str(egomotion.resolve()) in result.stderr


def test_default_layout_has_designer_and_no_critic(tmp_path: Path) -> None:
    result, layout, workspace, designer_workspace = _run_launcher(
        tmp_path,
        "--reviewer-agent",
        "hermes",
    )

    assert result.returncode == 0, result.stderr
    names = re.findall(r'pane name="([^\"]+)"', layout)
    assert names == [
        "planner-hermes",
        "reviewer-hermes",
        "implementer-hermes",
        "designer-hermes",
        "coordinator-hermes",
    ]
    assert "critic" not in layout.lower()
    assert f'cwd="{designer_workspace}"' in _pane(layout, "designer")
    assert f'cwd="{workspace}-coordinator"' in _pane(layout, "coordinator")
    for role in ("planner", "reviewer", "implementer"):
        assert f'cwd="{workspace}"' in _pane(layout, role)
    assert f"designer workspace: {designer_workspace}" in result.stdout
    assert f"coordinator workspace: {workspace}-coordinator" in result.stdout


def test_every_backend_exports_its_logical_origin_profile(tmp_path: Path) -> None:
    result, layout, _, _ = _run_launcher(
        tmp_path,
        "--planner-agent", "hermes",
        "--reviewer-agent", "codex",
        "--implementer-agent", "codewhale",
        "--designer-agent", "deepseek-reasonix",
        "--coordinator-agent", "claude",
    )

    assert result.returncode == 0, result.stderr
    for role in ("planner", "reviewer", "implementer", "designer", "coordinator"):
        assert f"HERMES_KANBAN_ORIGIN_PROFILE={role}" in _pane(layout, role)


def test_designer_cli_env_and_profile_keep_workspace_isolated(tmp_path: Path) -> None:
    result, layout, workspace, designer_workspace = _run_launcher(
        tmp_path,
        "--planner-agent",
        "codex",
        "-d",
        "codex",
        "--reviewer-agent",
        "hermes",
        env_updates={"KANBAN_DESIGNER_AGENT": "claude"},
    )

    assert result.returncode == 0, result.stderr
    planner = _pane(layout, "planner")
    designer = _pane(layout, "designer")
    assert "planner-codex" in planner
    assert "designer-codex" in designer
    assert f"CODEX_KANBAN_WORKSPACE={workspace}" in planner
    assert f"--workspace {workspace}" in planner
    assert f"CODEX_KANBAN_WORKSPACE={designer_workspace}" in designer
    assert f"--workspace {designer_workspace}" in designer
    assert ".codex-kanban/planner" in planner
    assert ".codex-kanban/designer" in designer
    assert "--profile planner" in planner
    assert "--profile designer" in designer
    assert "--claim-assignees designer" in designer


def test_designer_environment_mapping_is_used_without_cli_override(tmp_path: Path) -> None:
    override = tmp_path / "explicit-designer"
    result, layout, _, designer_workspace = _run_launcher(
        tmp_path,
        "--reviewer-agent",
        "hermes",
        env_updates={
            "KANBAN_DESIGNER_AGENT": "claude",
            "KANBAN_DESIGNER_WORKSPACE": str(override),
        },
    )

    assert result.returncode == 0, result.stderr
    designer = _pane(layout, "designer")
    assert "designer-claude" in designer
    assert f"CLAUDE_KANBAN_WORKSPACE={override}" in designer
    assert f"--workspace {override}" in designer


@pytest.mark.parametrize("option", ["-c", "--critic-agent"])
def test_removed_critic_option_exits_two_with_migration_hint(
    tmp_path: Path, option: str
) -> None:
    launcher, _, env = _sandboxed_launcher(tmp_path)
    result = subprocess.run(
        [str(launcher), "--board", "test-board", option, "none", "--dry-run"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "critic" in result.stderr.lower()
    assert "--designer-agent" in result.stderr
    assert "已移除" in result.stderr or "迁移" in result.stderr


def test_reviewer_supports_implementer_assist_delays_and_previous_worker_delay(
    tmp_path: Path,
) -> None:
    result, layout, _, _ = _run_launcher(
        tmp_path,
        "--designer-agent",
        "codex",
        "--reviewer-agent",
        "hermes",
        "--assist-role",
        "reviewer:implementer",
        "--assist-role",
        "planner:designer",
        "--assist-role-delay",
        "reviewer:implementer:12",
        "--previous-worker-delay-s",
        "77",
    )

    assert result.returncode == 0, result.stderr
    reviewer = _pane(layout, "reviewer")
    planner = _pane(layout, "planner")
    assert "HERMES_KANBAN_CLAIM_ASSIGNEES=reviewer,implementer" in reviewer
    assert "HERMES_KANBAN_ASSIST_CLAIM_DELAYS=implementer=12" in reviewer
    assert "HERMES_KANBAN_PREVIOUS_WORKER_DELAY_S=77" in reviewer
    assert "HERMES_KANBAN_CLAIM_ASSIGNEES=planner,designer" in planner


def test_designer_can_be_continue_primary_and_agent_mapping_is_dynamic(
    tmp_path: Path,
) -> None:
    result, layout, _, designer_workspace = _run_launcher(
        tmp_path,
        "--designer-agent",
        "codewhale",
        "--implementer-agent",
        "codewhale",
        "--reviewer-agent",
        "hermes",
        "--deepseek-continue-policy",
        "primary-only",
        "--deepseek-continue-primary",
        "designer",
    )

    assert result.returncode == 0, result.stderr
    designer = _pane(layout, "designer")
    implementer = _pane(layout, "implementer")
    assert "designer-codewhale" in designer
    assert f"CODEWHALE_KANBAN_WORKSPACE={designer_workspace}" in designer
    assert "--continue" in designer
    assert "--no-continue" not in designer
    assert "--no-continue" in implementer
    assert "primary=designer" in result.stdout


def test_planner_and_designer_hermes_start_with_equal_toolsets(
    tmp_path: Path,
) -> None:
    result, layout, _, designer_workspace = _run_launcher(
        tmp_path,
        "--designer-agent",
        "hermes",
        "--reviewer-agent",
        "hermes",
        "--hermes-toolsets",
        "file,terminal,skills",
    )

    assert result.returncode == 0, result.stderr
    planner = _pane(layout, "planner")
    designer = _pane(layout, "designer")
    expected = "HERMES_KANBAN_TOOLSETS=file\\\\,terminal\\\\,skills"
    assert expected in planner
    assert expected in designer
    assert "-p planner" in planner
    assert "-p designer" in designer
    assert f"cd {designer_workspace}" in designer


def test_help_documents_project_neutral_owner_workspaces(tmp_path: Path) -> None:
    launcher, _, env = _sandboxed_launcher(tmp_path)
    result = subprocess.run(
        [str(launcher), "--help"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "-d, --designer-agent" in result.stdout
    assert "--designer-workspace" in result.stdout
    assert "--coordinator-workspace" in result.stdout
    assert "<primary>-designer" in result.stdout
    assert "<primary>-coordinator" in result.stdout
    assert "/home/wyr/code/Egomotion4D-designer" not in result.stdout
    assert "reviewer:implementer" in result.stdout
    assert "designer:implementer" not in result.stdout
    assert "coordinator:implementer" not in result.stdout
    assert "--switch-owner-project" in result.stdout
    assert "--target-project" in result.stdout


def test_cli_overrides_both_owner_workspaces(tmp_path: Path) -> None:
    designer = tmp_path / "designer-explicit"
    coordinator = tmp_path / "coordinator-explicit"
    result, layout, _, _ = _run_launcher(
        tmp_path,
        "--designer-workspace",
        str(designer),
        "--coordinator-workspace",
        str(coordinator),
        "--reviewer-agent",
        "hermes",
    )

    assert result.returncode == 0, result.stderr
    assert f'cwd="{designer}"' in _pane(layout, "designer")
    assert f'cwd="{coordinator}"' in _pane(layout, "coordinator")


def test_launcher_delegates_real_owner_preparation_to_helper() -> None:
    source = START_SCRIPT.read_text(encoding="utf-8")

    assert 'OWNER_WORKSPACE_HELPER="$SCRIPT_DIR/hermes-kanban-owner-workspace"' in source
    assert '"$OWNER_WORKSPACE_HELPER" prepare --role designer' in source
    assert '"$OWNER_WORKSPACE_HELPER" prepare --role coordinator' in source
    assert 'if [ "$DRY_RUN" != "1" ]; then' in source


def test_stop_script_only_matches_active_roles() -> None:
    source = STOP_SCRIPT.read_text(encoding="utf-8")

    assert 'roles = {"coordinator", "planner", "implementer", "designer", "reviewer"}' in source
    assert "coordinator|planner|implementer|designer|reviewer" in source
    assert '"critic"' not in source


@pytest.mark.parametrize("script", [START_SCRIPT, STOP_SCRIPT])
def test_process_cleanup_requires_an_active_listener_profile(script: Path) -> None:
    source = script.read_text(encoding="utf-8")

    listener_branch = source.split("elif listener_re.search(cmd):", 1)[1].split(
        "else:", 1
    )[0]
    assert "profile =" in listener_branch
    assert "profile in roles and board_matches(cmd, env)" in listener_branch
