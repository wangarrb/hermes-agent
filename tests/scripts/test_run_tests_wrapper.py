import os
import subprocess
from pathlib import Path


def test_run_tests_skips_unusable_dotvenv_when_fallback_venv_exists():
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "run_tests.sh"
    env = os.environ.copy()
    env["HERMES_TEST_WORKERS"] = "0"

    result = subprocess.run(
        [str(script), "--version"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert "No module named pip" not in result.stdout
    assert result.returncode == 0, result.stdout


def test_run_tests_clears_kanban_worker_environment(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "run_tests.sh"
    probe = tmp_path / "test_kanban_env_probe.py"
    probe.write_text(
        "import os\n\n"
        "def test_kanban_env_is_clean():\n"
        "    leaked = sorted(k for k in os.environ if k.startswith('HERMES_KANBAN_'))\n"
        "    assert leaked == []\n"
    )
    env = os.environ.copy()
    env["HERMES_TEST_WORKERS"] = "0"
    env["HERMES_KANBAN_BOARD"] = "leaky-board"
    env["HERMES_KANBAN_TASK"] = "t_leaky"
    env["HERMES_KANBAN_WORKSPACE"] = "/tmp/leaky"

    result = subprocess.run(
        [str(script), str(probe), "-q"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout
