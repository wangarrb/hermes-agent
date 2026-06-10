import subprocess
from pathlib import Path

from hermes_cli.agent_proxy_env import LOCAL_NO_PROXY, without_agent_proxy_env


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_python_agent_proxy_env_removes_proxy_vars() -> None:
    env = without_agent_proxy_env(
        {
            "HTTP_PROXY": "http://127.0.0.1:7890/",
            "HTTPS_PROXY": "http://127.0.0.1:7890/",
            "http_proxy": "http://127.0.0.1:7890/",
            "https_proxy": "http://127.0.0.1:7890/",
            "ALL_PROXY": "socks5://127.0.0.1:7890",
            "all_proxy": "socks5://127.0.0.1:7890",
            "NO_PROXY": "localhost,127.0.0.0/8",
            "PATH": "/usr/bin:/bin",
        }
    )

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        assert key not in env
    assert env["NO_PROXY"] == LOCAL_NO_PROXY
    assert env["no_proxy"] == LOCAL_NO_PROXY
    assert env["PATH"] == "/usr/bin:/bin"


def test_shell_agent_env_helper_removes_proxy_vars() -> None:
    helper = REPO_ROOT / "local" / "bin" / "agent-env.sh"
    proc = subprocess.run(
        [
            "bash",
            "-lc",
            f"source {helper}; agent_env_disable_proxy; env",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={
            "HTTP_PROXY": "http://127.0.0.1:7890/",
            "HTTPS_PROXY": "http://127.0.0.1:7890/",
            "http_proxy": "http://127.0.0.1:7890/",
            "https_proxy": "http://127.0.0.1:7890/",
            "ALL_PROXY": "socks5://127.0.0.1:7890",
            "all_proxy": "socks5://127.0.0.1:7890",
            "NO_PROXY": "localhost,127.0.0.0/8",
        },
    )
    output = proc.stdout

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        assert f"{key}=" not in output
    assert f"NO_PROXY={LOCAL_NO_PROXY}" in output
    assert f"no_proxy={LOCAL_NO_PROXY}" in output
