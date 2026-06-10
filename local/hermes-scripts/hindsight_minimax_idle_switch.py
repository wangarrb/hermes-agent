#!/usr/bin/env python3
"""Safely switch Hindsight consolidation to MiniMax after the current op finishes.

This avoids interrupting an in-flight consolidation job. It recreates the container
with MiniMax env, keeps bge offline, applies the staged DB pool cap, restores the
locally patched consolidator/tuning, then resumes the drain.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

API = "http://127.0.0.1:8888"
LOG_PATH = Path("/home/wyr/.hermes/logs/hindsight-observations/minimax-idle-switch.log")
AUTH_JSON = Path("/home/wyr/.hermes/auth.json")
DRAIN_SCRIPT = "/home/wyr/.hermes/scripts/hindsight_observations_drain.py"
TMUX_DRAIN_SESSION = "hindsight-obs-drain"
MONITOR_SESSION = "hindsight-obs-monitor"
CURRENT_IMAGE_FALLBACK = "ghcr.io/vectorize-io/hindsight:latest"
MAX_WAIT_SECONDS = 4 * 3600

CONSOLIDATOR_HOST = Path("/tmp/hindsight_src/consolidator.py")
CONFIG_HOST = Path("/tmp/hindsight_src/config.py")
TUNING_HOST = Path("/home/wyr/.hermes/scripts/hindsight-consolidation-tuning-default-20x3.json")
CONSOLIDATOR_CONT = "/app/api/hindsight_api/engine/consolidation/consolidator.py"
CONFIG_CONT = "/app/api/hindsight_api/config.py"
TUNING_CONT = "/home/hindsight/.hindsight-consolidation-tuning.json"


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, check: bool = True, timeout: int = 120, redact: bool = False) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        out = proc.stderr.strip() or proc.stdout.strip()
        if redact:
            out = "[REDACTED]"
        raise RuntimeError(f"cmd failed {cmd[:4]}...: {out}")
    return proc.stdout


def docker_logs_since(since: str = "3h") -> str:
    return run(["docker", "logs", "--since", since, "hindsight"], check=False, timeout=120)


def latest_consolidation_op() -> str | None:
    out = docker_logs_since("3h")
    ops = re.findall(r"op=([0-9a-f-]{36}) type=consolidation", out)
    return ops[-1] if ops else None


def op_completed(op_id: str) -> bool:
    out = docker_logs_since("3h")
    if (
        f"Marked async operation as completed: {op_id}" in out
        or f"Marked async operation as failed: {op_id}" in out
        or f"Task {op_id} failed" in out
    ):
        return True
    # Fallback to PostgreSQL when logs do not contain the exact completion marker.
    try:
        psql = "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
        status = run([
            psql, "-h", "/tmp", "-p", "5432", "-U", "hindsight", "-d", "hindsight",
            "-q", "-t", "-A", "-c",
            f"select status from async_operations where operation_id='{op_id}' limit 1;",
        ], check=False, timeout=20).strip()
        return status in {"completed", "failed"}
    except Exception:
        return False


def stop_processes_containing(needle: str) -> None:
    out = run(["pgrep", "-af", needle], check=False)
    me = os.getpid()
    for line in out.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        if pid == me or "hindsight_minimax_idle_switch.py" in cmd:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"sent SIGTERM pid={pid} cmd={cmd[:140]}")
        except ProcessLookupError:
            pass
        except Exception as exc:
            log(f"WARN failed to terminate pid={pid}: {exc}")


def current_container_env() -> dict[str, str]:
    raw = run(["docker", "inspect", "hindsight", "--format", "{{json .Config.Env}}"], timeout=60)
    arr = json.loads(raw)
    env: dict[str, str] = {}
    for item in arr:
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    return env


def current_image() -> str:
    try:
        img = run(["docker", "inspect", "hindsight", "--format", "{{.Image}}"], timeout=60).strip()
        return img or CURRENT_IMAGE_FALLBACK
    except Exception:
        return CURRENT_IMAGE_FALLBACK


def minimax_key() -> str:
    data = json.loads(AUTH_JSON.read_text(encoding="utf-8"))
    pool = data.get("credential_pool") or {}
    for provider in ("minimax-cn", "minimax"):
        for rec in pool.get(provider, []):
            token = str(rec.get("access_token") or "").strip()
            if token and token not in {"***", "[REDACTED]"}:
                return token
    env_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError("No usable MiniMax API key found in auth.json or env")


def build_minimax_env() -> dict[str, str]:
    env = current_container_env()
    key = minimax_key()
    model = "MiniMax-M2.7"
    base_url = "https://api.minimaxi.com/v1"
    provider = "minimax"

    env.update({
        "HINDSIGHT_API_DB_POOL_MIN_SIZE": "5",
        "HINDSIGHT_API_DB_POOL_MAX_SIZE": "60",
        "HINDSIGHT_API_LLM_PROVIDER": provider,
        "HINDSIGHT_API_LLM_MODEL": model,
        "HINDSIGHT_API_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_LLM_API_KEY": key,
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER": provider,
        "HINDSIGHT_API_RETAIN_LLM_MODEL": model,
        "HINDSIGHT_API_RETAIN_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_RETAIN_LLM_API_KEY": key,
        "HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER": provider,
        "HINDSIGHT_API_CONSOLIDATION_LLM_MODEL": model,
        "HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY": key,
        "HINDSIGHT_API_REFLECT_LLM_PROVIDER": provider,
        "HINDSIGHT_API_REFLECT_LLM_MODEL": model,
        "HINDSIGHT_API_REFLECT_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_REFLECT_LLM_API_KEY": key,
        "HINDSIGHT_API_ENABLE_OBSERVATIONS": "true",
        "HINDSIGHT_API_WORKER_ENABLED": "true",
        "HINDSIGHT_API_WORKER_MAX_SLOTS": "9",
        "HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS": "1",
        "HINDSIGHT_API_LLM_MAX_CONCURRENT": "8",
        "HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT": "8",
        "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT": "8",
        "HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    # Avoid stale global proxy inside Docker unless explicitly set for Hindsight.
    if not os.environ.get("HINDSIGHT_DOCKER_HTTP_PROXY"):
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env.pop(k, None)
    return env


def write_env_file(env: dict[str, str]) -> str:
    fd, path = tempfile.mkstemp(prefix="hindsight-minimax-env-", suffix=".env")
    os.close(fd)
    os.chmod(path, 0o600)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in env.items():
            f.write(f"{k}={v}\n")
    return path


def wait_health(timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            with urlopen(f"{API}/health", timeout=5) as r:
                body = r.read().decode("utf-8", errors="replace")
            if "healthy" in body and "connected" in body:
                log(f"health ok: {body}")
                return
            last = body
        except Exception as exc:
            last = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"health did not recover: {last}")


def recreate_container_minimax() -> None:
    env = build_minimax_env()
    env_file = write_env_file(env)
    image = current_image()
    try:
        log("recreating hindsight container with MiniMax-M2.7 + DB pool cap 60")
        run(["docker", "rm", "-f", "hindsight"], check=False, timeout=120)
        cmd = [
            "docker", "run", "-d",
            "--name", "hindsight",
            "--network", "host",
            "--restart", "unless-stopped",
            "-v", "/home/wyr/.cache/huggingface:/home/hindsight/.cache/huggingface",
            "-v", "/home/wyr/.cache/huggingface:/home/wyr/.cache/huggingface",
            "-v", "/home/wyr/.cache/torch:/home/hindsight/.cache/torch",
            "-v", "/home/wyr/.cache/torch:/home/wyr/.cache/torch",
            "--env-file", env_file,
            image,
        ]
        out = run(cmd, timeout=180, redact=True).strip()
        log(f"container started: {out[:12]}")
    finally:
        try:
            os.remove(env_file)
        except FileNotFoundError:
            pass


def apply_hot_patches_and_restart() -> None:
    # Copy host-side patched files into the new container, then restart once so Python imports them.
    for src, dst in [(CONSOLIDATOR_HOST, CONSOLIDATOR_CONT), (CONFIG_HOST, CONFIG_CONT), (TUNING_HOST, TUNING_CONT)]:
        if not src.exists():
            log(f"WARN patch source missing: {src}")
            continue
        run(["docker", "cp", str(src), f"hindsight:{dst}"], timeout=120)
        log(f"copied patch {src} -> {dst}")
    run(["docker", "exec", "hindsight", "python3", "-m", "py_compile", CONSOLIDATOR_CONT, CONFIG_CONT], timeout=120)
    log("restarting container to load copied patches")
    run(["docker", "restart", "-t", "30", "hindsight"], timeout=180)
    wait_health()


def start_drain_tmux() -> None:
    run(["tmux", "kill-session", "-t", TMUX_DRAIN_SESSION], check=False)
    run(["tmux", "new-session", "-d", "-s", TMUX_DRAIN_SESSION, "python3", DRAIN_SCRIPT])
    log(f"started drain tmux session {TMUX_DRAIN_SESSION}")


def refresh_monitor() -> None:
    # Keep current monitor code; restart pane so it sees new liveness logs cleanly.
    run(["tmux", "kill-session", "-t", MONITOR_SESSION], check=False)
    run(["tmux", "new-session", "-d", "-s", MONITOR_SESSION, "-x", "170", "-y", "45", "python3", "/home/wyr/.hermes/scripts/hindsight_observations_monitor.py"], check=False)
    log(f"refreshed monitor tmux session {MONITOR_SESSION}")


def main() -> int:
    op_id = latest_consolidation_op()
    log(f"current_op={op_id or 'none'}; MiniMax idle-switch watcher started")
    stop_processes_containing("hindsight_observations_drain.py")
    stop_processes_containing("hindsight_pool_cap_idle_restart.py")
    stop_processes_containing("hindsight_drain_completion_watch.py")

    if op_id:
        start = time.time()
        while not op_completed(op_id):
            if time.time() - start > MAX_WAIT_SECONDS:
                log(f"TIMEOUT waiting for op completion: {op_id}")
                return 3
            log(f"waiting current consolidation op to finish before MiniMax switch: {op_id}")
            time.sleep(30)
        log(f"current consolidation op finished: {op_id}")
    else:
        log("no active consolidation op found in logs; switching immediately")

    recreate_container_minimax()
    wait_health()
    apply_hot_patches_and_restart()
    refresh_monitor()
    start_drain_tmux()
    log("DONE MiniMax switch completed; drain resumed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
