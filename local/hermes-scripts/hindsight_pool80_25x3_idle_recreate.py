#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib import request as urlrequest

PSQL = "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
TUNING_HOST = Path("/home/wyr/.hermes/scripts/hindsight-consolidation-tuning-25x3-pool80.json")
TUNING_CONT = "/home/hindsight/.hindsight-consolidation-tuning.json"
CONSOLIDATOR_HOST = Path("/tmp/hindsight_src/consolidator.py")
CONSOLIDATOR_CONT = "/app/api/hindsight_api/engine/consolidation/consolidator.py"
CONFIG_HOST = Path("/tmp/hindsight_src/config.py")
CONFIG_CONT = "/app/api/hindsight_api/config.py"
LOG = Path("/home/wyr/.hermes/logs/hindsight-observations/pool80-25x3-idle-recreate.log")
LOG.parent.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, timeout: int = 120, check: bool = True, redact: bool = False) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        out = proc.stderr.strip() or proc.stdout.strip()
        if redact:
            out = "[REDACTED]"
        raise RuntimeError(f"cmd failed rc={proc.returncode}: {' '.join(cmd[:6])}\n{out}")
    return proc.stdout


def psql(sql: str) -> str:
    return run([PSQL, "-h", "/tmp", "-p", "5432", "-U", "hindsight", "-d", "hindsight", "-q", "-t", "-A", "-F", "\t", "-c", sql], timeout=30)


def counts() -> dict[str, int]:
    out = psql("select status,count(*) from async_operations where bank_id='hermes' and operation_type='consolidation' group by status;").strip()
    d: dict[str, int] = {}
    for line in out.splitlines():
        if line.strip():
            k, v = line.split("\t")
            d[k] = int(v)
    return d


def backlog() -> str:
    return psql("""
select count(*) filter(where fact_type='observation'),
       count(*) filter(where fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null),
       count(*) filter(where fact_type in ('world','experience') and consolidation_failed_at is not null)
from memory_units where bank_id='hermes';
""").strip()


def pg_max_connections() -> int:
    return int(run([PSQL, "-h", "/tmp", "-p", "5432", "-U", "hindsight", "-d", "postgres", "-q", "-t", "-A", "-c", "show max_connections;"], timeout=20).strip())


def current_env() -> dict[str, str]:
    raw = run(["docker", "inspect", "hindsight", "--format", "{{json .Config.Env}}"], timeout=60)
    arr = json.loads(raw)
    env: dict[str, str] = {}
    for item in arr:
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    return env


def current_image() -> str:
    return run(["docker", "inspect", "hindsight", "--format", "{{.Image}}"], timeout=60).strip()


def write_env_file(env: dict[str, str]) -> str:
    fd, path = tempfile.mkstemp(prefix="hindsight-pool80-env-", suffix=".env")
    os.close(fd)
    os.chmod(path, 0o600)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in env.items():
            f.write(f"{k}={v}\n")
    return path


def health_ok(timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlrequest.urlopen("http://127.0.0.1:8888/health", timeout=5) as r:
                body = r.read().decode("utf-8", "replace")
            if "healthy" in body:
                log(f"health ok: {body}")
                return True
        except Exception as e:
            log(f"waiting health: {e}")
        time.sleep(3)
    return False


def recreate_pool80() -> None:
    env = current_env()
    env["HINDSIGHT_API_DB_POOL_MIN_SIZE"] = "5"
    env["HINDSIGHT_API_DB_POOL_MAX_SIZE"] = "80"
    # Keep LLM and all keys exactly as current container; do not read or modify .env.
    env_file = write_env_file(env)
    image = current_image()
    try:
        log("recreating container with HINDSIGHT_API_DB_POOL_MAX_SIZE=80")
        run(["docker", "rm", "-f", "hindsight"], timeout=120, check=False)
        cmd = [
            "docker", "run", "-d", "--name", "hindsight", "--network", "host", "--restart", "unless-stopped",
            "-v", "/home/wyr/.cache/huggingface:/home/hindsight/.cache/huggingface",
            "-v", "/home/wyr/.cache/huggingface:/home/wyr/.cache/huggingface",
            "-v", "/home/wyr/.cache/torch:/home/hindsight/.cache/torch",
            "-v", "/home/wyr/.cache/torch:/home/wyr/.cache/torch",
            "--env-file", env_file, image,
        ]
        out = run(cmd, timeout=180, redact=True).strip()
        log(f"container started: {out[:12]}")
    finally:
        try:
            os.remove(env_file)
        except FileNotFoundError:
            pass


def copy_patches_and_restart() -> None:
    for src, dst in [(CONSOLIDATOR_HOST, CONSOLIDATOR_CONT), (CONFIG_HOST, CONFIG_CONT), (TUNING_HOST, TUNING_CONT)]:
        if src.exists():
            run(["docker", "cp", str(src), f"hindsight:{dst}"], timeout=120)
            log(f"copied {src} -> {dst}")
        else:
            log(f"WARN source missing: {src}")
    run(["docker", "exec", "hindsight", "python3", "-m", "py_compile", CONSOLIDATOR_CONT, CONFIG_CONT], timeout=120)
    run(["docker", "restart", "-t", "30", "hindsight"], timeout=180)
    if not health_ok():
        raise RuntimeError("health failed after patch restart")


def start_drain() -> None:
    run(["tmux", "kill-session", "-t", "hindsight-obs-drain"], check=False)
    run(["tmux", "new-session", "-d", "-s", "hindsight-obs-drain", "-x", "170", "-y", "45", "python3", "/home/wyr/.hermes/scripts/hindsight_observations_drain.py"])
    log("restarted hindsight-obs-drain")


def main() -> int:
    max_conn = pg_max_connections()
    log(f"START pool80+25x3 idle recreate; postgres max_connections={max_conn}")
    if max_conn < 100:
        log(f"ERROR refusing pool80 because max_connections={max_conn} < 100")
        return 2
    run(["tmux", "kill-session", "-t", "hindsight-obs-drain"], check=False)
    log("stopped drain trigger while waiting for active processing to finish")
    last = 0.0
    while True:
        c = counts()
        if time.time() - last > 60 or c.get("processing", 0) == 0:
            log(f"counts={json.dumps(c, sort_keys=True)} backlog={backlog()}")
            last = time.time()
        if c.get("processing", 0) == 0:
            break
        time.sleep(10)
    recreate_pool80()
    if not health_ok():
        log("ERROR health failed after recreate")
        return 3
    copy_patches_and_restart()
    start_drain()
    log("DONE pool80 + 25x3 applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
