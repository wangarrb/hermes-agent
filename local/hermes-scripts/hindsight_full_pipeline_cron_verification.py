#!/usr/bin/env python3
"""One-shot cron verification for Hindsight downstream full pipeline.

Safe defaults:
- never edits .env
- never deletes data
- only executes the downstream full flow when Hindsight is idle
- skips daily/session retain via --skip-daily to avoid cron/self-session ingestion
- prints a human-readable report and exits 0 so the cron delivery contains the report
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
SCRIPTS = HERMES_HOME / "scripts"
API = os.environ.get("HINDSIGHT_BASE_URL", "http://127.0.0.1:8888")
BANK = os.environ.get("HINDSIGHT_BANK", "hermes")
DRY_RUN_ONLY = os.environ.get("HINDSIGHT_FULL_VERIFY_DRY_RUN_ONLY", "").lower() in {"1", "true", "yes"}
EXEC_TIMEOUT = int(os.environ.get("HINDSIGHT_FULL_VERIFY_EXEC_TIMEOUT_S", "14400"))
CMD_TIMEOUT = int(os.environ.get("HINDSIGHT_FULL_VERIFY_CMD_TIMEOUT_S", "300"))
IDLE_WAIT_S = int(os.environ.get("HINDSIGHT_FULL_VERIFY_IDLE_WAIT_S", "7200"))
IDLE_POLL_S = int(os.environ.get("HINDSIGHT_FULL_VERIFY_IDLE_POLL_S", "60"))


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run(cmd: list[str], *, timeout: int = CMD_TIMEOUT, cwd: Path | None = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
    )
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "seconds": round(time.time() - started, 2),
        "output": proc.stdout,
    }


def http_json(path: str, timeout: int = 20) -> Any:
    with urllib.request.urlopen(API.rstrip("/") + path, timeout=timeout) as r:
        return json.load(r)


def parse_json_output(result: dict[str, Any]) -> Any:
    return json.loads(result.get("output") or "{}")


def tail(text: str, lines: int = 80) -> str:
    xs = (text or "").splitlines()
    return "\n".join(xs[-lines:])


def operation_total(status: str) -> int:
    data = http_json(f"/v1/default/banks/{BANK}/operations?status={status}&limit=1&offset=0&exclude_parents=true")
    for key in ("total", "total_operations", "count"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    value = data.get("operations") or data.get("items") or data.get("results") or []
    if isinstance(value, list):
        return len(value)
    return 0


def queue_counts() -> tuple[int, int]:
    return operation_total("pending"), operation_total("processing")


def wait_for_idle(lines: list[str], warnings: list[str]) -> tuple[int, int]:
    deadline = time.time() + IDLE_WAIT_S
    pending, processing = queue_counts()
    if pending == 0 and processing == 0:
        return pending, processing
    lines.append(
        f"idle_wait: initial pending={pending} processing={processing}; "
        f"waiting up to {IDLE_WAIT_S}s poll={IDLE_POLL_S}s"
    )
    while pending or processing:
        if time.time() >= deadline:
            warnings.append(f"idle wait timed out: pending={pending}, processing={processing}")
            return pending, processing
        time.sleep(IDLE_POLL_S)
        pending, processing = queue_counts()
        lines.append(f"idle_wait_poll: {now()} pending={pending} processing={processing}")
    return pending, processing


def plan(mode: str, *extra: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    cmd = [sys.executable, str(SCRIPTS / "hindsight_memory_pipeline.py"), mode, *extra, "--plan-json"]
    res = run(cmd)
    if res["rc"] != 0:
        return None, res
    try:
        return parse_json_output(res), res
    except Exception:
        return None, res


def main() -> int:
    lines: list[str] = []
    failures: list[str] = []
    warnings: list[str] = []
    artifacts: list[str] = []

    lines.append("Hindsight full pipeline cron verification")
    lines.append(f"started_at: {now()}")
    lines.append(f"bank: {BANK}")
    lines.append(f"dry_run_only: {DRY_RUN_ONLY}")
    lines.append("")

    try:
        health = http_json("/health")
        lines.append(f"health: {health}")
        if health.get("status") != "healthy" or health.get("database") != "connected":
            failures.append("Hindsight health is not healthy+connected")
    except Exception as e:
        failures.append(f"health check failed: {type(e).__name__}: {e}")
        print("\n".join(lines + ["", "FAILURES:", *failures]))
        return 0

    preflight = run([sys.executable, str(SCRIPTS / "hindsight_pipeline_preflight.py"), "--strict-runtime", "--json"])
    if preflight["rc"] != 0:
        failures.append(f"strict preflight rc={preflight['rc']}")
        artifacts.append("strict preflight tail:\n" + tail(preflight["output"], 60))
    else:
        pf = parse_json_output(preflight)
        lines.append(
            f"strict_preflight: ok={pf.get('ok')} blocking={pf.get('blocking_count')} warnings={pf.get('warning_count')}"
        )
        if not pf.get("ok") or pf.get("blocking_count"):
            failures.append("strict preflight has blocking failures")

    status = run([sys.executable, str(SCRIPTS / "hindsight_consolidation_status.py"), "--skip-psql", "--json"])
    if status["rc"] != 0:
        warnings.append(f"status helper rc={status['rc']}; tail follows")
        artifacts.append("status helper tail:\n" + tail(status["output"], 80))
    else:
        try:
            st = parse_json_output(status)
            summary = st.get("summary") or st
            lines.append(
                "status: "
                + json.dumps(
                    {
                        "active_count": summary.get("active_count"),
                        "pending_count": summary.get("pending_count"),
                        "failed_count": summary.get("failed_count"),
                        "has_active_work": summary.get("has_active_work"),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            warnings.append("could not parse status helper JSON")

    try:
        pending, processing = queue_counts()
        lines.append(f"operations_api_exclude_parents: pending={pending} processing={processing}")
    except Exception as e:
        pending = processing = -1
        warnings.append(f"operations API count failed: {type(e).__name__}: {e}")

    daily_plan, daily_plan_res = plan("daily")
    if not daily_plan:
        failures.append(f"daily plan failed rc={daily_plan_res['rc']}")
        artifacts.append("daily plan tail:\n" + tail(daily_plan_res["output"], 80))
    else:
        daily_steps = [s.get("name") for s in daily_plan.get("steps", [])]
        retain_steps = [s for s in daily_plan.get("steps", []) if s.get("name") == "retain_session_manifest"]
        has_obs = bool(retain_steps and "--enable-observations" in retain_steps[0].get("command", []))
        lines.append(f"daily_plan_steps: {daily_steps}")
        lines.append(f"daily_retain_has_enable_observations: {has_obs}")
        if not has_obs:
            failures.append("daily retain step lost --enable-observations")

    full_plan, full_plan_res = plan("full", "--skip-daily")
    expected_downstream = [
        "preflight",
        "runtime_status",
        "weekly_reflect",
        "v2_rebuild_gate",
        "conflict_audit",
        "repair_zone_proposals",
        "proposal_review",
    ]
    if not full_plan:
        failures.append(f"full --skip-daily plan failed rc={full_plan_res['rc']}")
        artifacts.append("full plan tail:\n" + tail(full_plan_res["output"], 80))
    else:
        full_steps = [s.get("name") for s in full_plan.get("steps", [])]
        lines.append(f"full_skip_daily_plan_steps: {full_steps}")
        if full_steps != expected_downstream:
            failures.append(f"unexpected full --skip-daily steps: {full_steps}")

    execute_skipped = False
    execute_result: dict[str, Any] | None = None
    if DRY_RUN_ONLY:
        execute_skipped = True
        warnings.append("execute skipped because HINDSIGHT_FULL_VERIFY_DRY_RUN_ONLY=1")
    elif failures:
        execute_skipped = True
        warnings.append("execute skipped because pre-execute checks failed")
    elif pending < 0 or processing < 0:
        execute_skipped = True
        warnings.append("execute skipped because queue status is unknown")
    else:
        pending, processing = wait_for_idle(lines, warnings)
        if pending != 0 or processing != 0:
            execute_skipped = True
            warnings.append(f"execute skipped because queue did not become idle: pending={pending}, processing={processing}")
        else:
            cmd = [
                sys.executable,
                str(SCRIPTS / "hindsight_memory_pipeline.py"),
                "full",
                "--skip-daily",
                "--execute",
                "--confirm",
                "run-hindsight-pipeline",
                "--execute-proposal-review-llm",
                "--confirm-proposal-review",
                "review-hindsight-proposals",
                "--notify-proposal-review",
                "--timeout",
                os.environ.get("HINDSIGHT_FULL_VERIFY_PIPELINE_TIMEOUT_ARG", "7200"),
            ]
            lines.append("execute_command: " + " ".join(cmd))
            execute_result = run(cmd, timeout=EXEC_TIMEOUT)
            lines.append(f"execute_rc: {execute_result['rc']} seconds={execute_result['seconds']}")
            artifacts.append("execute tail:\n" + tail(execute_result["output"], 120))
            if execute_result["rc"] != 0:
                failures.append(f"full --skip-daily execute failed rc={execute_result['rc']}")

    try:
        health2 = http_json("/health")
        lines.append(f"post_health: {health2}")
    except Exception as e:
        warnings.append(f"post health failed: {type(e).__name__}: {e}")

    try:
        pending2 = operation_total("pending")
        processing2 = operation_total("processing")
        lines.append(f"post_operations_api_exclude_parents: pending={pending2} processing={processing2}")
    except Exception as e:
        warnings.append(f"post operations API count failed: {type(e).__name__}: {e}")

    lines.append("")
    if failures:
        lines.append("RESULT: FAIL")
        lines.append("FAILURES:")
        lines.extend(f"- {x}" for x in failures)
    elif execute_skipped:
        lines.append("RESULT: CHECKS_PASS_EXECUTE_SKIPPED")
    else:
        lines.append("RESULT: PASS")

    if warnings:
        lines.append("")
        lines.append("WARNINGS:")
        lines.extend(f"- {x}" for x in warnings)

    if artifacts:
        lines.append("")
        lines.append("ARTIFACT TAILS:")
        lines.extend(artifacts)

    lines.append("")
    lines.append(f"finished_at: {now()}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("Hindsight full pipeline cron verification")
        print(f"started_or_failed_at: {now()}")
        print("RESULT: SCRIPT_EXCEPTION")
        print(traceback.format_exc())
        raise SystemExit(0)
