#!/usr/bin/env python3
"""Guard rails and smoke tests for Hindsight native reflect/consolidation workflows.

This script is intentionally conservative:
- read-only status/preflight are default paths;
- cleanup requires an explicit confirm token;
- native consolidation smoke runs on a temporary bank and restores normal-local mode in finally.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

API = os.environ.get("HINDSIGHT_API", "http://127.0.0.1:8888")
BANK = os.environ.get("HINDSIGHT_BANK", "hermes")
PSQL = os.environ.get("HINDSIGHT_PSQL", "/home/wyr/.pg0/installation/18.1.0/bin/psql")
DB_ARGS = ["-h", os.environ.get("HINDSIGHT_PGHOST", "/tmp"), "-p", os.environ.get("HINDSIGHT_PGPORT", "5432"), "-U", os.environ.get("HINDSIGHT_PGUSER", "hindsight"), "-d", os.environ.get("HINDSIGHT_PGDATABASE", "hindsight")]
JSON_PARSER_PATCH_SCRIPT = SCRIPT_DIR / "patch_hindsight_minimax_json_parser.py"
CONSOLIDATION_BUDGET_PATCH_SCRIPT = SCRIPT_DIR / "patch_hindsight_native_consolidation_budget.py"
PATCH_SCRIPTS = [
    ("JSON parser / 429 backoff", JSON_PARSER_PATCH_SCRIPT),
    ("native consolidation per-job budget", CONSOLIDATION_BUDGET_PATCH_SCRIPT),
]
CONFIRM_CLEANUP = "cleanup-hindsight-payload-null"
CONFIRM_RUN_NATIVE_PAID = "run-native-paid-consolidation"
SMOKE_BANK_PREFIX = "hermes-native-smoke-"


class GuardError(RuntimeError):
    pass


def run(cmd: list[str], *, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise GuardError(f"command failed exit={proc.returncode}: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


def psql_at(sql: str, *, timeout: int = 60) -> list[list[str]]:
    cmd = [PSQL, *DB_ARGS, "-X", "-v", "ON_ERROR_STOP=1", "-At", "-F", "\t", "-c", sql]
    proc = run(cmd, timeout=timeout)
    rows: list[list[str]] = []
    for line in proc.stdout.splitlines():
        if line.strip():
            rows.append(line.split("\t"))
    return rows


def psql_scalar(sql: str, *, timeout: int = 60) -> str:
    rows = psql_at(sql, timeout=timeout)
    if not rows:
        return ""
    return rows[0][0]


def psql_exec(sql: str, *, timeout: int = 60) -> str:
    cmd = [PSQL, *DB_ARGS, "-X", "-v", "ON_ERROR_STOP=1", "-c", sql]
    return run(cmd, timeout=timeout).stdout


def api_json(method: str, path: str, *, payload: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    url = f"{API}{path}"
    r = requests.request(method, url, json=payload, timeout=timeout)
    r.raise_for_status()
    if not r.content:
        return None
    return r.json()


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_native_consolidation_estimate_sql(
    bank: str,
    *,
    fetch_size: int,
    llm_batch_size: int,
    max_facts: int | None = None,
) -> str:
    fetch_size = max(1, int(fetch_size))
    llm_batch_size = max(1, int(llm_batch_size))
    limit_clause = f"WHERE rn <= {max(1, int(max_facts))}" if max_facts is not None and int(max_facts) > 0 else ""
    return f"""
WITH ordered AS (
    SELECT
        row_number() OVER (ORDER BY created_at ASC) AS rn,
        COALESCE(
            array_to_string(
                ARRAY(SELECT unnest(COALESCE(tags, '{{}}'::varchar[])) ORDER BY 1),
                ','
            ),
            '<null>'
        ) AS tag_key
    FROM memory_units
    WHERE bank_id = {sql_literal(bank)}
      AND consolidated_at IS NULL
      AND consolidation_failed_at IS NULL
      AND fact_type IN ('experience', 'world')
),
limited AS (
    SELECT * FROM ordered
    {limit_clause}
),
groups AS (
    SELECT
        ((rn - 1) / {fetch_size})::int AS fetch_round,
        tag_key,
        count(*) AS n
    FROM limited
    GROUP BY 1, 2
)
SELECT
    COALESCE((SELECT count(*) FROM limited), 0) AS facts,
    COALESCE((SELECT count(DISTINCT fetch_round) FROM groups), 0) AS fetch_rounds,
    COALESCE((SELECT count(*) FROM groups), 0) AS tag_groups,
    COALESCE((SELECT sum(ceil(n::numeric / {llm_batch_size})::int) FROM groups), 0) AS llm_calls;
"""


def estimate_native_consolidation_calls(
    bank: str,
    *,
    fetch_size: int,
    llm_batch_size: int,
    max_facts: int | None = None,
) -> dict[str, int]:
    rows = psql_at(build_native_consolidation_estimate_sql(bank, fetch_size=fetch_size, llm_batch_size=llm_batch_size, max_facts=max_facts))
    if not rows:
        return {"facts": 0, "fetch_rounds": 0, "tag_groups": 0, "llm_calls": 0}
    row = rows[0]
    return {
        "facts": int(row[0] or 0),
        "fetch_rounds": int(row[1] or 0),
        "tag_groups": int(row[2] or 0),
        "llm_calls": int(row[3] or 0),
    }


def wait_operation(operation_id: str, *, poll_s: int = 10, timeout_s: int = 3600) -> dict[str, Any]:
    started = time.time()
    last: dict[str, Any] | None = None
    while True:
        rows = psql_at(
            f"""
            SELECT status,
                   COALESCE(error_message, ''),
                   COALESCE(result_metadata::text, ''),
                   COALESCE(operation_type, ''),
                   COALESCE(claimed_at::text, ''),
                   COALESCE(updated_at::text, ''),
                   COALESCE(completed_at::text, '')
            FROM async_operations
            WHERE operation_id = {sql_literal(operation_id)};
            """
        )
        if not rows:
            raise GuardError(f"operation disappeared: {operation_id}")
        row = rows[0]
        status = row[0]
        last = {
            "operation_id": operation_id,
            "status": status,
            "error_message": row[1],
            "result_metadata": row[2],
            "operation_type": row[3],
            "claimed_at": row[4],
            "updated_at": row[5],
            "completed_at": row[6],
        }
        print(json.dumps({"operation": last, "elapsed_s": round(time.time() - started, 1)}, ensure_ascii=False), flush=True)
        if status not in ("pending", "processing"):
            return last
        if timeout_s and time.time() - started > timeout_s:
            raise TimeoutError(f"operation {operation_id} did not finish within {timeout_s}s")
        time.sleep(poll_s)


def docker_env() -> dict[str, str]:
    keys = [
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_MODEL",
        "HINDSIGHT_API_LLM_BASE_URL",
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER",
        "HINDSIGHT_API_RETAIN_LLM_MODEL",
        "HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER",
        "HINDSIGHT_API_CONSOLIDATION_LLM_MODEL",
        "HINDSIGHT_API_REFLECT_LLM_PROVIDER",
        "HINDSIGHT_API_REFLECT_LLM_MODEL",
        "HINDSIGHT_API_ENABLE_OBSERVATIONS",
        "HINDSIGHT_API_WORKER_MAX_SLOTS",
        "HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS",
        "HINDSIGHT_API_LLM_MAX_RETRIES",
        "HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES",
        "HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES",
        "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES",
        "HINDSIGHT_API_CONSOLIDATION_MAX_ATTEMPTS",
        "HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB",
        "HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE",
        "HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE",
        "HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS",
        "HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS",
    ]
    script = "for k in " + " ".join(keys) + "; do v=$(docker exec hindsight printenv $k 2>/dev/null || true); [ -n \"$v\" ] && printf '%s=%s\\n' $k $v; done"
    proc = subprocess.run(["bash", "-lc", f"newgrp docker <<'SH'\n{script}\nSH"], text=True, capture_output=True, timeout=30)
    env: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def collect_status(bank: str = BANK) -> dict[str, Any]:
    health: Any
    stats: Any
    config: Any
    try:
        health = api_json("GET", "/health", timeout=10)
    except Exception as e:
        health = {"error": repr(e)}
    try:
        stats = api_json("GET", f"/v1/default/banks/{bank}/stats", timeout=20)
    except Exception as e:
        stats = {"error": repr(e)}
    try:
        config = api_json("GET", f"/v1/default/banks/{bank}/config", timeout=20)
    except Exception as e:
        config = {"error": repr(e)}

    op_rows = psql_at(
        f"""
        SELECT operation_type,status,count(*) AS n,
               count(*) FILTER (WHERE task_payload IS NULL) AS payload_null
        FROM async_operations
        WHERE bank_id = '{bank}'
        GROUP BY operation_type,status
        ORDER BY operation_type,status;
        """
    )
    operations = [
        {"operation_type": r[0], "status": r[1], "count": int(r[2]), "payload_null": int(r[3])}
        for r in op_rows
    ]
    mem_rows = psql_at(
        f"""
        SELECT fact_type, count(*) AS n,
               count(*) FILTER (WHERE consolidated_at IS NULL) AS unconsolidated,
               count(*) FILTER (WHERE consolidation_failed_at IS NOT NULL) AS consolidation_failed
        FROM memory_units
        WHERE bank_id = '{bank}'
        GROUP BY fact_type
        ORDER BY fact_type;
        """
    )
    memory_units = [
        {"fact_type": r[0], "count": int(r[1]), "unconsolidated": int(r[2]), "consolidation_failed": int(r[3])}
        for r in mem_rows
    ]
    active_payload_null = int(psql_scalar(
        f"""
        SELECT count(*) FROM async_operations
        WHERE bank_id='{bank}' AND status IN ('pending','processing') AND task_payload IS NULL;
        """
    ) or 0)
    unconsolidated_candidates = int(psql_scalar(
        f"""
        SELECT count(*) FROM memory_units
        WHERE bank_id='{bank}'
          AND fact_type IN ('experience','world')
          AND consolidated_at IS NULL
          AND consolidation_failed_at IS NULL;
        """
    ) or 0)
    return {
        "api": API,
        "bank": bank,
        "health": health,
        "stats": stats,
        "config_focus": focus_config(config),
        "provider_env": docker_env(),
        "operations": operations,
        "memory_units": memory_units,
        "active_payload_null": active_payload_null,
        "native_consolidation_unconsolidated_candidates": unconsolidated_candidates,
    }


def focus_config(config_response: Any) -> Any:
    if not isinstance(config_response, dict):
        return config_response
    cfg = config_response.get("config") or config_response
    keys = [
        "enable_observations",
        "retain_chunk_size",
        "retain_extraction_mode",
        "llm_provider",
        "llm_model",
        "retain_llm_provider",
        "consolidation_llm_provider",
        "consolidation_llm_model",
        "consolidation_max_memories_per_round",
        "consolidation_llm_batch_size",
        "consolidation_source_facts_max_tokens",
        "consolidation_max_attempts",
    ]
    return {k: cfg.get(k) for k in keys if k in cfg}


def cmd_status(args: argparse.Namespace) -> int:
    report = collect_status(args.bank)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def evaluate_preflight_report(
    report: dict[str, Any],
    *,
    allow_queue: bool = False,
    allow_failed: bool = False,
    expect_provider: str | None = None,
    require_observations_disabled: bool = False,
    max_unconsolidated: int | None = None,
) -> list[str]:
    stats = report.get("stats") if isinstance(report.get("stats"), dict) else {}
    env = report.get("provider_env") if isinstance(report.get("provider_env"), dict) else {}
    errors: list[str] = []
    health = report.get("health") if isinstance(report.get("health"), dict) else {}
    if health.get("status") != "healthy":
        errors.append(f"health is not healthy: {health}")
    pending = int(stats.get("pending_operations") or 0)
    processing = int(stats.get("processing_operations") or 0)
    failed = int(stats.get("failed_operations") or 0)
    if not allow_queue and (pending or processing):
        errors.append(f"queue not empty: pending={pending}, processing={processing}")
    if report.get("active_payload_null"):
        errors.append(f"active payload_null operations={report['active_payload_null']}")
    if failed and not allow_failed:
        errors.append(f"failed operations={failed}")
    provider = env.get("HINDSIGHT_API_LLM_PROVIDER")
    if expect_provider and provider != expect_provider:
        errors.append(f"provider mismatch: expected {expect_provider}, got {provider}")
    if require_observations_disabled:
        cfg_enable = (report.get("config_focus") or {}).get("enable_observations")
        env_enable = env.get("HINDSIGHT_API_ENABLE_OBSERVATIONS")
        if cfg_enable is not False:
            errors.append(f"bank config enable_observations is not false: {cfg_enable}")
        if env_enable not in (None, "false", "False", "0"):
            errors.append(f"env HINDSIGHT_API_ENABLE_OBSERVATIONS is not false: {env_enable}")
    candidates = int(report.get("native_consolidation_unconsolidated_candidates") or 0)
    if max_unconsolidated is not None and candidates > max_unconsolidated:
        errors.append(f"native consolidation candidate count {candidates} exceeds max {max_unconsolidated}")
    return errors


def cmd_preflight(args: argparse.Namespace) -> int:
    report = collect_status(args.bank)
    errors = evaluate_preflight_report(
        report,
        allow_queue=args.allow_queue,
        allow_failed=args.allow_failed,
        expect_provider=args.expect_provider,
        require_observations_disabled=args.require_observations_disabled,
        max_unconsolidated=args.max_unconsolidated,
    )
    out = {"ok": not errors, "errors": errors, "report": report}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def cmd_cleanup_payload_null(args: argparse.Namespace) -> int:
    rows = psql_at(
        f"""
        SELECT operation_id::text, operation_type, status, created_at::text, COALESCE(claimed_at::text,'')
        FROM async_operations
        WHERE bank_id='{args.bank}' AND status IN ('pending','processing') AND task_payload IS NULL
        ORDER BY created_at
        LIMIT 50;
        """
    )
    report = {
        "bank": args.bank,
        "matched_active_payload_null": len(rows),
        "sample": [
            {"operation_id": r[0], "operation_type": r[1], "status": r[2], "created_at": r[3], "claimed_at": r[4]}
            for r in rows
        ],
        "execute": args.execute,
        "required_confirm": CONFIRM_CLEANUP,
    }
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.confirm != CONFIRM_CLEANUP:
        report["error"] = "confirmation token mismatch; refusing to modify DB"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    sql = f"""
    UPDATE async_operations
    SET status='failed',
        error_message='invalid task_payload: null; marked failed by hindsight_native_workflow_guard',
        result_metadata = result_metadata || '{{"guard_cleanup":"payload_null","cleanup_ts":"{int(time.time())}"}}'::jsonb,
        completed_at=NOW(), updated_at=NOW(), worker_id=NULL, claimed_at=NULL
    WHERE bank_id='{args.bank}' AND status IN ('pending','processing') AND task_payload IS NULL;
    """
    out = psql_exec(sql)
    report["psql_output"] = out.strip()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_apply_patch(args: argparse.Namespace) -> int:
    failures: list[dict[str, Any]] = []
    for label, script in PATCH_SCRIPTS:
        if not script.exists():
            failures.append({"label": label, "script": str(script), "error": "missing patch script"})
            continue
        proc = run([sys.executable, str(script)], timeout=180, check=False)
        if proc.stdout.strip():
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        if proc.returncode != 0:
            failures.append({"label": label, "script": str(script), "returncode": proc.returncode})
    if failures:
        print(json.dumps({"ok": False, "failures": failures}, ensure_ascii=False, indent=2))
        return 2
    if args.restart:
        proc = subprocess.run(
            ["bash", "-lc", "newgrp docker <<'SH'\ndocker restart hindsight\nSH"],
            text=True,
            capture_output=True,
            timeout=180,
        )
        if proc.stdout.strip():
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        if proc.returncode != 0:
            return proc.returncode
    print(json.dumps({"ok": True, "patches": [label for label, _ in PATCH_SCRIPTS], "restart": args.restart}, ensure_ascii=False))
    return 0


def cmd_restore_local(args: argparse.Namespace) -> int:
    mgr = __import__("hindsight_minimax_import")
    mgr.switch_mode("normal-local", allow_existing_queue=True)
    print(json.dumps(collect_status(args.bank), ensure_ascii=False, indent=2))
    return 0


def cmd_run_native_consolidation_paid(args: argparse.Namespace) -> int:
    fetch_size = max(1, int(args.fetch_size or args.facts_per_job))
    llm_batch_size = max(1, int(args.llm_batch_size or args.facts_per_job))
    facts_per_job = max(1, int(args.facts_per_job))
    jobs = max(1, int(args.jobs))
    max_window_facts = jobs * facts_per_job

    report = collect_status(args.bank)
    errors = evaluate_preflight_report(
        report,
        allow_queue=args.allow_existing_queue,
        allow_failed=args.allow_failed,
        expect_provider="ollama" if args.expect_local_provider else None,
        require_observations_disabled=True,
        max_unconsolidated=args.max_unconsolidated,
    )
    if int(report.get("native_consolidation_unconsolidated_candidates") or 0) <= 0:
        errors.append("no native consolidation candidates")

    full_estimate = estimate_native_consolidation_calls(
        args.bank,
        fetch_size=fetch_size,
        llm_batch_size=llm_batch_size,
        max_facts=None,
    )
    window_estimate = estimate_native_consolidation_calls(
        args.bank,
        fetch_size=fetch_size,
        llm_batch_size=llm_batch_size,
        max_facts=max_window_facts,
    )
    summary: dict[str, Any] = {
        "cmd": "run-native-consolidation-paid",
        "execute": args.execute,
        "required_confirm": CONFIRM_RUN_NATIVE_PAID,
        "bank": args.bank,
        "llm_profile": args.llm_profile,
        "jobs": jobs,
        "facts_per_job": facts_per_job,
        "fetch_size": fetch_size,
        "llm_batch_size": llm_batch_size,
        "source_facts_max_tokens": args.source_facts_max_tokens,
        "preflight_ok": not errors,
        "preflight_errors": errors,
        "estimate_window": window_estimate,
        "estimate_full_backlog": full_estimate,
        "status_before_focus": {
            "health": report.get("health"),
            "config_focus": report.get("config_focus"),
            "provider_env": report.get("provider_env"),
            "active_payload_null": report.get("active_payload_null"),
            "native_candidates": report.get("native_consolidation_unconsolidated_candidates"),
        },
    }

    if not args.execute:
        summary["ok"] = not errors
        summary["next_step"] = (
            f"Add --execute --confirm {CONFIRM_RUN_NATIVE_PAID} to switch to paid provider and POST /consolidate."
            if not errors
            else "Fix preflight errors before paid/native consolidation."
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not errors else 2

    if args.confirm != CONFIRM_RUN_NATIVE_PAID:
        summary["ok"] = False
        summary["error"] = "confirmation token mismatch; refusing paid provider switch"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    if errors:
        summary["ok"] = False
        summary["error"] = "preflight failed; refusing paid provider switch"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    mgr = __import__("hindsight_minimax_import")
    profile = mgr.get_llm_profile(args.llm_profile)
    env_overrides = {
        "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_JOB": str(facts_per_job),
        "HINDSIGHT_NATIVE_CONSOLIDATION_BATCH_SIZE": str(fetch_size),
        "HINDSIGHT_NATIVE_CONSOLIDATION_LLM_BATCH_SIZE": str(llm_batch_size),
        "HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS": str(args.source_facts_max_tokens),
    }
    old_env: dict[str, str | None] = {k: os.environ.get(k) for k in env_overrides}
    summary["env_overrides"] = env_overrides
    summary["jobs_result"] = []
    exit_code = 0
    try:
        os.environ.update(env_overrides)
        mgr.switch_mode("import-minimax", allow_existing_queue=args.allow_existing_queue, enable_observations=True, llm_profile=profile)
        for job_idx in range(jobs):
            started = time.time()
            response = api_json("POST", f"/v1/default/banks/{args.bank}/consolidate", timeout=args.api_timeout)
            operation_id = str((response or {}).get("operation_id") or "")
            if not operation_id:
                raise GuardError(f"consolidate response has no operation_id: {response}")
            op = wait_operation(operation_id, poll_s=args.poll, timeout_s=args.operation_timeout)
            job_result = {
                "job_index": job_idx + 1,
                "operation_id": operation_id,
                "operation": op,
                "elapsed_s": round(time.time() - started, 1),
            }
            summary["jobs_result"].append(job_result)
            if op.get("status") != "completed":
                raise GuardError(f"consolidation operation {operation_id} ended with status={op.get('status')}")
        summary["ok"] = True
    except Exception as e:
        summary["ok"] = False
        summary["error"] = repr(e)
        exit_code = 1
    finally:
        try:
            mgr.switch_mode("normal-local", allow_existing_queue=True)
            restored = collect_status(args.bank)
            summary["restored_status_focus"] = {
                "health": restored.get("health"),
                "config_focus": restored.get("config_focus"),
                "provider_env": restored.get("provider_env"),
                "active_payload_null": restored.get("active_payload_null"),
                "native_candidates": restored.get("native_consolidation_unconsolidated_candidates"),
            }
        except Exception as e:
            summary["restore_error"] = repr(e)
            exit_code = 1
        for key, old in old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


def wait_bank_queue(bank: str, *, timeout_s: int, poll_s: int) -> None:
    started = time.time()
    while True:
        pending = int(psql_scalar(f"SELECT count(*) FROM async_operations WHERE bank_id='{bank}' AND status='pending';") or 0)
        processing = int(psql_scalar(f"SELECT count(*) FROM async_operations WHERE bank_id='{bank}' AND status='processing';") or 0)
        failed = int(psql_scalar(f"SELECT count(*) FROM async_operations WHERE bank_id='{bank}' AND status='failed';") or 0)
        print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "bank": bank, "pending": pending, "processing": processing, "failed": failed}, ensure_ascii=False), flush=True)
        if failed:
            raise GuardError(f"smoke bank has failed operations: {failed}")
        if pending == 0 and processing == 0:
            return
        if timeout_s and time.time() - started > timeout_s:
            raise TimeoutError(f"bank {bank} queue did not drain within {timeout_s}s")
        time.sleep(poll_s)


def cmd_smoke_reflect_local(args: argparse.Namespace) -> int:
    payload = {
        "query": args.query,
        "budget": "low",
        "max_tokens": args.max_tokens,
        "tags": ["__native_smoke_no_match__"],
        "tags_match": "any_strict",
        "response_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "string"},
            },
            "required": ["answer", "confidence"],
        },
    }
    started = time.time()
    data = api_json("POST", f"/v1/default/banks/{args.bank}/reflect", payload=payload, timeout=args.timeout)
    print(json.dumps({"elapsed_s": round(time.time() - started, 1), "response": data}, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke_consolidation_local(args: argparse.Namespace) -> int:
    mgr = __import__("hindsight_minimax_import")
    smoke_bank = args.smoke_bank or f"{SMOKE_BANK_PREFIX}{int(time.time())}"
    if not smoke_bank.startswith(SMOKE_BANK_PREFIX):
        raise GuardError(f"smoke bank must start with {SMOKE_BANK_PREFIX!r}")
    pre = collect_status(args.bank)
    stats = pre.get("stats") if isinstance(pre.get("stats"), dict) else {}
    if not args.allow_existing_queue and (int(stats.get("pending_operations") or 0) or int(stats.get("processing_operations") or 0)):
        raise GuardError("main bank queue is not empty; refusing native consolidation smoke")
    result: dict[str, Any] = {"smoke_bank": smoke_bank, "model": args.model, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        mgr.try_disable_observations_before_restart("native-consolidation-smoke")
        mgr.recreate_container(mgr.ollama_env(model=args.model, consolidation_slots=1))
        mgr.wait_health(timeout_s=120)
        # Patch in-container parser/backoff code without forcing a second restart; this run is local but validates patch presence.
        cmd_apply_patch(argparse.Namespace(restart=False))
        api_json("PATCH", f"/v1/default/banks/{args.bank}/config", payload={"updates": {"enable_observations": False}}, timeout=20)
        api_json(
            "PUT",
            f"/v1/default/banks/{smoke_bank}",
            payload={
                "retain_extraction_mode": "concise",
                "retain_chunk_size": 8000,
                "enable_observations": False,
                "observations_mission": "Synthesize only stable technical observations from the smoke-test facts.",
            },
            timeout=20,
        )
        retain_payload = {
            "async": True,
            "items": [
                {
                    "document_id": f"{smoke_bank}::doc1",
                    "context": "native workflow smoke test",
                    "tags": ["native-smoke"],
                    "content": (
                        "Native workflow smoke fact A: project Alpha uses a local guard before paid Hindsight consolidation. "
                        "Native workflow smoke fact B: the guard blocks payload_null and large unconsolidated backlogs before provider switch."
                    ),
                }
            ],
        }
        retain_response = api_json("POST", f"/v1/default/banks/{smoke_bank}/memories", payload=retain_payload, timeout=30)
        result["retain_response"] = retain_response
        wait_bank_queue(smoke_bank, timeout_s=args.timeout, poll_s=args.poll)
        before_obs = int(psql_scalar(f"SELECT count(*) FROM memory_units WHERE bank_id='{smoke_bank}' AND fact_type='observation';") or 0)
        cons_response = api_json("POST", f"/v1/default/banks/{smoke_bank}/consolidate", timeout=30)
        result["consolidate_response"] = cons_response
        wait_bank_queue(smoke_bank, timeout_s=args.timeout, poll_s=args.poll)
        after_obs = int(psql_scalar(f"SELECT count(*) FROM memory_units WHERE bank_id='{smoke_bank}' AND fact_type='observation';") or 0)
        unit_rows = psql_at(
            f"""
            SELECT fact_type, count(*) FROM memory_units
            WHERE bank_id='{smoke_bank}' GROUP BY fact_type ORDER BY fact_type;
            """
        )
        result["before_observations"] = before_obs
        result["after_observations"] = after_obs
        result["memory_units"] = [{"fact_type": r[0], "count": int(r[1])} for r in unit_rows]
        result["ok"] = after_obs >= before_obs
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["ok"] else 2
    finally:
        if args.cleanup:
            try:
                api_json("DELETE", f"/v1/default/banks/{smoke_bank}", timeout=30)
                print(f"deleted smoke bank: {smoke_bank}")
            except Exception as e:
                print(f"WARNING: failed to delete smoke bank {smoke_bank}: {e}", file=sys.stderr)
        print("restoring normal-local mode...", flush=True)
        try:
            mgr.switch_mode("normal-local", allow_existing_queue=True)
        except Exception as e:
            print(f"ERROR: failed to restore normal-local: {e}", file=sys.stderr)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hindsight native workflow guard/status/smoke tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p.add_argument("--bank", default=BANK)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("preflight")
    p.add_argument("--bank", default=BANK)
    p.add_argument("--allow-queue", action="store_true")
    p.add_argument("--allow-failed", action="store_true")
    p.add_argument("--expect-provider")
    p.add_argument("--require-observations-disabled", action="store_true")
    p.add_argument("--max-unconsolidated", type=int, default=None)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("cleanup-payload-null")
    p.add_argument("--bank", default=BANK)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm")
    p.set_defaults(func=cmd_cleanup_payload_null)

    p = sub.add_parser("apply-patch")
    p.add_argument("--restart", action="store_true")
    p.set_defaults(func=cmd_apply_patch)

    p = sub.add_parser("restore-local")
    p.add_argument("--bank", default=BANK)
    p.set_defaults(func=cmd_restore_local)

    p = sub.add_parser("run-native-consolidation-paid")
    p.add_argument("--bank", default=BANK)
    p.add_argument("--llm-profile", default=os.environ.get("HINDSIGHT_OFFLINE_LLM_PROFILE", "minimax"))
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--facts-per-job", type=int, default=50)
    p.add_argument("--fetch-size", type=int, default=None)
    p.add_argument("--llm-batch-size", type=int, default=None)
    p.add_argument("--source-facts-max-tokens", type=int, default=4096)
    p.add_argument("--max-unconsolidated", type=int, default=10000)
    p.add_argument("--allow-existing-queue", action="store_true")
    p.add_argument("--allow-failed", action="store_true")
    p.add_argument("--no-expect-local-provider", dest="expect_local_provider", action="store_false")
    p.set_defaults(expect_local_provider=True)
    p.add_argument("--execute", action="store_true", help="Actually switch to paid provider and trigger native /consolidate")
    p.add_argument("--confirm")
    p.add_argument("--api-timeout", type=int, default=30)
    p.add_argument("--operation-timeout", type=int, default=3600)
    p.add_argument("--poll", type=int, default=10)
    p.set_defaults(func=cmd_run_native_consolidation_paid)

    p = sub.add_parser("smoke-reflect-local")
    p.add_argument("--bank", default=BANK)
    p.add_argument("--query", default="Return a minimal JSON answer confirming native reflect is reachable; do not use external knowledge.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=int, default=240)
    p.set_defaults(func=cmd_smoke_reflect_local)

    p = sub.add_parser("smoke-consolidation-local")
    p.add_argument("--bank", default=BANK, help="main bank to protect/restore; smoke uses a temp bank")
    p.add_argument("--smoke-bank")
    p.add_argument("--model", default=os.environ.get("HINDSIGHT_NATIVE_SMOKE_MODEL", "qwen2:7b-instruct"))
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--poll", type=int, default=10)
    p.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-existing-queue", action="store_true")
    p.set_defaults(func=cmd_smoke_consolidation_local)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(args.func(args))
    except Exception as e:
        print(json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
