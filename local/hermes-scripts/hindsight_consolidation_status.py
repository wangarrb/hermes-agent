#!/usr/bin/env python3
"""Read-only Hindsight consolidation status snapshot.

Designed for skill publication: stdlib only, no production mutations, no .env
editing, no secret printing. It prefers the v0.6.1 Operations API for queue
status and keeps PostgreSQL as optional forensic fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def discover_default_psql() -> str:
    """Find a Hindsight-managed psql binary without pinning a version."""
    root = Path.home() / ".hindsight-docker" / "installation"
    candidates = sorted(root.glob("*/bin/psql"), key=lambda p: p.as_posix(), reverse=True)
    for p in candidates:
        if p.exists():
            return str(p)
    return "psql"


DEFAULT_API_URL = os.environ.get("HINDSIGHT_API_URL", "http://127.0.0.1:8888")
DEFAULT_TENANT = os.environ.get("HINDSIGHT_TENANT", "default")
DEFAULT_BANK = os.environ.get("HINDSIGHT_BANK", "hermes")
DEFAULT_PSQL = os.environ.get("HINDSIGHT_PSQL", discover_default_psql())
OPERATION_STATUSES = ["pending", "processing", "completed", "failed", "cancelled"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def api_get_json(url: str, *, params: dict[str, Any] | None = None, timeout: int = 10) -> dict[str, Any]:
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    if clean_params:
        sep = "&" if "?" in url else "?"
        url = url + sep + urllib.parse.urlencode(clean_params, doseq=True)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    data = json.loads(body) if body.strip() else {}
    if not isinstance(data, dict):
        return {"value": data}
    return data


def bank_url(api_url: str, tenant: str, bank: str, suffix: str) -> str:
    suffix = suffix.lstrip("/")
    return f"{api_url}/v1/{urllib.parse.quote(tenant, safe='')}/banks/{urllib.parse.quote(bank, safe='')}/{suffix}"


def operation_public_view(op: dict[str, Any]) -> dict[str, Any]:
    """Return an operation summary without task payloads or headers."""
    return {
        "id": op.get("id") or op.get("operation_id"),
        "status": op.get("status"),
        "task_type": op.get("task_type") or op.get("operation_type") or op.get("type"),
        "items_count": op.get("items_count"),
        "document_id": op.get("document_id"),
        "created_at": op.get("created_at"),
        "updated_at": op.get("updated_at"),
        "retry_count": op.get("retry_count"),
        "next_retry_at": op.get("next_retry_at"),
        "error_message": (str(op.get("error_message") or "")[:240] or None),
    }


def summarize_async_ops(ops: dict[str, dict[str, int]]) -> dict[str, Any]:
    totals_by_status = {status: sum(types.values()) for status, types in ops.items()}
    active = totals_by_status.get("processing", 0) + totals_by_status.get("running", 0)
    pending = totals_by_status.get("pending", 0) + totals_by_status.get("queued", 0)
    failed = totals_by_status.get("failed", 0) + totals_by_status.get("error", 0)
    completed = totals_by_status.get("completed", 0) + totals_by_status.get("done", 0)
    cancelled = totals_by_status.get("cancelled", 0)
    return {
        "totals_by_status": totals_by_status,
        "active_count": active,
        "pending_count": pending,
        "failed_count": failed,
        "completed_count": completed,
        "cancelled_count": cancelled,
        "has_active_work": active > 0 or pending > 0,
    }


def fetch_operations_api_summary(
    *,
    api_url: str,
    tenant: str,
    bank: str,
    timeout: int,
    exclude_parents: bool,
    sample_limit: int,
) -> dict[str, Any]:
    totals_by_status: dict[str, int] = {}
    samples_by_status: dict[str, list[dict[str, Any]]] = {}
    for status in OPERATION_STATUSES:
        data = api_get_json(
            bank_url(api_url, tenant, bank, "operations"),
            params={"status": status, "limit": min(max(sample_limit, 1), 100), "offset": 0, "exclude_parents": exclude_parents},
            timeout=timeout,
        )
        total = data.get("total")
        if isinstance(total, int):
            totals_by_status[status] = total
        else:
            totals_by_status[status] = len(data.get("operations") or [])
        samples_by_status[status] = [operation_public_view(op) for op in (data.get("operations") or [])[:sample_limit] if isinstance(op, dict)]

    recent = api_get_json(
        bank_url(api_url, tenant, bank, "operations"),
        params={"limit": min(max(sample_limit, 1), 100), "offset": 0, "exclude_parents": exclude_parents},
        timeout=timeout,
    )
    recent_ops = [operation_public_view(op) for op in (recent.get("operations") or [])[:sample_limit] if isinstance(op, dict)]
    summary = summarize_async_ops({status: {"all": count} for status, count in totals_by_status.items()})
    return {
        "source": "operations_api",
        "exclude_parents": exclude_parents,
        "total_operations": recent.get("total"),
        "summary": summary,
        "samples_by_status": samples_by_status,
        "recent_operations": recent_ops,
    }


def fetch_observability(api_url: str, tenant: str, bank: str, *, period: str, time_field: str, timeout: int) -> dict[str, Any]:
    out: dict[str, Any] = {"period": period, "time_field": time_field, "checks": {}}
    try:
        out["checks"]["memories_timeseries"] = {
            "ok": True,
            "data": api_get_json(
                bank_url(api_url, tenant, bank, "stats/memories-timeseries"),
                params={"period": period, "time_field": time_field},
                timeout=timeout,
            ),
        }
    except Exception as exc:
        out["checks"]["memories_timeseries"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        out["checks"]["audit_log_stats"] = {
            "ok": True,
            "data": api_get_json(bank_url(api_url, tenant, bank, "audit-logs/stats"), params={"period": period}, timeout=timeout),
        }
    except Exception as exc:
        out["checks"]["audit_log_stats"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def build_async_ops_sql(bank: str | None = None) -> str:
    # Keep SQL read-only. Some Hindsight schemas include bank_id, others do not;
    # callers can pass --no-bank-filter if needed.
    where = ""
    if bank:
        safe_bank = bank.replace("'", "''")
        where = f"WHERE bank_id = '{safe_bank}'"
    return (
        "SELECT status, operation_type, count(*) "
        "FROM async_operations "
        f"{where} "
        "GROUP BY status, operation_type "
        "ORDER BY status, operation_type;"
    )


def run_psql_tsv(
    *,
    psql: str,
    sql: str,
    host: str,
    port: str,
    user: str,
    dbname: str,
    timeout: int = 20,
) -> str:
    cmd = [
        psql,
        "-h",
        host,
        "-p",
        str(port),
        "-U",
        user,
        "-d",
        dbname,
        "-q",
        "-t",
        "-A",
        "-F",
        "\t",
        "-c",
        sql,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip()[:2000])
    return proc.stdout


def parse_async_ops_tsv(text: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        status, op_type, count_s = parts
        try:
            count = int(count_s)
        except ValueError:
            continue
        out[status][op_type] = count
    return {k: dict(v) for k, v in sorted(out.items())}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    api_url = args.api_url.rstrip("/")
    report: dict[str, Any] = {
        "schema_version": "hindsight-consolidation-status-v2",
        "generated_at": now_iso(),
        "api_url": api_url,
        "tenant": args.tenant,
        "bank": args.bank,
        "read_only": True,
        "v061_features": {
            "operations_api": not args.skip_operations_api,
            "operations_exclude_parents": not args.include_parents,
            "observability_api": not args.skip_observability,
            "psql_fallback": not args.skip_psql,
        },
        "checks": {},
    }

    try:
        report["checks"]["health"] = {"ok": True, "data": api_get_json(f"{api_url}/health", timeout=args.timeout)}
    except Exception as exc:
        report["checks"]["health"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    stats_url = bank_url(api_url, args.tenant, args.bank, "stats")
    try:
        stats = api_get_json(stats_url, timeout=args.timeout)
        report["checks"]["bank_stats"] = {"ok": True, "data": stats}
        report["bank_summary"] = {
            "total_documents": stats.get("total_documents"),
            "total_nodes": stats.get("total_nodes"),
            "total_observations": stats.get("total_observations"),
            "operations_by_status": stats.get("operations_by_status"),
        }
    except Exception as exc:
        report["checks"]["bank_stats"] = {"ok": False, "url": stats_url, "error": f"{type(exc).__name__}: {exc}"}

    if args.skip_operations_api:
        report["checks"]["operations_api"] = {"ok": None, "skipped": True}
    else:
        try:
            api_ops = fetch_operations_api_summary(
                api_url=api_url,
                tenant=args.tenant,
                bank=args.bank,
                timeout=args.timeout,
                exclude_parents=not args.include_parents,
                sample_limit=args.operations_sample_limit,
            )
            report["checks"]["operations_api"] = {"ok": True, "data": api_ops, "summary": api_ops["summary"]}
        except Exception as exc:
            report["checks"]["operations_api"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": "Use --skip-operations-api on old Hindsight versions; psql fallback remains optional."}

    if args.skip_observability:
        report["checks"]["observability"] = {"ok": None, "skipped": True}
    else:
        report["checks"]["observability"] = {
            "ok": True,
            "data": fetch_observability(api_url, args.tenant, args.bank, period=args.observability_period, time_field=args.timeseries_time_field, timeout=args.timeout),
        }

    if args.skip_psql:
        report["checks"]["async_operations_psql"] = {"ok": None, "skipped": True}
    else:
        psql = args.psql
        if not Path(psql).exists() and not args.allow_psql_from_path:
            report["checks"]["async_operations_psql"] = {
                "ok": False,
                "skipped": True,
                "error": f"psql not found: {psql}",
                "hint": "Set HINDSIGHT_PSQL or pass --allow-psql-from-path if psql is on PATH.",
            }
        else:
            sql = build_async_ops_sql(None if args.no_bank_filter else args.bank)
            try:
                raw = run_psql_tsv(
                    psql=psql,
                    sql=sql,
                    host=args.pg_host,
                    port=args.pg_port,
                    user=args.pg_user,
                    dbname=args.pg_db,
                    timeout=args.timeout,
                )
                ops = parse_async_ops_tsv(raw)
                report["checks"]["async_operations_psql"] = {"ok": True, "data": ops, "summary": summarize_async_ops(ops)}
            except Exception as exc:
                report["checks"]["async_operations_psql"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    checks = report["checks"]
    hard_failures = [name for name, check in checks.items() if isinstance(check, dict) and check.get("ok") is False and name in {"health", "bank_stats"}]
    api_summary = ((checks.get("operations_api") or {}).get("summary") or {}) if isinstance(checks.get("operations_api"), dict) else {}
    psql_summary = ((checks.get("async_operations_psql") or {}).get("summary") or {}) if isinstance(checks.get("async_operations_psql"), dict) else {}
    selected_summary = api_summary or psql_summary
    async_known = bool(api_summary or psql_summary)
    report["overall"] = {
        "ok": not hard_failures,
        "hard_failures": hard_failures,
        "async_status_source": "operations_api" if api_summary else ("psql" if psql_summary else "unknown"),
        "has_active_work": bool(selected_summary.get("has_active_work", False)) if async_known else None,
        "safe_to_restart_without_human_review": False,
        "restart_guidance": "If active/pending work exists or async status is unknown, wait for idle or get explicit human approval before restart/recreate.",
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Read-only Hindsight consolidation status snapshot")
    ap.add_argument("--api-url", default=DEFAULT_API_URL)
    ap.add_argument("--tenant", default=DEFAULT_TENANT)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--skip-operations-api", action="store_true", help="Disable v0.6.1 operations API status and rely on stats/psql")
    ap.add_argument("--include-parents", action="store_true", help="Include parent batch operations in Operations API queries; default excludes parents")
    ap.add_argument("--operations-sample-limit", type=int, default=5, help="Recent operations per status to include without payloads")
    ap.add_argument("--skip-observability", action="store_true", help="Skip v0.6.1 memories-timeseries and audit-log stats endpoints")
    ap.add_argument("--observability-period", default="1d", help="Period for v0.6.1 observability endpoints: 1d, 7d, or 30d")
    ap.add_argument("--timeseries-time-field", default="created_at", help="created_at, mentioned_at, or occurred_start")
    ap.add_argument("--psql", default=DEFAULT_PSQL)
    ap.add_argument("--allow-psql-from-path", action="store_true")
    ap.add_argument("--skip-psql", action="store_true")
    ap.add_argument("--no-bank-filter", action="store_true", help="Do not include WHERE bank_id=... in async_operations query")
    ap.add_argument("--pg-host", default=os.environ.get("HINDSIGHT_PGHOST", "/tmp"))
    ap.add_argument("--pg-port", default=os.environ.get("HINDSIGHT_PGPORT", "5432"))
    ap.add_argument("--pg-user", default=os.environ.get("HINDSIGHT_PGUSER", "hindsight"))
    ap.add_argument("--pg-db", default=os.environ.get("HINDSIGHT_PGDATABASE", "hindsight"))
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--json", action="store_true")
    return ap


def render_human_summary(report: dict[str, Any]) -> str:
    overall = report.get("overall") or {}
    health = (report.get("checks") or {}).get("health") or {}
    stats = report.get("bank_summary") or {}
    ops_api = (report.get("checks") or {}).get("operations_api") or {}
    ops_summary = ops_api.get("summary") or {}
    psql_check = (report.get("checks") or {}).get("async_operations_psql") or {}
    psql_summary = psql_check.get("summary") or {}
    selected = ops_summary or psql_summary
    obs = (report.get("checks") or {}).get("observability") or {}
    lines = [
        "Hindsight consolidation status",
        f"generated_at: {report.get('generated_at')}",
        f"bank: {report.get('tenant')}/{report.get('bank')}",
        f"health_ok: {health.get('ok')}",
        f"overall_ok: {overall.get('ok')}",
        f"async_status_source: {overall.get('async_status_source')}",
        f"active_work: {overall.get('has_active_work')}",
        f"documents/nodes/observations: {stats.get('total_documents')}/{stats.get('total_nodes')}/{stats.get('total_observations')}",
    ]
    if selected:
        lines.append(f"async_totals_by_status: {selected.get('totals_by_status')}")
    if ops_api.get("ok") is False:
        lines.append(f"operations_api_error: {ops_api.get('error')}")
    if psql_check.get("ok") is False and not psql_check.get("skipped"):
        lines.append(f"psql_async_operations_error: {psql_check.get('error')}")
    elif psql_check.get("skipped"):
        lines.append("psql_async_operations: skipped")
    if obs.get("skipped"):
        lines.append("observability: skipped")
    elif isinstance(obs.get("data"), dict):
        obs_checks = obs["data"].get("checks") or {}
        lines.append("observability_ok: " + str({k: v.get("ok") for k, v in obs_checks.items() if isinstance(v, dict)}))
    lines.append(f"restart_guidance: {overall.get('restart_guidance')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_human_summary(report))
    return 0 if report.get("overall", {}).get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
