#!/usr/bin/env python3
"""Install/runtime preflight for the Hindsight memory pipeline.

Read-only by default.  It validates paths, Hindsight health, bank/runtime safety,
and the default 20x3 consolidation tuning.  Local config/tuning files are written
only when explicitly requested.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
import urllib.parse
from urllib import request as urlrequest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_pipeline_common import (  # noqa: E402
    CONFIG_SCHEMA_VERSION,
    config_path,
    consolidation_tuning,
    ensure_local_dirs,
    load_config,
    path_from_config,
    script_path,
    write_default_config,
)

REQUIRED_SCRIPTS = [
    "hindsight_memory_pipeline.py",
    "hindsight_session_manifest.py",
    "hindsight_session_retain_runner.py",
    "hindsight_minimax_import.py",
    "offline_hindsight_reflect_consolidate.py",
    "hindsight_offline_v2_rebuild.py",
    "hindsight_conflict_audit.py",
    "hindsight_repair_proposal_build.py",
    "hindsight_proposal_review.py",
    "hindsight_native_client.py",
    "hindsight_consolidation_status.py",
    "hindsight_wait_native_consolidation.py",
]


def check(name: str, ok: bool, *, severity: str = "block", detail: Any = None, hint: str | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "severity": severity, "detail": detail, "hint": hint}


def run(cmd: list[str], *, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)


def read_url_json(url: str, *, timeout: int = 8, params: dict[str, Any] | None = None) -> dict[str, Any]:
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}
    if clean_params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(clean_params, doseq=True)
    with urlrequest.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def bank_api_url(api: str, bank: str, suffix: str) -> str:
    suffix = suffix.lstrip("/")
    return f"{api}/v1/default/banks/{urllib.parse.quote(bank, safe='')}/{suffix}"


def read_operations_summary(api: str, bank: str, *, timeout: int = 8) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for status in ["pending", "processing", "completed", "failed", "cancelled"]:
        data = read_url_json(
            bank_api_url(api, bank, "operations"),
            timeout=timeout,
            params={"status": status, "limit": 1, "offset": 0, "exclude_parents": True},
        )
        total = data.get("total")
        totals[status] = int(total) if isinstance(total, int) else len(data.get("operations") or [])
    active = totals.get("pending", 0) + totals.get("processing", 0)
    return {"totals_by_status": totals, "active_or_pending": active, "exclude_parents": True, "source": "operations_api"}


def docker_env_snapshot(container: str) -> tuple[bool, dict[str, str] | str]:
    if not shutil.which("docker"):
        return False, "docker command not found"
    proc = run(["docker", "inspect", container, "--format", "{{range .Config.Env}}{{println .}}{{end}}"], timeout=20)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "docker inspect failed").strip()[:500]
    env: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return True, env


def docker_exec_cat(container: str, path: str) -> tuple[bool, str]:
    if not shutil.which("docker"):
        return False, "docker command not found"
    proc = run(["docker", "exec", container, "sh", "-lc", f"test -f {sh_quote(path)} && cat {sh_quote(path)}"], timeout=20)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "docker exec failed").strip()[:500]
    return True, proc.stdout


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def expected_tuning_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    t = consolidation_tuning(cfg)
    return {
        "consolidation_batch_size": t["consolidation_batch_size"],
        "consolidation_llm_batch_size": t["consolidation_llm_batch_size"],
        # Official upstream Hindsight >=0.5.3 / 0.6.x name.
        "max_memories_per_round": t["max_memories_per_round"],
        # Legacy alias kept so old local tuning files/wrappers compare cleanly.
        "max_memories_per_job": t["max_memories_per_round"],
        # External/offline pipeline fanout budget, not an upstream Hindsight env/config field.
        "parallel_batches": t["parallel_batches"],
        "recall_budget": t["recall_budget"],
        "source_facts_max_tokens": t["source_facts_max_tokens"],
        "source_facts_max_tokens_per_observation": t["source_facts_max_tokens_per_observation"],
        "note": "Default publication profile: batch=64, llm_batch=8, max_memories_per_round=64, parallel_batches=8; verified 2026-05-14 (0 429s, ~$6 full drain). Hindsight v0.6.x also defaults consolidation_recall_budget=low and source_facts_max_tokens=4096.",
    }


def write_tuning_file(cfg: dict[str, Any], *, overwrite: bool = True) -> Path:
    p = path_from_config(cfg, "tuning_file")
    if p.exists() and not overwrite:
        raise FileExistsError(str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(expected_tuning_payload(cfg), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def compare_tuning(data: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    expected = expected_tuning_payload(cfg)
    actual = dict(data or {})
    if "max_memories_per_round" not in actual and "max_memories_per_job" in actual:
        actual["max_memories_per_round"] = actual["max_memories_per_job"]
    keys = [
        "consolidation_batch_size",
        "consolidation_llm_batch_size",
        "max_memories_per_round",
        "parallel_batches",
        "source_facts_max_tokens",
        "source_facts_max_tokens_per_observation",
    ]
    mismatches = {k: {"expected": expected[k], "actual": actual.get(k)} for k in keys if int(actual.get(k) or -1) != int(expected[k])}
    if str(actual.get("recall_budget", expected["recall_budget"])) != str(expected["recall_budget"]):
        mismatches["recall_budget"] = {"expected": expected["recall_budget"], "actual": actual.get("recall_budget")}
    return {"ok": not mismatches, "expected": {k: expected[k] for k in keys + ["recall_budget"]}, "actual": {k: actual.get(k) for k in keys + ["recall_budget", "max_memories_per_job"]}, "mismatches": mismatches}


def run_preflight(cfg: dict[str, Any], *, strict_runtime: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    home = Path(str(cfg.get("hermes_home"))).expanduser()
    scripts_dir = Path(str(cfg.get("scripts_dir"))).expanduser()
    api = str(cfg.get("api_url") or "http://127.0.0.1:8888").rstrip("/")
    bank = str(cfg.get("bank") or "hermes")
    docker_cfg = cfg.get("docker") or {}
    container = str(docker_cfg.get("container") or "hindsight")

    checks.append(check("config_schema", cfg.get("schema_version") == CONFIG_SCHEMA_VERSION, detail=cfg.get("schema_version"), hint=f"Expected {CONFIG_SCHEMA_VERSION}"))
    checks.append(check("hermes_home_exists", home.exists(), detail=str(home)))
    checks.append(check("scripts_dir_exists", scripts_dir.exists(), detail=str(scripts_dir)))
    for script in REQUIRED_SCRIPTS:
        p = Path(script_path(cfg, script))
        checks.append(check(f"script_exists:{script}", p.exists(), detail=str(p)))

    for key in ["sessions_dir", "manifest_dir", "approved_root", "proposal_root", "review_root", "status_root"]:
        p = path_from_config(cfg, key)
        parent = p if p.exists() else p.parent
        checks.append(check(f"path_parent_writable:{key}", parent.exists() and os.access(parent, os.W_OK), severity="block", detail=str(p)))

    # Hindsight API and bank checks.
    try:
        health = read_url_json(f"{api}/health")
        checks.append(check("hindsight_api_health", True, detail=health, severity="block"))
    except Exception as exc:
        checks.append(check("hindsight_api_health", False, detail=f"{type(exc).__name__}: {exc}", hint="Start Hindsight and verify api_url in pipeline_config.json"))
        health = {}

    try:
        stats = read_url_json(bank_api_url(api, bank, "stats"))
        checks.append(check("hindsight_bank_stats", True, detail={k: stats.get(k) for k in ["total_documents", "total_nodes", "total_observations", "operations_by_status"]}, severity="block"))
    except Exception as exc:
        checks.append(check("hindsight_bank_stats", False, detail=f"{type(exc).__name__}: {exc}", hint=f"Create/verify bank {bank}"))

    try:
        ops_summary = read_operations_summary(api, bank)
        checks.append(check("operations_api_read_exclude_parents", True, severity="block", detail=ops_summary))
        checks.append(check(
            "operations_api_no_active_work",
            int(ops_summary.get("active_or_pending") or 0) == 0,
            severity="warn",
            detail=ops_summary,
            hint="Active/pending work exists; avoid restart/recreate and production-adjacent changes unless the user accepts interruption",
        ))
    except Exception as exc:
        checks.append(check(
            "operations_api_read_exclude_parents",
            False,
            severity="warn",
            detail=f"{type(exc).__name__}: {exc}",
            hint="Hindsight v0.6.1 exposes /operations?exclude_parents=true; if this is an older server, use psql as forensic fallback",
        ))

    try:
        bank_cfg = read_url_json(bank_api_url(api, bank, "config"))
        actual_cfg = bank_cfg.get("config") if isinstance(bank_cfg, dict) else {}
        checks.append(check("bank_config_read", True, detail={k: actual_cfg.get(k) for k in ["enable_observations", "consolidation_llm_batch_size", "consolidation_max_memories_per_round", "consolidation_recall_budget", "consolidation_source_facts_max_tokens", "retain_chunk_size"]}, severity="warn"))
    except Exception as exc:
        checks.append(check("bank_config_read", False, severity="warn", detail=f"{type(exc).__name__}: {exc}"))

    try:
        timeseries = read_url_json(bank_api_url(api, bank, "stats/memories-timeseries"), params={"period": "1d", "time_field": "created_at"})
        audit_stats = read_url_json(bank_api_url(api, bank, "audit-logs/stats"), params={"period": "1d"})
        checks.append(check("v061_observability_endpoints", True, severity="warn", detail={"memories_timeseries_keys": sorted(timeseries.keys())[:10], "audit_log_stats_keys": sorted(audit_stats.keys())[:10]}))
    except Exception as exc:
        checks.append(check("v061_observability_endpoints", False, severity="warn", detail=f"{type(exc).__name__}: {exc}", hint="v0.6.1 exposes memories-timeseries and audit-log stats; daily reports should use them when available"))

    try:
        schema = read_url_json(f"{api}/v1/bank-template-schema")
        checks.append(check("v061_bank_template_schema", True, severity="warn", detail={"keys": sorted(schema.keys())[:10]}))
    except Exception as exc:
        checks.append(check("v061_bank_template_schema", False, severity="warn", detail=f"{type(exc).__name__}: {exc}", hint="v0.6.1 export/import snapshots need /v1/bank-template-schema"))

    # Proposal review governance checks.  Publication flow must stop at local
    # review packets unless an LLM advisory pass and human go/no-go are present.
    review_cfg = cfg.get("review") or {}
    checks.append(check(
        "proposal_review_requires_llm",
        review_cfg.get("require_llm_review") is True,
        severity="block",
        detail={"require_llm_review": review_cfg.get("require_llm_review")},
        hint="Set review.require_llm_review=true; production proposal promotion must have an LLM advisory judgement",
    ))
    checks.append(check(
        "proposal_review_requires_human_approval",
        review_cfg.get("require_human_approval") is True,
        severity="block",
        detail={"require_human_approval": review_cfg.get("require_human_approval")},
        hint="Set review.require_human_approval=true; final go/no-go must stay human-only",
    ))
    proposal_review_cfg = review_cfg.get("proposal_review") or {}
    checks.append(check(
        "proposal_review_confirm_token",
        proposal_review_cfg.get("confirm_token") == "review-hindsight-proposals",
        severity="block",
        detail={"confirm_token": proposal_review_cfg.get("confirm_token")},
        hint="Use confirm token review-hindsight-proposals for advisory LLM review; this still does not allow production writes",
    ))

    # Hermes Hindsight provider config safety.
    h_cfg_path = home / "hindsight" / "config.json"
    if h_cfg_path.exists():
        try:
            h_cfg = json.loads(h_cfg_path.read_text(encoding="utf-8"))
            checks.append(check("hermes_auto_retain_disabled", h_cfg.get("auto_retain") is False, severity="block", detail={"auto_retain": h_cfg.get("auto_retain"), "path": str(h_cfg_path)}, hint="auto_retain must stay false during offline/paid pipeline runs"))
        except Exception as exc:
            checks.append(check("hermes_hindsight_config_read", False, severity="warn", detail=f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(check("hermes_hindsight_config_present", False, severity="warn", detail=str(h_cfg_path)))

    # Docker/container checks.
    docker_required = bool(docker_cfg.get("required", True))
    docker_path = shutil.which("docker")
    checks.append(check("docker_command", bool(docker_path), severity="block" if docker_required else "warn", detail=docker_path))
    if docker_path:
        proc = run(["docker", "inspect", container, "--format", "{{.State.Running}}"], timeout=20)
        running = proc.returncode == 0 and proc.stdout.strip().lower() == "true"
        checks.append(check("hindsight_container_running", running, severity="block" if docker_required else "warn", detail={"container": container, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()[:200]}))
        env_ok, env_data = docker_env_snapshot(container)
        if env_ok and isinstance(env_data, dict):
            t = consolidation_tuning(cfg)
            env_expectations = {
                "HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND": str(t["max_memories_per_round"]),
                "HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET": str(t["recall_budget"]),
                "HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS": str(t["source_facts_max_tokens"]),
                "HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION": str(t["source_facts_max_tokens_per_observation"]),
                "HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA": "false",
            }
            env_mismatch = {k: {"expected": v, "actual": env_data.get(k)} for k, v in env_expectations.items() if str(env_data.get(k)) != v}
            checks.append(check("v061_runtime_env_official_tuning", not env_mismatch, severity="block" if strict_runtime else "warn", detail={"mismatches": env_mismatch, "checked_keys": sorted(env_expectations)}, hint="Restart/recreate Hindsight only after operations are idle so official v0.6.x env keys apply"))
            try:
                worker_max = int(env_data.get("HINDSIGHT_API_WORKER_MAX_SLOTS") or 0)
            except ValueError:
                worker_max = -1
            reserved_keys = [
                "HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS",
                "HINDSIGHT_API_WORKER_RETAIN_MAX_SLOTS",
                "HINDSIGHT_API_WORKER_REFRESH_MENTAL_MODEL_MAX_SLOTS",
            ]
            reservations: dict[str, int] = {}
            for key in reserved_keys:
                try:
                    reservations[key] = int(env_data.get(key) or 0)
                except ValueError:
                    reservations[key] = -1
            reserved_total = sum(v for v in reservations.values() if v > 0)
            checks.append(check("v061_worker_slot_reservations", worker_max > 0 and reserved_total <= worker_max and reservations.get("HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS", 0) >= 1, severity="warn", detail={"worker_max_slots": worker_max, "reservations": reservations, "reserved_total": reserved_total}, hint="Worker *_MAX_SLOTS are reservations inside WORKER_MAX_SLOTS, not additive pools"))
            read_db_url = env_data.get("HINDSIGHT_API_READ_DATABASE_URL")
            checks.append(check("v061_read_database_url", True, severity="warn", detail={"configured": bool(read_db_url)}, hint="Only configure HINDSIGHT_API_READ_DATABASE_URL when there is a real read-only replica/backend; same-writer is not a safety improvement"))
        else:
            checks.append(check("v061_runtime_env_inspect", False, severity="warn", detail=env_data))

    # PostgreSQL client is optional for public-release path, but useful for forensic gates.
    psql_path = str((cfg.get("psql") or {}).get("path") or "")
    psql_required = bool((cfg.get("psql") or {}).get("required", False))
    psql_ok = bool(psql_path and Path(psql_path).exists()) or bool(shutil.which("psql"))
    checks.append(check("psql_available", psql_ok, severity="block" if psql_required else "warn", detail=psql_path or shutil.which("psql"), hint="Set HINDSIGHT_PSQL or config.psql.path if DB-forensic checks are needed"))

    # Local tuning file and runtime tuning.
    tuning_file = path_from_config(cfg, "tuning_file")
    if tuning_file.exists():
        try:
            tuning_data = json.loads(tuning_file.read_text(encoding="utf-8"))
            cmp = compare_tuning(tuning_data, cfg)
            checks.append(check("local_consolidation_tuning_64x8", cmp["ok"], severity="block", detail=cmp, hint="Run preflight --write-tuning or fix pipeline_config.consolidation"))
        except Exception as exc:
            checks.append(check("local_consolidation_tuning_read", False, detail=f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(check("local_consolidation_tuning_file", False, detail=str(tuning_file), hint="Run preflight --write-tuning to create the default 64x8 tuning file"))

    # Runtime (container-side) tuning
    local_tuning_ok = any(c.get("key") == "local_consolidation_tuning_64x8" and c.get("ok") for c in checks)
    tuning_container_path = str(docker_cfg.get("tuning_container_path") or "")
    if docker_path and tuning_container_path:
        ok, raw = docker_exec_cat(container, tuning_container_path)
        if ok:
            try:
                runtime_cmp = compare_tuning(json.loads(raw), cfg)
                checks.append(check("runtime_consolidation_tuning_64x8", runtime_cmp["ok"], severity="block" if strict_runtime else "warn", detail=runtime_cmp, hint="Recreate/restart Hindsight through hindsight_minimax_import.py so the 64x8 tuning is copied into the container"))
            except Exception as exc:
                checks.append(check("runtime_consolidation_tuning_parse", False, severity="block" if strict_runtime else "warn", detail=f"{type(exc).__name__}: {exc}"))
        else:
            checks.append(check("runtime_consolidation_tuning_present", False, severity="block" if strict_runtime else "warn", detail=raw))

    blocking = [c for c in checks if not c["ok"] and c.get("severity") == "block"]
    warnings = [c for c in checks if not c["ok"] and c.get("severity") == "warn"]
    return {
        "schema_version": "hindsight-pipeline-preflight-v1",
        "ok": not blocking,
        "blocking_count": len(blocking),
        "warning_count": len(warnings),
        "config_path": cfg.get("config_path"),
        "api_url": api,
        "bank": bank,
        "checks": checks,
        "blocking": blocking,
        "warnings": warnings,
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Hindsight pipeline preflight: {'OK' if report['ok'] else 'BLOCKED'}",
        f"config: {report.get('config_path')}",
        f"api/bank: {report.get('api_url')} / {report.get('bank')}",
        f"blocking={report.get('blocking_count')} warnings={report.get('warning_count')}",
    ]
    for c in report.get("blocking") or []:
        lines.append(f"BLOCK {c['name']}: {c.get('detail')}" + (f" | {c.get('hint')}" if c.get("hint") else ""))
    for c in report.get("warnings") or []:
        lines.append(f"WARN  {c['name']}: {c.get('detail')}" + (f" | {c.get('hint')}" if c.get("hint") else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preflight Hindsight memory pipeline installation/runtime configuration")
    ap.add_argument("--config", type=Path, help="Path to pipeline_config.json")
    ap.add_argument("--init-config", action="store_true", help="Write default config JSON, then exit unless --run is also provided")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting config/tuning when used with --init-config/--write-tuning")
    ap.add_argument("--write-tuning", action="store_true", help="Write local default 64x8 consolidation tuning JSON")
    ap.add_argument("--strict-runtime", action="store_true", help="Treat runtime/container tuning mismatch as blocking")
    ap.add_argument("--run", action="store_true", help="Run checks after --init-config")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.init_config:
        p = write_default_config(args.config, overwrite=args.overwrite)
        if not args.run:
            out = {"ok": True, "wrote_config": str(p), "next": f"python3 {Path(__file__)} --config {p} --write-tuning --json"}
            print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else f"wrote config: {p}")
            return 0

    cfg = load_config(args.config)
    if args.write_tuning:
        p = write_tuning_file(cfg, overwrite=args.overwrite or True)
        cfg.setdefault("paths", {})["tuning_file"] = str(p)

    ensure_local_dirs(cfg)
    report = run_preflight(cfg, strict_runtime=args.strict_runtime)
    if args.write_tuning:
        report["wrote_tuning"] = str(path_from_config(cfg, "tuning_file"))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
