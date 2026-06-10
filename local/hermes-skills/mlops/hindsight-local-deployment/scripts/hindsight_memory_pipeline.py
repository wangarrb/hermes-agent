#!/usr/bin/env python3
"""Mode-based Hindsight memory/offline pipeline orchestrator.

Default is non-mutating plan/dry-run. Real production writes require:
  --execute --confirm run-hindsight-pipeline

Modes:
- daily: incremental session/json retain across default + Hindsight-backed profiles, native observations + daily offline reflect + local/v2 refresh gate.
- weekly: weekly/all-history reflect + repair-zone proposal sweep + conflict/lineage gate.
- full: daily + weekly + optional long-cycle wiki maintenance; default history is incremental, --history all forces re-retain of all sessions.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_pipeline_common import (  # noqa: E402
    consolidation_tuning,
    load_config,
    path_from_config,
    script_path,
)

ACTIVE_CONFIG = load_config()
HERMES_HOME = Path(str(ACTIVE_CONFIG.get("hermes_home"))).expanduser()
SCRIPTS_DIR = Path(str(ACTIVE_CONFIG.get("scripts_dir"))).expanduser()
PIPELINE_CONFIRM = "run-hindsight-pipeline"
RETAIN_CONFIRM = "retain-hindsight-session-manifest"
V2_PUBLISH_CONFIRM = "publish-hindsight-v2-canonical"
PROPOSAL_REVIEW_CONFIRM = "review-hindsight-proposals"
DEFAULT_BANK = "hermes"
DEFAULT_APPROVED_ROOT = path_from_config(ACTIVE_CONFIG, "approved_root")
DEFAULT_PROPOSAL_ROOT = path_from_config(ACTIVE_CONFIG, "proposal_root")
DEFAULT_STATUS_ROOT = path_from_config(ACTIVE_CONFIG, "status_root")


@dataclass
class Step:
    name: str
    description: str
    command: list[str]
    mutating: bool = False
    writes_local: bool = False
    optional: bool = False
    produces: dict[str, str] = field(default_factory=dict)
    consumes: dict[str, str] = field(default_factory=dict)


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def get_cfg(args: argparse.Namespace | None = None) -> dict[str, Any]:
    return getattr(args, "pipeline_config", None) or ACTIVE_CONFIG


def activate_config(args: argparse.Namespace) -> dict[str, Any]:
    global ACTIVE_CONFIG, HERMES_HOME, SCRIPTS_DIR, DEFAULT_APPROVED_ROOT, DEFAULT_PROPOSAL_ROOT, DEFAULT_STATUS_ROOT
    cfg = load_config(getattr(args, "config", None))
    args.pipeline_config = cfg
    ACTIVE_CONFIG = cfg
    HERMES_HOME = Path(str(cfg.get("hermes_home"))).expanduser()
    SCRIPTS_DIR = Path(str(cfg.get("scripts_dir"))).expanduser()
    DEFAULT_APPROVED_ROOT = path_from_config(cfg, "approved_root")
    DEFAULT_PROPOSAL_ROOT = path_from_config(cfg, "proposal_root")
    DEFAULT_STATUS_ROOT = path_from_config(cfg, "status_root")
    return cfg


def py(script: str, args: argparse.Namespace | None = None) -> str:
    return script_path(get_cfg(args), script)


def pipeline_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    t = consolidation_tuning(get_cfg(args))
    env.update({
        "HINDSIGHT_NATIVE_CONSOLIDATION_BATCH_SIZE": str(t["consolidation_batch_size"]),
        "HINDSIGHT_NATIVE_CONSOLIDATION_LLM_BATCH_SIZE": str(t["consolidation_llm_batch_size"]),
        "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_ROUND": str(t["max_memories_per_round"]),
        # Legacy alias for old local wrappers.
        "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_JOB": str(t["max_memories_per_round"]),
        "HINDSIGHT_NATIVE_CONSOLIDATION_PARALLEL_BATCHES": str(t["parallel_batches"]),
        "HINDSIGHT_NATIVE_CONSOLIDATION_RECALL_BUDGET": str(t["recall_budget"]),
        "HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS": str(t["source_facts_max_tokens"]),
        "HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION": str(t["source_facts_max_tokens_per_observation"]),
    })
    return env


def resolve_daily_date(mode: str) -> str:
    now = datetime.now()
    if mode == "today":
        return now.date().isoformat()
    if mode == "yesterday":
        return (now.date() - timedelta(days=1)).isoformat()
    if mode == "auto":
        return (now.date() - timedelta(days=1) if now.hour < 6 else now.date()).isoformat()
    return mode


def resolve_week(mode: str) -> str:
    now = datetime.now().date()
    if mode == "current":
        iso = now.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if mode == "previous":
        iso = (now - timedelta(days=7)).isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return mode


def require_confirm(args: argparse.Namespace) -> None:
    if getattr(args, "execute", False) and getattr(args, "confirm", None) != PIPELINE_CONFIRM:
        raise SystemExit(f"--execute requires --confirm {PIPELINE_CONFIRM}")


def common_wrapper_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.llm_profile:
        out += ["--llm-profile", args.llm_profile]
    out += ["--poll", str(args.poll), "--timeout", str(args.timeout)]
    return out


def preflight_step(args: argparse.Namespace) -> Step:
    cmd = [
        sys.executable,
        py("hindsight_pipeline_preflight.py", args),
        "--config",
        str(get_cfg(args).get("config_path")),
        "--write-tuning",
        "--json",
    ]
    if getattr(args, "strict_runtime", False):
        cmd.append("--strict-runtime")
    return Step(
        name="preflight",
        description="Read-only install/runtime preflight plus local 20x3 tuning file check/write. Blocks unsafe execute runs.",
        command=cmd,
        mutating=False,
        writes_local=True,
    )


def session_manifest_step(args: argparse.Namespace) -> Step:
    cmd = [
        sys.executable,
        py("hindsight_session_manifest.py"),
        "--bank-target",
        args.bank,
        "--output-dir",
        str(path_from_config(get_cfg(args), "manifest_dir")),
        "--profile-mode",
        str(getattr(args, "session_profile_mode", "hindsight") or "hindsight"),
        "--json",
    ]
    if getattr(args, "include_codex", True):
        cmd.append("--include-codex")
    else:
        cmd.append("--no-include-codex")
    if getattr(args, "session_limit", None):
        cmd += ["--limit", str(args.session_limit)]
    return Step(
        name="build_session_manifest",
        description="Build cleaned session/json manifest from default Hermes, Hindsight-backed profiles, and Codex rollout JSONL. Local files only; no Hindsight writes.",
        command=cmd,
        mutating=False,
        writes_local=True,
        produces={"manifest": "summary.paths.manifest"},
    )


def retain_session_step(args: argparse.Namespace) -> Step:
    cmd = [
        sys.executable,
        py("hindsight_minimax_import.py"),
        "session-manifest-retain-llm",
        *common_wrapper_args(args),
        "--enable-observations",
        "--manifest",
        "{{manifest}}",
        "--bank",
        args.bank,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.history == "all":
        cmd.append("--ignore-submit-state")
    if args.execute:
        cmd += ["--execute", "--confirm", RETAIN_CONFIRM]
    wait_timeout = int(getattr(args, "session_retain_wait_timeout", 86400) or 86400)
    cmd += ["--wait-timeout-s", str(wait_timeout)]
    return Step(
        name="retain_session_manifest",
        description="Retain session/json manifest through paid LLM wrapper. Incremental by submit_state unless --history all.",
        command=cmd,
        mutating=bool(args.execute),
        writes_local=not bool(args.execute),
        consumes={"manifest": "build_session_manifest"},
    )


def daily_reflect_step(args: argparse.Namespace) -> Step:
    day = resolve_daily_date(args.date_mode)
    if args.execute:
        cmd = [
            sys.executable,
            py("hindsight_minimax_import.py"),
            "offline-reflect-llm",
            *common_wrapper_args(args),
            "--",
            "--scope",
            "daily",
            "--bank",
            args.bank,
            "--date",
            day,
            "--daily-source",
            "facts",
            "--mode",
            "submit",
            "--prefilter",
            args.prefilter,
        ]
    else:
        cmd = [
            sys.executable,
            py("offline_hindsight_reflect_consolidate.py"),
            "--scope",
            "daily",
            "--bank",
            args.bank,
            "--date",
            day,
            "--daily-source",
            "facts",
            "--mode",
            "dry-run",
            "--prefilter",
            args.prefilter,
        ]
    return Step(
        name="daily_reflect",
        description="Daily processed-fact consolidation/reflection. Dry-run does not call LLM/write Hindsight; execute submits daily outputs.",
        command=cmd,
        mutating=bool(args.execute),
        writes_local=True,
    )


def weekly_reflect_step(args: argparse.Namespace) -> Step:
    week = resolve_week(args.week_mode)
    if args.execute:
        cmd = [
            sys.executable,
            py("hindsight_minimax_import.py"),
            "offline-reflect-llm",
            *common_wrapper_args(args),
            "--",
            "--scope",
            "weekly",
            "--bank",
            args.bank,
            "--week",
            week,
            "--weekly-window",
            "all-history",
            "--weekly-source",
            "daily",
            "--weekly-group-by",
            "topic",
            "--backfill-missing-daily",
            "--mode",
            "submit",
            "--prefilter",
            args.prefilter,
        ]
    else:
        cmd = [
            sys.executable,
            py("offline_hindsight_reflect_consolidate.py"),
            "--scope",
            "weekly",
            "--bank",
            args.bank,
            "--week",
            week,
            "--weekly-window",
            "all-history",
            "--weekly-source",
            "daily",
            "--weekly-group-by",
            "topic",
            "--backfill-missing-daily",
            "--mode",
            "dry-run",
            "--prefilter",
            args.prefilter,
            "--budget-json",
            "--budget-max-pending-units",
            str(args.weekly_budget_max_pending_units),
            "--budget-max-pending-chars",
            str(args.weekly_budget_max_pending_chars),
        ]
    return Step(
        name="weekly_reflect",
        description="Weekly/all-history high-dimensional integration from daily outputs; includes backfill-missing-daily gate.",
        command=cmd,
        mutating=bool(args.execute),
        writes_local=True,
    )


def v2_rebuild_step(args: argparse.Namespace) -> Step:
    mode = "publish" if args.execute else "local"
    cmd = [
        sys.executable,
        py("hindsight_offline_v2_rebuild.py"),
        "--mode",
        mode,
        "--bank",
        args.bank,
        "--json",
    ]
    if args.execute:
        cmd += ["--confirm-publish", V2_PUBLISH_CONFIRM]
    return Step(
        name="v2_rebuild_gate",
        description="Rebuild local/offline V2 cards and run eval/conflict gates; publish only in execute mode with confirm token.",
        command=cmd,
        mutating=bool(args.execute),
        writes_local=True,
    )


def conflict_audit_step(args: argparse.Namespace) -> Step:
    cmd = [
        sys.executable,
        py("hindsight_conflict_audit.py"),
        "--bank",
        args.bank,
        "--api",
        str(get_cfg(args).get("api_url") or "http://127.0.0.1:8888"),
        "--block-severity",
        "P1",
        "--json",
    ]
    return Step(
        name="conflict_audit",
        description="Read-only conflict/lineage audit; blocks production quality claims when P1+ cases remain.",
        command=cmd,
        mutating=False,
        writes_local=True,
    )


def repair_zone_proposals_step(args: argparse.Namespace) -> Step:
    cmd = [
        sys.executable,
        py("hindsight_repair_proposal_build.py"),
        "--approved-index",
        "{{approved_index}}",
        "--output-root",
        str(args.output_root),
        "--top",
        str(args.top_proposals),
        "--json",
    ]
    return Step(
        name="repair_zone_proposals",
        description="Sweep approved review-repair sidecars into proposal-only canonical bundles. No production writes.",
        command=cmd,
        mutating=False,
        writes_local=True,
        optional=True,
    )


def proposal_review_step(args: argparse.Namespace) -> Step:
    review_cfg = (get_cfg(args).get("review") or {}).get("proposal_review") or {}
    cmd = [
        sys.executable,
        py("hindsight_proposal_review.py", args),
        "--review-root",
        str(path_from_config(get_cfg(args), "review_root")),
        "--top",
        str(args.top_proposals),
        "--max-llm-calls",
        str(getattr(args, "max_proposal_review_llm_calls", None) or review_cfg.get("max_llm_calls") or 10),
        "--json",
        "{{proposal_jsons}}",
    ]
    if getattr(args, "execute_proposal_review_llm", False):
        cmd += ["--execute-llm", "--confirm-review", str(getattr(args, "confirm_proposal_review", None) or "")]
    if getattr(args, "notify_proposal_review", False):
        cmd.append("--notify")
    return Step(
        name="proposal_review",
        description="Build local LLM/human review packets for proposal bundles. Advisory LLM calls require a separate review confirm; no production writes.",
        command=cmd,
        mutating=False,
        writes_local=True,
        optional=True,
        consumes={"proposal_jsons": "repair_zone_proposals"},
        produces={"review_packet": "paths.review_json"},
    )


def wiki_step(args: argparse.Namespace) -> Step:
    cmd = [
        sys.executable,
        py("wiki_auto_maintenance.py"),
        "--wait-hindsight",
        "--wait-timeout",
        str(args.timeout or 21600),
        "--wait-poll",
        str(args.poll),
    ]
    return Step(
        name="wiki_auto_maintenance",
        description="Long-cycle wiki candidate maintenance. Writes isolated auto-maintenance reports, not main wiki files.",
        command=cmd,
        mutating=False,
        writes_local=True,
        optional=True,
    )


def status_step(args: argparse.Namespace) -> Step:
    return Step(
        name="runtime_status",
        description="Show Hindsight health/stats/provider mode.",
        command=[sys.executable, py("hindsight_minimax_import.py", args), "status"],
        mutating=False,
    )


def queue_drain_step(args: argparse.Namespace, *, name: str, description: str) -> Step:
    """Wait for existing async Hindsight work before a provider/container switch.

    The paid-LLM wrapper intentionally refuses to switch providers while a bank
    has active operations. Daily/full runs used to discover that only after the
    local manifest was built, and the wrapper's restore path could still restart
    Hindsight even though no retain work had started. Make the queue wait an
    explicit orchestrator step so a scheduled run either starts cleanly or fails
    before mutating/provider-switching work.
    """
    timeout_s = int(
        getattr(args, "pre_existing_queue_wait_timeout", None)
        or getattr(args, "native_consolidation_wait_timeout", 86400)
        or 86400
    )
    poll_s = int(
        getattr(args, "pre_existing_queue_poll", None)
        or getattr(args, "native_consolidation_poll", None)
        or getattr(args, "poll", 60)
        or 60
    )
    return Step(
        name=name,
        description=description,
        command=[
            sys.executable,
            py("hindsight_minimax_import.py", args),
            "wait-queue",
            "--bank",
            args.bank,
            "--poll",
            str(poll_s),
            "--timeout",
            str(timeout_s),
        ],
        mutating=False,
        writes_local=False,
    )


def native_consolidation_drain_step(args: argparse.Namespace, *, name: str, description: str) -> Step:
    """Wait for Hindsight native consolidation to finish before quality-sensitive stages."""
    cmd = [
        sys.executable,
        py("hindsight_wait_native_consolidation.py", args),
        "--api-url",
        str(get_cfg(args).get("api_url") or "http://127.0.0.1:8888"),
        "--bank",
        args.bank,
        "--max-pending",
        str(args.native_consolidation_max_pending),
        "--timeout-s",
        str(args.native_consolidation_wait_timeout),
        "--poll-s",
        str(args.native_consolidation_poll),
        "--json",
    ]
    if getattr(args, "allow_active_native_operations", False):
        cmd.append("--allow-active-operations")
    return Step(
        name=name,
        description=description,
        command=cmd,
        mutating=False,
        writes_local=True,
    )


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    cfg = activate_config(args)
    if not getattr(args, "bank", None):
        args.bank = str(cfg.get("bank") or DEFAULT_BANK)
    if not getattr(args, "llm_profile", None):
        args.llm_profile = str(cfg.get("llm_profile") or os.environ.get("HINDSIGHT_OFFLINE_LLM_PROFILE", "minimax"))
    if getattr(args, "output_root", None) is None:
        args.output_root = DEFAULT_PROPOSAL_ROOT
    if getattr(args, "skip_daily", False) and args.mode != "full":
        raise SystemExit("--skip-daily is only valid with full mode")
    if getattr(args, "native_consolidation_poll", None) is None:
        args.native_consolidation_poll = int(getattr(args, "poll", 60) or 60)
    require_confirm(args)
    steps: list[Step] = [preflight_step(args)]
    wait_native_consolidation = not bool(getattr(args, "no_wait_native_consolidation", False))
    if args.mode != "preflight":
        steps.append(status_step(args))
    if args.mode == "daily" or (args.mode == "full" and not getattr(args, "skip_daily", False)):
        steps.append(queue_drain_step(
            args,
            name="queue_drain_before_daily_retain",
            description="Wait for any existing Hindsight async queue before the daily paid-LLM retain/provider switch. Prevents active backlog from turning the daily observation step into a partial/no-op run.",
        ))
        steps.extend([session_manifest_step(args), retain_session_step(args), daily_reflect_step(args)])
        if wait_native_consolidation:
            steps.append(native_consolidation_drain_step(
                args,
                name="native_consolidation_drain_after_daily",
                description="Wait until Hindsight native source-fact consolidation is drained after daily/session ingestion, before V2 quality gates.",
            ))
        steps.append(v2_rebuild_step(args))
    if args.mode in {"weekly", "full"}:
        if wait_native_consolidation and (args.mode == "weekly" or getattr(args, "skip_daily", False)):
            steps.append(native_consolidation_drain_step(
                args,
                name="native_consolidation_drain_before_weekly",
                description="Wait for any existing Hindsight native consolidation backlog before weekly/full resume stages.",
            ))
        steps.append(weekly_reflect_step(args))
        if wait_native_consolidation:
            steps.append(native_consolidation_drain_step(
                args,
                name="native_consolidation_drain_after_weekly",
                description="Wait until Hindsight native consolidation from weekly/offline outputs is drained before V2/conflict/proposal gates.",
            ))
        steps.extend([v2_rebuild_step(args), conflict_audit_step(args)])
        if not getattr(args, "skip_repair_zone", False):
            steps.append(repair_zone_proposals_step(args))
            if not getattr(args, "skip_proposal_review", False):
                steps.append(proposal_review_step(args))
    if args.mode == "full" and args.include_wiki:
        steps.append(wiki_step(args))
    tuning = consolidation_tuning(cfg)
    return {
        "schema_version": "hindsight-memory-pipeline-plan-v2",
        "generated_at": iso_now(),
        "mode": args.mode,
        "history": args.history,
        "skip_daily": bool(getattr(args, "skip_daily", False)),
        "execute": bool(args.execute),
        "config_path": str(cfg.get("config_path")),
        "api_url": str(cfg.get("api_url")),
        "bank": args.bank,
        "consolidation_tuning": tuning,
        "wait_native_consolidation": wait_native_consolidation,
        "native_consolidation_gate": {
            "max_pending": int(getattr(args, "native_consolidation_max_pending", 0)),
            "timeout_s": int(getattr(args, "native_consolidation_wait_timeout", 86400)),
            "poll_s": int(getattr(args, "native_consolidation_poll", getattr(args, "poll", 60))),
            "allow_active_operations": bool(getattr(args, "allow_active_native_operations", False)),
        },
        "production_writes_possible": any(s.mutating for s in steps),
        "confirm_required_for_execute": PIPELINE_CONFIRM,
        "steps": [asdict(s) for s in steps],
    }


def redact_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(f"RUN {redact_cmd(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def parse_last_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    for idx in [text.rfind("\n{"), text.find("{")]:
        if idx < 0:
            continue
        start = idx + (1 if text[idx:idx+2] == "\n{" else 0)
        try:
            obj = json.loads(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def execute_plan(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    run_report: dict[str, Any] = {"started_at": iso_now(), "steps": [], "mode": args.mode, "history": args.history, "skip_daily": bool(getattr(args, "skip_daily", False)), "config_path": str(get_cfg(args).get("config_path")), "consolidation_tuning": consolidation_tuning(get_cfg(args)), "wait_native_consolidation": not bool(getattr(args, "no_wait_native_consolidation", False))}
    manifest_path: str | None = None
    proposal_jsons: list[str] = []
    for step in plan["steps"]:
        cmd = list(step["command"])
        if "{{manifest}}" in cmd:
            if not manifest_path:
                raise SystemExit("retain step requires manifest from build_session_manifest")
            cmd = [manifest_path if x == "{{manifest}}" else x for x in cmd]
        if "{{proposal_jsons}}" in cmd:
            if not proposal_jsons:
                run_report["steps"].append({"name": step["name"], "status": "skipped", "reason": "no proposal jsons generated in this run"})
                continue
            expanded: list[str] = []
            for x in cmd:
                if x == "{{proposal_jsons}}":
                    for proposal_json in proposal_jsons:
                        expanded += ["--proposal-json", proposal_json]
                else:
                    expanded.append(x)
            cmd = expanded
        if "{{approved_index}}" in cmd:
            approved_files = sorted(DEFAULT_APPROVED_ROOT.glob("*-observations_index.jsonl"))
            if not approved_files:
                run_report["steps"].append({"name": step["name"], "status": "skipped", "reason": "no approved sidecar indexes"})
                continue
            for approved in approved_files:
                stem = approved.name.replace("-observations_index.jsonl", "")
                c = [str(approved) if x == "{{approved_index}}" else x for x in cmd]
                c += ["--stem", stem]
                proc = run_command(c, env=pipeline_env(args))
                parsed = parse_last_json(proc.stdout or "")
                proposal_json = ((parsed.get("paths") or {}).get("proposal_json")) if isinstance(parsed, dict) else None
                if proposal_json:
                    proposal_jsons.append(str(proposal_json))
                run_report["steps"].append({
                    "name": step["name"],
                    "status": "ok",
                    "approved_index": str(approved),
                    "returncode": proc.returncode,
                    "proposal_json": proposal_json,
                })
            continue
        proc = run_command(cmd, env=pipeline_env(args))
        parsed = parse_last_json(proc.stdout or "")
        if step["name"] == "build_session_manifest":
            manifest_path = str(((parsed.get("paths") or {}).get("manifest")) or "")
            if not manifest_path:
                raise SystemExit("build_session_manifest did not return paths.manifest")
        run_report["steps"].append({"name": step["name"], "status": "ok", "returncode": proc.returncode, "parsed_keys": sorted(parsed.keys())[:20]})
    run_report["finished_at"] = iso_now()
    run_report["status"] = "ok"
    DEFAULT_STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    path = DEFAULT_STATUS_ROOT / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.mode}-pipeline-run.json"
    path.write_text(json.dumps(run_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_report["run_report_path"] = str(path)
    return run_report


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run/plan Hindsight memory pipeline modes: daily, weekly, full")
    ap.add_argument("mode", choices=["preflight", "daily", "weekly", "full"])
    ap.add_argument("--config", type=Path, help="Path to pipeline_config.json; defaults to $HINDSIGHT_PIPELINE_CONFIG or ~/.hermes/hindsight/pipeline_config.json")
    ap.add_argument("--history", choices=["incremental", "all"], default="incremental", help="incremental uses submit_state skip; all ignores submit_state and re-retains all manifest records")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--llm-profile", default=None)
    ap.add_argument("--prefilter", choices=["safe", "balanced", "strict"], default="safe")
    ap.add_argument("--date-mode", default="auto", help="auto/today/yesterday/YYYY-MM-DD")
    ap.add_argument("--week-mode", default="current", help="current/previous/YYYY-Www")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--session-retain-wait-timeout", type=int, default=86400, help="Seconds to wait for session retain operation ids; full observation loads can legitimately run for up to 24h")
    ap.add_argument("--native-consolidation-wait-timeout", type=int, default=86400, help="Seconds to wait for native Hindsight source-fact consolidation gates before V2/conflict/proposal stages")
    ap.add_argument("--native-consolidation-poll", type=int, default=None, help="Poll interval for native consolidation gates; defaults to --poll")
    ap.add_argument("--native-consolidation-max-pending", type=int, default=0, help="Allowed pending_consolidation source facts before continuing quality-sensitive stages")
    ap.add_argument("--pre-existing-queue-wait-timeout", type=int, default=None, help="Seconds to wait for already-active Hindsight operations before the first daily/full provider switch; defaults to --native-consolidation-wait-timeout")
    ap.add_argument("--pre-existing-queue-poll", type=int, default=None, help="Poll interval for the pre-existing queue drain; defaults to --native-consolidation-poll/--poll")
    ap.add_argument("--allow-active-native-operations", action="store_true", help="Native consolidation gate ignores active/pending operations and only checks pending_consolidation")
    ap.add_argument("--no-wait-native-consolidation", action="store_true", help="Skip native consolidation drain gates; only use for explicit emergency/resume debugging")
    ap.add_argument("--session-limit", type=int, default=None)
    ap.add_argument("--session-profile-mode", choices=["hindsight", "all", "none"], default=os.environ.get("HINDSIGHT_SESSION_PROFILE_MODE", "hindsight"), help="Session retain source discovery: hindsight=default + profiles with memory.provider=hindsight (default); all=all profiles; none=default only")
    ap.add_argument("--include-codex", dest="include_codex", action="store_true", default=os.environ.get("HINDSIGHT_INCLUDE_CODEX", "1").lower() not in {"0", "false", "no", "off"}, help="Include Codex rollout JSONL sessions in the daily manifest build (default on)")
    ap.add_argument("--no-include-codex", dest="include_codex", action="store_false", help="Disable Codex rollout JSONL scanning for this pipeline run")
    ap.add_argument("--weekly-budget-max-pending-units", type=int, default=12)
    ap.add_argument("--weekly-budget-max-pending-chars", type=int, default=500000)
    ap.add_argument("--include-wiki", action="store_true", help="For full mode, include long-cycle wiki candidate maintenance")
    ap.add_argument("--skip-daily", action="store_true", help="For full mode resume runs, skip session manifest/retain, daily reflect, and the daily V2 rebuild gate")
    ap.add_argument("--skip-repair-zone", action="store_true", help="For weekly/full, skip approved repair sidecar proposal sweep")
    ap.add_argument("--skip-proposal-review", action="store_true", help="For weekly/full, skip local proposal LLM/human review packet generation")
    ap.add_argument("--execute-proposal-review-llm", action="store_true", help="Call the advisory proposal-review LLM; still no production writes")
    ap.add_argument("--confirm-proposal-review", help=f"Required token for --execute-proposal-review-llm: {PROPOSAL_REVIEW_CONFIRM}")
    ap.add_argument("--notify-proposal-review", action="store_true", help="Emit notification block in proposal review output for Hermes/cron delivery")
    ap.add_argument("--max-proposal-review-llm-calls", type=int, default=None)
    ap.add_argument("--top-proposals", type=int, default=80)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--strict-runtime", action="store_true", help="Make preflight fail when live container tuning is not yet 20x3")
    ap.add_argument("--execute", action="store_true", help="Actually execute mutating pipeline steps; requires --confirm run-hindsight-pipeline")
    ap.add_argument("--confirm")
    ap.add_argument("--plan-json", action="store_true", help="Print plan JSON; default also prints JSON when not executing")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plan = build_plan(args)
    if args.plan_json:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.mode == "preflight" and not args.execute:
        report = execute_plan(plan, args)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    report = execute_plan(plan, args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
