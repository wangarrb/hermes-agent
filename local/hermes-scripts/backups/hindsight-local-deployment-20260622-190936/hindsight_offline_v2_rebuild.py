#!/usr/bin/env python3
"""Run the full Offline Hindsight V2 rebuild pipeline.

Default full rebuild (no raw retain):
1. Rebuild local v2 canonical cards from existing offline daily/weekly JSON.
2. Run baseline vs local-card eval on generic + local benchmarks.
3. Run fail-closed publish gate and emit proposal.
4. Publish canonical cards to Hindsight DB as fact_type=observation when gate passes.
5. Run audit/status verification.

No raw SQLite retain is performed by this script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
SCRIPT_DIR = HERMES_HOME / "scripts"
DEFAULT_OUTPUT_DIR = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_rebuild"
DEFAULT_CARDS_ROOT = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_cards"
DEFAULT_GENERIC_BENCH = HERMES_HOME / "hindsight" / "eval" / "benchmark_queries.jsonl"
DEFAULT_LOCAL_BENCH = HERMES_HOME / "hindsight" / "eval" / "benchmark_queries.local.jsonl"


def run_json(cmd: list[str], *, cwd: Path | None = None, timeout: int = 3600) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(cwd) if cwd else None, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            + " ".join(cmd)
            + f"\nexit={proc.returncode}\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-8000:]}"
        )
    text = proc.stdout.strip()
    try:
        return json.loads(text)
    except Exception:
        # Some scripts may print warnings before JSON. Prefer the last JSON object.
        start = text.rfind("\n{")
        if start >= 0:
            return json.loads(text[start + 1 :])
        raise RuntimeError(f"command did not return JSON: {' '.join(cmd)}\nSTDOUT={proc.stdout[-4000:]}\nSTDERR={proc.stderr[-4000:]}")


def run_text(cmd: list[str], *, timeout: int = 3600) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            + " ".join(cmd)
            + f"\nexit={proc.returncode}\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-8000:]}"
        )
    return proc.stdout


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def eval_pair(label: str, bench: Path, cards_root: Path, out_dir: Path, *, api: str, bank: str, raw_limit: int, limit: int, top_k: int, include_hindsight_observations: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    common = [
        sys.executable,
        str(SCRIPT_DIR / "hindsight_eval.py"),
        "--benchmark",
        str(bench),
        "--api",
        api,
        "--bank",
        bank,
        "--raw-limit",
        str(raw_limit),
        "--limit",
        str(limit),
        "--top-k",
        str(top_k),
        "--cards-root",
        str(cards_root),
        "--json",
    ]
    baseline_cmd = common + ["--no-local-cards", "--output-dir", str(out_dir / f"{label}-baseline")]
    cards_cmd = common + ["--use-local-cards", "--output-dir", str(out_dir / f"{label}-cards")]
    if not include_hindsight_observations:
        baseline_cmd.append("--no-hindsight-observations")
        cards_cmd.append("--no-hindsight-observations")
    baseline = run_json(baseline_cmd, timeout=1800)
    cards = run_json(cards_cmd, timeout=1800)
    return baseline, cards


def publish_safety_errors(args: argparse.Namespace) -> list[str]:
    if args.mode != "publish":
        return []
    errors: list[str] = []
    if args.confirm_publish != "publish-hindsight-v2-canonical":
        errors.append("publish requires --confirm-publish publish-hindsight-v2-canonical")
    if args.skip_conflict_audit:
        errors.append("publish with --skip-conflict-audit is unsafe and blocked")
    if args.skip_eval_gate:
        errors.append("publish with --skip-eval-gate is unsafe and blocked")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full Offline Hindsight V2 rebuild pipeline")
    ap.add_argument("--mode", choices=["dry-run", "local", "publish"], default="local")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--cards-root", default=str(DEFAULT_CARDS_ROOT))
    ap.add_argument("--api", default="http://127.0.0.1:8888")
    ap.add_argument("--bank", default="hermes")
    ap.add_argument("--generic-benchmark", default=str(DEFAULT_GENERIC_BENCH))
    ap.add_argument("--local-benchmark", default=str(DEFAULT_LOCAL_BENCH))
    ap.add_argument("--raw-limit", type=int, default=40)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-topics", type=int, default=30)
    ap.add_argument("--max-observations-per-card", type=int, default=120)
    ap.add_argument("--embedding-provider", choices=["auto", "docker", "none"], default="auto")
    ap.add_argument("--skip-eval-gate", action="store_true", help="Publish without eval gate; not recommended")
    ap.add_argument("--allow-gate-blocked-publish", action="store_true", help="Still publish when the gate is blocked; for forced rebuilds only")
    ap.add_argument("--include-hindsight-observations-in-gate", action="store_true", help="Gate baseline includes already-published Hindsight observations; default excludes them for raw baseline isolation")
    ap.add_argument("--skip-conflict-audit", action="store_true", help="Skip conflict/raw-lineage audit; publish remains blocked because this makes the run unsafe")
    ap.add_argument("--confirm-publish", help="Required exact confirmation string for publish mode: publish-hindsight-v2-canonical")
    ap.add_argument("--conflict-block-severity", choices=["P0", "P1", "P2", "P3"], default="P1")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    safety_errors = publish_safety_errors(args)
    if safety_errors:
        report = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "decision": "blocked_unsafe_publish_args",
            "published": False,
            "errors": [{"step": "preflight", "error": e} for e in safety_errors],
            "steps": {"preflight": {"skipped_side_effects": True, "safety_errors": safety_errors}},
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print("Offline Hindsight V2 rebuild blocked before side effects")
            for e in safety_errors:
                print(f"- {e}")
        raise SystemExit(2)

    started = datetime.now().isoformat(timespec="seconds")
    out_dir = Path(args.output_dir).expanduser()
    cards_root = Path(args.cards_root).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "started_at": started,
        "finished_at": None,
        "mode": args.mode,
        "output_dir": str(out_dir),
        "cards_root": str(cards_root),
        "steps": {},
        "decision": None,
        "published": False,
        "errors": [],
        "safety": "No raw SQLite retain; rebuild uses existing offline facts/summaries and direct canonical observation publish.",
    }

    try:
        reduce_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "hindsight_offline_v2_reduce.py"),
            "--mode",
            "local" if args.mode in {"local", "publish"} else "dry-run",
            "--scope",
            "all",
            "--output-root",
            str(cards_root),
            "--max-topics",
            str(args.max_topics),
            "--max-observations-per-card",
            str(args.max_observations_per_card),
            "--json",
        ]
        reduce_report = run_json(reduce_cmd, timeout=600)
        report["steps"]["reduce"] = reduce_report

        conflict_report = None
        conflict_pass = True
        if not args.skip_conflict_audit:
            conflict_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "hindsight_conflict_audit.py"),
                "--cards-root",
                str(cards_root),
                "--offline-root",
                str(HERMES_HOME / "hindsight" / "offline_reflect"),
                "--output-dir",
                str(out_dir / "conflict_audit"),
                "--bank",
                args.bank,
                "--block-severity",
                args.conflict_block_severity,
                "--json",
            ]
            conflict_report = run_json(conflict_cmd, timeout=300)
            report["steps"]["conflict_audit"] = conflict_report
            conflict_pass = conflict_report.get("decision") == "pass"
        else:
            conflict_pass = False
            report["steps"]["conflict_audit"] = {"skipped": True, "unsafe_for_publish": True, "reason": "--skip-conflict-audit"}

        gate_report = None
        if not args.skip_eval_gate:
            eval_root = out_dir / "eval"
            pairs = []
            for label, bench in [("generic", Path(args.generic_benchmark).expanduser()), ("local", Path(args.local_benchmark).expanduser())]:
                if not bench.exists():
                    report["errors"].append({"step": "eval", "label": label, "error": f"benchmark missing: {bench}"})
                    continue
                baseline, cards = eval_pair(
                    label,
                    bench,
                    cards_root,
                    eval_root,
                    api=args.api,
                    bank=args.bank,
                    raw_limit=args.raw_limit,
                    limit=args.limit,
                    top_k=args.top_k,
                    include_hindsight_observations=args.include_hindsight_observations_in_gate,
                )
                pairs.append((label, baseline, cards))
            report["steps"]["eval"] = {
                label: {"baseline": b.get("json_path"), "cards": c.get("json_path"), "baseline_summary": b.get("summary"), "cards_summary": c.get("summary")}
                for label, b, c in pairs
            }
            gate_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "hindsight_offline_v2_gate.py"),
                "--cards-root",
                str(cards_root),
                "--output-dir",
                str(out_dir / "gate"),
                "--emit-proposal",
                "--min-layer-hit-delta",
                "0",
                "--max-case-term-regressions",
                "2",
                "--case-regression-tolerance",
                "0.05",
            ]
            if conflict_report and conflict_report.get("json_path"):
                gate_cmd.extend([
                    "--conflict-audit-json",
                    str(conflict_report["json_path"]),
                    "--require-conflict-audit",
                    "--conflict-block-severity",
                    args.conflict_block_severity,
                ])
            elif not args.skip_conflict_audit:
                gate_cmd.append("--require-conflict-audit")
            gate_cmd.append("--json")
            for label, baseline, cards in pairs:
                gate_cmd.extend(["--pair", label, str(baseline["json_path"]), str(cards["json_path"])])
            gate_report = run_json(gate_cmd, timeout=300)
            report["steps"]["gate"] = gate_report
            report["decision"] = gate_report.get("decision")
        else:
            report["decision"] = "gate_skipped"

        publish_confirmed = args.confirm_publish == "publish-hindsight-v2-canonical"
        gate_allows_publish = (not args.skip_eval_gate and report.get("decision") == "eligible_for_local_proposal")
        if args.allow_gate_blocked_publish and publish_confirmed and not args.skip_eval_gate:
            gate_allows_publish = True
        should_publish = args.mode == "publish" and publish_confirmed and conflict_pass and gate_allows_publish
        if should_publish:
            publish_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "hindsight_offline_v2_publish.py"),
                "--cards-root",
                str(cards_root),
                "--bank",
                args.bank,
                "--mode",
                "publish",
                "--replace",
                "--embedding-provider",
                args.embedding_provider,
                "--json",
            ]
            publish_report = run_json(publish_cmd, timeout=1800)
            report["steps"]["publish"] = publish_report
            report["published"] = publish_report.get("inserted_observations", 0) > 0
        else:
            report["steps"]["publish"] = {
                "skipped": True,
                "reason": "mode_not_publish_or_publish_not_confirmed_or_gate_blocked_or_conflict_audit_blocked",
                "decision": report.get("decision"),
                "publish_confirmed": publish_confirmed,
                "conflict_audit_passed": conflict_pass,
                "gate_allows_publish": gate_allows_publish,
                "skip_eval_gate": args.skip_eval_gate,
            }

        # Post-publish verification uses default V2 recall, including Hindsight observations and local cards.
        audit_cmd = [sys.executable, str(SCRIPT_DIR / "hindsight_offline_v2_audit.py"), "--api", args.api, "--bank", args.bank, "--json"]
        try:
            report["steps"]["audit"] = run_json(audit_cmd, timeout=300)
        except Exception as e:
            report["errors"].append({"step": "audit", "error": repr(e)})

        try:
            status = run_text([sys.executable, str(SCRIPT_DIR / "hindsight_minimax_import.py"), "status"], timeout=120)
            report["steps"]["status_text"] = status
        except Exception as e:
            report["errors"].append({"step": "status", "error": repr(e)})
    except Exception as e:
        report["errors"].append({"step": "fatal", "error": repr(e)})
        report["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(out_dir / "latest.json", report)
        write_json(out_dir / f"v2-rebuild-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json", report)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_json(out_dir / "latest.json", report)
    write_json(out_dir / f"v2-rebuild-{ts}.json", report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Offline Hindsight V2 rebuild")
        print(f"decision: {report.get('decision')}")
        print(f"published: {report.get('published')}")
        reduce_step = report.get("steps", {}).get("reduce", {})
        print(f"cards: {reduce_step.get('card_count')} observations: {reduce_step.get('collected_observations')}")
        pub = report.get("steps", {}).get("publish", {})
        if pub:
            print(f"inserted_observations: {pub.get('inserted_observations')} backup: {pub.get('backup_path')}")
        print(f"summary_json: {out_dir / 'latest.json'}")


if __name__ == "__main__":
    main()
