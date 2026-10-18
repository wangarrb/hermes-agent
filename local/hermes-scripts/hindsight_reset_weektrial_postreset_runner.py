#!/usr/bin/env python3
"""Post-reset one-week Hindsight session/json trial runner.

Assumes destructive DB reset + migrations already completed and the filtered
production manifest already exists. Runs paid session retain, restores local via
wrapper, audits, and optionally runs a bounded native consolidation only if hard
gates pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REDACT_RE = re.compile(r"sk-[A-Za-z0-9_.-]+|(api[_-]?key=)[^\s]+", re.IGNORECASE)


def redact(text: str) -> str:
    return REDACT_RE.sub(lambda m: (m.group(1) + "[REDACTED]") if m.lastindex else "[REDACTED]", text)


def run_capture(cmd: list[str], *, log_path: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat()}] RUN: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        combined += "\n[stderr]\n" + proc.stderr
    log_path.write_text(redact(combined), encoding="utf-8")
    print(f"[{datetime.now().isoformat()}] EXIT {proc.returncode}: {' '.join(cmd[:3])}", flush=True)
    if check and proc.returncode != 0:
        tail = "\n".join(redact(combined).splitlines()[-80:])
        print(tail, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return proc


def run_stream(cmd: list[str], *, log_path: Path, check: bool = True) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat()}] STREAM RUN: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started_at={datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"# cmd={' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            rline = redact(line)
            log.write(rline)
            log.flush()
            # Keep parent output low-volume but preserve important milestones.
            if any(key in rline for key in [
                "queue:", "queue drained", "created/verified bank", "patched bank config",
                "restoring normal-local", "mode=normal-local", "submitted_items",
                "failed_operations", "ERROR", "429", "STUCK", "JSON parse error",
            ]):
                print(rline.rstrip(), flush=True)
        rc = proc.wait()
        log.write(f"# finished_at={datetime.now(timezone.utc).isoformat()} rc={rc}\n")
    print(f"[{datetime.now().isoformat()}] STREAM EXIT {rc}: {' '.join(cmd[:3])}", flush=True)
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    a = audit.get("audit", {})
    counts = a.get("counts", {})
    lineage = a.get("lineage", {})
    tag_quality = a.get("tag_quality", {})
    ops = a.get("operations", {})
    contamination = a.get("contamination_counts", {}) or {}
    by_status = ops.get("by_status", {}) or {}
    summary = {
        "documents": counts.get("documents"),
        "memory_units": counts.get("memory_units"),
        "observations": counts.get("observations"),
        "operations_by_status": by_status,
        "docs_without_units": lineage.get("docs_without_units"),
        "units_missing_document": lineage.get("units_missing_document"),
        "broad_system_tag_total": tag_quality.get("broad_system_tag_total"),
        "contamination_counts": contamination,
        "contamination_total": sum(int(v or 0) for v in contamination.values()),
        "warnings": audit.get("warnings", []),
    }
    summary["hard_gate_pass"] = (
        int(by_status.get("failed", 0) or 0) == 0
        and int(summary.get("docs_without_units") or 0) == 0
        and int(summary.get("units_missing_document") or 0) == 0
        and int(summary.get("broad_system_tag_total") or 0) == 0
        and int(summary.get("contamination_total") or 0) == 0
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--submit-state", required=True, type=Path)
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--bank", default="hermes")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--wait-timeout-s", type=int, default=21600)
    ap.add_argument("--health-timeout-s", type=int, default=600)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--run-consolidation", action="store_true")
    args = ap.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "bank": args.bank,
        "manifest": str(args.manifest),
        "submit_state": str(args.submit_state),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": [],
    }

    try:
        # Baseline after reset.
        run_capture(["python3", "/home/wyr/.hermes/scripts/hindsight_minimax_import.py", "status"], log_path=args.run_root / "09-status-before-paid-retain.txt")
        summary["stages"].append("status_before_paid")

        # Paid retain through wrapper. Wrapper itself switches to paid provider and restores normal-local in finally.
        retain_cmd = [
            "python3", "/home/wyr/.hermes/scripts/hindsight_minimax_import.py", "session-manifest-retain-llm",
            "--manifest", str(args.manifest),
            "--bank", args.bank,
            "--batch-size", str(args.batch_size),
            "--submit-state", str(args.submit_state),
            "--execute", "--confirm", "retain-hindsight-session-manifest",
            "--health-timeout-s", str(args.health_timeout_s),
            "--timeout", "0",
            "--poll", str(args.poll),
            "--wait-timeout-s", str(args.wait_timeout_s),
            "--poll-s", str(args.poll_s),
        ]
        rc = run_stream(retain_cmd, log_path=args.run_root / "10-paid-retain.log", check=False)
        summary["retain_returncode"] = rc
        summary["stages"].append("paid_retain_finished")

        # Always capture status after retain attempt.
        run_capture(["python3", "/home/wyr/.hermes/scripts/hindsight_minimax_import.py", "status"], log_path=args.run_root / "11-status-after-paid-retain.txt", check=False)
        summary["stages"].append("status_after_paid")
        if rc != 0:
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            summary["status"] = "retain_failed_or_restore_failed"
            write_json(args.run_root / "summary.json", summary)
            return rc

        # Facts + recall smoke audit.
        audit_stem = f"{args.run_id}-retain-audit"
        audit_proc = run_capture([
            "python3", "/home/wyr/.hermes/scripts/hindsight_bank_quality_audit.py",
            "--bank", args.bank,
            "--recall-smoke",
            "--stem", audit_stem,
            "--json",
        ], log_path=args.run_root / "12-retain-audit.json", check=False)
        summary["stages"].append("retain_audit_finished")
        if audit_proc.returncode == 0 and audit_proc.stdout.strip():
            audit = json.loads(audit_proc.stdout)
            write_json(args.run_root / "12-retain-audit-parsed.json", audit)
            summary["retain_audit_summary"] = audit_summary(audit)
        else:
            summary["retain_audit_error"] = redact((audit_proc.stdout or "") + "\n" + (audit_proc.stderr or ""))[-4000:]
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            summary["status"] = "audit_failed"
            write_json(args.run_root / "summary.json", summary)
            return audit_proc.returncode or 1

        hard_gate = bool(summary.get("retain_audit_summary", {}).get("hard_gate_pass"))
        if not args.run_consolidation:
            summary["consolidation"] = "skipped_by_flag"
        elif not hard_gate:
            summary["consolidation"] = "skipped_by_hard_gate"
        else:
            # Bounded native consolidation.
            pre = run_capture([
                "python3", "/home/wyr/.hermes/scripts/hindsight_native_workflow_guard.py", "preflight",
                "--bank", args.bank,
                "--max-unconsolidated", "350",
                "--require-observations-disabled",
            ], log_path=args.run_root / "13-native-consolidation-preflight.txt", check=False)
            summary["native_preflight_returncode"] = pre.returncode
            if pre.returncode == 0:
                cons_rc = run_stream([
                    "python3", "/home/wyr/.hermes/scripts/hindsight_native_workflow_guard.py", "run-native-consolidation-paid",
                    "--bank", args.bank,
                    "--jobs", "1",
                    "--facts-per-job", "50",
                    "--fetch-size", "50",
                    "--llm-batch-size", "50",
                    "--max-unconsolidated", "350",
                    "--execute", "--confirm", "run-native-paid-consolidation",
                    "--operation-timeout", "7200",
                    "--poll", "30",
                ], log_path=args.run_root / "14-native-consolidation-paid.log", check=False)
                summary["native_consolidation_returncode"] = cons_rc
                summary["consolidation"] = "completed" if cons_rc == 0 else "failed"
                run_capture(["python3", "/home/wyr/.hermes/scripts/hindsight_minimax_import.py", "status"], log_path=args.run_root / "15-status-after-consolidation.txt", check=False)
                post_stem = f"{args.run_id}-post-consolidation-audit"
                post = run_capture([
                    "python3", "/home/wyr/.hermes/scripts/hindsight_bank_quality_audit.py",
                    "--bank", args.bank,
                    "--recall-smoke",
                    "--stem", post_stem,
                    "--json",
                ], log_path=args.run_root / "16-post-consolidation-audit.json", check=False)
                if post.returncode == 0 and post.stdout.strip():
                    post_audit = json.loads(post.stdout)
                    write_json(args.run_root / "16-post-consolidation-audit-parsed.json", post_audit)
                    summary["post_consolidation_audit_summary"] = audit_summary(post_audit)
            else:
                summary["consolidation"] = "skipped_by_preflight"

        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["status"] = "completed"
        write_json(args.run_root / "summary.json", summary)
        md = [
            f"# {args.run_id}",
            "",
            f"bank: `{args.bank}`",
            f"manifest: `{args.manifest}`",
            f"status: `{summary['status']}`",
            f"retain_returncode: `{summary.get('retain_returncode')}`",
            f"consolidation: `{summary.get('consolidation')}`",
            "",
            "## retain audit summary",
            "```json",
            json.dumps(summary.get("retain_audit_summary"), ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
        if "post_consolidation_audit_summary" in summary:
            md += ["", "## post consolidation audit summary", "```json", json.dumps(summary["post_consolidation_audit_summary"], ensure_ascii=False, indent=2, sort_keys=True), "```"]
        (args.run_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        return 0
    except Exception as e:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["status"] = "exception"
        summary["exception"] = redact(repr(e))
        write_json(args.run_root / "summary.json", summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
