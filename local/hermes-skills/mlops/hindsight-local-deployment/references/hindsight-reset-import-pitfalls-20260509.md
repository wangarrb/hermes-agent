# Hindsight Reset / Session-Import Pitfalls (2026-05-09)

Session-derived traps encountered during a prod reset + earliest-week session/json retain.

## 1. Do not write one-off scripts for standard flows

The umbrella skill and its references already document the complete workflow.
Use the existing scripts:

- `hindsight_session_manifest.py` — generate manifest (--bank-target hermes, --json)
- `hindsight_minimax_import.py session-manifest-retain-llm` — execute paid retain with provider switching, queue drain, and normal-local restoration
- `hindsight_session_retain_runner.py` — lower-level dry-run / execute retain

**Do NOT write custom scripts** like `*_retain_a_group_production.py` or `run_week_retain.py`
unless the existing tools genuinely cannot express the needed flow. Even then, prefer
patching the existing tool over a one-off script.

## 2. Docker service: old systemd journal ≠ broken service

The hindsight service runs in a **Docker container** managed by systemd.
Systemd journal may show ancient errors (e.g., `ImportError: sentence-transformers`)
from previous restarts — the Docker container has all dependencies pre-installed.

**Always verify with `curl /health` first.** If it returns `healthy`, the service is fine.
Do not initiate pip install or dependency repair based on journal errors alone.

## 3. CCH debugging: always load the skill first

When CCH returns 503 or auth errors, load `hermes-operations-debugging` and consult
the dedicated `references/cch-debugging.md` before attempting manual curl diagnosis.

The user explicitly corrected: "不是有cch的skill吗？" — the skill already existed and
documents the `format_type_mismatch` / `no_available_providers` / `service_unavailable`
error taxonomy, the required `User-Agent: openai-codex/0.121.0` header, and the
admin panel at `http://cch.jmadas.com/zh-CN/login`.

## 4. Reset + session-import runbook is the authoritative reference

The one-week trial runbook at `references/hindsight-session-json-reset-one-week-trial.md`
is the canonical workflow. Key steps:

- Generate manifest → filter week → dry-run → backup → reset DB → execute paid retain → audit
- Use `--execute --confirm retain-hindsight-session-manifest` for real runs
- Fresh submit-state file for every reset trial
- `enable_observations=false` during retain; native consolidation only after facts audit passes
- completed operations ≠ facts produced; check `docs_without_units`
- Paid MiniMax retain of 83 early-week records takes ~1-2 hours; monitor with stats polling

## 5. Manifest already exists for common windows

Before generating a new manifest, check `$HOME/.hermes/hindsight/runs/` for
pre-existing filtered manifests (e.g., `week-04-09-04-16-production-*.jsonl`).

## 6. PostgreSQL access: use conda hindsight Python

The host Python lacks `psycopg2`. Always use the conda environment's Python for DB operations:
```bash
$HOME/miniconda/envs/hindsight/bin/python -c "import psycopg2; ..."
```
Or write scripts with shebang `#!$HOME/miniconda/envs/hindsight/bin/python`.
