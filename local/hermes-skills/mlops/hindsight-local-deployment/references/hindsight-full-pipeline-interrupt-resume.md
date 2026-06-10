# Hindsight full pipeline interruption and resume notes

Use when a full/offline Hindsight pipeline or watchdog exits with SIGTERM / rc=-15, or when a foreground orchestrator is stopped while native operations may have progressed.

## Diagnosis pattern

1. Treat `rc=-15` as SIGTERM until proven otherwise. It often means an external stop/kill, not an application failure.
2. Inspect the runroot logs before rerunning:
   - `watchdog.log`
   - `watchdog-rerun-*.log`
   - `run-command.txt`
   - `status-after-primary-exit.json`
3. Locate the last completed step and the first started-but-unfinished step. In the 2026-05-13 case:
   - preflight passed;
   - session manifest built;
   - session retain submitted nothing new because all production items were `unchanged`;
   - daily reflect for 2026-05-13 posted 3 async docs and queue drained;
   - rerun was SIGTERM'd immediately after starting `hindsight_offline_v2_rebuild.py --mode publish`.
4. Check current runtime read-only before deciding on resume:
   ```bash
   python3 $HERMES_HOME/scripts/hindsight_consolidation_status.py --skip-psql --json > /tmp/hindsight-status.json
   python3 - <<'PY' /tmp/hindsight-status.json
   import json,sys
   p=json.load(open(sys.argv[1]))
   ops=((p.get('checks') or {}).get('operations_api') or {}).get('summary') or {}
   stats=(((p.get('checks') or {}).get('bank_stats') or {}).get('data') or {})
   print({'active':ops.get('active_count'),'pending':ops.get('pending_count'),'failed':ops.get('failed_count'),'pending_consolidation':stats.get('pending_consolidation')})
   PY
   ```
5. Run only non-mutating validation for the interrupted step first. For V2:
   ```bash
   python3 $HERMES_HOME/scripts/hindsight_offline_v2_rebuild.py --mode local --bank hermes --json
   ```

## Resume rule

Do not blindly rerun full from the beginning after an interrupted full pipeline. Use the last completed step to resume from the next safe stage.

Preferred resume behavior for future orchestrator improvements:

- add `--start-at <step>` and/or `--skip-daily` / `--skip-weekly` flags to `hindsight_memory_pipeline.py`;
- make resume reuse existing session submit state and offline reflect progress;
- do not repeat daily reflect if the progress file already contains the same daily unit document IDs;
- for production mutations (`v2_rebuild --mode publish`, proposal LLM review, repair/merge), require the original confirm token and explicit human approval.

## Duplicate prevention checks

Before rerunning daily/weekly reflect, inspect progress:

```bash
python3 - <<'PY'
import json, pathlib
p=pathlib.Path.home()/'.hermes/hindsight/offline_reflect/offline_reflect_progress.json'
data=json.loads(p.read_text())
for x in data.get('processed_document_ids', []):
    if '2026-05-13' in x:
        print(x)
PY
```

If the relevant document IDs are already present and Hindsight operations are idle, prefer resuming after that stage rather than force-reposting.

## Reporting language

Report separately:

- process interrupted by SIGTERM / external stop;
- Hindsight native operations active vs idle;
- which pipeline stages completed;
- which stage is safe to resume;
- which next command is mutating and requires confirmation.
