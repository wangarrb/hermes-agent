# Hindsight full pipeline watchdog and session manifest ID lessons

Context: a production-adjacent full Hindsight pipeline run may spend many hours in native observation/consolidation work. Short outer-process timeouts can exit while Hindsight is still making progress.

## Durable session document IDs

When building `session_*.json` manifests, prefer the filename stem as the durable unique session/document ID.

Reason: some historical session JSON files can contain a stale embedded `session_id`. If the manifest trusts that embedded field, multiple files may collapse to the same `document_id`, causing duplicate-document/batch retain failures even though the filenames are unique.

Safe rule:

```text
session file: session_<durable-id>.json
manifest document_id/session_key: Path(file).stem
embedded session_id: metadata only; do not use as uniqueness authority
```

Regression test shape:

1. Create two `session_*.json` files with distinct filenames but identical/stale embedded `session_id`.
2. Build the manifest.
3. Assert exported document IDs are the two filename stems and are unique.

## Long retain / observation waits

Native observation creation can legitimately run for many hours after session retain submission. For full or broad incremental production runs, use a wait budget up to 86400 seconds rather than treating a short client timeout as failure.

Do not restart Hindsight just because the foreground pipeline timed out. First check operations/status/logs and whether observation or consolidation counters are still advancing.

## Watchdog pattern for full pipeline

For long full runs, wrap the outer pipeline with a watchdog that:

1. Writes logs into the pipeline run directory.
2. Waits for Hindsight to become idle if the foreground process exits due to timeout while native work is still active.
3. Reconciles/updates submit state after idle so already-submitted session retains are not submitted again blindly.
4. Re-runs the full pipeline from the next safe stage.
5. Runs post-status and recall smoke after completion.

This pattern is safer than killing/restarting the Hindsight service and avoids losing progress from long native processing.

## Reporting language

Report these states distinctly:

- `foreground pipeline process exited/timed out`
- `Hindsight native operations still active and advancing`
- `watchdog waiting for idle`
- `submit state reconciled`
- `pipeline resumed`
- `post-verify passed/failed`

Do not call the run complete until post-run status, recall smoke, conflict/proposal artifacts, and queue/drain state are verified.
