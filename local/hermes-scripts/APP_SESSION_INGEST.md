# Scoped app history ingestion (staged, not deployed)

## Authorized local-only persistence

`hindsight_app_local_archive.py` persists the already sealed first batch without
re-fetching history or running model code. It refuses an archive root overlapping
production sessions, profiles, or the production incremental ledger. The archive
has its own `app-archive` ledger, with documents at `captured` only. These are six
local input-document drafts, not six extracted Hindsight memories.

The authorized archive is
`~/.hermes/hindsight/app_archive/scoped-seqscale`: three native snapshots and
447 messages. A repeated run adds zero messages/documents. The first frozen batch
spans 2026-10-14/15, so this task's local reporter writes corresponding sidecars
and appends a marked 2026-10-14 block to the 2026-10-15 canonical daily report,
backing up and preserving its original content. The 2026-10-15 event-day sidecar
is labelled partial coverage; it is not a completed 2026-10-16 canonical report.
No default production source, service, scheduler or model configuration is
changed. The optional daily entry-point patch below remains unapplied because
that tracked production script contains the user's separate uncommitted edits;
future full regeneration must incorporate the preserved local sidecar.

This local processing was explicitly authorized by the parent after the initial
handoff. The first 264-test result remains the original validation; it was not
rerun. Local archive/report idempotence was checked by actually running this
local-only path twice and verifying the saved item IDs, file hashes and prior
daily body. Runtime evidence is in `evidence/local-persistence.json` on the task.

The existing nightly `hindsight_codex_sync.sh` invokes
`hindsight_backfill_codex_rollouts.py`, whose two sources are CLI rollout
directories. App threads returned by the supported `list_threads` / `read_thread`
tools are a separate source. Never guess a filesystem or hidden API for them.

## Supported bridge

`hindsight_app_export.js` runs inside the tool orchestrator (where `tools` is
provided). Call `collectAppHistory({scope, outputDir})` with an explicit thread-ID
map and a private staging directory. It discovers the scoped IDs in the bounded
listing, reads scoped IDs directly even when outside that listing, follows opaque
pagination cursors, and saves only public user/assistant items. Every invocation
refetches the head so newly completed turns can be discovered. It does not save
tool commands, outputs, reasoning, credentials, or authorize additional threads.
The listing is bounded; there is no claim of account-wide enumeration. Check
`reached_oldest` and `resume_cursor` if a chosen page budget is exhausted; increasing
the budget on a subsequent head scan is safe. This is an on-demand tool bridge,
not a newly installed unattended service or scheduler.

An app read is a **history summary view**, not a guaranteed full transcript.
Missing user/delegation bodies and missing exact message timestamps remain
unknown. Completed turns supply stable public item IDs. The adapter defers
in-progress/failed turns and rejects explicitly truncated or rewritten items.
Refetch a truncated item through a supported fuller export before publication;
the adapter does not invent its missing suffix.

`hindsight_app_sessions.py --export-dir STAGING --scope SCOPE.json
--output-sessions TARGET` is a read-only plan by default. `--write` writes only
native `session_codex-app-UUID.json` snapshots. It never calls Hindsight or a model.
An example scope value is
`{"UUID":{"parent_session_id":"PARENT_UUID","relation_basis":"user_supplied_delegation"}}`.
The scope is an authorization input, not inferred from conversation commands.

## Existing pipeline and identity

Only after authorization review, use the existing default-profile session
directory as TARGET. That makes the existing logical/physical arrival scanner
and existing bank-scoped ledger discover the feed without a second ingestion
pipeline. The source key is `default:hermes:codex-app-UUID`; the public native
origin is `codex-app:UUID:TURN:ITEM`. A content change under the same ID fails
closed. Partial later exports union with old snapshots and cannot erase history.
Repeated input creates no new messages or ledger batches.

App parent IDs are `app_parent_session_id`, with an explicit evidence basis.
They are not `parent_session_id`: the latter means Hermes compression ancestry
and would incorrectly merge independently delegated app conversations. Each
message retains source/session/turn/item/parent/time provenance in its input text,
and the existing manifest tagger recognizes `source:codex-app`.

An existing CLI snapshot ending in the same UUID blocks import for that session
until its event identity is reconciled; heuristic content dedup is not used.
Review both default and reviewer exports before approval. This is conservative
overlap rejection, not a completed CLI/app migration merge.

## Calls and daily output

`hindsight_app_usage.aggregate_llm_events` counts only explicit normalized
`llm.request/attempt/retry/success/failure` events, preserving raw references,
scope, and conflicts. A `token_count`, assistant item, tool call, or completed
turn is not such an event. Supported app history currently exposes none of
these telemetry events. Unknown columns remain unknown, including zero-looking
days without telemetry coverage. The normalization API is ready for a separately
verified telemetry source; it does not prove that such a source is available.

`render_app_daily(sessions, "YYYY-MM-DD")` produces an appendable daily block
with source-session totals and item references. The interval is explicitly
UTC+8 `[00:00,next 00:00)`. Message allocation uses turn start/completion as a
labelled approximation, not exact message event times; cross-day turns remain
an attribution limit. The report does not add these counts to the existing
CLI LLM-call total. Existing production `daily_report.py` has uncommitted user
changes, so this branch deliberately leaves that file untouched. The remaining
small integration is to load approved app snapshots and append this block to
its calendar-day report window, after those changes are reconciled by the parent.
`app-daily-integration.patch` is a prepared, unapplied patch against the observed
production file including its existing user changes. It adds an explicit
`--app-sessions` argument and appends the sidecar using the report window's UTC+8
day. Verify the recorded baseline hash and use `git apply --check --unidiff-zero`
before applying with `git apply --unidiff-zero` (the patch omits context lines);
do not restore the branch's older `daily_report.py`. No default cron changes are
included, and the patch has not been applied to production or this branch's old
reporter.
The prepared patch also labels the existing external CLI counter as usage events
and explains that `token_count` does not prove request/attempt/success totals.

## Data flow approval boundary

Local staging is private and outside the production feed. Planned production
targets are `~/.hermes/sessions`, the existing `hermes` incremental ledger,
Hindsight `http://127.0.0.1:8888`, its configured local PostgreSQL database, and
the existing wiki daily report. None is written by the read-only plan.

`pipeline_config.json` declares publication embedding at
`http://127.0.0.1:11434/v1`, `bge-m3` (local Ollama). Its consolidation profile is
`oneapi-auto-cheep`, whose built-in route is `http://127.0.0.1:3000/v1`, model
`auto_cheep_model`. Runtime overrides and the proxy's active upstream routing
must be attested without reading credentials before sending new app data.
The current reporter identifies One API's DB as
`~/.local/share/one-api/data/one-api.db`. A read-only query of non-secret channel
fields finds the enabled priority-100 `codex-subscription-luna-muse` route for
`auto_cheep_model` at `http://172.17.0.1:4458`. Disabled routes also exist and are
not presumed to be processing these data. This configuration is not a per-request
runtime trace: the bridge's remote destination remains unattested. The user-bus
and Docker socket were not accessible in this delegated sandbox; no permission
or service changes were made to inspect them.
The already-generated `2026-10-15.md` daily report's public configuration table
records retain/reflect/consolidation/default at the same local One API URL and
`auto_cheep_model` under `systemd:hindsight`. This is the latest observed runtime
evidence, not a fresh service attestation. Do not label the known local embedding
endpoint as external. Before execution, confirm that runtime overrides and the
bridge's ultimate destination have not changed.

The native retain stage receives cleaned public conversation text plus source
provenance. Offline summary generation then receives newly extracted
world/experience facts with fact/document IDs and timestamps. Publication embeds
the accepted canonical text locally. Formal retain, external processing and wiki
rebuild remain gated on the parent's authorization check of targets, categories,
destinations and counts. No credentials, permissions, proxy settings, cron jobs,
old histories or unrelated SeqScale files are changed by this branch.
