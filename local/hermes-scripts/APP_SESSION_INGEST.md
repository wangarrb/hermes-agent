# App history: supported capture, local archive and daily evidence

## Scope and coverage

The first installed ChatGPT app date is **2026-09-09 UTC+8**, proven by the
same-user package install log and filesystem birth metadata. The inclusive
boundary is `2026-09-09T00:00:00+08:00`. Filter individual event/turn boundary
times, never session creation dates; an older conversation can gain new content.
Dates never own deduplication identity.

The supported current listing returned 50 non-pinned conversations (47 Codex,
3 ordinary ChatGPT); Codex archived pagination returned 1 more. This **51-ID
visible subset is not the full account**: the current listing has no cursor,
and archived ordinary ChatGPT returns an explicit local-desktop-context refusal.
Do not switch interfaces to evade that refusal. The parent conversation can
return no turns. These limitations stay in `collection-status.json` and the daily.
Only same-account visible user/assistant text is preserved. No tool output,
commands, hidden reasoning, credentials, attachment URLs or internal sessions.

`hindsight_app_export.js` needs the supported orchestrator-provided `tools`.
It handles both `codex` and `chatgpt`, requests up to 20,000 output characters
per item, follows opaque cursors, and re-fetches the head each run. It is an
on-demand bridge, **not a Node cron collector**. Always write to a private,
user-owned staging directory and record a receipt for the complete cursor chain.
A returned oldest page alone does not prove a continuous chain or full transcript.

`read_thread` provides a public summary view. Missing user continuation bodies,
exact item timestamps and summarized-away history remain missing even when
pagination ends. Failed/in-progress/interrupted turns stay deferred. Explicit
truncation is separately recorded; it never becomes a complete stable event.
Transport errors resume only from unsealed cursors. Never manufacture message text.

## Local preservation and identity

`hindsight_app_sessions.py --export-dir DIR --scope SCOPE.json
--output-sessions TARGET --since 2026-09-09T00:00:00+08:00 --allow-partial`
is a dry plan unless `--write` is supplied. It makes no network/model calls.
Native snapshot IDs are `codex-app-UUID` / `chatgpt-app-UUID`; origins are
`SOURCE:UUID:TURN:ITEM`. Stable identity owns union/dedup; a rewritten stable
item fails closed. First-observed append order preserves sealed JSON prefixes
even when newly discovered history has older timestamps. Partial snapshots cannot remove previously saved history.

Parent metadata is `app_parent_session_id`, never Hermes compression
`parent_session_id`. Relation evidence comes from an initial public delegation
wrapper or an explicit authorized assertion. Same project/title is no proof.
Public provenance includes session, turn, item, time and parent basis. Timestamps
use labelled turn start/completion fallbacks, not precise item event times.
The existing manifest tagger recognizes both source labels.

`hindsight_app_local_archive.py` reuses the real arrival scanner and ledger,
inside `~/.hermes/hindsight/app_archive/account-visible`. This root is outside
production `sessions`, `profiles` and the production incremental ledger; overlap
is rejected. Local documents remain **captured, not retained memories**. The
independent `app-archive` ledger never runs Hindsight. An oversized public message
is preserved whole locally, using a local document budget large enough for that
message; production/model budgets are unchanged. Re-running the same input adds
zero messages and zero documents. The original frozen three-session archive at
`app_archive/scoped-seqscale` remains separately available.

Compare source UUIDs with existing CLI exports before any production ingestion.
Seventeen currently visible IDs overlap the local CLI/VScode state store; twelve
already have CLI exports. These twelve are labelled `legacy_cli_session_overlap`:
the App view is retained locally but is **not additive to CLI totals** and cannot
be re-imported as a second production source. The old exports lack original item
IDs, so an exact between-view message delta remains unknown. Do not guess it
from text similarity or different clocks. Within App history, audit exact shared
turn/item IDs across sessions; report occurrences and unique shared history
separately and never assume a fork from similar text.

## Daily integration and actual automation status

`hindsight_app_usage` counts only explicit `llm.request`, `llm.attempt`,
`llm.retry`, `llm.success`, `llm.failure` events. Raw references, identities,
retry outcomes, conflicts and UTC+8 boundaries are preserved. App history and
currently correlated logs expose none, so all five call columns remain unknown.
Assistant messages, tool calls, completed turns and token_count events are not
actual request/attempt counts. The historical CLI `calls` field still counts
usage events internally; the production daily column now says **用量事件**.

`hindsight_app_daily_sidecar.py` is deployed as a reader of `account-visible`.
The existing **08:30 `daily_report.py` entry point now invokes it** for the prior
UTC+8 calendar day. Only the small insertion and truthful CLI event labels/doc
were applied to the current dirty production file; all prior uncommitted edits
were preserved and backed up by hash. The isolated worktree's older reporter
was never copied over it. Deployment receipts are outside Git under task evidence.
The exact patch is `patches/daily_report_app_local_integration.patch`; the older
`app-daily-integration.patch` and six-line sidecar-only patch are superseded,
unapplied alternatives and must not be reapplied.

Historical evidence sidecars are generated for observed days. Completed prior
days may append a single idempotent marked block to an existing next-day report,
with a private backup. Today's sidecar stays `.partial.md`; no future canonical
report is invented. CLI-overlap rows are explicitly non-additive. Source item
references permit checking totals without publishing private conversation bodies.

**Automatic App discovery/history fetch is not deployed.** The existing 23:00
script now first consumes App exports locally and then syncs CLI rollouts;
00:01 runs the existing memory pipeline. The deployed consumer was exercised
against a real supported-API export and added zero messages/documents on replay.
A
same-user official local app-server transport was read-only verified, but that
store returned only 52 CLI/VScode threads and excluded the known App samples.
It cannot be presented as a full App collector. A supported App-native task
context/schedule must supply list/read capability and private staging for actual
daily App collection. No new schedule, service, credentials, permissions, proxy
or exporter was created. The daily shows unknown/stale coverage when collection
has not been proved current. A report reader alone is not a daily collector.

## Formal Hindsight processing and approval boundary

External processing was approved for **only three frozen SeqScale conversations,
447 cleaned public messages**. `hindsight_app_approved_run.py` validates their
exact proof, refuses unrelated pending work, writes only these three snapshots
to `~/.hermes/sessions`, and invokes the existing native arrival pipeline with
its original integrity guards and publication gates. Its first execution was
blocked by `SourceChangedError` **before capture**: zero batches, network calls
and offline LLM calls. The confirmed cause is an unrelated physical replacement of `~/.hermes/state.db`
(inode 41059815 → 41059857; SQLite schema cookie 163 → 208). All 1508 baseline
session identities and all 277999 baseline row hashes were proved unchanged in
one read-only snapshot; the 17 default-policy unknown rows also match. There
are 7278 later SQL rows, which must remain future. The original CLI has no epoch
migration command. A new opt-in exact header-transition candidate preserves all
original row/session/unknown/profile/JSON checks and the original feed epoch.
It requires a certificate and independently reviewed SHA. Its 199 targeted tests
passed; the real certificate remains DRAFT pending the parent's independent pin.
No baseline, gate or source unknown policy is weakened. Local archives are not memory
extraction, and retained/submitted/published are separate receipt states.

The accessible archive currently contains 50 sessions (47 Codex,3 ordinary
ChatGPT), 10366 cleaned source occurrences /7602 unique App items and154 captured
local documents. All pagination/coverage limitations remain explicit. Re-running
the whole archive adds zero messages/documents. A single newest-turn refresh in
an existing session added5 completed messages, then zero on replay; those5 are
local-only and do not expand the frozen447 external scope. The real source-level
raw view has10799 occurrences /7800 unique items, with2999 shared occurrences;
cleaning/deferred-turn selection explains the different normalized counts.

Fresh safe-field attestation proved native retain/consolidate/reflect/default
and offline summary use `auto_cheep_model` via `http://127.0.0.1:3000/v1`.
The enabled One API route reaches `http://172.17.0.1:4458`, which selects among:

- LongCat `longcat-2.5-preview-free`: `https://opencode.ai/zen/go/v1/chat/completions`;
- Muse `muse-spark-1.3-contributor`: `https://opencode.ai/zen/go/v1/responses`;
- Luna `gpt-6-luna`: `https://chatgpt.com/backend-api/codex/responses`.

No failure cross-model or paid fallback was enabled. Publication embedding is
**local** Ollama-compatible `http://127.0.0.1:11434/v1`, `bge-m3`; the provider
name `openai` describes the protocol and is not evidence of external embedding.
No credential values are read into reports or logs.

Retain sends cleaned public text plus provenance; offline summary sends newly
extracted world/experience facts with IDs/time; local publication embedding gets
accepted canonical text. Other discovered conversations, including ordinary or
sensitive discussions, remain local-only. Before expanding external extraction,
the parent must approve the concrete session/message inventory, categories,
ultimate destinations and purposes. A successful operation is not a measured
LLM attempt; unknown native request/retry telemetry remains unknown.
