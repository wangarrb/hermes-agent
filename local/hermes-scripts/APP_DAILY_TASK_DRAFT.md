# Proposed App collection task — not created

Proposed schedule: daily **22:45 Asia/Shanghai (UTC+8)**, ahead of the existing
23:00 local consumer/CLI sync, 00:01 Hindsight pipeline and 08:30 report.
This is a proposed new App-native task. No existing system cron, service,
credentials, account, permissions or proxy will be changed. Approval is required
under the instruction not to create/change scheduling configuration.

Run in this same authenticated App context and local user workspace. First check
access to a previously authorized sentinel thread; if refused, stop without
listing another account, refreshing credentials or changing endpoints. Use only
supported user-visible list/read tools. Never request archived ordinary ChatGPT
again from the context that refused it; preserve that explicit coverage gap.

Load the reviewed hindsight_app_export.js in the supported tool orchestrator.
Call discoverVisibleAppScope with the prior account-visible scope, preserving
known authorized IDs and adding only newly user-visible IDs in this same context.
Save the scope privately. Call collectAppHistory to project public user/assistant
text to ~/.hermes/hindsight/app_archive/account-visible/exports (directory700,
files600). Preserve cursor receipts, truncation, failed/in-progress turns and
source-kind metadata. Refetch heads and save versioned pages. Return only counts,
IDs and coverage; never emit private body text in the assistant output.

Run hindsight_app_ingest_pending.py --write after capture. The existing 23:00
script will also consume exports idempotently once its reviewed code patch is
deployed. Keep the production external-processing scope at the original frozen
three sessions/447 messages. All other messages, including new turns in those
sessions, stay in the independent local archive. Do not call Hindsight, embedding,
extraction, summary or consolidation for expanded history, and never write its
snapshots to production source directories.

Keep successful API capture time separate from local consumption time. A bounded
50-item current listing is not the full account. Record missing/archive-denied,
truncated and active states separately. Resume transient failures at unsealed
cursors; preserve old pages. Page-budget exhaustion is pending, not complete.
LLM request totals stay unknown without explicit request telemetry.

The discovery/export code was runtime-verified using supported tools: 51 visible
IDs (48 Codex including1 archived,3 ordinaryChatGPT), plus a known sample public
history. It needs orchestrator-provided tools and cannot execute as Node inside
shell cron. No App-native task has been created. The scheduled task's tool/file
access still requires runtime verification after approval.
