---
name: kanban-worker
description: Pitfalls, examples, and edge cases for Hermes Kanban workers. The lifecycle itself is auto-injected into every worker's system prompt as KANBAN_GUIDANCE (from agent/prompt_builder.py); this skill is what you load when you want deeper detail on specific scenarios.
version: 2.0.0
metadata:
  hermes:
    tags: [kanban, multi-agent, collaboration, workflow, pitfalls]
    related_skills: [kanban-orchestrator, upgrading-deepseek-tui, upgrading-reasonix, upgrading-codewhale]
---

# Kanban Worker — Pitfalls and Examples

> You're seeing this skill because the Hermes Kanban dispatcher spawned you as a worker with `--skills kanban-worker` — it's loaded automatically for every dispatched worker. The **lifecycle** (6 steps: orient → work → heartbeat → block/complete) also lives in the `KANBAN_GUIDANCE` block that's auto-injected into your system prompt. This skill is the deeper detail: good handoff shapes, retry diagnostics, edge cases.

## Task lookup — your task may live on a named board

When you're dispatched as a kanban worker, your task ID (`$HERMES_KANBAN_TASK`) comes from the dispatcher but the task itself may live on a **named board** DB like `kanban/boards/<slug>/kanban.db`, not in the default `kanban.db`. The default board query can silently return empty rows without error.

**First thing upon claiming a task:** verify the task exists and has the expected status:

```bash
# Quick check across ALL kanban databases
for db in $(find ~/.hermes -name "kanban.db" -not -path "*/cache/*"); do
    result=$(sqlite3 "$db" "select id, status from tasks where id='$HERMES_KANBAN_TASK';" 2>/dev/null)
    if [ -n "$result" ]; then
        echo "FOUND in $db: $result"
        break
    fi
done
echo "--- board slug: $(basename $(dirname $(dirname $db)))"
```

If the task shows `done`, `blocked`, or `archived`, you shouldn't be running — `kanban_show` first is the reliable check (per earlier guidance), but direct SQL queries are a fast fallback when the CLI is unavailable or the board isn't the default.

**Important:** Your `$HOME` may be overridden to the Hermes profile home (e.g. `/home/wyr/.hermes/profiles/implementer/home`). The actual home directory for user files lives under `getent passwd <user> | cut -d: -f6`. Use the actual `$HOME` from `getent` when locating hermes kanban DBs:

```bash
REAL_HOME=$(getent passwd $(whoami) | cut -d: -f6)
# Then: find "$REAL_HOME/.hermes" -name "kanban.db" ...
```

## Workspace handling

Your workspace kind determines how you should behave inside `$HERMES_KANBAN_WORKSPACE`:

| Kind | What it is | How to work |
|---|---|---|
| `scratch` | Fresh tmp dir, yours alone | Read/write freely; it gets GC'd when the task is archived. |
| `dir:<path>` | Shared persistent directory | Other runs will read what you write. Treat it like long-lived state. Path is guaranteed absolute (the kernel rejects relative paths). |
| `worktree` | Git worktree at the resolved path | If `.git` doesn't exist, run `git worktree add <path> <branch>` from the main repo first, then cd and work normally. Commit work here. |

## Tenant isolation

If `$HERMES_TENANT` is set, the task belongs to a tenant namespace. When reading or writing persistent memory, prefix memory entries with the tenant so context doesn't leak across tenants:

- Good: `business-a: Acme is our biggest customer`
- Bad (leaks): `Acme is our biggest customer`

## Good summary + metadata shapes

The `kanban_complete(summary=..., metadata=...)` handoff is how downstream workers read what you did. Patterns that work:

**Coding task:**
```python
kanban_complete(
    summary="shipped rate limiter — token bucket, keys on user_id with IP fallback, 14 tests pass",
    metadata={
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "decisions": ["user_id primary, IP fallback for unauthenticated requests"],
    },
)
```

**Research task:**
```python
kanban_complete(
    summary="3 competing libraries reviewed; vLLM wins on throughput, SGLang on latency, Tensorrt-LLM on memory efficiency",
    metadata={
        "sources_read": 12,
        "recommendation": "vLLM",
        "benchmarks": {"vllm": 1.0, "sglang": 0.87, "trtllm": 0.72},
    },
)
```

**Review task:**
```python
kanban_complete(
    summary="reviewed PR #123; 2 blocking issues found (SQL injection in /search, missing CSRF on /settings)",
    metadata={
        "pr_number": 123,
        "findings": [
            {"severity": "critical", "file": "api/search.py", "line": 42, "issue": "raw SQL concat"},
            {"severity": "high", "file": "api/settings.py", "issue": "missing CSRF middleware"},
        ],
        "approved": False,
    },
)
```

Shape `metadata` so downstream parsers (reviewers, aggregators, schedulers) can use it without re-reading your prose.

## Claiming cards you actually created

If your run produced new kanban tasks (via `kanban_create`), pass the ids in `created_cards` on `kanban_complete`. The kernel verifies each id exists and was created by your profile; any phantom id blocks the completion with an error listing what went wrong, and the rejected attempt is permanently recorded on the task's event log. **Only list ids you captured from a successful `kanban_create` return value — never invent ids from prose, never paste ids from earlier runs, never claim cards another worker created.**

```python
# GOOD — capture return values, then claim them.
c1 = kanban_create(title="remediate SQL injection", assignee="security-worker")
c2 = kanban_create(title="fix CSRF middleware", assignee="web-worker")

kanban_complete(
    summary="Review done; spawned remediations for both findings.",
    metadata={"pr_number": 123, "approved": False},
    created_cards=[c1["task_id"], c2["task_id"]],
)
```

```python
# BAD — claiming ids you don't have captured return values for.
kanban_complete(
    summary="Created remediation cards t_a1b2c3d4, t_deadbeef",  # hallucinated
    created_cards=["t_a1b2c3d4", "t_deadbeef"],                   # → gate rejects
)
```

If a `kanban_create` call fails (exception, tool_error), the card was NOT created — do not include a phantom id for it. Retry the create, or omit the id and mention the failure in your summary. The prose-scan pass also catches `t_<hex>` references in your free-form summary that don't resolve; these don't block the completion but show up as advisory warnings on the task in the dashboard.

## Block reasons that get answered fast

Bad: `"stuck"` — the human has no context.

Good: one sentence naming the specific decision you need. Leave longer context as a comment instead.

```python
kanban_comment(
    task_id=os.environ["HERMES_KANBAN_TASK"],
    body="Full context: I have user IPs from Cloudflare headers but some users are behind NATs with thousands of peers. Keying on IP alone causes false positives.",
)
kanban_block(reason="Rate limit key choice: IP (simple, NAT-unsafe) or user_id (requires auth, skips anonymous endpoints)?")
```

The block message is what appears in the dashboard / gateway notifier. The comment is the deeper context a human reads when they open the task.

## Heartbeats worth sending

Good heartbeats name progress: `"epoch 12/50, loss 0.31"`, `"scanned 1.2M/2.4M rows"`, `"uploaded 47/120 videos"`.

Bad heartbeats: `"still working"`, empty notes, sub-second intervals. Every few minutes max; skip entirely for tasks under ~2 minutes.

## Retry scenarios

If you open the task and `kanban_show` returns `runs: [...]` with one or more closed runs, you're a retry. The prior runs' `outcome` / `summary` / `error` tell you what didn't work. Don't repeat that path. Typical retry diagnostics:

- `outcome: "timed_out"` — the previous attempt hit `max_runtime_seconds`. You may need to chunk the work or shorten it.
- `outcome: "crashed"` — OOM or segfault. Reduce memory footprint.
- `outcome: "spawn_failed"` + `error: "..."` — usually a profile config issue (missing credential, bad PATH). Ask the human via `kanban_block` instead of retrying blindly.
- `outcome: "reclaimed"` + `summary: "task archived..."` — operator archived the task out from under the previous run; you probably shouldn't be running at all, check status carefully.
- `outcome: "blocked"` — a previous attempt blocked; the unblock comment should be in the thread by now.

## Do NOT

- Call `delegate_task` as a substitute for `kanban_create`. `delegate_task` is for short reasoning subtasks inside YOUR run; `kanban_create` is for cross-agent handoffs that outlive one API loop.
- Modify files outside `$HERMES_KANBAN_WORKSPACE` unless the task body says to.
- Create follow-up tasks assigned to yourself — assign to the right specialist.
- Complete a task you didn't actually finish. Block it instead.
- **Start writing code from scratch without checking whether the target files already exist with a complete implementation.** In multi-agent workflows, another agent may have already completed the work while the task was `running` or from a previous claim. Before writing any file: check if it exists, check its size/completeness, and assess whether the current task's work is already done. If the existing implementation is more complete than what you'd produce, report that and complete the task with a summary pointing to existing artifacts.
- **Create a duplicate task to "fix" a body/title that was wrong.** Use `hermes kanban update <task_id> --body "..."` instead. The only exception is when the task has already been claimed by a worker — then block it first and create a replacement with a `supersedes` comment.

## Review tasks — workflow and checklist

### Role switching: task assignee overrides pane/profile

When you are working via the Hermes TUI (not dispatched as a kanban worker), the pane/profile you're in normally determines your role (implementer, critic, coordinator). However, the user may instruct you to override this:

> "Pane/profile: critic; task assignee/role: implementer. The Kanban task assignee decides the role for this run."

When such an instruction is given:
- **The task's `assignee` field determines what role you should act in**, regardless of which pane/profile you're in.
- If assignee says `implementer`, write code, create files, run experiments — even if you're in the critic pane.
- If assignee says `critic`, review files, verify artifacts, assess completeness — even if you're in the implementer pane.
- The user may also specify this explicitly: "虽然当前 Hermes pane/profile 是 critic，本次必须按 implementer 职责工作" (even though the current pane is critic, you must work as implementer). Obey the role instruction, not the pane label.

This pattern arises when the user manually assigns tasks across profiles from a single TUI session. It's not a bug — it's a deliberate workflow.

### Pre-flight: don't start before implementer finishes

When you are assigned a **review/critic task** that has a parent implementer task:

1. **Check parent task status FIRST.** `kanban_show(parent_id)` before reading any code. If the parent is `running` or `ready` (not `done`), the review has nothing to review yet.
2. **Read source + tests.** Verify all expected files exist on disk before diving into review logic. If files are missing, check whether the parent implementer is still running.
3. **If blocked by parent, mark it and move on.** Don't wait in a loop for the implementer to finish — that keeps the kanban worker alive unnecessarily. Instead:

```python
kanban_complete(
    summary="CANNOT REVIEW — parent implementer task t_xxx is still 'running'. Source files do not exist on disk yet.",
    metadata={
        "review_outcome": "blocked",
        "block_reason": "Parent implementer task not yet completed",
        "parent_status": "running",
    },
)
```

The task will be re-promoted to `ready` when the parent completes and a fresh worker picks it up. This is more efficient than holding a worker process idle.

Common signals that the parent hasn't finished:
- `hermes kanban show <parent_id>` shows status=`running`
- Expected source files don't exist: `find ... -name "*.py"` returns empty
- The parent task's `runs` has no completed run with substantive summary

### Reviewer checklist (for code review tasks with acceptance criteria)

When reviewing implementer-created code (common pattern in multi-task pipelines):

1. **Read the source implementation** — understand the core logic, edge cases, and how it matches the task spec
2. **Read the test file** — check coverage for expected scenarios (happy path, edge cases, error handling, real-data smoke)
3. **Run the tests** — `python -m pytest tests/test_xxx.py -v` — 100% pass required
4. **Verify against ALL acceptance criteria** listed in the task body. Check each one explicitly.
5. **Check pullback/results** if the task involves gpuserver execution — verify output files exist and contain expected values
6. **Classify each finding** with severity: `pass` (meets criteria), `blocking` (must fix), `non-blocking` (minor, fix later)
7. **Complete with structured metadata** using the `findings` array so downstream can consume without re-reading prose:

```python
metadata = {
    "review_outcome": "approved",  # or "needs_revision"
    "tests_passed": N,
    "tests_total": N,
    "acceptance_criteria": {
        "F1_something": "PASS",
        "F2_something_else": "PASS",
    },
    "findings": [
        {"id": "F1", "severity": "pass",
         "summary": "Short description of what was verified",
         "evidence": "Code line references or test assertions"},
        {"id": "F2", "severity": "non-blocking",
         "summary": "Minor issue description",
         "evidence": "Location in code"},
    ],
}
```

### Common CLI bugs to watch for in reviews

- **`nargs='*'` ambiguity with defaults**: `--variants type=str nargs='*' default=[...]` — when user passes `--variants all`, argparse treats `all` as a single-element list, NOT as "use defaults". Fix: check `if "all" in args.variants: args.variants = [...default list...]` after parsing.
- **Import typos / unused imports**: Duplicate import names with typos (e.g. `commpare_` with double m) are harmless at runtime but confuse static analysis and are a cleanliness signal.
- **Missing valid_mask handling**: When computing per-pixel metrics from compacted depth_residual arrays, always reconstruct the full array via valid_mask before computing coverage-aware statistics.
- **Dead import + discarded return**: CLI orchestrators sometimes import a library function and call it, but immediately overwrite the return value with a fallback (e.g., `w = compute_weight(); w = np.ones_like(depth)`). The import and call become dead code. Flag this — it means the integration is a placeholder, not a real connection.
- **Synthetic/random data in production CLI path**: If a CLI script uses `np.random.*` to generate inputs for a library function (e.g., reprojection errors, viewing camera assignments), that step is producing fake data, not running the real pipeline. Flag as non-blocking but note it means the CLI is a skeleton, not production-ready.

This pattern applies to any review task where the implementer produces files — code review, document review, benchmark validation.

**Task state can change between dispatch and your startup.** Between when the dispatcher claimed and when your process actually booted, the task may have been blocked, reassigned, or archived. Always `kanban_show` first. If it reports `blocked` or `archived`, stop — you shouldn't be running.

**Workspace may have stale artifacts.** Especially `dir:` and `worktree` workspaces can have files from previous runs. Read the comment thread — it usually explains why you're running again and what state the workspace is in.

**Don't rely on the CLI when the guidance is available.** The `kanban_*` tools work across all terminal backends (Docker, Modal, SSH). `hermes kanban <verb>` from your terminal tool will fail in containerized backends because the CLI isn't installed there. When in doubt, use the tool.

**`--board <slug>` is the only reliable way to address a non-default board.** `hermes kanban boards switch <slug>` does NOT persist across CLI invocations — each `hermes kanban` call defaults to the `default` board. If a task exists on `egomotion4d` but you query without `--board egomotion4d`, it silently returns "no such task". Always use `--board <slug> inline` on every `show`/`list`/`complete` call when working with a named board. Example: `hermes kanban --board egomotion4d show t_xxx`.

**Complete/block/edit commands also need `--board <slug>`.** `kanban complete`, `kanban block`, and `kanban edit` all default to the `default` board, same as `show`/`list`. If the task is on a different board (e.g. `egomotion4d`), the completion silently fails with "cannot complete t_xxx (unknown id or terminal state)". Always pass `--board <slug>` for these commands too: `hermes kanban --board egomotion4d complete t_xxx ...`.

**`claim` is for the dispatcher, not for manual cross-role adoption.** Running `hermes kanban claim t_xxx` outside the dispatcher will fail with "cannot claim t_xxx: status=todo lock=(none)" if the task is assigned to a different profile or the task is not in a claimable state. The dispatcher handles claiming automatically when spawning workers. If you need to adopt a task assigned to another profile, use `hermes kanban reassign t_xxx <your-profile>` first, then work on it. For critic-taken tasks just do the work directly — don't try to claim first.

**Dead process, stuck "running" task.** If a worker process dies (OOM, killed, host reboot) but the run had already completed its work internally, the task may be stuck in `running` with no live process. **The correct recovery depends on whether you own the lock:**

- **If you own the lock (same profile that claimed it):** `hermes kanban --board <slug> complete <id> --summary "..."` transitions directly to `done`. Verified on egomotion4d board: `kanban complete t_6155a028` succeeded on a running task.

- **If you do NOT own the lock (the task is running under another profile):** `kanban complete` will fail with "unknown id or terminal state". Use `kanban edit` which forcibly transitions to `done` regardless of lock state: `hermes kanban --board <slug> edit <id> --result "..." --summary "..."`.

- **`kanban edit` does NOT work on running tasks when you own the lock.** It returns "cannot edit <id> (unknown id or task is not done)". `edit` is only for backfilling results on already-done tasks.

- **Always verify** the work is actually complete (tests pass, files exist) before doing this — don't mark done prematurely.

**`--metadata` JSON with special characters breaks bash quoting.** When passing `--metadata` to `hermes kanban complete`, JSON strings containing parentheses, single quotes, or other shell-sensitive characters will cause bash eval errors (e.g., `未预期的记号 ")" 附近有语法错误`). The fix: write the metadata JSON to a temp file first, then read it into the shell variable:

```bash
# Write metadata to temp file to avoid bash quoting issues
METADATA=$(cat /tmp/metadata.json)
hermes kanban --board <slug> complete <id> --summary "..." --metadata "$METADATA"
```

Never try to inline complex JSON with strings containing `(`, `)`, `'`, or `"` directly in the `--metadata` argument — bash will mangle the quoting regardless of single/double quote nesting.

## Board inventory/audit reviews

When a review task also asks you to audit the kanban board for stale, duplicate, or overlapping tasks (common in coordinator-level or end-of-phase reviews):

### Step 1: List all tasks grouped by status
```bash
hermes kanban --board <slug> list --json | python3 -c "
import json, sys
data = json.load(sys.stdin)
by_status = {}
for t in data:
    s = t['status']
    by_status.setdefault(s, []).append(t)
for s in sorted(by_status.keys()):
    tasks = by_status[s]
    print(f'\n=== {s} ({len(tasks)} tasks) ===')
    for t in tasks:
        print(f'  {t[\"task_id\"]}  [{t[\"assignee\"]}]  {t[\"title\"][:80]}')
"
```

### Step 2: Identify duplicates and overlaps
For each `ready` or `running` task, check whether its scope overlaps with another task:
- **Auto-generated audit reviews** (created by `coordinator-audit` when an implementer completes without a linked critic) often overlap with manually-created review tasks that cover broader scope. The broader review supersedes the narrower one — archive the narrower duplicate.
- **Sequential M0→M1→M2→M3→M4 tasks** (common in MegaSaM-lite-style chunked plans): only the first unblocked chunk should be dispatched; the rest are valid but premature.
- **Blocked tasks** should stay blocked unless the blocker condition has genuinely changed (e.g., P0 match ratio was <50% but a new fix brought it above 50%).

### Step 3: Report inventory in the review completion
Include a `board_inventory` section in the metadata:
```python
metadata["board_inventory"] = {
    "duplicate_tasks": ["t_fe33b0a3"],  # tasks to archive
    "correctly_blocked": ["t_51280acc"],
    "total_done": 333,
    "total_ready": 10,
    "total_blocked": 1,
    "total_running": 1,
}
```

### Re-review tasks (verifying fixes from a prior NEEDS_REVISION)

When a review task was completed with `needs_revision` and blocking findings, a re-review task may be created to verify the fixes. The re-review workflow is different from a first-pass review:

1. **Read the original review metadata** — find the prior review task ID and read its findings. The metadata JSON is typically at `/tmp/t_<id>_metadata.json` or in the task's completion summary.
2. **Check each BLOCKING finding individually** — for each blocking finding from the original review, verify the specific fix:
   - F7 "missing CLI script X" → `ls scripts/.../X.py` — does it exist now?
   - F8 "garbled report text" → read the specific line(s) — is the text clean now?
   - F11 "missing CLI script Y" → same as F7 pattern
3. **Re-run tests** — verify the test suite still passes after fixes.
4. **If ALL blocking findings are fixed** → APPROVED. Note what was fixed.
5. **If ANY blocking finding remains unfixed** → NEEDS_REVISION again. Reference the same finding IDs. Note that no fix task was created or the fix task was not completed.
6. **Do NOT re-do the full first-pass review** — focus on the blocking findings. Non-blocking findings from the original review can be mentioned but should not be re-elevated to blocking unless they worsened.

**Critical anti-pattern:** Do not approve a re-review if the blocking findings are still present. The re-review exists specifically because the first review found problems. If nothing changed, the verdict must still be NEEDS_REVISION, with a note that a fix implementer task should be created.

### Auto-generated audit review deduplication

When a coordinator-audit bot creates individual `[Review] Impl M1: <title>` tasks for each completed implementer subtask, and a broader review task already covered that scope (e.g., "Review M0-M4 implementation"), the individual audit review is redundant:

1. **Check if the scope was already reviewed** — look at completed review tasks on the board. If a broader review (e.g., t_46ce38b5 covering M0-M4) already approved the same code with the same tests passing, the narrower audit review (e.g., M1 only) is a duplicate.
2. **Approve with reference** — complete the duplicate with `APPROVED (already reviewed in t_xxx)` and set `duplicate_of: "t_xxx"` in metadata.
3. **Do NOT re-run the full review** — this wastes time and tokens. The broader review already verified the code.
4. **Exception:** If the broader review had blocking findings specifically in the narrow scope, the audit review is NOT a duplicate — it needs its own review to verify the fix.

This pattern applies to any review task where the parent task is covered by a broader review that already completed. Common in coordinator-audit auto-generated tasks.

### Pitfall: Auto-generated audit review duplicates manual review
When a coordinator-audit bot creates `[Review] <impl_task_title>` for every completed implementer task, and a human also creates a broader review task (e.g., "Review all deliverables from Impl A, B, C"), the auto-generated review for Impl C is a subset of the broader review. **Archive the auto-generated duplicate after the broader review completes** — don't dispatch two critics to review overlapping scope.

### Pitfall: 333+ done tasks cluttering the board
Large boards with hundreds of `done` tasks make `list` unwieldy. Archive old done tasks by date range if the user asks.

## Interactive launcher pitfalls (codex/deepseek/reasonix kanban_interactive.py)

These three scripts share the same architecture: a launcher starts a watcher subprocess and a TUI process, both writing to a log file under `~/.hermes/kanban/boards/<board>/logs/`. If that directory doesn't exist yet (first run on a new board), `log_path.open("a")` crashes with `FileNotFoundError`.

**Fix applied**: All three scripts now have `log_path.parent.mkdir(parents=True, exist_ok=True)` before `log_path.open()`. If you create a new interactive launcher script, always include this guard.

**Files patched**:
- `plugins/kanban/codex_listener/codex_kanban_interactive.py` (line ~825)
- `plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py` (line ~1404)
- `plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py` (line ~1417)

**Default task delivery**: `inject` only. Self-poll mode was removed on 2026-06-06. All three interactive scripts use inject mode (watcher polls kanban → detects TUI idle → zellij injects task text). The `--task-delivery` flag accepts only `inject`.

**Auto-start mode**: All three interactive launchers accept `--auto-start` to skip the `input()` "press Enter" prompt and start immediately. This is used by `start-kanban.sh` so all panes launch without manual interaction. When `--auto-start` is not passed, the launcher prints "按 Enter 进入 interactive <agent>" and waits for user input — this is the default for manual/interactive use.

### Reasonix-specific pitfalls

- **v1.1.0 (Go rewrite) breaks multiple v0.x CLI flags.** `reasonix code <workspace>` → `reasonix chat --dir <workspace>`, `--system-append-file` → `system_prompt_file` in per-workspace `reasonix.toml`, `--no-mouse`/`--no-dashboard` → `--yolo`, `--new` → omit `--continue`. The `_REASONIX_IDLE_MARKERS` must include v1.1.0's new TUI idle prompts (`对话上下文将跨轮保留`, `输入 'exit' 或按 ctrl-d 退出`, `YOLO · 已跳过批准`) or the watcher permanently skips claiming. See `upgrading-reasonix` skill for full migration details.
- **v1.1.0 `--continue` does NOT recognize v0.x migrated sessions.** Even though session files are copied to `~/.config/reasonix/sessions/`, `--continue` exits rc=1 with "没有可恢复的会话". For kanban mode, omit `--continue` entirely — sessions are ephemeral (watcher injects tasks).
- **`--session` is NOT supported.** Reasonix CLI does not have a `--session` parameter (unlike deepseek-tui). Session names are auto-generated from the workspace directory name (e.g. `code-Egomotion4D`). Passing `--session <name>` causes `error: unknown option '--session'` and the TUI exits immediately.
- **Session control is `--continue` / `--new` only.** Use `--continue` to resume the most recent session for the workspace, `--new` to force a new session. There is no way to name or select a specific session from the CLI.
- **Multiple panes in same workspace are independent processes.** Since there's no `--session` to differentiate, each reasonix process gets its own session automatically. The `other_continue_reasonix_active()` guard from deepseek-tui doesn't apply the same way — just use `--continue` for primary role and `--new` for secondary roles.
- **Reasonix MCP config: `~/.config/reasonix/config.toml` `[[plugins]]` with `auto_start`.** The v1.x Go rewrite moved MCP config from `~/.reasonix/config.json` `mcp` array to `~/.config/reasonix/config.toml` `[[plugins]]` sections. Each plugin has `name`, `command`, `args` (optional), and `auto_start` (bool, default false). **`auto_start = true` is required** for MCP servers to connect automatically at startup; without it, you must manually run `/mcp add <name>` in chat. Common mistakes: (1) putting `auto_start=true` in the `args` array instead of as a standalone field, (2) putting command arguments in the `command` string instead of splitting into `args` array, (3) adding a duplicate codegraph `[[plugins]]` entry that conflicts with the built-in one. See `upgrading-reasonix` skill 陷阱 3 for full details.
- **Reasonix does NOT read `.mcp.json`.** Unlike Hermes, reasonix does not auto-discover MCP servers from project-local `.mcp.json` files. MCP servers must be registered in `~/.config/reasonix/config.toml` `[[plugins]]` sections.
- **Idle detection: whitespace normalization required.** Reasonix TUI renders idle markers with variable spacing (e.g. `输入任何内容  ·  / 使用命令  ·  @ 引用文件` with double spaces around `·`, vs the marker `输入任何内容 · / 使用命令 · @ 引用文件` with single spaces). Without whitespace normalization before substring matching, the idle marker never matches and the listener perpetually reports "skip claim: Reasonix pane still shows an active/pending Kanban prompt" even when the TUI is clearly idle. **Fix**: normalize whitespace (collapse consecutive spaces to one) in both `_looks_like_idle_reasonix_pane()` and `_looks_like_busy_reasonix_pane()` before checking markers. This applies to deepseek-tui as well if its TUI rendering changes spacing.
- **Log directory auto-creation.** All three interactive launchers (codex/deepseek/reasonix) write logs to `~/.hermes/kanban/boards/<board>/logs/`. On first run with a new board, this directory doesn't exist and `log_path.open("a")` crashes with `FileNotFoundError`. All three now have `log_path.parent.mkdir(parents=True, exist_ok=True)` before the open call. Always include this guard in any new interactive launcher script.

## Kanban DB corruption — all workers crash at startup

If every kanban pane worker (hermes, codex, deepseek) exits within seconds of launch and the zellij session goes dead, check the board database first:

```bash
# Quick check — file type should be "SQLite 3.x database", NOT "data"
file ~/.hermes/kanban/boards/<board>/kanban.db
```

**Symptom**: All 4 pane workers crash immediately with `sqlite3.DatabaseError: file is not a database` in `kanban_db.py:connect()`. The zellij session stays alive but all panes show empty shells (commands exited). Zellij logs show `Input/output error (os error 5)` as PTYs close after child exit.

**Root cause**: The `kanban.db` file is corrupted (header bytes damaged, `file` reports `data` instead of `SQLite 3.x database`). Every worker's first action is `claim_and_inject_one()` → `kb.connect()` → `PRAGMA journal_mode=WAL`, which fails on a corrupted DB.

**Fix**:
1. Check for repair backups: `ls -lt ~/.hermes/kanban/boards/<board>/kanban.db.repair-test-*`
2. Validate a backup: `file <backup>` should say `SQLite 3.x database`
3. Verify content: `sqlite3 <backup> "SELECT count(*) FROM tasks;"`
4. Replace: `cp kanban.db kanban.db.corrupted-$(date +%Y%m%d) && cp <backup> kanban.db`
5. Restart kanban session

**Prevention**: Consider adding an integrity check in `kanban_db.py:connect()` that falls back to the latest valid repair-test backup if `PRAGMA integrity_check` fails, instead of crashing.

**Full debugging trace and recommended code fix**: See `references/kanban-db-corruption-debugging-2026-05-22.md`.

## Interactive launcher crash-loop protections (2026-06-06)

The three interactive launchers (codex/deepseek/reasonix `*_kanban_interactive.py`) share a common architecture: launcher starts a watcher subprocess + TUI process, loops checking both. Three failure modes were discovered and patched:

### Failure 1: TUI process dies but watcher loops forever

**Symptom**: The TUI (reasonix/deepseek/codex) exits, but the watcher subprocess keeps running. The zellij pane shows stale screen content. The watcher's `_pane_can_accept_new_kanban_task()` sees the old content and logs `skip claim: Reasonix pane still shows an active/pending Kanban prompt` every poll interval, forever. Both processes eventually die but the watcher never detects the TUI is gone.

**Root cause**: The launcher's main loop only checks `proc.poll()` for the TUI process. If the process exits abnormally (SIGKILL, OOM, etc.) and `poll()` hasn't caught up (or the PID was recycled), the loop continues indefinitely.

**Fix applied** (all three launchers): Added `_pid_alive(proc.pid)` check in the launcher's main loop. If the PID is dead but `poll()` hasn't returned, double-check after 1 second, then force-break with `rc=-1`. This ensures the launcher exits when the TUI is truly gone.

### Failure 2: Watcher crash-loop without restart limit

**Symptom**: The watcher subprocess hits a transient error (DB I/O, etc.) and exits with rc≠0. The launcher restarts it. It crashes again immediately. This repeats hundreds of times (observed: 319 restarts), flooding the log with identical error messages.

**Root cause**: The launcher's `_should_restart_watcher()` only checks `rc != 0`, with no cap on restart count.

**Fix applied** (all three launchers): Added `MAX_WATCHER_RESTARTS = 10`. After 10 consecutive restarts, the launcher sets `watcher = None` instead of restarting, letting the TUI continue running without a background watcher (graceful degradation).

### Failure 3: DB OperationalError crashes the watcher

**Symptom**: The watcher's `kb.connect()` or subsequent DB operations hit `sqlite3.OperationalError: disk I/O error`. The exception propagates up, crashing the watcher. Combined with Failure 2, this creates an infinite crash-loop.

**Root cause**: The watcher's DB operations have no try/except for `OperationalError`. A transient I/O error (disk pressure, SQLite lock contention, filesystem hiccup) is fatal.

**Fix applied** (all three launchers): Added `_db_connect_with_retry()` / `_ensure_conn()` wrapper in `watcher_main()` that:
- Retries `kb.connect()` up to 3 times with exponential backoff (2/4/8 seconds)
- Tracks `consecutive_db_errors` across retries; after 5 consecutive failures, exits the watcher cleanly
- Resets the counter on any successful DB operation
- Replaces `kb.connect(board=board)` calls in the watcher loop with `_ensure_conn()`
- On `DatabaseError("malformed"/"corrupt")`, attempts REINDEX repair before giving up

**Applied to**: All three launchers (reasonix, deepseek, codex) — patched 2026-06-07.

### Recovering stuck tasks after crash-loop

When a crash-loop leaves a task stuck in `running` status with a dead worker PID:

```bash
# Check for stuck tasks
sqlite3 ~/.hermes/kanban/boards/<board>/kanban.db \
  "SELECT id, status, worker_pid FROM tasks WHERE status='running';"

# If worker_pid is dead, reclaim the task
hermes kanban --board <board> reclaim <task_id> --reason "worker dead after crash-loop"

# If reclaim sets it to running with a new (dead) pid, force reset via SQL:
sqlite3 ~/.hermes/kanban/boards/<board>/kanban.db \
  "UPDATE tasks SET status='todo', worker_pid=NULL, claim_lock=NULL, claim_expires=NULL, started_at=NULL WHERE id='<task_id>';"
```

### Failure 5: KANBAN_TASK_BOUNDARY causes SSH kill-0 infinite loops

**Symptom**: After a task boundary injection, the AI (reasonix/deepseek/codex) starts running `ssh gpuserver 'kill -0 <PID>'` in a loop every 60-120 seconds, even though the remote process has already exited or the task was already blocked. The loop continues indefinitely because each `kill -0` failure triggers another retry.

**Root cause**: The `KANBAN_TASK_BOUNDARY` injection text previously said `"读取并执行当前 Hermes Kanban 任务文件；不要延续上一轮输出。"` — this explicitly tells the AI to forget the previous round's context. When the AI had been monitoring a remote SSH process via `kill -0`, the boundary injection erases that context. The AI then re-derives the same monitoring strategy from scratch (because the conversation history still contains `kill -0` traces), but without knowing the process already exited or the task was already blocked.

**Fix applied** (reasonix_kanban_interactive.py, 2026-06-06):
1. Changed the `KANBAN_TASK_BOUNDARY` injection text from "不要延续上一轮输出" to explicit context-continuation rules:
   - "不要重复上一轮已经在做或已完成的动作"
   - "如果上一轮启动的远程进程 PID 已经不在（kill -0 返回非零），不要再轮询它"
   - "如果上一轮已经 hermes kanban block 了某个任务，不要再对该任务做任何操作"
   - "远程命令：不要用 ssh + kill -0 循环轮询。改用 nohup + 输出重定向 + cat/tail 检查日志"
2. Added "远程进程管理" rule (item 7) to `build_interactive_prompt()` instructing workers:
   - Never use `ssh + kill -0` in a loop
   - Use `nohup + output redirect`, then check result files/logs
   - `kill -0` at most once, never in a loop
   - If `kill -0` returns non-zero, immediately check output files instead of retrying

**Prevention for future prompt changes**: Any text injected at task boundaries must NOT instruct the AI to discard previous context. Instead, it should provide explicit rules about what NOT to repeat and how to handle in-progress operations from the previous round.

### Failure 6: Idle-pane reclaim in adopt_orphaned_running_task causes claim-reclaim loop

**Symptom**: The watcher restarts, detects an orphaned running task, sees the pane is idle, and immediately reclaims it. The task goes back to `ready`, gets claimed and injected again, the watcher crashes (often from a DB error during post-inject operations), restarts, and the cycle repeats. Observed: the same task reclaimed every 8 seconds (matching `--startup-delay-s 8.0`) across dozens of watcher restarts.

**Root cause**: `adopt_orphaned_running_task()` checks `_pane_looks_idle()` immediately after startup. When a prompt was just injected but the TUI hasn't started processing yet (rendering delay, AI thinking time), the pane still shows idle markers. The watcher has no grace period — it trusts the idle detection immediately and reclaims.

**The crash-reclaim feedback loop**:
1. Watcher starts → waits startup_delay → `adopt_orphaned_running_task` → pane idle → **reclaim immediately**
2. Task returns to `ready` → `claim_and_inject_one` → prompt injected into pane
3. Post-inject DB ops (`add_comment`, `heartbeat_worker`) hit `OperationalError: disk I/O error` → watcher crashes
4. Launcher restarts watcher → goto 1

**Fix applied** (reasonix + deepseek, 2026-06-06):
1. **Grace period in `adopt_orphaned_running_task`**: Before reclaiming based on idle-pane detection, check `task_runs.started_at`. If the run started less than 30 seconds ago, **adopt** the task instead of reclaiming — the AI likely hasn't had time to start processing the prompt. Fall through to the adopt path.

```python
# In adopt_orphaned_running_task, when pane is idle:
run_started_at = conn.execute(
    "SELECT started_at FROM task_runs WHERE id = ?", (run_id,)
).fetchone()
grace_period_s = 30
if run_started_at and (now - int(run_started_at)) < grace_period_s:
    log_line(log_path, f"orphaned task {task_id} run={run_id} pane idle but run started "
             f"{now - int(run_started_at)}s ago (< {grace_period_s}s); adopting instead of reclaiming")
    # Fall through to adopt path
else:
    # Original reclaim logic
```

2. **DB error protection in `claim_and_inject_one`**: Wrapped all `kb.connect()` calls in `claim_and_inject_one()` with try/except so DB I/O errors don't crash the watcher:
   - **Claim-phase DB errors** → `return None, None` (claim failed, retry next poll)
   - **Post-inject DB errors** (`add_comment` / `heartbeat_worker`) → log only, don't crash (prompt already injected successfully)
   - **Reclaim-on-failure DB errors** → log only (reclaim failure is non-fatal)

This applies to both reasonix_kanban_interactive.py and deepseek_kanban_interactive.py. Codex listener doesn't use the same pane-idle-detection pattern.

**Key insight**: The `add_comment`/`heartbeat_worker` calls after a successful injection are purely record-keeping. If they fail, the task is still running in the TUI — crashing the watcher is worse than missing a comment. The watcher's main loop heartbeat will catch up on the next iteration.

### Self-poll mode removed (2026-06-06)

Self-poll task delivery mode has been removed from all three interactive launchers. Only `inject` mode is supported now.

**Files changed**:
- `plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py` — removed `self_poll_prompt_dir`, `write_self_poll_startup_prompt`, `write_role_instructions_file`, `write_workspace_reasonix_toml`, self-poll launcher branch, `--task-delivery` now only accepts `inject`
- `plugins/kanban/codex_listener/codex_kanban_interactive.py` — removed `self_poll_prompt_dir`, `write_self_poll_startup_prompt`, self-poll branch, `--task-delivery` now only accepts `inject`
- `plugins/kanban/codex_listener/codex_kanban_listener.py` — removed `_start_self_poll_codex`, self-poll branch, monitoring logic, `--task-delivery` now only accepts `inject`
- `plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py` — removed `self_poll_prompt_dir`, `write_self_poll_startup_prompt`, self-poll branch, `--task-delivery` now only accepts `inject`
- `plugins/kanban/deepseek_listener/deepseek_kanban_listener.py` — removed `_start_self_poll_deepseek`, self-poll branch, monitoring logic, `--task-delivery` now only accepts `inject`

**Pitfall during removal**: The `write_role_instructions_file` and `write_workspace_reasonix_toml` functions in reasonix_kanban_interactive.py were initially deleted along with self-poll code, because they were only called from the self-poll launcher branch. However, **inject mode also needs these** — the inject-mode `launcher_main` calls `build_reasonix_cmd(args, role_instructions_path=role_instructions_path)` to set up `reasonix.toml`'s `system_prompt_file`. Deleting them caused `NameError: name 'role_instructions_path' is not defined` at startup. Fix: restored a combined `_write_role_instructions()` function that writes both the role-instructions.md and reasonix.toml, called from `launcher_main` regardless of delivery mode.

**`worker_runtime.py` functions preserved** — `build_self_poll_startup_prompt`, `reset_self_poll_claims`, `default_self_poll_owner` still exist in the shared library but are no longer called from any listener. They can be removed in a future cleanup.

**Migration**: If `HERMES_KANBAN_TASK_DELIVERY=self-poll` is set in `.env` or shell, it will be ignored — `--task-delivery` now defaults to `inject` unconditionally.

**General lesson**: When removing a code path (like self-poll), check that no functions called exclusively from that path are ALSO needed by other paths (like inject). Shared utility functions may appear to be "self-poll only" because they're only called in the self-poll branch, but other branches may reference their return values later in the same function.

**Pitfall: Shell launch script out of sync with Python scripts.** When removing `--task-delivery` from the Python interactive scripts (codex/deepseek/reasonix `*_kanban_interactive.py`), the shell launch script `start-kanban.sh` must also be updated to stop passing `--task-delivery inject` and related env vars (`HERMES_KANBAN_TASK_DELIVERY`, `HERMES_KANBAN_SELF_POLL_OWNER`). Otherwise `start-kanban.sh` will fail with `error: unrecognized arguments: --task-delivery inject`. The fix: remove `--task-delivery` from all `build_role_command` branches in `start-kanban.sh`, remove `CODEX_LISTEN`/`DEEPSEEK_LISTEN` variables and worker-mode branches, and keep `--task-delivery` as a backward-compatible no-op in the shell arg parser (warns and forces `inject`).

### Failure 4: Assist-role pane claims same task as primary pane

**Symptom**: A task assigned to `implementer` gets claimed by both the implementer pane AND the critic pane (or any pane with `--claim-assignees critic,implementer`). The event log shows alternating `claimed`/`reclaimed` events between two different PIDs, both with `:reasonix-interactive` suffix but different pane profiles.

**Root cause**: The `--assist-role critic:implementer` flag makes the critic pane's watcher able to claim `implementer` tasks. This is intentional — when the implementer is idle, the critic assists. But when the implementer watcher is crash-looping (Failure 2), its claim expires and gets reclaimed. The critic's watcher then picks up the same task. The implementer watcher restarts and immediately tries to adopt the orphaned task back. This creates an alternating claim/reclaim cycle between two panes for the same task.

**Why `--assist-claim-delay-for` doesn't prevent this**: The assist-claim-delay only applies when a task **first becomes ready**. When a task is reclaimed (claim expired/crashed), it transitions back to `ready` but the delay threshold may not reset properly. The critic watcher sees the task as immediately claimable and grabs it before the implementer watcher recovers.

**Prevention options** (pick based on stability of the primary pane):
1. **Remove assist-role for that pane** if the primary pane is frequently unstable: `start-kanban.sh -b egomotion4d` without `--assist-role critic:implementer`. The critic will only claim critic tasks.
2. **Increase assist-claim-delay**: `--assist-claim-delay-for implementer=300` (5 minutes) gives the primary pane time to recover before the assist pane can claim.
3. **Monitor crash-loops and manually reclaim**: When a crash-loop is detected (see Diagnostic section), reset the stuck task back to `todo` and restart only the primary pane's listener.

**Diagnostic**: Check the event log for alternating claims between different pane profiles:
```bash
sqlite3 ~/.hermes/kanban/boards/<board>/kanban.db \
  "SELECT id, run_id, kind, substr(payload, 1, 200), datetime(created_at, 'unixepoch', 'localtime')
   FROM task_events WHERE task_id='<id>' ORDER BY created_at DESC LIMIT 20;"
```
If you see `claimed` events from two different `claim_lock` suffixes (e.g. `:reasonix-interactive` from PID X and PID Y), that's a double-claim from two panes.

### Failure 7: sqlite3.OperationalError disk I/O error — multi-process WAL contention

**Symptom**: `sqlite3.OperationalError: disk I/O error` in watcher logs, often on SELECT queries (not just writes). Multiple `.corrupt.*.bak` files in the board DB directory. Only one listener (usually the one with crash-loop) shows the error; others don't.

**NOT a disk hardware problem** — if `PRAGMA integrity_check` returns `ok` and the filesystem is local ext4 (not NFS), the root cause is SQLite WAL-mode concurrency, not disk failure.

**Root cause chain**:
1. **Multiple processes write the same kanban.db simultaneously**: gateway dispatch daemon (5s poll) + 3-5 interactive listeners (6-8s poll each) = 5+ concurrent writers
2. **Short connections amplify contention**: `claim_and_inject_one` opens 4 separate `kb.connect()` calls per invocation (claim, pre-inject reclaim, inject-fail reclaim, add_comment/heartbeat). Each connection does PRAGMA setup, then operation, then close.
3. **WAL checkpoint competition**: `wal_autocheckpoint=100` means multiple processes can trigger checkpoints simultaneously. Checkpoints need exclusive locks that conflict with read/write operations.
4. **Crash-loop amplification**: Idle-pane reclaim causes watcher crash, then restart, then immediate DB operation hits lock contention, crashes again. The 8-second restart cycle (matching `--startup-delay-s`) creates resonance with the 5-second gateway dispatch cycle.

**Diagnostic methodology** (follow this order):
1. Check `.corrupt.*.bak` file count and timestamps
2. Run `PRAGMA integrity_check` — if `ok`, it is NOT hardware corruption
3. Check filesystem type — must be local (ext4/xfs), NOT network (NFS/SMB/FUSE). WAL is incompatible with NFS.
4. Check busy_timeout — should be 120000ms (Hermes default), NOT 5000ms (Python default). If 5s, someone is using raw `sqlite3.connect()` instead of `kanban_db.connect()`.
5. Count concurrent processes — `fuser <db_path>` or `ps aux | grep kanban_interactive | grep watch-child`
6. Check WAL/SHM sidecar files — large WAL means pending checkpoints
7. Grep watcher log for error frequency
8. Check if errors cluster with crash-loops — see if preceding line is a reclaim/restart

**Solutions** (in order of effectiveness):
1. **Force DELETE journal mode instead of WAL** ✅ IMPLEMENTED 2026-06-08: `kanban_db.py:connect()` now defaults to `HERMES_KANBAN_FORCE_DELETE_JOURNAL=1`, which skips `apply_wal_with_fallback()` and instead does `PRAGMA journal_mode=DELETE`. If the on-disk DB is already WAL, it first runs `PRAGMA wal_checkpoint(TRUNCATE)` then switches to DELETE. DELETE mode serializes writes via SQLite's rollback journal — slower than WAL for concurrent reads, but eliminates WAL checkpoint contention entirely. For kanban's write pattern (lightweight claim/complete operations), the performance difference is negligible while the stability improvement is decisive. Set `HERMES_KANBAN_FORCE_DELETE_JOURNAL=0` to revert to WAL if needed.

   **Critical pitfall: Residual processes block journal_mode switching.** When switching from WAL to DELETE, ALL processes holding open connections to the DB must be killed first. `PRAGMA journal_mode=DELETE` silently fails if another process has the DB open in WAL mode (SQLite requires exclusive access to change journal mode). After `hermes gateway stop`, watcher subprocesses may still hold WAL connections — always check with `fuser <db_path>` and `kill -9` any remaining PIDs before attempting the switch. Only after all holders are killed will `PRAGMA journal_mode=DELETE` succeed and persist. Symptom: you set DELETE, verify it, start gateway, and it reverts to WAL — this means a residual process (often a watcher `--watch-child` subprocess) was still holding a WAL connection and re-applied WAL on its next connect().

   **Critical pitfall: `os.environ.setdefault` doesn't work for systemd-launched gateway.** The gateway process is launched by systemd user service, which does NOT inherit the shell's environment. `os.environ.setdefault("HERMES_KANBAN_FORCE_DELETE_JOURNAL", "1")` only sets the var if it's not already in the process's environment — but since systemd doesn't pass it, `setdefault` would set it. The real issue is that `setdefault` may run too late (after `apply_wal_with_fallback` was already imported and called). The correct pattern is `os.environ.get("HERMES_KANBAN_FORCE_DELETE_JOURNAL", "1")` with default "1" inline, and check the value immediately — do NOT rely on `setdefault` to influence code running in the same function.

   **Critical pitfall: Gateway stderr is a socket, not a file.** When debugging gateway code, `print(..., file=sys.stderr)` will NOT appear in `journalctl --user -u hermes-gateway` or in `~/.hermes/logs/gateway.log`. The gateway's fd/2 points to a socket (`socket:[...]`), not a pipe or file. To log from `kanban_db.py:connect()`, use `logging.getLogger("kanban_db").warning(...)` instead of `print(file=sys.stderr)`.

   **Critical pitfall: `_guard_existing_db_is_healthy()` opens a probe connection before `_INIT_LOCK`.** This probe inherits the on-disk journal_mode (WAL if that's what the header says). The probe is short-lived (opened, integrity check, closed), but if other connections are also open, the probe holds a WAL reader slot temporarily. This is harmless for the probe itself, but it means you cannot assume the first `connect()` call is the only one touching the DB during initialization.
2. **Fix the crash-loop first** (idle grace period + DB error non-fatal handling — see Failure 6). This stops the resonance pattern.
3. **Reduce connection count per operation** ✅ IMPLEMENTED 2026-06-07: `claim_and_inject_one` now accepts optional `conn` parameter; post-inject `add_comment` + `heartbeat_worker` merged into claim connection; pre-inject and inject-fail reclaim reuse same connection. Connection count from 4 → 1-2 per call. Applied to reasonix + deepseek + codex listeners.
4. **Increase WAL checkpoint interval**: `PRAGMA wal_autocheckpoint=1000` (default is 100) reduces checkpoint frequency. (Only relevant if WAL mode is re-enabled.)
5. **Watchers share a single connection per main loop** ✅ IMPLEMENTED 2026-06-07: `watcher_main()` now uses `_ensure_conn()` persistent connection with 60s recycling, liveness probe, retry on OperationalError, and `finally` close. Applied to all three listeners (reasonix, deepseek, codex). The codex listener was the last to receive Plan 2 (2026-06-07, second patch); previously it used `with kb.connect()` on every loop iteration (3 connections per claim cycle), causing the worst WAL lock churn of all listeners.

**Pitfall: kb.connect() context manager does NOT close connections.** `with kb.connect() as conn:` only calls `__enter__` (PRAGMA setup) and `__exit__` (commit/rollback). It NEVER calls `conn.close()`. Every `with kb.connect()` leaks a file descriptor and holds a WAL reader lock until GC collects the connection. With 8 call sites per listener file, a single claim-and-inject cycle leaks 4+ FDs. The persistent connection pattern (Plan 2, `_ensure_conn()` + `finally close`) eliminates this leak by keeping one connection alive for the watcher's lifetime and explicitly closing it on exit. When passing `conn` to sub-functions like `claim_and_inject_one`, use the `_owns_conn` pattern: if the function created the connection, it closes it in `finally`; if the caller passed it in, the function does NOT close it.

**Pitfall: Don't assume disk I/O error means disk failure.** SQLite's `disk I/O error` covers WAL lock contention, failed fcntl/flock calls, and interrupted checkpoints — all of which are transient software issues, not hardware problems. Always run `PRAGMA integrity_check` before investigating hardware.

**Pitfall: busy_timeout 5ms vs 120000ms.** If you test with `sqlite3.connect(db)` directly (Python default), you get 5000ms timeout. If you test with `kanban_db.connect(board=...)`, you get 120000ms. Always use the kanban_db.connect path for testing — raw sqlite3.connect will mislead you about the actual timeout in production.

**Pitfall: FD leak causes disk I/O error, not just WAL contention.** The original analysis attributed disk I/O errors primarily to WAL checkpoint competition between concurrent writers. The deeper root cause is **FD exhaustion**: `with kb.connect() as conn:` never closes connections (see above). Over time, FDs accumulate toward `ulimit -n` (default 1024). When SQLite cannot `open()` the WAL or SHM file because the process hit its FD limit, it returns `SQLITE_IOERR` which Python surfaces as `OperationalError: disk I/O error`. This is NOT a disk problem — it's a process resource limit problem that SQLite maps to the wrong error code.

**Three-layer fix** (all implemented 2026-06-07):

| Layer | Fix | Purpose |
|-------|-----|---------|
| **Root cause** | Plan 1 + Plan 2 (persistent connections, `_owns_conn` pattern) | Eliminate FD leak entirely |
| **Process limits** | `resource.setrlimit(RLIMIT_NOFILE, (4096, hard))` in `watcher_main()` | Raise safety margin from 1024 to 4096 FDs |
| **Auto-repair** | `REINDEX` on `DatabaseError("malformed"/"corrupt")` in `_ensure_conn()` and active-task loop | Recover from index corruption without crashing |

**ulimit fix detail**: Add at the start of `watcher_main()`:
```python
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 4096:
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
except Exception:
    pass
```

**REINDEX auto-repair detail**: In `_ensure_conn()`, add a `DatabaseError` handler (after `OperationalError`, since `OperationalError` is a subclass of `DatabaseError`):
```python
except sqlite3.DatabaseError as exc:
    msg = str(exc).lower()
    if "malformed" in msg or "corrupt" in msg:
        log_line(log_path, f"DB corruption detected: {exc}; attempting REINDEX repair")
        try:
            repair_conn = sqlite3.connect(str(kb.kanban_db_path(board=board)),
                                           isolation_level=None, timeout=120.0)
            repair_conn.execute("PRAGMA integrity_check")
            repair_conn.execute("REINDEX")
            repair_conn.execute("PRAGMA integrity_check")
            repair_conn.close()
            # Retry connect after repair
            _conn = kb.connect(board=board)
            _conn_created_at = time.time()
            consecutive_db_errors = 0
            return _conn
        except Exception as repair_exc:
            log_line(log_path, f"REINDEX repair failed: {repair_exc}")
    # Fall through to retry
    consecutive_db_errors += 1
    ...
```

The same pattern applies in the active-task loop exception handler — catch `DatabaseError` with malformed/corrupt detection, attempt REINDEX, then reconnect and continue.

**Applied to**: reasonix_kanban_interactive.py, deepseek_kanban_interactive.py, codex_kanban_interactive.py (all patched 2026-06-07).

**Reference**: See `references/sqlite-wal-disk-io-error-2026-06-06.md` for the original diagnostic trace. See `references/persistent-connection-implementation-2026-06-07.md` for Plan 1 + Plan 2 implementation. See `references/sqlite-fd-leak-and-reindex-repair-2026-06-07.md` for FD leak root cause analysis, ulimit fix, REINDEX auto-repair, and dispatch prevention details. See `references/kanban-db-wal-to-delete-migration-2026-06-08.md` for the WAL→DELETE migration implementation, critical pitfalls (residual processes blocking switch, gateway stderr socket, os.environ.setdefault timing), and post-migration verification checklist.

### Failure 8: Dispatch daemon steals tasks from interactive listeners

**Symptom**: A task with parents completing shows `status=running` in kanban, but the zellij pane remains idle — no task prompt was injected. The user sees "显示在执行但没看到真正被执行". Eventually the user manually claims the task to get it running in the visible pane.

**Root cause**: `dispatch_once` (called by `hermes kanban dispatch` or `hermes kanban daemon`) and the interactive listeners (reasonix/codex/deepseek `*_kanban_interactive.py`) both compete for the same `ready` tasks. When `dispatch_once` wins the race:

1. It calls `claim_task()` with `_claimer_id()` → claim_lock format `hostname:pid` (no `:reasonix-interactive` suffix)
2. It spawns a **headless** worker via `_default_spawn()` → `hermes -p <profile> chat -q "work kanban task <id>"`
3. This worker runs in the background — it does NOT inject into any zellij pane
4. The interactive listener sees the task is already `running` (claimed by someone else) and skips it
5. The headless worker may produce heartbeats and even complete the task, but the user sees nothing in their visible panes

**Diagnostic**: Check the claim_lock format in task_events:
```bash
sqlite3 ~/.hermes/kanban/boards/<board>/kanban.db \
  "SELECT te.created_at, te.kind, te.payload
   FROM task_events te
   WHERE te.task_id='<id>' AND te.kind='claimed'
   ORDER BY te.created_at DESC LIMIT 5;"
```
- claim_lock with `:reasonix-interactive` or `:codex-interactive` suffix → claimed by interactive listener (correct)
- claim_lock with just `hostname:pid` (no suffix) → claimed by dispatch daemon's `_default_spawn` (headless, wrong)

**Prevention** (three mechanisms, all implemented 2026-06-07):

1. **AGENTS.md §8.0 hard rule**: "禁止执行 `hermes kanban dispatch`、`hermes kanban daemon`、或任何会调用 `dispatch_once`/`dispatch_loop` 的命令". Explicitly documented that the only legal task delivery is interactive listener inject mode. This prevents both human and AI agents from accidentally running dispatch commands.

2. **start-kanban.sh auto-kill**: Before launching the zellij session, `start-kanban.sh` now checks for stray dispatch/daemon processes and kills them:
```bash
for proc_pattern in "hermes.*kanban.*dispatch" "hermes.*kanban.*daemon"; do
    if pgrep -f "$proc_pattern" >/dev/null 2>&1; then
        echo "⚠️  发现残留的 kanban dispatch/daemon 进程，正在杀掉："
        pgrep -af "$proc_pattern"
        pkill -f "$proc_pattern" 2>/dev/null || true
        sleep 1
        if pgrep -f "$proc_pattern" >/dev/null 2>&1; then
            pkill -9 -f "$proc_pattern" 2>/dev/null || true
        fi
    fi
done
```
This ensures no headless dispatch workers are running when the interactive kanban session starts, preventing claim competition.

3. **Dispatch is not auto-started**: `dispatch_once` and `dispatch_loop` are NOT automatically started by the gateway or any systemd service. They only run when a human (or AI agent) explicitly calls `hermes kanban dispatch` or `hermes kanban daemon`. There is no background daemon that spawns headless workers without user action.

**Key insight**: The `claim_lock` format distinguishes the claimer:
- `hostname:pid:reasonix-interactive` → claimed by interactive listener (correct, visible in TUI)
- `hostname:pid` (no suffix) → claimed by `_default_spawn` via dispatch daemon (headless, invisible)

**If both must coexist**: Set `max_in_progress_per_profile` in the dispatch daemon config, or restrict the daemon to only handle profiles that don't have interactive listeners. The interactive listeners should be the primary claim path for any profile with a visible pane.

**Recovery**: If a task was claimed by the dispatch daemon and you want it in the visible pane:
1. Reclaim the task: `hermes kanban --board <slug> reclaim <id> --reason \"headless dispatch; need visible pane\"`
2. The interactive listener will pick it up on the next poll cycle

### Failure 9: Go-based TUI tools (reasonix, deepseek-tui) need PTY — bubbletea raw-mode crash

**Symptom**: A kanban pane that runs `reasonix` or `deepseek-tui` exits immediately with `bubbletea: error opening TTY` or `Failed to enable raw mode: No such device or address`. The watcher sees `reasonix exited rc=0` (clean exit) and stops. If all panes exit, zellij `auto_close` deletes the entire session, making it look like a total system failure rather than a single TTY issue.

**Root cause**: Both `reasonix` and `deepseek-tui` are Go binaries using the bubbletea TUI framework, which requires opening `/dev/tty` and enabling terminal raw mode. When the process's stdin is a pipe (not a PTY), `open("/dev/tty")` fails with `OSError: No such device or address` and bubbletea crashes before rendering anything.

**When this happens**:
- **Hermes agent terminal**: `stdin.isatty()` is `False` — the agent's terminal tool provides a pipe, not a PTY. Go TUI tools cannot run here.
- **Direct shell without TTY**: `bash -c 'reasonix chat --dir /path'` from a non-interactive context (cron, `ssh host 'command'`, subprocess without PTY) fails the same way.
- **Zellij panes with `bash -lc`**: Usually works because zellij creates a PTY for each pane, and `bash -lc` inherits it. But if the pane's command chain breaks the PTY inheritance (e.g., nested `bash -c` inside `bash -lc`), `/dev/tty` may become unavailable to child processes.

**Diagnostic**: Test TTY availability before launching a Go TUI tool:
```bash
# In the target environment, check if /dev/tty is accessible:
python3 -c "import sys; print('isatty:', sys.stdin.isatty()); open('/dev/tty'); print('/dev/tty: OK')"
# If this prints isatty: False and raises OSError, Go TUI tools will crash.
```

**Fix options** (in order of preference):
1. **Use zellij panes directly** — zellij provides a PTY; launch with `bash -lc 'reasonix ...'` in the KDL layout (current approach in `kanban-launcher.kdl`).
2. **Use `script -qc` as PTY wrapper** — `script -qc 'reasonix chat --dir /path' /dev/null` allocates a pseudo-terminal for the child process. This works even when stdin is a pipe.
3. **Don't launch Go TUI from Hermes agent** — the Hermes agent's `terminal()` tool cannot provide a PTY. Use `start-kanban.sh` from a real terminal instead.

**Zellij auto_close cascade**: When all panes in a zellij session exit (even if they exit normally with rc=0), the session is automatically deleted if `auto_close` is enabled (default). This creates a misleading "total failure" appearance when the real cause was a single pane's TTY crash triggering others to exit. To diagnose: check `zellij list-sessions` — if the session is gone entirely, it was likely an auto_close cascade, not a simultaneous failure of all components.

**Key insight**: A process exiting with `rc=0` in a watcher log doesn't mean "everything is fine." For Go TUI tools, `rc=0` after a bubbletea crash means "the TUI framework exited cleanly after failing to initialize." Always check the pane content or stderr for TTY/raw-mode errors when a pane exits unexpectedly with rc=0.

**Reference**: See `references/go-tui-pty-requirement-2026-06-07.md` for the full diagnostic trace of the 2026-06-07 cascade failure.

### Failure 10: Watcher exits cleanly (rc=0) and is NOT restarted — tasks go unclaimed

**Symptom**: A task is `ready` on the board, but no pane picks it up. The watcher log shows `interactive watcher stopped` followed by `watcher exited cleanly rc=0 while Reasonix is still running; not restarting`. The pane sits idle at a YOLO prompt indefinitely.

**Root cause**: The launcher's `_should_restart_watcher()` only restarted on **abnormal** exit (rc≠0). When the watcher exited with rc=0 (clean shutdown), the launcher treated it as intentional and did NOT restart it. But the watcher can exit cleanly for non-intentional reasons:

1. **DB corruption cascade → recovery → clean exit**: The watcher hits DB errors, crashes, gets restarted, recovers, processes the active task, then exits with rc=0 when done. The launcher sees rc=0 and stops restarting permanently.

2. **Watcher receives SIGTERM/SIGINT while TUI is still alive**: External process cleanup (stale worker kill, OOM killer, manual intervention) sends SIGTERM. The watcher's `while not _STOP` loop ends, returning rc=0. The launcher does not restart.

**Fix applied** (2026-06-07, reasonix_kanban_interactive.py):

Modified `_should_restart_watcher()` to restart the watcher **regardless of exit code** as long as the TUI (Reasonix) process is still alive. The only case where restart is skipped is when the TUI has also exited (no point running a watcher for a dead TUI).

```python
def _should_restart_watcher(returncode: int | None, *, reasonix_alive: bool = True) -> bool:
    if returncode is None:
        return False  # Process hasn't exited yet
    if not reasonix_alive:
        return False  # TUI is gone; no point restarting watcher
    return True  # TUI alive + watcher exited → always restart
```

The launcher's main loop passes `reasonix_alive=True` because it only reaches the watcher-restart check when `reasonix_proc.poll()` returned None (TUI still alive). The log message was also updated from `"watcher exited cleanly rc=0 while Reasonix is still running; not restarting"` to `"watcher exited rc={rc} and Reasonix is gone; not restarting"` (the only case where restart is skipped).

**Applied to**: reasonix_kanban_interactive.py (2026-06-07). **Open TODO**: patch `_should_restart_watcher` in `codex_kanban_interactive.py` and `deepseek_kanban_interactive.py` to accept `reasonix_alive`/`tui_alive` kwarg and restart on rc=0 when TUI is alive. The launcher loop in each file also needs the corresponding call-site update.

### Failure 11: Pane screen residual text causes permanent "skip claim"

**Symptom**: The watcher log shows repeated `skip claim: Reasonix pane still shows an active/pending Kanban prompt` every poll interval (6s), even though the TUI is clearly idle at a YOLO prompt. The watcher never claims any task.

**Root cause**: After a task completes, the Reasonix TUI may still display residual text from the previous Kanban injection (e.g., `KANBAN_TASK_BOUNDARY`, `hermes kanban 已领取任务`, `完成后必须运行`). The watcher's `_pane_can_accept_new_kanban_task()` reads the pane screen via `zellij dump-screen`, takes the last 20 non-empty lines, and checks for BUSY markers. If residual Kanban injection text is in that tail, the function returns False (busy) even though the TUI is at an idle prompt. The IDLE markers (YOLO, 已跳过批准) are present but the old logic gave BUSY markers absolute priority — any BUSY marker, even stale, overrode all IDLE markers.

This is especially likely when:
- The AI completed the task but the conversation history still shows the injection text
- The TUI renders a long conversation where the injection text scrolls but remains in the last 20 non-empty lines
- The AI's response includes references to "Kanban" in its output (e.g., "hermes kanban --board egomotion4d context t_xxx")

**Diagnostic**: Check the watcher log for a long sequence of `skip claim` entries with no `claimed+injected` in between:
```
[14:51:56] skip claim: Reasonix pane still shows an active/pending Kanban prompt
[14:52:01] skip claim: Reasonix pane still shows an active/pending Kanban prompt
...  (continues for hours)
```

**Fix applied** (2026-06-07, reasonix_kanban_interactive.py):

Introduced `_STALE_KANBAN_INJECTION_MARKERS` — markers that come from a previous task injection and are NOT evidence of an active task. These are distinguished from truly busy markers (tool execution status like `processing`, `run running`, etc.).

Modified `_pane_can_accept_new_kanban_task()` with a two-tier logic:

1. **Queued input is always a hard block** — `pending inputs` / `edit last queued message` means the TUI has buffered but unprocessed input.

2. **When an IDLE marker is present** (YOLO, 已跳过批准, etc.):
   - Only check **non-stale BUSY markers** (`processing`, `run running`, `read running`, `write running`, `edit running`)
   - Ignore stale Kanban injection markers (`kanban_task_boundary`, `hermes kanban 已领取任务`, `完成后必须运行`)
   - If no non-stale BUSY markers → **allow injection** (idle with stale text is still idle)

3. **When no IDLE marker is present**:
   - Fall back to the original strict check: no BUSY markers at all → allow injection

```python
_STALE_KANBAN_INJECTION_MARKERS = (
    "kanban_task_boundary",
    "hermes kanban 已领取任务",
    "完成后必须运行",
)

def _pane_can_accept_new_kanban_task(text: str) -> bool:
    # ... queued input always blocks ...
    has_idle = any(marker in tail_norm for marker in _REASONIX_IDLE_MARKERS)
    if has_idle:
        truly_busy = tuple(m for m in _REASONIX_BUSY_MARKERS
                          if m not in _STALE_KANBAN_INJECTION_MARKERS)
        if any(marker in tail_norm for marker in truly_busy):
            return False  # Genuinely busy (tool executing)
        return True  # Idle with only stale markers — safe
    return not any(marker in tail_norm for marker in (*_REASONIX_BUSY_MARKERS, *_REASONIX_QUEUED_INPUT_MARKERS))
```

**Key insight**: The presence of an IDLE marker (YOLO prompt) is strong evidence the agent is waiting for input. Stale Kanban injection text in the scrollback does NOT mean the agent is still processing — it means the screen hasn't been cleared. Only active tool-execution status markers indicate genuine busyness.

**Applied to**: reasonix_kanban_interactive.py. Codex and deepseek listeners do not use this screen-check pattern (codex doesn't check screen at all; deepseek has its own markers). If they add similar checks in the future, the same stale-marker distinction should apply. The `_STALE_KANBAN_INJECTION_MARKERS` tuple and two-tier logic in `_pane_can_accept_new_kanban_task` are ready to copy.

### Failure 13: Slow-API idle-reclaim loop — watcher reclaims every idle_pane_reclaim_s because it can't distinguish "waiting for API" from "truly idle"

**Symptom**: The watcher log shows `idle pane reclaim` events every 600s (or whatever `--idle-pane-reclaim-s` is set to), with the same task being reclaimed and re-injected over and over. The TUI (reasonix/deepseek) is actually working — it sent an API request and is waiting for the response — but the screen doesn't update during the wait, so the watcher's idle detection fires.

**Observed pattern** (implementer, 2026-06-08):
```
10:00:12  claimed+injected t_037b1e0c
10:01:07  idle pane observed for active task t_037b1e0c
10:11:09  idle pane reclaim t_037b1e0c: idle 602s >= 600s
10:11:19  claimed+injected t_037b1e0c (re-injected same task)
10:11:49  idle pane observed
10:21:51  idle pane reclaim (again)
... (16 reclaim cycles over 1h23m, ¥7.60-8.24 spent total)
11:23:38  reasonix exited rc=-15 (external SIGTERM killed all panes)
```

**Root cause**: The watcher's idle detection relies on screen content not changing. When the TUI sends an API request to the LLM provider and waits for a streaming response:
1. Screen may show "思考了 N 秒" or a partial tool call, but no new output for several minutes
2. The watcher counts this as "idle time" and accumulates toward `idle_pane_reclaim_s`
3. After 600s of no screen changes, it reclaims the task and re-injects
4. This **interrupts the agent's work in progress** — the API call is cancelled, context resets
5. The agent starts over, hits the same slow API call, and the cycle repeats

**Why this is destructive**: Unlike the crash-loop reclaim (Failure 6, every 8 seconds), this one happens at the configured interval (600s default). Each reclaim interrupts real work, wastes tokens (¥0.05 per cycle for thinking + API call initiation), and prevents the agent from ever completing its task. The agent might spend ¥7+ across 16 cycles but never finish because each cycle is cut short.

**Distinguishing from crash-loop reclaim (Failure 6)**:
- Failure 6: reclaim every ~8 seconds (startup_delay_s), caused by DB crash → watcher restart → immediate re-reclaim
- Failure 13: reclaim every ~600 seconds (idle_pane_reclaim_s), caused by screen appearing idle during API wait

**Mitigation options** (in order of effectiveness):

1. **Increase `--idle-pane-reclaim-s`**: Change from default 600s to 1800s or 3600s. This gives the agent more time for slow API responses. Drawback: genuinely stuck agents take longer to detect and reclaim.

2. **Session file activity check**: Before declaring idle, check the agent's session log file (e.g., `~/.config/reasonix/sessions/<session>.jsonl`) for recent writes. If the file was modified within the last N seconds, the agent IS working (writing tool calls/results) even if the screen hasn't updated. This is the most reliable signal.

3. **API request tracking**: Check if the agent process has an active network connection to the LLM provider (e.g., `ss -tnp | grep <pid> | grep api.deepseek.com`). Active connection = waiting for response, not idle.

4. **Screen content semantic analysis**: Distinguish "思考了 N 秒" / "processing" / "tool call in progress" from genuine idle markers. The current `_pane_looks_idle()` already has some of this via `_REASONIX_BUSY_MARKERS`, but "thinking" display may not be in the busy marker list.

**Immediate action**: Set `--idle-pane-reclaim-s 1800` (30 minutes) in `start-kanban.sh` for reasonix panes. This gives agents enough time for complex API responses (DeepSeek v4-pro planner model can take 2-5 minutes per response) while still detecting genuinely stuck agents within 30 minutes.

**Long-term fix**: Add session-file timestamp check to `_pane_looks_idle()`. If the reasonix session jsonl was written within the last 120 seconds, the agent is active even if the screen hasn't changed. This would allow keeping `idle_pane_reclaim_s` at 600s for genuinely stuck detection while never interrupting active API-wait periods.

**Reference**: See `references/slow-api-idle-reclaim-loop-2026-06-08.md` for the full diagnostic trace of the 2026-06-08 incident, including the 16-cycle reclaim pattern, simultaneous process death analysis, and double-launcher finding.

### Failure 14: Reasonix max_steps=6 default silently overrides global config

**Symptom**: A kanban pane's Reasonix pauses after 6 tool-call rounds with a message like "paused after 6 tool-call rounds (agent.max_steps)". The agent is mid-task and cannot continue without manual intervention. This is especially common for complex tasks (remote execution + verification + code search) that need 10+ tool-call rounds.

**Root cause**: Reasonix's built-in default `max_steps` is 6. The config resolution order is `flag > ./reasonix.toml > ~/.config/reasonix/config.toml > built-in defaults`. Even if the global config (`~/.config/reasonix/config.toml`) sets `max_steps = 0` (unlimited), the project-level `reasonix.toml` takes priority — and if it doesn't specify `max_steps`, the built-in default of 6 wins.

The kanban listener's `_write_role_instructions()` auto-generates `reasonix.toml` with only `system_prompt_file`. Without an explicit `[agent] max_steps = 0`, the default of 6 applies to all panes using this project workspace.

**Fix applied** (2026-06-08): `_write_role_instructions()` now appends `[agent] max_steps = 0` to the generated `reasonix.toml`, ensuring the global config's unlimited setting is preserved at the project level.

**Pitfall: Auto-generated configs must include all non-default settings.** When a tool generates a config file that overrides the user's global config, it must explicitly set any values that differ from built-in defaults. Otherwise, the generated file silently degrades behavior (e.g., `max_steps` goes from 0 in global to 6 from built-in default).

**Diagnostic**: If a kanban pane pauses with "max_steps" in the message:
1. Check `cat <workspace>/reasonix.toml` — is `[agent] max_steps` present?
2. If missing, the built-in default of 6 is active
3. Fix: add `[agent]\nmax_steps = 0` to the toml, and ensure `_write_role_instructions()` includes it

**Applied to**: reasonix_kanban_interactive.py `_write_role_instructions()` (2026-06-08). Codex and deepseek listeners don't use `reasonix.toml` — this pitfall is Reasonix-specific.

**Also required**: `auto_plan = "off"` with `max_steps = 0` (added 2026-06-08).

**Watcher flock bug (2026-06-08)**: `start_new_session=True` breaks flock → watcher refuses to start. See `references/watcher-flock-bug3-2026-06-08.md`.

### Zellij session recovery: dead panes require full session recreation

When a zellij session has panes whose shell processes have exited (e.g., watcher crash, Reasonix exit, OOM kill), `zellij write-chars` can send text to the pane but no shell is running to execute it. The pane appears alive (title remains) but is effectively dead — no new commands can run.

**Diagnostic**: If `zellij --session <name> action list-clients` only shows 1 client (coordinator), the other panes' processes are gone. `ps aux | grep reasonix_kanban | grep -v grep` shows no launcher/watcher processes.

**Recovery**: Kill the entire zellij session and recreate from the layout:
```bash
zellij kill-session <name>
zellij delete-session <name> 2>/dev/null || true
# Recreate with PTY support (script -qc provides a PTY)
script -qc "zellij --session <name> --new-session-with-layout <layout-name>" /dev/null
```

**Pitfall: `zellij --new-session-with-layout` needs a TTY.** Running it from the Hermes `terminal()` tool (which provides a pipe, not a PTY) creates the session but it immediately exits (shows as "EXITED" in `list-sessions`). Use `script -qc` as a PTY wrapper, or start from a real terminal.

**Pitfall: Don't try to revive individual panes with `zellij write-chars`.** If the pane's shell process has exited, writing characters does nothing useful — there's no shell to interpret them. The only reliable recovery is full session recreation.

### Double-launcher risk during manual restart

**Symptom**: Two launcher processes start for the same pane, creating duplicate watchers. Both watchers try to claim/inject tasks into the same pane. The watcher log shows two consecutive `launcher starting watcher` entries at slightly different times (e.g., 10:00:12 and 10:00:27).

**Root cause**: During manual intervention (upgrade, restart), the operator may start a reasonix/watcher manually (via `zellij write-chars` or direct command) while `start-kanban.sh` or the auto-restart mechanism also starts one. The second launcher doesn't check for an existing process on the same pane.

**Impact**: The second watcher's `adopt_orphaned_running_task` may find a task already claimed by the first watcher and "adopt" it (changing the PID), which is benign but adds confusion. Both watchers poll the same pane, potentially causing race conditions on claim/inject.

**Prevention**: Before manually starting a launcher/watcher, check if one is already running for the target pane:
```bash
# Check existing processes for this pane
ps aux | grep "kanban_interactive.*--profile <role>" | grep -v grep
```
If one exists, kill it first, then start the new one. Don't start a second one alongside.

**Diagnostic**: When the watcher log shows two consecutive launcher starts within seconds, one of them is a duplicate. Check which PID survived and whether the other died or stayed alive causing interference.

The watcher logs are at `~/.hermes/kanban/boards/<board>/logs/<agent>-interactive-<profile>.log`. When diagnosing a crashed interactive pane:

1. Check the log tail: `tail -50 ~/.hermes/kanban/boards/<board>/logs/reasonix-interactive-implementer.log`
2. Look for patterns: `skip claim` loops, `OperationalError`, `reasonix exited rc=`, `watcher exited unexpectedly`
3. Check if `.corrupt.*.bak` files exist in the board DB directory — multiple corrupt backups indicate recurring DB issues
4. Verify DB integrity: `sqlite3 ~/.hermes/kanban/boards/<board>/kanban.db "PRAGMA integrity_check;"`
5. Check event log for double-claims: query `task_events` for the stuck task to see if multiple panes are alternating claims

For a complete diagnostic playbook including process tree inspection, health check script, and key log patterns, see `references/watcher-health-diagnostics.md`.

## Injection prompt architecture — fixed rules vs. task-specific content

**Principle**: When injecting task prompts into TUI panes via `zellij write-chars`, keep the injection minimal. Fixed rules belong in `role-instructions.md` (injected via `reasonix.toml`'s `system_prompt_file`). The per-task injection should only contain task-specific info.

**Three layers** (each serves a distinct purpose, no duplication):

| Layer | Content | Mechanism | When loaded |
|-------|---------|-----------|-------------|
| `role-instructions.md` | Common rules + role description | `system_prompt_file` in `reasonix.toml` | Always present, per-watcher startup |
| zellij `inject_text` | Task boundary marker + ID + role + prompt file + finish command | `zellij write-chars` | Per-task, ~120 chars |
| prompt file (`build_interactive_prompt`) | Task ID/title + board/role/workspace + complete/block commands + task context | Written to `.reasonix-kanban/<board>/<role>/` | Per-task, ~500 chars |

**User preference (2026-06-08)**: "固定的内容应该放入agents.md定义的角色说明里，注入以简洁的提示就可以了" — fixed content goes in role instructions, injection must be concise. "描述应当简洁完整" — descriptions should be concise AND complete, not verbose.

**Implementation** (reasonix_kanban_interactive.py):
- `role_guidance(profile)` → common rules (7 items, ~280 chars, bullet-list format) + one-line role description. Written to `role-instructions.md` on watcher startup.
- `build_interactive_prompt()` → ~15 lines with only task-specific content (task ID/title, board/role/workspace, complete/block commands, context).
- `inject_text` → 3 lines: `KANBAN_TASK_BOUNDARY`, task ID+title+role+file, finish command. No "上下文衔接规则" — those rules belong in `role-instructions.md`.
- `_write_role_instructions()` → writes both `role-instructions.md` and `reasonix.toml` with `system_prompt_file` pointing to it.

**Key design rule**: `role_guidance()` common section must be CONCISE. Use bullet-list format (one rule per line), not numbered paragraphs with examples. Project-specific rules (like AGENTS.md commands) are referenced by name ("项目命令遵守仓库 AGENTS.md") rather than inlined verbatim. Role-specific guidance (e.g., planner's plan quality requirements) goes in the role line, not the common section — implementer doesn't need planner rules.

**Result**: Total injection went from ~5000 chars (inject_text ~800 + prompt ~2000 + role-instructions ~100) to ~1000 chars (inject_text ~120 + prompt ~500 + role-instructions ~600-823).

**Pitfall: Don't put board/task-specific placeholders in `role-instructions.md`.** The role instructions are written once per watcher startup and shared across all tasks. Placeholders like `<board>` or `<task_id>` won't be filled in — use generic descriptions (e.g., "hermes kanban --board <board> context <task_id>") instead.

**Pitfall: `inject_text` must not duplicate rules already in `role-instructions.md`.** Before this refactoring, `inject_text` contained 12 lines of "上下文衔接规则" (context continuation rules about kill -0 loops, remote process management, etc.) that were already covered by `role_guidance()`. Every character in `inject_text` costs tokens on every subsequent TUI turn — duplication is a direct waste.

**Pitfall: Don't inline project-specific rules in `role_guidance()`.** The original common section had a full paragraph about Egomotion4D's conda env and gpuserver commands. This is project-specific and already covered by AGENTS.md. Replace with a generic reference ("项目命令遵守仓库 AGENTS.md"). Different projects use the same kanban system — hardcoding one project's rules breaks others.

## Board status summarization (user asks "summarize X-line results" or "any unmentioned completed tasks?")

When the user asks for a summary of a specific work line or to find tasks not yet discussed:

### Step 1: Query all tasks on the board
```bash
hermes kanban --board <slug> list --json | python3 -c "
import json, sys
data = json.load(sys.stdin)
# Filter by title keywords matching the line (A/B/C/P0/MegaSaM/Turn)
# Exclude coordinator follow-ups
# Group by status
"
```

### Step 2: Cross-reference with already-discussed task IDs
Maintain a set of task IDs already mentioned in conversation. Filter them out to find "unmentioned" tasks. This avoids re-summarizing what the user already knows.

### Step 3: Pull summaries from task events
For each unmentioned task, extract the completion summary from the `runs` section (look for `completed` events with `summary` field). If no summary in events, check `Latest summary` from `hermes kanban show`.

### Step 4: Group by work line
Common groupings for Egomotion4D:
- **A line**: multi-frontend TSDF/Occ fusion closure, scene0 QA, metrics
- **B line**: ground dense reconstruction, ShelfOcc benchmarks, pose audit
- **C line**: SfM-for-NVS, RoMa2 geometry guidance, sparse depth refinement
- **P0 line**: pipeline orchestrator, schema loader, source agreement, TSDF, occ exporters
- **MegaSaM line**: M0-M4 implementation chunks
- **Turn line**: turn-window tracking, PnP edges, CoTracker benchmarks

### Step 5: Present concise summary
For each task: task ID, status, one-line result. For blocked tasks: explain the blocker. For done tasks: key finding or outcome.

### Pitfall: `list --json` returns all 300+ done tasks
Filter aggressively — exclude coordinator follow-ups (title starts with "Coordinator"), and only show tasks relevant to the user's question. Don't dump the full board.

### Pitfall: Task summaries are in events, not top-level fields
The `list --json` output doesn't include summaries. You need `show <id>` to get them, but that's one API call per task. For a quick overview, just show title + status + assignee. Only pull full summaries for tasks the user specifically asks about.

### Pitfall: Worker conclusions live in `comments`, not `runs[].summary`
Many implementer tasks complete with `result: null` and no `summary` in any run — the actual work description is in the `comments` array (author=`worker`). When extracting conclusions:
1. First check `runs[-1].summary` — if present, use it.
2. If no summary in runs, fall back to `comments` where `author == 'worker'` — these contain the implementer's self-description of what was done.
3. For critic/review tasks, `runs[-1].summary` is usually populated (critic completes with structured metadata).

Extraction pattern:
```python
data = json.load(sys.stdin)  # from `show <id> --json`
# Try runs first
for run in reversed(data.get('runs', [])):
    s = run.get('summary') or ''
    m = run.get('metadata') or {}
    if s or m.get('tests'):
        # Use this
        break
else:
    # Fall back to worker comments
    for c in data.get('comments', []):
        if c.get('author') == 'worker':
            # c['body'] has the work description
            break
```

This is especially common for tasks that were `reclaimed` (run outcome=`reclaimed`) — the reclaimed run has no summary, and the replacement run may also lack one if the worker commented instead of using `kanban_complete(summary=...)`.

### Failure 12: No external watchdog for watcher process health — IMPLEMENTED

**Problem**: The launcher inside each pane restarts its own watcher on crash (after Failure 10 fix), but if the launcher itself dies (OOM, segfault, DB corruption crash-loop exceeding MAX_WATCHER_RESTARTS), nothing restarts it. The pane becomes a zombie — the TUI may still be running but no new tasks are claimed.

**Implemented solution (2026-06-07)**: Added `_check_watcher_health()` in `hermes_cli/kanban_listener.py`, called from the coordinator's `/listen-kanban` loop every 180 seconds.

**Implementation details**:

1. **Function**: `_check_watcher_health(board: str, interval_seconds: float = 180.0) -> None` in `kanban_listener.py`
2. **Trigger**: Called in `_listener_loop` when `state.assignee == "coordinator"` and `task_info is None` (idle polling), gated by `_last_watcher_check` timestamp (180s minimum interval)
3. **Detection logic**:
   - Scans `/proc` for all `kanban_interactive` processes (launcher with `--auto-start` and watcher with `--watch-child`)
   - Matches processes to the target board via `HERMES_KANBAN_BOARD` env var or `--board` CLI arg
   - Extracts profile name from `--profile` CLI arg
   - Detects coordinator's hermes process separately (no launcher/watcher subprocess; hermes IS both)
   - Identifies anomalies: launcher PID alive but watcher PID missing
4. **Output**:
   - Anomaly detected: Red warning with profile name, launcher PID, and recovery command (`zellij kill-session --force kanban-<board> && bash scripts/start-kanban.sh`)
   - All healthy: `watcher health check OK (N panes)` info message
5. **Safety**: Only detects and warns — does NOT auto-restart (avoids conflict with launcher internal state). Recovery is manual.

**Files changed**:
- `hermes_cli/kanban_listener.py`:
  - Added `_last_watcher_check: float = 0.0` to `ListenerState` dataclass
  - Added `_check_watcher_health()` function (~110 lines)
  - Added 180s interval check in `_listener_loop` idle branch

**How to activate**: The coordinator's hermes process must be restarted to pick up the code change. The health check runs automatically once the coordinator is running with the updated code.

**User preference (2026-06-07)**: User prefers reusing an existing periodic mechanism rather than creating a new dedicated cron job. The `/listen-kanban` loop (15s poll for tasks, 180s for watcher health) serves as the polling backbone without needing a separate cron job.

**Future enhancement**: Auto-restart of dead watchers (currently only detection + warning). This would require careful coordination with the launcher's internal state to avoid conflicts.

## Kanban watchdog monitoring (scheduled cron checks)

When running as a scheduled watchdog (e.g., hourly cron for Egomotion4D kanban), follow this structured checklist. The goal is to detect anomalies and act on them, NOT to produce long reports. If everything is healthy, output exactly `[SILENT]` (nothing else) to suppress delivery.

### Step 1: Board overview
```bash
hermes kanban --board <slug> stats
hermes kanban --board <slug> list --status ready --json
hermes kanban --board <slug> list --status running --json
hermes kanban --board <slug> list --status blocked --json
```

### Step 2: Check specific tracked tasks
For each task ID in the current wave, run `hermes kanban --board <slug> show <id>` and note:
- `done` — proceed, no action needed
- `ready` — check time since creation; if >10 min and no dispatch, consider dispatching
- `running` — check heartbeat timestamps; if >60 min stale with no new heartbeat/comment/result, flag as stalled
- `blocked` — check if the blocker is fixable (missing artifact, wrong assignee) vs genuinely waiting on upstream

### Step 3: Anomaly detection rules

| Condition | Action |
|-----------|--------|
| `ready` task >10 min unclaimed | Run `dispatch --dry-run --json`; if reasonable, run `dispatch --max 1 --json`. Don't抢 tasks assigned to other roles. |
| `running` task >60 min stale heartbeat | Check if worker PID is still alive: `ps -p <worker_pid> -o pid,stat,etime`. If PID is dead but task shows `running`, the lock is stale — `dispatch` will reclaim and respawn automatically. Don't manually reclaim unless dispatch doesn't handle it. |
| `running` task with dead worker PID | `hermes kanban --board <slug> dispatch --dry-run --json` will show `reclaimed: N` for stale-locked runs. If it reports reclaims, run `dispatch --max <N> --json` to reclaim+respawn in one step. This is cleaner than manual `reclaim` + separate `claim`. |
| `blocked` task with fixable cause | Create a fix + re-review task if none exists |
| Critic didn't check output artifacts first | Add precise comment to critic task; create fix + re-review if needed |
| Final planner review ready but not started | Claim/trigger planner review to keep pipeline flowing |
| Campaign completed (all iterations done) | Verify final review is `done` and no orphan tasks remain; output `[SILENT]` |

### Step 4: Act on anomalies
- **Dispatch**: `hermes kanban --board <slug> dispatch --max 1 --json`
- **Comment**: `hermes kanban --board <slug> comment <id> --body "precise observation"`
- **Create fix task**: `hermes kanban --board <slug> create "fix + re-review: ..." --assignee implementer --parent <critic-id>`
- **Reclaim stalled run**: `hermes kanban --board <slug> reclaim <run-id> --reason "stalled >60min"`

### Step 5: Output
- **If anomalies found**: Report concisely (what, why, action taken)
- **If no anomalies**: Output exactly `[SILENT]` — no additional text

### Pitfalls for watchdog monitors
- **Don't treat completed campaigns as stalled.** After 6 iterations complete with a final review `done`, the board naturally has 0 ready/running tasks for that campaign. New campaigns (FPMR, etc.) may have their own separate task IDs — don't confuse the two.
- **Don't flag test/experimental tasks.** Tasks like "test-planner: ls 当前目录" are non-research and should be ignored.
- **Don't count blocked GFPR/other-campaign tasks as W13-W16 anomalies.** Different campaigns are independent.
- **Don't dispatch tasks whose assignee isn't ready for auto-dispatch.** Some tasks (planner reviews) require human judgment before starting.
- **Don't produce verbose reports.** The watchdog is a health check, not a status dashboard. Keep it under 500 words unless anomalies require detailed explanation.
- **Dead worker PID ≠ task failure.** A worker process may die (OOM, killed, host reboot) while the task is still `running` in Kanban. Always verify with `ps -p <pid>` before assuming the task is stalled. If the PID is dead, `dispatch` will automatically reclaim the stale lock and respawn — no manual intervention needed beyond triggering dispatch.
- **Mass-stall ≠ single stale heartbeat.** When one listener PID claims 5+ tasks and all show zero heartbeats, the listener itself is the bottleneck (alive but not processing). Reclaiming all at once is correct; dispatching more workers is not. Verify one worker works before scaling back up.
- **`dispatch --dry-run --json` is your diagnostic.** Before committing to any dispatch, always run dry-run first. The `reclaimed` count tells you how many stale-locked runs the dispatcher found; the `spawned` list previews which tasks will get new workers. Only run `dispatch --max N --json` after confirming the dry-run plan is sensible.

## CLI fallback (for scripting)

Every tool has a CLI equivalent for human operators and scripts:
- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>] [--idempotency-key <wave-prefix>-<task-slug>]`
- `kanban_update` ↔ `hermes kanban update <id> [--title "new title"] [--body "new body"] [--body-file <path>] [--priority N] [--assignee <profile>] [--reopen] [--json]` (for modifying non-done, non-archived tasks; add `--reopen` to reopen a done task: done→ready, closes dangling runs, records `reopened` audit event; REFUSED on archived — use `promote` first)
- `kanban_edit` ↔ `hermes kanban edit <id> --result "..." --summary "..."` (for backfilling results on done tasks; does NOT work on running tasks when you own the lock)
- etc.

Use the tools from inside an agent; the CLI exists for the human at the terminal.

## Handling rework returns (打回返工)

When a task you previously completed reappears as `ready` with new comments, it means a critic/planner reviewer rejected your output and sent it back for rework. The comment thread contains structured review feedback with three required elements: problem identification, rework requirements, and updated acceptance criteria.

**What to do when you receive a rework return:**

1. **Read the comment thread** — look for comments with `【REVIEW 不通过】` prefix. These contain the specific problems and what needs fixing.
2. **Check the updated body** — the task body may have been modified with a `## 返工要求` section listing additional requirements.
3. **Don't repeat the failed approach** — the review explicitly identifies what was wrong. Fix the specific issue, don't redo the entire task from scratch.
4. **Address each point individually** — verify each review complaint is resolved before completing again.
5. **Check iteration count** — max 3 reworks per task. If this is the 3rd return, the review comment will explicitly say "这是第 3 次打回". After 3 reworks, the task should be blocked for planner reassessment.

**Common patterns:**
- Metrics missing from output → re-run eval with correct fields
- Output contract incomplete → add missing files/data
- Code quality issues → fix specific bugs, don't rewrite whole module
- Visual artifacts missing → regenerate specific visualizations

**Anti-patterns:**
- Don't complete the task again without actually fixing the review issues — the same reviewer will catch it again
- Don't create a new task instead of fixing the existing one — violates UPDATE-over-recreate rule
- Don't ignore the comment thread and guess what "might" be wrong — the reviewer gave explicit instructions

For the orchestrator-side rework flow (how to send a task back), see `kanban-orchestrator` skill → `references/kanban-task-rework-flow.md`. The key command is `hermes kanban update <task_id> --reopen [--body "..."]` which transitions done→ready, closes dangling runs, and records the audit event in one step.

## UPDATE-first rule (for planner/coordinator roles)

When working in a planner or coordinator role that creates and publishes kanban tasks:

- **If a task's body/title/priority needs fixing after creation, use `hermes kanban update <task_id> --body "..."` instead of creating a duplicate task.** Creating a duplicate and archiving the original is a HARD RULE violation per the kanban-orchestrator anti-temptation rules. For done tasks, add `--reopen` to allow the update.
- **The only exception**: when the task has already been claimed/started by a worker. In that case, block the original first, then create a replacement with a clear `supersedes` comment.
- **Always use `--idempotency-key`** on `kanban create` calls in batch/wave scripts to prevent duplicate creation on retry.
