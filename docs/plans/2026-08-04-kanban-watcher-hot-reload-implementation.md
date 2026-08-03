# Kanban Watcher Safe Hot Reload Implementation Plan

> **For agentic workers:** REQUIRED: Use `superpowers:executing-plans` in the current checkout. Do not create a worktree/branch or mechanically spawn one subagent per task; use Luna only for a bounded read-only audit when its expected benefit exceeds startup cost.

**Goal:** Add watcher-only rolling hot reload that preserves a running Kanban claim and agent/TUI session while preventing duplicate claim loops across launcher and supervisor paths.

**Architecture:** A small Kanban runtime module owns watcher identity, local `flock`, reload handoff and ACK files. `BaseInteractiveListener` installs the signal/checkpoint/self-exec lifecycle, an extensionless local CLI performs scoped rolling reload, and the supervisor becomes board/session-scoped with a second pre-restart discovery. All state-changing behavior is exercised through subprocess/temporary-runtime tests rather than source-text assertions.

**Tech Stack:** Python stdlib (`fcntl`, `signal`, `os.execve`, `json`, `hashlib`, `pathlib`, `/proc`), SQLite task identity checks, Bash launch wrappers, pytest via `scripts/run_tests.sh`.

**Design:** `docs/design/2026-08-04-kanban-watcher-hot-reload.md` at commit `5e3b6a2`.

---

## Chunk 1: Runtime identity and singleton foundation

### Task 1: Replace the invalid Codex-home source-text test

**Files:**
- Create: `local/lib/codex_role_home.sh`
- Modify: `local/bin/start-kanban.sh:947-954`
- Modify: `tests/test_start_kanban_codex_agents.sh`

- [ ] Extract `ensure_codex_role_home <real_home> <role_home>` into `local/lib/codex_role_home.sh`; it creates `sessions/` and links existing shared files/directories, including `agents`, without overwriting existing entries.
- [ ] Rewrite the shell test to use two `mktemp -d` homes, create a fake `agents/luna.toml`, call the function, and assert the role home resolves the same file. It must not read `start-kanban.sh` or touch `/home/wyr`.
- [ ] Run `bash tests/test_start_kanban_codex_agents.sh`; expected PASS.
- [ ] Run `bash -n local/lib/codex_role_home.sh local/bin/start-kanban.sh`; expected no output.
- [ ] Commit only these files: `fix(kanban): test Codex role home sharing behavior`.

### Task 2: Add watcher identity and local lock runtime

**Files:**
- Create: `plugins/kanban/watcher_runtime.py`
- Create: `tests/test_kanban_watcher_runtime.py`

- [ ] Write RED tests for `WatcherIdentity.from_values()` requiring non-empty board/profile/session/pane and for stable distinct digests.
- [ ] Write a subprocess RED test proving two processes using the same identity cannot both acquire the lock, while different pane IDs can.
- [ ] Write a RED test that a leftover metadata file without a held kernel lock does not block acquisition.
- [ ] Implement `WatcherIdentity`, `runtime_root(env, uid)`, `WatcherLock.acquire()`, metadata validation and context-manager cleanup. Prefer owned `XDG_RUNTIME_DIR`; fall back to a mode-`0700` UID-specific directory under `tempfile.gettempdir()`. Never hardcode `~/.hermes`.
- [ ] Store metadata `{pid, proc_start_time, identity, code_revision}` with mode `0600`; use `fcntl.LOCK_EX | LOCK_NB` for correctness.
- [ ] Run `scripts/run_tests.sh tests/test_kanban_watcher_runtime.py -q`; expected all PASS and no writes outside pytest temp roots.
- [ ] Commit: `feat(kanban): add watcher identity locks`.

### Task 3: Enforce one cold-start claim loop

**Files:**
- Modify: `plugins/kanban/base_listener.py:watcher_main`
- Modify: `tests/test_kanban_watcher_runtime.py`
- Modify: relevant listener tests only if behavior requires it

- [ ] Add a RED fake-listener subprocess test: incomplete `--watch-child` identity exits nonzero before opening the board; two identical watchers yield one live loop and one lock-contention exit.
- [ ] At the start of `watcher_main`, construct the identity and acquire/adopt the runtime lock before DB access, heartbeat, claim or injection.
- [ ] Log incomplete identity and lock contention with the owner PID/start-time; do not delete a lock file or kill another process.
- [ ] Keep the lock object alive through the entire watcher function, including cleanup.
- [ ] Run `scripts/run_tests.sh tests/test_kanban_watcher_runtime.py tests/test_hermes_role_context_listener.py -q`.
- [ ] Commit: `fix(kanban): enforce one watcher claim loop per pane`.

---

## Chunk 2: Claim-preserving self-exec reload

### Task 4: Add handoff and ACK primitives

**Files:**
- Modify: `plugins/kanban/watcher_runtime.py`
- Modify: `tests/test_kanban_watcher_runtime.py`

- [ ] Write RED tests for mode-`0600` atomic handoff/ACK writes, nonce matching, PID/start-time checks and cleanup.
- [ ] Write RED tests for a mode-`0600` atomic reload request containing nonce, identity and expected lock owner; a bare signal without a valid request must not reload.
- [ ] Write RED tests that changing any of task ID, run ID, generation, claim lock, worker PID or watcher identity rejects restoration.
- [ ] Implement `ReloadRequest`, `ReloadHandoff`, atomic request/handoff/ACK helpers and bounded cleanup. These records contain no prompt, token or credentials.
- [ ] Add inherited-lock adoption: mark the held FD inheritable immediately before exec, pass FD + nonce in env, verify `fstat`/metadata/identity after exec, then return it to non-inheritable mode.
- [ ] Run `scripts/run_tests.sh tests/test_kanban_watcher_runtime.py -q`.
- [ ] Commit: `feat(kanban): add watcher reload handoff state`.

### Task 5: Implement safe-boundary SIGUSR1 self-exec

**Files:**
- Modify: `plugins/kanban/base_listener.py`
- Modify: `tests/test_kanban_watcher_runtime.py`
- Create: `tests/fixtures/fake_reload_watcher.py`

- [ ] Write a RED subprocess integration test: fake watcher holds a lock and active claim fixture, receives `SIGUSR1`, self-execs with unchanged PID, restores state, emits one ACK and never emits a second claim/inject marker.
- [ ] Write RED cases for a failed preflight and injected `execve` exception; the old process must continue heartbeat and must not enter active-claim cleanup.
- [ ] Install a watcher-only SIGUSR1 handler that sets a reload flag. Do not perform I/O or exec in the handler; at the safe checkpoint, require and verify the request JSON before starting reload.
- [ ] At loop-safe checkpoints, snapshot and revalidate active task/run/generation/claim-lock/worker-PID, run the same entry point with hidden `--reload-preflight`, then call `os.execve` using the same interpreter and argv.
- [ ] Add startup adoption before normal claim discovery. Restore heartbeat only after DB row equality; do not claim or inject during restoration. Write success ACK only after one successful heartbeat.
- [ ] Catch pre-exec and `execve` errors inside the active loop, reopen resources and continue old heartbeat. Ensure the outer `finally` does not clear the active claim on a failed reload attempt.
- [ ] Coalesce concurrent signals to the latest pending nonce.
- [ ] Run `scripts/run_tests.sh tests/test_kanban_watcher_runtime.py tests/test_hermes_role_context_listener.py -q`.
- [ ] Commit: `feat(kanban): hot reload watchers without losing claims`.

---

## Chunk 3: Operator command and scoped supervisor

### Task 6: Add the rolling reload command

**Files:**
- Create: `local/bin/hermes-kanban-reload-watchers`
- Create: `tests/test_kanban_reload_watchers.py`

- [ ] Write RED tests using a fake proc tree/runtime root: filter exact board/session/profile; identify only inner Python watchers; ignore `conda run` wrappers and unrelated shell text; select the lock owner PID.
- [ ] Write RED tests for stable profile order, atomic request-before-signal ordering, nonce-matched ACK validation, 15-second configurable timeout, nonzero exit and stop-on-first-failure behavior.
- [ ] Implement CLI arguments `--board`, `--session`, repeatable `--profile`, `--ack-timeout-s` and optional test-only injected proc/runtime roots.
- [ ] Generate a nonce and atomically write the request JSON, then send SIGUSR1 only after PID/start-time/key/lock-owner revalidation. Report `REQUESTED`, `ACK`, `FAILED`, `SKIPPED`; never send Zellij input or change Kanban task state.
- [ ] Make the script executable and link it into `~/.local/bin` only after tests pass and only if the destination is absent or already points to this source.
- [ ] Run `scripts/run_tests.sh tests/test_kanban_reload_watchers.py -q`.
- [ ] Commit: `feat(kanban): add rolling watcher reload command`.

### Task 7: Scope and singleton the supervisor

**Files:**
- Modify: `local/bin/kanban-watcher-supervisor.py`
- Modify: `tests/test_kanban_watcher_supervisor.py`

- [ ] Add RED tests that discovery excludes conda wrappers and other board/session keys; same supervisor identity is single-instance while different identities coexist.
- [ ] Add a RED restart-race test: a launcher replacement appears during `restart_delay`, so no supervisor child is spawned.
- [ ] Add `--board` and filter discovery before tracking/restart/cleanup. Acquire a supervisor lock keyed by board/session. Key restart counts by full watcher identity.
- [ ] After restart delay, rediscover and repeat live-replacement plus pane-ownership checks immediately before `Popen`.
- [ ] Preserve wrapper startup diagnostics without grouping wrappers as functional watchers.
- [ ] Run `scripts/run_tests.sh tests/test_kanban_watcher_supervisor.py -q`.
- [ ] Commit: `fix(kanban): isolate watcher supervisors by board session`.

### Task 8: Integrate scoped startup without global pkill

**Files:**
- Modify: `local/bin/start-kanban.sh:1154-1165`
- Modify: `local/bin/stop-kanban.sh` only if its current supervisor cleanup is also global
- Create or modify: a behavioral shell test under `tests/` using fake `pgrep/pkill/nohup/zellij` commands and temporary HOME

- [ ] Write a RED behavioral test proving starting board/session A does not signal supervisor B.
- [ ] Replace global `pkill -f kanban-watcher-supervisor` with exact board/session process matching and pass both `--board` and `--session` to the supervisor.
- [ ] Ensure dry-run does not start or stop any process.
- [ ] Run the new shell test plus `bash tests/test_start_kanban_reviewer_mode.sh` and `bash -n local/bin/start-kanban.sh local/bin/stop-kanban.sh`.
- [ ] Run `scripts/run_tests.sh tests/test_kanban_watcher_supervisor.py tests/test_kanban_reload_watchers.py -q`.
- [ ] Commit: `fix(kanban): scope watcher supervisor lifecycle`.

---

## Chunk 4: Regression and activation

### Task 9: Focused regression suite

**Files:**
- Modify only tests or implementation required by observed failures

- [ ] Run:

```bash
scripts/run_tests.sh \
  tests/test_kanban_watcher_runtime.py \
  tests/test_kanban_reload_watchers.py \
  tests/test_kanban_watcher_supervisor.py \
  tests/test_hermes_role_context_listener.py -q
```

Expected: all PASS, no retry/flaky summary.

- [ ] Run existing listener/result/composer-focused tests discovered with `rg`; include only real affected suites.
- [ ] Run `git diff --check` and inspect the full diff against the plan base; confirm the existing unrelated modification in `plugins/kanban/codex_listener/codex_kanban_interactive.py` was not staged or altered.
- [ ] If fixes were required, commit one focused regression commit; otherwise do not create an empty commit.

### Task 10: Isolated end-to-end smoke and one-time activation

**Files:**
- Update: `docs/design/2026-08-04-kanban-watcher-hot-reload.md` only if measured behavior differs

- [ ] Start a temporary fake board/session and a fake watcher with an active SQLite claim; execute `hermes-kanban-reload-watchers` and verify PID, run, generation and claim-lock remain unchanged, heartbeat advances, and injection count remains one.
- [ ] Verify two simultaneous same-key spawn attempts still yield one lock owner.
- [ ] Record exact commands and compact outputs in the final handoff; do not commit runtime artifacts.
- [ ] For the first deployment only, do not send SIGUSR1 to legacy watchers that lack the handler. Activate new code with one explicitly announced watcher-only maintenance restart at a safe boundary; do not rerun `start-kanban.sh` and do not restart agent/TUI panes.
- [ ] After activation, run the real reload command once on `kanban-egomotion4d`, verify ACK for every live role, unchanged pane commands/TUI sessions and continued heartbeat. If any role fails, stop rolling reload and retain the old watcher; do not force through.
- [ ] Commit any final documentation correction, then report commits, tests, smoke evidence and the exact operational command.

## Execution constraints

- Work in `/home/wyr/.hermes/hermes-agent-repo` on current `main`; do not create a branch/worktree.
- Preserve the pre-existing unstaged change in `plugins/kanban/codex_listener/codex_kanban_interactive.py`.
- Do not push unless the user explicitly asks.
- Do not use Graphify. Use focused local search; Serena/RepoWise may be used when available.
- Do not run raw `pytest`; always use `scripts/run_tests.sh`.
- Do not modify Hermes core files outside the Kanban plugin/local integration surfaces named above.
