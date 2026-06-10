# CodeWhale-tui Watcher Injection: Busy Markers and PTY Input Corruption (2026-06-09)

## Problem 1: `active ctx` False Busy Marker

**Symptom**: Critic and implementer watchers continuously `skip claim: DeepSeek pane still shows an active/pending Kanban prompt` despite the pane being idle.

**Root cause**: CodeWhale-tui v0.8.53 status bar shows `active ctx 13%` (context-usage indicator). The watcher's `_DEEPSEEK_BUSY_MARKERS` contained `" active ctx"`, which substring-matched `active ctx 13%` → `_looks_like_idle_deepseek_pane()` returned False → watcher never attempted to claim.

**Fix**: Remove `" active ctx"` from `_DEEPSEEK_BUSY_MARKERS`. The genuine busy marker is `" active ·"` (with middle dot), not `" active ctx"` (context percentage).

**File**: `~/.hermes/hermes-agent/plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py` line ~346

**Why planner was unaffected**: Planner uses codex/Reasonix, whose status bar doesn't show `active ctx`.

## Problem 2: `Activity: thinking` Not Recognized as Busy

**Symptom**: Watcher injects a new kanban prompt while codewhale-tui is still processing an LLM response (status bar shows `Ctrl+O Activity: thinking`). The injected characters go into PTY buffer and corrupt input state when the composer re-enables.

**Root cause**: codewhale-tui status bar `Activity: thinking` means LLM is streaming output. The pane's second-to-last line still shows idle marker `编写任务或使用 /。`, so `_looks_like_idle_deepseek_pane()` returned True. But the TUI is NOT idle — it's rendering LLM output and the composer is disabled.

**Fix**: Add `"activity: thinking"` to `_DEEPSEEK_BUSY_MARKERS`. Now when codewhale shows `Activity: thinking`, the watcher correctly treats the pane as busy and skips injection.

## Problem 3: PTY Input Degradation After Injection

**Symptom**: After watcher injects a task prompt and the task completes, typing in codewhale-tui becomes unreliable — keys don't register, require multiple attempts. Fresh restart fixes it.

**Contributing factors**:
1. **High-frequency dump-screen polling**: 3 watchers × 6s poll = 0.5Hz zellij PTY reads, potentially interfering with codewhale-tui's ink input event processing.
2. **Injection during thinking state**: If watcher injects while codewhale is in `Activity: thinking`, `write-chars` text goes into PTY buffer. When LLM finishes and composer re-enables, buffered chars flood the input handler.
3. **No idle confirmation before inject**: Single check of pane state could race with codewhale transitioning from busy→idle.

**Fixes applied**:
1. `DAY_POLL_SECONDS` changed from 6.0 to 60.0 in `kanban_listener_policy.py` — reduces PTY poll frequency by 10x.
2. Idle confirmation loop added in `claim_and_inject_one()`: before injecting, wait for 2 consecutive idle checks (1s apart). If either fails, abort injection and reclaim the task.
3. `"activity: thinking"` BUSY marker (Problem 2 fix) prevents injection during LLM response.

**Remaining risk**: The root cause of input degradation is not fully proven. The PTY buffer accumulation theory is plausible but not instrumented. If degradation persists after these fixes, the next diagnostic step is to strace codewhale-tui's read/write on the PTY fd during and after watcher injection to see if write-chars leaves unconsumed bytes.

## Diagnostic Commands

```bash
# Check if watcher is stuck in skip-claim loop
tail -20 ~/.hermes/kanban/boards/egomotion4d/logs/deepseek-interactive-critic.log

# Verify codewhale-tui status bar
zellij --session kanban-egomotion4d action dump-screen --pane-id 3 --full 2>&1 | tail -5

# Check current BUSY markers in source
grep -A20 "_DEEPSEEK_BUSY_MARKERS" ~/.hermes/hermes-agent/plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py

# Restart watcher (kill PID, launcher auto-restarts)
pgrep -af "watch-child.*critic"
kill <PID>
```

## Key Insight

**Python module-level constants are loaded once at process start.** Patching the source file does NOT affect already-running watcher processes. You must restart the watcher (kill it; the launcher auto-restarts with the new code) for fixes to take effect.
