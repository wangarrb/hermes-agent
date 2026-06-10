# Fork Customizations Manifest

This file tracks all customizations added on top of official hermes-agent releases.
Purpose: prevent lost customizations during future upgrades (rsync --delete wipes fork-only files).

Last updated: 2026-06-06 (v0.16.0 upgrade)

---

## New Files (fork-only, deleted by rsync --delete)

These files do NOT exist in official releases. After any rsync upgrade, they must be
restored from git history or this manifest's embedded content.

### 1. hermes_cli/kanban_listener.py (989 lines)

**Feature**: `/listen-kanban` command — worker-loop that runs inside a Hermes CLI pane.
Continuously finds ready tasks, claims them, foreground-executes via natural language.

**Restore**:
```bash
git show 82cb4c264:hermes_cli/kanban_listener.py > hermes_cli/kanban_listener.py
```

### 2. hermes_cli/kanban_listener_policy.py (185 lines)

**Feature**: Shared timing/retry policy constants for all kanban listeners.
Active/quiet hours poll intervals, health check cadences, idle reclaim thresholds.

**Restore**:
```bash
git show 82cb4c264:hermes_cli/kanban_listener_policy.py > hermes_cli/kanban_listener_policy.py
```

### 3. hermes_cli/kanban_worker_runtime.py (506 lines)

**Feature**: Worker runtime state tracking — claim locks, heartbeats, progress monitoring.

**Restore**:
```bash
git show 82cb4c264:hermes_cli/kanban_worker_runtime.py > hermes_cli/kanban_worker_runtime.py
```

---

## Modified Files (patches on top of official)

These are official files with fork-specific code injected. After rsync, re-apply patches.
Line numbers are approximate — search by anchor comments/strings instead.

### 4. plugins/memory/holographic/__init__.py — Chinese auto_extract patterns

**Feature**: 26 Chinese patterns (3 groups) + matching logic in `_auto_extract()`.
Official only has English patterns; Chinese memories would never auto-extract without this.

**Patch points**:
- After the English pattern definitions (~line 380): Add `CN_PATTERNS_1`, `CN_PATTERNS_2`, `CN_PATTERNS_3`
- In `_auto_extract()`: Add CN pattern matching logic (try CN patterns if EN patterns don't match)

**Verify**: `grep -c 'CN_PATTERNS' plugins/memory/holographic/__init__.py` should be ≥ 6

### 5. gateway/run.py — `/mycompress` command

**Feature**: `/mycompress` = load skill body + call `/compress <skill_body>`.
Allows compressing context with a skill as focus topic.

**Patch points** (search by function name, not line number):
- `_AGENT_PENDING_SENTINEL` definition area: Add 3 module-level helpers
  - `_load_gateway_mycompress_focus_topic()`
  - `_parse_mycompress_args()`
  - `_find_mycompress_tail_start()`
- In command dispatch (search `canonical == "compress"`): Add `elif canonical == "mycompress": return await self._handle_mycompress_command(event)`
- After `_handle_compress_command`: Add `_handle_mycompress_command()` method

**Verify**: `grep -c 'mycompress' gateway/run.py` should be ≥ 17

### 6. cli.py — `/mycompress` command (CLI mode)

**Feature**: Same as gateway but for CLI mode.

**Patch points**:
- In command dispatch (search `canonical == "compress"`): Add `elif canonical == "mycompress": self._handle_mycompress_command(cmd_original)`
- After `_manual_compress()`: Add `_handle_mycompress_command()` method

**Verify**: `grep -c 'mycompress' cli.py` should be ≥ 7

### 7. tools/memory_tool.py — Failed write backup

**Feature**: When a memory/user write is rejected (injection blocked, char limit exceeded),
the content is backed up to `memories/failed-backups/` with a timestamped
filename and an INDEX.md for review. Prevents data loss from rejected writes.

**Patch points**:
- Add `import re` and `from datetime import datetime` to file header
- In `MemoryTool` class after `save_to_disk()`: Add `_backup_failed_write()` method (~50 lines)
- In `MemoryTool` class: Add `_append_backup_index()` static method (~30 lines)
- In `add()`: Call `_backup_failed_write()` on injection rejection and char_limit rejection
- In `replace()`: Call `_backup_failed_write()` on injection rejection and char_limit rejection

**Verify**: `grep -c 'failed.backup\|failed_backup' tools/memory_tool.py` should be ≥ 3

---

## Features Covered by Official (no fork patch needed)

These fork features are now in the official release. Check on each upgrade if still covered.

| Feature | Since official version | Notes |
|---------|----------------------|-------|
| DeepSeek v4 + reasoning | v0.16.0 | MiMo support, host-driven detection |
| credential_pool key_env | v0.16.0 | In agent_init.py + chat_completion_helpers.py |
| hindsight conditional auto-recall | v0.16.0 | auto_recall_mode, recall_trigger_keywords, recall_cache |
| kanban_db expanded schema | v0.16.0 | runs, heartbeats, context tracking |
| Provider threat patterns refactored | v0.16.0 | Now in tools/threat_patterns.py |

---

## Config-level (no code changes needed)

- **Bailian alias**: Handled in `config.yaml` provider definitions
- **DeepSeek default model**: Set via `hermes config set`

---

## Upgrade Procedure

```bash
# 1. Backup current state
git add -A && git commit -m "pre-upgrade snapshot"

# 2. Download & extract official tarball
aria2c/curl the tarball, extract to /tmp/hermes-agent-<version>/

# 3. rsync (preserves .git/, venv/, config/)
rsync -az \
  --exclude .git/ --exclude venv/ --exclude config/ \
  --exclude node_modules/ --exclude .env \
  /tmp/hermes-agent-<version>/ \
  ~/.hermes/hermes-agent/

# 4. Restore new files (deleted by rsync --delete)
for f in kanban_listener.py kanban_listener_policy.py kanban_worker_runtime.py; do
  git show <last-fork-commit>:hermes_cli/$f > hermes_cli/$f
done

# 5. Re-apply patches to modified files (see sections 4-7 above)

# 6. Verify
python3 -c "compile(open('gateway/run.py').read(), 'gateway/run.py', 'exec')"
python3 -c "compile(open('cli.py').read(), 'cli.py', 'exec')"
python3 -c "compile(open('plugins/memory/holographic/__init__.py').read(), '_', 'exec')"
python3 -c "compile(open('tools/memory_tool.py').read(), '_', 'exec')"
PYTHONPATH=. python3 -c "from hermes_cli import kanban_listener_policy; print('OK')"
PYTHONPATH=. python3 -c "from hermes_cli import kanban_worker_runtime; print('OK')"
PYTHONPATH=. python3 -c "from hermes_cli import kanban_listener; print('OK')"

# 7. Restart gateway
hermes gateway restart

# 8. Functional tests
# - CCH chat
# - /mycompress
# - /compress
# - holographic CN auto_extract
# - kanban listener
```

---

## Git anchor commit

The last fork commit before v0.16.0 rsync: `82cb4c264`
Use this to restore any fork-only files: `git show 82cb4c264:<path>`
