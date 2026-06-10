# Session-Liveness Gate & Task Rework via `--reopen` (2026-06-08)

## Session-Liveness Gate for Idle Detection

### Bug 1: Wrong directory and suffix

`_sessions_latest_mtime()` only scanned `~/.reasonix/sessions/*.json` (Codex legacy, stale).
Active Reasonix sessions are at `~/.config/reasonix/sessions/*.jsonl`.

Fix: Now scans both directories, both suffixes:
```python
_REASONIX_SESSIONS_DIRS = [
    Path.home() / ".config" / "reasonix" / "sessions",  # Reasonix v1.x
    Path.home() / ".reasonix" / "sessions",              # Codex / legacy
]
_REASONIX_SESSION_SUFFIXES = (".jsonl", ".json")
```

### Bug 2: Idle detection ignored liveness data

Even when liveness found session updates, idle detection only checked screen content.
Screen unchanged for 600s during API wait → false reclaim.

Fix: Idle detection now checks `latest_mtime` before starting idle timer:
```
Screen idle? → session file updated recently (< idle_pane_reclaim_s)?
  Yes → agent working, reset idle timer
  No  → genuinely idle, start/continue timer
```

### Pitfall for future

When adding liveness checks that depend on file mtime, always verify the actual file paths
and suffixes match what the running tool produces. A liveness check scanning the wrong
directory is worse than no liveness check — it creates false confidence while being blind.

## Task Rework via `update --reopen`

When critic/planner review fails and the task needs to go back to the executor:

1. **Done tasks**: `comment <id> "review feedback"` + `update <id> --reopen [--body "..."]`.
   - `--reopen` transitions `done → ready`, clears claim lock, closes dangling runs, logs `reopened` event.
   - Can also change assignee: `update <id> --reopen --assignee <new_profile>`.
   - Does NOT work on `archived` tasks — use `promote` first.

2. **Running/started tasks**: `reclaim` first, then `comment` + `update`.

3. **Blocked tasks**: `comment` + `update` + `unblock`.

4. **Todo/ready tasks**: Direct `update` to modify body/title.

### Comment must include:
- Problem: which check failed, evidence
- Rework requirement: what to change, how
- Acceptance criteria: how to judge pass

Example:
```
【REVIEW 不通过】输出合同检查失败。
- 问题：metrics.json 缺少 ATE_metric 字段
- 证据：server_results/xxx/run/metrics.json 第 12 行
- 返工要求：重新运行 eval，确保输出 ATE_metric、ATE_rel、RPE_trans、RPE_rot
- 验收标准：metrics.json 包含四字段，ATE_metric < 0.5m
```
