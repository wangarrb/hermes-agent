# Kanban DB Corruption — Debugging Trace (2026-05-22)

## Symptom
All 4 kanban pane workers (hermes coordinator, codex planner, deepseek implementer, hermes critic) exit within seconds of launch. Zellij session stays alive but all panes show empty shells.

## Debugging Path

1. **Zellij log** (`/tmp/zellij-1000/zellij-log/zellij.log`): Shows `Input/output error (os error 5)` on PTY close after child exit — this is a symptom, not the cause.

2. **Kanban worker logs** (`~/.hermes/kanban/boards/<board>/logs/`): The actual crash trace:
```
sqlite3.DatabaseError: file is not a database
  at kanban_db.py:920, in connect() → conn.execute("PRAGMA journal_mode=WAL")
  at codex_kanban_interactive.py:255, in claim_and_inject_one() → with kb.connect(board=board) as conn
```

3. **Verify DB corruption**: `file ~/.hermes/kanban/boards/<board>/kanban.db` returns `data` instead of `SQLite 3.x database`.

4. **Find repair backup**: `ls -lt ~/.hermes/kanban/boards/<board>/kanban.db.repair-test-*` — validate with `file` and `sqlite3 <backup> "SELECT count(*) FROM tasks;"`.

5. **Replace**:
```bash
cd ~/.hermes/kanban/boards/<board>/
cp kanban.db kanban.db.corrupted-$(date +%Y%m%d)
cp kanban.db.repair-test-<timestamp> kanban.db
```

6. **Clean dead zellij session** (secondary issue found during debugging):
```bash
zellij kill-session kanban-<board>   # kills processes
zellij delete-session kanban-<board>  # removes session record
```

## Secondary Issue: Zellij Dead Sessions

`zellij kill-session` only kills processes inside the session. The session itself becomes EXITED (dead) but the name stays occupied. Next `zellij --session <name> --new-session-with-layout` fails with "Session with name X already exists, but is dead".

Fix: Always call `zellij delete-session <name>` after `kill-session`. Both `start-kanban.sh` and `stop-kanban.sh` at `/home/wyr/bin/` have been patched to do kill+delete.

Bulk cleanup of accumulated dead sessions:
```bash
zellij list-sessions --no-formatting --short | while read -r s; do zellij delete-session "$s"; done
```

## Recommended Prevention

Add integrity check in `kanban_db.py:connect()` that:
1. Catches `sqlite3.DatabaseError` on PRAGMA
2. Scans for `kanban.db.repair-test-*` backups sorted by mtime
3. Validates the most recent backup with `PRAGMA integrity_check`
4. Falls back to the valid backup with a warning log
5. Only raises the original error if no valid backup exists
