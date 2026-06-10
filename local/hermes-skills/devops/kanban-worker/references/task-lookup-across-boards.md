# Task Lookup Across Multiple Kanban Board Databases

## Scenario

A worker is dispatched for a task ID (e.g. `t_7fe2bed7`), but the task exists on a named board's DB (e.g. `kanban/boards/egomotion4d/kanban.db`), not in the default `~/.hermes/kanban.db`.

## Symptom

- `kanban_show` fails with "unknown id" (because CLI defaults to the `default` board)
- Direct SQL query on the default DB returns empty rows
- The task ID appears nowhere in the default board

## Root Cause

The kanban dispatcher can route a task from any board to any worker. The worker receives the task ID but not the board slug. The worker must discover which board DB it lives on.

## Solution: Search All Board DBs

```python
import subprocess, os, sys

def find_kanban_db(task_id):
    """Find which kanban database a task lives on."""
    real_home = subprocess.run(
        ["getent", "passwd", os.environ.get("USER", "wyr")],
        capture_output=True, text=True
    ).stdout.strip().split(":")[5] if ":" in ... else os.environ.get("HOME", "/home/wyr")

    import glob
    candidates = glob.glob(f"{real_home}/.hermes/**/kanban.db", recursive=True)
    candidates = [c for c in candidates if "cache" not in c]

    for db_path in sorted(candidates):
        result = subprocess.run(
            ["sqlite3", db_path, f"select id, status, title, assignee from tasks where id='{task_id}';"],
            capture_output=True, text=True, timeout=5
        )
        row = result.stdout.strip()
        if row:
            status = row.split("|")[1] if "|" in row else "unknown"
            slug = db_path.replace(f"{real_home}/.hermes/kanban/boards/", "").split("/")[0] if "boards/" in db_path else "default"
            return {"db_path": db_path, "board_slug": slug, "status": status, "raw": row}

    return None
```

## Real Example

```
Task: t_7fe2bed7 ("T7: Run benchmark on gpuserver and pull back results")
Status: done
Found in: /home/wyr/.hermes/kanban/boards/egomotion4d/kanban.db
Board slug: egomotion4d
```

## $HOME Override Pitfall

When Hermes runs as a profile (e.g. `implementer`), `$HOME` may be set to the profile's home directory (e.g. `/home/wyr/.hermes/profiles/implementer/home`). The real user home `/home/wyr` is only accessible via `getent passwd $(whoami)`.

Affected paths:
- `~/.hermes/kanban.db` resolves relative to overridden `$HOME`, so you won't find the user's actual kanban board
- Use absolute paths: `/home/wyr/.hermes/kanban.db`
- Use `getent` to discover the real home at runtime