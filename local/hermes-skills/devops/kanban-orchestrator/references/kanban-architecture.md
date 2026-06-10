# Kanban Architecture FAQ

Common conceptual questions about how Hermes Kanban works under the hood.

## Profile vs Session Identity

**Q: Does each kanban assignee need a separate Hermes profile?**

Not necessarily. The assignee label maps to a profile name, but the same profile can run multiple concurrent workers:

```
Task A (assignee=default)  →  hermes -p default  (session 1)
Task B (assignee=default)  →  hermes -p default  (session 2)
```

Each worker is an independent Hermes process with its own session. They share the profile's config (model, API keys, skills) but don't interfere — unless they write to the same workspace.

**When to create a new profile:**
- Different model needed for that role
- Different API credentials (e.g., researcher uses a cheap model, reviewer uses a capable one)
- Different skill set
- Different memory namespace

**When a single profile is enough:**
- Simple pipeline with sequential tasks (same model/skills for all steps)
- Lean setup — 2-3 profiles can cover a full orchestrator→worker pipeline

## Worker Visibility

**Q: Can I see the worker's CLI window?**

No. The dispatcher spawns workers via `subprocess.Popen()`:

```python
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,    # no input
    stdout=worker_log_file,      # output to log file
    stderr=subprocess.STDOUT,    # errors to same log
    start_new_session=True,      # detached
)
```

Workers are **headless background processes** — no terminal, no PTY, no CLI UI.

**What you CAN see:**

| Surface | Command |
|---------|---------|
| Worker output log | `hermes kanban log <task_id>` |
| Live event stream | `hermes kanban tail <task_id>` |
| Task state + history | `hermes kanban show <task_id>` |
| Attempt history | `hermes kanban runs <task_id>` |
| Visual board | dashboard at `http://127.0.0.1:9119/` → Kanban tab |
| OS processes | `ps aux | grep hermes` |

## Manual Claiming (Control-Plane Lanes)

**Q: Can I run a task interactively instead of letting the dispatcher spawn a headless worker?**

Yes. Create a task with an assignee name that does NOT match a real Hermes profile:

```bash
# Create a task for manual handling
hermes kanban create "Research X" --assignee manual-worker
```

The dispatcher sees `manual-worker` is not a real profile → **skips it** (logged as `skipped_nonspawnable`). The task stays `ready` until a human claims it:

```bash
# In a terminal window
hermes kanban claim <task_id>

# Now you have full Hermes CLI interaction for this task
```

**Use cases for manual claiming:**
- Human-in-the-loop research/analysis that needs your judgment
- Debugging — run a task interactively to see what happens
- Tasks that need domain knowledge the AI can't provide
- Training/onboarding — see how a worker would experience the task

**Dashboard indicators:** The kanban dashboard shows these tasks as "waiting for manual pickup" — no auto-spawn attempt.

## How Roles Communicate

**Q: How do different kanban workers pass information to each other?**

**No real-time IPC/RPC exists.** Communication is purely asynchronous through the board's persistent state:

### 1. Parent-Child Handoff (Primary)

The `parents=[...]` mechanism gates task execution and passes structured data:

```
researcher task ──kanban_complete(summary=..., metadata={...})──▶  DB
                                                                      │
                                                             parent_of
                                                                      │
analyst task ◀──────── build_worker_context() reads parent results ───┘
```

When a child task is spawned, `build_worker_context()` (in `kanban_db.py`) assembles:
- Task title, body, assignee, workspace
- Prior attempts on this task (capped at 10, with summaries/errors/metadata)
- **Structured handoff from every done parent** — `summary` + `metadata` from each parent's last completed run
- Cross-task role history (last 5 completed runs by this assignee)
- Comment thread (last 30 comments)

### 2. Comments (Async)

```bash
# Any worker or human can leave a comment
hermes kanban comment <task_id> "Found a dependency issue..."

# Comments persist in the event log and appear in build_worker_context()
```

### 3. Block/Unblock (Human Gate)

```python
# Worker signals it needs input
kanban_block(reason="Which scaling method: linear or exponential backoff?")
```

The task moves to `blocked`. After the human decides and unblocks, the dispatcher respawns the worker with the full comment thread in context.

### 4. Orchestrator as Coordinator

An orchestrator profile (no `HERMES_KANBAN_TASK` env var) can freely read/write any task on the board — it creates subtasks, reads intermediate results, and manages the pipeline. The orchestrator is the "project manager" that sees everything.

## Headless vs Interactive: Choosing the Mode

| Aspect | Auto-Spawn (Headless) | Manual Claim (Interactive) |
|--------|----------------------|---------------------------|
| Who runs it | Dispatcher via `subprocess.Popen()` | You in a terminal |
| Visibility | Logs + dashboard only | Full CLI interaction |
| Best for | Routine/automated tasks | Exploratory/debugging tasks |
| Profile needed | Assignee must match a real profile | Assignee can be any label |
| Concurrency | Multiple workers from same profile | One task per terminal |
