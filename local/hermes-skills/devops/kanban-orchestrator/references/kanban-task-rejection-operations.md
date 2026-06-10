# Kanban Task Rejection / Return Operations

User asked (2026-05-21): "kanban的任务有问题，应该怎么处理？类似于拒绝/退回"

Hermes Kanban supports several ways to reject, return, or redirect a task without deleting it. All preserve the audit trail.

| Operation | Command | Effect | Use case |
|-----------|---------|--------|----------|
| **Block** | `hermes kanban --board <board> block <task_id> '<reason>'` | Status → `blocked`, not auto-picked by listeners | Task has fundamental issue; needs human intervention before retry |
| **Reclaim + reassign** | `hermes kanban reclaim <task_id> --reason '<why>'` then `hermes kanban assign <task_id> <new_profile>` | Releases claim, resets to `ready` under a different assignee | Wrong profile got the task; return to planner or reassign to coordinator |
| **Reassign only** | `hermes kanban reassign <task_id> <new_profile> --reclaim` | One-shot: reclaim + assign | Same as above, atomic |
| **Complete with rejection reason** | `hermes kanban --board <board> complete <task_id> --summary 'REJECT. <reason>' --result '<decision>'` | Status → `done` with explicit rejection in result field | Task is valid but decision is "no go"; useful for review/audit tasks that return NO_CLAIM |
| **Block + create follow-up** | `block` the bad task, `create` a new task with `parents=[bad_task_id]` | Bad task stays blocked as root cause; new task has dependency | "Needs revision" — block review, create implementer fix task linked from review |

## Coordinator-audit created tasks

Rule-based audit auto-creates `[Review]` and `Coordinator follow-up` tasks. When these accumulate as `ready` with no one to pick them up (common when the coordinator or critic pane is not running):

1. Check the state: `hermes kanban --board <board> list --assignee coordinator --status ready`
2. Reclaim stale running tasks first: `hermes kanban --board <board> reclaim <stuck_task_id>`
3. If the coordinator-audit chain is complete and no action remains, complete it: `kanban_complete(summary="NO_OP: already processed", result="No action needed")`
4. If it needs a real decision, manually complete with proper rejection/blocking

## Quick status checks

```bash
# See all tasks by assignee and status
hermes kanban --board egomotion4d list --json

# Show detailed task state including claim expiry
hermes kanban --board egomotion4d show <task_id>

# Claim expiry detection (stale running tasks)
python3 -c "import sqlite3; r = sqlite3.connect('/home/wyr/.hermes/kanban/boards/egomotion4d/kanban.db').execute('SELECT id, claim_lock, claim_expires, status FROM tasks WHERE status=\"running\" AND claim_expires < strftime(\"%s\",\"now\")').fetchall(); print(r)"
```