# Kanban Task Rework Flow (打回返工)

When a critic/planner review fails and the task needs to go back to the original implementer for fixes, use the **rework flow** — not simple rejection or new task creation. Rework preserves context, avoids duplicate tasks, and keeps the audit trail intact.

This is distinct from:
- **Block**: task has a fundamental issue needing human intervention
- **Complete with REJECT**: task is valid but the decision is "no go"
- **Block + create follow-up**: bad task stays blocked, new task created

Rework = "the task is sound but the output needs fixes; send it back with specific instructions."

## Decision table by task status

| Task status | Rework steps |
|-------------|-------------|
| `done` (implementer completed, critic gate failed) | `comment` → `update --reopen` (optionally with `--body`/`--assignee`) |
| `started`/`running` (execution blocked, needs rework) | `reclaim` → `comment` → `update` (optional) |
| `blocked` (waiting on external input, now needs rework) | `comment` → `update` (optional) → `unblock` |
| `todo`/`ready` (requirements need modification before execution) | `update` only (no reclaim needed) |

## Standard rework flow (done status — most common scenario)

The `--reopen` flag on `update` handles the entire done→ready transition in one command. It:
1. Transitions task status from `done` to `ready` (can be re-claimed by watcher)
2. Clears claim_lock / claim_expires / worker_pid / completed_at
3. Closes any dangling run records (marks as `reopened`)
4. Records a `reopened` event in the audit trail

```bash
# Step 1: Add structured review comment (REQUIRED — before or after reopen)
hermes kanban --board <board> comment <task_id> \
  "【REVIEW 不通过】<具体问题>。需要修改：<具体要求>。验收标准：<如何判断通过>"

# Step 2: Reopen and optionally update body (one command)
hermes kanban --board <board> update <task_id> --reopen \
  --body "$(cat <<'EOF'
<原 body 内容>

---
## 返工要求（YYYY-MM-DD）
- <新增要求 1>
- <新增要求 2>

## 验收标准（更新）
- <验收条件 1>
- <验收条件 2>
EOF
)"

# If you only need to reopen without changing body (comment-driven rework)
hermes kanban --board <board> update <task_id> --reopen
```

After reopen, the task is `ready` and the original assignee's watcher will automatically claim it.

### Why --reopen instead of reclaim + unblock?

The old flow required 4 steps: `reclaim` → `comment` → `update` → `unblock`. The `--reopen` flag combines the reclaim + status-transition + dangling-run-cleanup into a single atomic operation. Benefits:
- Fewer commands = fewer failure points
- Dangling runs from interactive listener completions are automatically closed
- Audit trail captures `reopened` event with `closed_run_id` for traceability
- No risk of `reclaim` or `unblock` silently failing on done-status tasks

### What if --reopen is not available?

If running an older Hermes version without `--reopen`, fall back to the legacy flow:

```bash
# Legacy 4-step flow (for Hermes versions before --reopen)
hermes kanban --board <board> reclaim <task_id> --reason "review 不通过，需要返工"
hermes kanban --board <board> comment <task_id> "【REVIEW 不通过】..."
hermes kanban --board <board> update <task_id> --body "..."
hermes kanban --board <board> unblock <task_id> --reason "已添加返工要求，重新开放 claim"
```

## Comment format (mandatory)

A rework comment MUST contain all three elements. "不通过" alone is not acceptable.

1. **Problem identification**: which checkpoint failed, what's the evidence
2. **Rework requirements**: what to change, how to change it
3. **Acceptance criteria**: how to judge the rework passes

Example:
```
【REVIEW 不通过】输出合同检查失败。
- 问题：metrics.json 缺少 ATE_metric 字段，只有 ATE_rel
- 证据：server_results/xxx/run/metrics.json 第 12 行
- 返工要求：重新运行 eval 脚本，确保输出包含 ATE_metric、ATE_rel、RPE_trans、RPE_rot 四个字段
- 验收标准：metrics.json 包含完整四字段，且 ATE_metric < 0.5m
```

## Rework to a different worker

If the original worker can't continue, change assignee during the reopen:

```bash
hermes kanban --board <board> update <task_id> --reopen \
  --assignee <new_profile> \
  --body "$(cat <<'EOF'
<原 body 内容>

---
## 返工说明（YYYY-MM-DD）
原执行者 <old_profile> 因 <原因> 无法继续，转由 <new_profile> 接手。
接手者请先阅读上方 comment 了解返工原因。
EOF
)"
```

## Iteration limits

- Max **3 rework cycles** per task
- On the 3rd rework, comment must explicitly state: "这是第 3 次打回，如仍不通过将触发 planner 介入"
- Beyond 3 reworks: `block` the task and notify planner to reassess feasibility

## Archived tasks

`--reopen` does NOT work on archived tasks. To reopen an archived task, first `promote` it back to todo/ready, then use normal operations.

## Relationship to UPDATE-over-recreate rule

The project's hard rule (AGENTS.md §8.4) applies to rework too:
- **Must use `update --reopen` to modify a done task** — never create a new task to replace it
- Only exception: task already claimed/started AND needs a complete plan replacement → block first, then create replacement with `supersedes <old_task_id>` comment

## Why not just create a new task?

Creating a new task for rework:
- Loses the comment thread and review history
- Creates duplicate titles on the board (triggers duplicate detection)
- Breaks parent-child dependency chains
- Violates the UPDATE-over-recreate hard rule

Using `update --reopen` + comment:
- Preserves full audit trail (including reopened event with closed_run_id)
- Original worker sees the review feedback in context
- No board pollution
- Dependency chain stays intact
- Dangling runs automatically cleaned up
