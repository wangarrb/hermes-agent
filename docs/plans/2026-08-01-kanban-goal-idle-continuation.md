# Kanban Goal Idle Completion Check Implementation Plan

> **For agentic workers:** Execute locally with TDD; no subagent is required for this focused lifecycle fix.

**Goal:** Prevent large `/goal` tasks from silently stopping or completing on an intermediate result, without assuming every normal idle state means “continue”.

**Architecture:** Keep the existing watcher and Kanban cards. At stable normal idle, ask “任务都完成了吗？” once per idle episode; only the existing API/model failure path injects “继续”. Suppress the question while an explicitly registered reviewer checkpoint is open. Reuse one Kanban-specific completion rubric in CLI/tool goal judges. No polling card, continuation card, watchdog, or new model tool.

**Tech Stack:** Python, SQLite Kanban DB, Zellij interactive listeners, pytest.

---

### Task 1: RED tests for idle completion checks

**Files:**
- Create: `tests/plugins/test_kanban_idle_continuation.py`
- Modify: `plugins/kanban/base_listener.py`
- Modify: `plugins/kanban/hermes_listener/hermes_kanban_interactive.py`

- [x] Add tests proving an idle `goal_mode` task receives exactly one neutral completion check per idle episode and busy activity resets the episode.
- [x] Add tests proving an idle reviewer receives one lifecycle nudge, while an open `REVIEWER_CHECKPOINT_PENDING` suppresses the owner completion check.
- [x] Run the new tests RED, implement the shared base/Hermes helper, then verify GREEN.

### Task 2: RED tests for strict Kanban completion semantics

**Files:**
- Modify: `hermes_cli/goals.py`
- Modify: `hermes_cli/kanban.py`
- Modify: `tools/kanban_tools.py`
- Modify: `tests/hermes_cli/test_kanban_goal_mode.py`

- [x] Add a test capturing the goal sent by CLI completion and require the Kanban rubric: north-star evidence, intermediate NO_CLAIM is not terminal unless the body says so, reviewer/user terminal authorization for true abandonment.
- [x] Run the focused test RED, add shared `kanban_goal_text()` to CLI/tool/worker goal paths, then verify GREEN.

### Task 3: Regression verification

**Files:**
- Test only.

- [x] Run listener fencing, plugin idle, goal-mode CLI/tool, and Hermes listener tests: `241 passed`.
- [x] Run `git diff --check` and inspect the focused diff.
- [x] Record that `/goal` stays on the same card, normal idle asks for completion assessment, API failure alone injects “继续”, and ordinary non-goal/non-review tasks are unchanged.

### 更新记录

| 日期 | 关键数据/结果 | 结论 | 关键转折及原因 |
|------|--------------|------|---------------|
| 2026-08-01 | focused regression: 241 passed | PASS | 普通空闲由命令“继续”改为中性的“任务都完成了吗？”，避免预设任务未完成；等待 reviewer 时以 durable checkpoint 抑制追问。 |
| 2026-08-01 | supervisor/listener regression: 13 passed；五个 pane watcher 单实例稳定 | PASS | 重启时发现 supervisor 把含 listener 文本的 `bash -c` 误识别为 watcher；改为匹配真实 argv basename，消除整条运维命令被重放和重复 watcher。 |
| 2026-08-01 22:55 | 真实 Hermes TUI idle/busy fixtures；focused regression `58 passed` | PASS | 新 TUI 空闲栏 `⚕ ❯ msg=interrupt · ...` 被旧逻辑同时当成“非裸 prompt”和 busy，导致 goal completion check 不可达。仅放行精确静态栏，移除历史输出型 busy marker，增加 `ruminating`/process wait 等真实活动 marker；completion prompt 明确未达北极星时不得向用户升级普通技术选项。 |
