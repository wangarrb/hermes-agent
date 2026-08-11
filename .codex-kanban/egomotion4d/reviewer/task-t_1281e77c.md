─── Kanban 任务 t_1281e77c ───
        标题：IMPLEMENT-WATCHER-HOT-RELOAD
        角色：reviewer 

        关键规则：
        1. 完成后调用 `hermes kanban --board egomotion4d complete t_1281e77c --run-id 2058 --generation 2 --summary "..."`
        2. 阻塞时调用 `hermes kanban --board egomotion4d block t_1281e77c "..." --run-id 2058 --generation 2`
        3. 默认中文；路径/命令保留英文

        执行边界：task_id=t_1281e77c run_id=2058 generation=2
        HERMES_KANBAN_TASK=t_1281e77c HERMES_KANBAN_RUN_ID=2058 HERMES_KANBAN_GENERATION=2

        

        上下文：
        # Kanban task t_1281e77c: IMPLEMENT-WATCHER-HOT-RELOAD

Assignee: reviewer
Status:   running
Workspace: dir @ /home/wyr/.hermes/hermes-agent-repo
Max runtime: 14400s
Terminal timeout: 14370s
Workspace contract: {"artifact_namespace": null, "auto_generated": true, "base_commit": "", "branch": "", "common_dir": "", "generation": 2, "mismatches": [], "repository": "", "target_branch": "", "task_id": "t_1281e77c", "valid": true, "version": "workspace_contract.v1", "workspace_kind": "dir", "worktree": "/home/wyr/.hermes/hermes-agent-repo", "write_set": null}

## Body
OWNER=reviewer
WORKSPACE=/home/wyr/.hermes/hermes-agent-repo
BRANCH=main
BASE_SHA=f4759918dc33dc3f11f4acec8a8e7fbc7e752860
COMMIT_OWNERSHIP=implementer may create local focused commits for declared WRITE_SET; never push.

Execute docs/plans/2026-08-04-kanban-watcher-hot-reload-implementation.md through Tasks 1-9 plus the isolated fake-board portion of Task 10. Reviewer retains the real kanban-egomotion4d live activation and final operational verdict. Use TDD and scripts/run_tests.sh only. Read repository AGENTS.md first. Do not use Graphify, create a branch/worktree, modify Hermes core, or restart current watchers/agent panes.

WRITE_SET:
local/lib/codex_role_home.sh
local/bin/start-kanban.sh
local/bin/stop-kanban.sh only if scoped cleanup requires it
plugins/kanban/watcher_runtime.py
plugins/kanban/base_listener.py
local/bin/hermes-kanban-reload-watchers
local/bin/kanban-watcher-supervisor.py
tests/test_start_kanban_codex_agents.sh
tests/test_kanban_watcher_runtime.py
tests/fixtures/fake_reload_watcher.py
tests/test_kanban_reload_watchers.py
tests/test_kanban_watcher_supervisor.py
other existing listener tests only when necessary for real behavioral compatibility
docs/design/2026-08-04-kanban-watcher-hot-reload.md only for measured corrections.

Preserve and do not stage the pre-existing user modification in plugins/kanban/codex_listener/codex_kanban_interactive.py.

INTERFACE_GUARDS:
1. Same board/profile/session/pane admits exactly one functional claim loop; tests/test_kanban_watcher_runtime.py.
2. SIGUSR1 request/self-exec keeps PID and verified running task/run/generation/claim_lock, advances heartbeat, and does not claim/inject twice; fake subprocess integration test.
3. Supervisor/reload CLI affect only exact board/session and recheck replacement immediately before restart; supervisor and reload tests.

DELIVERY: focused commit SHA list, git diff base..delivery, exact test outputs, isolated reload smoke. Do not merely summarize the plan; implement and verify it completely before completing this card.

## Current authoritative handback
RETURN FOR REWORK: 真实 self-exec 路径先 acquire 后 adopt 会自锁退出；fake tests 未 exec，idle/FAILED ACK、PID-start/FD identity 与 source-grep test 均不满足合同。读取 reviewer durable comment A-F，原卡 gen2 修复并重新交付。

## Prior attempts on this task
_(1 prior attempt from an older generation omitted; inspect durable history on demand)_
### Attempt 1 — reclaimed (implementer, 2026-08-04 04:06, 48m ago)
_error_: manual_reclaim: implementer gen2 delivery complete (A-F + independent review PASS); reviewer retains real kanban-egomotion4d live activation and final operational verdict per task contract
_metadata_: `{"host_local": true, "prev_pid": 1439151, "sigkill": true, "terminated": false, "termination_attempted": true}`

### Attempt 2 — reclaimed (reviewer, 2026-08-04 04:35, 19m ago)
_error_: manual_reclaim: one-time reviewer legacy watcher activation; durable progress saved, same generation resumes at safe boundary
_metadata_: `{"host_local": true, "prev_pid": 1458957, "sigkill": true, "terminated": false, "termination_attempted": true}`

## Comment thread
comment from worker `hermes-interactive-listener` at 2026-08-04 04:06, 48m ago:
Injected into Zellij pane 1; prompt file: /home/wyr/.hermes/hermes-agent-repo/.hermes-kanban/egomotion4d/implementer/task-t_1281e77c.md

comment from worker `implementer` at 2026-08-04 04:34, 20m ago:
IMPLEMENTER GEN2 DELIVERY COMPLETE — awaiting reviewer live activation + final verdict.

独立审查 (deleg_a9277d65 leaf subagent) verdict=PASS：A-F 逐项核实，13 个关键测试（真实 exec 集成 + adopt fail-closed + 隔离 smoke）全过；commit 分布干净（fix 只改 3 个生产文件）。

证据：
- COMMITS: eac30bff4 (fix: adopt-first fail-closed + strict ACK contract) + 3e00dad47 (test: real os.execve self-exec + strict ACK/fail-closed)
- fresh focused suite: 66 passed / 0 failed (watcher_runtime + reload_watchers + watcher_supervisor + hermes_role_context_listener)
- shell suites: test_start_kanban_codex_agents.sh + test_start_kanban_reviewer_mode.sh PASS; bash -n 3 脚本 clean
- Task 10 isolated fake-board smoke: 真实 CLI REQUESTED→ACK 退出码 0; watcher 同 PID / EXEC_GENERATION 1→2 / LOCK_ADOPTED / CLAIM_INJECT 恰 1 次 / claim 文件不变 / ACK_WRITTEN 恰一次
- git diff --check clean; plugins/kanban/codex_listener/codex_kanban_interactive.py 用户改动保持未暂存
- 未真实 reload kanban-egomotion4d（按合同留给 reviewer）

下一步（reviewer 职责）: 完成真实 kanban-egomotion4d live activation 和最终 operational verdict；若失败则停止 rolling 保留旧 watcher。

comment from worker `implementer` at 2026-08-04 04:35, 19m ago:
Delivery SHA 更新：test commit 已 amend 为 3e00dad47（清理 fake 未使用的 --handoff-dir 参数，无行为变化，watcher_runtime 39 测试重跑全过）。最终 delivery = eac30bff4 + 3e00dad47（base f4759918d..HEAD）。独立审查报告全文：/home/wyr/.hermes/profiles/implementer/cache/delegation/subagent-summary-0-20260804_043238_902832.txt（verdict PASS，A-F 逐项证据）。请 reviewer 完成真实 kanban-egomotion4d live activation 与最终 operational verdict。

comment from worker `codex-interactive-listener` at 2026-08-04 04:35, 19m ago:
Injected into Zellij pane 2; prompt file: /home/wyr/.hermes/hermes-agent-repo/.codex-kanban/egomotion4d/reviewer/task-t_1281e77c.md

comment from worker `default` at 2026-08-04 04:54, 1m ago:
REVIEWER ACTIVATION PROGRESS：gen2 delivery 经 focused diff 后直接补三组生产边界，commits c16f1959c、b26ee270d。fresh 验证：73 passed/0 failed；Codex role-home 与 reviewer-mode 两个 shell suite PASS；bash -n 与 git diff --check clean。真实 rolling：coordinator PID 1737560 ACK；designer PID 1737562 ACK；implementer 首次从 pre-c16 版本升级时正确 FAILED missing_handoff，scoped supervisor 冷启动 PID 1745464 后 ACK；planner PID 1737973 ACK。均未重启 agent/TUI；planner 远端计算仍继续。当前只剩 reviewer legacy watcher 的一次性接管：reclaim run2057 后新 watcher应在安全边界领取同一 gen2；随后对 reviewer 自身执行真实热重载，核对同 PID、run/generation/claim_lock、heartbeat 与单实例，再给最终 verdict。


## Effective Role Context

pane_profile=reviewer effective_role=reviewer assist=false backend=codex

fixed_control_commit=

role_context_manifest=/home/wyr/.hermes/hermes-agent-repo/.codex-kanban/egomotion4d/reviewer/task-t_1281e77c/role-context.json

HERMES_KANBAN_CONTROL_PROMPT_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

### Project Role Prompt


### Workspace Contract
{"artifact_namespace": null, "auto_generated": true, "base_commit": "", "branch": "", "common_dir": "", "generation": 2, "mismatches": [], "repository": "", "target_branch": "", "task_id": "t_1281e77c", "valid": true, "version": "workspace_contract.v1", "workspace_kind": "dir", "worktree": "/home/wyr/.hermes/hermes-agent-repo", "write_set": null}

        ─── 开始执行任务 t_1281e77c ───