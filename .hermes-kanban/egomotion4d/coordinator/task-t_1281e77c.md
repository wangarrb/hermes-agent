─── Kanban 任务 t_1281e77c ───
        标题：IMPLEMENT-WATCHER-HOT-RELOAD
        角色：implementer （注意：当前 profile=coordinator，但本任务 assignee=implementer，按 implementer 职责执行）

        关键规则：
        1. 完成后调用 `hermes kanban --board egomotion4d complete t_1281e77c --run-id 2054 --generation 1 --summary "..."`
        2. 阻塞时调用 `hermes kanban --board egomotion4d block t_1281e77c "..." --run-id 2054 --generation 1`
        3. 默认中文；路径/命令保留英文

        执行边界：task_id=t_1281e77c run_id=2054 generation=1
        HERMES_KANBAN_TASK=t_1281e77c HERMES_KANBAN_RUN_ID=2054 HERMES_KANBAN_GENERATION=1

        当前任务角色说明：
职责由 Kanban profile/assignee 决定，而不是由底层 agent 类型决定；即使用不同 agent 运行，也要按当前角色工作。
你是 implementer：主要协助 reviewer 完成合同已冻结的确定性 diff/测试/artifact 盘点、复现或小修；也可接受 owner 的例外委派，但它不是 owner 耗时工作的默认路径。正式成功率、算法方向、路线重置、审核结论和最终 handback 只由 reviewer 决定。可写任务必须显式给出绝对 workspace、branch、base SHA、write set 和 commit ownership；缺失时只做只读证据工作。

        上下文：
        # Kanban task t_1281e77c: IMPLEMENT-WATCHER-HOT-RELOAD

Assignee: implementer
Status:   running
Workspace: dir @ /home/wyr/.hermes/hermes-agent-repo
Max runtime: 14400s
Terminal timeout: 14370s
Workspace contract: {"artifact_namespace": null, "auto_generated": true, "base_commit": "", "branch": "", "common_dir": "", "generation": 1, "mismatches": [], "repository": "", "target_branch": "", "task_id": "t_1281e77c", "valid": true, "version": "workspace_contract.v1", "workspace_kind": "dir", "worktree": "/home/wyr/.hermes/hermes-agent-repo", "write_set": null}

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


## Effective Role Context

pane_profile=coordinator effective_role=implementer assist=true backend=hermes

fixed_control_commit=

role_context_manifest=/home/wyr/.hermes/hermes-agent-repo/.hermes-kanban/egomotion4d/coordinator/task-t_1281e77c/role-context.json

HERMES_KANBAN_CONTROL_PROMPT_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

### Project Role Prompt


### Role Description
Egomotion4D implementer role for task-scoped assist execution.

### Workspace Contract
{"artifact_namespace": null, "auto_generated": true, "base_commit": "", "branch": "", "common_dir": "", "generation": 1, "mismatches": [], "repository": "", "target_branch": "", "task_id": "t_1281e77c", "valid": true, "version": "workspace_contract.v1", "workspace_kind": "dir", "worktree": "/home/wyr/.hermes/hermes-agent-repo", "write_set": null}

        ─── 开始执行任务 t_1281e77c ───