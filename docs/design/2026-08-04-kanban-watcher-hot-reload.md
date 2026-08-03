# Kanban watcher 安全热重载与单例设计

日期：2026-08-04

## 目标与边界

在不重启 agent/TUI、不重新运行 `start-kanban.sh`、不丢失 running task claim
的前提下，安全地重载同一 Zellij session 内的全部 watcher。实现只位于 Hermes
仓库托管的 Kanban 插件与 `local/bin/`，不修改 Hermes 主体运行时。

非目标：切换 agent 模型、重启 pane、迁移 task、改变 claim/notification 语义，
或用热重载替代完整 session 启停。

## 身份与单例

实际 claim loop 的唯一身份为：

```text
(board, profile, zellij_session, zellij_pane_id)
```

`--watch-child` 缺少任一字段时 fail closed。watcher 在进入 DB/claim 循环前，
必须获取该 identity 的 non-blocking `flock`。锁位于本机
`/run/user/$UID/hermes-kanban/watchers/` 的 `0700` 目录，文件名使用 identity
摘要，文件内容只记录 PID、proc start-time、identity 和 code revision，不能包含
token 或 task body。锁 FD 在 watcher 整个生命周期及 self-exec 期间保持有效；
同 key 的第二个 watcher 只记录 owner 后退出。锁文件残留不等于锁残留，正确性
只依赖内核锁。

Python 默认把新 FD 标为 close-on-exec，因此 watcher 必须在 exec 前把 lock FD
显式设为 inheritable，并通过 `HERMES_KANBAN_WATCHER_LOCK_FD` 传给新进程；新
进程优先接管该 FD，不能重新 open 同一文件并与自己竞争。正常冷启动不得信任
外部伪造的 FD：只有 reload nonce、lock metadata、`fstat` 与当前 identity 全部
匹配才允许接管。

supervisor 自身也按 `(board, zellij_session)` 获取单例锁。它只发现并管理完全
匹配当前 board/session 的内部 Python watcher；`conda run` wrapper 只作为启动
诊断，不计作 claim loop。restart counter 按完整 watcher identity 统计。

## 热重载协议

新增命令：

```bash
hermes-kanban-reload-watchers --board <board> --session <session> [--profile <role>]
```

命令为每个 watcher 生成 nonce，先在 mode `0600` 的 runtime reload 目录原子
写入 request JSON，再向匹配的内部 Python watcher 发送 `SIGUSR1`，不 kill
wrapper、agent 或 pane。request 只含 nonce、identity、owner PID/start-time、
request time 和期望 code revision，不含 task/prompt/凭证。命令根据 lock metadata
只选择真正持锁的 PID；重复但未持锁进程只报告，
不得发送 reload。默认按 profile 稳定排序滚动处理；一个 watcher ACK 后才处理
下一个，任一失败即停止剩余 reload 并返回非零。同一 watcher 的并发 reload
请求按 nonce 去重，运行中请求只保留最新一个 pending 请求。

signal handler 只设置 reload flag，不做 I/O、DB 或 exec。watcher 在下一次安全
循环边界读取并验证 request JSON 后执行 reload：当前没有打开的 DB transaction，
也不在 inject/composer 操作中。即使存在 running task，也无需等待任务结束。
没有合法 request 的裸 SIGUSR1 只记录并忽略，不能自行生成 reload。

热重载步骤：

1. 将 `active_task/current_run_id/generation/claim_lock`、watcher identity、原 PID
   和 reload nonce 写入 runtime 目录下 mode `0600` 的 handoff JSON；不保存
   prompt 或凭证。写入前后都用当前 DB row 验证一次 active identity。
2. 用当前 interpreter 对 watcher entry point 执行无副作用 reload preflight
   （至少覆盖语法、import 和参数构造）；preflight 失败只写 FAILED ACK，旧 watcher
   继续运行。
3. 关闭 DB connection 和普通日志句柄；保留并显式继承 watcher lock FD。
4. 使用当前 interpreter、原 argv 和过滤后的原 env 执行 `os.execve`；PID 不变。
   `execve` 返回异常时必须在旧进程内重新打开 DB/日志、继续原 heartbeat，并写
   FAILED ACK；不得经过会清理 active claim 的 `finally` 路径。
5. 新进程先接管继承的 lock FD，再用 task DB 验证 handoff：task 仍为 running，
   run/generation/claim_lock/worker_pid 均匹配。验证成功才恢复 heartbeat；不得重新
   claim 或重新注入原 prompt。
6. 成功发送一次恢复后的 heartbeat，再原子写入 ACK。ACK 位于 mode `0600` 的
   runtime reload 目录，包含 nonce、PID、proc start-time、code revision 和恢复的
   task/run。reload 命令验证 PID 未变化、nonce 匹配和 heartbeat 成功后继续下一
   role；完成后清理对应 handoff/ACK，保留限量摘要日志。

handoff 缺失或不一致时 fail closed：不 claim、不 inject，记录精确原因并退出，
让 scoped supervisor 按同一 identity 恢复；不得猜测或清除原 task。

## Supervisor 配合

- `start-kanban.sh` 启动 supervisor 时同时传 `--board` 与 `--session`；移除全局
  `pkill -f kanban-watcher-supervisor`，只处理同 identity 的旧 supervisor。
- watcher 死亡后等待 restart delay，再重新 discover；只有仍无同 key live
  watcher才 spawn，关闭 launcher replacement race。
- self-exec 保持 PID 存活，supervisor 不应把正常热重载当死亡或消耗重启预算。
- supervisor spawn 的 watcher仍受同一 watcher lock 约束，因此任何遗漏的 spawn
  路径都不能形成第二个 claim loop。

## 失败处理与可观察性

- reload 命令输出每个 role 的 `REQUESTED / ACK / FAILED / SKIPPED`，并以非零退出
  表示未全部成功。
- ACK 默认超时 15 秒，可显式覆盖；超时不强杀仍存活的 watcher，只报告其 PID、
  identity 和最后状态。
- `execve` 调用失败时旧 watcher必须原地恢复；只有新代码在 exec 后启动崩溃时
  才交给 supervisor 受限重启。该异常路径不能承诺无缝 handoff，必须显式告警并
  保留原 task 供现有 orphan recovery，而不能伪报 reload 成功。如果旧 watcher
  卡死且持锁，supervisor必须做 PID/start-time 与 pane ownership 校验后才能终止，
  禁止删除锁文件冒充释放锁。
- 热重载不发送任何 Zellij prompt，也不改变 Kanban task 状态。

## 验证

最小自动测试覆盖：

1. 同 key 两进程只有一个能进入 claim loop；不同 key 可并行。
2. identity 不完整 fail closed；残留锁文件不阻塞新进程。
3. fake watcher 收到 `SIGUSR1` 后在安全边界 self-exec，PID 不变、nonce ACK、锁不
   释放给竞争者。
4. running claim handoff 验证后继续 heartbeat，且不发生第二次 claim/inject；篡改
   run/generation/claim_lock/worker_pid 任一字段均拒绝恢复。
5. reload CLI 过滤其他 board/session、忽略 conda wrapper、按 role 滚动并在失败时
   停止。
6. 双 supervisor 同 identity 只有一个运行；不同 board/session 可并行。
7. restart delay 内出现 launcher replacement 时不再 spawn。
8. preflight/`execve` 失败时旧进程继续 heartbeat且不触发 active-claim cleanup；
   exec 后启动崩溃必须可观察且不得产生成功 ACK。
9. 现有 watcher supervisor、listener、result notification 与 composer-safe tests
   全部保持通过。

人工 smoke 使用测试 board/session：保持一个 running task，执行 reload 命令，
核对 watcher PID 不变、claim/run/generation 不变、heartbeat 继续、agent pane/TUI
未重启且没有重复注入。
