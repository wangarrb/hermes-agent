# Listener 架构重构（2026-06-17）— 已完成

重构完成。基类 `BaseInteractiveListener` + 4 子类继承，替代原 4 个独立 `*_interactive.py`。

## 架构

| 文件 | 类 | 行数 | 说明 |
|------|-----|------|------|
| `base_listener.py` | `BaseInteractiveListener` | ~1000L | 基类：watcher_main + launcher_main 骨架 + 所有共享方法 |
| `codex_listener/codex_kanban_interactive.py` | `CodexInteractiveListener` | ~120L | 子类：Codex CLI idle("> ") |
| `deepseek_listener/deepseek_kanban_interactive.py` | `CodeWhaleInteractiveListener` | ~700L | 子类：steering dismiss + progress watch + idle reclaim + task timeout + watcher restart |
| `reasonix_listener/reasonix_kanban_interactive.py` | `ReasonixInteractiveListener` | ~500L | 子类：whitespace-normalized idle detection + 2-round confirm |
| `claude_listener/claude_kanban_interactive.py` | `ClaudeInteractiveListener` | ~250L | 子类：Claude Code CLI + --append-system-prompt |

## 核心设计

- **抽象方法**：`build_tui_cmd()`, `has_saved_sessions()`, `inject_text()`, `pane_label()`
- **可选 hook**：`on_claim_pre_check`, `on_claim_post_confirm`, `on_task_running_monitor`, `on_watcher_loop_idle`, `build_launch_env`, `build_watcher_extra_args`
- **idle/busy markers 作为类属性**：子类定义 tuple，基类 `_looks_like_idle_pane()` 统一实现
- **inject_text 规则**：Codex/CodeWhale/Claude **禁止 \n**（LF≠Enter 在 raw mode）；Reasonix 是唯一例外（prompt_toolkit 正确处理 LF）

## 注入 bug 根因（已修复）

- **根因**：`write-chars` 写入 `\n`(LF, 0x0A) 在 PTY raw mode 下≠Enter(CR, 0x0D)
- **表现**：文字输入到 TUI 输入框但未提交（"typed but not sent"）
- **修复**：inject_text 压单行 + `write-chars` 后 sleep 0.8s + `send-keys Enter`
- **Ctrl+U 语法**：`send-keys "Ctrl" "u"` 是错的，正确是 `send-keys "Ctrl u"` 单参数

## 废代码（待删除）

| 文件 | 说明 |
|------|------|
| `codex_kanban_listener.py` | 单进程模式，从未启动 |
| `deepseek_kanban_listener.py` | 单进程模式，从未启动 |
| `codex_listener/bin/codex-kanban-listen` | 对应废弃 bin 入口 |
| `deepseek_listener/bin/deepseek-kanban-listen` | 对应废弃 bin 入口 |
