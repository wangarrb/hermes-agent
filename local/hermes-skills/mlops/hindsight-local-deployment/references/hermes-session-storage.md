# Hermes Session Storage: JSON vs SQLite

## 数据源

| 源 | 路径 | 数量 |
|---|---|---|
| JSON sessions | `~/.hermes/sessions/session_*.json` | ~1492 |
| SQLite | `~/.hermes/state.db` | ~562 sessions + `main` |

## 字段对比

### 消息级别

| 字段 | JSON | SQLite | 说明 |
|---|---|---|---|
| content | ✓ | ✓ | 一致 |
| tool_calls | ✓ | ✓ | 一致，JSON 格式更完整 |
| reasoning | ✓ | ✓ | 一致 |
| reasoning_content | ✓ | ✓ | 一致 |
| codex_reasoning_items | ✓ | ✓ | gpt-5.x 专有 |
| codex_message_items | ✓ | ✓ | gpt-5.x 专有 |
| finish_reason | ✓ | ✓ | 一致 |
| role | ✓ | ✓ | 一致 |
| tool_call_id | ✓ | ✓ | 一致 |
| timestamp | — | ✓ | SQLite 专有（Unix float） |
| token_count | — | ✓ | SQLite 专有（单条消息） |
| tool_name | — | ✓ | SQLite 专有（单独字段） |

### Session 级别

| 字段 | JSON | SQLite | 说明 |
|---|---|---|---|
| model | ✓ | ✓ | 一致 |
| started_at | ✓ (ISO) | ✓ (Unix) | 格式不同 |
| ended_at / last_updated | ✓ | ✓ | 字段名不同 |
| title | — | ✓ | SQLite 专有 |
| source / platform | platform | source | 字段名不同 |
| message_count | ✓ | ✓ | 一致 |
| input_tokens | — | ✓ | SQLite 汇总 |
| output_tokens | — | ✓ | SQLite 汇总 |
| cache_read/write_tokens | — | ✓ | SQLite 汇总 |
| reasoning_tokens | — | ✓ | SQLite 汇总 |
| system_prompt | ✓ | ✓ | JSON 更完整 |
| tools | ✓ | — | JSON 专有（完整 schema） |
| base_url | ✓ | — | JSON 专有 |
| billing/cost | — | ✓ | SQLite 专有 |
| parent_session_id | — | ✓ | SQLite 专有（续接关系） |

## 关键差异

### SQLite 更丰富
- 汇总统计：input_tokens、output_tokens、cache tokens、reasoning_tokens
- billing/cost：estimated_cost_usd、actual_cost_usd、billing_provider
- 实时数据：`main` session 是当前会话，实时写入
- 续接关系：parent_session_id

### JSON 更丰富
- 完整 system_prompt（~27k chars）
- 完整 tools 列表（带 schema）
- base_url（调用哪个 gateway）
- 覆盖更全：1492 vs 562

## 推荐用法

| 场景 | 推荐 | 原因 |
|---|---|---|
| 历史完整导入 | JSON | 覆盖更全 |
| 实时监控/复盘当天 | SQLite `main` | 当前会话实时更新 |
| 带 token/cost 分析 | SQLite | 有汇总统计 |
| 带 system/tools | JSON | 有完整上下文 |

## 导入脚本

当前 `import_sessions_to_hindsight.py` 使用 JSON 路线，正确。

如需实时复盘，可从 SQLite `main` session 抽取最新消息补充。