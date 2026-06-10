# SQLite vs JSON Session Data 详细对比

## 数据源定位

| 数据源 | 路径 | 覆盖范围 |
|--------|------|---------|
| SQLite | `~/.hermes/state.db` | 562 sessions + `main`（实时） |
| JSON | `~/.hermes/sessions/session_*.json` | 1492 个历史 session |

**关键差异**：SQLite 有实时 `main` session，JSON 还没写出来。

## Message 级别字段对比

| 字段 | SQLite | JSON | 说明 |
|------|--------|------|------|
| `content` | ✓ ✓ | ✓ ✓ | 一致，完整对话文本 |
| `tool_calls` | ✓ | ✓ | 一致，JSON 格式完整 |
| `reasoning` | ✓ | ✓ | 一致 |
| `reasoning_content` | ✓ | ✓ | 一致（gpt-5.x 专有） |
| `codex_reasoning_items` | ✓ | ✓ | 一致（gpt-5.x 专有） |
| `codex_message_items` | ✓ | ✓ | 一致（gpt-5.x 专有） |
| `finish_reason` | ✓ | ✓ | 一致 |
| `role` | ✓ | ✓ | 一致 |
| `timestamp` | ✓ | — | SQLite 专有（Unix float，精确到微秒） |
| `token_count` | ✓ | — | SQLite 专有（单条消息 token 数） |
| `tool_call_id` | ✓ | ✓ | 一致 |
| `tool_name` | ✓ | — | SQLite 专有（单独字段提取） |

## Session 级别字段对比

| 字段 | SQLite | JSON | 说明 |
|------|--------|------|------|
| `model` | ✓ | ✓ | 一致 |
| `started_at` | ✓（Unix float） | ✓（ISO string） | 格式不同，信息等价 |
| `ended_at` | ✓ | `last_updated` | SQLite 更精确 |
| `title` | ✓ | — | SQLite 专有 |
| `source` | ✓（cli/gateway） | `platform` | 字段名不同 |
| `message_count` | ✓ | ✓ | 一致 |
| `input_tokens` | ✓（汇总） | — | SQLite 专有 |
| `output_tokens` | ✓（汇总） | — | SQLite 专有 |
| `cache_read_tokens` | ✓ | — | SQLite 专有 |
| `cache_write_tokens` | ✓ | — | SQLite 专有 |
| `reasoning_tokens` | ✓ | — | SQLite 专有 |
| `api_call_count` | ✓ | — | SQLite 专有 |
| `system_prompt` | ✓（压缩） | ✓（完整 27k chars） | JSON 有完整原文 |
| `tools` | — | ✓（完整 schema） | JSON 有完整工具列表 |
| `base_url` | — | ✓ | JSON 有 gateway 地址 |
| `billing_provider` | ✓ | — | SQLite 专有 |
| `estimated_cost_usd` | ✓ | — | SQLite 专有 |
| `actual_cost_usd` | ✓ | — | SQLite 专有 |
| `parent_session_id` | ✓ | — | SQLite 专有（对话续接关系） |

## SQLite 独特优势

1. **实时 `main` session**：当前会话实时写入，JSON 还未生成
2. **Token/Cost 统计**：完整的 input/output/cache/reasoning tokens + 成本估算
3. **精确时间戳**：每条消息有 Unix float timestamp（微秒精度）
4. **对话续接关系**：`parent_session_id` 记录会话树结构
5. **Billing 信息**：provider、mode、estimated/actual cost

## JSON 独特优势

1. **完整 system_prompt**：27k+ chars 原文，便于复盘分析
2. **完整 tools schema**：7 tools + 参数定义，便于理解调用上下文
3. **base_url 记录**：知道用的是哪个 gateway/provider

## 增量导入选择

**优先 SQLite**，原因：
- 有 `timestamp` 字段，天然支持时间范围过滤
- 有实时 `main` session，当前会话也能导入
- 截止时间记录可靠（取 `last_message_at`）

**JSON 用于**：
- 历史完整迁移（SQLite 可能早期没有）
- 需要完整 system_prompt/tools 的场景

## SQLite 查询示例

```sql
-- 最近 24 小时的 sessions（含 main）
SELECT id, source, model, started_at, message_count
FROM sessions
WHERE started_at > (strftime('%s', 'now') - 86400)
ORDER BY started_at DESC;

-- 某个 session 的消息
SELECT role, content, timestamp
FROM messages
WHERE session_id = 'main'
ORDER BY timestamp;

-- Token 统计
SELECT
  SUM(input_tokens) as total_input,
  SUM(output_tokens) as total_output,
  SUM(estimated_cost_usd) as total_cost
FROM sessions
WHERE started_at > (strftime('%s', 'now') - 86400);
```

---

*更新: 2026-05-04*