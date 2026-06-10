# Hindsight 每日复盘策略：修复 Retain 遗漏

## 2026-05-04 状态修正

这个用户环境里，**已实现且验证过的导入脚本有两套数据源**：

### JSON session 路线（历史完整导入）
- `$HOME/.hermes/scripts/import_sessions_to_hindsight.py` — 逐条导入
- `$HOME/.hermes/scripts/import_sessions_to_hindsight_aggregate.py` — 合并 bundle（推荐，降低 API 调用）

### SQLite 路线（近期实时导入）
- `$HOME/.hermes/scripts/import_sqlite_to_hindsight.py` — 支持 `--hours` / `--days` 参数，包含实时 `main` session

### 管理脚本
- `$HOME/.hermes/scripts/cancel_hindsight_bank_pending.py`
- `$HOME/.hermes/scripts/monitor_hindsight_hermes_queue.py`

两条导入路线都已包含 `RATE_LIMIT_BACKOFF_SECONDS = 300`，遇到 `429` / `throttling` 会 sleep 300s 再重试。

### 数据源对比

| 来源 | 覆盖范围 | 特点 |
|------|---------|------|
| JSON session | 1492 个历史 session | 覆盖全，适合历史迁移 |
| SQLite | 562 + main 实时会话 | 有实时数据、token/cost 统计 |

### 推荐导入策略（优先 SQLite + 增量）

**用户决定：只用 SQLite 导入，不再用 Aggregate JSON session。理由：SQLite有实时main、有token统计、数据规范。Aggregate bundle碎片化严重（355 bundles）。已禁用Auto-retain（避免烧额度）。Holographic auto_extract保留（本地免费）。**

1. **日常增量导入**（推荐）：
   ```bash
   # 默认增量模式，自动使用上次记录的截止时间
   python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --mode submit
   ```
   - 自动读取 `~/.hermes/hindsight/sqlite_import_progress.json` 里的 `last_imported_timestamp`
   - 只导入比这个时间戳新的 session
   - 成功后更新截止时间（取所有成功 bundle 中最晚的消息时间）

2. **全量导入**（首次或重置）：
   ```bash
   python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --full --mode submit
   ```
   - `--full` 忽略截止时间记录，导入所有 session

3. **指定时间范围**：
   ```bash
   # 最近 24 小时
   python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --hours 24 --mode submit

   # 最近 7 天
   python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --days 7 --mode submit

   # 从指定时间点开始
   python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --since "2026-05-01T00:00:00" --mode submit
   ```

4. **Dry-run 预览**：
   ```bash
   python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --mode dry-run
   ```
   - 显示：sessions 数量、bundles 数量、总字符数
   - 显示：新的截止时间会是多少

### Progress 文件结构

`~/.hermes/hindsight/sqlite_import_progress.json`:
```json
{
  "processed": ["bundle_doc_id_1", "bundle_doc_id_2", ...],
  "last_imported_timestamp": 1777830755.363191,
  "last_imported_iso": "2026-05-04T02:12:35.363191",
  "last_run": "2026-05-04T06:00:00",
  "total_sessions_imported": 436,
  "total_bundles_imported": 83
}
```

### 为什么优先 SQLite

- 覆盖更全：563 sessions vs JSON 1492（SQLite 有实时 main session）
- 实时数据：`main` session 是当前会话，JSON 还没写出来
- 增量友好：有 `timestamp` 字段，天然支持时间范围过滤
- Token/Cost 统计：有 `input_tokens`、`output_tokens` 等汇总信息

## 核心思路

日常 Retain 用低频或本地模型减少成本；每天/按需用强模型重新处理原始对话，补充遗漏细节。

## 关键认知

**Consolidation 不能修复 Retain 遗漏**：
- Consolidation 只处理已有的 facts，看不到原始对话
- 只有重新 Retain 才能提取遗漏的信息

## 流程

```
Day 1:
  对话发生 → Hermes auto-retain (qwen3.5:9b-local)
  → 提取 facts（可能遗漏数值细节、复杂关系）
  → Facts 存入 Hindsight

Day 2 凌晨（定时复盘）:
  从 Hermes 原始 session 记录取出 Day 1 对话
  → 优先读取 ~/.hermes/sessions/session_*.json
  → 如明确需要 SQLite，可从 ~/.hermes/state.db 的 messages 表做兜底
  → 重新 Retain (MiniMax/DeepSeek)
  → 提取新的 facts（补充遗漏）
  → document_id 加 "-review-" 后缀（避免覆盖）
  → Consolidation 合并新旧 facts
  → 生成/更新完整 observations
```

## document_id 策略

**推荐**：使用新 document_id，保留旧数据

```python
# Day 1 auto-retain
document_id = "session_20260504_xxx"

# Day 2 复盘 retain
document_id = "session_20260504_xxx-review-20260505"
```

优点：
- 旧 facts 保留
- 新 facts 补充
- Consolidation 自动合并

缺点：
- 有冗余（Consolidation 需正确识别）

## 能修复的遗漏类型

| 遗漏类型 | 能否修复 | 修复方式 |
|----------|----------|----------|
| 数值细节 | ✅ | 重新 Retain 提取具体数值 |
| 实体遗漏 | ✅ | 重新 Retain 识别更多实体 |
| 关系遗漏 | ✅ | 重新 Retain 建立更多链接 |
| 提取错误 | ⚠️ 部分 | 新 facts 与旧 facts 冲突，Consolidation 需判断 |
| 格式错误 | ❌ | Consolidation 不改变 facts 格式 |

## 局限性

1. **同模型重新处理效果有限**
   - qwen3.5:9b 第二次处理可能仍遗漏
   - 推荐：复盘 Retain 用强模型

2. **新旧 facts 冗余**
   - 如果提取相似内容 → Consolidation 需正确合并
   - 如果 Consolidation 判断错误 → 可能保留冗余或误删

3. **冲突处理**
   - 新旧 facts 可能矛盾（提取错误）
   - Consolidation 需正确判断保留哪个
   - 弱模型 Consolidation 可能判断错误

## 推荐配置

```
日常 Retain → 本地模型（免费，高频）
复盘 Retain → MiniMax/DeepSeek（付费，每天一次）
复盘 Consolidation → MiniMax/DeepSeek（付费，每天一次）
```

成本估算（每天一次）：
- 10-20 个 session 复盘 retain → ~30-60 次 MiniMax API
- 一次 consolidation → ~10-20 次 API
- 总计每天 ~40-80 次（MiniMax 1500次/5小时限制内）

## 实现脚本示例

```python
# 概念示例：SQLite 兜底路线。当前已有正式脚本优先走 ~/.hermes/sessions/session_*.json，
# 不要把这个 SQLite 示例误当成当前默认实现。
# ~/.hermes/scripts/daily_review_retain.py

import sqlite3
import requests
from datetime import datetime, timedelta

def get_yesterday_sessions():
    """从 SQLite 取昨天的 session"""
    conn = sqlite3.connect("~/.hermes/state.db")
    yesterday = datetime.now() - timedelta(days=1)

    sessions = conn.execute("""
        SELECT session_id, messages
        FROM sessions
        WHERE date(created_at) = date(?)
    """, (yesterday.strftime("%Y-%m-%d"),)).fetchall()

    conn.close()
    return sessions

def extract_conversation(messages):
    """提取 user/assistant 主文本"""
    # 解析 messages JSON，只保留 user/assistant 内容
    content = ""
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            content += f"{msg['role']}: {msg['content']}\n\n"
    return content

def review_retain(session_id, content, api_url="http://127.0.0.1:8888"):
    """用强模型重新 retain"""
    # 新 document_id
    document_id = f"{session_id}-review-{datetime.now().strftime('%Y%m%d')}"

    # 调用主实例（MiniMax）
    requests.post(f"{api_url}/v1/default/banks/hermes/memories", json={
        "items": [{
            "content": content,
            "document_id": document_id,
            "context": "daily_review"
        }],
        "async": True  # 异步处理
    })

def main():
    sessions = get_yesterday_sessions()

    print(f"找到 {len(sessions)} 个昨天的 session")

    for session_id, messages in sessions:
        content = extract_conversation(messages)
        if len(content) > 500:  # 只处理有实质内容的
            review_retain(session_id, content)
            print(f"已提交复盘 retain: {session_id}")

if __name__ == "__main__":
    main()
```

## Cron 配置

```bash
# 每天凌晨 2 点执行复盘
0 2 * * * python3 ~/.hermes/scripts/daily_review_retain.py >> ~/.hermes/logs/daily_review.log 2>&1
```

## 验证

```bash
# 检查复盘 retain 是否完成
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats | jq '.queue_status'

# 检查是否有 review document
curl -s "http://127.0.0.1:8888/v1/default/banks/hermes/documents" | jq '.documents[] | select(.document_id | contains("-review-"))'
```

## 相关文档

- `references/hindsight-operations-model-sizing.md` — Retain/Consolidation 模型能力要求
- `references/qwen35-9b-rtx2070-offload-measurements.md` — qwen3.5:9b-local 效率实测