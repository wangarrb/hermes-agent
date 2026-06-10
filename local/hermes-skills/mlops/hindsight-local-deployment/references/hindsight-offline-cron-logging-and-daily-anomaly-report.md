# Hindsight Offline Cron Logging + Daily Anomaly Report

Captured from the 2026-05-06 cron hardening session.

## Goal

Scheduled Hindsight/Wiki jobs must leave enough structured evidence for next-day triage, and the daily report must explicitly say whether anomalies occurred.

## Implemented pattern

### Hindsight offline runner

`~/.hermes/scripts/hindsight_offline_cron_runner.py` should write:

- Raw log: `~/.hermes/logs/hindsight-offline-pipeline/<timestamp>-<task>.log`
- Per-run JSON summary: `~/.hermes/logs/hindsight-offline-pipeline/summaries/<timestamp>-<task>.json`
- Latest aliases:
  - `~/.hermes/logs/hindsight-offline-pipeline/latest-daily.json`
  - `~/.hermes/logs/hindsight-offline-pipeline/latest-weekly.json`
  - `~/.hermes/logs/hindsight-offline-pipeline/latest-summary.json`
- A `.log.path` pointer for humans/tools.

Summary JSON should include at least:

```json
{
  "started_at": "...",
  "finished_at": "...",
  "status": "ok|failed",
  "task": "daily|weekly|both",
  "log_path": "...",
  "summary_path": "...",
  "anomaly_count": 0,
  "anomalies": []
}
```

### Wiki auto-maintenance runner

`~/.hermes/scripts/wiki_auto_maintenance_cron_runner.py` should follow the same shape:

- Raw log: `~/.hermes/logs/wiki-auto-maintenance/<timestamp>.log`
- Per-run JSON summary: `~/.hermes/logs/wiki-auto-maintenance/summaries/<timestamp>.json`
- Latest alias: `~/.hermes/logs/wiki-auto-maintenance/latest-summary.json`

### Daily report

`~/.hermes/scripts/daily_report.py` should report:

- Cron job state/next run/last status.
- Hindsight latest daily/weekly summary status and anomaly count.
- Wiki latest summary status and anomaly count.
- Current Hindsight health/provider/queue state.

**用户日报格式偏好（2026-05-06 修正）**：

- **工作摘要**：
  - 必须显示统计数据：处理了多少对话、多少轮消息、多少次工具调用
  - 必须显示 consolidation 统计：多少个 daily consolidation、多少个 weekly/global consolidation
  - 必须显示 wiki 更新统计：更新了多少个页面
  - 简要总结工作内容，不要只列出会话标题

- **模型使用统计**：
  - 必须包含 Hermes 会话模型 + Hindsight 后台模型
  - 必须明确显示本地模型（如 Ollama `qwen3.5:9b-local`）
  - 必须显示 LLM 调用次数（`api_calls` 列）
  - 必须明确显示本地 Hindsight 状态为"正常"或"异常"，不要模糊

- **异常报告**：
  - 简要告诉用户有没有异常即可
  - 有异常时最多列出 2-3 条，不要详细展开
  - 不要把完整 pipeline_section 全部展示

- **Consolidation 内容**：
  - 不要罗列文件条目
  - 简要总结内容（提取 Executive Summary 段落）
  - 显示最近 3-5 个 consolidation 的内容摘要

- **Wiki 更新**：
  - 不要罗列文件条目
  - 简要总结更新内容（提取文件前几行或关键内容）
  - 显示最近 3-5 个更新页面的内容摘要

Desired daily report headline:

```text
📊 Hermes 日报 - YYYY-MM-DD
✅ 无异常
```

or:

```text
📊 Hermes 日报 - YYYY-MM-DD
⚠️ 发现 N 项异常
```

**日报结构示例**：

```text
📊 Hermes 日报 - 2026-05-06
⚠️ 发现 1 项异常

【工作摘要】
- 处理 22 个对话，3104 轮消息，1671 次工具调用
- 生成 57 个 daily consolidation，23 个 weekly/global consolidation
- 更新 5 个 wiki 页面
- 最近工作内容：
  - Egomotion4D temporal scale 验证
  - Hermes 日报功能增强

【模型使用统计】
| 模型/用途 | 位置 | Provider | 端点 | 会话数 | LLM调用 | 输入tokens | 输出tokens |
|-----------|------|----------|------|--------|---------|------------|------------|
| gpt-5.5 | 远端 | custom | cch.jmadas.com/v1 | 20 | 1171 | 12,049,365 | 766,164 |
| qwen3.5:9b-local | 本地 | ollama | 127.0.0.1:11434/v1 | 后台 | - | - | - |

- 本地 Hindsight 状态：正常

【异常报告】
- 发现 1 项异常/告警
  - ⚠️ hermes-daily-report: state=error

【Consolidation 内容】
- 共 80 个 consolidation
- 最近内容摘要：
  - [weekly] 终端代理配置是CLI下载性能的决定性因素...
  - [weekly] 两处修复：hermes Hindsight provider...

【Wiki 更新】
- 更新 5 个页面
- 更新内容摘要：
  - auto-maintenance/wiki-auto-maintenance-20260506...
```

## Anomaly detection rules

High-signal log patterns:

- `Traceback`, `ERROR`, `Exception`
- `Connection refused`, `ConnectionError`
- `JSON parse error`, `STUCK`, `429`
- non-zero `EXIT N`
- `post failed`, `failed with code`
- `status=failed` / `"status":"failed"`

Important benign case:

- Ignore transient `queue_poll_error` / `Connection refused` if the same log later contains `queue drained`. This is a Docker restart window that self-recovers and should not make the daily report noisy.

Avoid false positives:

- Do not scan the full Hermes cron output prompt/skill dump as a primary anomaly source if a structured summary exists. Cron output often contains historical troubleshooting text like `JSON parse error` from loaded skills.
- Treat stale `jobs.json.last_error` as non-fatal if the job is now `scheduled`, `last_status=ok`, and has a valid `next_run_at`.

## Verification

Run:

```bash
python3 -m py_compile \
  ~/.hermes/scripts/daily_report.py \
  ~/.hermes/scripts/hindsight_offline_cron_runner.py \
  ~/.hermes/scripts/wiki_auto_maintenance_cron_runner.py

python3 ~/.hermes/scripts/daily_report.py
hermes cron list
python3 ~/.hermes/scripts/hindsight_minimax_import.py status
```

Expected normal state:

- Daily report says `✅ 无异常` unless real failures exist.
- Hindsight provider is `ollama`.
- `enable_observations=false`.
- `worker_consolidation_max_slots=0`.
- `pending/processing/failed=0/0/0`.
