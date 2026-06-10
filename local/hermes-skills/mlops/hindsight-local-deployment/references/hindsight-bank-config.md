# Hindsight Bank Configuration

## Key Parameters

| 参数 | 默认值 | 推荐值 | 单位 | 说明 |
|------|--------|--------|------|------|
| `retain_chunk_size` | 3000 | 8000（当前保守默认）；48000（激进降调用，需质量抽样） | **chars（字符）** | 内部分块大小，越大调用次数越少，但长 chunk 可能降低细节提取质量 |
| `retain_extraction_mode` | verbose | concise | - | 提取模式，concise 只提取高价值事实 |
| `retain_chunk_batch_size` | 100 | 100 | - | 每批处理的 chunk 数 |
| `consolidation_llm_batch_size` | 8 | 8 | - | Consolidation 批次大小 |

**⚠️ 单位澄清**：`retain_chunk_size` 单位是 **chars（字符）**，不是 tokens。官方文档明确说明。

## retain_chunk_size vs MAX_BUNDLE_CHARS

**这是两个不同层级的参数**：

| 参数 | 位置 | 含义 |
|------|------|------|
| `MAX_BUNDLE_CHARS` | 导入脚本 | 每个 bundle 的打包大小（120000 chars） |
| `retain_chunk_size` | Hindsight bank config | 每个 bundle 内部的切分大小 |

**计算**：bundle 120000 chars / retain_chunk_size = 内部 LLM 调用次数

| retain_chunk_size | 120000 chars bundle 调用次数 |
|-------------------|------------------------------|
| 3000 (默认) | 40 次 |
| 48000 (推荐) | 2-3 次 |

## 查询 Bank Config

```bash
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/config | jq '.config'
```

## 更新 Bank Config

```bash
# 正确的 PATCH 格式（注意 updates wrapper）
curl -X PATCH http://127.0.0.1:8888/v1/default/banks/hermes/config \
  -H "Content-Type: application/json" \
  -d '{"updates": {"retain_chunk_size": 48000, "retain_extraction_mode": "concise"}}'
```

**常见错误**：
- 直接传 `{"retain_chunk_size": 48000}` → 422 missing `updates` field
- 正确格式：`{"updates": {"retain_chunk_size": 48000}}`

## 配置优先级

| 层级 | 优先级 | 说明 |
|------|--------|------|
| Bank config API | 最高 | 单 bank 配置，覆盖全局默认 |
| Server-wide env vars | 低 | `HINDSIGHT_API_RETAIN_CHUNK_SIZE` 等环境变量 |

**已验证**：通过 bank config API 设置的值会覆盖环境变量默认值。

## 三个 auto_retain/auto_extract 系统

| 系统 | 配置位置 | 额度消耗 | 功能 |
|------|----------|----------|------|
| **Hindsight auto_retain** | `~/.hermes/hindsight/config.json` | 烧 MiniMax | 对话过程中调用云端 Hindsight retain |
| **Hermes plugin auto_extract** | `~/.hermes/config.yaml` → `plugins.hermes-memory-store` | 烧 MiniMax | 对话结束时调用 Hindsight retain |
| **Holographic auto_extract** | Hermes 本地 memory 系统 | **免费** | 本地 SQLite 提取，不调用云端 |

**关键区别**：
- **Holographic auto_extract** 是本地免费提取，**应该保留**
- 用户改进的中文模式扩展（18→26个）是 Holographic 层
- **Hindsight auto_retain** 和 **Hermes plugin auto_extract** 都会烧云端额度

**推荐配置**：

```json
// ~/.hermes/hindsight/config.json（日常默认）
{
  "auto_retain": false,
  "retain_every_n_turns": 50,
  "memory_mode": "context",
  "recall_prefetch_method": "recall"
}
```

说明：当前用户环境采用“离线导入构建 Hindsight，日常只 recall 不在线写”的模式；避免 Hermes 在线 auto-retain 或 plugin auto_extract 误触发 MiniMax。Holographic 本地 auto_extract 与 Hindsight 写入是两套系统，保留本地提取不等于开启 Hindsight 付费写入。

## 2026-05-04 实际配置

```json
// Hindsight bank config（当前保守默认）
{
  "retain_chunk_size": 8000,
  "retain_extraction_mode": "concise",
  "enable_observations": false
}
```

```json
// ~/.hermes/hindsight/config.json（日常）
{
  "auto_retain": false,
  "retain_every_n_turns": 50,
  "memory_mode": "context",
  "recall_prefetch_method": "recall"
}
```

## 效果估算

| 优化项 | 旧值 | 新值 | 效果 |
|--------|------|------|------|
| Bundle 分组 | week-topic | month-topic | bundle 数 355 → ~20 |
| retain_chunk_size | 3000 chars | 8000 chars（保守）/ 48000 chars（激进） | 120k bundle 内部约 40 → 15 / 3 次 |
| retain_every_n_turns | 1 | 50 | 对话中 auto-retain 减少 50 倍 |
| **总调用** | 355×40=14200 | 20×3=60 | **减少 ~200 倍** |