# MiniMax API 调用分析与优化

## Rate Limit

- **MiniMax Token Plan Plus**: 1500 requests per 5 hours
- **429 响应**: 需等待 reset time 继续
- **脚本已处理**: `RATE_LIMIT_BACKOFF_SECONDS = 300`

## 调用分布实测（2026-05-04）

### 已统计的导入调用

| 来源 | Bundles/Documents | API 调用 |
|------|------------------|----------|
| SQLite 导入 | 83 bundles | 83 retain |
| Aggregate 导入 | 355 bundles | 355 retain |
| **导入合计** | 438 bundles | 438 retain |

### 未统计的调用（估算）

| 来源 | 估算调用 | 说明 |
|------|---------|------|
| Auto-retain | ~350 次 | Hermes 日常对话自动触发 |
| Consolidation | ~350 次 | Hindsight 内部多次 LLM 调用 |
| Reflect | ~7 次 | 手动触发 |

### 实际调用远超 bundle 数

**关键洞察**：每个 document 的 retain 内部会产生多次 LLM 调用：
- Entity extraction
- Relation building
- 每个约 49 facts，每个 fact 可能多次调用

```
Hindsight Stats (2026-05-04):
- Documents: 713
- Facts (nodes): 35119
- Links: 2.75M
- Facts/document: 49.3
```

所以 **438 个 bundle → 远超 438 次 LLM 调用**。

## 为什么调用偏多？

### 1. Bundle 分得太细

```
SQLite: 平均 5.3 sessions/bundle
Aggregate: 按 week-topic 分组
  - W17__egomotion4d: 134 bundles（最多）
  - W18__egomotion4d: 58 bundles
```

### 2. Auto-retain 未控制

- Hermes 没有显式配置 `auto_retain: false`
- 日常对话中自动触发，每次对话可能调用多次

### 3. Consolidation 内部调用多

- 处理 pending facts 时，每个 batch 可能多次 LLM 调用
- 35119 facts × 多次调用

## 优化策略

### 1. 控制 bundle 大小

**当前**: `import_sqlite_to_hindsight.py` 默认 `MAX_BUNDLE_CHARS = 120000`，并已支持 `--max-bundle-chars` 参数。

**注意**:
- 这个参数只控制“多 session 合包”大小；单个超大的 live `main` session 不会被硬切开。
- Hindsight 仍会按 bank config 的 `retain_chunk_size` 在 bundle 内部二次切分，所以调用估算要看 `Estimated retain chunks`，不要只看 bundle 数。

**建议**:
- 默认保持 `120000`，配合 `retain_chunk_size=8000`，质量更保守。
- 若历史增量内容质量可接受、且想继续降调用，可试 `--max-bundle-chars 200000`；正式改默认前先 dry-run 看 `Estimated retain chunks` 和最大 bundle。

### 1.0. retain_chunk_size 调整建议

`retain_chunk_size` 会直接影响 Hindsight 内部 retain chunks，因此是“实际减少调用次数”的关键参数。它的单位是 **chars**，不是 tokens。

当前建议分三档：

| 档位 | retain_chunk_size | 用途 | 风险 |
|------|-------------------|------|------|
| 保守 | 8000 | 当前默认；技术细节保留更稳 | 调用较多 |
| 平衡 | 12000 或 16000 | full 导入推荐试验档；通常能明显降调用 | 轻微细节遗漏风险 |
| 激进 | 24000+ / 48000 | 只适合已抽样确认质量后使用 | 长 chunk 可能漏数值、条件、细粒度决策 |

当前 full/no-main/day-topic 候选矩阵（2026-05-05，实际 dry-run）：

| prefilter | threshold | chunks@8k | chunks@12k | chunks@16k | chunks@24k |
|-----------|-----------|-----------|------------|------------|------------|
| none | default | 1144 | 787 | 608 | 426 |
| safe | default | 992 | 678 | 524 | 365 |
| balanced | default | 903 | 616 | 478 | 334 |
| balanced | 15 | 722 | 492 | 383 | 265 |
| strict | default | 820 | 561 | 434 | 302 |
| strict | 18 | 550 | 383 | 294 | 216 |
| strict | 22 | 412 | 287 | 222 | 161 |

推荐安全档：`--prefilter balanced --retain-chunk-size 12000` 或 `16000`。如果特别想压调用且愿意抽样验证，可试 `--prefilter balanced --prefilter-threshold 15 --retain-chunk-size 16000`，大约 383 chunks。

不建议直接把 full 默认改到 48000；这会把 chunk 变成很长的混合上下文，容易漏掉数值证据、实验条件、文件路径、决策边界。

### 1.1. Full 导入前的本地初筛（已支持）

当前 `import_sqlite_to_hindsight.py` 已支持 `--prefilter none|safe|balanced|strict` 和 `--prefilter-threshold N`。

定位：这是**非 LLM、本地确定性初筛**，目的是在 full 导入前减少送入 Hindsight retain 的原文体积，从而实际减少 MiniMax/Hindsight 内部 retain chunks。它不会替代 Hindsight 的 fact extraction。

模式说明：
- `none`：不筛，最大保真；适合最终基线估算。
- `safe`：只去明显无价值噪声，如极短确认、压缩 handoff、untrusted metadata 包装等；低风险。
- `balanced`：full 导入推荐先试；保留有项目、配置、错误、验证、结论、偏好、数值证据等信号的消息/会话。
- `strict`：高强度降调用；只适合 dry-run 对比和抽样验证后使用。

当前 full/no-main/day-topic/retain_chunk_size=16000 的 dry-run 参考（2026-05-05，Ollama 0.23 + 更严格规则）：

| mode | records | bundles | chunks@16k | chars | vs none |
|------|---------|---------|------------|-------|---------|
| none | 448 | 117 | 610 | 8.73M | baseline |
| safe | 417 | 100 | 524 | 7.54M | -14.1% chunks |
| balanced default (threshold=7) | 381 | 83 | 448 | 6.49M | -26.6% |
| balanced default + llama3.1/qwen2 consensus | 404 | 85 | 450 | 6.51M | -26.2% |
| balanced threshold15 | 340 | 73 | 378 | 5.40M | -38.0% |
| balanced threshold15 + llama3.1/qwen2 consensus | 379 | 80 | 389 | 5.47M | -36.2% |
| strict default (threshold=12) | 358 | 78 | 380 | 5.40M | -37.7% |
| strict threshold18 | 295 | 68 | 286 | 3.97M | -53.1% |
| strict threshold22 | 227 | 56 | 212 | 2.92M | -65.2% |

Recommendation: prefer `balanced --prefilter-threshold 15 + llama3.1 primary + qwen2 backup + consensus` before MiniMax submit. It gives nearly the same chunk reduction as strict default while allowing model rescue for session-level gray-zone drops. Treat strict/high-threshold modes as high-risk unless dropped samples are manually audited.

推荐 full 导入前先跑：

```bash
python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --mode dry-run --full --group-by day-topic --no-main --prefilter none
python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --mode dry-run --full --group-by day-topic --no-main --prefilter safe
python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --mode dry-run --full --group-by day-topic --no-main --prefilter balanced
```

若要更激进，再试：

```bash
python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py --mode dry-run --full --group-by day-topic --no-main --prefilter strict --prefilter-threshold 22
```

正式 submit 前必须看 `Estimated retain chunks`，不要只看 bundles。

### 1.2. 规则 + 本地模型辅助筛选（2026-05-05）

当前本地模型小样本甄别结果（2026-05-05，Ollama 0.23.0）：
- `llama3.1:8b-local`：JSON 稳定，12/12 可解析，筛选准确 11/12；推荐作为主筛选模型。
- `qwen2:7b-instruct`：修复 manifest 后 JSON 稳定，12/12 可解析，筛选准确 10/12；偏保守，适合作为备份/救援模型。
- `gemma4:4b-local`：升级后可运行，但在严格 JSON gate 下经常输出 `{}`，低价值内容几乎全保留；不适合自动 drop gate。
- `qwen3:8b-local`、`deepseek-r1:7b-distill-qwen-local`：输出不稳定、慢或过度保留，不推荐做自动 drop gate。

推荐架构：规则 hard keep/drop → `llama3.1:8b-local` 判断灰区 → `qwen2:7b-instruct` 复核将被 drop 的弱信号内容 → 分歧则保留。模型只做辅助筛选，不替代 Hindsight retain；保留内容尽量用原文，不先做本地总结，避免长期细节丢失。

脚本已接入本地模型复核和抽样报告：

```bash
python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py \
  --mode dry-run --full --group-by day-topic --no-main \
  --prefilter balanced --retain-chunk-size 16000 \
  --local-filter llama3.1:8b-local \
  --backup-filter qwen2:7b-instruct \
  --drop-policy consensus \
  --sample-report 8
```

实现约束：本地模型只复核规则准备丢弃的 session 级灰区；drop-policy=`consensus` 时只有主备都 drop 才丢弃。脚本串行调用 Ollama；触发备份模型前会卸载主模型，备份完成后卸载备份模型，避免双模型同时占显存。dry-run 会输出 kept/dropped 抽样与 drop reasons。

完整设计已写入：`$HOME/.hermes/hindsight/sqlite_full_import_filter_design.md`

### 2. 改 group-by 策略

| group-by | Bundle 数 | 说明 |
|----------|-----------|------|
| `day-topic` | 83 | 当前默认 |
| `day` | ~15 | 减少 5 倍 |
| `month-topic` | ~20 | 减少 4 倍 |
| `week-topic` | ~35 | Aggregate 默认 |

**建议**: 默认仍用 `--group-by day-topic`，因为它在调用次数和主题纯度之间更稳。`--group-by day` 可显著减少 bundle 数，但会混主题；只有在 dry-run 显示调用仍偏多且抽样 recall/事实质量通过后，再用于正式导入。

### 3. 禁用 Hermes/Hindsight 日常 auto-retain

当前用户默认策略是：日常 8888 用 Ollama/local，只 recall，不在线写 Hindsight；正式 SQLite 导入才临时切 MiniMax。

关键配置：
- `~/.hermes/hindsight/config.json`: `auto_retain=false`, `memory_mode=context`, `recall_prefetch_method=recall`
- Hindsight env/bank config: `enable_observations=false`, consolidation worker slots = 0

**效果**: 日常对话不再自动触发付费 retain；Recall 不调用 LLM。

### 4. 降低 consolidation 频率

检查 Hindsight 配置：
- consolidation 频率（每周一次？）
- batch size
- 用本地模型处理非关键 facts

## 快速行动清单

1. **已完成**: `import_sqlite_to_hindsight.py` 支持 `--max-bundle-chars`，dry-run 会输出 `Estimated retain chunks`。
2. **已完成**: full 导入支持本地确定性初筛 `--prefilter none|safe|balanced|strict` 和 `--prefilter-threshold N`；它不额外调用 LLM，但能实际减少送进 Hindsight retain 的字符数/chunks。
3. **已完成**: 支持本地模型灰区复核 `--local-filter/--backup-filter/--drop-policy consensus`，并支持 `--sample-report N` 输出 kept/dropped 抽样报告。
4. **已完成/默认要求**: Hermes Hindsight `auto_retain=false`，日常 `memory_mode=context` + `recall_prefetch_method=recall`，避免在线写入烧 MiniMax。
5. **full 安全档推荐**: 先 dry-run 对比 `--prefilter none/safe/balanced`；默认优先 `balanced + retain_chunk_size=16000`，再用包装器正式导入。
6. **可选优化**: 若调用仍偏多，在保证质量前提下优先试 `--max-bundle-chars 200000`；`--group-by day` 会进一步降 bundle 数但混主题，需抽样验证 recall/事实质量后再作为默认；`strict`/高阈值只作为抽样验证后的高风险降调用模式。

## 调用估算公式

```
总调用 ≈ Import retain × (1 + α × Facts/document)
       + Auto-retain × N对话
       + Consolidation × β × Pending facts

其中：
- α ≈ 2-3（每个 fact 的内部调用）
- β ≈ 0.5-1（consolidation batch factor）

假设：
- Import: 438 bundles → 438 × (1 + 3 × 49) ≈ 65000 次（理论最大）
- 实际 MiniMax 只做 retain，内部调用可能是 qwen3.5 本地
- 所以实际 MiniMax 调用 ≈ 438 + Auto-retain + Consolidation
```

---

*更新: 2026-05-04*