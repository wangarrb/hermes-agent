# SQLite 历史导入污染与安全清理（2026-05-05）

## 触发场景

用户曾中断 Hindsight 历史对话扫描/导入，表现为 LLM 调用次数持续增长。后续需要判断“已有结果能否用、接着跑还是重跑、会不会再次爆炸”。

## 关键结论

- 已落库的 `documents` / `memory_units` 不一定废弃，部分 recall 仍可用。
- 但中断前的旧策略可能把 heartbeat、进度日志、compaction handoff、模型循环输出等噪声作为高价值内容 retain，造成 facts 爆炸并污染 recall。
- 不要直接续跑旧 pending/failed 队列；先确认队列为空、provider 是日常 local/recal-only，再 dry-run 新导入策略。
- 若发现单个 document 产生异常大量 facts，应优先做定点清理，而不是立即全库重置。

## 已验证故障形态

示例坏 document：

```text
hermes-sqlite::day-topic::2026-03-25__egomotion4d::0001::617e7f545e2b
```

它单个 document 产生约：

```text
facts: 6582
HEARTBEAT facts: 4627
```

症状：
- Egomotion4D / Hindsight / OpenClaw 相关 recall 都返回 heartbeat / glm-5 循环内容。
- `memory_units` 总数被少数坏 document 支配。
- Hindsight stats 可能显示 pending=0，但 recall 质量仍被已完成的噪声 facts 污染。

## 安全审计步骤

连接 PostgreSQL：

```bash
PSQL=$HOME/.pg0/installation/18.1.0/bin/psql
$PSQL -h /tmp -p 5432 -U hindsight -d hindsight
```

先看队列状态：

```sql
SELECT operation_type, status, COUNT(*) AS n,
       COUNT(*) FILTER (WHERE task_payload IS NULL) AS payload_null
FROM async_operations
GROUP BY operation_type, status
ORDER BY operation_type, status;
```

确认最近 worker 真实队列（Stats API 不一定完整反映内存队列）：

```bash
docker logs hindsight --since 30m 2>&1 | grep -E 'PENDING_BREAKDOWN|payload_null|STUCK|pending='
```

找异常 document：

```sql
SELECT d.id AS document_id,
       COUNT(mu.*) AS facts,
       COUNT(*) FILTER (WHERE mu.text ILIKE '%HEARTBEAT%') AS heartbeat_facts,
       LEFT(MAX(d.original_text), 120) AS sample
FROM documents d
LEFT JOIN memory_units mu ON mu.document_id = d.id
WHERE d.bank_id='hermes'
  AND d.id LIKE 'hermes-sqlite::day-topic::%'
GROUP BY d.id
ORDER BY facts DESC
LIMIT 20;
```

做 recall smoke test：
- `Egomotion4D DAGE local selector ATE_m RPE`
- `hindsight minimax sqlite import`
- `OpenClaw gateway probe bug`

如果三个无关查询都返回同一类 heartbeat/循环日志，说明 recall 已被噪声 document 污染。

## 清理策略

优先级：

1. 定点删除坏 document（推荐第一步）
2. 清理旧 failed/payload_null ops（如果存在，且确认不会误删新任务）
3. 重新 recall smoke test
4. 只有定点清理后仍污染严重，才考虑删除全部 `hermes-sqlite::day-topic::%` 后 full rerun

定点删除前必须备份数据库或至少导出候选 ID。删除 document 可能通过外键 cascade 删除关联 `memory_units` / links；这是期望行为，但必须向用户确认影响范围。

示例：

```sql
-- 先确认影响范围
SELECT d.id, COUNT(mu.*) AS facts
FROM documents d
LEFT JOIN memory_units mu ON mu.document_id=d.id
WHERE d.id='hermes-sqlite::day-topic::2026-03-25__egomotion4d::0001::617e7f545e2b'
GROUP BY d.id;

-- 用户确认后再执行
DELETE FROM documents
WHERE bank_id='hermes'
  AND id='hermes-sqlite::day-topic::2026-03-25__egomotion4d::0001::617e7f545e2b';
```

## 续跑/重跑判断

- “接着旧任务跑”：不建议。旧 pending/failed 可能恢复后继续烧 LLM。
- “按 progress 增量跑新内容”：可以，但先确认 provider 是 Ollama/local、auto_retain=false、pending=0。
- “full 重跑”：不能直接叠加跑。新 filter 会改变 bundle/document_id，可能产生重复 facts；若要重跑，先 purge 旧 SQLite 导入 documents 或新建 bank。

推荐正式流程：

```bash
python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py \
  --mode dry-run --full --group-by day-topic --no-main \
  --prefilter balanced --prefilter-threshold 15 \
  --retain-chunk-size 16000 \
  --local-filter llama3.1:8b-local \
  --backup-filter qwen2:7b-instruct \
  --drop-policy consensus \
  --sample-report 12
```

确认 dropped samples 可接受后，正式提交必须走包装器临时切 MiniMax，完成后恢复 local：

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py sqlite-import-minimax -- \
  --mode submit --full --group-by day-topic --no-main \
  --prefilter balanced --prefilter-threshold 15 \
  --retain-chunk-size 16000 \
  --local-filter llama3.1:8b-local \
  --backup-filter qwen2:7b-instruct \
  --drop-policy consensus \
  --sample-report 12
```

## 防爆炸判据

继续前必须同时满足：

- `pending_operations=0`，`processing_operations=0`
- Docker logs 最近无异常 `PENDING_BREAKDOWN pending>0` / `payload_null` 循环；注意：当前版本里正常运行中的/已完成的 `batch_retain` 行也可能显示 `task_payload IS NULL`，若同时有 `WORKER_TASK` 在推进、`STREAMING RETAIN COMPLETE` 持续出现、pending 数下降，不要把单个 `payload_null=1` 误判为 stuck bug
- 日常 provider 是 Ollama/local；MiniMax 只在 import wrapper 生命周期内启用
- `auto_retain=false`，observations/consolidation 日常关闭
- dry-run 输出 retain chunks 可估算，且用户确认样本
- 单个 session 也必须受 `--max-bundle-chars` 约束。2026-05-05 已修复 `import_sqlite_to_hindsight.py`：单个超大会话会先硬切成虚拟 session；否则 25 万~35 万字符的 bundle 会让 MiniMax retain 长时间 STUCK 或触发 JSON parse retry/内容劫持。若队列里已经提交了这类 pending bundle，先把原 op 标记为 skipped/superseded，再按 6~8 万字符分片重提，记录到 `~/.hermes/hindsight/sqlite_import_skipped_bundles.md`。详细 live queue 处置流程见 `references/hindsight-sqlite-json-loop-guard.md`。

不要只用 documents/bundles 数估算成本；Hindsight 内部会按 `retain_chunk_size` 再分 chunks，每 chunk 可能触发多次 LLM 调用。