# Full Pipeline Run & Recall Smoke Verification

实战经验收集，来自 2026-05-20 的 full pipeline 执行 + recall smoke 验证。

## 1. Full Pipeline 调用

### 正确的命令

```bash
cd /home/wyr/.hermes && python3 scripts/hindsight_memory_pipeline.py full \
  --execute \
  --confirm run-hindsight-pipeline \
  --history incremental \
  --poll 60 \
  --native-consolidation-poll 60 \
  --native-consolidation-wait-timeout 86400 \
  --native-consolidation-max-pending 0 \
  --allow-active-native-operations \
  2>&1 | tail -150
```

### 坑

- **`--no-skip-daily` 不识别** — pipeline 不支持这个参数。`full` 模式默认做 daily 步骤，不需要显式传 `--no-skip-daily`。
- **`--allow-active-native-operations`** — 当有 processing 中的 operations 时，不加这个参数会失败。但通常最好先确认 state 干净再跑。

### Full pipeline 做了什么

按顺序：
1. preflight — 健康检查 + patch container
2. queue_drain_before_daily_retain — 等现有队列清空
3. status_step — 快照前状态
4. session_manifest_step — 构建增量 manifest
5. retain_session_step — 通过 MiniMax 增量 retain
6. daily_reflect_step — 每日 reflect consolidation
7. native_consolidation_drain_after_daily — 等 consolidation/observation 完成
8. v2_rebuild_step — 更新 V2 知识图谱卡片
9. weekly_reflect_step — 周度 reflect
10. native_consolidation_drain_after_weekly — 再次等 consolidation
11. conflict_audit_step — 冲突审计
12. repair_zone_proposals — 修复提案
13. proposal_review — 提案审查

## 2. Recall Smoke Verification

### 问题

Hindsight recall API 返回的 JSON 中有控制字符（`\x00-\x08\x0b\x0c\x0e-\x1f`），python 的 `json.loads()` 默认会拒绝这类字符。

### 解决方案

使用 `re.sub` 清洗后再解析：

```python
import re
raw = response_bytes
clean = re.sub(rb'[\x00-\x08\x0b\x0c\x0e-\x1f]', b'', raw)
data = json.loads(clean)
```

或者在 curl 管道中：

```bash
curl -s -X POST 'http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall' \
  -H 'Content-Type: application/json' \
  -d '{"query":"关键词","limit":1}' | \
  python3 -c "
import json, sys, re
raw = sys.stdin.buffer.read()
clean = re.sub(rb'[\x00-\x08\x0b\x0c\x0e-\x1f]', b'', raw)
data = json.loads(clean)
hits = data.get('results',[])
print(f'total={len(hits)}')
for h in hits[:2]:
    print(f'score={h.get(\"score\",\"?\")} text={h.get(\"text\",\"\")[:100]}')
"
```

### 推荐 smoke test 查询词

这些是经过验证能准确命中已有内容的查询（基于本环境的知识库）：

| 查询词 | 预期命中场景 |
|--------|------------|
| `极安AEB轻卡回灌制动覆盖率` | 老王 AEB 工作细节，含具体数值 |
| `单目多模态检测` | 检测相关项目内容 |
| `外厂主目标` | 主目标测距测速对比 |
| `VGGT4D` | Egomotion4D 项目 |
| `Egomotion4D Phase2` | Egomotion4D 项目 |
| `Hermes 升级后` | Hermes fork 升级相关 |

### 验证判断标准

- `total` > 0 基本通过
- `text` 包含查询词或相关领域内容 → 命中准确
- 多个不同领域的查询词都能命中 → 数据覆盖面正常

## 3. 执行后数据参考

2026-05-20 full pipeline 执行结果：

| 指标 | 执行前 | 执行后 | 增量 |
|------|--------|--------|------|
| documents | 2,822 | 3,131 | +309 |
| nodes | 33,071 | 36,460 | +3,389 |
| observations | 21,531 | 23,264 | +1,733 |
| pending_consolidation | 0 | 0 | — |
| failed_consolidation | 0 | 0 | — |