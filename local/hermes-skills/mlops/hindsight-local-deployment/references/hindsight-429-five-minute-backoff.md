# Hindsight 429 → 300s Backoff Patch (2026-05-03)

## Trigger

适用场景：
- Hindsight retain 队列在 MiniMax / Bailian 这类 provider 上反复报 429 / throttling
- 默认秒级指数退避不足以跨过 provider 的小时级额度窗口
- 用户明确要求“遇到 429 报错，就暂停 5 分钟再试”

## What was learned

实测发现两层重试要分开看：

1. 本地提交脚本层
- `~/.hermes/scripts/import_sessions_to_hindsight.py`
- `~/.hermes/scripts/import_sessions_to_hindsight_aggregate.py`
- 这些脚本只负责往 `/memories` 提交 async retain 请求
- 若提交 API 自身返回 429，应把 429/throttling 单独映射到 300 秒 backoff

2. Hindsight 容器内部 worker / LLM 调用层
- 真实消耗配额、最容易被 429 卡住的是容器内部 worker
- 关键代码路径：
  - `/app/api/hindsight_api/engine/providers/openai_compatible_llm.py`
  - `/app/api/hindsight_api/engine/retain/fact_extraction.py`
- 默认配置虽然支持 `initial_backoff/max_backoff`，但默认上限只有 60s
- 遇到小时级 quota exhausted 时，60s 内持续重试会空转

## Patch applied

### A. Local importer scripts

新增：
- `RATE_LIMIT_BACKOFF_SECONDS = 300`
- `get_retry_delay(error_text, attempt)`

规则：
- 如果错误文本含 `429` 或 `throttling` → 固定 300s
- 否则保持原有短 backoff 列表 `[5, 15, 30, 60]`

### B. Hindsight container internal provider layer

文件：
- `/app/api/hindsight_api/engine/providers/openai_compatible_llm.py`

在 `except APIStatusError as e:` 分支里增加：
- `e.status_code == 429` 或错误字符串含 `throttling`
- `sleep_time = max(300.0, min(initial_backoff * (2**attempt), max_backoff))`
- `await asyncio.sleep(sleep_time)`
- 然后 `continue`

这样 429 不再走普通秒级 jitter backoff，而是至少暂停 5 分钟。

## Important caveat

这次修改后，容器重启过程中出现过额外问题：
- `Connection reset by peer`
- `Connection refused`
- `docker ps` 一度显示 `Exited (1)`
- 日志警告：
  - `pg0 data directory exists ... but no PG_VERSION found`
  - 指向可能的 data corruption / incomplete shutdown 风险

结论：
- 429 → 300s backoff 的补丁本身语法通过，逻辑已写入
- 但如果容器重启后健康检查异常，不能直接把问题归咎于补丁逻辑本身
- 更大概率是：在大量 in-flight tasks + pg0 嵌入式数据库恢复状态下重启，触发了服务恢复不稳定

## Operational advice

1. 改 Hindsight 容器内部代码前，优先评估是否必须重启当前主实例。
2. 如果主实例还在跑大队列，热补丁会伴随中断风险；更稳的方式是：
   - 先接受当前 provider 的自然耗尽
   - 在下一次 planned restart 时带上环境级 backoff 配置或代码补丁
3. 若必须重启：
   - 记录当前 `stats`
   - 保存 `docker logs`
   - 重启后先验证 `/health`
   - 再验证 `/stats`
   - 最后观察 429 日志是否出现新的 `sleeping 300s before retry`
4. 若看到 `PG_VERSION` 相关警告，优先怀疑 pg0 恢复状态，不要立即删除数据卷。

## Useful evidence from this session

- Hindsight 默认重试相关 env：
  - `HINDSIGHT_API_LLM_INITIAL_BACKOFF`
  - `HINDSIGHT_API_LLM_MAX_BACKOFF`
  - `HINDSIGHT_API_RETAIN_LLM_INITIAL_BACKOFF`
  - `HINDSIGHT_API_RETAIN_LLM_MAX_BACKOFF`
- 默认值：
  - initial backoff = 1s
  - max backoff = 60s
- retain chunk 级重试另有一层：
  - `fact_extraction.py` 中 `MAX_CHUNK_RETRIES`
  - 日志形如 `Chunk 0/1 extraction failed (attempt 2/3): RateLimitError. Retrying in 4s...`
- 所以仅改外层提交脚本，不足以控制容器内部的 quota-hit retry 行为。

## Recommendation for future sessions

如果用户再提“429 就等更久再试”：
- 先判断他指的是“本地提交脚本”还是“Hindsight worker 内部”
- 默认优先回答：真正关键的是 Hindsight worker 内部 retain/consolidation LLM 调用层
- 若要稳妥上线，最好把这类 backoff 做成环境变量/配置项，而不是每次手 patch 容器源码

如果用户要求“429 sleep 减半保护”或类似表达，不要默认把 300s backoff 改短；更稳的解释和实现是：保留 429 长 backoff（默认 300s），并在调度层把后续并发减半（floor=1），减少继续撞限流的概率。对 consolidation 并行化，配套在 provider 层暴露/消费 rate-limit-hit 标志，scheduler 检测后把 `HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES` 的运行时有效值从 8 → 4 → 2 → 1 逐级降压。
