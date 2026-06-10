# Bailian GLM-5 在 Hindsight 大规模 retain 中的 Rate Limit 问题

## 现象

2026-05-03 把 Hindsight 切到 Bailian GLM-5 处理 625+ pending aggregate retain 队列时，Docker 日志持续报：

```
API error after 11 attempts: Error code: 429
{'error': {'code': 'throttling', 'message': 'hour allocated quota exceeded.'}}
```

## 关键点

1. **GLM-5 并非"无限"额度**。它也有小时级 quota，大规模 retain 队列同样会打满。
2. **与 MiniMax 的区别**：MiniMax 是 1500次/5小时（Token Plan Plus），GLM-5 是小时级 quota 但具体阈值未公开。
3. **对大规模队列的影响**：两者都不适合"无脑跑完几千 pending"的场景，区别只在于打满的速度和重置周期。
4. **表现差异**：GLM-5 429 是 `hour allocated quota exceeded`，MiniMax 是 `Token Plan Plus` 5 小时窗口限制。

## Worker 状态（打满时）

- 8+ `batch_retain` 任务处于 STUCK 状态（stage_age 可达 3000s+）
- 每个任务卡在 `retain_extract_facts` 阶段，不断 retry（attempt N/11）
- Worker stats 显示 `idle=3 in_use=0 waiters=100`（worker 都在重试退避）
- pending 队列数字几乎不下降

## 应对策略

1. 等待 GLM-5 小时 quota 重置
2. 切回 MiniMax M2.7（如果 MiniMax 窗口还没过期）
3. 减少 worker 并发（改 Hindsight 配置限制并发数）
4. 取消部分低优先级 pending operation，让高优先级先处理

## 结论

对于 600+ pending 的大批量 retain，**两个 provider 都会被打满**。真正有效的是：
- 合并推理（从 3400+ 降到 355 个 bundle）
- 按需 cancel 低优先级 pending
- 等待额度重置窗口
