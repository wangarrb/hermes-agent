# Hindsight + MiniMax 运行与队列处理记录

适用：Hindsight Docker 本地部署使用 MiniMax 作为 LLM，且需要持续消化 Hermes 队列。

## 已验证配置

- LLM provider: `minimax`
- model: `MiniMax-M2.7`
- base_url: `https://api.minimaxi.com/v1`
- embeddings: `local`
- reranker: `local`
- Hermes 默认写入 bank: `hermes`

## 已验证坑

1. 不要把 MiniMax base_url 设成 `/anthropic`
   - 错误配置：`https://api.minimaxi.com/anthropic`
   - 现象：容器启动时在 `verify_connection` 阶段报 `404 page not found`
   - 正确：`https://api.minimaxi.com/v1`

2. MiniMax 仍会遇到 429
   - Token Plan Plus 有 5 小时额度上限
   - 日志会出现 `usage limit exceeded` / `1500/1500 used`
   - Hindsight 会继续重试，但吞吐会明显下降

3. Hindsight 的 async retain 只是排队
   - `POST /v1/default/banks/{bank_id}/memories` + `async=true` 只表示入队
   - 真正可用要看 `/stats` 和 `/operations`

4. 可取消 pending operation
   - `DELETE /v1/default/banks/{bank_id}/operations/{operation_id}`
   - 仅用于取消未完成的 pending 操作
   - 适合清理旧 bank（例如 `hermes-sessions`）的排队，避免继续抢配额

## 推荐监控

```bash
curl http://localhost:8888/health
curl http://localhost:8888/v1/default/banks/hermes/stats
curl http://localhost:8888/v1/default/banks/hermes/operations?status=pending&limit=100
```

## 经验结论

- 旧 bank 的 pending 队列如果不清，会持续占用 worker，让新 bank（`hermes`）推进变慢
- 清理旧 `hermes-sessions` 的 pending 后，worker 会逐步转向 `hermes`
- 监控以 `pending_operations` 下降和 `total_documents` 上升为准，不看脚本退出码