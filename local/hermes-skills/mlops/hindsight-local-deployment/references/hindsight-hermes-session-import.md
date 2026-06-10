# Hermes Session History → Hindsight Import

适用场景：需要把本机 Hermes 历史会话批量导入到本地 Hindsight，且要支持中断续传。

## 环境事实

在这个用户环境里，历史会话主存储不是 SQLite，而是：

- `~/.hermes/sessions/session_*.json`
- 索引文件：`~/.hermes/sessions/sessions.json`

因此批量导入时，优先遍历 `session_*.json`。

## 推荐导入原则

1. 只保留 `user/assistant` 主文本
   - 跳过 `tool` 消息
   - 跳过 `reasoning` / `reasoning_content` / thinking 噪声
   - 否则内容膨胀非常快，会显著拖慢 fact extraction

2. 大会话必须分块
   - 单条 retain 可能被 Hindsight 统计成 4 万 token 级别的大请求
   - 经验上应先按字符粗分块（例如 8k-12k chars/chunk）
   - 每块都带独立 `document_id`，例如：
     - `session_20260410_xxx::chunk000`
     - `session_20260410_xxx::chunk001`

3. 调用 data-plane，不要调 control-plane HTML
   - 正确：`POST http://localhost:8888/v1/default/banks/{bank_id}/memories`
   - 错误：`localhost:9999/api/...`（那是 UI / HTML 路径）

4. 批量导入时优先 `async=true`
   - 同步 retain 很容易被客户端 `read timeout` 打断
   - `async=true` 可以把长处理排入 Hindsight 后台队列，显著降低导入脚本的阻塞感

5. 先跑 fresh，再跑 failed retry
   - 主流程只处理从未尝试过的文件
   - 已失败的单独做二次重试 pass
   - 这样不会让旧失败项反复卡住主队列

## 推荐 payload

```json
{
  "async": true,
  "items": [
    {
      "content": "User: ...\n\nAssistant: ...",
      "document_id": "session_20260410_130439_7c89e6::chunk000",
      "context": "hermes_conversation",
      "timestamp": "2026-04-10T13:04:39.167922",
      "metadata": {
        "session_id": "20260410_130439_7c89e6",
        "chunk_index": "0",
        "chunk_total": "3",
        "platform": "cli",
        "model": "glm-5",
        "source": "hermes_session_import"
      },
      "tags": ["hermes", "session-history", "cli", "glm-5"]
    }
  ]
}
```

## 失败分类建议

### 1. `Empty or too short`
正常跳过即可，通常是很短的测试会话。

### 2. `Read timed out`
通常说明：
- 请求太大
- provider 太慢
- 没有分块
- 用了同步 retain

优先动作：
- 改成 `async=true`
- 缩小 chunk
- 增加退避重试

### 3. `429 throttling` / `concurrency allocated quota exceeded`
说明 provider 并发配额不够。

优先动作：
- 小批次
- 延长退避
- 不要并发轰炸
- fresh / failed 分开跑

### 4. GLM fenced JSON 导致 Hindsight parse retry
Hindsight 日志里会看到类似：
- `JSON parse error from LLM response`
- content preview 里是 ```json fenced block

这不是导入脚本 JSON 错，而是上游 LLM 输出格式影响 Hindsight fact extraction。

缓解方式：
- 缩短单次内容
- 降低上下文噪声
- 使用 async 队列
- 必要时切更稳的 provider

## 导入脚本设计要点

推荐保留这些能力：
- `import_progress.json` 记录 `processed/failed/total/last_run`
- failed 去重（按 file 覆盖旧错误）
- `should_retry()` 专门识别 timeout / 429 / throttling
- backoff 例如 `[5, 15, 30, 60]`
- 每处理 5 个保存一次进度

## 验证命令

健康检查：
```bash
curl http://localhost:8888/health
```

看进度：
```bash
python3 -c "import json, os; p=json.load(open(os.path.expanduser('~/.hermes/hindsight/import_progress.json'))); print(len(p['processed']), len(p['failed']))"
```

看 Hindsight 日志：
```bash
newgrp docker << 'EOF'
docker logs hindsight --tail 50 2>&1
EOF
```

## 本次会话提炼出的关键结论

- 旧版同步导入脚本在 Bailian/GLM 上很容易被 `read timeout=120` 卡死
- 改成“去噪 + 分块 + async=true + retry/backoff”后，导入速度明显改善
- 1 分钟内 processed 数从 145 提升到 244，说明思路是对的
