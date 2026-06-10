# Shadow Hindsight + Ollama 本地化备注

场景：主 Hindsight API（8888）已在跑，不想直接改 Docker 主实例的 LLM 配置，但需要先把 Hermes memory provider 切到本地 Ollama 做 retain/recall 验证。

## 实测可行结构

- 主 Docker Hindsight：`127.0.0.1:8888`
- Shadow Hindsight：`127.0.0.1:8889`
- Ollama OpenAI-compatible API：`127.0.0.1:11434/v1`
- Hermes Hindsight config：`~/.hermes/hindsight/config.json`

`config.json` 关键字段：
```json
{
  "mode": "local_external",
  "api_url": "http://127.0.0.1:8889",
  "bank_id": "hermes",
  "auto_recall": true,
  "auto_retain": true,
  "memory_mode": "hybrid"
}
```

## Shadow API 关键环境变量

```bash
HINDSIGHT_API_LLM_PROVIDER=ollama
HINDSIGHT_API_LLM_MODEL=qwen3.5:9b-local
HINDSIGHT_API_LLM_BASE_URL=http://127.0.0.1:11434/v1
HINDSIGHT_API_RERANKER_PROVIDER=rrf
HINDSIGHT_API_ENABLE_OBSERVATIONS=false
HINDSIGHT_API_SKIP_LLM_VERIFICATION=true
```

## 已验证行为

- `/health` 正常
- `POST /memories` retain 成功
- `recall` 成功
- Hermes 指向 8889 后，本地 Hindsight retain/recall 路径可工作

## 已验证限制

Ollama 下的 `qwen3.5:9b-local` 不支持 tools。Hindsight `reflect` / consolidation 依赖 tool-capable LLM 时会失败，典型报错：

```text
registry.ollama.ai/library/qwen3.5:9b-local does not support tools
```

因此这条路径当前适合作为：
- 本地 retain/recall memory backend
- Hermes 的本地备选模型

不适合作为：
- 完整替代 reflect / consolidation 的 Hindsight 单模型方案

## 进程状态误判坑

如果工具提示启动 wrapper 脚本 exit `-15`，不能直接断言 shadow API 已死。应直接探测：
- 8889 `/health`
- 11434 `/api/version`

在本次环境里，wrapper 被监控层判定结束后，真正的 `python -m hindsight_api.main --port 8889` 仍继续存活。

## 建议

要长期使用 8889，最好做成 systemd --user 服务；否则会话结束或机器重启后不保证自动恢复。
