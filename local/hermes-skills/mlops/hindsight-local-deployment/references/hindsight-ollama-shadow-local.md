# Hindsight + Ollama Shadow Local Reference

适用场景：已有 Docker Hindsight 跑在 8888，但想快速验证另一条本地 LLM 路径（如 Ollama / Qwen3.5）且不想动现有容器。

## 推荐做法：shadow API

不要直接改线上 8888 容器；另起一个 shadow Hindsight API，例如 `127.0.0.1:8889`，先验证 retain/recall，再决定是否切换 Hermes。

已验证过的一组可用组合：
- API: `127.0.0.1:8889`
- LLM: `ollama / qwen3.5:9b-local`
- Embeddings: `local`（BAAI/bge-small-en-v1.5）
- Reranker: `rrf`
- observations: `false`

## 关键限制：Qwen3.5 本地 Ollama 在 Hindsight 上是“能跑但很慢”，不是“完全不能跑”

早期日志里可能出现过 tools 相关报错或长时间卡住，但后续实测表明：`qwen3.5:9b-local` 在 shadow Hindsight 上并非完全不能做 reflect/consolidation，而是**能返回、但延迟很高且质量一般**。

已验证现象：
- retain: 可用
- recall: 可用
- consolidation: 可完成，但 1 memory 级别也可能要 ~431s
- reflect: 可返回 200，但一个很小的 query 也可能要 ~133s

质量风险：
- 输出可能不严格 obey 指令
- 可能泄露 prompt / analysis / `<think>` 风格内容
- 不适合作为高频默认 reflect 引擎

因此更准确的定位是：
- 若目标是“先把本地长期记忆跑起来”：可以接受 qwen3.5 做 retain/recall，低频时再手动跑 consolidation
- 若目标是“完整替代线上 reflect/reasoning”：不要只以 HTTP 200 为验收，要额外看延迟、输出质量、是否泄露思维痕迹
- observations 仍建议默认关闭，除非你已确认该模型在你的数据规模上能稳定承受后台任务

## 经验步骤

1. 保留现有 8888 Docker 服务不动。
2. 起 shadow API 到 8889，指向 Ollama OpenAI 兼容端点。
3. 先测：`/health`、list banks、retain、recall。
4. 明确检查日志里实际使用的：LLM / Embeddings / Reranker。
5. 如日志出现 `does not support tools`，立刻关闭 observations，不要继续等后台任务卡死。
6. 再把 Hermes 的 `~/.hermes/hindsight/config.json` 指向 8889。
7. 改完 Hermes Hindsight 配置后，重开 Hermes 会话再验证内置 retain/recall，避免旧进程没热更新。

## 最小验收

```bash
curl http://127.0.0.1:8889/health
curl http://127.0.0.1:8889/v1/default/banks
```

再做一次真实 retain/recall，确认不是只有 health 活着。

## 切换判断

适合切到本地 Ollama Hindsight：
- 你主要需要 retain/recall
- 能接受 reflect 暂时不可用
- 想把云端 quota 压力降下来

不适合直接切：
- 你依赖 reflect / consolidation
- 你需要 observations 持续运行
- 你还没确认模型是否支持 tools
