# Hindsight Operations Model Sizing & Cost Optimization

**Session**: 2026-05-04
**Context**: User wanted to understand Hindsight LLM call points and optimize costs while maintaining performance.

---

## LLM Call Points in Hindsight

### Retain
- **触发**: 每次 `hindsight_retain` API 调用
- **内部流程**:
  1. Fact extraction (1 LLM call) - 从对话提取结构化事实
  2. Entity resolution (1 LLM call) - 识别实体并链接到知识图谱
  3. Link creation (可能 1 LLM call) - 建立 semantic/temporal/causal 关系
- **LLM 调用次数**: 1-3 次
- **Prompt 类型**: Extraction prompt + entity schema
- **输出**: Structured facts → 存储 + 向量索引

### Reflect
- **触发**: 用户/API 明确请求 (`POST /v1/default/banks/{bank}/memories/reflect`)
- **内部流程**: Agentic loop
  1. 多轮 recall 检索相关记忆
  2. 每轮调用 LLM 决定下一步查询
  3. 最终综合推理生成回答
- **LLM 调用次数**: 多次（取决于 agentic loop depth）
- **Prompt 类型**: Reasoning prompt + bank disposition + retrieved context
- **关键**: **不是自动后台任务**

### Consolidation
- **触发**: retain 完成后自动后台运行
- **内部流程**:
  1. Internal recall 找相似记忆
  2. Hydrate source facts
  3. LLM 判断是否合并/去重/标记矛盾
  4. 生成 Observation（合成知识）
- **LLM 调用次数**: 多次（每批记忆）
- **Prompt 类型**: Similarity + contradiction detection prompt
- **关键**: **自动后台**，但可通过环境变量控制频率和强度

### Recall
- **LLM 调用**: **零次**
- **流程**: 向量检索 → BM25 → Reranker → 返回
- **成本**: Embeddings + Reranker（若用 local 则免费）

---

## Model Capability Requirements

### Why 4-8B Can Handle Retain/Consolidation

**Retain 任务分解**:
- Named Entity Recognition (NER): 4-8B 足够
- Relation extraction: 简单模式匹配，不需要复杂推理
- Structured output (JSON/YAML): Qwen/GLM 4-8B 训练数据充分
- Instruction following: 单步任务，context 短

**Consolidation 任务分解**:
- Similarity detection: 语义相似度，4-8B 可判断
- Contradiction detection: 简单逻辑推理
- Merge decision: 规则-based，不需要 multi-step reasoning

**实测结论**:
- qwen3.5:9b-local retain: 可用，质量略低于 MiniMax 但可接受
- qwen3.5:9b-local consolidation: 431s（极慢但能完成）

### Why 4-8B Cannot Handle Reflect

**Reflect 任务特点**:
- Multi-turn agentic loop: 需要维护长期 context
- Retrieval + reasoning interleaved: 每步决策影响下一步
- Synthesis from multiple sources: 需要 cross-document reasoning
- Quality sensitivity: 用户直接看到输出，不能容忍低质量

**4-8B 失败模式**:
- Context 链条断裂：忘记之前的查询结果
- 推理跳跃：无法综合多来源信息
- Prompt leakage：输出中混入 system prompt
- 不遵循 bank disposition：推理风格不符合用户设定

**实测数据**:
- qwen3.5:9b-local reflect: 133s，输出质量不稳定
- MiniMax-M2.7 reflect: 15-30s，质量稳定

---

## Cost Optimization Architecture

### Two-Instance Pattern (User's Current Setup)

| 实例 | 端口 | LLM | 数据量 | 用途 |
|------|------|-----|--------|------|
| 主实例 | 8888 | MiniMax-M2.7 (付费) | 632 docs, 23K nodes | 高质量 reflect |
| Shadow | 8889 | qwen3.5:9b-local (免费) | 16 docs | 日常 retain/recall |

**Hermes config**: 指向 shadow 8889 → 日常 retain 不花钱

### Recommended Production Config

```yaml
# Hermes config
hindsight:
  api_url: http://127.0.0.1:8889  # shadow for retain/recall
  auto_retain: true

# 主实例 8888 配置
HINDSIGHT_API_LLM_PROVIDER: minimax
HINDSIGHT_API_LLM_MODEL: MiniMax-M2.7
HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND: 50  # 降低频率
HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET: low

# Shadow 8889 配置
HINDSIGHT_API_LLM_PROVIDER: openai  # OpenAI-compatible endpoint
HINDSIGHT_API_LLM_MODEL: qwen3.5:9b-local
HINDSIGHT_API_LLM_BASE_URL: http://172.17.0.1:11434/v1
HINDSIGHT_API_EMBEDDINGS_PROVIDER: local
HINDSIGHT_API_RERANKER_PROVIDER: local
```

### Cost Breakdown (Estimated)

| 操作 | Shadow 8889 | 主实例 8888 | 成本节省 |
|------|-------------|-------------|----------|
| Retain (每对话) | 0 元 | ~0.02 元/次 | 100% |
| Recall | 0 元 | 0 元 | 0% |
| Consolidation | 0 元（慢） | ~0.5 元/批 | 100%（慢） |
| Reflect (手动) | 不用 | ~1-2 元/次 | 按需付费 |

---

## Consolidation Tuning Parameters

From Hindsight 0.5.3+ docs:

```bash
# 每轮处理上限 → 控制 LLM batch size
HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND=100  # default
# 可降到 10-50 来减少后台开销

# Internal recall 预算 → 控制检索开销
HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET=low  # default
# 选项: low, mid, high

# Source facts token 上限 → 控制 prompt 长度
HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096  # default
# 大 bank 时这个参数很关键，防止 prompt explosion

# LLM batch size
HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=10  # default

# Reranker memory arena（内存优化）
HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA=false  # default
```

**没有"关闭 consolidation"的开关**，但可以:
- `MAX_MEMORIES_PER_ROUND=1` → 极慢，几乎不运行
- 或完全依赖 shadow 做 consolidation

---

## Key Takeaways

1. **Reflect 不是自动的** → 用户明确请求才触发，不用担心后台烧钱
2. **Consolidation 是自动的** → 但可通过参数大幅降低开销
3. **Recall 不调用 LLM** → embeddings/reranker local = 完全免费
4. **4-8B 可胜任 retain/consolidation** → 不需要强模型
5. **Reflect 需要强模型** → 这是唯一真正烧钱的操作
6. **分层架构最优** → shadow 免费 + 主实例按需

---

*Generated from session 2026-05-04, user cost optimization inquiry.*