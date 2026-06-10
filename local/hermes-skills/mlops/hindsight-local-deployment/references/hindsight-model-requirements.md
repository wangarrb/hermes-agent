# Hindsight LLM Model Requirements

Session: 2026-05-04
Context: Evaluating qwen3.5:9b-local vs MiniMax M2.7 for different Hindsight operations

## Operation-Level Model Requirements

### Retain (Fact Extraction)

**Task**: 从对话提取结构化事实、实体识别、建立链接

**4-8B 能力评估**：
- ✅ 基础事实提取：能做，基本准确
- ✅ 实体识别：能做，与强模型相近
- ⚠️ 数值细节：可能遗漏具体数值
- ⚠️ 复杂关系：可能简化深层关系
- ⚠️ 结构化输出：JSON 格式可能不稳定

**强模型优势**：
- 保留数值细节（ATE_metric = 2.471）
- 完整关系抽取（scene0/4/15）
- 稳定的结构化输出

**推荐**：本地模型可接受（高频低成本），关键内容用强模型

---

## TopenRouter / deepseek-v4-flash (Current Default, 2026-05-24)

Switched from MiniMax-M2.7 to TopenRouter/deepseek-v4-flash. Cost is ~10x lower, quality comparable for consolidation/retain. Must use `provider=openai` (not `topenrouter`) due to Hindsight provider whitelist validation. See `references/hindsight-llm-provider-switching.md` for full switching procedure.

---

### Consolidation (Observation Synthesis)

**Task**: 合并相似 facts、识别矛盾、生成 observations

**这是质量敏感操作，4-8B 不推荐**。

**4-8B 问题**：
1. **Observation 过于简化**
   ```
   输入: 5 条关于用户偏好的 facts
   弱模型输出: "用户喜欢简洁回复"
   强模型输出: "用户偏好高信息密度回复，要求先结论后解释，
                在 Egomotion4D 项目明确要求 actionable overviews"
   ```

2. **遗漏关键细节**
   - 数值被模糊化
   - 场景/context 被丢弃
   - 关系被简化

3. **误导后续 Reflect**
   - 低质量 observation → Reflect 从 raw facts 重新推理
   - 反而增加 Reflect 成本

**强模型优势**：
- 精炼但保留关键细节
- 准确判断相似度/矛盾
- 高质量 observations → Reflect 高效

**推荐**：必须用强模型（MiniMax/DeepSeek），但可以低频（每天/每周）

---

### Reflect (Agentic Reasoning)

**Task**: 多轮检索、综合推理、生成洞察

**4-8B 完全不推荐**。

**实测数据（RTX 2070 8GB + qwen3.5:9b-local）**：
- Simple reflect: ~133 秒（极慢）
- 质量不稳定：可能泄露 prompt、不 obey 指令

**问题**：
1. Agentic loop 需要强推理能力
2. Context 综合需要大模型理解力
3. Multi-step reasoning 容易断裂

**推荐**：必须用强模型，手动触发

---

## Cost-Quality Trade-offs

| 策略 | Retain | Consolidation | Reflect | 成本 | 质量 |
|------|--------|---------------|---------|------|------|
| **全本地** | qwen3.5 | qwen3.5 | qwen3.5 | 免费 | 低 |
| **分层混合** | qwen3.5 | MiniMax（每天） | MiniMax（手动） | 低 | 中高 |
| **全付费** | MiniMax | MiniMax | MiniMax | 高 | 高 |
| **topenrouter** | deepseek-v4-flash (via topenrouter) | deepseek-v4-flash (via topenrouter) | deepseek-v4-flash (via topenrouter) | 中低 | 高 |
| **推荐 (2026-05-26)** | topenrouter/v4-flash | topenrouter/v4-flash | topenrouter/v4-flash | 中低 | 高 |

---

## Current Production LLM Configuration (2026-05-24)

Provider switched from MiniMax-M2.7 to TopenRouter/deepseek-v4-flash.

- All 4 groups (llm/retain/consolidation/reflect) use the same provider
- `HINDSIGHT_API_LLM_PROVIDER=openai` (Hindsight's OpenAI-compatible mode)
- `HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash`
- `HINDSIGHT_API_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1`
- Latency: ~12-16s per consolidation LLM call (comparable to MiniMax)
- 1M context window

**Key lesson**: Hindsight only accepts specific provider names (see `references/hindsight-consolidation-config.md`). Use `openai` as the provider name for any OpenAI-compatible endpoint (TopenRouter, OpenCode, etc.), then point `BASE_URL` to the actual endpoint.

---

## Local Model Hardware Constraints

**RTX 2070 8GB + qwen3.5:9b-local (Q4_K_M)**:

| num_ctx | GPU 状态 | 吞吐 | 适用 |
|---------|----------|------|------|
| 128 | 纯 GPU (33/33) | ~40 tok/s | 极小 prompt |
| 4096 | 混合 (32/33) | ~18.6 tok/s | 短对话 retain |
| 16384 | 混合 | ~14 tok/s | 中等对话 |
| 32768 | 混合 (29/33) | ~10.2 tok/s | 长对话/reflect |

**结论**：
- 理论上"能跑"，但长 prompt 只能混合 CPU/GPU
- Reflect/Consolidation 的长 prompt → 极慢
- 不适合作为 Hermes/Hindsight 默认引擎

**替代方案**：
- qwen2:7b-instruct (Q4_0): 4k-16k 纯 GPU，~69 tok/s
- Qwen3.5-4B: 更可能在 8GB 上纯 GPU

详见：`references/rtx2070-local-model-sizing-notes.md`

---

## Daily Review Retain Strategy

**概念**：每日用强模型复盘前一天原始对话，补充遗漏

**流程**：
```
Day 1: auto-retain (本地) → facts（可能有遗漏）
Day 2: 从 SQLite 取原始对话
       → 重新 retain (强模型)
       → document_id: 加 "-review-" 后缀
       → Consolidation (强模型) 合并新旧
```

**document_id 策略**：
- 新 ID（推荐）：`session_xxx-review-20260505`
  - 旧 facts 保留
  - 新 facts 补充
  - Consolidation 合并
- 同 ID：删除旧 + 重新处理
  - 无冗余
  - 但丢失已建立的链接

**局限**：
- 同模型重新处理 → 可能提取相似结果（遗漏仍在）
- 建议用更强的模型做复盘 retain

---

## Key Takeaways

1. **Retain**：本地模型可接受（高频低成本）
2. **Consolidation**：强模型必须（质量敏感）
3. **Reflect**：强模型必须（推理敏感）
4. **成本控制**：降低频率而非降低模型质量
5. **硬件限制**：8GB 卡 + 9B 模型 → 长 prompt 只能混合推理

---

_Updated: 2026-05-04_