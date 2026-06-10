# Hindsight Observations Gap — 诊断与修复方案

## 问题现象

2026-05-06 检查发现：
- Hindsight stats: experience=3283, world=744, **observation=22**
- observations 占比仅 0.5%（应为 10-20%）
- offline consolidation 输出 JSON 无 observations 字段

## 根因分析

`offline_hindsight_reflect_consolidate.py` 的 LLM prompt/response 格式：
- 只输出 `executive_summary` + `markdown`
- 未设计 `observations` 数组结构
- submit 模式未自动 retain 回 Hindsight

检查方法：
```bash
# 检查 daily 输出结构
cat ~/.hermes/hindsight/offline_reflect/daily/2026-05-05/*.json | jq 'keys'
# 应返回: ["document_id", "unit", "model", "llm_json", "raw_text", "markdown_path"]
# 缺失: "observations"
```

## 影响

Observations 是 Hindsight 核心价值：
- 跨 facts 的洞察（如"用户偏好 X"、"项目决策 Y"）
- Recall 只返回散 facts，无高层结论
- Reflect 质量降低（无 observations 作为推理基础）

## 修复方案

### 方案 A：修改 prompt + response 格式

```python
# 在 call_llm 的 system_prompt_for 中要求输出 observations
response_format = {
    "executive_summary": "...",
    "observations": [
        {"insight": "...", "confidence": "high/medium/low", "scope": "user/project/method/domain"}
    ],
    "key_entities": [...],
    "period_summary": "..."
}
```

### 方案 B：添加 --retain-output 参数

让 submit 模式自动：
1. 调用 LLM 获取 observations
2. 将 markdown + observations 组装成 document
3. POST 到 `/v1/default/banks/{bank}/memories`

### 方案 C：独立 offline-retain 命令

专门处理已有 daily/weekly 输出的 retain：
```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-retain \
  --scope daily --date 2026-05-05 \
  --mode submit
```

## 验收标准

修复后：
- observations 数量应达 200-500（占比 10-20%）
- Recall 返回应包含 observations 类型 facts
- 查询"用户偏好"应返回高层洞察而非散 facts

## 相关文件

- `$HOME/.hermes/scripts/offline_hindsight_reflect_consolidate.py`
- `$HOME/.hermes/scripts/hindsight_minimax_import.py`
- `$HOME/.hermes/hindsight/offline_reflect/daily/`
- `$HOME/.hermes/hindsight/offline_reflect/weekly/`

---

_记录时间: 2026-05-06_