---
name: topic-doc-ingestion
description: 处理本地主题文档与 details，做增量提炼、去重合并，并同步到 fact_store 与 wiki
tags: knowledge, ingestion, docs, wiki, fact_store
---

# Topic Doc Ingestion

处理本地已有专题文档、details 文档、研究笔记，做增量整理。

## 适用场景
- 用户要求整理 `memory/专题-*`、`memory/details/*`、研究笔记目录
- 已有文档，需要更新、合并、结构化沉淀
- 输入不是原始聊天批次，而是主题文档本身

## 目标
1. 识别新增内容与重复内容
2. 将稳定结论抽取到 fact_store
3. 更新已有 wiki 页面或新建必要专题页
4. 保持索引和日志一致

## 标准流程

### 1. 扫描主题与 details
重点找：
- 主题文档：专题总览、路线分析、方法总结
- details：案例、补充论据、细节推导、零散记录

### 2. 判断文档角色
每个文档归为：
- 主专题
- 补充细节
- 案例/实验
- 待归档零散记录

### 3. 提炼稳定结论
优先提炼：
- 主题的核心定义
- 稳定技术判断
- 对比结论
- 关键参数/约束
- 已纠正错误

写入 fact_store 的内容应短、稳定、可检索。

### 4. 更新 wiki
优先动作：
- 给已有 wiki 页面补充新结论
- 合并重复主题
- 对 details 做“归并”而不是原文搬运

### 5. 标注尾项
对仍然不成熟、待验证、分散的内容，列成“待补充问题”，不要强行定稿。

## 存储原则
- fact_store：短结论、约束、关系
- wiki：长文总结、专题脉络
- 原文 details：保留，但不等于结构化完成

## 处理 Hindsight / Canonical Memory 输出

当 wiki auto-maintenance 报告包含 Hindsight consolidation/canonical observation 候选时，优先加载并遵循 `hindsight-wiki-maintenance` 技能；本节只保留与本地 topic-doc ingestion 的衔接规则。

### 输入格式识别
- 高层 canonical：`v2_rebuild/gate/canonical-retain-proposal.md`、`v2_cards/**/*.md`、`observations_index.jsonl`
- Weekly/daily fallback：`~/.hermes/hindsight/offline_reflect/weekly/` 和 `daily/`
- 典型旧格式：`标题 - conclusion: ... - evidence: ... - applicability: ... - limitations: ... - tags: ...`
- auto-maintenance 报告路径：`~/wiki/auto-maintenance/latest.md`

### 整理流程
1. **先看 canonical/gate 状态**：确认 published/local、conflict summary、blocking cases、evidence/source refs。
2. **按话题分类**：优先从 wiki `SCHEMA.md`/index/tags 和 `WIKI_MAINTENANCE_KEYWORDS` 推断话题，不在通用流程硬编码项目名。
3. **去重合并**：同一结论的多个 candidate 应合并，保留最新 scope、条件和证据链。
4. **结构化输出**：每个话题生成：
   - 演进过程：按时间顺序描述关键进展
   - 关键发现：核心结论和洞察（精简版）
   - 当前状态：最新配置/方案/结论
   - 待解决问题：limitations/conflict audit 中的开放问题
   - Provenance：canonical observation/evidence/source 文件路径或 ID
5. **LLM 辅助**：可使用 delegate_task 处理大量 candidates，提示词应明确去重、scope、evidence 和“不直接改主 wiki”。

### 输出存放规则（重要）

**自动维护生成的 wiki 内容不直接写入主 wiki！**

| 输出类型 | 存放位置 | 后续处理 |
|----------|----------|----------|
| 原始 auto-maintenance 报告 | `wiki/auto-maintenance/latest.md` | 自动生成 |
| 整理后的候选页面 | `wiki/auto-maintenance/*.md` | **等待用户审查** |
| 整理报告 | `wiki/auto-maintenance/hindsight_candidates_organized.md` | 记录去重映射 |

**流程**：
1. 整理后的候选页面放在 `wiki/auto-maintenance/`
2. 用户审查确认后，**手动**移动到 `wiki/concepts/`
3. 更新 `wiki/index.md` 和 `wiki/log.md`

**注意**：不要未经审查直接写入主 wiki，这是用户明确要求的边界。

## 注意事项
- 不要把 details 全文复制进 wiki
- 不要把尚未稳定的推测当成 fact_store 事实
- 同主题优先更新已有页面，而不是重复建页
- 如果发现 wiki 页面已有陈旧说法，先修正再补充
- **Hindsight consolidation 是原始素材，需加工后入 wiki**：用户希望看到整理过的演进过程/关键信息/进展/结论，而非原始 conclusion/evidence/applicability bullet points

## Obsidian 兼容性

### PDF 可见性配置
Obsidian 默认不显示 PDF 文件。确保 `.obsidian/app.json` 包含：
```json
{
  "showUnsupportedFiles": true
}
```
否则用户无法在 Obsidian 中打开 `raw/papers/*.pdf` 文件。

### graphify 占位文件清理
graphify 生成的 `_COMMUNITY_*.md` 是空占位文件，Obsidian 打开会显示空白。应删除这些文件，保留实际内容页面在 `concepts/`、`papers/` 等目录。

### 论文摘要页面
论文不应只有概念汇总页。关键论文应创建详细摘要页面：
- 位置: `papers/[方法名].md`
- 内容: 基本信息、核心创新、技术架构、与相关方法关系、实验数据集
- 双向链接: 链接到 `concepts/*` 和 `datasets/*`

## graphify Integration

When processing paper corpora, graphify outputs can drive wiki structure:
- Communities → Concept pages (one per topic cluster)
- God nodes → Core entities featured in pages
- Surprising connections → Cross-reference notes
- Dataset nodes → Dataset pages

See `references/wiki-integration-pattern.md` for the full workflow.

## Batch paper wiki creation

For importing 30+ PDF papers to wiki, use the parallel delegate_task + graphify workflow: `references/batch-paper-wiki-creation-2026-05-06.md`. Key pattern: pdftotext extraction → 3-batch semantic extraction → graphify clustering → wiki pages by community → Obsidian config fix (showUnsupportedFiles: true).

## Curator consolidation: absorbed narrow skills (2026-05-01)

This umbrella now owns the following formerly separate session/narrow skills. Full original SKILL.md bodies are preserved under `references/`.

- **batch-chat-ingestion** — 处理批量对话目录（如 chat-memo_xxx），先分类聚类，再分别写入 fact_store 与 wiki See `references/absorbed-batch-chat-ingestion.md`.
- **project-knowledge-graph** — 为 `$HOME/.hermes/hermes_use/projects/` 下的项目卡片升级非代码项目知识图谱；适用于用户要求"项目知识用图谱管理"、回忆项目上下文、维护项目卡片/实验结论/工作流关系时。 See `references/absorbed-project-knowledge-graph.md`.
- **wiki-knowledge-ingestion** — 将对话记录目录整理成结构化知识条目，写入 wiki 系统 See `references/absorbed-wiki-knowledge-ingestion.md`.
- **project-knowledge-graph** — 为 `$HOME/.hermes/hermes_use/projects/` 下的项目卡片升级非代码项目知识图谱；适用于用户要求"项目知识用图谱管理"、回忆项目上下文、维护项目卡片/实验结论/工作流关系时。 See `references/absorbed-project-knowledge-graph.md`.

## 批量论文知识图谱导入流程（2026-05-06）

适用于用户要求"将目录下所有PDF生成结构化Wiki知识库"的场景。

### Pitfall：不要只创建"核心"页面
- **错误做法**：只创建5篇"核心论文"的详细页面，其余用知识图谱报告覆盖
- **正确做法**：为**所有论文**创建详细wiki页面（用户明确抱怨"不是有三十多个pdf吗？"）
- **触发信号**：用户说"批量"、"所有PDF"、"目录下全部"时，必须全覆盖

### 标准流程

**Phase 1: 知识图谱构建**
1. 检测文件类型：`graphify detect`
2. PDF文本提取：`pdftotext` 或 `pymupdf`（优先pdftotext，pymupdf安装可能失败）
3. 语义提取：`delegate_task` 并行处理（3批，每批10-11篇）
4. 图谱构建：`graphify build` + 社区检测 + God Nodes分析

**Phase 2: Wiki页面生成**
1. **概念页**：按社区聚类创建（如 gaussian-splatting-dynamic-scenes、vggt-variants）
2. **数据集页**：为共用数据集创建（nuScenes、Waymo、KITTI）
3. **论文详细页**：为**每篇论文**创建详细页面（50-80行模板）
4. **汇总页**：知识图谱报告 + 总索引更新

**Phase 3: Obsidian配置**
- 检查 `.obsidian/app.json` 中 `showUnsupportedFiles`
- 必须设为 `true` 才能看到 PDF 文件
- 删除 graphify 生成的空 `_COMMUNITY_*.md` 占位文件

### 论文详细页模板

```markdown
---
title: [方法名]
type: paper-summary
tags: [trajectory-reconstruction, 4d, dynamic-scene]
---

# [方法名]

## 基本信息
- **论文**: [完整标题]
- **PDF**: [[raw/papers/[文件].pdf]]

## 核心创新
[1-2条]

## 关系
- **基础方法**: [[相关方法]]
- **相关概念**: [[concepts/xxx]]

## 数据集
- [[datasets/nuScenes]]
```

### 输出结构

```
wiki/
├── papers/           # 论文详细页（每篇50-80行）
├── concepts/         # 技术概念页（按聚类）
├── datasets/         # 数据集页
├── raw/papers/
│   └── graphify-out/ # 图谱可视化 + 报告
│   └── extracted_texts/ # PDF提取文本
```

### 性能参考
- 32篇PDF → 约10分钟处理（含3批并行delegate_task）
- 最终输出：77个markdown文件（32论文页 + 19概念页 + 3数据集页 + 其他）

## 相关参考文件

- `references/global-consistent-4d-reconstruction-comparison.md` — 4D重建方法精度排名与技术路线推荐（基于papers知识库分析）
