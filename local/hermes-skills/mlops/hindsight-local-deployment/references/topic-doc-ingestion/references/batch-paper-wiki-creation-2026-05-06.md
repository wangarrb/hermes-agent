# Batch Paper Wiki Creation Workflow (2026-05-06)

## 场景

批量导入 papers 目录（30+ PDF）到 wiki，需要：
- 提取论文核心信息
- 建立方法间关系（引用、改进、对比）
- 生成结构化 wiki 页面
- 创建双向链接网络

## 推荐流程

### 1. PDF 文本提取

pymupdf 可能不可用，用 pdftotext 替代：

```bash
mkdir -p extracted_texts
for pdf in $(find . -name "*.pdf" -type f); do
    stem=$(basename "$pdf" .pdf)
    pdftotext "$pdf" "extracted_texts/${stem}.txt"
done
```

### 2. 语义提取（并行）

使用 delegate_task 分批并行处理，避免单次 token 超限：

```python
# 分 3 批，每批 10-12 个文件
delegate_task("处理第一批论文...", role="leaf", toolsets=["file", "terminal"])
delegate_task("处理第二批论文...", role="leaf", toolsets=["file", "terminal"])
delegate_task("处理第三批论文...", role="leaf", toolsets=["file", "terminal"])
```

提取 schema:
```json
{
  "nodes": [{"id": "file_prefix_entity", "label": "Human Name", "file_type": "paper", "source_file": "path"}],
  "edges": [{"source": "id", "target": "id", "relation": "cites|improves|compares_to|shares_data_with", "confidence": "EXTRACTED", "confidence_score": 1.0}],
  "hyperedges": [{"id": "he_topic", "nodes": ["id1","id2"], "concept": "Topic Name"}]
}
```

### 3. 知识图谱构建

使用 graphify 构建 + 社区检测：

```bash
$HOME/.local/share/pipx/venvs/graphifyy/bin/python -c "
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
# 合并批次结果，构建图谱
G = build_from_json(merged)
communities = cluster(G)
"
```

### 4. Wiki 页面生成

按社区聚类生成概念页 + 数据集页 + 论文详情页：

```
wiki/
├── concepts/     # 技术聚类页面（如 gaussian-splatting-dynamic-scenes.md）
├── datasets/     # 数据集页面
├── papers/       # 论文详情页（每篇 50-80 行）
└── index.md      # 主索引（按类别分组）
```

### 5. Obsidian 配置修复

如果 PDF 在 Obsidian 中不可见：

```json
# .obsidian/app.json
{
  "showUnsupportedFiles": true  // 必须 true 才能看到 PDF
}
```

## 输出统计示例

- 输入: 32 PDF + 6 md
- 输出: 45 nodes, 36 edges, 21 communities
- Wiki: 从 15 页扩展到 47 页

## 注意事项

1. delegate_task 并行处理时，每批 10-12 个文件，避免 token 超限
2. PDF 链接格式统一：`[[raw/papers/trajectory_4drecon/[文件名].pdf]]`
3. 双向链接使用 Obsidian 语法 `[[页面名]]`
4. index.md 按技术聚类分组，便于导航

## 相关技能

- `paper-parse` - 单篇论文深度研读
- `graphify` - 知识图谱构建与可视化
- `topic-doc-ingestion` - 本地文档增量整理