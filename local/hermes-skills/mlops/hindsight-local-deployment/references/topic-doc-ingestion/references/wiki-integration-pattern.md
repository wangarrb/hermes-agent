# Wiki Integration Pattern

Using graphify outputs to generate structured Wiki knowledge pages.

## Workflow

1. **Build graph** — Run graphify on paper corpus
2. **Extract insights**:
   - God nodes → Core methods/entities (wiki hub pages)
   - Surprising connections → Cross-community relationships
   - Community labels → Topic clusters (concept pages)
3. **Generate wiki pages** — Create pages per community, with bidirectional links

## Page Types

| Graph Output | Wiki Page Type | Example |
|-------------|---------------|---------|
| Community (method cluster) | Concept page | `concepts/gaussian-splatting-dynamic-scenes.md` |
| Dataset nodes | Dataset page | `datasets/nuScenes.md` |
| God nodes | Featured in concept pages | "DVGT (4 edges)" |
| Surprising connections | Cross-reference notes | "DenoiseGS → DrivingGaussian" |

## Bidirectional Linking

Use Obsidian-style `[[page-name]]` syntax:
- Concept pages link to datasets used by methods
- Dataset pages link to methods that use them
- Summary page links to all concept pages

## Example Structure

```
wiki/
├── concepts/
│   ├── gaussian-splatting-dynamic-scenes.md  # Community 0
│   ├── vggt-variants.md                      # Community 1/2
│   └── dust3r-family.md                      # Community 3
├── datasets/
│   ├── nuScenes.md
│   ├── Waymo-Open-Dataset.md
│   └── KITTI.md
├── papers/
│   └── trajectory-4d-reconstruction-kg.md    # Summary
└── index.md                                   # Updated index
```

## Integration with topic-doc-ingestion

For users with existing topic-doc-ingestion skill:
- graphify output can feed into the wiki generation step
- community detection replaces manual topic classification
- god_nodes provide "Executive Summary" candidates

## HTML Visualization

The generated `graph.html` can be linked from wiki summary page:
```markdown
## 可视化
- HTML 图谱: `/path/to/graphify-out/graph.html`
```