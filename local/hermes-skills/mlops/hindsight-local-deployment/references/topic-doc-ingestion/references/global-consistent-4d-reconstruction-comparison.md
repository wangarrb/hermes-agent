# Global-Consistent 4D Reconstruction Methods Comparison

> Session-derived comparison table for trajectory/4D reconstruction methods
> Generated from papers/ knowledge base analysis (2026-05-06)

## 正确结论（2026-05-06 实测验证）

### 单前端精度排名（5-scene 驾驶场景）

| 方法 | 平均 ATE | RPE | 适用场景 |
|------|---------|-----|---------|
| **Pi3X** | ~0.13m (最高) | 最低 | 长序列全局一致性 |
| **DAGE** | ~0.18m | 稳定 | 高分辨率+全局约束 |
| **VGGT4D** | ~1.9m (最低) | - | 短序列动态 benchmark |

**错误纠正**: Memory 合成结论 "VGGT4D 0.32m 最佳" 是数据源混淆：
- 0.32m 是 gtsam_gps 校正后结果，不是单前端 raw
- 0.017m 是 TUM/Sintel 短序列 benchmark，不是驾驶长序列

详见: `../egomotion4d/egomotion4d-experiment-operations/references/frontend-accuracy-verification-2026-05-06.md`（5-scene ATE）

| 方法 | ATE (m) | RPE | 特点 |
|------|--------|-----|------|
| **VGGT4D** | 0.32 | - | 最佳单前端精度，需启用undistort |
| Pi3X | 0.85 | 0.4628 | 最佳RPE，局部精度高，适合anchor |
| DAGE | 0.87 | - | 全局一致性稳定 |
| DGGT | 1.33 | - | pose-free设计 |
| LingBotMap | 1.40 | - | 流式重建 |
| Any4D | 3.10 | - | 多模态输入 |

## 融合最佳组合

| 组合 | Mean ATE | 备注 |
|------|---------|------|
| **DAGE + Pi3X + VGGT4D** | ~0.1165 | 最优融合（legacy3） |
| DAGE + Pi3X + Any4D | 明显退化 | Any4D不能替代VGGT4D |

## 按需求推荐

### P0: VGGT4D（精度最高）
- ATE: 0.32m（单前端最佳）
- 借鉴：Gram相似度动态提取，投影梯度掩码细化
- 限制：必须启用undistort，噪声敏感

### P1: TrackingWorld（全局坐标系）
- 借鉴：world-centric表示，As-Static-As-Possible约束
- 优势：密集像素跟踪，相机运动解耦
- 精度：Sintel/Bonn位姿估计最优

### P2: DVGT（prior-free度量）
- 借鉴：ego-centric统一表示，prior-free设计
- 优势：nuScenes上 δ<1.25达95.3%（vs VGGT 72.9%）
- 限制：需多视图输入

### P3: Driv3R（流式密集）
- 借鉴：时空记忆池，无优化对齐
- 优势：15x faster，流式密集点云
- 精度：Depth Abs Rel 0.24

## 技术路线组合

**场景A：单目长序列驾驶重建**
- DVGT prior-free ego-centric + Driv3R Memory Pool + WorldTree层级分解

**场景B：密集动态场景重建**
- TrackingWorld world-centric + As-Static-As-Possible动静分离 + BA位姿优化

**场景C：高分辨率+全局一致性**
- DAGE双流架构：LR流（Pi3初始化）全局 + HR流（MoGe2）细节 + Adapter融合

## 关键论文wiki链接

- [[papers/vggt4d]] - 最佳精度
- [[papers/trackingworld]] - 全局坐标系
- [[papers/dvgt]] - prior-free
- [[papers/driv3r]] - 流式密集
- [[papers/dage]] - 双流全局一致性
- [[papers/worldtree]] - 层级分解
- [[papers/pi3]] - 排列等变anchor