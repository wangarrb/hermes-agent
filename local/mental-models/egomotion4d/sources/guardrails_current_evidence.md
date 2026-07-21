# Egomotion4D Research Guardrails - Curated Current Evidence (D19-D43)

## 顶层承重共识

## 顶层承重共识（任何算法、指标或实验前先检查）

> **D76：统一坐标系不等于米制坐标系。** `canonical_fused_pose` 只保证
> pose/depth/point cloud 处于同一 gauge；它本身是 temporal + Pi3X-prior，
> 不天然提供真实 meter。字段名 `camera_z_m`、`pose_*_m`、`*_p90_m` 或
> `world_frame_id=*:canonical_fused_pose` 都只是 schema/历史命名，不能充当
> metric provenance。

1. **先过 source-unit gate**：只有 manifest 明确给出独立 metric calibration
   （例如 GT/LiDAR/CAN/外部标定），且 `promotion_safe_geometry=true`、
   pose/depth coupling 无 diagnostic blocker 时，当前 packet 的数值才可按米解释。
2. **再过 algorithm-equivariance gate**：source 可按米解释，也不代表固定长度参数
   能跨 scene/frontend 泛化。新算法若要跨 gauge，长度参数必须由 frozen source
   尺度推导或改成无量纲，并通过整体缩放 `x -> s*x` 后 membership/输出/claim
   按合同等变的测试。
3. **两门不可互相替代**：metric-calibrated source 不自动证明算法跨域；
   scale-equivariant 算法也不会把 diagnostic source 变成米制。未结清任一门时，
   绝对米阈值与 `_m` claim 只能 diagnostic/no-claim。
4. **当前边界**：D70/D72 recovered DAGE/Pi3X manifests 因逐场景 independent GT
   calibration 可报告米制结果；它们不证明 GT-free canonical gauge 是米制。
   详见 D76 与 `topics/canonical-geometry/00-current.md`。

> **D79：correction group 不是 renderable SurfacePatch 拓扑。** INFO/Pi3X
> 93f frozen evidence 已证明 legacy fixed-normal group graph 同时大面积
> fragmentation 且局部 transitive chaining。后续 face、boundary 和 patch
> topology 只能来自 frozen organized-depth support；legacy groups 仅可提供
> plane/correction evidence。禁止重开 group adjacency/radius/extent/normal/
> plane-gap sweep，cross-frame association 在新 gate 前必须保持 direct evidence，
> 不得隐式传递闭包。

> **D80：association pair 不是独立优化约束，共享 mesh vertex 只能有一个输出
> 位置。** Stage C 的全脸 per-frame planar regions 与 direct canonical-surfel
> incidence 已通过，但同一 canonical surfel 跨多个 regions 会组合地产生大量
> 高相关 pair rows；pair table 只允许作 audit/coverage，Stage D 必须回到底层
> `region x canonical-surfel` incidence，禁止 pair-degree weighting 或传递闭包。
> 相邻 patches 若对同一 frozen vertex 给出不同平移，不能靠复制 seam 顶点把矛盾
> 变成裂缝；最终几何必须对每个 frozen within-frame vertex 求唯一位置，并以
> whole-region active-or-abstain 使用 correction evidence。首个 geometry solve 与
> cross-frame duplicate fusion 必须分门验收。

## D19-D43 Exact Current Decisions

| ID | 结论 | 状态 |
|---|---|---|
| D19 | 多源融合采用 DLT 几何框架 + 每源独立 depth 积分进 TSDF；淘汰直接对齐前端 scale 再融合的旧路线 | current |
| D20 | DLT anchor-guided depth refine 不改善跨帧一致性。DLT 双目匹配引入噪声破坏 Pi3X depth 时序平滑性 | current |
| D21 | RoMa2 宽高比导致 DLT 深度尺度偏移：padding 到 1280×1280 后 cam-Z median 从 32.6m 降到 9.7m | current |
| D22 | p6 RoMa2 缓存匹配坐标 + 融合位姿不兼容：frame index mapping 和 K 缩放导致 DLT 锚点尺度偏差 4-5× | current |
| D23 | 当前重建主线是 fixed-pose shared surfel refine，不再推进后置多前端 TSDF 补洞式融合。任何"多前端提升"必须在 canonical geometry、same denominator、best single-source baseline、tail/coverage gate 和 RGB/投影可视化全部通过后才能 claim | current |
| D24 | **Canonical geometry 是 promotion-safe 的强制边界**：DLT/TSDF/ground/PLY/evaluator 不得直接消费 raw depths/intrinsics/poses；必须通过 canonical adapter 显式证明 camera_z_m、K/pixel canvas、world_from_camera、world_frame_id 一致，且 promotion_safe_geometry=true | current |
| D25 | DLT anchor PASS ≠ dense-depth promotion-safe。DLT candidate generation 已修通，但 canonical fused pose translation gauge 与 dense depth metric gauge 未对齐（anchor-depth ratio p50 0.1496）。DLT 只能 diagnostic，不得驱动 dense-depth refine 或 TSDF precision claim | current |
| D26 | Static surfel/shared-surface 当前只能 diagnostic，不得 promotion 到 map-level product。residual/road/other_static 有正信号，但 frame-colored layering 三场景全部退化（Δ +0.220~+0.563m） | current |
| D27 | Window-conditioned scene0/Pi3X/100f support-denominator v2 是单场景 promotion-safe 正信号。下一波必须同配置、同分母、同 mask 口径 | current |
| D28 | MAD-BA-core vs temporal layer consensus 线已关闭 `ROUTE_CLOSED / NO_CLAIM_MULTISCENE_FAIL`。Baseline 是 runtime-safe project adaptation（非 full paper-faithful MAD-BA），三场景 delta 全 0.0 | current |
| D29 | Patch/surfel surface 路线不关闭，但 DAGE positive-control / Pi3X regression 边界已明确：DAGE 对弱前端有机制正信号，Pi3X full-region 迁移失败。不得 product promotion，不得跑 scene22 | current |
| D30 | Static-surface 下一轮必须从 B0-fallback high-confidence local update 开始。只允许在信息充足的 local windows/regions 做更新，其余 abstain 保留 B0 | current |
| D31 | Static-surface 重启必须把 patch/surface 作为共享优化变量，不能再退回 per-surfel local update。`patch_surface_consensus` MVP 已实现但 100f timeout，不能标 PASS | current |
| D32 | 下一版 static-surface optimizer 必须采用 Shared Surface Objective v2 的 L0-L3 分层和 active objective。禁止 mean-plane averaging；temporal consistency/visibility/occlusion 必须进入优化 | current |
| D33 | EGTR Stage7-lite non-sky TSDF 当前只允许 scoped diagnostic / objective-smoke。Independent QA `CODE_PASS_EVIDENCE_PASS_NO_CLAIM`。Promotion 阻塞：需再生 heldout source data，至少两个 independent scene/heldout 改善 | current |
| D35 | Active `region_conflict_selector` (mode=tail_entry_freeze) 是当前 fixed-pose Pi3X/100f static surfel 的 scoped/default selector policy。scene22 非零移动通过，scene0/scene4 安全 no-op。不能外推为 DAGE/TSDF/dynamic actor 通用 product pass；W5/W10 仍 diagnostic-only | current |
| D36 | Pi3X raw-depth/K fix 是 scene0 clean-gauge mechanism signal，不是 promotion-safe geometry claim。Canonical blockers (pose_scale_diagnostic_only, depth_gauge_frontend_declared_only, pose_depth_coupling_not_proven) 继续严格。DAGE `0.002m` noise floor 下有 optimizer mechanism pass 但 top-level 仍 BLOCKED_BY_INPUT_QUALITY | current |
| D37 | CTLD locality-decoupling 关闭 `MECHANISM_NO_GAIN_PRODUCT_LEVEL_CLOSED`。36-cell sweep 均未通过 strict locality。screening_lambda 局部有效（eff_halo 0.625→0.053）但 product layer 只接受 no-op。重开条件：减少 raw moving set 或 core count 的新机制 | current |
| D38 | PDAF affine-depth 线关闭 `PDAF_H2_FINAL_CLOSED_COORDINATE_RISK_NO_CLAIM`。H1/H2 的 p90 改善是 nonphysical `b` 造成的 valid-set collapse / fit artifact。不继续扩 affine range；只允许以新的具体 reprojection falsification 或 parallax-triangulation plan 重开 | current |
| D39 | USN/SPN unified-scale normalization：R3-FINAL 关闭 `USN_R3_FINAL_NO_CLAIM_NOT_SAME_DENOMINATOR_CLOSED`；USN-EVAL 进一步关闭 `NO_CLAIM_NOT_TRUE_FIXED_OBSERVATION_CLOSED`。CP2 评估器根本缺陷仍成立：改 depth scale 会改变 surfel population，off/on 不是同一批点。USN-EVAL 的真实 scene22/DAGE run 仍使用 surfel-derived sample proxy，未产出 per-pixel fixed-observation NPZ，OFF/ON/common 为 `54589/36443/36443`，ON residual p90 `1.846→2.052m` 变差，因此不得发布 W3/W4，也不得称为 valid true no-gain。归一化框架本身仍正确可用；重开只能走 true fixed pixel-observation export/evaluator。 | current |
| D40 | DDG 线关闭 `DDG_FINAL_CLOSED_GLOBAL_COEFF_BOUNDARY`。**真进展：affine depth adapter 基础设施 + DAGE scene0 全局单位 bug 修复**（DAGE median 0.63m 非物理→`scale≈11.58` 拉回米制，flow p90 `108→42px`）。**⚠️ 与 D41 调和（reviewer 2026-07-02）：DDG 全程在 DAGE 上做，corrected 后 obs/pred spread `10.7x` 是"把 DAGE 拉成 Pi3X 后继承了 Pi3X 的 obs/pred 形状"，而该形状已由 D41(NEARFIELD) 归因为三角化/低视差 measurement artifact，非深度 depth-dependent 病。故不成立"depth-dependent 不一致未解、需逐 bin 修深度"这一 framing（那是重开 D41 已关闭的问题）。D40 确定结论收窄为：全局系数只 recenter obs/pred 中心、不改跨深度形状——但该形状本就是 measurement artifact 非待修深度病。** DAGE 单位修复可作 diagnostic/source-gauge 契约层（promotion_safe_geometry=false）。不得继续全局系数扫。 | current |
| D41 | **跨深度 obs/pred 漂移已归因（NEARFIELD，非深度病）——此前只在实验层，现提升为决策以约束后续推理。** height 锚（独立于三角化，0-40m）实测：**Pi3X 网络深度近处基本正确**（net/height=1.06，0-10m），DAGE net/height=1.60（前端尺度偏大 1.54×，纯 source 差），三角化 Ztri/height=0.346（偏小 3×，三角化/低视差侧问题）。⇒ **obs/pred 跨深度 8.6-10.7× 漂移不是网络深度的 depth-dependent 病，而是"近处深度对 + 远处三角化/测量在低视差失效"的 measurement artifact**；远处(40m+)为几何不可判（DEPTH-TRI，非未解病）。**约束：后续不得把 obs/pred 跨深度 spread 重新 framing 成"深度需逐 bin 修正"的未解问题（D40 曾犯此错，见其修订）。** | current |
| D42 | **知识管理双层优化(元决策)——治"知识在库里≠推理时被查到"。** 本 session 实测三次未对齐(NEARFIELD 已归因却重开=D40重开D41;DDG-CR 误派 scene22 无 matches;off/on bit-identical 两次误读),根因均为"凭新数据下结论,不先核对已确立约束",且关键结论卡在 KG 实验层未提升到决策层。落地两层全局机制:(a) AGENTS.md §8.4 硬规则=下结论/发波前必检索已有 D 号写 related_decisions,能约束后续方向的结论达成即提升为 D 号带"约束:后续不得…",新 decision 落库前做冲突检索;(b) `docs/process/kanban-task-body-template.md` 把 related_decisions/load_bearing_assumptions/数据前提/同分母 设为 ⭐ 必填字段(自觉变填空)。**曾试 (c) PreToolUse hook 硬拦 kanban create,已撤销:hook 仅作用于 Claude Code 单 profile(其他 profile 走各自 runtime 不经过 .claude/settings.json),覆盖面不全,且正则误报(echo/grep 里提到 kanban create 就被拦)。全局防线应在 AGENTS.md(所有 profile 都读)或 hermes CLI 层,不在单 profile hook。** 边界(诚实):(a)(b) 是"应做"非流程强制,降低重开已归因问题的频率但不归零;"读结果/下判断"节点仍靠自觉+(a)规则。 | current |
| D43 | **PPMB V0→V2/V2R1 scalar-offset planar PatchMatch 线关闭 `PPMB_LINE_CLOSED_PRODUCT_NO_CLAIM_100F_CONFIRMATION`。** V0 sparse mechanism、V1 iterative refresh、V2 PCA/robust multi-direction 都未形成 fixed-denominator product improvement；V2 20f render p90 `13.062→13.003` 的边际信号在 V2R1 scene22/DAGE/100f 消失：`accepted_scale=0.0`，residual p90 `6.4933→6.4933`，temporal p90 `8.7167→8.7167`，render p90 `22.0656→22.0656`。**约束：后续不得以 candidate count / propagation changed / shared objective / 20f render p90 / tolerance sweep / scalar-offset PatchMatch 调参重开产品 claim；重开只能是新 higher-DOF surface state 或直接 product-risk-aware objective，并先证明 fixed-denominator product gate 可达。** | current |
