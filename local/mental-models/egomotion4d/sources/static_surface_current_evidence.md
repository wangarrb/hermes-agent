# 静态 SurfacePatch 当前证据

这是从 current D 决策派生的有界构建输入，仅下列当前约束属于本模型范围。

- **D64**：停止 HCR/泛化参数扫描。Route 2 必须对齐真实当前关联、替代目标、重关联算法检查和独立 fixed-common 产品门；Route 3 必须先有显式容量证据。
- **D69**：100f 的关键阻塞是 product observer 与 optimizer association 的有效域不一致，不是缺少位姿或更高 surface DoF。`fixed_gated_3d` 使覆盖率核算保持独立。
- **D72**：同一冻结 optimizer/config 已通过 DAGE/Pi3X × scene0/4/22 ×100f 残差门，时序与覆盖分母不变。该结果依赖逐场景独立 GT 校准，不证明 GT-free metric scale 或任意新域可推广。
- **D76**：统一到 canonical gauge 不等于获得米制尺度。source-unit provenance 与 algorithm-equivariance 是两个独立门；D70/D72 仅因逐场景独立 GT 校准才可报告米制结果。
- **D77**：已接受的标量 patch-position 机制采用整 patch 移动或弃权，不丢失任何 baseline-valid member；它不支持联合位姿或额外 DoF 的结论。
- **D79**：legacy correction group 同时存在碎片化和局部链式连接，不能提供可渲染 SurfacePatch 拓扑。拓扑由 organized-depth faces 持有，legacy group 只提供校正证据。
- **D80**：region-pair rows 是相关的审计证据，不是独立约束。优化必须使用 region × canonical-surfel incidence；每个冻结顶点只能有一个输出位置，禁止 seam duplication 和 transitive closure。
- **D82**：exact-incidence reconciliation 是机制正信号，但保持只读，不构成 heldout、mesh 或产品 claim。
- **D83**：full-step geometry 违反局部朝向；唯一允许的是一个全局 topology-certified dyadic step，禁止删面、逐顶点 clamp 或局部 alpha。
- **D84**：topology-certified geometry shadow 保留全部顶点、面和精确 transport，但仍是 diagnostic，不建立 heldout 或产品收益。
- **D85**：leakage-free heldout D2F gain 仅 `0.1775%`，低于 `5%` gate；target fidelity 与同 packet residual 不能替代 heldout benefit。
- **D86**：93f full-face RGB PLY 是真实 visual/representation PASS，不证明 D2 objective 改善几何。
- **D87**：transfer attribution 将收益损失定位为 region aggregation、耦合幅度收缩和方向抵消，而不是 support abstention 或 topology alpha。
- **D88**：exact-incidence projection 证明冻结拓扑下 scalar region translation 无法承载 D77 信号；scalar weight/alpha/support sweep 已关闭。
- **D89**：下一项有界容量测试是 coupled affine-height shared-vertex oracle，不是 deformation-graph optimizer。实现前必须通过 identity、inactive-zero、exact incidence、D88 nesting、scale equivariance、vertex norm、topology step 和 frozen heldout gates。

D44+ 实现时间线、任务状态和未列出的路线均不是本模型的权威内容。
