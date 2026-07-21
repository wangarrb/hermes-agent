# Dynamic Actor Current Evidence

This is a bounded current source for the precision-first dynamic-actor model.

- **D76**: Canonical alignment does not imply metric scale. Actor distances, velocity and box dimensions require explicit source-unit provenance; dimensionless checks remain valid without it.
- **D91**: Dynamic pseudo-labels are identity-first and kinematics-first. Stable track/frame-local instance identity owns the actor; position, velocity, direction, visibility/confidence and trajectory are required. Box extent/yaw are emitted only when observable. Occ/OccFlow are downstream derivatives, not identity owners.
- **D92**: The real P0 source freezes one INFO/Pi3X 93f population and two clean visual ranges: vehicle proposal 1 frames 0-62 and person/rider proposal 2 frames 7-34. This proves bounded source/identity evidence only, not full-sequence stable identity, metric geometry, kinematics, boxes or product quality.
- Stage 1 remains camera-only, no GT annotation and no training. External pretrained 3D-box detection cannot own the core pipeline.
- Pretrained identity/mask proposals are evidence providers. Per-frame detector IDs are not stable actor IDs; ambiguous overlap remains identity-unknown and absent masks do not become static/free space.
- Frame-local raw actor depth is the B0 geometry observation. Static surface may provide background depth, occlusion order, ground support and contamination checks, but cannot replace actor depth.
- Dynamic surface consistency, if attempted later, must motion-compensate actor micro-patches into an actor frame. Direct world-frame fusion would staticize real motion and is forbidden.
- Reference LiDAR or external 4D pseudo-labels are evaluator-only. They never enter generation or fitting; GT-free metrics remain the main research driver.
- Precision is preferred over coverage. Actor, frame or segment abstention is valid whenever identity, depth, kinematics or observability is insufficient.

No Phase3/4DGS history, task status or frame-local detector ID is authority for an accepted actor.
