# Pose And Gauge Current Evidence

This is a bounded current-D-derived source for the pose/gauge mental model.

- **D19**: Multi-source fusion uses DLT geometry plus independent per-source depth integration, not direct frontend-scale alignment.
- **D20**: DLT anchor-guided depth refinement does not improve temporal consistency and can inject matching noise.
- **D21**: RoMa2 padding/aspect-ratio changes can move cam-Z median from 32.6m to 9.7m; canvas and intrinsics must be transformed consistently.
- **D22**: Cached match coordinates, frame mapping and K scaling can create a 4-5x DLT anchor scale error.
- **D24**: DLT/TSDF/ground/PLY/evaluators may not consume raw depths, intrinsics or poses for promotion; canonical camera-z, canvas/K, world transform and frame identity must be proven.
- **D25**: A DLT anchor PASS is not dense-depth promotion safety. Anchor-depth ratio p50 0.1496 exposes pose/depth gauge mismatch; DLT remains diagnostic.
- **D36**: Pi3X raw-depth/K fixes and DAGE noise-floor optimizer motion are mechanism signals, not promotion-safe geometry claims.
- **D61**: Source-gauge recovery must use Route A evidence-first with independent per-scene pose-depth coupling. Route B is diagnostic-only and cannot bypass product gates.
- **D62**: Scene0 scale cannot be copied to scene4/22. DAGE audits found distinct ratios; source evidence must be per scene.
- **D63**: Scale correction must be a derived source manifest consumed explicitly. Scaled/unscaled runs change populations, so cross-run claims require a fixed-common evaluator.
- **D70**: Pi3X scene0 became promotion-safe only after the same independently calibrated scalar was applied to native pose translation and raw depth.
- **D72**: Six calibrated DAGE/Pi3X cells establish optimizer transfer on promotion-safe sources, not GT-free metric scale, arbitrary domains or dynamic actors.
- **D76**: Canonical gauge is not meter. Source-unit provenance and scale-equivariant algorithm behavior are independent gates; neither substitutes for the other.

Do not infer metric truth from field names, frontend declarations, directory names or a copied scalar.
