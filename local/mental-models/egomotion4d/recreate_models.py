#!/usr/bin/env python3
"""Recreate all 4 mental models with optimized source_query.

Key improvements over v1:
- Explicit D-ref ranges to guide retrieval
- Explicit exclusion of implementation details, progress logs, config dumps
- Bilingual query
- Tighter scope per model
"""
import hashlib, json, os, requests, time, sys

API = "http://127.0.0.1:8888"
BANK = "hermes"

models = [
    {
        "id_suffix": "research-guardrails",
        "name_a": "Egomotion4D 算法设计避坑指南 (active A)",
        "name_b": "Egomotion4D 算法设计避坑指南 (candidate B)",
        "source_query": """生成一份精简的 Egomotion4D 算法设计避坑指南。只包含当前有效结论，删除已被 D 编号 supersede 的内容。

Produce a concise current-only Egomotion4D algorithm-design guardrail. Focus on D19-D43 decisions.

Include:
- Load-bearing invariants (max 8): canonical geometry promotion-safe, fixed denominator, same mask口径, observer/optimizer consistency
- Closed routes with reopen conditions (max 8): DLT dense-depth refine, USN scale normalization, PDAF affine-depth, DDG global coeff, PPMB scalar-offset, CTLD locality, MAD-BA temporal layer
- Pitfall router: P-id, trigger, one-line warning, locator (no full details)
- Open hypotheses with cheapest falsifier

Exclude:
- Implementation details (variable names, CLI commands, config dumps, dependency issues)
- Progress logs, task status, timeline
- Non-algorithmic bugs

Cite D/task/report/artifact anchors for every actionable item. Mark unanchored as UNVERIFIED_LEAD. Never use other mental models as sources.""",
        "max_tokens": 1536,
    },
    {
        "id_suffix": "static-surface",
        "name_a": "Egomotion4D 静态面避坑指南 (active A)",
        "name_b": "Egomotion4D 静态面避坑指南 (candidate B)",
        "source_query": """生成一份精简的 Egomotion4D 静态面（static surface, surfel, shared surface, patch surface）算法设计避坑指南。只包含当前有效结论。

Focus on D23-D35 decisions about static surface optimization.

Include:
- Current design goal: fixed-pose shared surfel refine, not multi-frontend TSDF patch
- Hard constraints: canonical geometry promotion-safe (D24), fixed denominator, same mask口径
- Closed routes: static surfel map-level promotion (D26, frame-colored layering退化), B0-fallback to patch_surface consensus (D30-D31), Shared Surface Objective v2 L0-L3 (D32), region_conflict_selector scope (D35)
- Pitfall router: P-id, trigger, one-line warning, locator
- Open hypotheses: window-conditioned support-denominator v2 (D27)

Exclude:
- Implementation details, config parameters, resolution settings
- Progress logs, task status
- Non-algorithmic bugs

Cite D/task/report anchors. Mark unanchored as UNVERIFIED_LEAD.""",
        "max_tokens": 2048,
    },
    {
        "id_suffix": "pose-gauge",
        "name_a": "Egomotion4D 位姿与尺度避坑指南 (active A)",
        "name_b": "Egomotion4D 位姿与尺度避坑指南 (candidate B)",
        "source_query": """生成一份精简的 Egomotion4D 位姿与尺度（pose, CAN, gauge, scale, frontend fusion, pose-depth coupling）算法设计避坑指南。只包含当前有效结论。

Focus on D19-D25 and D36-D41 decisions about pose and gauge.

Include:
- Current design goal: fixed-pose shared surfel refine (D23), not GTSAM or GPS/CAN
- Hard constraints: canonical geometry (D24), DLT diagnostic-only (D25), pose_scale_diagnostic_only, depth_gauge_frontend_declared_only
- Closed routes: DLT anchor-guided depth refine (D20), RoMa2 aspect ratio scale offset (D21), p6 cached coords incompatibility (D22), USN/SPN normalization (D39), DDG global coeff (D40), PDAF affine-depth (D38)
- Key finding: obs/pred cross-depth spread is NEARFIELD measurement artifact, NOT depth-dependent disease (D41)
- Pitfall router: P-id, trigger, one-line warning, locator

Exclude:
- Implementation details, config parameters
- Progress logs, task status
- GTSAM/GPS/CAN implementation specifics (already abandoned per D23)

Cite D/task/report anchors. Mark unanchored as UNVERIFIED_LEAD.""",
        "max_tokens": 1536,
    },
    {
        "id_suffix": "dynamic-actor",
        "name_a": "Egomotion4D 动态目标避坑指南 (active A)",
        "name_b": "Egomotion4D 动态目标避坑指南 (candidate B)",
        "source_query": """生成一份精简的 Egomotion4D 动态目标（dynamic actor, mask, tube, moving object, identity, kinematics）算法设计避坑指南。只包含当前有效结论。

Focus on dynamic actor decisions and Phase 3 plans.

Include:
- Current status: dynamic modeling is experimental line and Phase3 target, not main delivery (contrast with D23 static background)
- Design constraints: identity-first approach, frame-local raw-depth observations, kinematic smoother
- Closed routes and blockers: 4DGS deformation branch frozen (dx/ds/dr), no product promotion for dynamic
- Pitfall router: P-id, trigger, one-line warning, locator
- Open hypotheses: person branch kinematic smoother, SAM2 mask integration

Exclude:
- Implementation details, config parameters
- Progress logs, task status
- Non-algorithmic bugs

Cite D/task/report anchors. Mark unanchored as UNVERIFIED_LEAD.""",
        "max_tokens": 2048,
    },
]

spec_root = os.path.expanduser("~/.hermes/mental-models/egomotion4d/specs")
for model in models:
    spec_path = os.path.join(spec_root, f"{model['id_suffix']}.json")
    if not os.path.isfile(spec_path):
        continue
    with open(spec_path, encoding="utf-8") as spec_file:
        spec = json.load(spec_file)
    query_parts = [spec["source_query"].rstrip()]
    if spec.get("inline_source_files"):
        model_root = os.path.dirname(spec_root)
        for relative_path in spec.get("source_files", []):
            source_path = os.path.join(model_root, relative_path)
            with open(source_path, encoding="utf-8") as source_file:
                source_content = source_file.read().rstrip()
            with open(source_path, "rb") as source_file:
                source_sha = hashlib.sha256(source_file.read()).hexdigest()
            query_parts.append(
                f"BEGIN AUTHORITATIVE SOURCE {relative_path} sha256={source_sha}\n"
                f"{source_content}\n"
                f"END AUTHORITATIVE SOURCE {relative_path}"
            )
    model["source_query"] = "\n\n".join(query_parts) + "\n"
    model["max_tokens"] = spec["max_tokens"]
    model["tags"] = spec["tags"]

trigger = {
    "mode": "full",
    "refresh_after_consolidation": False,
    "fact_types": ["observation", "world"],
    "exclude_mental_models": True,
    "tags_match": "all_strict",
    "include_chunks": False,
    "recall_max_tokens": 16384,
}

created = []
for m in models:
    for slot in ["a", "b"]:
        payload = {
            "id": f"egomotion4d-{m['id_suffix']}-{slot}",
            "name": m[f"name_{slot}"],
            "source_query": m["source_query"],
            "tags": m.get("tags", ["egomotion4d"]),
            "max_tokens": m["max_tokens"],
            "trigger": trigger,
        }
        try:
            resp = requests.post(f"{API}/v1/default/banks/{BANK}/mental-models", json=payload, timeout=15)
            d = resp.json()
            if resp.status_code == 200:
                print(f"Created: {payload['id']} (op={d.get('operation_id','?')[:8]})")
                created.append(payload['id'])
            else:
                print(f"FAILED: {payload['id']} - {d}")
        except Exception as e:
            print(f"ERROR: {payload['id']} - {e}")

print(f"\nCreated {len(created)} models. Waiting for reflect operations to complete...")


# After creating models, update registry with per-model evidence SHAs from bundle.
# This ensures new models have deterministic evidence identity from the start.
evidence_path = os.path.expanduser("~/.hermes/mental-models/egomotion4d/evidence_bundle.json")
registry_path = os.path.expanduser("~/.hermes/mental-models/egomotion4d/registry.json")

if os.path.exists(evidence_path) and os.path.exists(registry_path):
    with open(evidence_path) as ef:
        bundle = json.load(ef)
    with open(registry_path) as rf:
        registry = json.load(rf)

    for logical_id, model in registry.get("models", {}).items():
        per_model = bundle.get("per_model", {}).get(logical_id)
        if per_model:
            model["source_evidence_sha"] = per_model["evidence_sha256"]
            model["source_watermark"] = per_model["evidence_sha256"]
            print(f"  Updated {logical_id}: source_evidence_sha={per_model['evidence_sha256'][:16]}...")

    registry["evidence_bundle_sha"] = bundle.get("bundle_sha256", "")
    registry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    tmp = registry_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    os.rename(tmp, registry_path)
    print("registry.json updated with per-model evidence SHAs from recreate")
