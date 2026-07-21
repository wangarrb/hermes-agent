from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "local" / "hermes-scripts" / "hindsight_daily_noagent.py"
RUN_AB = MODEL_ROOT / "benchmark" / "run_ab.py"
PITFALL_WRITER = MODEL_ROOT / "pitfall_writer.py"
WRAPPER = REPO_ROOT / "local" / "hermes-scripts" / "daily_mental_model_wrapper.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def daily():
    return _load_module("hindsight_daily_noagent_test", SCRIPT)


def _write_evidence(home: Path, logical_id: str, evidence_sha: str) -> None:
    root = home / "mental-models" / "egomotion4d"
    root.mkdir(parents=True)
    (root / "evidence_bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "per_model": {
                    logical_id: {
                        "d_ids": ["D92"],
                        "sources": {},
                        "evidence_sha256": evidence_sha,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_registry(
    home: Path,
    logical_id: str,
    *,
    content: str,
    evidence_sha: str,
    accepted: bool = True,
) -> None:
    content_sha = hashlib.sha256(content.encode()).hexdigest()
    model = {
        "logical_id": logical_id,
        "active_slot": "a",
        "physical_ids": {"a": f"{logical_id}-a", "b": f"{logical_id}-b"},
        "last_verdict": "PASS_PUBLISH" if accepted else "INITIAL",
        "source_evidence_sha": evidence_sha,
        "source_watermark": evidence_sha,
    }
    if accepted:
        model["accepted_revision"] = {
            "slot": "a",
            "content_sha": content_sha,
            "source_evidence_sha": evidence_sha,
            "accepted_at": "2026-07-21T00:00:00Z",
        }
    root = home / "mental-models" / "egomotion4d"
    (root / "registry.json").write_text(
        json.dumps({"schema_version": 2, "models": {logical_id: model}}),
        encoding="utf-8",
    )


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_preflight_accepts_exact_per_model_evidence(tmp_path, monkeypatch, daily, capsys):
    logical_id = "egomotion4d-dynamic-actor"
    content = "# accepted\nD92-bound content"
    evidence_sha = "e" * 64
    _write_evidence(tmp_path, logical_id, evidence_sha)
    _write_registry(tmp_path, logical_id, content=content, evidence_sha=evidence_sha)
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Response({"content": content})
    )

    assert daily.mental_model_preflight(logical_id) == 0
    assert content in capsys.readouterr().out


def test_active_context_drops_content_hash_mismatch(tmp_path, monkeypatch, daily):
    logical_id = "egomotion4d-dynamic-actor"
    content = "tampered content"
    evidence_sha = "e" * 64
    _write_evidence(tmp_path, logical_id, evidence_sha)
    _write_registry(tmp_path, logical_id, content="accepted content", evidence_sha=evidence_sha)
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Response({"content": content})
    )

    assert daily._build_active_context("http://unused") == ""


def test_candidate_transaction_binds_evidence_and_transaction_id(daily):
    candidate_sha = "c" * 64
    evidence_sha = "e" * 64
    transaction = daily._make_candidate_transaction(
        "model", "b", candidate_sha, evidence_sha, "2026-07-21T00:00:00Z"
    )

    assert daily._validate_candidate_transaction(
        transaction,
        logical_id="model",
        slot="b",
        candidate_sha=candidate_sha,
        source_evidence_sha=evidence_sha,
    )
    assert not daily._validate_candidate_transaction(
        transaction,
        logical_id="model",
        slot="b",
        candidate_sha=candidate_sha,
        source_evidence_sha="f" * 64,
    )
    transaction["tx_id"] = "forged"
    assert not daily._validate_candidate_transaction(
        transaction,
        logical_id="model",
        slot="b",
        candidate_sha=candidate_sha,
        source_evidence_sha=evidence_sha,
    )


def test_candidate_completeness_requires_all_anchors_and_terminal_marker(daily):
    requirements = {
        "required_anchors": ["D19", "D43", "D76", "D79", "D80"],
        "required_terminal_marker": "END_GUARDRAILS",
        "require_source_facts": True,
        "required_prefix": "# Guardrail",
    }

    incomplete = "D19 D43 D76 D79"
    complete = "# Guardrail\nD19 D43 D76 D79 D80\nEND_GUARDRAILS"

    assert daily._candidate_completeness_errors(
        incomplete, requirements, source_fact_count=0
    ) == [
        "missing anchors: D80",
        "missing terminal marker: END_GUARDRAILS",
        "no source facts in reflect_response.based_on",
        "invalid content prefix: expected # Guardrail",
    ]
    assert daily._candidate_completeness_errors(
        complete, requirements, source_fact_count=5
    ) == []


def test_parse_adjudication_response_accepts_fenced_json_and_string_braces(daily):
    response = """<think>internal</think>
```json
{
  "verdict": "PASS_PUBLISH",
  "conflicts": [
    {
      "d_id": "D19",
      "candidate_claim": "literal braces {inside} nested object",
      "d_text": "current",
      "severity": "low"
    }
  ],
  "stale_items": [],
  "unanchored_items": [],
  "quality_score": 96,
  "notes": "literal braces {inside} a JSON string"
}
```
"""

    parsed, error = daily._parse_adjudication_response(response)

    assert error is None
    assert parsed["verdict"] == "PASS_PUBLISH"
    assert parsed["quality_score"] == 96
    assert parsed["conflicts"][0]["d_id"] == "D19"


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        ("not json", "no JSON object"),
        (
            json.dumps(
                {
                    "verdict": "PASS_PUBLISH",
                    "conflicts": [],
                    "stale_items": [],
                    "unanchored_items": [],
                    "quality_score": 101,
                    "notes": "invalid score",
                }
            ),
            "quality_score",
        ),
        (
            json.dumps(
                {
                    "verdict": "MAYBE",
                    "conflicts": [],
                    "stale_items": [],
                    "unanchored_items": [],
                    "quality_score": 90,
                    "notes": "invalid verdict",
                }
            ),
            "verdict",
        ),
    ],
)
def test_parse_adjudication_response_fails_closed_on_invalid_schema(
    daily, response, expected_error
):
    parsed, error = daily._parse_adjudication_response(response)

    assert parsed is None
    assert expected_error in error


def test_adjudication_prompt_respects_current_status_and_bounded_scope(daily):
    prompt = daily._adjudication_system_prompt()

    assert "higher D number does not supersede" in prompt
    assert "D anchor alone is sufficient" in prompt
    assert "outside the declared generation scope" in prompt
    assert "active slot is comparison context, not an authority" in prompt


def test_model_benchmark_contract_uses_target_spec(tmp_path, monkeypatch, daily):
    root = tmp_path / "mental-models" / "egomotion4d"
    (root / "specs").mkdir(parents=True)
    benchmark = root / "benchmark" / "questions-dynamic-actor.json"
    benchmark.parent.mkdir()
    benchmark.write_text('{"questions": []}', encoding="utf-8")
    (root / "specs" / "dynamic-actor.json").write_text(
        json.dumps(
            {
                "benchmark_file": "questions-dynamic-actor.json",
                "smoke_ids": ["DA01", "DA04"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)

    path, smoke_ids = daily._model_benchmark_contract("egomotion4d-dynamic-actor")

    assert path == benchmark
    assert smoke_ids == ["DA01", "DA04"]


def test_render_model_source_query_binds_inline_source_hash(
    tmp_path, monkeypatch, daily
):
    root = tmp_path / "mental-models" / "egomotion4d"
    (root / "specs").mkdir(parents=True)
    (root / "sources").mkdir()
    source = root / "sources" / "static.md"
    source.write_text("D64 authoritative source", encoding="utf-8")
    (root / "specs" / "static-surface.json").write_text(
        json.dumps(
            {
                "source_query": "Generate bounded output.",
                "inline_source_files": True,
                "source_files": ["sources/static.md"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)

    query = daily._render_model_source_query("egomotion4d-static-surface")

    assert query.startswith("Generate bounded output.")
    assert "D64 authoritative source" in query
    assert hashlib.sha256(source.read_bytes()).hexdigest() in query
    assert "BEGIN AUTHORITATIVE SOURCE" in query


def test_render_model_source_query_appends_simplified_chinese_contract(
    tmp_path, monkeypatch, daily
):
    root = tmp_path / "mental-models" / "egomotion4d"
    (root / "specs").mkdir(parents=True)
    (root / "specs" / "static-surface.json").write_text(
        json.dumps(
            {
                "source_query": "Generate bounded output.",
                "output_language": "zh-CN",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)

    query = daily._render_model_source_query("egomotion4d-static-surface")

    assert "必须使用简体中文输出" in query
    assert "技术标识符" in query


def test_candidate_completeness_rejects_non_chinese_when_required(daily):
    requirements = {
        "required_anchors": ["D64"],
        "output_language": "zh-CN",
    }

    assert daily._candidate_completeness_errors(
        "# Egomotion4D Static Surface\n- D64: English only.",
        requirements,
        source_fact_count=1,
    ) == ["output language must be Simplified Chinese"]
    assert daily._candidate_completeness_errors(
        (
            "# Egomotion4D 静态面约束\n"
            "- D64：必须使用简体中文完整说明当前结论、适用边界、禁止项和下一步条件。"
        ),
        requirements,
        source_fact_count=1,
    ) == []


def test_refresh_watermark_separates_accepted_and_rejected_candidates(daily):
    evidence_sha = "e" * 64

    initial = {}
    assert daily._model_needs_refresh(initial, evidence_sha, 0)

    rejected = {"last_candidate_evidence_sha": evidence_sha}
    assert not daily._model_needs_refresh(rejected, evidence_sha, 0)
    assert daily._model_needs_refresh(rejected, "f" * 64, 0)

    accepted = {
        "accepted_revision": {
            "slot": "a",
            "content_sha": "c" * 64,
            "source_evidence_sha": evidence_sha,
            "accepted_at": "2026-07-21T00:00:00Z",
        },
        "last_candidate_evidence_sha": "f" * 64,
    }
    assert not daily._model_needs_refresh(accepted, evidence_sha, 0)
    assert daily._model_needs_refresh(accepted, evidence_sha, 1)


def test_refresh_evidence_bundle_recomputes_changed_sources(tmp_path, monkeypatch, daily):
    logical_id = "egomotion4d-dynamic-actor"
    source = tmp_path / "current.md"
    source.write_text("v1", encoding="utf-8")
    root = tmp_path / "mental-models" / "egomotion4d"
    root.mkdir(parents=True)
    bundle_path = root / "evidence_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "per_model": {
                    logical_id: {
                        "d_ids": ["D92"],
                        "sources": {
                            "topic": {"path": str(source), "sha256": "stale"}
                        },
                        "evidence_sha256": "old",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)
    daily._refresh_evidence_bundle()
    refreshed = json.loads(bundle_path.read_text(encoding="utf-8"))
    expected_source_sha = hashlib.sha256(b"v1").hexdigest()
    assert refreshed["per_model"][logical_id]["sources"]["topic"]["sha256"] == expected_source_sha
    assert refreshed["per_model"][logical_id]["evidence_sha256"] != "old"


def test_refresh_evidence_bundle_rejects_unmarked_current_evidence(
    tmp_path, monkeypatch, daily
):
    logical_id = "egomotion4d-dynamic-actor"
    source = tmp_path / "dynamic_actor_current_evidence.md"
    source.write_text("D92 manually maintained copy", encoding="utf-8")
    root = tmp_path / "mental-models" / "egomotion4d"
    root.mkdir(parents=True)
    (root / "evidence_bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "per_model": {
                    logical_id: {
                        "d_ids": ["D92"],
                        "sources": {
                            "curated": {"path": str(source), "sha256": "stale"}
                        },
                        "evidence_sha256": "old",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)

    with pytest.raises(ValueError, match="derived build input contract"):
        daily._refresh_evidence_bundle()


def test_refresh_evidence_bundle_accepts_registered_current_evidence(
    tmp_path, monkeypatch, daily
):
    logical_id = "egomotion4d-dynamic-actor"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "dynamic_actor_current_evidence.md"
    source.write_text("D92 derived snapshot", encoding="utf-8")
    (source_dir / "derived-build-inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    source.name: {
                        "authority": "kg-current-evidence",
                        "replaceable": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "mental-models" / "egomotion4d"
    root.mkdir(parents=True)
    (root / "evidence_bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "per_model": {
                    logical_id: {
                        "d_ids": ["D92"],
                        "sources": {
                            "curated": {"path": str(source), "sha256": "stale"}
                        },
                        "evidence_sha256": "old",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)

    refreshed = daily._refresh_evidence_bundle()

    assert refreshed["per_model"][logical_id]["sources"]["curated"]["sha256"] == (
        hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_refresh_evidence_bundle_derives_decision_scope_from_spec(
    tmp_path, monkeypatch, daily
):
    logical_id = "egomotion4d-research-guardrails"
    root = tmp_path / "mental-models" / "egomotion4d"
    specs = root / "specs"
    specs.mkdir(parents=True)
    (specs / "research-guardrails.json").write_text(
        json.dumps({"decision_ids": ["D19", "D43", "D76", "D79", "D80"]}),
        encoding="utf-8",
    )
    (root / "evidence_bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "per_model": {
                    logical_id: {
                        "d_ids": ["D19", "D43"],
                        "sources": {},
                        "evidence_sha256": "old",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)

    refreshed = daily._refresh_evidence_bundle()

    assert refreshed["per_model"][logical_id]["d_ids"] == [
        "D19",
        "D43",
        "D76",
        "D79",
        "D80",
    ]


@pytest.mark.parametrize(
    ("stage_a", "adjudicate", "smoke", "expected"),
    [(1, None, None, 1), (2, None, 0, 0), (0, 2, 0, 2), (0, 0, 0, 0), (0, 3, 0, 3)],
)
def test_daily_branch_exit_codes_run_real_smoke(daily, stage_a, adjudicate, smoke, expected):
    calls = []

    def run_stage_a():
        calls.append("stage_a")
        return stage_a

    def run_adjudicate():
        calls.append("adjudicate")
        return adjudicate

    def run_smoke():
        calls.append("smoke")
        return smoke

    assert daily._run_mental_model_daily(run_stage_a, run_adjudicate, run_smoke) == expected
    if stage_a != 1:
        assert "smoke" in calls


def test_all_model_smoke_checks_each_accepted_revision(tmp_path, monkeypatch, daily):
    root = tmp_path / "mental-models" / "egomotion4d"
    root.mkdir(parents=True)
    (root / "registry.json").write_text(
        json.dumps(
            {
                "models": {
                    "model-a": {"accepted_revision": {"content_sha": "a"}},
                    "model-b": {"accepted_revision": {"content_sha": "b"}},
                    "model-c": {},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daily, "HERMES_HOME", tmp_path)
    calls = []

    def fake_smoke(_api_url, *, logical_id, **_kwargs):
        calls.append(logical_id)
        return 0

    monkeypatch.setattr(daily, "_run_smoke_regression", fake_smoke)

    assert daily._run_all_model_smoke("http://unused") == 0
    assert calls == ["model-a", "model-b"]


def test_render_mental_model_index_uses_registry_truth(daily):
    registry = {
        "models": {
            "model-ready": {
                "active_slot": "b",
                "max_tokens": 2048,
                "last_verdict": "PASS_PUBLISH",
                "verdict_detail": "quality=95",
                "accepted_revision": {
                    "slot": "b",
                    "content_sha": "a" * 64,
                    "source_evidence_sha": "e" * 64,
                    "accepted_at": "2026-07-21T00:00:00Z",
                },
            },
            "model-blocked": {
                "active_slot": "a",
                "max_tokens": 1024,
                "last_verdict": "REJECT",
            },
        }
    }

    rendered = daily._render_mental_model_index(registry, "2026-07-21")

    assert "model-ready" in rendered
    assert "PASS_PUBLISH" in rendered
    assert "aaaaaaaaaaaa" in rendered
    assert "current/model-ready.md" in rendered
    assert "model-blocked" in rendered
    assert "REJECT" in rendered
    assert "current/model-blocked.md" not in rendered
    assert "Knowledge Ownership" in rendered
    assert "authoritative project truth" in rendered
    assert "reproducible derived build inputs" in rendered
    assert "must not be maintained as a third truth store" in rendered


def test_research_digest_validation_rejects_noncanonical_action_and_line_counts(daily):
    good = """# 2026-07-20 Research Digest
## 主线
- bounded result
## 已验证结果
| 对象 | 证据/指标 | 结论 |
|---|---|---|
| packet | sha | PASS |
## 决策与转折
- no claim
## 来源
- current.md
"""
    bad = good + "\n## 下一步\n- retry\n- 128 行实现\n"

    assert daily._research_digest_errors(good) == []
    errors = daily._research_digest_errors(bad)
    assert "forbidden section: 下一步" in errors
    assert "implementation line-count narration is forbidden" in errors


def test_pitfall_writer_records_lifecycle_timestamps():
    module = _load_module("pitfall_writer_timestamps_test", PITFALL_WRITER)
    index = module.PitfallIndex()

    p_id = index.add_entry(
        title="bounded algorithm pitfall",
        status=module.PitfallStatus.CANDIDATE,
        is_algorithm_level=True,
        trigger="when",
        root_cause="why",
        lesson="lesson",
        tags=["egomotion4d"],
        source="test",
        detail_locator="",
    )
    created = index._get(p_id).created_at
    index.adjudicate_candidates(
        {p_id: (module.PitfallStatus.REJECTED, None, None)}
    )

    assert created is not None
    assert index._get(p_id).updated_at is not None


def test_pitfall_writer_migrates_retired_source_without_fabricating_timestamps(tmp_path):
    module = _load_module("pitfall_writer_v3_migration_test", PITFALL_WRITER)
    path = tmp_path / "pitfall_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_file": (
                    "/home/wyr/wiki/auto-maintenance/project/egomotion4d/pitfalls.md"
                ),
                "entries": [
                    {
                        "p_id": "P1",
                        "title": "legacy",
                        "status": "current",
                        "is_algorithm_level": True,
                        "trigger": "when",
                        "root_cause": "why",
                        "lesson": "lesson",
                        "tags": ["egomotion4d"],
                        "date": "2026-07-20",
                        "source": "legacy",
                        "detail_locator": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    index = module.PitfallIndex.load(str(path))

    assert index.schema_version == 3
    assert index.source_file.endswith("/mental-models/pitfall-catalog.md")
    assert index.entries[0].created_at is None
    assert index.entries[0].updated_at is None


def test_run_ab_is_import_safe_and_scores_numeric_verdict():
    module = _load_module("run_ab_test", RUN_AB)
    question = {"required_terms": ["D76", "非米制"]}
    assert module.score_response("D76 说明统一 gauge 是非米制。", question) == 1.0
    summary = module.summarize_results(
        [
            {"A": {"score": 0.0, "error": None}, "B": {"score": 1.0, "error": None}},
            {"A": {"score": 1.0, "error": None}, "B": {"score": 1.0, "error": None}},
        ]
    )
    assert summary["paired_complete"] == 2
    assert summary["mean_score_a"] == 0.5
    assert summary["mean_score_b"] == 1.0
    assert summary["verdict"] == "PASS_B_BETTER"


def test_smoke_terms_normalize_dash_and_support_predeclared_alternatives(daily):
    answer = "D92: frames 0–62 and 7–34; 这不证明全序列稳定 identity。"

    assert daily._matches_expected_term(answer, "0-62")
    assert daily._matches_expected_term(answer, "7-34")
    assert daily._matches_expected_term(answer, ["no", "不证明", "does not prove"])
    assert daily._matches_expected_term("promotion_safe_geometry", "promotion-safe")

    module = _load_module("run_ab_term_normalization_test", RUN_AB)
    question = {
        "key_d_refs": ["D92"],
        "expected_pitfall_triggers": [
            "0-62",
            "7-34",
            ["no", "不证明", "does not prove"],
            "full-sequence",
        ],
    }
    assert module.score_response(answer, question) == 0.8


def test_gate_question_exposes_required_decision_anchors(daily):
    question = {"question": "Which frame?", "key_d_refs": ["D91", "D92"]}

    assert daily._format_gate_question(question).endswith(
        "Required decision anchors: D91, D92."
    )

    module = _load_module("run_ab_gate_prompt_test", RUN_AB)
    assert module.format_gate_question(question).endswith(
        "Required decision anchors: D91, D92."
    )


def test_run_ab_can_isolate_one_accepted_model(tmp_path, monkeypatch):
    module = _load_module("run_ab_isolation_test", RUN_AB)
    registry = {
        "models": {
            "model-a": {"accepted_revision": {"content_sha": "a"}},
            "model-b": {"accepted_revision": {"content_sha": "b"}},
        }
    }
    root = tmp_path / "mental-models" / "egomotion4d"
    root.mkdir(parents=True)
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    preflight = tmp_path / "preflight.py"
    preflight.write_text("# fixture", encoding="utf-8")
    monkeypatch.setattr(module, "HERMES_HOME", tmp_path)
    monkeypatch.setattr(module, "PREFLIGHT_SCRIPT", preflight)

    class Result:
        returncode = 0
        stdout = "MODEL_A_CONTEXT"

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        assert command[-1] == "model-a"
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.build_active_context(logical_ids=["model-a"]) == "MODEL_A_CONTEXT"
    assert len(calls) == 1


def test_pitfall_validator_detects_false_locator_and_missing_provenance(tmp_path):
    module = _load_module("pitfall_writer_test", PITFALL_WRITER)
    details = tmp_path / "pitfalls.md"
    details.write_text("# Pitfalls\n\n## P1: Real title\n", encoding="utf-8")
    index_path = tmp_path / "pitfall_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "next_p_id": 3,
                "entries": [
                    {
                        "p_id": "P2",
                        "title": "Missing detail",
                        "status": "current",
                        "is_algorithm_level": True,
                        "trigger": "x",
                        "root_cause": "y",
                        "lesson": "z",
                        "tags": ["egomotion4d"],
                        "date": "2026-07-21",
                        "source": "memory",
                        "detail_locator": "pitfalls.md#P2",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    index = module.PitfallIndex.load(str(index_path))
    errors = index.validate(details_path=details, require_provenance=True)
    assert any("locator" in error.lower() for error in errors)
    assert any("provenance" in error.lower() for error in errors)


def test_pitfall_reconcile_uses_canonical_details_and_quarantines_false_current(tmp_path):
    module = _load_module("pitfall_writer_reconcile_test", PITFALL_WRITER)
    details = tmp_path / "pitfalls.md"
    details.write_text(
        "## P2: Canonical title\n\n"
        "**坑**: trigger\n**实际**: cause\n**教训**: lesson\n"
        "**标签**: #评估\n**日期**: 2026-07-21\n**来源**: report\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "pitfall_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "next_p_id": 4,
                "entries": [
                    {
                        "p_id": "P2", "title": "Wrong title", "status": "rejected_non_algorithmic",
                        "is_algorithm_level": False, "trigger": "x", "root_cause": "y",
                        "lesson": "z", "tags": ["x"], "date": "2026-07-20",
                        "source": "memory", "detail_locator": "pitfalls.md#P2",
                    },
                    {
                        "p_id": "P3", "title": "False current", "status": "current",
                        "is_algorithm_level": True, "trigger": "x", "root_cause": "y",
                        "lesson": "z", "tags": ["x"], "date": "2026-07-20",
                        "source": "memory", "detail_locator": "pitfalls.md#P3",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    index = module.PitfallIndex.load(str(index_path))
    index.reconcile_details(details)
    p2 = index.find_by_id("P2")
    p3 = index.find_by_id("P3")
    assert p2 and p2.title == "Canonical title" and p2.status == "current"
    assert p2.source_memory_id == "pitfalls.md:P2" and p2.source_content_hash
    assert p3 and p3.status == "candidate" and p3.detail_locator == ""


def test_legacy_pitfall_append_writer_is_retired():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pitfalls_file.write_text" not in source
    assert "with open(catalog_file" not in source


def test_pitfall_writer_exports_exact_status_counts(tmp_path):
    module = _load_module("pitfall_writer_catalog_test", PITFALL_WRITER)
    index_path = tmp_path / "pitfall_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "entries": [
                    {
                        "p_id": "P1", "title": "Current", "status": "current",
                        "is_algorithm_level": True, "trigger": "x", "root_cause": "y",
                        "lesson": "z", "tags": ["x"], "date": "2026-07-21",
                        "source": "test", "detail_locator": "pitfalls.md#P1",
                    },
                    {
                        "p_id": "P2", "title": "Candidate", "status": "candidate",
                        "is_algorithm_level": True, "trigger": "x", "root_cause": "y",
                        "lesson": "", "tags": ["x"], "date": "2026-07-21",
                        "source": "test", "detail_locator": "",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    index = module.PitfallIndex.load(str(index_path))
    output = tmp_path / "catalog.md"
    index.export_catalog(output, generated_date="2026-07-21")
    text = output.read_text(encoding="utf-8")
    assert "Current: 1" in text
    assert "Candidate: 1" in text
    assert index.status_counts["current"] == 1


def test_consumption_feedback_supports_four_week_metrics(tmp_path, monkeypatch, daily):
    monkeypatch.setattr(daily, "LOG_DIR", tmp_path)
    daily._log_role_trigger(
        "reviewer",
        "mental_model_consumed",
        logical_id="egomotion4d-dynamic-actor",
        revision="a" * 64,
        context_tokens=120,
    )
    daily._record_mental_model_feedback(
        "avoided_repeat",
        logical_id="egomotion4d-dynamic-actor",
        task_id="t_test",
        note="closed route found before experiment",
        role="reviewer",
    )

    metrics = daily._mental_model_metrics(days=28)
    assert metrics["total_consumptions"] == 1
    assert metrics["consumption_by_role"] == {"reviewer": 1}
    assert metrics["context_tokens"] == 120
    assert metrics["feedback_by_outcome"] == {"avoided_repeat": 1}
    assert metrics["avoided_repeat_rate"] == 1.0


@pytest.mark.parametrize(("child_rc", "cron_rc"), [(0, 0), (1, 1), (2, 0), (3, 0)])
def test_cron_wrapper_maps_research_outcomes_to_operational_success(
    monkeypatch, child_rc, cron_rc
):
    wrapper = _load_module(f"daily_wrapper_{child_rc}", WRAPPER)
    monkeypatch.setattr(wrapper.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": child_rc})(),
    )
    assert wrapper.run() == cron_rc
