from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "hindsight_daily_noagent.py"


def _load_module(tmp_path: Path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HINDSIGHT_DAILY_HERMES_HOME", str(hermes_home))
    spec = importlib.util.spec_from_file_location(
        f"hindsight_daily_review_export_{tmp_path.name}", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, hermes_home


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def _review_fixture(tmp_path: Path, monkeypatch):
    module, hermes_home = _load_module(tmp_path, monkeypatch)
    model_root = hermes_home / "mental-models" / "egomotion4d"
    decisions = model_root / "10-current-decisions.md"
    derived = model_root / "derived.md"
    decision_sha = _write(decisions, "| D1 | current | frozen decision |\n")
    derived_sha = _write(derived, "derived evidence\n")
    content = "accepted model\n" + ("complete evidence line\n" * 8)
    content_sha = hashlib.sha256(content.encode()).hexdigest()
    evidence_sha = "e" * 64
    evidence_bundle = {
        "per_model": {
            "model-a": {
                "evidence_sha256": evidence_sha,
                "d_ids": ["D1"],
                "sources": {
                    "10-current-decisions.md": {
                        "path": str(decisions),
                        "sha256": decision_sha,
                    },
                    "derived.md": {
                        "path": str(derived),
                        "sha256": derived_sha,
                    },
                },
            }
        }
    }
    _write(
        model_root / "evidence_bundle.json",
        json.dumps(evidence_bundle, sort_keys=True),
    )
    registry = {
        "models": {
            "model-a": {
                "active_slot": "a",
                "physical_ids": {"a": "physical-a"},
                "accepted_revision": {
                    "slot": "a",
                    "content_sha": content_sha,
                    "source_evidence_sha": evidence_sha,
                    "accepted_at": "2026-08-01T00:00:00Z",
                },
            }
        }
    }
    manifest = tmp_path / "review_exports.json"
    _write(
        manifest,
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "model-a": {"enabled": True, "extra_decision_ids": []}
                },
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "_model_generation_spec",
        lambda logical_id: {
            "source_files": ["derived.md"],
            "required_anchors": ["D1"],
            "decision_ids": [],
        },
    )
    monkeypatch.setattr(module, "_model_generation_requirements", lambda _: {})
    monkeypatch.setattr(module, "_candidate_completeness_errors", lambda *a, **k: [])
    return module, registry, manifest, content, decisions


def test_review_export_is_identity_bound_and_history_is_immutable(
    tmp_path, monkeypatch,
):
    module, registry, manifest, content, _ = _review_fixture(tmp_path, monkeypatch)
    export_root = tmp_path / "exports"

    first = module._publish_review_exports(
        "http://unused",
        registry=registry,
        manifest_path=manifest,
        export_root=export_root,
        fetch_model=lambda _: {"content": content},
        generated_at="2026-08-01T01:00:00Z",
    )
    second = module._publish_review_exports(
        "http://unused",
        registry=registry,
        manifest_path=manifest,
        export_root=export_root,
        fetch_model=lambda _: {"content": content},
        generated_at="2026-08-02T01:00:00Z",
    )

    assert first["aggregate"] == "PASS_ALL"
    assert second["aggregate"] == "PASS_ALL"
    history = Path(first["models"]["model-a"]["history"])
    current = Path(first["models"]["model-a"]["current"])
    assert history.read_bytes() == current.read_bytes()
    rendered = history.read_text(encoding="utf-8")
    assert "<!-- BEGIN_ACCEPTED_MODEL_BYTES -->" in rendered
    assert content in rendered
    assert "| D1 | current | frozen decision |" in rendered
    assert "first_generated_at: 2026-08-01T01:00:00Z" in rendered
    assert "2026-08-02T01:00:00Z" not in rendered


def test_review_export_blocks_when_evidence_source_hash_changes(
    tmp_path, monkeypatch,
):
    module, registry, manifest, content, decisions = _review_fixture(
        tmp_path, monkeypatch
    )
    decisions.write_text("tampered\n", encoding="utf-8")

    result = module._publish_review_exports(
        "http://unused",
        registry=registry,
        manifest_path=manifest,
        export_root=tmp_path / "exports",
        fetch_model=lambda _: {"content": content},
        generated_at="2026-08-01T01:00:00Z",
    )

    assert result["aggregate"] == "BLOCK_ALL"
    assert result["models"]["model-a"]["status"] == "BLOCKED"
    assert "evidence source SHA mismatch" in result["models"]["model-a"]["errors"][0]
    assert not list((tmp_path / "exports" / "review" / "history").glob("*.md"))


def test_review_export_allows_evidence_only_model_without_decision_source(
    tmp_path, monkeypatch,
):
    module, registry, manifest, content, _ = _review_fixture(tmp_path, monkeypatch)
    model_root = module.HERMES_HOME / "mental-models" / "egomotion4d"
    evidence_bundle_path = model_root / "evidence_bundle.json"
    evidence_bundle = json.loads(evidence_bundle_path.read_text(encoding="utf-8"))
    evidence_entry = evidence_bundle["per_model"]["model-a"]
    evidence_entry["d_ids"] = []
    evidence_entry["sources"].pop("10-current-decisions.md")
    evidence_bundle_path.write_text(
        json.dumps(evidence_bundle, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(
        module,
        "_model_generation_spec",
        lambda logical_id: {
            "source_files": ["derived.md"],
            "required_anchors": ["evidence-only section"],
            "decision_ids": [],
        },
    )

    result = module._publish_review_exports(
        "http://unused",
        registry=registry,
        manifest_path=manifest,
        export_root=tmp_path / "exports",
        fetch_model=lambda _: {"content": content},
        generated_at="2026-08-01T01:00:00Z",
    )

    assert result["aggregate"] == "PASS_ALL"
    assert result["models"]["model-a"]["status"] == "PASS"


def test_daily_no_refresh_still_runs_smoke_then_review_export(
    tmp_path, monkeypatch,
):
    module, _ = _load_module(tmp_path, monkeypatch)
    calls: list[str] = []

    result = module._run_mental_model_daily(
        lambda: 2,
        lambda: calls.append("adjudicate") or 0,
        lambda: calls.append("smoke") or 0,
        lambda: calls.append("review") or 0,
    )

    assert result == 0
    assert calls == ["smoke", "review"]
