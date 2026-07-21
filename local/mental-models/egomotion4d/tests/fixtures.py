"""Deterministic fixtures for mental_model daily pipeline five branches.

Each fixture produces a known registry state and mock API response so
that the pipeline's five exit paths can be tested deterministically.

Branch paths:
1. busy: another instance holds the lock
2. no-change: Stage A finds no stale models (return 2)
3. reject: adjudicate returns REJECT (return 2)
4. publish: adjudicate returns PASS_PUBLISH (return 0)
5. escalate: adjudicate returns ESCALATE_D_REVIEW (return 3)
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _make_registry(*, last_verdict: str, active_content_sha: str | None) -> dict:
    """Build a minimal registry with one model at given verdict."""
    return {
        "schema_version": 1,
        "project": "egomotion4d",
        "last_updated": "2026-07-21T00:00:00Z",
        "models": {
            "egomotion4d-test-fixture": {
                "logical_id": "egomotion4d-test-fixture",
                "active_slot": "a",
                "physical_ids": {"a": "egomotion4d-test-fixture-a", "b": "egomotion4d-test-fixture-b"},
                "content_sha": active_content_sha,
                "active_content_sha": active_content_sha,
                "source_watermark": "2026-07-21T00:00:00+00:00",
                "last_refresh": "2026-07-21T00:00:00+00:00",
                "last_verdict": last_verdict,
                "verdict_detail": "Fixture for testing",
                "tags": ["egomotion4d"],
                "max_tokens": 4096,
                "source_evidence_sha": "test_evidence_sha",
                "relevant_current_decisions": ["D19", "D20"],
            }
        },
        "pitfalls": {
            "catalog_file": "wiki/auto-maintenance/project/egomotion4d/mental-models/pitfall-catalog.md",
            "index_file": "~/.hermes/mental-models/egomotion4d/pitfall_index.json",
            "last_updated": "2026-07-21T00:00:00Z",
            "total_canonical": 20,
            "total_superseded": 20,
            "total_rejected": 13,
        },
    }


def _make_content_body(verdict: str) -> str:
    """Return plausible content body for a given verdict state."""
    return (
        f"# Test Fixture Content ({verdict})\n\n"
        f"This is a deterministic fixture for {verdict} branch.\n"
        f"Generated at: 2026-07-21T00:00:00Z\n"
    )


def setup_fixture_registry(
    tmp_path: Path,
    *,
    verdict: str,
    exit_code: int,
    active_content_sha: str | None = "test_content_sha",
) -> Path:
    """Write a fixture registry to tmp_path and return its path."""
    reg = _make_registry(
        last_verdict=verdict,
        active_content_sha=active_content_sha,
    )
    reg_path = tmp_path / "registry.json"
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)
    return reg_path


def setup_fixture_questions(tmp_path: Path) -> Path:
    """Write fixture questions file (subset of real benchmark)."""
    q = {
        "version": 1,
        "created": "2026-07-21",
        "description": "Fixture questions for testing",
        "questions": [
            {
                "id": "Q01",
                "question": "DLT anchor-guided depth refine 是否可以用于 dense-depth refine？",
                "ground_truth": "不能。D25",
                "key_d_refs": ["D25"],
                "expected_pitfall_triggers": ["DLT", "anchor"],
                "pass_condition": "回答必须说明 DLT 只能 diagnostic",
            },
        ],
    }
    q_path = tmp_path / "questions.json"
    with open(q_path, "w") as f:
        json.dump(q, f, indent=2)
    return q_path


# ── Fixture configurations ───────────────────────────────────────────────

FIXTURES: dict[str, dict[str, Any]] = {
    "busy": {
        "description": "Another instance holds the lock (simulated by lock file)",
        "registry_verdict": "PASS_PUBLISH",
        "expected_exit_code": 1,
        "setup_hook": "create_lock_file",
    },
    "no-change": {
        "description": "Stage A finds no stale models (maintain returns 2)",
        "registry_verdict": "PASS_PUBLISH",
        "expected_exit_code": 0,
        "setup_hook": "maintain_returns_2",
    },
    "reject": {
        "description": "Adjudicate returns REJECT (exit 2)",
        "registry_verdict": "REJECT",
        "expected_exit_code": 2,
    },
    "publish": {
        "description": "Adjudicate returns PASS_PUBLISH (exit 0)",
        "registry_verdict": "PASS_PUBLISH",
        "active_content_sha": "test_content_sha",
        "expected_exit_code": 0,
    },
    "escalate": {
        "description": "Adjudicate returns ESCALATE_D_REVIEW (exit 3)",
        "registry_verdict": "ESCALATE_D_REVIEW",
        "active_content_sha": "test_content_sha",
        "expected_exit_code": 3,
    },
}
