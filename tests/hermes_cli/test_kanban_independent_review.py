"""Independent non-author review gate for implementer deliveries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_independent_review as review


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_SESSION_ID", "author-session")
    monkeypatch.setenv("HERMES_KANBAN_CONTROL_PROMPT_SHA256", "control-sha")
    kb.init_db()
    return home


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_payload(task, artifact: Path, **updates):
    payload = {
        "schema": "independent_review.v1",
        "task_id": task.id,
        "generation": task.generation,
        "base_sha": task.base_commit,
        "delivery_sha": task.base_commit,
        "diff_tree_sha": review.EMPTY_DIFF_TREE_SHA256,
        "reviewer_session_id": "reviewer-session",
        "author_session_ids": ["author-session"],
        "control_prompt_sha": "control-sha",
        "commands": ["pytest -q"],
        "artifact_hashes": {str(artifact.resolve()): _sha(artifact)},
        "verdict": "PASS",
    }
    payload.update(updates)
    return payload


def _create_implementer_task(conn, tmp_path: Path):
    return kb.create_task(
        conn,
        title="deliver evidence",
        assignee="implementer",
        workspace_kind="worktree",
        workspace_path=str(tmp_path / "wt"),
        branch_name="implementer/task/g1",
        base_commit="a" * 40,
        target_branch="main",
    )


def _metadata(tmp_path: Path, review_path: Path, artifact: Path, **updates):
    metadata = {
        "task_type": "formal-artifact",
        "independent_review_required": True,
        "independent_review_artifact": str(review_path),
        "artifacts": [str(artifact)],
    }
    metadata.update(updates)
    return metadata


def test_valid_review_is_bound_and_persisted_on_completion(kanban_home, tmp_path):
    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"score": 1}\n', encoding="utf-8")
    with kb.connect() as conn:
        task_id = _create_implementer_task(conn, tmp_path)
        task = kb.get_task(conn, task_id)
        review_path = tmp_path / "review.json"
        review_path.write_text(json.dumps(_review_payload(task, artifact)), encoding="utf-8")

        metadata = _metadata(tmp_path, review_path, artifact)
        assert kb.complete_task(conn, task_id, summary="done", metadata=metadata)

        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        stored = json.loads(run["metadata"])
        assert stored["independent_review"]["verdict"] == "PASS"
        assert stored["independent_review_sha256"] == _sha(review_path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"reviewer_session_id": "author-session"}, "non-author"),
        ({"verdict": "FAIL"}, "PASS"),
        ({"generation": 99}, "generation"),
        ({"delivery_sha": "b" * 40}, "delivery"),
        ({"control_prompt_sha": "stale"}, "control prompt"),
        ({"artifact_hashes": {}}, "artifact"),
    ],
)
def test_stale_or_self_authored_review_blocks_completion(
    kanban_home, tmp_path, mutation, match,
):
    artifact = tmp_path / "metrics.json"
    artifact.write_text("before\n", encoding="utf-8")
    with kb.connect() as conn:
        task_id = _create_implementer_task(conn, tmp_path)
        task = kb.get_task(conn, task_id)
        review_path = tmp_path / "review.json"
        review_path.write_text(
            json.dumps(_review_payload(task, artifact, **mutation)), encoding="utf-8"
        )

        with pytest.raises(review.IndependentReviewError, match=match):
            kb.complete_task(
                conn,
                task_id,
                metadata=_metadata(tmp_path, review_path, artifact),
            )
        assert kb.get_task(conn, task_id).status == "ready"
        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert event["kind"] == "completion_blocked_independent_review"
        assert match.replace("\\", "") in json.loads(event["payload"])["reason"]


def test_artifact_change_after_review_blocks_completion(kanban_home, tmp_path):
    artifact = tmp_path / "metrics.json"
    artifact.write_text("before\n", encoding="utf-8")
    with kb.connect() as conn:
        task_id = _create_implementer_task(conn, tmp_path)
        task = kb.get_task(conn, task_id)
        review_path = tmp_path / "review.json"
        review_path.write_text(json.dumps(_review_payload(task, artifact)), encoding="utf-8")
        artifact.write_text("after\n", encoding="utf-8")

        with pytest.raises(review.IndependentReviewError, match="artifact"):
            kb.complete_task(
                conn,
                task_id,
                metadata=_metadata(tmp_path, review_path, artifact),
            )


def test_missing_review_blocks_implementer_but_not_designer(kanban_home, tmp_path):
    with kb.connect() as conn:
        implementer_id = _create_implementer_task(conn, tmp_path)
        with pytest.raises(review.IndependentReviewError, match="required"):
            kb.complete_task(conn, implementer_id, metadata={"task_type": "code"})

        designer_id = kb.create_task(
            conn, title="direct design implementation", assignee="designer"
        )
        assert kb.complete_task(conn, designer_id, metadata={"task_type": "code"})


@pytest.mark.parametrize("task_type", ["plan", "research", "readonly"])
def test_explicit_non_delivery_opt_out_is_allowed(kanban_home, tmp_path, task_type):
    with kb.connect() as conn:
        task_id = _create_implementer_task(conn, tmp_path)
        assert kb.complete_task(
            conn,
            task_id,
            metadata={
                "task_type": task_type,
                "independent_review_required": False,
            },
        )


def test_code_delivery_cannot_opt_out(kanban_home, tmp_path):
    with kb.connect() as conn:
        task_id = _create_implementer_task(conn, tmp_path)
        with pytest.raises(review.IndependentReviewError, match="cannot opt out"):
            kb.complete_task(
                conn,
                task_id,
                metadata={
                    "task_type": "code",
                    "independent_review_required": False,
                },
            )
