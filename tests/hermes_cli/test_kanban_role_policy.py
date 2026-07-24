"""Board-neutral role lifecycle policy enforcement."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from plugins.kanban import base_listener


SCHEMA_VERSION = "kanban-role-policy.v1"


def _policy_module():
    return importlib.import_module("hermes_cli.kanban_role_policy")


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_policy(
    path: Path,
    *,
    active=("builder", "planner"),
    historical=("auditor",),
    schema_version=SCHEMA_VERSION,
):
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "active_roles": list(active),
                "historical_roles": list(historical),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _activate(board_home, **kwargs):
    path = _write_policy(board_home / "role-policy.json", **kwargs)
    return path, kb.activate_board_role_policy("default", path)


def test_loader_is_board_neutral_and_cache_invalidates_on_sha(board_home):
    policy_module = _policy_module()
    path = _write_policy(
        board_home / "policy.json", active=("alpha",), historical=("retired",),
    )
    first = policy_module.load_role_policy(path)
    original_stat = path.stat()

    _write_policy(path, active=("bravo",), historical=("retired",))
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = policy_module.load_role_policy(path)

    assert first.active_roles == frozenset({"alpha"})
    assert second.active_roles == frozenset({"bravo"})
    assert first.content_sha != second.content_sha
    assert second.path == path.resolve()


@pytest.mark.parametrize(
    "payload,error",
    [
        ("{not-json", "corrupt"),
        (
            json.dumps(
                {
                    "schema_version": "kanban-role-policy.v999",
                    "active_roles": ["builder"],
                    "historical_roles": ["auditor"],
                }
            ),
            "unsupported",
        ),
        (
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "active_roles": ["builder", "auditor"],
                    "historical_roles": ["auditor"],
                }
            ),
            "both active and historical",
        ),
    ],
)
def test_loader_rejects_corrupt_unsupported_and_overlapping_roles(
    board_home, payload, error,
):
    policy_module = _policy_module()
    path = board_home / "bad-policy.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(policy_module.RolePolicyConfigurationError, match=error):
        policy_module.load_role_policy(path)


def test_activation_atomically_persists_canonical_policy_identity(board_home):
    path, activated = _activate(board_home)
    metadata_path = kb.board_metadata_path("default")
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    assert activated["role_policy_enforced"] is True
    assert persisted["role_policy_path"] == str(path.resolve())
    assert persisted["role_policy_schema_version"] == SCHEMA_VERSION
    assert persisted["role_policy_content_sha"] == expected_sha
    assert not list(metadata_path.parent.glob(f".{metadata_path.name}.*.tmp"))


def test_legacy_board_remains_compatible_before_activation(board_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy", assignee="auditor")
        kb.add_comment(conn, task_id, "user", "legacy mutation remains valid")
        claimed = kb.claim_task(conn, task_id, claimer="legacy-pane")

    assert claimed is not None
    assert claimed.assignee == "auditor"


def test_active_policy_keeps_historical_rows_readable_but_immutable(board_home):
    policy_module = _policy_module()
    with kb.connect() as conn:
        historical_id = kb.create_task(
            conn, title="historical evidence", assignee="auditor",
        )
        active_id = kb.create_task(conn, title="active work", assignee="builder")
    _activate(board_home)

    with kb.connect() as conn:
        assert kb.get_task(conn, historical_id).title == "historical evidence"
        assert historical_id in {task.id for task in kb.list_tasks(conn)}
        kb.add_comment(conn, active_id, "user", "active mutation")
        with pytest.raises(policy_module.RolePolicyDenied, match="auditor.*historical"):
            kb.add_comment(conn, historical_id, "user", "must fail")
        with pytest.raises(policy_module.RolePolicyDenied):
            kb.claim_task(conn, historical_id, claimer="auditor-pane")
        with pytest.raises(policy_module.RolePolicyDenied):
            kb.create_task(conn, title="new historical", assignee="auditor")


@pytest.mark.parametrize(
    "operation",
    [
        "assign",
        "edit",
        "complete",
        "block",
        "unblock",
        "return",
        "archive",
        "link",
        "unlink",
    ],
)
def test_central_db_mutation_entries_reject_historical_tasks(
    board_home, operation,
):
    policy_module = _policy_module()
    with kb.connect() as conn:
        historical_id = kb.create_task(conn, title="history", assignee="auditor")
        active_id = kb.create_task(conn, title="active", assignee="builder")
    _activate(board_home)

    def invoke(conn):
        calls = {
            "assign": lambda: kb.assign_task(conn, historical_id, "builder"),
            "edit": lambda: kb.edit_completed_task_result(
                conn, historical_id, result="changed",
            ),
            "complete": lambda: kb.complete_task(conn, historical_id),
            "block": lambda: kb.block_task(conn, historical_id),
            "unblock": lambda: kb.unblock_task(conn, historical_id),
            "return": lambda: kb.return_task_for_rework(
                conn, historical_id, actor="planner", reason="redo",
            ),
            "archive": lambda: kb.archive_task(conn, historical_id),
            "link": lambda: kb.link_tasks(conn, historical_id, active_id),
            "unlink": lambda: kb.unlink_tasks(conn, historical_id, active_id),
        }
        return calls[operation]()

    with kb.connect() as conn:
        with pytest.raises(policy_module.RolePolicyDenied):
            invoke(conn)


def test_historical_actor_cannot_mutate_active_task(board_home, monkeypatch):
    policy_module = _policy_module()
    with kb.connect() as conn:
        active_id = kb.create_task(conn, title="active", assignee="builder")
    _activate(board_home)
    monkeypatch.setenv("HERMES_PROFILE", "auditor")

    with kb.connect() as conn:
        with pytest.raises(policy_module.RolePolicyDenied, match="actor.*historical"):
            kb.add_comment(conn, active_id, "auditor", "forbidden")


@pytest.mark.parametrize("failure", ["missing", "corrupt", "sha-mismatch"])
def test_enforced_policy_configuration_failure_blocks_claim_and_mutation(
    board_home, failure,
):
    policy_module = _policy_module()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="active", assignee="builder")
    path, _ = _activate(board_home)
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_text("{broken", encoding="utf-8")
    else:
        _write_policy(path, active=("planner",), historical=("auditor",))

    with kb.connect() as conn:
        with pytest.raises(policy_module.RolePolicyConfigurationError):
            kb.claim_task(conn, task_id, claimer="builder-pane")
        with pytest.raises(policy_module.RolePolicyConfigurationError):
            kb.add_comment(conn, task_id, "user", "blocked")
        assert kb.get_task(conn, task_id).status == "ready"


def test_control_ack_for_historical_role_is_rejected(board_home):
    policy_module = _policy_module()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="history", assignee="auditor")
        assert kb.claim_task(conn, task_id, claimer="auditor-pane")
        returned = kb.return_task_for_rework(
            conn, task_id, actor="planner", reason="supersede",
        )
        control_id = returned.control_ids[0]
    _activate(board_home)

    with kb.connect() as conn:
        with pytest.raises(policy_module.RolePolicyDenied):
            kb.ack_control_message(conn, control_id, receiver="auditor-pane")


def test_cli_activation_and_historical_read_only_surface(board_home):
    with kb.connect() as conn:
        historical_id = kb.create_task(
            conn, title="historical cli", assignee="auditor",
        )
    policy_path = _write_policy(board_home / "cli-policy.json")

    activated = json.loads(
        kc.run_slash(f"role-policy activate {policy_path} --json")
    )
    shown = json.loads(kc.run_slash(f"show {historical_id} --json"))
    listed = kc.run_slash("list")
    denied_comment = kc.run_slash(f"comment {historical_id} forbidden")
    denied_create = kc.run_slash("create forbidden --assignee auditor")

    assert activated["role_policy_enforced"] is True
    assert shown["task"]["id"] == historical_id
    assert "historical cli" in listed
    assert "historical" in denied_comment.lower()
    assert "historical" in denied_create.lower()


def test_interactive_candidate_skips_historical_and_claim_guard_is_shared(
    board_home,
):
    policy_module = _policy_module()
    with kb.connect() as conn:
        historical_id = kb.create_task(
            conn, title="old", assignee="auditor", priority=100,
        )
        active_id = kb.create_task(
            conn, title="new", assignee="builder", priority=1,
        )
    _activate(board_home)
    args = argparse.Namespace(
        profile="builder",
        claim_assignees="auditor,builder",
        assist_claim_delay_s=0.0,
        assist_claim_delay_for=[],
        assist_claim_profile_delay=[],
        previous_worker_delay_s=0.0,
    )

    with kb.connect() as conn:
        candidate = base_listener._select_ready_candidate(conn, args)
        with pytest.raises(policy_module.RolePolicyDenied):
            kb.claim_task(conn, historical_id, claimer="auditor-pane")

    assert candidate.id == active_id
