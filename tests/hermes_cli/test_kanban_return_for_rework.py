"""Focused DB-contract tests for return-for-rework control delivery."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_return_for_rework_delivers_control_and_releases_root_after_ack(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="implementation", assignee="implementer")
        child = kb.create_task(
            conn, title="review", assignee="reviewer", parents=[root]
        )
        assert kb.complete_task(conn, root)
        claimed = kb.claim_task(conn, child, claimer="review-pane")
        assert claimed is not None
        old_run_id = claimed.current_run_id
        assert old_run_id is not None

        returned = kb.return_task_for_rework(
            conn, root, actor="reviewer", reason="reject result"
        )
        root_after_return = kb.get_task(conn, root)
        child_after_return = kb.get_task(conn, child)
        old_run = kb.get_run(conn, old_run_id)
        controls = kb.list_control_messages(conn, return_task_id=root)
        peeked = kb.peek_control_message(
            conn, profiles=["reviewer"], receiver="review-pane"
        )

        assert len(returned.control_ids) == 1
        assert root_after_return is not None
        assert root_after_return.status == "todo"
        assert root_after_return.rework_hold is True
        assert child_after_return is not None
        assert child_after_return.status == "todo"
        assert child_after_return.current_run_id is None
        assert old_run is not None
        assert old_run.outcome == "returned_for_rework"
        assert [control.id for control in controls] == returned.control_ids
        assert peeked is not None
        assert peeked.id == returned.control_ids[0]
        assert peeked.status == "pending"
        assert peeked.target_profile == "reviewer"

        leased = kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="review-pane"
        )
        assert leased is not None
        assert leased.id == returned.control_ids[0]
        assert leased.status == "delivering"
        assert kb.mark_control_delivered(
            conn, leased.id, receiver="review-pane"
        )
        assert kb.ack_control_message(conn, leased.id, receiver="review-pane")

        released = kb.get_task(conn, root)
        acked = kb.list_control_messages(conn, return_task_id=root, status="acked")

    assert released is not None
    assert released.rework_hold is False
    assert released.status == "ready"
    assert [control.id for control in acked] == returned.control_ids


def test_release_control_lease_makes_pause_available_to_another_receiver(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementation", assignee="reviewer")
        assert kb.claim_task(conn, task_id, claimer="first-pane") is not None
        returned = kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="retry correctly"
        )
        control_id = returned.control_ids[0]

        first_lease = kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="first-pane"
        )
        assert first_lease is not None
        assert first_lease.id == control_id
        assert kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="second-pane"
        ) is None
        assert kb.release_control_lease(
            conn, control_id, receiver="first-pane"
        )

        second_lease = kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="second-pane"
        )
        assert second_lease is not None
        assert second_lease.id == control_id
        assert second_lease.delivery_owner == "second-pane"
        assert kb.ack_control_message(conn, control_id, receiver="second-pane")
        assert kb.ack_control_message(conn, control_id, receiver="second-pane")
        acked = kb.list_control_messages(
            conn, return_task_id=task_id, status="acked"
        )

    assert acked


def test_rework_hold_drains_only_after_every_control_ack(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root")
        left = kb.create_task(conn, title="left", assignee="reviewer", parents=[root])
        right = kb.create_task(conn, title="right", assignee="critic", parents=[root])
        assert kb.complete_task(conn, root)
        assert kb.claim_task(conn, left, claimer="reviewer-pane") is not None
        assert kb.claim_task(conn, right, claimer="critic-pane") is not None

        returned = kb.return_task_for_rework(
            conn, root, actor="reviewer", reason="both reviews are obsolete"
        )
        assert len(returned.control_ids) == 2
        returned_root = kb.get_task(conn, root)
        assert returned_root is not None
        assert returned_root.rework_hold is True

        assert kb.ack_control_message(
            conn, returned.control_ids[0], receiver="reviewer-pane"
        )
        after_first = kb.get_task(conn, root)
        assert after_first is not None
        assert after_first.rework_hold is True
        assert after_first.status == "todo"

        assert kb.ack_control_message(
            conn, returned.control_ids[1], receiver="critic-pane"
        )
        after_second = kb.get_task(conn, root)

    assert after_second is not None
    assert after_second.rework_hold is False
    assert after_second.status == "ready"


def test_old_generation_cannot_complete_or_block_current_generation(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementation", assignee="implementer")
        first = kb.claim_task(conn, task_id, claimer="old-pane")
        assert first is not None

        returned = kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="retry correctly"
        )
        assert kb.ack_control_message(
            conn, returned.control_ids[0], receiver="old-pane"
        )
        second = kb.claim_task(conn, task_id, claimer="new-pane")
        assert second is not None

        assert not kb.complete_task(
            conn,
            task_id,
            summary="late old completion",
            expected_run_id=first.current_run_id,
            expected_generation=first.generation,
        )
        assert kb.get_task(conn, task_id).status == "running"
        assert not kb.block_task(
            conn,
            task_id,
            reason="late old block",
            expected_run_id=first.current_run_id,
            expected_generation=first.generation,
        )
        assert kb.get_task(conn, task_id).status == "running"
        assert kb.complete_task(
            conn,
            task_id,
            summary="valid new completion",
            expected_run_id=second.current_run_id,
            expected_generation=second.generation,
        )
