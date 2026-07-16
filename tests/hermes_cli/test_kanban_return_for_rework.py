"""Behavior contract for generation-aware Kanban return-for-rework."""

from __future__ import annotations

import subprocess
import socket
from concurrent.futures import ThreadPoolExecutor
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


def _init_git_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=path, check=True,
        capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path,
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Kanban Test"], cwd=path,
        check=True, capture_output=True, text=True,
    )
    (path / "artifact.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "artifact.txt"], cwd=path, check=True,
        capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"], cwd=path, check=True,
        capture_output=True, text=True,
    )


def _complete_current(conn, task_id: str, *, summary: str = "done") -> None:
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert kb.complete_task(
        conn,
        task_id,
        summary=summary,
        expected_run_id=task.current_run_id,
        expected_generation=task.generation,
    )


def test_schema_exposes_generation_hold_and_control_inbox(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="schema")
        task = kb.get_task(conn, task_id)
        task_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(tasks)")
        }
        run_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_runs)")
        }
        link_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_links)")
        }
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert task.generation == 1
    assert task.rework_hold is False
    assert {"generation", "rework_hold", "rework_baseline_fingerprint"} <= task_cols
    assert "generation" in run_cols
    assert "parent_generation" in link_cols
    assert {"task_control_messages", "task_control_holds"} <= tables


@pytest.mark.parametrize(
    ("child_status", "expected_status", "generation_delta"),
    [
        ("triage", "todo", 1),
        ("todo", "todo", 1),
        ("scheduled", "todo", 1),
        ("ready", "todo", 1),
        ("blocked", "todo", 1),
        ("review", "todo", 1),
        ("done", "stale", 1),
        ("stale", "stale", 1),
        ("archived", "archived", 0),
    ],
)
def test_return_for_rework_applies_descendant_state_matrix(
    kanban_home, child_status, expected_status, generation_delta,
):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="implementation", assignee="implementer")
        child = kb.create_task(
            conn, title="review", assignee="reviewer", parents=[parent]
        )
        _complete_current(conn, parent)

        if child_status == "done":
            _complete_current(conn, child)
        else:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ?",
                (child_status, child),
            )
            conn.commit()

        before = kb.get_task(conn, child)
        result = kb.return_task_for_rework(
            conn,
            parent,
            actor="reviewer",
            reason="implementation does not satisfy the contract",
        )
        parent_after = kb.get_task(conn, parent)
        child_after = kb.get_task(conn, child)

    assert result.task_id == parent
    assert parent_after.status == "ready"
    assert parent_after.generation == 2
    assert child_after.status == expected_status
    assert child_after.generation == before.generation + generation_delta


def test_running_descendant_is_paused_and_closes_old_run(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="implementation", assignee="implementer")
        child = kb.create_task(
            conn, title="review", assignee="reviewer", parents=[parent]
        )
        _complete_current(conn, parent)
        claimed = kb.claim_task(conn, child, claimer="review-pane")
        assert claimed is not None
        old_run = claimed.current_run_id

        result = kb.return_task_for_rework(
            conn, parent, actor="reviewer", reason="reject result"
        )
        parent_after = kb.get_task(conn, parent)
        child_after = kb.get_task(conn, child)
        run = kb.get_run(conn, old_run)
        controls = kb.list_control_messages(conn, return_task_id=parent)
        child_comments = kb.list_comments(conn, child)

    assert parent_after.status == "todo"
    assert parent_after.rework_hold is True
    assert child_after.status == "todo"
    assert child_after.current_run_id is None
    assert run.outcome == "returned_for_rework"
    assert len(result.control_ids) == 1
    assert [c.id for c in controls] == result.control_ids
    assert controls[0].status == "pending"
    assert controls[0].target_profile == "reviewer"
    assert "PAUSE FOR REWORK" in child_comments[-1].body
    assert "do not complete" in child_comments[-1].body


def test_three_level_done_dag_replays_in_dependency_order(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="implementation")
        review = kb.create_task(conn, title="review", parents=[root])
        close = kb.create_task(conn, title="close", parents=[review])
        for task_id in (root, review, close):
            _complete_current(conn, task_id)

        kb.return_task_for_rework(
            conn, root, actor="reviewer", reason="root must change"
        )
        assert [kb.get_task(conn, t).status for t in (root, review, close)] == [
            "ready", "stale", "stale"
        ]

        _complete_current(conn, root, summary="root v2")
        assert kb.get_task(conn, review).status == "ready"
        assert kb.get_task(conn, close).status == "stale"
        _complete_current(conn, review, summary="review v2")
        assert kb.get_task(conn, close).status == "ready"
        _complete_current(conn, close, summary="close v2")

        assert [kb.get_task(conn, t).status for t in (root, review, close)] == [
            "done", "done", "done"
        ]
        edges = conn.execute(
            "SELECT l.parent_id, l.parent_generation, t.generation "
            "FROM task_links l JOIN tasks t ON t.id = l.parent_id "
            "ORDER BY l.parent_id"
        ).fetchall()

    assert edges
    assert all(row["parent_generation"] == row["generation"] for row in edges)


def test_old_run_cannot_complete_new_generation(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementation", assignee="implementer")
        first = kb.claim_task(conn, task_id, claimer="old-pane")
        assert first is not None
        old_run = first.current_run_id
        old_generation = first.generation

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
            expected_run_id=old_run,
            expected_generation=old_generation,
        )
        assert kb.get_task(conn, task_id).status == "running"
        assert kb.complete_task(
            conn,
            task_id,
            summary="valid new completion",
            expected_run_id=second.current_run_id,
            expected_generation=second.generation,
        )


def test_old_run_cannot_block_new_generation(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementation")
        first = kb.claim_task(conn, task_id, claimer="old-pane")
        returned = kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="retry"
        )
        kb.ack_control_message(conn, returned.control_ids[0], receiver="old-pane")
        second = kb.claim_task(conn, task_id, claimer="new-pane")

        assert not kb.block_task(
            conn,
            task_id,
            reason="late old block",
            expected_run_id=first.current_run_id,
            expected_generation=first.generation,
        )
        assert kb.get_task(conn, task_id).status == "running"
        assert kb.block_task(
            conn,
            task_id,
            reason="valid current block",
            expected_run_id=second.current_run_id,
            expected_generation=second.generation,
        )


def test_return_racing_old_completion_never_completes_new_generation(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementation")
        claimed = kb.claim_task(conn, task_id, claimer="worker")
        run_id = claimed.current_run_id
        generation = claimed.generation

    def complete_old() -> bool:
        with kb.connect() as thread_conn:
            return kb.complete_task(
                thread_conn,
                task_id,
                summary="old result",
                expected_run_id=run_id,
                expected_generation=generation,
            )

    def return_it():
        with kb.connect() as thread_conn:
            return kb.return_task_for_rework(
                thread_conn,
                task_id,
                actor="reviewer",
                reason="result rejected",
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        complete_future = pool.submit(complete_old)
        return_future = pool.submit(return_it)
        complete_future.result()
        returned = return_future.result()

    with kb.connect() as conn:
        final = kb.get_task(conn, task_id)
        controls = kb.list_control_messages(conn, return_task_id=task_id)

    assert final.generation == 2
    assert final.status in {"ready", "todo"}
    assert final.status != "done"
    assert returned.generation == 2
    if controls:
        assert final.rework_hold is True


def test_rework_hold_drains_only_after_final_control_ack(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root")
        left = kb.create_task(conn, title="left", assignee="reviewer", parents=[root])
        right = kb.create_task(conn, title="right", assignee="critic", parents=[root])
        _complete_current(conn, root)
        assert kb.claim_task(conn, left, claimer="reviewer-pane")
        assert kb.claim_task(conn, right, claimer="critic-pane")

        returned = kb.return_task_for_rework(
            conn, root, actor="reviewer", reason="both reviews are obsolete"
        )
        assert len(returned.control_ids) == 2
        assert kb.get_task(conn, root).rework_hold is True
        assert kb.ack_control_message(
            conn, returned.control_ids[0], receiver="first-pane"
        )
        after_first = kb.get_task(conn, root)
        assert after_first.rework_hold is True
        assert after_first.status == "todo"

        assert kb.ack_control_message(
            conn, returned.control_ids[1], receiver="second-pane"
        )
        after_second = kb.get_task(conn, root)

    assert after_second.rework_hold is False
    assert after_second.status == "ready"


def test_repeated_return_keeps_existing_pause_control_hold(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementation", assignee="implementer")
        assert kb.claim_task(conn, task_id, claimer="old-pane")
        first = kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="first rejection"
        )
        second = kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="more precise rejection"
        )
        held = kb.get_task(conn, task_id)

        assert second.control_ids == first.control_ids
        assert held.rework_hold is True
        assert held.status == "todo"
        assert kb.ack_control_message(
            conn, first.control_ids[0], receiver="old-pane"
        )
        released = kb.get_task(conn, task_id)

    assert released.rework_hold is False
    assert released.status == "ready"
    assert released.generation == 3


def test_nested_return_one_control_releases_every_held_root(kanban_home):
    with kb.connect() as conn:
        ancestor = kb.create_task(conn, title="implementation")
        descendant = kb.create_task(
            conn, title="review", assignee="reviewer", parents=[ancestor]
        )
        assert kb.complete_task(conn, ancestor)
        assert kb.claim_task(conn, descendant, claimer="review-pane")

        first = kb.return_task_for_rework(
            conn, ancestor, actor="reviewer", reason="implementation rejected"
        )
        second = kb.return_task_for_rework(
            conn, descendant, actor="reviewer", reason="review contract clarified"
        )
        assert second.control_ids == first.control_ids
        assert kb.get_task(conn, ancestor).rework_hold is True
        assert kb.get_task(conn, descendant).rework_hold is True

        assert kb.ack_control_message(
            conn, first.control_ids[0], receiver="review-pane"
        )
        ancestor_after = kb.get_task(conn, ancestor)
        descendant_after = kb.get_task(conn, descendant)

    assert ancestor_after.rework_hold is False
    assert ancestor_after.status == "ready"
    assert descendant_after.rework_hold is False
    assert descendant_after.status == "todo"


def test_unchanged_workspace_is_rejected_after_rework(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="implementation",
            workspace_kind="dir",
            workspace_path=str(repo),
        )
        (repo / "artifact.txt").write_text("invalid attempt\n", encoding="utf-8")
        _complete_current(conn, task_id)
        kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="artifact is invalid"
        )
        task = kb.get_task(conn, task_id)

        with pytest.raises(kb.ReworkEvidenceUnchangedError):
            kb.complete_task(
                conn,
                task_id,
                summary="same artifact again",
                expected_generation=task.generation,
            )

        (repo / "artifact.txt").write_text("corrected attempt\n", encoding="utf-8")
        assert kb.complete_task(
            conn,
            task_id,
            summary="corrected artifact",
            expected_generation=task.generation,
        )


def test_analysis_only_rework_can_explicitly_allow_no_change(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="analysis",
            workspace_kind="dir",
            workspace_path=str(repo),
        )
        _complete_current(conn, task_id)
        kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="re-evaluate evidence"
        )
        task = kb.get_task(conn, task_id)
        assert kb.complete_task(
            conn,
            task_id,
            summary="analysis conclusion updated",
            expected_generation=task.generation,
            allow_no_change=True,
            no_change_reason="analysis-only task; evidence changed in the summary",
        )


def test_reclaim_rechecks_parent_gate_instead_of_forcing_ready(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.complete_task(conn, parent)
        claimed = kb.claim_task(conn, child, claimer="interactive-listener")
        assert claimed is not None
        # Simulate a concurrent operator reopening the parent through an older
        # client while this child run is still active.
        conn.execute(
            "UPDATE tasks SET status = 'todo', generation = generation + 1 "
            "WHERE id = ?",
            (parent,),
        )
        conn.commit()

        assert kb.reclaim_task(conn, child, reason="operator recovery")
        reclaimed = kb.get_task(conn, child)
        run = kb.get_run(conn, claimed.current_run_id)

    assert reclaimed.status == "todo"
    assert run.outcome == "reclaimed"


def test_reassign_with_reclaim_updates_run_and_assignee_atomically(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="task", assignee="implementer")
        claimed = kb.claim_task(conn, task_id, claimer="old-listener")
        assert claimed is not None

        assert kb.reassign_task(
            conn,
            task_id,
            "planner",
            reclaim_first=True,
            reason="move recovery ownership",
        )
        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, claimed.current_run_id)
        events = kb.list_events(conn, task_id)

    assert task.assignee == "planner"
    assert task.status == "ready"
    assert task.current_run_id is None
    assert run.outcome == "reclaimed"
    assert any(event.kind == "reassigned" for event in events)


def test_reclaim_never_signals_interactive_listener_process(kanban_home):
    signals: list[tuple[int, int]] = []
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="task", assignee="reviewer")
        assert kb.claim_task(
            conn,
            task_id,
            claimer=f"{socket.gethostname()}:123:codex-interactive",
        )
        conn.execute("UPDATE tasks SET worker_pid = 123 WHERE id = ?", (task_id,))
        conn.commit()

        assert kb.reclaim_task(
            conn,
            task_id,
            reason="return reviewer task",
            signal_fn=lambda pid, sig: signals.append((pid, sig)),
        )

    assert signals == []
