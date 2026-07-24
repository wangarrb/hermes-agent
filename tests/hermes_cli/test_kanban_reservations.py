"""Atomic write-set and artifact reservations for Kanban tasks."""

from __future__ import annotations

import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


BASE_COMMIT = "a" * 40


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "kanban.db"
    kb.init_db(path)
    return path


def _reservations_module():
    return importlib.import_module("hermes_cli.kanban_reservations")


def _create_task(conn, title: str) -> str:
    return kb.create_task(
        conn,
        title=title,
        assignee="implementer",
        workspace_kind="worktree",
        base_commit=BASE_COMMIT,
        target_branch="main",
    )


def _reserve(
    conn,
    task_id: str,
    *,
    write_set=("src/feature",),
    artifact_namespace="/tmp/hermes-artifacts/default",
):
    return kb.reserve_task_scopes(
        conn,
        task_id,
        write_set=write_set,
        artifact_namespace=artifact_namespace,
        expected_generation=1,
        expected_base_commit=BASE_COMMIT,
    )


@pytest.mark.parametrize(
    ("left", "right", "overlaps"),
    [
        ("foo", "foo", True),
        ("foo", "foo/bar.py", True),
        ("foo/bar.py", "foo", True),
        ("foo", "foobar", False),
        ("src/a", "src/ab", False),
    ],
)
def test_path_overlap_uses_canonical_components(left, right, overlaps):
    reservations = _reservations_module()

    assert reservations.path_scopes_overlap(left, right) is overlaps
    assert reservations.canonical_write_scope(f"./{left}") == left


def test_artifact_namespace_uses_absolute_component_prefixes():
    reservations = _reservations_module()

    assert reservations.artifact_scopes_overlap(
        "/tmp/artifacts/task", "/tmp/artifacts/task/logs",
    )
    assert not reservations.artifact_scopes_overlap(
        "/tmp/artifacts/task", "/tmp/artifacts/task-next",
    )
    with pytest.raises(ValueError, match="absolute"):
        reservations.canonical_artifact_namespace("artifacts/task")
    with pytest.raises(ValueError, match="parent"):
        reservations.canonical_write_scope("src/../secrets")
    with pytest.raises(ValueError, match="required"):
        reservations.reservation_manifest(
            task_id="t_missing_artifact",
            generation=1,
            base_commit=BASE_COMMIT,
            write_set=("src",),
            artifact_namespace=None,
        )


def test_canonical_write_set_collapses_redundant_descendants():
    reservations = _reservations_module()

    assert reservations.canonical_write_set(
        ("src", "src/feature.py", "tests/unit.py", "src"),
    ) == ("src", "tests/unit.py")


def test_reserve_disjoint_scopes_and_fail_closed_with_owner_evidence(db_path):
    reservations = _reservations_module()
    with kb.connect(db_path) as conn:
        owner = _create_task(conn, "owner")
        disjoint = _create_task(conn, "disjoint")
        path_conflict = _create_task(conn, "path conflict")
        artifact_conflict = _create_task(conn, "artifact conflict")

        first = _reserve(
            conn,
            owner,
            write_set=("src/foo",),
            artifact_namespace="/tmp/artifacts/owner",
        )
        second = _reserve(
            conn,
            disjoint,
            write_set=("src/foobar",),
            artifact_namespace="/tmp/artifacts/disjoint",
        )

        assert first.status == second.status == "active"
        with pytest.raises(reservations.ReservationConflictError) as path_error:
            _reserve(
                conn,
                path_conflict,
                write_set=("src/foo/child.py",),
                artifact_namespace="/tmp/artifacts/path-conflict",
            )
        with pytest.raises(reservations.ReservationConflictError) as artifact_error:
            _reserve(
                conn,
                artifact_conflict,
                write_set=("tests/independent.py",),
                artifact_namespace="/tmp/artifacts/owner/results",
            )

    assert path_error.value.owner_task_id == owner
    assert path_error.value.owner_generation == 1
    assert "src/foo" in str(path_error.value)
    assert owner in str(path_error.value)
    assert artifact_error.value.owner_task_id == owner
    assert "/tmp/artifacts/owner" in str(artifact_error.value)


def test_two_sqlite_writers_cannot_reserve_the_same_scope(db_path):
    with kb.connect(db_path) as conn:
        task_ids = [_create_task(conn, f"writer {index}") for index in range(2)]

    barrier = threading.Barrier(2)

    def attempt(task_id):
        with kb.connect(db_path) as conn:
            barrier.wait(timeout=5)
            try:
                reservation = _reserve(
                    conn,
                    task_id,
                    write_set=("shared/component",),
                    artifact_namespace=f"/tmp/artifacts/{task_id}",
                )
            except Exception as exc:
                return "conflict", exc
            return "reserved", reservation

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, task_ids))

    assert sorted(result[0] for result in results) == ["conflict", "reserved"]
    conflict = next(result[1] for result in results if result[0] == "conflict")
    assert isinstance(
        conflict, _reservations_module().ReservationConflictError,
    )
    with kb.connect(db_path) as conn:
        active = conn.execute(
            "SELECT COUNT(*) FROM task_scope_reservations WHERE status = 'active'"
        ).fetchone()[0]
    assert active == 1


def test_crash_resume_is_idempotent_and_generation_supersedes_old_manifest(db_path):
    with kb.connect(db_path) as conn:
        task_id = _create_task(conn, "resume")
        first = _reserve(
            conn,
            task_id,
            write_set=("b/file.py", "a/file.py", "b/file.py"),
            artifact_namespace="/tmp/artifacts/resume",
        )

    with kb.connect(db_path) as conn:
        resumed = _reserve(
            conn,
            task_id,
            write_set=("a/file.py", "b/file.py"),
            artifact_namespace="/tmp/artifacts/resume/.",
        )
        assert resumed.id == first.id
        assert resumed.manifest_hash == first.manifest_hash
        assert conn.execute(
            "SELECT COUNT(*) FROM task_scope_reservations WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 1

        conn.execute(
            "UPDATE tasks SET generation = 2 WHERE id = ?",
            (task_id,),
        )
        conn.commit()
        generation_two = kb.reserve_task_scopes(
            conn,
            task_id,
            write_set=("a/file.py", "b/file.py"),
            artifact_namespace="/tmp/artifacts/resume",
            expected_generation=2,
            expected_base_commit=BASE_COMMIT,
        )
        old_status = conn.execute(
            "SELECT status FROM task_scope_reservations WHERE id = ?",
            (first.id,),
        ).fetchone()[0]

    assert generation_two.id != first.id
    assert generation_two.generation == 2
    assert old_status == "abandoned"


def test_release_records_state_and_audit_event(db_path):
    with kb.connect(db_path) as conn:
        task_id = _create_task(conn, "release")
        reservation = _reserve(conn, task_id)

        released = kb.release_task_reservation(
            conn,
            reservation.id,
            task_id=task_id,
            expected_generation=1,
        )
        event = conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()

    assert released.status == "released"
    assert event["kind"] == "reservation_released"
    assert json.loads(event["payload"])["reservation_id"] == reservation.id


@pytest.mark.parametrize("status", ["integrated", "abandoned"])
def test_reservation_terminal_states_are_audited(db_path, status):
    with kb.connect(db_path) as conn:
        task_id = _create_task(conn, status)
        reservation = _reserve(
            conn,
            task_id,
            write_set=(f"src/{status}",),
            artifact_namespace=f"/tmp/artifacts/{status}",
        )

        transitioned = kb.transition_task_reservation(
            conn,
            reservation.id,
            status=status,
            task_id=task_id,
            expected_generation=1,
        )
        event = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()

    assert transitioned.status == status
    assert event["kind"] == f"reservation_{status}"


def test_existing_board_initialization_adds_reservation_schema(db_path):
    with kb.connect(db_path) as conn:
        conn.execute("DROP TABLE task_reservation_scopes")
        conn.execute("DROP TABLE task_scope_reservations")
        conn.commit()

    kb.init_db(db_path)

    with kb.connect(db_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "task_scope_reservations",
        "task_reservation_scopes",
    } <= tables


def test_writable_claim_requires_matching_active_reservation(db_path):
    with kb.connect(db_path) as conn:
        task_id = _create_task(conn, "writable claim")

        assert kb.claim_task(
            conn,
            task_id,
            claimer="worker:no-reservation",
            require_reservation=True,
        ) is None
        assert kb.get_task(conn, task_id).status == "ready"

        reservation = _reserve(conn, task_id)
        claimed = kb.claim_task(
            conn,
            task_id,
            claimer="worker:reserved",
            require_reservation=True,
        )

    assert reservation.status == "active"
    assert claimed is not None
    assert claimed.status == "running"
