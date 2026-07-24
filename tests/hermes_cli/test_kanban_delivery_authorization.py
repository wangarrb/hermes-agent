"""Delivery authorization bound to immutable Git and task identities."""

from __future__ import annotations

import importlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


def _delivery_module():
    return importlib.import_module("hermes_cli.kanban_delivery")


def _git(path: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


@pytest.fixture
def delivery_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    publisher = tmp_path / "publisher"
    publisher.mkdir()
    _git(publisher, "init")
    _git(publisher, "config", "user.email", "test@example.com")
    _git(publisher, "config", "user.name", "Test User")
    _git(publisher, "branch", "-M", "main")
    (publisher / "README.md").write_text("base\n", encoding="utf-8")
    _git(publisher, "add", "README.md")
    _git(publisher, "commit", "-m", "base")
    base = _git(publisher, "rev-parse", "HEAD")

    upstream = tmp_path / "upstream.git"
    subprocess.run(
        ["git", "init", "--bare", str(upstream)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(publisher, "remote", "add", "origin", str(upstream))
    _git(publisher, "push", "-u", "origin", "main")
    _git(upstream, "symbolic-ref", "HEAD", "refs/heads/main")

    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", str(upstream), str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    kb.init_db()
    return {
        "home": home,
        "publisher": publisher,
        "repo": repo,
        "base": base,
        "tmp_path": tmp_path,
    }


def _create_task(conn, env, label="delivery"):
    worktree = env["tmp_path"] / f"worktree-{label}"
    task_id = kb.create_task(
        conn,
        title=label,
        assignee="implementer",
        workspace_kind="worktree",
        workspace_path=str(worktree),
        branch_name="implementer/{task_id}/g{generation}",
        base_commit=env["base"],
        target_branch="main",
    )
    task = kb.get_task(conn, task_id)
    _git(env["repo"], "branch", f"implementer/{task.id}/g0", env["base"])
    _git(
        env["repo"],
        "worktree",
        "add",
        str(worktree),
        f"implementer/{task.id}/g0",
    )
    return task, worktree


def _prepare(conn, env, label="delivery"):
    task, worktree = _create_task(conn, env, label)
    contract = kb.prepare_task_worktree(
        conn, task.id, source_branch="main", upstream="origin",
    )
    reservation = kb.reserve_task_scopes(
        conn,
        task.id,
        write_set=(f"src/{label}.py",),
        artifact_namespace=f"/tmp/hermes-delivery/{task.id}/g1",
        expected_generation=1,
        expected_base_commit=env["base"],
    )
    prepared = kb.prepare_task_delivery(
        conn,
        task.id,
        workspace_contract=contract,
        reservation_id=reservation.id,
        actor="planner",
        source="workflow",
    )
    return task, worktree, reservation, prepared


def _deliver(conn, env, label="delivery"):
    task, worktree, reservation, _ = _prepare(conn, env, label)
    kb.start_task_delivery(
        conn, task.id, actor="implementer", source="worker",
    )
    path = worktree / f"{label}.txt"
    path.write_text(f"{label}\n", encoding="utf-8")
    _git(worktree, "add", path.name)
    _git(worktree, "commit", "-m", f"deliver {label}")
    frozen = kb.freeze_task_delivery(conn, task.id)
    delivered = kb.mark_task_delivered(
        conn,
        task.id,
        workspace_contract=frozen,
        actor="implementer",
        source="worker",
    )
    return task, worktree, reservation, frozen, delivered


def _authorize(conn, env, label="delivery"):
    task, worktree, reservation, frozen, _ = _deliver(conn, env, label)
    accepted = kb.accept_task_delivery(
        conn, task.id, actor="reviewer", source="review",
    )
    authorized = kb.authorize_task_delivery(
        conn,
        task.id,
        integrator="integrator-pane",
        actor="user:alice",
        source="interactive",
    )
    return task, worktree, reservation, frozen, accepted, authorized


def test_legal_delivery_chain_binds_authorization_and_never_merges(delivery_env):
    delivery = _delivery_module()
    with kb.connect() as conn:
        task, worktree, reservation, frozen, accepted, authorized = _authorize(
            conn, delivery_env,
        )
        target_before = _git(delivery_env["repo"], "rev-parse", "refs/heads/main")

        integrated = kb.integrate_task_delivery(
            conn, task.id, integrator="integrator-pane",
        )
        persisted_reservation = kb.get_scope_reservation(conn, reservation.id)
        events = kb.list_events(conn, task.id)

    expected_tuple = {
        "task_id": task.id,
        "generation": 1,
        "delivery_sha": frozen["delivery_commit"],
        "target_ref": "refs/heads/main",
        "target_sha": delivery_env["base"],
        "integrator": "integrator-pane",
    }
    assert accepted.state == "accepted"
    assert accepted.accepted_delivery_sha == frozen["delivery_commit"]
    assert accepted.accepted_delivery_tree == frozen["delivery_tree"]
    assert authorized.state == "authorized"
    assert authorized.authorization == expected_tuple
    assert delivery.authorization_matches(authorized, expected_tuple)
    assert integrated.state == "integrated"
    assert persisted_reservation.status == "integrated"
    assert _git(delivery_env["repo"], "rev-parse", "refs/heads/main") == target_before
    assert _git(worktree, "rev-parse", "HEAD") == frozen["delivery_commit"]
    auth_event = next(event for event in events if event.kind == "delivery_authorized")
    assert auth_event.payload["actor"] == "user:alice"
    assert auth_event.payload["source"] == "interactive"
    assert auth_event.payload["authorization"] == expected_tuple


def test_illegal_transition_fails_closed_with_durable_event(delivery_env):
    delivery = _delivery_module()
    with kb.connect() as conn:
        task, _, _, prepared = _prepare(conn, delivery_env, "illegal")

        with pytest.raises(delivery.DeliveryTransitionError, match="prepared.*accepted"):
            kb.accept_task_delivery(
                conn, task.id, actor="reviewer", source="review",
            )

        persisted = kb.get_task_delivery(conn, task.id)
        rejection = kb.list_events(conn, task.id)[-1]

    assert prepared.state == persisted.state == "prepared"
    assert rejection.kind == "delivery_transition_rejected"
    assert rejection.payload["from_state"] == "prepared"
    assert rejection.payload["to_state"] == "accepted"


def test_writable_claim_advances_prepared_delivery_to_running(delivery_env):
    with kb.connect() as conn:
        task, _, _, prepared = _prepare(conn, delivery_env, "claim")

        claimed = kb.claim_task(
            conn,
            task.id,
            claimer="implementer-pane",
            require_reservation=True,
        )
        running = kb.get_task_delivery(conn, task.id)

    assert prepared.state == "prepared"
    assert claimed is not None
    assert running.state == "running"
    assert running.generation == claimed.generation


def test_delivered_rejects_forged_workspace_identity(delivery_env):
    delivery = _delivery_module()
    with kb.connect() as conn:
        task, worktree, _, _ = _prepare(conn, delivery_env, "forged")
        kb.start_task_delivery(
            conn, task.id, actor="implementer", source="worker",
        )
        (worktree / "forged.txt").write_text("delivery\n", encoding="utf-8")
        _git(worktree, "add", "forged.txt")
        _git(worktree, "commit", "-m", "forged delivery")
        frozen = kb.freeze_task_delivery(conn, task.id)
        forged = {**frozen, "repository": str(delivery_env["publisher"])}

        with pytest.raises(delivery.DeliveryTransitionError, match="workspace identity"):
            kb.mark_task_delivered(
                conn,
                task.id,
                workspace_contract=forged,
                actor="implementer",
                source="worker",
            )

        persisted = kb.get_task_delivery(conn, task.id)
        rejection = kb.list_events(conn, task.id)[-1]

    assert persisted.state == "running"
    assert rejection.kind == "delivery_transition_rejected"


def test_automated_smoke_cannot_forge_user_authorization(delivery_env):
    delivery = _delivery_module()
    with kb.connect() as conn:
        task, _, _, _, _ = _deliver(conn, delivery_env, "smoke")
        kb.accept_task_delivery(
            conn, task.id, actor="reviewer", source="review",
        )

        with pytest.raises(delivery.DeliveryAuthorizationError, match="user source"):
            kb.authorize_task_delivery(
                conn,
                task.id,
                integrator="integrator-pane",
                actor="pytest-smoke",
                source="automated-smoke",
            )

        persisted = kb.get_task_delivery(conn, task.id)
        rejection = kb.list_events(conn, task.id)[-1]

    assert persisted.state == "accepted"
    assert rejection.kind == "delivery_authorization_rejected"
    assert rejection.payload["actor"] == "pytest-smoke"
    assert rejection.payload["source"] == "automated-smoke"


def test_authorize_rejects_acceptance_for_a_different_delivery(delivery_env):
    delivery = _delivery_module()
    with kb.connect() as conn:
        task, _, _, _, _ = _deliver(conn, delivery_env, "stale-review")
        accepted = kb.accept_task_delivery(
            conn, task.id, actor="reviewer", source="review",
        )
        tampered = accepted.to_dict()
        tampered["delivery_sha"] = "d" * 40
        conn.execute(
            "UPDATE tasks SET delivery_json = ? WHERE id = ?",
            (json.dumps(tampered), task.id),
        )
        conn.commit()

        with pytest.raises(delivery.DeliveryAuthorizationError, match="accepted review"):
            kb.authorize_task_delivery(
                conn,
                task.id,
                integrator="integrator-pane",
                actor="user:alice",
                source="interactive",
            )

        persisted = kb.get_task_delivery(conn, task.id)
        rejection = kb.list_events(conn, task.id)[-1]

    assert persisted.state == "accepted"
    assert rejection.kind == "delivery_authorization_rejected"


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("task_id", "t_other"),
        ("generation", 2),
        ("delivery_sha", "b" * 40),
        ("target_ref", "refs/heads/release"),
        ("target_sha", "c" * 40),
        ("integrator", "other-integrator"),
    ],
)
def test_any_authorization_tuple_change_invalidates_authorization(
    delivery_env, field, replacement,
):
    delivery = _delivery_module()
    with kb.connect() as conn:
        _, _, _, _, _, authorized = _authorize(
            conn, delivery_env, f"tuple-{field}",
        )

    changed = dict(authorized.authorization)
    changed[field] = replacement
    assert not delivery.authorization_matches(authorized, changed)


@pytest.mark.parametrize("mutation", ["target", "delivery", "dirty", "generation"])
def test_integrate_rechecks_immutable_git_and_task_identity(delivery_env, mutation):
    delivery = _delivery_module()
    with kb.connect() as conn:
        task, worktree, _, frozen, _, _ = _authorize(
            conn, delivery_env, f"recheck-{mutation}",
        )
        if mutation == "target":
            target_file = delivery_env["repo"] / "target-change.txt"
            target_file.write_text("target changed\n", encoding="utf-8")
            _git(delivery_env["repo"], "add", target_file.name)
            _git(delivery_env["repo"], "commit", "-m", "move target")
        elif mutation == "delivery":
            delivery_file = worktree / "post-authorization.txt"
            delivery_file.write_text("changed\n", encoding="utf-8")
            _git(worktree, "add", delivery_file.name)
            _git(worktree, "commit", "-m", "move delivery")
        elif mutation == "dirty":
            (worktree / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        else:
            conn.execute(
                "UPDATE tasks SET generation = 2 WHERE id = ?", (task.id,),
            )
            conn.commit()

        with pytest.raises(delivery.DeliveryAuthorizationError):
            kb.integrate_task_delivery(
                conn, task.id, integrator="integrator-pane",
            )

        persisted = kb.get_task_delivery(conn, task.id)
        rejection = kb.list_events(conn, task.id)[-1]

    assert persisted.state == "authorized"
    assert persisted.delivery_sha == frozen["delivery_commit"]
    assert rejection.kind == "delivery_integration_rejected"


def test_new_generation_can_prepare_after_old_delivery_is_superseded(delivery_env):
    with kb.connect() as conn:
        task, _, _, old_prepared = _prepare(conn, delivery_env, "superseded")
        conn.execute(
            "UPDATE tasks SET generation = 2 WHERE id = ?",
            (task.id,),
        )
        conn.commit()

        next_contract = dict(old_prepared.workspace_contract)
        next_contract["generation"] = 2
        next_contract["branch_name"] = f"implementer/{task.id}/g2"
        reservation = kb.reserve_task_scopes(
            conn,
            task.id,
            write_set=("src/superseded-g2.py",),
            artifact_namespace=f"/tmp/hermes-delivery/{task.id}/g2",
            expected_generation=2,
            expected_base_commit=delivery_env["base"],
        )

        prepared = kb.prepare_task_delivery(
            conn,
            task.id,
            workspace_contract=next_contract,
            reservation_id=reservation.id,
            actor="planner",
            source="workflow",
        )
        persisted = kb.get_task_delivery(conn, task.id)
        events = kb.list_events(conn, task.id)

    assert old_prepared.generation == 1
    assert prepared.state == persisted.state == "prepared"
    assert prepared.generation == persisted.generation == 2
    assert any(event.kind == "delivery_superseded" for event in events)


def test_old_tasks_have_no_delivery_state_after_migration(delivery_env):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy task")
        task = kb.get_task(conn, task_id)

    assert task.delivery_state is None
    assert task.delivery is None


def test_legacy_board_migrates_delivery_columns(delivery_env):
    legacy_path = delivery_env["tmp_path"] / "legacy-kanban.db"
    kb.init_db(legacy_path)
    with kb.connect(legacy_path) as conn:
        task_id = kb.create_task(conn, title="pre-h6 task")

    raw = sqlite3.connect(legacy_path)
    try:
        raw.execute("ALTER TABLE tasks DROP COLUMN delivery_state")
        raw.execute("ALTER TABLE tasks DROP COLUMN delivery_json")
        raw.commit()
    finally:
        raw.close()

    kb.init_db(legacy_path)
    with kb.connect(legacy_path) as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(tasks)")
        }
        task = kb.get_task(conn, task_id)

    assert {"delivery_state", "delivery_json"} <= columns
    assert task.delivery_state is None
    assert task.delivery is None


def test_cli_delivery_operations_and_show_json(delivery_env):
    with kb.connect() as conn:
        task, worktree = _create_task(conn, delivery_env, "cli")

    prepared = json.loads(kc.run_slash(
        f"prepare {task.id} --source-branch main "
        f"--write-set src/cli.py "
        f"--artifact-namespace /tmp/hermes-delivery/{task.id}/g1 --json"
    ))
    assert prepared["state"] == "prepared"

    assert "Claimed" in kc.run_slash(f"claim {task.id}")
    (worktree / "cli.txt").write_text("cli\n", encoding="utf-8")
    _git(worktree, "add", "cli.txt")
    _git(worktree, "commit", "-m", "cli delivery")

    assert json.loads(kc.run_slash(f"freeze {task.id} --json"))["state"] == "delivered"
    assert json.loads(kc.run_slash(
        f"accept {task.id} --actor reviewer --source review --json"
    ))["state"] == "accepted"
    authorized = json.loads(kc.run_slash(
        f"authorize {task.id} --integrator integrator-pane "
        f"--actor user:alice --source interactive --json"
    ))
    assert authorized["state"] == "authorized"

    shown = json.loads(kc.run_slash(f"show {task.id} --json"))
    assert shown["delivery"]["state"] == "authorized"
    assert shown["delivery"]["authorization"] == authorized["authorization"]
    assert "delivery: authorized" in kc.run_slash(f"show {task.id}")

    integrated = json.loads(kc.run_slash(
        f"integrate {task.id} --integrator integrator-pane --json"
    ))
    assert integrated["state"] == "integrated"

    with kb.connect() as conn:
        abandoned_task, _ = _create_task(conn, delivery_env, "abandon-cli")
    abandoned = kc.run_slash(
        f"prepare {abandoned_task.id} --source-branch main "
        f"--write-set src/abandon.py "
        f"--artifact-namespace /tmp/hermes-delivery/{abandoned_task.id}/g1 --json"
    )
    assert json.loads(abandoned)["state"] == "prepared"
    result = json.loads(kc.run_slash(
        f"abandon {abandoned_task.id} --actor planner --reason superseded --json"
    ))
    assert result["state"] == "abandoned"
