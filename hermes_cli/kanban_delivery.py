"""Immutable delivery authorization and Git identity validation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


DELIVERY_STATES = (
    "prepared",
    "running",
    "delivered",
    "accepted",
    "authorized",
    "integrated",
    "abandoned",
)
ALLOWED_TRANSITIONS = {
    None: frozenset({"prepared"}),
    "prepared": frozenset({"running", "abandoned"}),
    "running": frozenset({"delivered", "abandoned"}),
    "delivered": frozenset({"accepted", "abandoned"}),
    "accepted": frozenset({"authorized", "abandoned"}),
    "authorized": frozenset({"integrated", "abandoned"}),
    "integrated": frozenset(),
    "abandoned": frozenset(),
}
USER_AUTHORIZATION_SOURCES = frozenset({"interactive", "manual", "user"})
AUTHORIZATION_FIELDS = (
    "task_id",
    "generation",
    "delivery_sha",
    "target_ref",
    "target_sha",
    "integrator",
)


class DeliveryError(ValueError):
    """A delivery identity or operation is invalid."""


class DeliveryTransitionError(DeliveryError):
    """A delivery state transition is not permitted."""


class DeliveryAuthorizationError(DeliveryError):
    """An authorization is missing, forged, or stale."""


@dataclass(frozen=True)
class DeliveryRecord:
    state: str
    task_id: str
    generation: int
    base_commit: str
    reservation_id: int
    workspace_contract: dict[str, Any]
    delivery_sha: str | None = None
    delivery_tree: str | None = None
    accepted_by: str | None = None
    accepted_source: str | None = None
    accepted_delivery_sha: str | None = None
    accepted_delivery_tree: str | None = None
    authorization: dict[str, Any] | None = None
    authorization_actor: str | None = None
    authorization_source: str | None = None

    def with_state(self, state: str, **changes: Any) -> "DeliveryRecord":
        return replace(self, state=state, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "task_id": self.task_id,
            "generation": self.generation,
            "base_commit": self.base_commit,
            "reservation_id": self.reservation_id,
            "workspace_contract": dict(self.workspace_contract),
            "delivery_sha": self.delivery_sha,
            "delivery_tree": self.delivery_tree,
            "accepted_by": self.accepted_by,
            "accepted_source": self.accepted_source,
            "accepted_delivery_sha": self.accepted_delivery_sha,
            "accepted_delivery_tree": self.accepted_delivery_tree,
            "authorization": (
                dict(self.authorization) if self.authorization is not None else None
            ),
            "authorization_actor": self.authorization_actor,
            "authorization_source": self.authorization_source,
        }


def parse_delivery(
    state: str | None,
    value: str | Mapping[str, Any] | None,
) -> DeliveryRecord | None:
    if state is None and value is None:
        return None
    try:
        data = json.loads(value) if isinstance(value, str) else dict(value or {})
    except (TypeError, ValueError) as exc:
        raise DeliveryError(f"invalid stored delivery record: {exc}") from exc
    resolved_state = str(state or data.get("state") or "").strip()
    if resolved_state not in DELIVERY_STATES:
        raise DeliveryError(f"invalid stored delivery state: {resolved_state!r}")
    return DeliveryRecord(
        state=resolved_state,
        task_id=str(data["task_id"]),
        generation=int(data["generation"]),
        base_commit=str(data["base_commit"]),
        reservation_id=int(data["reservation_id"]),
        workspace_contract=dict(data["workspace_contract"]),
        delivery_sha=data.get("delivery_sha"),
        delivery_tree=data.get("delivery_tree"),
        accepted_by=data.get("accepted_by"),
        accepted_source=data.get("accepted_source"),
        accepted_delivery_sha=data.get("accepted_delivery_sha"),
        accepted_delivery_tree=data.get("accepted_delivery_tree"),
        authorization=(
            dict(data["authorization"]) if data.get("authorization") else None
        ),
        authorization_actor=data.get("authorization_actor"),
        authorization_source=data.get("authorization_source"),
    )


def dumps_delivery(record: DeliveryRecord) -> str:
    return json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)


def transition_allowed(current: str | None, target: str) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())


def authorization_matches(
    record: DeliveryRecord,
    candidate: Mapping[str, Any],
) -> bool:
    if record.state not in {"authorized", "integrated"} or not record.authorization:
        return False
    return all(
        record.authorization.get(field) == candidate.get(field)
        for field in AUTHORIZATION_FIELDS
    )


def acceptance_matches(record: DeliveryRecord) -> bool:
    return bool(
        record.accepted_by
        and record.accepted_source
        and record.accepted_delivery_sha == record.delivery_sha
        and record.accepted_delivery_tree == record.delivery_tree
    )


def target_ref_for_task(task: Any) -> str:
    target = str(getattr(task, "target_branch", None) or "").strip()
    if not target:
        raise DeliveryAuthorizationError("task target branch is missing")
    return target if target.startswith("refs/") else f"refs/heads/{target}"


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "git command failed").strip()
        raise DeliveryAuthorizationError(detail)
    return result.stdout.strip()


def repository_for_record(record: DeliveryRecord) -> Path:
    repository = Path(str(record.workspace_contract.get("repository") or ""))
    if not repository.is_absolute() or not repository.is_dir():
        raise DeliveryAuthorizationError("delivery repository identity is invalid")
    return repository.resolve(strict=False)


def current_target_identity(task: Any, record: DeliveryRecord) -> tuple[str, str]:
    target_ref = target_ref_for_task(task)
    target_sha = _git(repository_for_record(record), "rev-parse", "--verify", f"{target_ref}^{{commit}}")
    return target_ref, target_sha


def authorization_tuple(
    task: Any,
    record: DeliveryRecord,
    *,
    integrator: str,
) -> dict[str, Any]:
    name = str(integrator or "").strip()
    if not name:
        raise DeliveryAuthorizationError("integrator is required")
    if not record.delivery_sha:
        raise DeliveryAuthorizationError("delivery SHA is missing")
    target_ref, target_sha = current_target_identity(task, record)
    return {
        "task_id": str(task.id),
        "generation": int(task.generation),
        "delivery_sha": record.delivery_sha,
        "target_ref": target_ref,
        "target_sha": target_sha,
        "integrator": name,
    }


def validate_frozen_contract(task: Any, contract: Mapping[str, Any]) -> None:
    if not contract.get("frozen"):
        raise DeliveryError("delivery workspace contract is not frozen")
    required = ("delivery_commit", "delivery_tree", "branch", "worktree")
    missing = [field for field in required if not contract.get(field)]
    if missing:
        raise DeliveryError("frozen delivery identity missing: " + ", ".join(missing))
    if str(contract.get("task_id")) != str(task.id):
        raise DeliveryError("frozen delivery task identity mismatch")
    if int(contract.get("generation", -1)) != int(task.generation):
        raise DeliveryError("frozen delivery generation mismatch")
    stored = getattr(task, "workspace_contract", None)
    identity_fields = (
        "version",
        "repository",
        "worktree",
        "common_dir",
        "base_commit",
        "target_branch",
        "branch",
        "task_id",
        "generation",
        "manifest_path",
        "delivery_commit",
        "delivery_tree",
        "frozen",
    )
    if stored is None or any(
        stored.get(field) != contract.get(field) for field in identity_fields
    ):
        raise DeliveryError("frozen delivery workspace identity mismatch")
    worktree = Path(str(contract["worktree"]))
    if _git(worktree, "status", "--porcelain"):
        raise DeliveryError("delivery worktree must be clean")
    commit = _git(worktree, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git(worktree, "rev-parse", "--verify", "HEAD^{tree}")
    if commit != contract["delivery_commit"] or tree != contract["delivery_tree"]:
        raise DeliveryError("frozen delivery commit/tree identity mismatch")


def integration_mismatches(
    task: Any,
    record: DeliveryRecord,
    *,
    integrator: str,
) -> list[str]:
    mismatches: list[str] = []
    authorization = record.authorization or {}
    stored_contract = getattr(task, "workspace_contract", None)
    if stored_contract is None or any(
        stored_contract.get(field) != record.workspace_contract.get(field)
        for field in (
            "repository",
            "worktree",
            "common_dir",
            "branch",
            "task_id",
            "generation",
            "delivery_commit",
            "delivery_tree",
        )
    ):
        mismatches.append("workspace_contract")
    expected = {
        "task_id": str(task.id),
        "generation": int(task.generation),
        "delivery_sha": record.delivery_sha,
        "target_ref": target_ref_for_task(task),
        "integrator": str(integrator or "").strip(),
    }
    for field, value in expected.items():
        if authorization.get(field) != value:
            mismatches.append(field)
    try:
        target_ref, target_sha = current_target_identity(task, record)
        if authorization.get("target_ref") != target_ref:
            mismatches.append("target_ref")
        if authorization.get("target_sha") != target_sha:
            mismatches.append("target_sha")
        repository = repository_for_record(record)
        branch = str(record.workspace_contract.get("branch") or "")
        delivery_sha = _git(
            repository, "rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}",
        )
        if delivery_sha != record.delivery_sha:
            mismatches.append("delivery_sha")
        worktree = Path(str(record.workspace_contract.get("worktree") or ""))
        if _git(worktree, "status", "--porcelain"):
            mismatches.append("workspace_clean")
    except DeliveryAuthorizationError as exc:
        mismatches.append(f"git_identity:{exc}")
    return list(dict.fromkeys(mismatches))
