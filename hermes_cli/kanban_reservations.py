"""Canonical scope identities for Kanban write reservations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping


RESERVATION_STATUSES = frozenset(
    {"active", "released", "integrated", "abandoned"}
)


class ReservationError(ValueError):
    """A reservation request is invalid or stale."""


class ReservationConflictError(ReservationError):
    """An active reservation already owns an overlapping scope."""

    def __init__(
        self,
        *,
        owner_task_id: str,
        owner_generation: int,
        owner_reservation_id: int,
        scope_kind: str,
        owner_scope: str,
        requested_scope: str,
    ) -> None:
        self.owner_task_id = owner_task_id
        self.owner_generation = owner_generation
        self.owner_reservation_id = owner_reservation_id
        self.scope_kind = scope_kind
        self.owner_scope = owner_scope
        self.requested_scope = requested_scope
        super().__init__(
            f"{scope_kind} scope {requested_scope!r} conflicts with "
            f"reservation {owner_reservation_id} owned by task "
            f"{owner_task_id} generation {owner_generation}: {owner_scope!r}"
        )


@dataclass(frozen=True)
class ScopeReservation:
    id: int
    task_id: str
    generation: int
    base_commit: str
    manifest_hash: str
    write_set: tuple[str, ...]
    artifact_namespace: str | None
    status: str
    created_at: int
    updated_at: int


def _reject_ambiguous_path(value: str, *, label: str) -> PurePosixPath:
    text = str(value or "").strip()
    if not text:
        raise ReservationError(f"{label} cannot be empty")
    if "\\" in text:
        raise ReservationError(f"{label} must use POSIX separators: {text!r}")
    raw_parts = text.split("/")
    if ".." in raw_parts:
        raise ReservationError(f"{label} cannot contain parent traversal: {text!r}")
    return PurePosixPath(text)


def canonical_write_scope(value: str) -> str:
    """Canonicalize a repository-relative path component scope."""
    path = _reject_ambiguous_path(value, label="write scope")
    if path.is_absolute():
        raise ReservationError(f"write scope must be repository-relative: {value!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        raise ReservationError("write scope cannot resolve to repository root")
    return "/".join(parts)


def canonical_write_set(values: Iterable[str]) -> tuple[str, ...]:
    scopes = sorted(
        {canonical_write_scope(value) for value in values},
        key=lambda scope: (len(PurePosixPath(scope).parts), scope),
    )
    if not scopes:
        raise ReservationError("write_set must contain at least one path")
    minimal: list[str] = []
    for scope in scopes:
        if any(_component_prefix(parent, scope) for parent in minimal):
            continue
        minimal.append(scope)
    return tuple(sorted(minimal))


def canonical_artifact_namespace(value: str | None) -> str | None:
    """Canonicalize an optional absolute artifact namespace."""
    if value is None:
        return None
    path = _reject_ambiguous_path(value, label="artifact namespace")
    if not path.is_absolute():
        raise ReservationError(
            f"artifact namespace must be absolute: {value!r}"
        )
    parts = tuple(part for part in path.parts if part not in {"", ".", "/"})
    if not parts:
        raise ReservationError("artifact namespace cannot be filesystem root")
    return "/" + "/".join(parts)


def _parts(value: str) -> tuple[str, ...]:
    return tuple(part for part in PurePosixPath(value).parts if part != "/")


def _component_prefix(left: str, right: str) -> bool:
    left_parts = _parts(left)
    right_parts = _parts(right)
    shorter = min(len(left_parts), len(right_parts))
    return left_parts[:shorter] == right_parts[:shorter]


def path_scopes_overlap(left: str, right: str) -> bool:
    """Return whether two repository paths overlap by whole components."""
    return _component_prefix(
        canonical_write_scope(left), canonical_write_scope(right),
    )


def artifact_scopes_overlap(left: str, right: str) -> bool:
    """Return whether absolute artifact namespaces are equal or nested."""
    canonical_left = canonical_artifact_namespace(left)
    canonical_right = canonical_artifact_namespace(right)
    assert canonical_left is not None and canonical_right is not None
    return _component_prefix(canonical_left, canonical_right)


def reservation_manifest(
    *,
    task_id: str,
    generation: int,
    base_commit: str,
    write_set: Iterable[str],
    artifact_namespace: str | None,
) -> dict[str, Any]:
    base = str(base_commit or "").strip()
    if not base:
        raise ReservationError("base_commit is required for scope reservation")
    artifact = canonical_artifact_namespace(artifact_namespace)
    if artifact is None:
        raise ReservationError(
            "artifact_namespace is required for scope reservation"
        )
    return {
        "task_id": str(task_id),
        "generation": int(generation),
        "base_commit": base,
        "write_set": list(canonical_write_set(write_set)),
        "artifact_namespace": artifact,
    }


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
