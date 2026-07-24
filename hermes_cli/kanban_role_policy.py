"""Board-neutral active and historical Kanban role policy."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "kanban-role-policy.v1"
READ_ONLY_OPERATIONS = frozenset({"show", "list", "export"})


class RolePolicyError(ValueError):
    """Base role policy failure."""


class RolePolicyConfigurationError(RolePolicyError):
    """An enforced policy cannot be loaded or verified."""


class RolePolicyDenied(RolePolicyError):
    """A role lifecycle policy denied an operation."""


@dataclass(frozen=True)
class RolePolicy:
    path: Path
    schema_version: str
    content_sha: str
    active_roles: frozenset[str]
    historical_roles: frozenset[str]


_CACHE: dict[tuple[str, int, str], RolePolicy] = {}
_CACHE_LOCK = threading.RLock()


def canonical_role(value: Any) -> str | None:
    role = str(value or "").strip().lower()
    return role or None


def _role_set(value: Any, *, field: str) -> frozenset[str]:
    if not isinstance(value, list):
        raise RolePolicyConfigurationError(f"{field} must be a JSON array")
    roles = frozenset(
        role for role in (canonical_role(item) for item in value) if role
    )
    if len(roles) != len(value):
        raise RolePolicyConfigurationError(
            f"{field} contains blank or duplicate roles"
        )
    return roles


def load_role_policy(
    path: Path | str,
    *,
    expected_sha: str | None = None,
    expected_schema_version: str | None = None,
) -> RolePolicy:
    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute():
        raise RolePolicyConfigurationError("role policy path must be absolute")
    canonical_path = raw_path.resolve(strict=False)
    try:
        content = canonical_path.read_bytes()
        stat = canonical_path.stat()
    except OSError as exc:
        raise RolePolicyConfigurationError(
            f"role policy is missing or unreadable: {canonical_path}: {exc}"
        ) from exc
    content_sha = hashlib.sha256(content).hexdigest()
    if expected_sha is not None and content_sha != str(expected_sha):
        raise RolePolicyConfigurationError(
            f"role policy SHA mismatch: expected {expected_sha}, actual {content_sha}"
        )
    cache_key = (str(canonical_path), stat.st_mtime_ns, content_sha)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            if (
                expected_schema_version is not None
                and cached.schema_version != expected_schema_version
            ):
                raise RolePolicyConfigurationError(
                    "role policy schema version does not match board metadata"
                )
            return cached
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RolePolicyConfigurationError(
            f"corrupt role policy {canonical_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RolePolicyConfigurationError("corrupt role policy: expected JSON object")
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != SCHEMA_VERSION:
        raise RolePolicyConfigurationError(
            f"unsupported role policy schema version: {schema_version or '<missing>'}"
        )
    if (
        expected_schema_version is not None
        and schema_version != expected_schema_version
    ):
        raise RolePolicyConfigurationError(
            "role policy schema version does not match board metadata"
        )
    active = _role_set(payload.get("active_roles"), field="active_roles")
    historical = _role_set(
        payload.get("historical_roles"), field="historical_roles",
    )
    overlap = active & historical
    if overlap:
        raise RolePolicyConfigurationError(
            "roles cannot be both active and historical: " + ", ".join(sorted(overlap))
        )
    policy = RolePolicy(
        path=canonical_path,
        schema_version=schema_version,
        content_sha=content_sha,
        active_roles=active,
        historical_roles=historical,
    )
    with _CACHE_LOCK:
        for key in [key for key in _CACHE if key[0] == str(canonical_path)]:
            _CACHE.pop(key, None)
        _CACHE[cache_key] = policy
    return policy


def enforced_policy_from_metadata(
    metadata: Mapping[str, Any],
) -> RolePolicy | None:
    if not metadata.get("role_policy_enforced"):
        return None
    path = metadata.get("role_policy_path")
    schema_version = metadata.get("role_policy_schema_version")
    content_sha = metadata.get("role_policy_content_sha")
    if not path or not schema_version or not content_sha:
        raise RolePolicyConfigurationError(
            "enforced role policy metadata is incomplete"
        )
    return load_role_policy(
        str(path),
        expected_sha=str(content_sha),
        expected_schema_version=str(schema_version),
    )


def assert_operation_allowed(
    policy: RolePolicy | None,
    operation: str,
    *,
    actor_role: str | None = None,
    target_roles: Iterable[str | None] = (),
) -> None:
    if policy is None or operation in READ_ONLY_OPERATIONS:
        return
    actor = canonical_role(actor_role)
    if actor in policy.historical_roles:
        raise RolePolicyDenied(
            f"actor role {actor!r} is historical and cannot perform {operation}"
        )
    for raw_role in target_roles:
        role = canonical_role(raw_role)
        if role is None:
            continue
        if role in policy.historical_roles:
            raise RolePolicyDenied(
                f"role {role!r} is historical and read-only; {operation} is denied"
            )
        if role not in policy.active_roles:
            raise RolePolicyDenied(
                f"role {role!r} is not active; {operation} is denied"
            )


def activated_metadata(
    metadata: Mapping[str, Any],
    policy: RolePolicy,
) -> dict[str, Any]:
    result = dict(metadata)
    result.pop("db_path", None)
    result.update(
        {
            "role_policy_enforced": True,
            "role_policy_path": str(policy.path),
            "role_policy_schema_version": policy.schema_version,
            "role_policy_content_sha": policy.content_sha,
        }
    )
    return result


def atomic_write_metadata(path: Path | str, metadata: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(metadata), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
