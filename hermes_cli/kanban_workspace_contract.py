"""Immutable Git workspace identities for Kanban worktree tasks."""

from __future__ import annotations

import json
import string
import subprocess
from pathlib import Path
from typing import Any, Mapping


CONTRACT_VERSION = "workspace_contract.v1"
_BRANCH_FIELDS = frozenset({"task_id", "generation", "assignee"})


class WorkspaceContractError(ValueError):
    """A task's declared repository identity does not match its worktree."""


def _run_git(
    path: Path, *args: str, check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkspaceContractError(f"git command failed for {path}: {exc}") from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "git command failed").strip()
        raise WorkspaceContractError(detail)
    return result


def validate_branch_name(branch: str) -> str:
    """Return a valid Git branch name or raise a stable user-facing error."""
    value = str(branch or "").strip()
    if not value:
        raise WorkspaceContractError("invalid git branch: branch is empty")
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", value],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise WorkspaceContractError(f"invalid git branch: {value!r}")
    return value


def render_branch_template(
    template: str,
    *,
    task_id: str,
    generation: int,
    assignee: str | None,
) -> str:
    """Resolve the safe task identity placeholders in a branch template."""
    formatter = string.Formatter()
    try:
        parts = list(formatter.parse(str(template or "")))
    except ValueError as exc:
        raise WorkspaceContractError(f"invalid branch template: {exc}") from exc
    for _, field_name, format_spec, conversion in parts:
        if field_name is None:
            continue
        if field_name not in _BRANCH_FIELDS:
            raise WorkspaceContractError(
                f"unknown branch placeholder {{{field_name}}}; allowed: "
                "{task_id}, {generation}, {assignee}"
            )
        if format_spec or conversion:
            raise WorkspaceContractError(
                f"branch placeholder {{{field_name}}} does not allow formatting"
            )
    values = {
        "task_id": str(task_id),
        "generation": str(int(generation)),
        "assignee": str(assignee or "unassigned").strip().lower(),
    }
    try:
        rendered = str(template).format_map(values)
    except (KeyError, ValueError) as exc:
        raise WorkspaceContractError(f"invalid branch template: {exc}") from exc
    return validate_branch_name(rendered)


def _repository_from_worktree(path: Path) -> Path:
    result = _run_git(path, "worktree", "list", "--porcelain")
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            return Path(line.removeprefix("worktree ")).expanduser().resolve()
    raise WorkspaceContractError(f"cannot resolve repository for worktree {path}")


def resolve_workspace_contract(task: Any, workspace: Path | str) -> dict[str, Any]:
    """Validate a linked worktree and return its immutable task identity."""
    raw_path = Path(workspace).expanduser()
    if not raw_path.is_absolute():
        raise WorkspaceContractError("worktree path must be absolute")
    path = raw_path.resolve(strict=False)
    if not path.is_dir():
        raise WorkspaceContractError(f"worktree does not exist: {path}")

    git_dir_text = _run_git(
        path, "rev-parse", "--path-format=absolute", "--git-dir",
    ).stdout.strip()
    common_dir_text = _run_git(
        path, "rev-parse", "--path-format=absolute", "--git-common-dir",
    ).stdout.strip()
    git_dir = Path(git_dir_text).expanduser().resolve(strict=False)
    common_dir = Path(common_dir_text).expanduser().resolve(strict=False)
    if git_dir == common_dir:
        raise WorkspaceContractError(f"path is not a linked worktree: {path}")

    status = _run_git(path, "status", "--porcelain").stdout
    if status.strip():
        raise WorkspaceContractError(f"worktree must be clean: {path}")

    actual_branch = _run_git(path, "branch", "--show-current").stdout.strip()
    if not actual_branch:
        raise WorkspaceContractError(f"worktree has no attached branch: {path}")
    expected_branch = validate_branch_name(getattr(task, "branch_name", None) or "")
    if actual_branch != expected_branch:
        raise WorkspaceContractError(
            f"branch mismatch: expected {expected_branch!r}, actual {actual_branch!r}"
        )

    declared_base = str(getattr(task, "base_commit", None) or "").strip()
    if not declared_base:
        raise WorkspaceContractError("base commit is required for workspace contract")
    resolved_base_result = _run_git(
        path, "rev-parse", "--verify", f"{declared_base}^{{commit}}", check=False,
    )
    if resolved_base_result.returncode != 0:
        raise WorkspaceContractError(
            f"base commit is not reachable in repository: {declared_base}"
        )
    resolved_base = resolved_base_result.stdout.strip()
    ancestor = _run_git(
        path, "merge-base", "--is-ancestor", resolved_base, "HEAD", check=False,
    )
    if ancestor.returncode != 0:
        raise WorkspaceContractError(
            f"base commit is not an ancestor of worktree HEAD: {resolved_base}"
        )

    target_branch = validate_branch_name(
        str(getattr(task, "target_branch", None) or "")
    )
    return {
        "version": CONTRACT_VERSION,
        "valid": True,
        "mismatches": [],
        "repository": str(_repository_from_worktree(path)),
        "worktree": str(path),
        "common_dir": str(common_dir),
        "base_commit": resolved_base,
        "target_branch": target_branch,
        "branch": actual_branch,
        "task_id": str(task.id),
        "generation": int(task.generation),
        "write_set": None,
        "artifact_namespace": None,
    }


def parse_contract(value: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else dict(value)
    except (TypeError, ValueError) as exc:
        raise WorkspaceContractError(f"invalid stored workspace contract: {exc}") from exc
    if not isinstance(parsed, dict):
        raise WorkspaceContractError("invalid stored workspace contract: expected object")
    return parsed


def contract_for_task(task: Any) -> dict[str, Any] | None:
    """Return the stored contract annotated against the task's current identity."""
    contract = parse_contract(getattr(task, "workspace_contract_json", None))
    if contract is None:
        return None
    result = dict(contract)
    mismatches: list[str] = []
    expected = {
        "version": CONTRACT_VERSION,
        "task_id": str(task.id),
        "generation": int(task.generation),
        "branch": getattr(task, "branch_name", None),
        "base_commit": getattr(task, "base_commit", None),
        "target_branch": getattr(task, "target_branch", None),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            mismatches.append(field)
    workspace_path = getattr(task, "workspace_path", None)
    if workspace_path and result.get("worktree") != str(
        Path(workspace_path).expanduser().resolve(strict=False)
    ):
        mismatches.append("worktree")
    result["valid"] = not mismatches
    result["mismatches"] = mismatches
    result.setdefault("write_set", None)
    result.setdefault("artifact_namespace", None)
    return result


def validate_or_resolve_contract(task: Any, workspace: Path | str) -> dict[str, Any]:
    """Validate an existing contract, or create the first contract for a task."""
    stored = contract_for_task(task)
    if stored is not None and not stored["valid"]:
        raise WorkspaceContractError(
            "stored workspace contract mismatch: " + ", ".join(stored["mismatches"])
        )
    resolved = resolve_workspace_contract(task, workspace)
    if stored is not None:
        identity_fields = (
            "version", "repository", "worktree", "common_dir", "base_commit",
            "target_branch", "branch", "task_id", "generation",
        )
        changed = [
            field
            for field in identity_fields
            if stored.get(field) != resolved.get(field)
        ]
        if changed:
            raise WorkspaceContractError(
                "workspace contract identity mismatch: " + ", ".join(changed)
            )
    return resolved


def dumps_contract(contract: Mapping[str, Any]) -> str:
    return json.dumps(dict(contract), ensure_ascii=False, sort_keys=True)
