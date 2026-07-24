"""Immutable Git workspace identities for Kanban worktree tasks."""

from __future__ import annotations

import json
import os
import string
import subprocess
import tempfile
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


def _resolve_ref(repository: Path, ref: str, *, label: str) -> str:
    result = _run_git(
        repository, "rev-parse", "--verify", f"{ref}^{{commit}}", check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise WorkspaceContractError(f"cannot resolve {label}: {ref}")
    return result.stdout.strip()


def _snapshot_source_refs(
    repository: Path | str,
    *,
    source_branch: str,
    upstream: str,
) -> dict[str, str]:
    repo = Path(repository).expanduser().resolve(strict=False)
    source = validate_branch_name(source_branch)
    remote = str(upstream or "").strip()
    if not remote or any(ch.isspace() for ch in remote):
        raise WorkspaceContractError(f"invalid upstream remote: {upstream!r}")
    return {
        "source_branch": source,
        "source_sha": _resolve_ref(
            repo, f"refs/heads/{source}", label="source branch",
        ),
        "upstream": remote,
        "upstream_sha": _resolve_ref(
            repo, f"refs/remotes/{remote}/{source}", label="upstream branch",
        ),
    }


def resolve_fetched_base(
    repository: Path | str,
    *,
    source_branch: str,
    upstream: str = "origin",
    base_commit: str | None = None,
) -> dict[str, str]:
    """Fetch and resolve a stable source/upstream snapshot and task base."""
    repo = Path(repository).expanduser().resolve(strict=False)
    if not repo.is_dir():
        raise WorkspaceContractError(f"repository does not exist: {repo}")
    source = validate_branch_name(source_branch)
    remote = str(upstream or "").strip()
    fetch = _run_git(
        repo, "fetch", "--no-tags", remote, source, check=False,
    )
    if fetch.returncode != 0:
        detail = (fetch.stderr or fetch.stdout or "fetch failed").strip()
        raise WorkspaceContractError(f"cannot fetch {remote}/{source}: {detail}")

    before = _snapshot_source_refs(
        repo, source_branch=source, upstream=remote,
    )
    requested_base = str(base_commit or before["upstream_sha"]).strip()
    resolved_base = _resolve_ref(repo, requested_base, label="base commit")
    reachable = _run_git(
        repo,
        "merge-base",
        "--is-ancestor",
        resolved_base,
        before["upstream_sha"],
        check=False,
    )
    if reachable.returncode != 0:
        raise WorkspaceContractError(
            f"base commit is not reachable from fetched upstream: {resolved_base}"
        )
    after = _snapshot_source_refs(
        repo, source_branch=source, upstream=remote,
    )
    if before != after:
        raise WorkspaceContractError(
            "source branch/upstream changed during resolution"
        )
    return {**after, "base_commit": resolved_base}


def generation_branch(role: str, task_id: str, generation: int) -> str:
    normalized_role = str(role or "").strip().lower()
    normalized_task = str(task_id or "").strip()
    if not normalized_role or not normalized_task:
        raise WorkspaceContractError("role and task_id are required")
    return validate_branch_name(
        f"{normalized_role}/{normalized_task}/g{int(generation)}"
    )


def _linked_worktree_identity(worktree: Path | str) -> tuple[Path, Path, Path]:
    raw = Path(worktree).expanduser()
    if not raw.is_absolute():
        raise WorkspaceContractError("worktree path must be absolute")
    path = raw.resolve(strict=False)
    if not path.is_dir():
        raise WorkspaceContractError(f"worktree does not exist: {path}")
    git_dir = Path(
        _run_git(
            path, "rev-parse", "--path-format=absolute", "--git-dir",
        ).stdout.strip()
    ).resolve(strict=False)
    common_dir = Path(
        _run_git(
            path, "rev-parse", "--path-format=absolute", "--git-common-dir",
        ).stdout.strip()
    ).resolve(strict=False)
    if git_dir == common_dir:
        raise WorkspaceContractError(f"path is not a linked worktree: {path}")
    return path, _repository_from_worktree(path), common_dir


def _require_clean(worktree: Path) -> None:
    if _run_git(worktree, "status", "--porcelain").stdout.strip():
        raise WorkspaceContractError(f"worktree must be clean: {worktree}")


def manifest_path_for(task: Any, worktree: Path | str) -> Path:
    _, _, common_dir = _linked_worktree_identity(worktree)
    return (
        common_dir
        / "hermes-kanban"
        / "workspace-contracts"
        / f"{task.id}.{CONTRACT_VERSION}.json"
    )


def _atomic_write_manifest(path: Path, contract: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(contract), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkspaceContractError(f"invalid lifecycle manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkspaceContractError(f"invalid lifecycle manifest {path}: expected object")
    return value


_LIFECYCLE_IMMUTABLE_FIELDS = (
    "version",
    "repository",
    "worktree",
    "common_dir",
    "base_commit",
    "target_branch",
    "branch",
    "task_id",
    "generation",
    "source_branch",
    "source_sha",
    "upstream",
    "upstream_sha",
    "role",
    "write_set",
    "artifact_namespace",
    "manifest_path",
)


def _manifest_mismatches(
    existing: Mapping[str, Any], expected: Mapping[str, Any],
) -> list[str]:
    return [
        field
        for field in _LIFECYCLE_IMMUTABLE_FIELDS
        if existing.get(field) != expected.get(field)
    ]


def _branch_exists(repository: Path, branch: str) -> bool:
    return _run_git(
        repository,
        "show-ref",
        "--verify",
        f"refs/heads/{branch}",
        check=False,
    ).returncode == 0


def prepare_generation_worktree(
    task: Any,
    worktree: Path | str,
    *,
    source_branch: str,
    upstream: str = "origin",
    base_commit: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Prepare one clean long-lived worktree for a task generation."""
    path, repository, common_dir = _linked_worktree_identity(worktree)
    _require_clean(path)
    task_role = str(role or getattr(task, "assignee", None) or "").strip().lower()
    expected_branch = generation_branch(task_role, task.id, task.generation)
    declared_branch = str(getattr(task, "branch_name", None) or "").strip()
    if declared_branch and declared_branch != expected_branch:
        raise WorkspaceContractError(
            f"generation branch mismatch: declared {declared_branch!r}, "
            f"expected {expected_branch!r}"
        )

    resolved = resolve_fetched_base(
        repository,
        source_branch=source_branch,
        upstream=upstream,
        base_commit=base_commit or getattr(task, "base_commit", None),
    )
    manifest_path = manifest_path_for(task, path)
    prior_contract = contract_for_task(task) or {}
    expected = {
        "version": CONTRACT_VERSION,
        "valid": True,
        "mismatches": [],
        "repository": str(repository),
        "worktree": str(path),
        "common_dir": str(common_dir),
        "base_commit": resolved["base_commit"],
        "target_branch": validate_branch_name(
            str(getattr(task, "target_branch", None) or "")
        ),
        "branch": expected_branch,
        "task_id": str(task.id),
        "generation": int(task.generation),
        "source_branch": resolved["source_branch"],
        "source_sha": resolved["source_sha"],
        "upstream": resolved["upstream"],
        "upstream_sha": resolved["upstream_sha"],
        "role": task_role,
        "write_set": prior_contract.get("write_set"),
        "artifact_namespace": prior_contract.get("artifact_namespace"),
        "manifest_path": str(manifest_path),
        "lease_released": False,
    }
    existing = _read_manifest(manifest_path)
    current_branch = _run_git(path, "branch", "--show-current").stdout.strip()
    if existing is not None:
        mismatches = _manifest_mismatches(existing, expected)
        if mismatches:
            raise WorkspaceContractError(
                "lifecycle manifest mismatch: " + ", ".join(mismatches)
            )
        if current_branch != expected_branch:
            raise WorkspaceContractError(
                f"lifecycle manifest branch ownership mismatch: {current_branch!r}"
            )
        return existing

    owned_prefix = f"{task_role}/{task.id}/g"
    if current_branch != expected_branch and not current_branch.startswith(owned_prefix):
        raise WorkspaceContractError(
            f"worktree branch ownership mismatch: {current_branch!r} is not "
            f"owned by {task_role}/{task.id}"
        )

    if current_branch == expected_branch:
        if _resolve_ref(path, "HEAD", label="worktree HEAD") != resolved["base_commit"]:
            raise WorkspaceContractError(
                "existing generation branch does not start at resolved base commit"
            )
    elif _branch_exists(repository, expected_branch):
        branch_tip = _resolve_ref(
            repository, f"refs/heads/{expected_branch}", label="generation branch",
        )
        if branch_tip != resolved["base_commit"]:
            raise WorkspaceContractError(
                "existing generation branch does not match resolved base commit"
            )
        _run_git(path, "switch", expected_branch)
    else:
        _run_git(path, "switch", "-c", expected_branch, resolved["base_commit"])

    if _run_git(path, "branch", "--show-current").stdout.strip() != expected_branch:
        raise WorkspaceContractError("failed to activate generation branch")
    if _resolve_ref(path, "HEAD", label="worktree HEAD") != resolved["base_commit"]:
        raise WorkspaceContractError("generation branch HEAD does not match resolved base")
    _require_clean(path)
    _atomic_write_manifest(manifest_path, expected)
    return expected


def _active_lifecycle_manifest(task: Any, worktree: Path | str) -> tuple[Path, dict[str, Any]]:
    path, repository, common_dir = _linked_worktree_identity(worktree)
    manifest_path = manifest_path_for(task, path)
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        raise WorkspaceContractError(f"lifecycle manifest is missing: {manifest_path}")
    expected = {
        "version": CONTRACT_VERSION,
        "repository": str(repository),
        "worktree": str(path),
        "common_dir": str(common_dir),
        "base_commit": getattr(task, "base_commit", None),
        "target_branch": getattr(task, "target_branch", None),
        "branch": getattr(task, "branch_name", None),
        "task_id": str(task.id),
        "generation": int(task.generation),
    }
    mismatches = [field for field, value in expected.items() if manifest.get(field) != value]
    if mismatches:
        raise WorkspaceContractError(
            "lifecycle manifest mismatch: " + ", ".join(mismatches)
        )
    return path, manifest


def freeze_delivery(task: Any, worktree: Path | str) -> dict[str, Any]:
    """Freeze a clean committed delivery and record commit/tree identities."""
    path, manifest = _active_lifecycle_manifest(task, worktree)
    _require_clean(path)
    current_branch = _run_git(path, "branch", "--show-current").stdout.strip()
    if current_branch != manifest["branch"]:
        raise WorkspaceContractError(
            f"delivery branch mismatch: {current_branch!r}"
        )
    delivery_commit = _resolve_ref(path, "HEAD", label="delivery commit")
    tree_result = _run_git(
        path, "rev-parse", "--verify", "HEAD^{tree}", check=False,
    )
    if tree_result.returncode != 0 or not tree_result.stdout.strip():
        raise WorkspaceContractError("cannot resolve delivery tree")
    delivery_tree = tree_result.stdout.strip()
    frozen = {
        **manifest,
        "delivery_commit": delivery_commit,
        "delivery_tree": delivery_tree,
        "frozen": True,
    }
    _atomic_write_manifest(Path(manifest["manifest_path"]), frozen)
    return frozen


def release_worktree_lease(task: Any, worktree: Path | str) -> dict[str, Any]:
    """Detach a frozen long-lived worktree without deleting its delivery branch."""
    path, manifest = _active_lifecycle_manifest(task, worktree)
    _require_clean(path)
    delivery_commit = str(manifest.get("delivery_commit") or "").strip()
    delivery_branch = str(manifest.get("branch") or "").strip()
    if not manifest.get("frozen") or not delivery_commit:
        raise WorkspaceContractError("delivery must be frozen before releasing lease")
    if _run_git(path, "branch", "--show-current").stdout.strip() != delivery_branch:
        raise WorkspaceContractError("delivery branch is not active in worktree")
    _run_git(path, "switch", "--detach", delivery_commit)
    persisted_branch = _resolve_ref(
        path, f"refs/heads/{delivery_branch}", label="delivery branch",
    )
    if persisted_branch != delivery_commit:
        raise WorkspaceContractError("delivery branch changed while releasing lease")
    released = {
        **manifest,
        "lease_released": True,
        "released_head": delivery_commit,
    }
    _atomic_write_manifest(Path(manifest["manifest_path"]), released)
    return released
