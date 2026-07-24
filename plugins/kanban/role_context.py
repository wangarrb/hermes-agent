"""Shared effective-role context rendering for interactive Kanban listeners."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_PROFILES_ROOT = Path("/home/wyr/.hermes/profiles")
DEFAULT_SHARED_SKILLS_ROOT = Path("/home/wyr/.hermes/skills")
MAX_SOURCE_BYTES = 64 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_SKILL_NAME = re.compile(r"^[A-Za-z0-9._/-]+$")


class RoleContextError(RuntimeError):
    """Role context could not be built from trusted, complete inputs."""


def _validate_name(value: str, *, kind: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or not _SAFE_NAME.fullmatch(normalized):
        raise RoleContextError(f"invalid_{kind}:{normalized!r}")
    return normalized


def _read_file_source(
    path: Path,
    *,
    allowed_root: Path,
    kind: str,
    name: str,
    expected_sha: str | None = None,
) -> dict[str, str]:
    root = allowed_root.expanduser().resolve(strict=False)
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RoleContextError(f"path_outside_allowed_root:{kind}:{name}") from exc
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise RoleContextError(f"missing_source:{kind}:{name}") from exc
    if len(payload) > MAX_SOURCE_BYTES:
        raise RoleContextError(f"source_too_large:{kind}:{name}:{len(payload)}")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise RoleContextError(f"sha_mismatch:{kind}:{name}")
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RoleContextError(f"source_not_utf8:{kind}:{name}") from exc
    return {"path": str(resolved), "sha256": digest, "content": content}


def _find_named_skill(name: str, roots: Iterable[Path]) -> tuple[Path, Path]:
    if not name or not _SAFE_SKILL_NAME.fullmatch(name) or ".." in Path(name).parts:
        raise RoleContextError(f"invalid_skill_name:{name!r}")
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=False)
        matches: set[Path] = set()
        candidates = [root / name / "SKILL.md"]
        if "/" not in name and root.is_dir():
            candidates.extend(root.rglob(f"{name}/SKILL.md"))
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            if resolved.is_file():
                matches.add(resolved)
        if len(matches) > 1:
            raise RoleContextError(
                f"ambiguous_skill:{name}:"
                + ",".join(str(path) for path in sorted(matches))
            )
        if matches:
            return next(iter(matches)), root
    raise RoleContextError(f"missing_source:task_skill:{name}")


def _git_read_control_prompt(
    contract: Mapping[str, Any],
    *,
    board: str,
    effective_role: str,
) -> tuple[str, dict[str, str]]:
    repository = Path(str(contract.get("repository") or "")).expanduser()
    if not repository.is_absolute() or not repository.is_dir():
        raise RoleContextError("invalid_workspace_contract:repository")
    repository = repository.resolve(strict=False)
    requested_commit = str(
        contract.get("control_commit") or contract.get("base_commit") or ""
    ).strip()
    if not requested_commit:
        raise RoleContextError("missing_fixed_control_commit")
    resolved = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--verify",
            f"{requested_commit}^{{commit}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if resolved.returncode != 0 or not resolved.stdout.strip():
        raise RoleContextError("missing_source:fixed_control_prompt:commit")
    control_commit = resolved.stdout.strip()
    relative_path = (
        f".hermes-kanban/{board}/{effective_role}/kanban-system-prompt.md"
    )
    shown = subprocess.run(
        ["git", "-C", str(repository), "show", f"{control_commit}:{relative_path}"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if shown.returncode != 0:
        raise RoleContextError(
            f"missing_source:fixed_control_prompt:{effective_role}"
        )
    payload = shown.stdout
    if len(payload) > MAX_SOURCE_BYTES:
        raise RoleContextError(
            f"source_too_large:fixed_control_prompt:{effective_role}:{len(payload)}"
        )
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RoleContextError(
            f"source_not_utf8:fixed_control_prompt:{effective_role}"
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    expected_sha = contract.get("project_prompt_sha256")
    if expected_sha is not None and digest != str(expected_sha):
        raise RoleContextError(
            f"sha_mismatch:fixed_control_prompt:{effective_role}"
        )
    return control_commit, {
        "path": f"{repository}@{control_commit}:{relative_path}",
        "sha256": digest,
        "content": content,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _render_text(payload: Mapping[str, Any]) -> str:
    sections = [
        "## Effective Role Context",
        (
            f"pane_profile={payload['pane_profile']} "
            f"effective_role={payload['effective_role']} "
            f"assist={str(payload['assist']).lower()} "
            f"backend={payload['backend']}"
        ),
        f"fixed_control_commit={payload['fixed_control_commit']}",
        f"role_context_manifest={payload['manifest_path']}",
        (
            "HERMES_KANBAN_CONTROL_PROMPT_SHA256="
            + payload["control_prompt_sha256"]
        ),
        "### Project Role Prompt\n" + payload["project_prompt"]["content"],
    ]
    description = payload.get("role_description")
    if description:
        sections.append("### Role Description\n" + str(description))
    for skill in payload.get("declared_skills", []):
        sections.append(
            f"### Declared Role Skill: {skill['name']}\n{skill['content']}"
        )
    for skill in payload.get("task_skills", []):
        sections.append(f"### Task Skill: {skill['name']}\n{skill['content']}")
    sections.append(
        "### Workspace Contract\n"
        + json.dumps(payload["workspace_contract"], ensure_ascii=False, sort_keys=True)
    )
    return "\n\n".join(sections).strip()


def render_effective_role_context(
    *,
    board: str,
    workspace: Path,
    pane_profile: str,
    task: Any,
    output_path: Path,
    profiles_root: Path = DEFAULT_PROFILES_ROOT,
    shared_skills_root: Path = DEFAULT_SHARED_SKILLS_ROOT,
    backend: str,
) -> str:
    """Render and emit one task-assignee-driven role context."""
    payload: dict[str, Any] = {
        "schema_version": "kanban-role-context.v1",
        "status": "blocked",
        "errors": [],
    }
    try:
        board = _validate_name(board, kind="board")
        pane_profile = _validate_name(pane_profile, kind="pane_profile")
        effective_role = _validate_name(
            str(getattr(task, "assignee", None) or pane_profile),
            kind="effective_role",
        )
        contract = getattr(task, "workspace_contract", None)
        if not isinstance(contract, dict) or not contract.get("valid", False):
            raise RoleContextError("missing_or_invalid_workspace_contract")
        payload.update(
            {
                "pane_profile": pane_profile,
                "effective_role": effective_role,
                "assist": effective_role != pane_profile,
                "backend": str(backend),
                "workspace": str(Path(workspace).resolve(strict=False)),
                "workspace_contract": dict(contract),
                "manifest_path": str(output_path.resolve(strict=False)),
            }
        )

        role_root = (profiles_root / effective_role).resolve(strict=False)
        manifest_source = _read_file_source(
            role_root / "role-context.json",
            allowed_root=role_root,
            kind="role_manifest",
            name=effective_role,
        )
        try:
            manifest = json.loads(manifest_source["content"])
        except json.JSONDecodeError as exc:
            raise RoleContextError(
                f"invalid_role_manifest:{effective_role}"
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("role") != effective_role
        ):
            raise RoleContextError(f"invalid_role_manifest:{effective_role}")
        description = manifest.get("description")
        if not isinstance(description, str) or not description.strip():
            raise RoleContextError(f"invalid_role_description:{effective_role}")

        control_commit, project_prompt = _git_read_control_prompt(
            contract, board=board, effective_role=effective_role,
        )
        declared_skills: list[dict[str, str]] = []
        declared = manifest.get("skills")
        if not isinstance(declared, list):
            raise RoleContextError(f"invalid_role_skills:{effective_role}")
        for item in declared:
            if not isinstance(item, dict):
                raise RoleContextError(f"invalid_role_skill_entry:{effective_role}")
            name = str(item.get("name") or "").strip()
            relative_path = str(item.get("path") or "").strip()
            expected_sha = str(item.get("sha256") or "").strip()
            if not name or not relative_path or not expected_sha:
                raise RoleContextError(f"missing_expected_sha:declared_skill:{name}")
            source = _read_file_source(
                role_root / relative_path,
                allowed_root=role_root / "skills",
                kind="declared_skill",
                name=name,
                expected_sha=expected_sha,
            )
            declared_skills.append({"name": name, **source})

        task_skills: list[dict[str, str]] = []
        for name in dict.fromkeys(
            str(skill).strip()
            for skill in (getattr(task, "skills", None) or [])
            if str(skill).strip()
        ):
            path, root = _find_named_skill(
                name, (role_root / "skills", shared_skills_root),
            )
            task_skills.append(
                {
                    "name": name,
                    **_read_file_source(
                        path,
                        allowed_root=root,
                        kind="task_skill",
                        name=name,
                    ),
                }
            )
        payload.update(
            {
                "status": "ready",
                "fixed_control_commit": control_commit,
                "project_prompt": project_prompt,
                "control_prompt_sha256": project_prompt["sha256"],
                "role_description": description.strip(),
                "declared_skills": declared_skills,
                "task_skills": task_skills,
                "sources": {
                    "role_manifest": {
                        "path": manifest_source["path"],
                        "sha256": manifest_source["sha256"],
                    }
                },
            }
        )
        rendered = _render_text(payload)
        payload["rendered_context"] = rendered
        _atomic_write_json(output_path, payload)
        return rendered
    except RoleContextError as exc:
        payload["errors"] = [str(exc)]
        _atomic_write_json(output_path, payload)
        raise
