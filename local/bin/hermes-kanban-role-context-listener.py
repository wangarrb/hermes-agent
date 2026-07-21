#!/usr/bin/env python3
"""Task-scoped assist-role context overlay for the Hermes Kanban watcher.

This is a local control-plane extension selected by ``start-kanban.sh``.  It
wraps ``kanban_db.build_worker_context`` after a task is claimed, so the
existing upstream listener and prompt writer remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROFILES_ROOT = Path("/home/wyr/.hermes/profiles")
DEFAULT_SHARED_SKILLS_ROOT = Path("/home/wyr/.hermes/skills")
DEFAULT_HERMES_KANBAN_ROOT = Path("/home/wyr/.hermes/hermes-agent/plugins/kanban")
MAX_SOURCE_BYTES = 64 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_SKILL_NAME = re.compile(r"^[A-Za-z0-9._/-]+$")

class RoleContextError(RuntimeError):
    """A fail-visible, non-secret role-context resolution error."""


def _validate_name(value: str, *, kind: str) -> str:
    value = (value or "").strip()
    if not value or not _SAFE_NAME.fullmatch(value):
        raise RoleContextError(f"invalid_{kind}:{value!r}")
    return value


def _source_record(path: Path, *, allowed_root: Path, kind: str, name: str) -> tuple[str, str]:
    root = allowed_root.expanduser().resolve(strict=False)
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RoleContextError(f"path_outside_allowed_root:{kind}:{name}") from exc
    if not resolved.is_file():
        raise RoleContextError(f"missing_or_unreadable_source:{kind}:{name}")
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise RoleContextError(f"missing_or_unreadable_source:{kind}:{name}") from exc
    if len(payload) > MAX_SOURCE_BYTES:
        raise RoleContextError(f"source_too_large:{kind}:{name}:{len(payload)}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RoleContextError(f"source_not_utf8:{kind}:{name}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    marker = (
        f"source_kind={kind} source_name={name} source_path={resolved} "
        f"source_sha256={digest}"
    )
    return marker, text


def _find_named_skill(name: str, roots: Iterable[Path]) -> tuple[Path, Path]:
    if not name or not _SAFE_SKILL_NAME.fullmatch(name) or ".." in Path(name).parts:
        raise RoleContextError(f"invalid_skill_name:{name!r}")

    # Roots are ordered by authority: the effective target profile shadows the
    # shared skill root. Ambiguity is an error within one authority tier; a
    # lower-tier duplicate is only a fallback and is never read.
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=False)
        matches: set[Path] = set()
        if "/" in name:
            candidates = [root / name / "SKILL.md"]
        else:
            candidates = [root / name / "SKILL.md"]
            if root.is_dir():
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
            locations = ",".join(str(path) for path in sorted(matches))
            raise RoleContextError(f"ambiguous_skill:{name}:{locations}")
        if matches:
            return next(iter(matches)), root

    raise RoleContextError(f"unknown_skill:{name}")


def _append_error(errors: list[str], exc: Exception) -> None:
    errors.append(str(exc).replace("\n", " "))


def _unique_skill_record(
    path: Path,
    *,
    allowed_root: Path,
    kind: str,
    name: str,
    loaded_paths: set[Path],
    loaded_hashes: set[str],
) -> tuple[str, str] | None:
    """Read one skill unless its canonical path or content was already loaded."""
    marker, text = _source_record(path, allowed_root=allowed_root, kind=kind, name=name)
    resolved = path.expanduser().resolve(strict=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if resolved in loaded_paths or digest in loaded_hashes:
        return None
    loaded_paths.add(resolved)
    loaded_hashes.add(digest)
    return marker, text


def render_role_context(
    *,
    board: str,
    workspace: Path,
    pane_profile: str,
    task_assignee: str,
    task_skills: Iterable[str] | None,
    profiles_root: Path = DEFAULT_PROFILES_ROOT,
    shared_skills_root: Path = DEFAULT_SHARED_SKILLS_ROOT,
    backend: str = "hermes",
) -> str:
    """Render safe, task-scoped role/skill context with traceable sources."""
    errors: list[str] = []
    sections: list[str] = []
    loaded_skill_paths: set[Path] = set()
    loaded_skill_hashes: set[str] = set()

    try:
        board = _validate_name(board, kind="board")
        pane_profile = _validate_name(pane_profile, kind="pane_profile")
        task_assignee = _validate_name(task_assignee, kind="task_assignee")
    except RoleContextError as exc:
        return f"BLOCK_ASSIST_ROLE_CONTEXT\n- {exc}"

    skills = list(
        dict.fromkeys(str(skill).strip() for skill in (task_skills or []) if str(skill).strip())
    )
    is_assist = task_assignee != pane_profile
    if not is_assist and not skills:
        return ""

    effective_profile_root = profiles_root / task_assignee
    effective_skills_root = effective_profile_root / "skills"

    sections.append(
        "## Task-Scoped Role Context\n"
        f"pane_profile={pane_profile} effective_role={task_assignee} "
        f"assist={str(is_assist).lower()} backend={backend}"
    )

    if is_assist:
        manifest_path = effective_profile_root / "role-context.json"
        try:
            marker, manifest_text = _source_record(
                manifest_path,
                allowed_root=effective_profile_root,
                kind="role_manifest",
                name=task_assignee,
            )
            manifest = json.loads(manifest_text)
            if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
                raise RoleContextError(f"invalid_role_manifest_schema:{task_assignee}")
            if manifest.get("role") != task_assignee:
                raise RoleContextError(f"role_manifest_mismatch:{task_assignee}")
            description = manifest.get("description")
            if not isinstance(description, str) or not description.strip() or len(description) > 2000:
                raise RoleContextError(f"invalid_role_description:{task_assignee}")
            sections.append(f"### Safe Profile Description\n{marker}\n{description.strip()}")

            declared = manifest.get("skills", [])
            if not isinstance(declared, list):
                raise RoleContextError(f"invalid_role_skills:{task_assignee}")
            for item in declared:
                try:
                    if not isinstance(item, dict):
                        raise RoleContextError(f"invalid_role_skill_entry:{task_assignee}")
                    skill_name = str(item.get("name") or "").strip()
                    relative_path = str(item.get("path") or "").strip()
                    if not skill_name or not relative_path:
                        raise RoleContextError(f"invalid_role_skill_entry:{task_assignee}")
                    if backend != "hermes":
                        sections.append(
                            f"WARN_ASSIST_ROLE_SKILL_CAPABILITY backend={backend} "
                            f"skill={skill_name} content_not_embedded"
                        )
                        continue
                    skill_path = effective_profile_root / relative_path
                    skill_record = _unique_skill_record(
                        skill_path,
                        allowed_root=effective_skills_root,
                        kind="role_skill",
                        name=skill_name,
                        loaded_paths=loaded_skill_paths,
                        loaded_hashes=loaded_skill_hashes,
                    )
                    if skill_record is not None:
                        skill_marker, skill_text = skill_record
                        sections.append(
                            f"### Declared Role Skill: {skill_name}\n{skill_marker}\n{skill_text}"
                        )
                except (RoleContextError, OSError) as exc:
                    _append_error(errors, exc)
        except (RoleContextError, OSError, json.JSONDecodeError) as exc:
            _append_error(errors, exc)

        project_prompt = (
            workspace
            / ".hermes-kanban"
            / board
            / task_assignee
            / "kanban-system-prompt.md"
        )
        try:
            prompt_root = workspace / ".hermes-kanban" / board / task_assignee
            marker, prompt_text = _source_record(
                project_prompt,
                allowed_root=prompt_root,
                kind="project_role_prompt",
                name=task_assignee,
            )
            sections.append(f"### Project Role Prompt\n{marker}\n{prompt_text}")
        except RoleContextError as exc:
            if "missing_or_unreadable_source" in str(exc):
                errors.append(f"missing_project_role_prompt:{task_assignee}")
            else:
                _append_error(errors, exc)

    if skills:
        for skill_name in skills:
            try:
                if backend != "hermes":
                    sections.append(
                        f"WARN_TASK_SKILL_CAPABILITY backend={backend} "
                        f"skill={skill_name} content_not_embedded"
                    )
                    continue
                skill_path, allowed_root = _find_named_skill(
                    skill_name,
                    (effective_skills_root, shared_skills_root),
                )
                skill_record = _unique_skill_record(
                    skill_path,
                    allowed_root=allowed_root,
                    kind="task_skill",
                    name=skill_name,
                    loaded_paths=loaded_skill_paths,
                    loaded_hashes=loaded_skill_hashes,
                )
                if skill_record is not None:
                    marker, skill_text = skill_record
                    sections.append(f"### Task Skill: {skill_name}\n{marker}\n{skill_text}")
            except RoleContextError as exc:
                _append_error(errors, exc)

    if errors:
        label = "BLOCK_ASSIST_ROLE_CONTEXT" if is_assist else "BLOCK_TASK_SKILL_CONTEXT"
        sections.insert(0, label + "\n" + "\n".join(f"- {error}" for error in errors))
    return "\n\n".join(sections).strip()


def install_worker_context_overlay(
    kb_module: Any,
    *,
    pane_profile: str,
    board: str,
    workspace: Path,
    profiles_root: Path = DEFAULT_PROFILES_ROOT,
    shared_skills_root: Path = DEFAULT_SHARED_SKILLS_ROOT,
    backend: str = "hermes",
) -> None:
    """Wrap the real worker-context builder so Task.assignee/skills drive context."""
    if getattr(kb_module, "_task_role_context_overlay_installed", False):
        return
    original_builder = kb_module.build_worker_context

    def wrapped_build_worker_context(conn: Any, task_id: str) -> str:
        base_context = original_builder(conn, task_id)
        task = kb_module.get_task(conn, task_id)
        if task is None:
            return base_context + "\n\nBLOCK_ASSIST_ROLE_CONTEXT\n- missing_claimed_task"
        task_assignee = getattr(task, "assignee", None) or pane_profile
        task_skills = getattr(task, "skills", None) or []
        role_context = render_role_context(
            board=board,
            workspace=workspace,
            pane_profile=pane_profile,
            task_assignee=task_assignee,
            task_skills=task_skills,
            profiles_root=profiles_root,
            shared_skills_root=shared_skills_root,
            backend=backend,
        )
        if not role_context:
            return base_context
        return base_context + "\n\n" + role_context

    wrapped_build_worker_context._role_context_original = original_builder  # type: ignore[attr-defined]
    kb_module.build_worker_context = wrapped_build_worker_context
    kb_module._task_role_context_overlay_installed = True


def _arg_value(argv: list[str], flag: str, default: str) -> str:
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return default


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    plugin_root = DEFAULT_HERMES_KANBAN_ROOT
    listener_root = plugin_root / "hermes_listener"
    for path in (str(plugin_root), str(listener_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import hermes_kanban_interactive as upstream_listener

    pane_profile = _arg_value(args, "--profile", "implementer")
    board = _arg_value(args, "--board", "egomotion4d")
    workspace = Path(_arg_value(args, "--workspace", str(Path.cwd()))).expanduser().resolve()
    install_worker_context_overlay(
        upstream_listener.kb,
        pane_profile=pane_profile,
        board=board,
        workspace=workspace,
        backend="hermes",
    )
    return upstream_listener.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
