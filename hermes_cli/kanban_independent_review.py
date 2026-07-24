"""Validation for implementer-owned, non-author delivery reviews."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


SCHEMA = "independent_review.v1"
OPT_OUT_TASK_TYPES = frozenset({"plan", "research", "readonly"})
EMPTY_DIFF_TREE_SHA256 = hashlib.sha256(b"").hexdigest()
REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "generation",
        "base_sha",
        "delivery_sha",
        "diff_tree_sha",
        "reviewer_session_id",
        "author_session_ids",
        "control_prompt_sha",
        "commands",
        "artifact_hashes",
        "verdict",
    }
)


class IndependentReviewError(ValueError):
    """Raised when an implementer delivery lacks a valid independent review."""


def review_required(task: Any, metadata: Mapping[str, Any] | None) -> bool:
    """Return whether this completion is subject to the implementer gate."""
    if str(getattr(task, "assignee", "") or "").strip().lower() != "implementer":
        return False
    data = metadata or {}
    requested = data.get("independent_review_required")
    if requested is False:
        task_type = str(data.get("task_type") or "").strip().lower()
        if task_type not in OPT_OUT_TASK_TYPES:
            raise IndependentReviewError(
                "implementer delivery cannot opt out of independent review; "
                "only plan, research, or readonly tasks may opt out"
            )
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(paths: Sequence[Any]) -> dict[str, str]:
    """Hash the exact completion artifacts using canonical absolute paths."""
    result: dict[str, str] = {}
    for raw in paths:
        path = Path(str(raw)).expanduser().resolve(strict=False)
        if not path.is_file():
            raise IndependentReviewError(f"artifact is unavailable: {path}")
        result[str(path)] = _sha256_file(path)
    return result


def _run_git(repository: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or b"git command failed").decode(
            errors="replace"
        ).strip()
        raise IndependentReviewError(f"cannot compute delivery diff tree: {detail}")
    return proc.stdout


def expected_binding(task: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Compute authoritative identities that an independent review must bind."""
    base_sha = str(getattr(task, "base_commit", None) or "").strip()
    if not base_sha:
        raise IndependentReviewError("implementer delivery base SHA is missing")

    record = getattr(task, "delivery", None)
    delivery_sha = str(getattr(record, "delivery_sha", None) or base_sha).strip()
    if record is not None and not getattr(record, "delivery_sha", None):
        raise IndependentReviewError("final implementer delivery SHA is missing")

    if delivery_sha == base_sha:
        diff_tree_sha = EMPTY_DIFF_TREE_SHA256
    else:
        contract = getattr(record, "workspace_contract", None) or {}
        repository = Path(str(contract.get("repository") or ""))
        if not repository.is_absolute() or not repository.is_dir():
            raise IndependentReviewError("delivery repository identity is invalid")
        diff_bytes = _run_git(
            repository,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            base_sha,
            delivery_sha,
        )
        diff_tree_sha = hashlib.sha256(diff_bytes).hexdigest()

    control_prompt_sha = os.environ.get(
        "HERMES_KANBAN_CONTROL_PROMPT_SHA256", ""
    ).strip()
    if not control_prompt_sha:
        raise IndependentReviewError(
            "authoritative control prompt SHA is missing from the worker environment"
        )
    paths = metadata.get("artifacts") or []
    if not isinstance(paths, (list, tuple)):
        raise IndependentReviewError("completion artifacts must be a list")
    return {
        "task_id": str(task.id),
        "generation": int(task.generation),
        "base_sha": base_sha,
        "delivery_sha": delivery_sha,
        "diff_tree_sha": diff_tree_sha,
        "control_prompt_sha": control_prompt_sha,
        "artifact_hashes": artifact_hashes(paths),
    }


def _load_review(path_value: Any) -> tuple[Path, dict[str, Any], str]:
    if not path_value:
        raise IndependentReviewError("independent review artifact is required")
    path = Path(str(path_value)).expanduser().resolve(strict=False)
    if not path.is_file():
        raise IndependentReviewError(f"independent review artifact is required: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentReviewError(f"invalid independent review artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise IndependentReviewError("independent review artifact must be a JSON object")
    return path, payload, hashlib.sha256(raw).hexdigest()


def validate_completion(
    task: Any,
    metadata: MutableMapping[str, Any] | None,
) -> MutableMapping[str, Any] | None:
    """Validate and bind review evidence before a task may become done."""
    if not review_required(task, metadata):
        return metadata
    if metadata is None:
        raise IndependentReviewError("independent review artifact is required")

    _path, payload, review_sha = _load_review(
        metadata.get("independent_review_artifact")
    )
    missing = sorted(REQUIRED_FIELDS.difference(payload))
    if missing:
        raise IndependentReviewError(
            "independent review artifact missing fields: " + ", ".join(missing)
        )
    if payload.get("schema") != SCHEMA:
        raise IndependentReviewError("unsupported independent review schema")
    if payload.get("verdict") != "PASS":
        raise IndependentReviewError("independent review verdict must be PASS")

    expected = expected_binding(task, metadata)
    labels = {
        "task_id": "task",
        "generation": "generation",
        "base_sha": "base",
        "delivery_sha": "delivery",
        "diff_tree_sha": "diff tree",
        "control_prompt_sha": "control prompt",
        "artifact_hashes": "artifact",
    }
    for field, label in labels.items():
        if payload.get(field) != expected[field]:
            raise IndependentReviewError(f"independent review {label} binding mismatch")

    reviewer = str(payload.get("reviewer_session_id") or "").strip()
    authors = payload.get("author_session_ids")
    if not reviewer or not isinstance(authors, list) or not authors:
        raise IndependentReviewError("independent review session identities are incomplete")
    author_ids = {str(value).strip() for value in authors if str(value).strip()}
    current_author = os.environ.get("HERMES_SESSION_ID", "").strip()
    if current_author and current_author not in author_ids:
        raise IndependentReviewError("current author session is absent from review artifact")
    if reviewer in author_ids:
        raise IndependentReviewError("independent review must use a non-author session")
    commands = payload.get("commands")
    if not isinstance(commands, list) or not any(str(item).strip() for item in commands):
        raise IndependentReviewError("independent review commands are required")

    metadata["independent_review"] = payload
    metadata["independent_review_sha256"] = review_sha
    return metadata
