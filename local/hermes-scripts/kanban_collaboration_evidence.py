#!/usr/bin/env python3
"""Build a deterministic source snapshot for the Egomotion4D Kanban model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


TERMINAL_MARKER = "END_KANBAN_COLLABORATION_EVIDENCE"
_HEADING_RE = re.compile(r"^(#{1,6})\s+.*$")
_SHELL_FUNCTION_RE = re.compile(r"(?m)^[ \t]*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\(\)[ \t]*\{")
_HEREDOC_RE = re.compile(
    r"<<(?P<strip>-?)[ \t]*(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)


def _normalized_text(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    return raw, raw.decode("utf-8").replace("\r\n", "\n")


def _markdown_heading(text: str, heading: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.rstrip("\n") == heading]
    if not matches:
        return []
    depth = len(heading) - len(heading.lstrip("#"))
    extracted: list[str] = []
    for start in matches:
        end = len(lines)
        for index in range(start + 1, len(lines)):
            candidate = _HEADING_RE.match(lines[index].rstrip("\n"))
            if candidate and len(candidate.group(1)) <= depth:
                end = index
                break
        extracted.append("".join(lines[start:end]))
    return extracted


def _symbol_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def _python_symbol(text: str, symbol: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"invalid Python source for selector {symbol!r}: {exc.msg}") from exc

    nodes: list[ast.AST] = []
    if "." in symbol:
        class_name, method_name = symbol.split(".", 1)
        if "." in method_name:
            raise ValueError(f"unsupported Python symbol {symbol!r}")
        for candidate in tree.body:
            if isinstance(candidate, ast.ClassDef) and candidate.name == class_name:
                nodes.extend(method
                    for method in candidate.body
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name == method_name
                )
    else:
        nodes = [
            candidate
            for candidate in tree.body
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
            and candidate.name == symbol
        ]

    lines = text.splitlines(keepends=True)
    extracted: list[str] = []
    for node in nodes:
        start = _symbol_start(node) - 1
        extracted.append("".join(lines[start : node.end_lineno]))
    return extracted


def _skip_heredoc(text: str, opening: int) -> int:
    match = _HEREDOC_RE.match(text, opening)
    if match is None:
        raise ValueError("unsupported shell heredoc")
    header_end = text.find("\n", match.end())
    if header_end == -1 or "<<" in text[match.end() : header_end]:
        raise ValueError("unsupported shell heredoc")
    delimiter = match.group("delimiter")
    strip_tabs = bool(match.group("strip"))
    position = header_end + 1
    while position <= len(text):
        line_end = text.find("\n", position)
        if line_end == -1:
            line_end = len(text)
        line = text[position:line_end]
        if (line.lstrip("\t") if strip_tabs else line) == delimiter:
            return line_end if line_end == len(text) else line_end + 1
        if line_end == len(text):
            break
        position = line_end + 1
    raise ValueError("unterminated shell heredoc")


def _shell_closing_brace(text: str, opening: int) -> int:
    depth = 0
    escaped = False
    comment = False
    contexts: list[dict[str, str | None]] = [{"quote": None, "command": None}]
    index = opening
    while index < len(text):
        char = text[index]
        context = contexts[-1]
        quote = context["quote"]
        if comment:
            if char == "\n":
                comment = False
            index += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == "`" and quote == '"':
                raise ValueError("unsupported shell construct: backticks")
            elif char == quote:
                context["quote"] = None
            elif char == "$" and quote == '"' and text[index + 1 : index + 2] == "(":
                contexts.append({"quote": None, "command": "$("})
                index += 2
                continue
            index += 1
            continue
        if char == "$" and text[index + 1 : index + 2] == "(":
            contexts.append({"quote": None, "command": "$("})
            index += 2
            continue
        if char == "`":
            raise ValueError("unsupported shell construct: backticks")
        if char == "\\":
            if index + 1 >= len(text):
                raise ValueError("unsupported shell escape")
            index += 2
            continue
        if char == "<" and text[index + 1 : index + 2] == "<":
            index = _skip_heredoc(text, index)
            continue
        if char in ("'", '"'):
            context["quote"] = char
        elif char == "#" and (index == 0 or text[index - 1].isspace()):
            comment = True
        elif char == ")" and context["command"] == "$(":
            contexts.pop()
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                newline = text.find("\n", index)
                return len(text) if newline == -1 else newline + 1
        index += 1
    raise ValueError("unclosed shell function body")


def _shell_function(text: str, name: str) -> list[str]:
    extracted = []
    for match in _SHELL_FUNCTION_RE.finditer(text):
        if match.group("name") == name:
            extracted.append(text[match.start() : _shell_closing_brace(text, match.end() - 1)])
    return extracted


def _resolve_selector(text: str, selector: dict[str, Any]) -> str:
    kind = selector.get("kind")
    if kind == "whole_file":
        matches = [text]
    else:
        value = selector.get("value")
        if not isinstance(value, str) or not value:
            raise ValueError(f"selector {kind!r} requires a non-empty value")
        if kind == "markdown_heading":
            matches = _markdown_heading(text, value)
        elif kind == "python_symbol":
            matches = _python_symbol(text, value)
        elif kind == "shell_function":
            matches = _shell_function(text, value)
        else:
            raise ValueError(f"unsupported selector kind {kind!r}")
    if len(matches) != 1:
        raise ValueError(f"selector {selector!r} resolved {len(matches)} times")
    if not matches[0]:
        raise ValueError(f"selector {selector!r} resolved empty content")
    return matches[0]


def _render(snapshot: dict[str, object]) -> str:
    lines = [
        "# Egomotion4D Kanban Collaboration Evidence",
        "",
        f"schema_version: {snapshot['schema_version']}",
        f"logical_id: {snapshot['logical_id']}",
        "",
    ]
    for evidence in snapshot["evidence"]:  # type: ignore[index]
        item = evidence  # type: ignore[assignment]
        lines.extend(
            [
                f"## {item['source_name']}",  # type: ignore[index]
                f"path: {item['source_path']}",  # type: ignore[index]
                f"whole_file_sha256: {item['whole_file_sha256']}",  # type: ignore[index]
                f"selector: {json.dumps(item['selector'], ensure_ascii=False, sort_keys=True)}",  # type: ignore[index]
                f"bounded_byte_length: {len(item['bounded_bytes'].encode('utf-8'))}",  # type: ignore[index]
                "BEGIN_EXACT_BOUNDED_BYTES",
            ]
        )
        bounded = item["bounded_bytes"]  # type: ignore[index]
        lines.append(bounded if bounded.endswith("\n") else f"{bounded}\n")
        lines.extend(["END_EXACT_BOUNDED_BYTES", ""])
    lines.append(TERMINAL_MARKER)
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_snapshot(manifest_path: Path, output_path: Path) -> dict[str, object]:
    """Resolve a source manifest and atomically replace its evidence snapshot."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    logical_id = manifest.get("logical_id")
    if not isinstance(logical_id, str) or not logical_id:
        raise ValueError("manifest logical_id must be non-empty")
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("manifest sources must be a list")
    if not sources:
        raise ValueError("manifest sources must be non-empty")

    snapshot_sources: list[dict[str, str]] = []
    evidence: list[dict[str, object]] = []
    source_names: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each manifest source must be an object")
        name, path_text, selectors = source.get("name"), source.get("path"), source.get("selectors")
        if not isinstance(name, str) or not isinstance(path_text, str) or not isinstance(selectors, list):
            raise ValueError("each source requires name, path, and selectors")
        if not name:
            raise ValueError("source name must be non-empty")
        if not path_text:
            raise ValueError("source path must be non-empty")
        if not selectors:
            raise ValueError("source selectors must be non-empty")
        if name in source_names:
            raise ValueError("source names must be unique")
        source_names.add(name)
        path = Path(path_text)
        if not path.is_absolute():
            raise ValueError(f"source path must be absolute: {path}")
        raw, text = _normalized_text(path)
        sha256 = hashlib.sha256(raw).hexdigest()
        snapshot_sources.append({"name": name, "path": path_text, "sha256": sha256})
        for selector in selectors:
            if not isinstance(selector, dict):
                raise ValueError("each selector must be an object")
            evidence.append(
                {
                    "source_name": name,
                    "source_path": path_text,
                    "whole_file_sha256": sha256,
                    "selector": selector,
                    "bounded_bytes": _resolve_selector(text, selector),
                }
            )
    snapshot: dict[str, object] = {
        "schema_version": 1,
        "logical_id": logical_id,
        "sources": snapshot_sources,
        "evidence": evidence,
    }
    _atomic_write(Path(output_path), _render(snapshot))
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_snapshot(args.manifest, args.output)


if __name__ == "__main__":
    main()
