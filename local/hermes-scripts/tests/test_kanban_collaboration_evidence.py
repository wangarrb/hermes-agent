"""Contract tests for the Kanban collaboration evidence snapshot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "kanban_collaboration_evidence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("kanban_collaboration_evidence", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path, sources: list[dict[str, object]]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "logical_id": "egomotion4d-kanban-collaboration",
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return path


def _source(path: Path, selectors: list[dict[str, str]]) -> dict[str, object]:
    return {"name": path.name, "path": str(path.resolve()), "selectors": selectors}


def test_build_snapshot_resolves_all_selector_kinds_and_normalizes_crlf(tmp_path: Path) -> None:
    module = _load_module()
    markdown = tmp_path / "guide.md"
    markdown.write_bytes(b"# Intro\r\n## Target\r\nbody\r\n### Nested\r\nkeep\r\n## Next\r\nstop\r\n")
    python_file = tmp_path / "module.py"
    python_file.write_text(
        "@decorator\n"
        "def top():\n"
        "    return 1\n\n"
        "@next_decorator\n"
        "def following():\n"
        "    return 9\n\n"
        "class Worker:\n"
        "    @classmethod\n"
        "    def run(cls):\n"
        "        return 2\n\n"
        "    def later(self):\n"
        "        return 3\n",
        encoding="utf-8",
    )
    shell = tmp_path / "tool.sh"
    shell.write_text(
        "usage() {\n"
        "  printf '%s' 'brace } stays quoted'\n"
        "  echo \"{ quoted too }\"\n"
        "}\n"
        "next() { :; }\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        tmp_path,
        [
            _source(markdown, [{"kind": "markdown_heading", "value": "## Target"}]),
            _source(
                python_file,
                [
                    {"kind": "python_symbol", "value": "top"},
                    {"kind": "python_symbol", "value": "Worker.run"},
                ],
            ),
            _source(shell, [{"kind": "shell_function", "value": "usage"}]),
        ],
    )
    output = tmp_path / "evidence.md"

    snapshot = module.build_snapshot(manifest, output)

    bounded = [item["bounded_bytes"] for item in snapshot["evidence"]]
    assert bounded == [
        "## Target\nbody\n### Nested\nkeep\n",
        "@decorator\ndef top():\n    return 1\n",
        "    @classmethod\n    def run(cls):\n        return 2\n",
        "usage() {\n  printf '%s' 'brace } stays quoted'\n  echo \"{ quoted too }\"\n}\n",
    ]
    assert snapshot["sources"][0]["sha256"] == hashlib.sha256(markdown.read_bytes()).hexdigest()
    rendered = output.read_text(encoding="utf-8")
    assert "\r" not in rendered
    assert rendered.endswith("END_KANBAN_COLLABORATION_EVIDENCE")


def test_shell_function_ignores_braces_and_quotes_in_comments(tmp_path: Path) -> None:
    module = _load_module()
    shell = tmp_path / "comment.sh"
    shell.write_text("usage() {\n  # unmatched ' and brace }\n  :\n}\n", encoding="utf-8")
    manifest = _manifest(tmp_path, [_source(shell, [{"kind": "shell_function", "value": "usage"}])])

    snapshot = module.build_snapshot(manifest, tmp_path / "evidence.md")

    assert snapshot["evidence"][0]["bounded_bytes"].endswith("  :\n}\n")


def test_python_symbol_ends_at_the_selected_ast_node(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "symbols.py"
    source.write_text(
        "@decorator\n"
        "def top():\n"
        "    return 1\n"
        "top_level_value = 2\n"
        "# top-level tail\n\n"
        "class Worker:\n"
        "    @decorator\n"
        "    def run(self):\n"
        "        return 3\n"
        "    class_value = 4\n"
        "    # class tail\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        tmp_path,
        [
            _source(
                source,
                [
                    {"kind": "python_symbol", "value": "top"},
                    {"kind": "python_symbol", "value": "Worker.run"},
                ],
            )
        ],
    )

    snapshot = module.build_snapshot(manifest, tmp_path / "evidence.md")

    assert [item["bounded_bytes"] for item in snapshot["evidence"]] == [
        "@decorator\ndef top():\n    return 1\n",
        "    @decorator\n    def run(self):\n        return 3\n",
    ]


def test_shell_function_handles_escaped_braces_heredocs_and_nested_subshells(tmp_path: Path) -> None:
    module = _load_module()
    shell = tmp_path / "complex.sh"
    shell.write_text(
        "usage() {\n"
        "  printf '%s\\n' \\}\n"
        "  cat <<'EOF'\n"
        "} literal heredoc brace\n"
        "EOF\n"
        "  nested=$(printf '%s' \"$(printf '%s' '}')\")\n"
        "  ( printf '%s' 'subshell }' )\n"
        "}\n"
        "after() { :; }\n",
        encoding="utf-8",
    )
    subprocess.run(["bash", "-n", str(shell)], check=True)
    manifest = _manifest(tmp_path, [_source(shell, [{"kind": "shell_function", "value": "usage"}])])

    snapshot = module.build_snapshot(manifest, tmp_path / "evidence.md")

    extracted = snapshot["evidence"][0]["bounded_bytes"]
    assert "} literal heredoc brace" in extracted
    assert "nested=$(printf" in extracted
    assert extracted.rstrip().endswith("}")


def test_shell_function_rejects_unsupported_backticks(tmp_path: Path) -> None:
    module = _load_module()
    shell = tmp_path / "backticks.sh"
    shell.write_text("usage() { value=`printf '%s' '}'`; }\n", encoding="utf-8")
    subprocess.run(["bash", "-n", str(shell)], check=True)
    manifest = _manifest(tmp_path, [_source(shell, [{"kind": "shell_function", "value": "usage"}])])

    with pytest.raises(ValueError, match="unsupported shell construct"):
        module.build_snapshot(manifest, tmp_path / "evidence.md")


def test_shell_function_ignores_fake_declarations_inside_heredocs(tmp_path: Path) -> None:
    module = _load_module()
    shell = tmp_path / "fake-only.sh"
    shell.write_text(
        "cat <<'EOF'\n"
        "usage() {\n"
        "  fake body\n"
        "}\n"
        "EOF\n",
        encoding="utf-8",
    )
    subprocess.run(["bash", "-n", str(shell)], check=True)
    manifest = _manifest(tmp_path, [_source(shell, [{"kind": "shell_function", "value": "usage"}])])

    with pytest.raises(ValueError, match="resolved 0 times"):
        module.build_snapshot(manifest, tmp_path / "evidence.md")


def test_shell_function_finds_only_real_declaration_after_a_heredoc(tmp_path: Path) -> None:
    module = _load_module()
    shell = tmp_path / "fake-before-real.sh"
    shell.write_text(
        "cat <<'EOF'\n"
        "usage() { fake }\n"
        "EOF\n"
        "usage() { :; }\n",
        encoding="utf-8",
    )
    subprocess.run(["bash", "-n", str(shell)], check=True)
    manifest = _manifest(tmp_path, [_source(shell, [{"kind": "shell_function", "value": "usage"}])])

    snapshot = module.build_snapshot(manifest, tmp_path / "evidence.md")

    assert snapshot["evidence"][0]["bounded_bytes"] == "usage() { :; }\n"


def test_shell_function_does_not_treat_a_here_string_as_a_heredoc(tmp_path: Path) -> None:
    module = _load_module()
    shell = tmp_path / "here-string.sh"
    shell.write_text("usage() { read value <<< \"}\"; }\n", encoding="utf-8")
    subprocess.run(["bash", "-n", str(shell)], check=True)
    manifest = _manifest(tmp_path, [_source(shell, [{"kind": "shell_function", "value": "usage"}])])

    snapshot = module.build_snapshot(manifest, tmp_path / "evidence.md")

    assert snapshot["evidence"][0]["bounded_bytes"] == "usage() { read value <<< \"}\"; }\n"


def test_shell_function_ignores_fake_declarations_in_quotes_and_comments(tmp_path: Path) -> None:
    module = _load_module()
    shell = tmp_path / "quoted-fakes.sh"
    shell.write_text(
        "# usage() { fake comment }\n"
        "fake='usage() { fake quote }'\n"
        "usage() { :; }\n",
        encoding="utf-8",
    )
    subprocess.run(["bash", "-n", str(shell)], check=True)
    manifest = _manifest(tmp_path, [_source(shell, [{"kind": "shell_function", "value": "usage"}])])

    snapshot = module.build_snapshot(manifest, tmp_path / "evidence.md")

    assert snapshot["evidence"][0]["bounded_bytes"] == "usage() { :; }\n"


@pytest.mark.parametrize(
    ("sources", "error"),
    [
        ([], "sources must be non-empty"),
        ([{"name": "source", "path": "PATH", "selectors": []}], "selectors must be non-empty"),
        ([{"name": "", "path": "PATH", "selectors": [{"kind": "whole_file"}]}], "name must be non-empty"),
        ([{"name": "source", "path": "", "selectors": [{"kind": "whole_file"}]}], "path must be non-empty"),
        (
            [
                {"name": "duplicate", "path": "PATH", "selectors": [{"kind": "whole_file"}]},
                {"name": "duplicate", "path": "PATH", "selectors": [{"kind": "whole_file"}]},
            ],
            "source names must be unique",
        ),
    ],
)
def test_build_snapshot_rejects_empty_or_ambiguous_source_declarations(
    tmp_path: Path, sources: list[dict[str, object]], error: str
) -> None:
    module = _load_module()
    source = tmp_path / "source.txt"
    source.write_text("value\n", encoding="utf-8")
    rendered_sources = [
        {**item, "path": str(source.resolve()) if item.get("path") == "PATH" else item.get("path")}
        for item in sources
    ]
    manifest = _manifest(tmp_path, rendered_sources)

    with pytest.raises(ValueError, match=error):
        module.build_snapshot(manifest, tmp_path / "evidence.md")


def test_real_manifest_rebuild_matches_the_tracked_snapshot(tmp_path: Path) -> None:
    module = _load_module()
    repo = Path(__file__).parents[3]
    manifest = repo / "local" / "mental-models" / "egomotion4d" / "sources" / "kanban_collaboration_sources.json"
    tracked = repo / "local" / "mental-models" / "egomotion4d" / "sources" / "kanban_collaboration_current_evidence.md"
    rebuilt = tmp_path / "evidence.md"

    snapshot = module.build_snapshot(manifest, rebuilt)

    assert len(snapshot["sources"]) == 11
    assert len(snapshot["evidence"]) == 19
    assert rebuilt.read_bytes() == tracked.read_bytes()


def test_build_snapshot_extracts_real_start_kanban_usage(tmp_path: Path) -> None:
    module = _load_module()
    start_kanban = Path(__file__).parents[2] / "bin" / "start-kanban.sh"
    manifest = _manifest(
        tmp_path,
        [
            _source(
                start_kanban,
                [
                    {"kind": "shell_function", "value": "workspace_for_role"},
                    {"kind": "shell_function", "value": "build_role_command"},
                    {"kind": "shell_function", "value": "usage"},
                ],
            )
        ],
    )
    snapshot = module.build_snapshot(manifest, tmp_path / "evidence.md")

    workspace, build_command, usage = [item["bounded_bytes"] for item in snapshot["evidence"]]
    assert workspace.startswith("workspace_for_role() {")
    assert build_command.startswith("build_role_command() {")
    assert build_command.rstrip().endswith("}")
    assert usage.startswith("usage() {")
    assert "用法:" in usage
    assert usage.rstrip().endswith("}")


@pytest.mark.parametrize(
    ("contents", "selector", "error"),
    [
        ("# Else\n", {"kind": "markdown_heading", "value": "## Missing"}, "resolved 0 times"),
        ("## Same\nfirst\n## Same\nsecond\n", {"kind": "markdown_heading", "value": "## Same"}, "resolved 2 times"),
        ("", {"kind": "whole_file"}, "empty"),
    ],
)
def test_build_snapshot_rejects_missing_duplicate_and_empty_selectors(
    tmp_path: Path, contents: str, selector: dict[str, str], error: str
) -> None:
    module = _load_module()
    source = tmp_path / "source.md"
    source.write_text(contents, encoding="utf-8")
    manifest = _manifest(tmp_path, [_source(source, [selector])])

    with pytest.raises(ValueError, match=error):
        module.build_snapshot(manifest, tmp_path / "evidence.md")


def test_build_snapshot_is_deterministic_and_preserves_old_output_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    source = tmp_path / "source.txt"
    source.write_text("stable\n", encoding="utf-8")
    manifest = _manifest(tmp_path, [_source(source, [{"kind": "whole_file"}])])
    output = tmp_path / "evidence.md"

    first = module.build_snapshot(manifest, output)
    first_bytes = output.read_bytes()
    second = module.build_snapshot(manifest, output)
    assert first == second
    assert output.read_bytes() == first_bytes

    def fail_replace(source_path: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        module.build_snapshot(manifest, output)
    assert output.read_bytes() == first_bytes
