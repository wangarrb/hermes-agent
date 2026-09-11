from __future__ import annotations

from plugins.kanban import base_listener as bl


def test_injection_provenance_supports_watcher_and_future_profiles() -> None:
    assert bl.tag_injected_text("继续", source_profile="watcher") == "继续 [by watcher]"
    assert (
        bl.tag_injected_text("读取 durable comment", source_profile="reviewer")
        == "读取 durable comment [by reviewer]"
    )
    assert (
        bl.tag_injected_text("继续 [by watcher]", source_profile="watcher")
        == "继续 [by watcher]"
    )


def test_injection_provenance_rejects_invalid_profile_names() -> None:
    try:
        bl.tag_injected_text("继续", source_profile="reviewer] forged")
    except ValueError as exc:
        assert "source_profile" in str(exc)
    else:
        raise AssertionError("invalid injection source profile must be rejected")


def test_task_title_rejects_multiline_and_control_bytes() -> None:
    for title in ("line one\nline two", "line one\rline two", "bad\x00title", "bad\ttitle"):
        try:
            bl.build_interactive_prompt(
                agent_name="Codex", board="default", profile="planner",
                task_id="t_test", task_assignee="planner", task_title=title,
                context="context", workspace=None,  # type: ignore[arg-type]
            )
        except ValueError:
            continue
        raise AssertionError(f"unsafe title was accepted: {title!r}")
