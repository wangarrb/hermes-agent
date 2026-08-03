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
