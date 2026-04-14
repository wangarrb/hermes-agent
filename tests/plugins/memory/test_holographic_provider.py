"""Tests for the holographic memory provider plugin."""

from pathlib import Path

from plugins.memory.holographic import HolographicMemoryProvider


def _make_provider(tmp_path: Path) -> HolographicMemoryProvider:
    provider = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "memory_store.db"),
            "auto_extract": True,
            "default_trust": 0.5,
        }
    )
    provider.initialize(session_id="session-1")
    return provider


def test_on_session_end_auto_extracts_chinese_user_preference(tmp_path: Path):
    provider = _make_provider(tmp_path)

    provider.on_session_end([
        {"role": "user", "content": "记住：我每周五下午都要交周报。"},
    ])

    facts = provider._store.list_facts(limit=10)
    assert any(
        fact["category"] == "user_pref" and "每周五下午都要交周报" in fact["content"]
        for fact in facts
    )


def test_on_session_end_auto_extracts_assistant_confirmation(tmp_path: Path):
    provider = _make_provider(tmp_path)

    provider.on_session_end([
        {"role": "assistant", "content": "已记住：项目默认使用 gpuserver 作为 SSH 别名。"},
    ])

    facts = provider._store.list_facts(limit=10)
    assert any(
        fact["category"] == "general" and "gpuserver" in fact["content"]
        for fact in facts
    )
