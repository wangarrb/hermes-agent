"""Tests for gateway /mycompress true context replacement."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/mycompress") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _make_runner(history: list[dict[str, str]]):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    runner._session_key_for_source = lambda source: "quick-key"
    runner._resolve_session_agent_runtime = MagicMock(return_value=("test-model", {"api_key": "***"}))
    return runner, session_entry


@pytest.mark.asyncio
async def test_gateway_mycompress_rewrites_transcript_and_returns_minimal_status():
    history = _make_history()
    compressed = [history[0], history[-1]]
    runner, session_entry = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.api_mode = "chat_completions"
    agent_instance._cached_system_prompt = "system"
    agent_instance.context_compressor._generate_summary = MagicMock(return_value="orig")
    agent_instance.context_compressor._compute_summary_budget.return_value = 64
    agent_instance.context_compressor._serialize_for_summary.return_value = "serialized turns"
    agent_instance.context_compressor._with_summary_prefix.side_effect = lambda summary: f"[CONTEXT SUMMARY]: {summary}"
    agent_instance.context_compressor._previous_summary = None
    agent_instance.context_compressor._summary_failure_cooldown_until = 123.0
    agent_instance.context_compressor.protect_first_n = 0
    agent_instance.context_compressor._align_boundary_forward.return_value = 0
    agent_instance.context_compressor._find_tail_cut_by_tokens.return_value = 2
    agent_instance.session_id = "sess-2"
    agent_instance._current_main_runtime.return_value = {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.6",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }

    def fake_compress(history_arg, system_prompt, approx_tokens):
        agent_instance.context_compressor._generate_summary(history_arg)
        return compressed, ""

    agent_instance._compress_context.side_effect = fake_compress

    llm_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="handoff summary"))]
    )

    with (
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=100),
        patch("agent.skill_commands.build_skill_invocation_message", return_value="skill prompt") as mock_build,
        patch("agent.auxiliary_client.call_llm", return_value=llm_response) as mock_call,
    ):
        result = await runner._handle_skill_compress_command(_make_event("/mycompress 保留关键结论"), "/mycompress", "保留关键结论")

    assert result == "已压缩完成。"
    mock_build.assert_called_once()
    mock_call.assert_called_once()
    runner.session_store.rewrite_transcript.assert_called_once_with("sess-2", compressed)
    runner.session_store.update_session.assert_called_once_with(session_entry.session_key, last_prompt_tokens=0)
    runner.session_store._save.assert_called_once()
    assert session_entry.session_id == "sess-2"
    assert agent_instance.context_compressor._previous_summary == "handoff summary"


@pytest.mark.asyncio
async def test_gateway_mycompress_keep_last_rounds_adjusts_tail_cut():
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    compressed = [history[0], history[-2], history[-1]]
    runner, _session_entry = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.api_mode = "chat_completions"
    agent_instance._cached_system_prompt = "system"
    original_find_tail_cut = MagicMock(return_value=5)
    agent_instance.context_compressor._generate_summary = MagicMock(return_value="orig")
    agent_instance.context_compressor._find_tail_cut_by_tokens = original_find_tail_cut
    agent_instance.context_compressor._align_boundary_backward.side_effect = lambda messages, idx: idx
    agent_instance.context_compressor._compute_summary_budget.return_value = 64
    agent_instance.context_compressor._serialize_for_summary.return_value = "serialized turns"
    agent_instance.context_compressor._with_summary_prefix.side_effect = lambda summary: f"[CONTEXT SUMMARY]: {summary}"
    agent_instance.context_compressor._previous_summary = None
    agent_instance.context_compressor._summary_failure_cooldown_until = 123.0
    agent_instance.context_compressor.protect_first_n = 0
    agent_instance.context_compressor.protect_last_n = 1
    agent_instance.session_id = "sess-1"
    agent_instance._current_main_runtime.return_value = {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.6",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }

    observed = {}

    def fake_compress(history_arg, system_prompt, approx_tokens):
        observed["cut_idx"] = agent_instance.context_compressor._find_tail_cut_by_tokens(history_arg, 0)
        observed["protect_last_n_during_call"] = agent_instance.context_compressor.protect_last_n
        return compressed, ""

    agent_instance._compress_context.side_effect = fake_compress

    llm_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="handoff summary"))]
    )

    with (
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=100),
        patch("agent.skill_commands.build_skill_invocation_message", return_value="skill prompt"),
        patch("agent.auxiliary_client.call_llm", return_value=llm_response),
    ):
        result = await runner._handle_skill_compress_command(_make_event("/mycompress -n 2 保留关键结论"), "/mycompress", "-n 2 保留关键结论")

    assert result == "已压缩完成。"
    assert observed["cut_idx"] == 2
    assert observed["protect_last_n_during_call"] == 4
    assert agent_instance.context_compressor.protect_last_n == 1
    assert agent_instance.context_compressor._find_tail_cut_by_tokens is original_find_tail_cut
