"""Tests for gateway /mycompress command routing and arg parsing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_messages = {}
    runner._update_prompt_pending = {}
    runner._draining = False
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.config = SimpleNamespace(streaming=None)
    runner._session_key_for_source = lambda source: "quick-key"
    runner._is_user_authorized = lambda source: True
    runner._handle_message_with_agent = AsyncMock(return_value="ok")
    return runner


def test_gateway_parse_mycompress_args_supports_keep_last_rounds_flags():
    assert gateway_run._parse_mycompress_args("-n 3 保留关键结论") == (3, "保留关键结论")
    assert gateway_run._parse_mycompress_args("--keep-last-turns=2") == (2, "")
    assert gateway_run._parse_mycompress_args("--keep-last-rounds 4 聚焦 AEB") == (4, "聚焦 AEB")


@pytest.mark.asyncio
async def test_gateway_handle_message_routes_mycompress_to_special_handler():
    runner = _make_runner()
    event = _make_event("/mycompress -n 3 保留关键结论")

    with (
        patch("agent.skill_commands.get_skill_commands", return_value={"/mycompress": {"name": "mycompress"}}),
        patch("agent.skill_commands.resolve_skill_command_key", return_value="/mycompress"),
        patch.object(runner, "_handle_skill_compress_command", AsyncMock(return_value="已压缩完成。"), create=True) as mock_handler,
    ):
        result = await runner._handle_message(event)

    assert result == "已压缩完成。"
    mock_handler.assert_awaited_once_with(event, "/mycompress", "-n 3 保留关键结论")
    runner._handle_message_with_agent.assert_not_awaited()
