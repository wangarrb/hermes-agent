"""Tests for /mycompress skill-backed context replacement."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.cli.test_cli_init import _make_cli


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def test_process_command_routes_mycompress_to_special_handler():
    shell = _make_cli()
    process_globals = shell.process_command.__func__.__globals__

    with patch.dict(process_globals, {"_skill_commands": {"/mycompress": {"name": "mycompress"}}}), \
         patch.object(shell, "_handle_skill_compress_command") as mock_handler:
        shell.process_command("/mycompress 保留关键结论")

    mock_handler.assert_called_once_with("/mycompress", "保留关键结论")


def test_mycompress_replaces_history_with_skill_summary():
    shell = _make_cli()
    handler_globals = shell._handle_skill_compress_command.__func__.__globals__
    history = _make_history()
    compressed = [history[0], history[-1]]
    shell.conversation_history = list(history)
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = "system"

    compressor = MagicMock()
    original_generate_summary = MagicMock(return_value="orig")
    compressor._generate_summary = original_generate_summary
    compressor._compute_summary_budget.return_value = 64
    compressor._serialize_for_summary.return_value = "serialized turns"
    compressor._with_summary_prefix.side_effect = lambda summary: f"[CONTEXT SUMMARY]: {summary}"
    compressor._previous_summary = None
    compressor._summary_failure_cooldown_until = 123.0
    shell.agent.context_compressor = compressor

    def fake_compress(history_arg, system_prompt, approx_tokens):
        compressor._generate_summary(history_arg)
        return compressed, ""

    shell.agent._compress_context.side_effect = fake_compress

    llm_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="handoff summary"))]
    )
    mock_build = MagicMock(return_value="skill prompt")

    def estimate(messages):
        return 100 if messages == history else 2

    with patch.dict(handler_globals, {"build_skill_invocation_message": mock_build}), \
         patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=estimate), \
         patch("agent.auxiliary_client.call_llm", return_value=llm_response) as mock_call:
        shell._handle_skill_compress_command("/mycompress", "保留关键结论")

    assert shell.conversation_history == compressed
    assert compressor._previous_summary == "handoff summary"
    assert compressor._summary_failure_cooldown_until == 0.0
    assert compressor._generate_summary is original_generate_summary
    mock_build.assert_called_once()
    assert mock_build.call_args.kwargs["runtime_note"]
    mock_call.assert_called_once()
