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


def test_mycompress_parses_keep_last_rounds_flags():
    shell = _make_cli()
    handler_globals = shell._handle_skill_compress_command.__func__.__globals__
    parse_args = handler_globals["_parse_mycompress_args"]

    assert parse_args("-n 3 保留关键结论") == (3, "保留关键结论")
    assert parse_args("--keep-last-turns=2") == (2, "")
    assert parse_args("--keep-last-rounds 4 聚焦 AEB") == (4, "聚焦 AEB")


def test_mycompress_keep_last_rounds_overrides_tail_cut_and_strips_flag_from_prompt():
    shell = _make_cli()
    handler_globals = shell._handle_skill_compress_command.__func__.__globals__
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    compressed = [history[0], history[-2], history[-1]]
    shell.conversation_history = list(history)
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = "system"
    shell.agent.api_mode = "chat_completions"

    compressor = MagicMock()
    original_generate_summary = MagicMock(return_value="orig")
    original_find_tail_cut = MagicMock(return_value=5)
    compressor._generate_summary = original_generate_summary
    compressor._find_tail_cut_by_tokens = original_find_tail_cut
    compressor._align_boundary_backward.side_effect = lambda messages, idx: idx
    compressor._compute_summary_budget.return_value = 64
    compressor._serialize_for_summary.return_value = "serialized turns"
    compressor._with_summary_prefix.side_effect = lambda summary: f"[CONTEXT SUMMARY]: {summary}"
    compressor._previous_summary = None
    compressor._summary_failure_cooldown_until = 123.0
    compressor.protect_first_n = 0
    compressor.protect_last_n = 1
    shell.agent.context_compressor = compressor
    shell.agent._current_main_runtime.return_value = {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.6",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }

    observed = {}

    def fake_compress(history_arg, system_prompt, approx_tokens):
        observed["cut_idx"] = compressor._find_tail_cut_by_tokens(history_arg, 0)
        observed["protect_last_n_during_call"] = compressor.protect_last_n
        compressor._generate_summary(history_arg[:observed["cut_idx"]])
        return compressed, ""

    shell.agent._compress_context.side_effect = fake_compress

    llm_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="handoff summary"))]
    )
    mock_build = MagicMock(return_value="skill prompt")

    with patch.dict(handler_globals, {"build_skill_invocation_message": mock_build}), \
         patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=100), \
         patch("agent.auxiliary_client.call_llm", return_value=llm_response):
        shell._handle_skill_compress_command("/mycompress", "-n 2 保留关键结论")

    assert observed["cut_idx"] == 2
    assert observed["protect_last_n_during_call"] == 4
    assert shell.conversation_history == compressed
    assert compressor.protect_last_n == 1
    assert compressor._find_tail_cut_by_tokens is original_find_tail_cut
    mock_build.assert_called_once()
    assert mock_build.call_args.args[1] == "保留关键结论"


def test_mycompress_replaces_history_with_skill_summary():
    shell = _make_cli()
    handler_globals = shell._handle_skill_compress_command.__func__.__globals__
    history = _make_history()
    compressed = [history[0], history[-1]]
    shell.conversation_history = list(history)
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = "system"
    shell.agent.api_mode = "chat_completions"

    compressor = MagicMock()
    original_generate_summary = MagicMock(return_value="orig")
    compressor._generate_summary = original_generate_summary
    compressor._compute_summary_budget.return_value = 64
    compressor._serialize_for_summary.return_value = "serialized turns"
    compressor._with_summary_prefix.side_effect = lambda summary: f"[CONTEXT SUMMARY]: {summary}"
    compressor._previous_summary = None
    compressor._summary_failure_cooldown_until = 123.0
    shell.agent.context_compressor = compressor
    shell.agent._current_main_runtime.return_value = {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.6",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }

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
    assert mock_call.call_args.kwargs["main_runtime"] == shell.agent._current_main_runtime.return_value


def test_mycompress_uses_main_codex_chain_for_codex_responses():
    shell = _make_cli()
    handler_globals = shell._handle_skill_compress_command.__func__.__globals__
    history = _make_history()
    compressed = [history[0], history[-1]]
    shell.conversation_history = list(history)
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = "system prompt"
    shell.agent.api_mode = "codex_responses"

    compressor = MagicMock()
    original_generate_summary = MagicMock(return_value="orig")
    compressor._generate_summary = original_generate_summary
    compressor._compute_summary_budget.return_value = 64
    compressor._serialize_for_summary.return_value = "serialized turns"
    compressor._with_summary_prefix.side_effect = lambda summary: f"[CONTEXT SUMMARY]: {summary}"
    compressor._previous_summary = None
    compressor._summary_failure_cooldown_until = 123.0
    shell.agent.context_compressor = compressor
    shell.agent._current_main_runtime.return_value = {
        "provider": "cch",
        "model": "gpt-5.4",
        "base_url": "http://cch.jmadas.com/v1",
        "api_mode": "codex_responses",
    }

    def fake_compress(history_arg, system_prompt, approx_tokens):
        compressor._generate_summary(history_arg)
        return compressed, ""

    shell.agent._compress_context.side_effect = fake_compress
    shell.agent._build_api_kwargs.return_value = {
        "model": "gpt-5.4",
        "instructions": "system prompt",
        "input": [{"role": "user", "content": "prompt"}],
        "tools": [{"type": "function", "name": "noop"}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    shell.agent._run_codex_stream.return_value = SimpleNamespace(output=[])
    shell.agent._normalize_codex_response.return_value = (
        SimpleNamespace(content="handoff summary", reasoning=None),
        "stop",
    )

    mock_build = MagicMock(return_value="skill prompt")

    def estimate(messages):
        return 100 if messages == history else 2

    with patch.dict(handler_globals, {"build_skill_invocation_message": mock_build}), \
         patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=estimate), \
         patch("agent.auxiliary_client.call_llm") as mock_call:
        shell._handle_skill_compress_command("/mycompress", "保留关键结论")

    mock_call.assert_not_called()
    shell.agent._build_api_kwargs.assert_called_once()
    shell.agent._run_codex_stream.assert_called_once()
    sent_kwargs = shell.agent._run_codex_stream.call_args.args[0]
    assert "tools" not in sent_kwargs
    assert "tool_choice" not in sent_kwargs
    assert "parallel_tool_calls" not in sent_kwargs
    assert sent_kwargs["max_output_tokens"] == 128
    assert compressor._previous_summary == "handoff summary"


def test_mycompress_suppresses_stream_callbacks_while_generating_codex_summary():
    shell = _make_cli()
    handler_globals = shell._handle_skill_compress_command.__func__.__globals__
    history = _make_history()
    compressed = [history[0], history[-1]]
    shell.conversation_history = list(history)
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = "system prompt"
    shell.agent.api_mode = "codex_responses"

    compressor = MagicMock()
    original_generate_summary = MagicMock(return_value="orig")
    compressor._generate_summary = original_generate_summary
    compressor._compute_summary_budget.return_value = 64
    compressor._serialize_for_summary.return_value = "serialized turns"
    compressor._with_summary_prefix.side_effect = lambda summary: f"[CONTEXT SUMMARY]: {summary}"
    compressor._previous_summary = None
    compressor._summary_failure_cooldown_until = 123.0
    shell.agent.context_compressor = compressor
    shell.agent._current_main_runtime.return_value = {
        "provider": "cch",
        "model": "gpt-5.4",
        "base_url": "http://cch.jmadas.com/v1",
        "api_mode": "codex_responses",
    }

    def fake_compress(history_arg, system_prompt, approx_tokens):
        compressor._generate_summary(history_arg)
        return compressed, ""

    shell.agent._compress_context.side_effect = fake_compress
    shell.agent._build_api_kwargs.return_value = {
        "model": "gpt-5.4",
        "instructions": "system prompt",
        "input": [{"role": "user", "content": "prompt"}],
        "tools": [{"type": "function", "name": "noop"}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    streamed_chunks = []

    def primary_stream_cb(text):
        streamed_chunks.append(("primary", text))

    def secondary_stream_cb(text):
        streamed_chunks.append(("secondary", text))

    shell.agent.stream_delta_callback = primary_stream_cb
    shell.agent._stream_callback = secondary_stream_cb

    def fake_run_codex_stream(_api_kwargs):
        if shell.agent.stream_delta_callback is not None:
            shell.agent.stream_delta_callback("LEAKED_SUMMARY")
        if shell.agent._stream_callback is not None:
            shell.agent._stream_callback("LEAKED_SUMMARY")
        return SimpleNamespace(output=[])

    shell.agent._run_codex_stream.side_effect = fake_run_codex_stream
    shell.agent._normalize_codex_response.return_value = (
        SimpleNamespace(content="handoff summary", reasoning=None),
        "stop",
    )

    mock_build = MagicMock(return_value="skill prompt")

    def estimate(messages):
        return 100 if messages == history else 2

    with patch.dict(handler_globals, {"build_skill_invocation_message": mock_build}), \
         patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=estimate), \
         patch("agent.auxiliary_client.call_llm") as mock_call:
        shell._handle_skill_compress_command("/mycompress", "保留关键结论")

    mock_call.assert_not_called()
    assert streamed_chunks == []
    assert shell.agent.stream_delta_callback is primary_stream_cb
    assert shell.agent._stream_callback is secondary_stream_cb
    assert compressor._previous_summary == "handoff summary"


def test_mycompress_only_prints_minimal_completion_status(capsys):
    shell = _make_cli()
    handler_globals = shell._handle_skill_compress_command.__func__.__globals__
    history = _make_history()
    compressed = [history[0], history[-1]]
    shell.conversation_history = list(history)
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = "system prompt"
    shell.agent.api_mode = "codex_responses"

    compressor = MagicMock()
    original_generate_summary = MagicMock(return_value="orig")
    compressor._generate_summary = original_generate_summary
    compressor._compute_summary_budget.return_value = 64
    compressor._serialize_for_summary.return_value = "serialized turns"
    compressor._with_summary_prefix.side_effect = lambda summary: f"[CONTEXT SUMMARY]: {summary}"
    compressor._previous_summary = None
    compressor._summary_failure_cooldown_until = 123.0
    shell.agent.context_compressor = compressor
    shell.agent._current_main_runtime.return_value = {
        "provider": "cch",
        "model": "gpt-5.4",
        "base_url": "http://cch.jmadas.com/v1",
        "api_mode": "codex_responses",
    }

    def fake_compress(history_arg, system_prompt, approx_tokens):
        compressor._generate_summary(history_arg)
        return compressed, ""

    shell.agent._compress_context.side_effect = fake_compress
    shell.agent._build_api_kwargs.return_value = {
        "model": "gpt-5.4",
        "instructions": "system prompt",
        "input": [{"role": "user", "content": "prompt"}],
        "tools": [{"type": "function", "name": "noop"}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    shell.agent._run_codex_stream.return_value = SimpleNamespace(output=[])
    shell.agent._normalize_codex_response.return_value = (
        SimpleNamespace(content="handoff summary", reasoning=None),
        "stop",
    )

    mock_build = MagicMock(return_value="skill prompt")

    def estimate(messages):
        return 100 if messages == history else 2

    with patch.dict(handler_globals, {"build_skill_invocation_message": mock_build}), \
         patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=estimate), \
         patch("agent.auxiliary_client.call_llm") as mock_call:
        shell._handle_skill_compress_command("/mycompress", "保留关键结论")

    mock_call.assert_not_called()
    output = capsys.readouterr().out
    assert "已压缩完成。" in output
    assert "Skill-compressing" not in output
    assert "/mycompress debug" not in output
    assert "Skill compressed" not in output
