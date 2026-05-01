from contextlib import nullcontext

from cli import HermesCLI
from hermes_cli.commands import resolve_command


class DummyAgent:
    def __init__(self):
        self.compression_enabled = True
        self._cached_system_prompt = "FULL CACHED SYSTEM PROMPT SHOULD NOT BE NESTED"
        self.session_id = "new-session"
        self.calls = []

    def _compress_context(self, messages, system_message, *, approx_tokens=None, focus_topic=None):
        self.calls.append(
            {
                "messages": messages,
                "system_message": system_message,
                "approx_tokens": approx_tokens,
                "focus_topic": focus_topic,
            }
        )
        return ([{"role": "user", "content": "[CONTEXT SUMMARY]: compacted"}], "new system prompt")


def test_manual_compress_does_not_pass_cached_system_prompt(monkeypatch):
    """Manual /compress should rebuild the next prompt without nesting the old one."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.conversation_history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    cli.agent = DummyAgent()
    cli.session_id = "old-session"
    cli._pending_title = "old title"
    cli._busy_command = lambda _message: nullcontext()

    monkeypatch.setattr(
        "agent.manual_compression_feedback.summarize_manual_compression",
        lambda *args, **kwargs: {
            "noop": False,
            "headline": "compressed",
            "token_line": "tokens reduced",
            "note": "",
        },
    )

    cli._manual_compress("/compress database schema")

    assert len(cli.agent.calls) == 1
    call = cli.agent.calls[0]
    assert call["system_message"] is None
    assert call["system_message"] != cli.agent._cached_system_prompt
    assert call["focus_topic"] == "database schema"
    assert cli.session_id == "new-session"
    assert cli._pending_title is None


def test_process_command_mycompress_routes_to_wrapper():
    """/mycompress must be handled as a real command, not fall through to skill injection."""
    assert resolve_command("mycompress") is not None

    cli = HermesCLI.__new__(HermesCLI)
    called = []

    def fake_mycompress(cmd_original):
        called.append(cmd_original)

    cli._handle_mycompress_command = fake_mycompress

    assert cli.process_command("/mycompress -n 2 keep facts") is True
    assert called == ["/mycompress -n 2 keep facts"]


def test_mycompress_wrapper_uses_skill_focus_and_protects_last_user_rounds(monkeypatch):
    """The wrapper should delegate to /compress while translating -n user rounds to protect_last_n."""
    cli = HermesCLI.__new__(HermesCLI)

    class DummyCompressor:
        protect_last_n = 20

    class MyCompressAgent:
        context_compressor = DummyCompressor()

    cli.agent = MyCompressAgent()
    cli.conversation_history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a4"},
    ]

    # Mock build_skill_invocation_message to return a string (not dict)
    monkeypatch.setattr(
        "agent.skill_commands.build_skill_invocation_message",
        lambda *args, **kwargs: "SKILL_FOCUS",
    )

    calls = []

    def fake_manual_compress(cmd_original):
        calls.append((cmd_original, cli.agent.context_compressor.protect_last_n))

    cli._manual_compress = fake_manual_compress

    cli._handle_mycompress_command("/mycompress -n 2 保留关键结论")

    assert len(calls) == 1
    compress_cmd, protect_last_n_during_call = calls[0]
    assert compress_cmd.startswith("/compress SKILL_FOCUS")
    assert "用户补充要求：保留关键结论" in compress_cmd
    # Last 2 user rounds start at message index 4, so 8 - 4 = 4 tail messages protected.
    assert protect_last_n_during_call == 4
    assert cli.agent.context_compressor.protect_last_n == 20