"""PS10 preserves the live Xunfei/OneAPI selector, not a host-policy change.

Pure request-shape unit tests: no model requests, credentials, or mocked responses.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from providers.base import ProviderProfile


@pytest.mark.parametrize("profile_path", [False, True], ids=["legacy", "profile"])
@pytest.mark.parametrize("reasoning,expected", [
    (None, "enabled"), ({"enabled": False}, "disabled"),
    ({"enabled": 0}, "enabled"), ({"enabled": "false"}, "enabled"),
    ({"effort": "none"}, "enabled"),
])
@pytest.mark.parametrize("provider,base", [
    ("xunfei-coding", "https://unit-xunfei.invalid/v1"),
    (" XUNFEI-CODING ", None),
    ("custom", "http://127.0.0.1:3001/v1"),
    ("named-oneapi", "http://LOCALHOST:9876/arbitrary"),
    ("unrelated-local", "http://localhost:11434/v1"),
])
def test_live_thinking_selector_before_profile_split(profile_path, reasoning, expected, provider, base):
    overrides = {"extra_body": {"thinking": {"type": "caller-value"}, "unit_marker": "keep"}}
    before = deepcopy(overrides)
    kw = ChatCompletionsTransport().build_kwargs(
        model="unit-model", messages=[{"role": "user", "content": "unit"}], tools=[],
        provider_name=provider, base_url=base, reasoning_config=reasoning,
        request_overrides=overrides, extra_body_additions={"addition": "keep"},
        provider_profile=ProviderProfile(name="unit-profile") if profile_path else None,
    )
    assert kw["extra_body"]["thinking"] == {"type": expected}
    assert kw["extra_body"]["unit_marker"] == "keep"
    assert kw["extra_body"]["addition"] == "keep"
    assert overrides == before


@pytest.mark.parametrize("profile_path", [False, True], ids=["legacy", "profile"])
@pytest.mark.parametrize("provider,base", [
    ("custom", "https://localhost.evil.invalid/v1"),
    ("custom", "https://127.0.0.1.evil.invalid/v1"),
    ("custom", "http://[::1]:3001/v1"),
    ("custom", "https://cloud.invalid/localhost/127.0.0.1"),
    ("deepseek", "https://api.deepseek.com/v1"),
    ("xunfei", "https://unit-xunfei.invalid/v1"),
    ("custom", None),
])
def test_nonmatching_routes_are_not_given_thinking(profile_path, provider, base):
    kw = ChatCompletionsTransport().build_kwargs(
        model="unit-model", messages=[{"role": "user", "content": "unit"}], tools=[],
        provider_name=provider, base_url=base, reasoning_config={"enabled": False},
        provider_profile=ProviderProfile(name="unit-profile") if profile_path else None,
    )
    assert "thinking" not in kw.get("extra_body", {})


def test_worktree_import_provenance():
    import agent.transports.chat_completions as cc
    assert Path(cc.__file__).resolve() == Path(__file__).resolve().parents[3] / "agent/transports/chat_completions.py"
    print("PS10_IMPORT", cc.__file__, "PYTHON", sys.executable)

@pytest.mark.parametrize("profile_path", [False, True], ids=["legacy", "profile"])
@pytest.mark.parametrize("overrides", [None, {}, {"extra_body": {"unit_override": "keep"}}])
def test_matching_route_retains_generated_body_and_additions(profile_path, overrides):
    from providers import get_provider_profile

    additions = {"must_keep": {"value": "unit-marker"}}
    reasoning = {"enabled": False, "effort": "none"}
    before = deepcopy((overrides, additions, reasoning))
    kw = ChatCompletionsTransport().build_kwargs(
        model="unit-model", messages=[{"role": "user", "content": "unit"}], tools=[],
        provider_name="custom", base_url="http://localhost:11434/v1",
        is_openrouter=True, provider_preferences={"order": ["unit-provider"]},
        provider_profile=get_provider_profile("openrouter") if profile_path else None,
        reasoning_config=reasoning, request_overrides=overrides, extra_body_additions=additions,
    )
    assert kw["extra_body"]["must_keep"] == additions["must_keep"]
    assert kw["extra_body"]["provider"] == {"order": ["unit-provider"]}
    assert kw["extra_body"]["thinking"] == {"type": "disabled"}
    if overrides and overrides.get("extra_body"):
        assert kw["extra_body"]["unit_override"] == "keep"
    assert (overrides, additions, reasoning) == before


@pytest.mark.parametrize("profile_path", [False, True], ids=["legacy", "profile"])
def test_nonmatching_route_retains_existing_override_precedence(profile_path):
    overrides = {"extra_body": {"thinking": {"type": "caller"}, "marker": "override"}}
    additions = {"addition": "keep", "marker": "addition"}
    before = deepcopy((overrides, additions))
    kw = ChatCompletionsTransport().build_kwargs(
        model="unit-model", messages=[], provider_name="custom", base_url="https://remote.invalid/v1",
        request_overrides=overrides, extra_body_additions=additions,
        provider_profile=ProviderProfile(name="unit-profile") if profile_path else None,
    )
    expected = {**additions, **overrides["extra_body"]} if profile_path else overrides["extra_body"]
    assert kw["extra_body"] == expected
    assert (overrides, additions) == before


@pytest.mark.parametrize("provider", ["xunfei-coding", "unit-remote-control"])
@pytest.mark.parametrize("reasoning,expected", [
    (None, "enabled"), ({"enabled": False}, "disabled"),
    ({"enabled": 0}, "enabled"), ({"enabled": "false"}, "enabled"),
    ({"effort": "none"}, "enabled"),
])
def test_production_profile_caller_uses_profile_identity(monkeypatch, provider, reasoning, expected):
    """No manually injected provider_name: the real caller must compose with transport."""
    from types import SimpleNamespace
    from urllib.parse import urlparse
    import providers
    from agent.chat_completion_helpers import _build_chat_completions_kwargs

    profile = ProviderProfile(name=provider)
    lookups = []

    def lookup(name):
        lookups.append(name)
        return profile if name == provider else None

    monkeypatch.setattr(providers, "get_provider_profile", lookup)
    base = "https://unit-remote.invalid/v1"
    transport = ChatCompletionsTransport()
    agent = SimpleNamespace(
        provider=provider, model="unit-model", base_url=base,
        _base_url_lower=base, _base_url_hostname=urlparse(base).hostname,
        session_id="unit-session", max_tokens=None, _ollama_num_ctx=None,
        openrouter_min_coding_score=None, providers_allowed=None, providers_ignored=None,
        providers_order=None, provider_sort=None, provider_require_parameters=False,
        provider_data_collection=None,
        _get_transport=lambda: transport, _is_qwen_portal=lambda: False,
        _is_openrouter_url=lambda: False, _resolved_api_call_timeout=lambda: 30,
        _supports_reasoning_extra_body=lambda: False,
        _max_tokens_param=lambda value: {"max_tokens": value},
        _prepare_messages_for_non_vision_model=lambda messages: messages,
    )
    overrides = {"extra_body": {"must_keep": {"value": "unit-marker"}}}
    before = deepcopy((overrides, reasoning))
    kw = _build_chat_completions_kwargs(agent, [{"role": "user", "content": "unit"}], [],
                                       reasoning, overrides, "unit-cache")
    assert lookups and lookups[0] == provider
    assert kw["extra_body"]["must_keep"] == overrides["extra_body"]["must_keep"]
    if provider == "xunfei-coding":
        assert kw["extra_body"]["thinking"] == {"type": expected}
    else:
        assert "thinking" not in kw["extra_body"]
    assert (overrides, reasoning) == before
