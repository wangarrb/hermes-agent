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
    if profile_path:
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
