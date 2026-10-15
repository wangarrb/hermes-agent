"""PS05 route contracts. Unit-only fake keys; no model requests or responses."""
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    from agent import auxiliary_client as ac
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model: {}\n")
    monkeypatch.setattr(ac, "_ALIAS_TABLE", None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unit-only-dashscope-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    def forbidden(*args, **kwargs):
        pytest.fail("Bailian route must not discover another cloud")
    monkeypatch.setattr(ac, "_try_discovery_chain", forbidden)
    yield home


def test_worktree_import_provenance():
    from agent import auxiliary_client as ac
    expected = Path(__file__).resolve().parents[2]
    assert Path(ac.__file__).resolve() == expected / "agent/auxiliary_client.py"
    print("PS05_IMPORT", ac.__file__, "PYTHON", sys.executable)


@pytest.mark.parametrize("label", ["bailian", " Bailian ", "custom:bailian"])
def test_bailian_resolves_alibaba_without_named_override(label):
    from agent import auxiliary_client as ac
    assert ac._normalize_aux_provider(label) == "alibaba"
    client, model = ac.resolve_provider_client(label, "unit-qwen", api_mode="chat_completions")
    assert client is not None
    assert str(client.base_url).rstrip("/") == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert client.api_key == "unit-only-dashscope-key"
    assert model == "unit-qwen"


@pytest.mark.parametrize("label", ["bailian", "custom:bailian"])
@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
def test_named_bailian_overrides_alias_without_losing_identity(isolated_config, label, mode):
    from agent import auxiliary_client as ac
    endpoint = "https://named-bailian.invalid/v1"
    (isolated_config / "config.yaml").write_text(json.dumps({"providers": {"bailian": {
        "name": "Bailian", "base_url": endpoint, "api_key": "unit-only-named-key", "api_mode": mode,
    }}}))
    client, model = ac.resolve_provider_client(label, "unit-qwen")
    assert client is not None
    if mode == "codex_responses":
        assert isinstance(client, ac.CodexAuxiliaryClient)
    else:
        assert not isinstance(client, ac.CodexAuxiliaryClient)
    assert str(client.base_url).rstrip("/") == endpoint
    assert client.api_key == "unit-only-named-key"
    assert model == "unit-qwen"


@pytest.mark.parametrize("task", ["compression", "vision", "title_generation"])
def test_explicit_bailian_task_routes_to_dashscope(isolated_config, task):
    from agent import auxiliary_client as ac
    (isolated_config / "config.yaml").write_text(json.dumps({"auxiliary": {task: {
        "provider": "bailian", "model": "unit-qwen", "api_mode": "chat_completions",
    }}}))
    provider, model, base, key, mode = ac._resolve_task_provider_model(task)
    client, resolved = ac.resolve_provider_client(provider, model, explicit_base_url=base,
                                                 explicit_api_key=key, api_mode=mode)
    assert client is not None
    assert str(client.base_url).rstrip("/") == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert client.api_key == "unit-only-dashscope-key"
    assert resolved == "unit-qwen"


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
def test_auto_keeps_bailian_main_endpoint_key_and_wire(mode):
    from agent import auxiliary_client as ac
    runtime = {"provider": "bailian", "model": "unit-runtime-qwen",
               "base_url": "https://session-bailian.invalid/v1", "api_key": "unit-only-session-key",
               "api_mode": mode}
    client, model, provider = ac._resolve_auto_route(main_runtime=runtime, task="compression")
    assert client is not None
    assert str(client.base_url).rstrip("/") == runtime["base_url"]
    assert client.api_key == runtime["api_key"]
    assert model == runtime["model"]
    assert provider == "bailian"
    assert isinstance(client, ac.CodexAuxiliaryClient) == (mode == "codex_responses")

@pytest.mark.parametrize("label", ["bailian", "custom:bailian"])
@pytest.mark.parametrize("entry_key", ["bailian", "unit-named-key"])
@pytest.mark.parametrize("flag", [False, "false", 0, True, "absent"])
def test_bailian_disabled_policy_precedes_alias_registry(isolated_config, monkeypatch, label, entry_key, flag):
    """Real config, credentials and router; observe lookup without replacing its result."""
    from agent import auxiliary_client as ac
    from hermes_cli import auth

    entry = {"name": "Bailian", "base_url": "https://named-bailian.invalid/v1",
             "api_key": "unit-only-named-key", "enabled": flag}
    config = {} if flag == "absent" else {"providers": {entry_key: entry}}
    (isolated_config / "config.yaml").write_text(json.dumps(config))
    credential_lookups = []
    real_lookup = auth.resolve_api_key_provider_credentials

    def observe_lookup(*args, **kwargs):
        credential_lookups.append(args)
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(auth, "resolve_api_key_provider_credentials", observe_lookup)
    client, model = ac.resolve_provider_client(label, "unit-qwen", api_mode="chat_completions")
    if flag in (False, "false", 0):
        assert client is None
        assert model is None
        assert credential_lookups == []
    else:
        assert client is not None
        assert model == "unit-qwen"
        if flag == "absent":
            assert client.api_key == "unit-only-dashscope-key"
            assert str(client.base_url).rstrip("/") == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            assert credential_lookups
        else:
            assert client.api_key == "unit-only-named-key"
            assert str(client.base_url).rstrip("/") == entry["base_url"]
            assert credential_lookups == []
