"""G04: retained cloned credentials are not active duplicate bot claims.

Offline native Telegram file/claim unit contracts; no adapter is constructed or
started. These cases do not stand in for an external Weixin plugin E2E.
"""
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path

import pytest
import hermes_yaml as yaml

import hermes_constants
from agent import secret_scope as ss
from hermes_cli import gateway_migrate as gm
from hermes_cli.profile_channels import shared_channel_credentials

TOKEN = "111111:offline-cloned-fake-token"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    role = root / "profiles" / "role"
    role.mkdir(parents=True)
    return root, role


def _raw(top=None, nested=None):
    raw = {}
    if top is not None:
        raw["platforms"] = {"telegram": top}
    if nested is not None:
        raw["gateway"] = {"platforms": {"telegram": nested}}
    return raw


def _write(home, raw, token_source):
    # Independent file-backed homes, as after an opt-in channel clone.
    raw = yaml.safe_load(yaml.safe_dump(raw))
    env = "# retained fake credential\nUNRELATED_SETTING=preserve-me\n"
    if token_source == "env":
        env += f"TELEGRAM_BOT_TOKEN={TOKEN}\n"
    else:
        raw.setdefault("platforms", {}).setdefault("telegram", {})["token"] = TOKEN
    (home / ".env").write_text(env, encoding="utf-8")
    if raw:
        (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")


def _snapshot(homes):
    return {str(path): (path.read_bytes(), sha256(path.read_bytes()).hexdigest())
            for home in homes for path in (home / ".env", home / "config.yaml")
            if path.is_file()}


@pytest.mark.parametrize("token_source", ["env", "config"])
@pytest.mark.parametrize("disabled_side", [0, 1], ids=["source", "clone"])
@pytest.mark.parametrize("raw, warns", [
    (_raw({"enabled": False}), False),
    (_raw(nested={"enabled": False}), False),
    (_raw({"enabled": False}, {"enabled": True}), False),
    (_raw({"enabled": True}, {"enabled": False}), True),
    (_raw({}, {"enabled": False}), False),
    (_raw({"enabled": True}), True),
    (_raw(nested={"enabled": True}), True),
    (_raw({"enabled": "false"}), False),  # approved real Gateway typed semantics, not raw bool
    ({}, True),  # no explicit disable; env-only profiles can have no config file
], ids=["top-false", "nested-false", "top-false-wins", "top-true-wins",
        "top-unspecified", "top-true", "nested-true", "quoted-false", "unset"])
def test_explicitly_disabled_cloned_channel_does_not_warn_as_duplicate(
        homes, token_source, disabled_side, raw, warns):
    for i, home in enumerate(homes):
        _write(home, raw if i == disabled_side else _raw({"enabled": True}), token_source)
    before = _snapshot(homes)
    # Config-only token presence does not auto-enable the canonical adapter. Keep
    # the old unset control, but assert its real typed claim rather than raw inventory.
    if raw == {} and token_source == "config":
        assert not _claims(homes)[disabled_side]
        warns = False
    expected = ["telegram"] if warns else []
    assert shared_channel_credentials(homes[1], homes[0]) == expected
    assert shared_channel_credentials(homes[0], homes[1]) == expected
    assert _snapshot(homes) == before


@pytest.mark.parametrize("token_source", ["env", "config"])
@pytest.mark.parametrize("enabled", [(True, True), (False, True), (True, False), (False, False)])
def test_warning_inventory_agrees_with_real_migration_claim_builder(homes, token_source, enabled):
    for home, active in zip(homes, enabled):
        _write(home, _raw({"enabled": active}), token_source)
    before = _snapshot(homes)
    claims = _claims(homes)
    for active, actual in zip(enabled, claims):
        assert bool(actual) is active
        assert set(actual.values()) == ({"telegram"} if active else set())
    overlap = set(claims[0]) & set(claims[1])
    assert bool(overlap) is all(enabled)
    expected = ["telegram"] if overlap else []
    assert shared_channel_credentials(homes[1], homes[0]) == expected
    assert shared_channel_credentials(homes[0], homes[1]) == expected
    assert _snapshot(homes) == before


def _claims(homes):
    token = ss.set_multiplex_context(True)
    try:
        return [gm._credential_claims(gm._profile_gateway_config(home)) for home in homes]
    finally:
        ss.reset_multiplex_context(token)


def _state():
    return (dict(os.environ), hermes_constants.get_hermes_home_override(),
            ss.current_secret_scope(), ss.current_secret_scope_home(),
            ss._MULTIPLEX_CONTEXT.get(), ss._MULTIPLEX_ACTIVE,
            hermes_constants._PINNED_PROCESS_HERMES_HOME, ss._AUTO_PINNED_HOME)


@contextmanager
def _surrounding(home, multiplex):
    before = _state()
    mapping = {"TELEGRAM_BOT_TOKEN": "999999:surrounding-fake-token"}
    ht = hermes_constants.set_hermes_home_override(home)
    st = ss.set_secret_scope(mapping, profile_home=str(home))
    mt = ss.set_multiplex_context(multiplex)
    surrounding = _state()
    try:
        yield
    finally:
        try:
            assert _state() == surrounding
            assert ss.current_secret_scope() is mapping
        finally:
            ss.reset_multiplex_context(mt)
            ss.reset_secret_scope(st)
            hermes_constants.reset_hermes_home_override(ht)
        assert _state() == before


def _write_shape(home, shape, token=TOKEN):
    raw = {}
    env = "# synthetic retained input\nUNRELATED_SETTING=preserve-me\n"
    if shape == "env-bare":
        env += f"TELEGRAM_BOT_TOKEN={token}\n"
    elif shape == "env-single":
        env += f"export TELEGRAM_BOT_TOKEN='{token}'\n"
    elif shape == "env-double":
        env += f'TELEGRAM_BOT_TOKEN="{token}" # retained comment\n'
    elif shape == "top-token":
        raw = _raw({"enabled": True, "token": token})
    elif shape == "nested-token":
        raw = _raw(nested={"enabled": True, "token": token})
    elif shape == "shorthand-token":
        raw = {"gateway": {"telegram": {"enabled": True, "token": token}}}
    else:
        assert shape == "missing"
    (home / ".env").write_text(env, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")


@pytest.mark.parametrize("left,right", [
    ("env-bare", "top-token"), ("env-single", "nested-token"),
    ("env-double", "shorthand-token"), ("top-token", "nested-token"),
    ("nested-token", "shorthand-token"), ("shorthand-token", "env-bare"),
])
def test_all_credential_shapes_share_real_connecting_claims(homes, left, right):
    for home, shape in zip(homes, (left, right)):
        _write_shape(home, shape)
    before = _snapshot(homes)
    claims = _claims(homes)
    assert all(set(c.values()) == {"telegram"} for c in claims)
    assert claims[0] == claims[1]
    assert shared_channel_credentials(*homes) == ["telegram"]
    assert shared_channel_credentials(*reversed(homes)) == ["telegram"]
    assert _snapshot(homes) == before


@pytest.mark.parametrize("token_source", ["env", "config"])
@pytest.mark.parametrize("side", [0, 1])
def test_gateway_shorthand_overrides_top_false(homes, token_source, side):
    for i, home in enumerate(homes):
        raw = _raw({"enabled": False if i == side else True})
        if i == side:
            raw["gateway"] = {"telegram": {"enabled": True}}
        _write(home, raw, token_source)
    before = _snapshot(homes)
    claims = _claims(homes)
    assert claims[0] and claims[0] == claims[1]
    assert shared_channel_credentials(*homes) == ["telegram"]
    assert shared_channel_credentials(*reversed(homes)) == ["telegram"]
    assert _snapshot(homes) == before


@pytest.mark.parametrize("side", [0, 1])
def test_distinct_env_tokens_override_identical_yaml(homes, side):
    for i, home in enumerate(homes):
        _write(home, _raw({"enabled": True, "token": TOKEN}), "env")
        (home / ".env").write_text(f"TELEGRAM_BOT_TOKEN={i + side + 222222}:distinct-{i}\n")
    before = _snapshot(homes)
    claims = _claims(homes)
    assert all(claims)
    assert not (claims[0].keys() & claims[1].keys())
    assert shared_channel_credentials(*homes) == []
    assert shared_channel_credentials(*reversed(homes)) == []
    assert _snapshot(homes) == before


@pytest.mark.parametrize("multiplex", [False, True])
@pytest.mark.parametrize("process_active", [False, True])
@pytest.mark.parametrize("missing", [(True, True), (False, True), (True, False)])
def test_missing_tokens_never_borrow_ambient_or_outer_scope(
        homes, monkeypatch, multiplex, process_active, missing):
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", process_active)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", str(homes[0]))
    for home, absent in zip(homes, missing):
        _write_shape(home, "missing" if absent else "env-bare")
    before = _snapshot(homes)
    with _surrounding(homes[1], multiplex):
        claims = _claims(homes)
        assert [bool(c) for c in claims] == [not absent for absent in missing]
        assert shared_channel_credentials(*homes) == []
        assert shared_channel_credentials(*reversed(homes)) == []
    assert _snapshot(homes) == before


@pytest.mark.parametrize("fail_at", [None, 1, 2], ids=["success", "first-loader", "second-loader"])
@pytest.mark.parametrize("multiplex", [False, True])
@pytest.mark.parametrize("process_active", [False, True])
def test_canonical_loader_restores_real_scope_on_success_and_exception(
        homes, monkeypatch, fail_at, multiplex, process_active):
    from gateway import config as gc
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", process_active)
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", str(homes[0]))
    for home in homes:
        _write_shape(home, "env-bare")
    before = _snapshot(homes)
    calls = []
    real_load = gc.load_gateway_config

    def observing_load():
        # Real canonical caller has installed this profile's home and secret scope.
        home = homes[len(calls)]
        assert hermes_constants.get_hermes_home() == home
        assert ss.current_secret_scope_home() == str(home)
        assert ss.get_secret("TELEGRAM_BOT_TOKEN") == TOKEN
        assert ss.is_multiplex_active()
        result = real_load()
        assert gm._credential_claims(result)
        calls.append(home)
        if len(calls) == fail_at:
            raise RuntimeError("injected after real canonical load")
        return result

    monkeypatch.setattr(gc, "load_gateway_config", observing_load)
    with _surrounding(homes[0], multiplex):
        if fail_at is None:
            assert shared_channel_credentials(*homes) == ["telegram"]
            assert calls == list(homes)
        else:
            with pytest.raises(RuntimeError, match="injected after real canonical load"):
                shared_channel_credentials(*homes)
            assert calls == list(homes[:fail_at])
    assert _snapshot(homes) == before
