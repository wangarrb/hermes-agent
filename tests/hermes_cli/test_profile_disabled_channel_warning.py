"""G04: retained cloned credentials are not active duplicate bot claims.

Offline native Telegram file/claim unit contracts; no adapter is constructed or
started. These cases do not stand in for an external Weixin plugin E2E.
"""
from hashlib import sha256
from pathlib import Path

import pytest
import hermes_yaml as yaml

import hermes_constants
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
    (_raw({"enabled": "false"}), True),  # existing helper uses bool, not a new parser
    ({}, True),  # no explicit disable; env-only profiles can have no config file
], ids=["top-false", "nested-false", "top-false-wins", "top-true-wins",
        "top-unspecified", "top-true", "nested-true", "quoted-false", "unset"])
def test_explicitly_disabled_cloned_channel_does_not_warn_as_duplicate(
        homes, token_source, disabled_side, raw, warns):
    for i, home in enumerate(homes):
        _write(home, raw if i == disabled_side else _raw({"enabled": True}), token_source)
    before = _snapshot(homes)
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
    with gm._multiplex_read_mode():
        claims = [gm._credential_claims(gm._profile_gateway_config(home)) for home in homes]
    for active, actual in zip(enabled, claims):
        assert bool(actual) is active
        assert set(actual.values()) == ({"telegram"} if active else set())
    overlap = set(claims[0]) & set(claims[1])
    assert bool(overlap) is all(enabled)
    expected = ["telegram"] if overlap else []
    assert shared_channel_credentials(homes[1], homes[0]) == expected
    assert shared_channel_credentials(homes[0], homes[1]) == expected
    assert _snapshot(homes) == before
