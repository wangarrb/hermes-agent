"""Inert, read-only profile config/claim reader for warnings and diagnostics.

Reads one profile's Gateway config the way the multiplexer reads it — under the profile's home
override and its own secret scope — but as a READ-ONLY reader: no ``gateway.run`` import (Gateway
start-up initialisation mutates the process environment and attempts host writes), no secret-source
hydration (external providers are resolved only by the owning loader), no plugin discovery or
activation, no multiplex activation, no terminal scope. Anything the reader cannot evaluate is
reported as an explicit unknown instead of being silently assumed complete.

Owner of the composed read behind ``gateway_migrate._profile_gateway_config`` and
``profile_channels.shared_channel_credentials``.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator

#: Explicit unknowns: inputs this inert reader could not evaluate.
UNKNOWN_SECRET_SOURCES = "declared-secret-sources-not-hydrated"
UNKNOWN_PARKED_PLUGIN_PLATFORMS = "parked-profile-plugin-platforms-not-evaluated"


@contextlib.contextmanager
def profile_read_scope(profile_home: Path) -> Iterator[None]:
    """Home override + the profile's private secret scope for one read; never hydrates.

    The read-only mirror of ``gateway.run._profile_runtime_scope`` minus hydration, terminal
    policy and multiplex activation: a diagnostic read must not resolve external secret providers,
    install terminal scope, or flip process mode. ``build_profile_secret_scope`` still layers the
    profile's ``.env``, any already-hydrated source snapshot and managed-env policy, so a missing
    token stays missing instead of borrowing the ambient launch credentials.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope

    home_token = set_hermes_home_override(str(profile_home))
    try:
        secrets = build_profile_secret_scope(Path(profile_home))
        secret_token = set_secret_scope(secrets, profile_home=str(profile_home))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


def load_profile_gateway_config(profile_home: Path) -> Any:
    """This profile's ``GatewayConfig`` read the way the multiplexer reads it, without Gateway
    start-up initialisation.

    A PARKED profile stays in the inventory and the duplicate-credential guard (parking must not
    hide a conflict), but it is inert for the host: its config is read without discovering its
    plugins, whose ``register()`` would otherwise run in the live gateway on every boot (#123386).
    Only its plugin-defined platforms fall out of the guard — their credentials cannot be evaluated
    without importing the plugin; builtin bot tokens still collide.

    ``load_gateway_config`` is resolved lazily so the registered ``gateway.config`` attribute is
    the one called (the canonical caller and its monkeypatching tests observe that attribute).
    """
    from gateway.config import load_gateway_config
    from hermes_cli.plugins_discovery import suppress_plugin_discovery
    from hermes_cli.profiles import profile_is_parked

    profile_home = Path(profile_home)
    scope = suppress_plugin_discovery() if profile_is_parked(profile_home) else contextlib.nullcontext()
    with profile_read_scope(profile_home), scope:
        return load_gateway_config()


def profile_claim_unknowns(profile_home: Path) -> tuple[str, ...]:
    """Explicit unknowns for one profile read; never silently assumed complete.

    * ``secrets:`` declared in the profile's config.yaml but no hydrated source snapshot for that
      home: external values are only known after ``env_loader.hydrate_profile_secret_sources``
      resolved them, and this reader never hydrates — so credentials those sources may supply are
      UNKNOWN, not absent.
    * A parked profile's plugin platforms: evaluating their credentials would require the plugin
      discovery/activation this read-only path suppresses.
    """
    from hermes_cli.env_loader import _load_secrets_config, get_secret_source_values
    from hermes_cli.profiles import profile_is_parked

    profile_home = Path(profile_home)
    unknown: list[str] = []
    with contextlib.suppress(Exception):
        if _load_secrets_config(profile_home) and not get_secret_source_values(profile_home):
            unknown.append(UNKNOWN_SECRET_SOURCES)
    if profile_is_parked(profile_home):
        unknown.append(UNKNOWN_PARKED_PLUGIN_PLATFORMS)
    return tuple(unknown)


def profile_effective_claims(profile_home: Path) -> tuple[dict[tuple, str], tuple[str, ...]]:
    """``(claims, unknowns)`` for one profile home — one inert, config-only read."""
    from gateway.credential_claims import effective_credential_claims

    config = load_profile_gateway_config(profile_home)
    return effective_credential_claims(config), profile_claim_unknowns(profile_home)
