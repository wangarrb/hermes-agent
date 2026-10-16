"""Canonical, inert credential-fingerprint/claim logic shared by the gateway and read-only readers.

The multiplexer's duplicate-credential guard (``GatewayRunner``) and every read-only reader (the
migration preflight, the clone/shared-credential warnings) must compute the SAME
``(platform, fingerprint)`` claims from one implementation. This module owns it so the readers
never import ``gateway.run``: importing that module executes Gateway start-up initialisation
(process-environment mutation, host write attempts) that a warning must not trigger.

INERT CONTRACT: never import ``gateway.run`` or ``dotenv`` here, never hydrate secret sources,
never activate plugins or the multiplexer. Only the canonical algorithm lives here.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

#: Domain-separation prefix of the fingerprint hash; changing it invalidates recorded claims.
FINGERPRINT_SALT = "hermes-mux:"


def credential_fingerprint(adapter: Any) -> Optional[str]:
    """Salted, log-safe hash of an adapter's credential; ``None`` when none is discoverable
    (conflict detection is then skipped)."""
    # Many adapters (Discord) keep the token on `config`; without that fallback the check is skipped.
    candidates = [
        (adapter, attr) for attr in (
            "token", "bot_token", "_token", "api_token", "_bot_token",
            "_project_secret",  # Photon/Spectrum: project credentials, not a bot token
            "_app_id",  # Feishu/Lark app_id — stable, log-safe, already the _app_lock_identity
            "_client_id", "_bot_id",  # Teams / WeCom app-style id pairs
        )
    ] + [(getattr(adapter, "config", None), attr) for attr in ("token", "bot_token")]
    for obj, attr in candidates:
        val = getattr(obj, attr, None)
        if isinstance(val, str) and val.strip():
            import hashlib
            return hashlib.sha256((FINGERPRINT_SALT + val.strip()).encode("utf-8")).hexdigest()[:16]
    return None


def credential_probe(platform_config: Any) -> SimpleNamespace:
    """Config-shaped stand-in for the adapter attributes :func:`credential_fingerprint` probes:
    token/api_key plus the id-style credentials adapters expose from ``extra``."""
    extra = getattr(platform_config, "extra", None) or {}
    return SimpleNamespace(
        token=getattr(platform_config, "token", None) or getattr(platform_config, "api_key", None),
        _app_id=extra.get("app_id"), _client_id=extra.get("client_id"), _bot_id=extra.get("bot_id"),
        _project_secret=extra.get("project_secret"), config=platform_config,
    )


def effective_credential_claims(config: Any) -> dict[tuple, str]:
    """``(platform, fingerprint)`` for every enabled platform with a discoverable credential."""
    claims: dict[tuple, str] = {}
    for platform, platform_config in config.platforms.items():
        if not platform_config.enabled:
            continue
        fp = credential_fingerprint(credential_probe(platform_config))
        if fp is not None:
            claims[(platform.value, fp)] = platform.value
    return claims
