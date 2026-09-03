"""``x-opencode-session`` — OpenCode relay session-affinity header.

OpenCode (opencode.ai Zen/Go/free relay) pins requests that share an
``x-opencode-session`` value to the same upstream backend, which is what
keeps its prompt cache warm across the turns of one conversation. The value
only has to be opaque and consistent per conversation, so it is derived the
same way as the other conversation-affinity hints Hermes already sends
(OpenRouter's sticky ``session_id``, xAI's ``x-grok-conv-id``): the
host-declared routing scope first, then the ambient conversation root, then
the physical session id — normalized through ``_cache_scope_from_session_id``
so cron fires of one job share a scope.

Every OpenCode request — main turn on any transport, auxiliary calls
(compression, titles, vision, MoA) — goes through :func:`opencode_session_headers`
so the header cannot drift per code path.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

OPENCODE_SESSION_HEADER = "x-opencode-session"


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """True when *provider* or *base_url* addresses the OpenCode relay.

    Matches the built-in OpenCode provider names, custom ``opencode-*``
    providers, and any endpoint hosted on ``opencode.ai``.  This local
    v0.20-compatible detection intentionally does not depend on provider
    helpers introduced after the backport target.
    """
    provider_id = str(provider or "").strip().lower()
    if provider_id in {"opencode-go", "opencode-zen", "opencode-free"}:
        return True
    if provider_id.startswith("opencode-"):
        return True
    try:
        host = (urlparse(str(base_url or "")).hostname or "").lower().rstrip(".")
    except Exception:
        return False
    return host == "opencode.ai" or host.endswith(".opencode.ai")


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return ``{"x-opencode-session": <key>}`` for OpenCode targets, else ``{}``."""
    if not is_opencode_target(provider, base_url):
        return {}
    # v0.20 exposes the conversation lineage through portal_tags. Newer
    # releases may add an explicit affinity scope; use it when present but
    # keep this backport independent of that later helper.
    try:
        from agent import portal_tags

        affinity_scope = getattr(portal_tags, "get_affinity_scope", lambda: None)()
        conversation_scope = portal_tags.get_conversation_context()
    except Exception:
        affinity_scope = None
        conversation_scope = None
    key = str(affinity_scope or conversation_scope or session_id or "").strip()
    return {OPENCODE_SESSION_HEADER: key} if key else {}


def merge_opencode_session_headers(
    kwargs: dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Merge the affinity header into ``kwargs["extra_headers"]`` (in place).

    Existing per-request headers win, so a caller-pinned value is preserved.
    Non-OpenCode targets are left untouched.
    """
    headers = opencode_session_headers(provider, base_url, session_id)
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
