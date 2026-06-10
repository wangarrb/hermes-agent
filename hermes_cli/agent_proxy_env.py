"""Environment cleanup for local visible agent sessions."""
from __future__ import annotations

import os
from collections.abc import MutableMapping

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
LOCAL_NO_PROXY = "localhost,127.0.0.1,::1"


def clear_agent_proxy_env(env: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    """Remove inherited proxy settings from an agent subprocess environment."""
    target = os.environ if env is None else env
    for key in PROXY_ENV_KEYS:
        target.pop(key, None)
    target["NO_PROXY"] = LOCAL_NO_PROXY
    target["no_proxy"] = LOCAL_NO_PROXY
    return target


def without_agent_proxy_env(env: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of *env* with proxy settings removed."""
    copied = dict(os.environ if env is None else env)
    clear_agent_proxy_env(copied)
    return copied
