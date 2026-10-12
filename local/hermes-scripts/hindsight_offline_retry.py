"""Standalone offline retry policy (stdlib; bundle this beside the workers).

This is intentionally not an optional import of agent.retry_utils: installed
script bundles do not contain the agent package. Numeric/HTTP-date Retry-After
semantics match that helper, with an injected clock and strict finite deadline.
"""
from __future__ import annotations

import math
import random
import socket
import time
import urllib.error
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Callable


class RemoteCallError(RuntimeError):
    def __init__(self, error_class: str, *, status_code=None, retry_after=None):
        super().__init__(error_class)
        self.error_class = error_class
        self.status_code = status_code
        self.retry_after = retry_after


class RetryDeferred(RemoteCallError):
    def __init__(self, error_class: str, *, next_retry_at: float, attempts: int = 0):
        super().__init__(error_class)
        self.next_retry_at = next_retry_at
        self.attempts = attempts


def parse_retry_after(value, *, now: float) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (str, int, float)):
        value = value.get('Retry-After') or value.get('retry-after')
    if value is None:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            when = parsedate_to_datetime(str(value))
            seconds = when.timestamp() - now
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def classify_error(error) -> tuple[str, bool]:
    """Only structured status/types; never infer gateway backend from text."""
    status = getattr(error, 'status_code', None)
    response = getattr(error, 'response', None)
    if status is None and response is not None:
        status = getattr(response, 'status_code', None)
    if status is not None:
        return f'http_{status}', status in {429, 500, 502, 503, 504}
    category = getattr(error, 'error_class', None)
    if category:
        return category, category in {'transport', 'rate_limit', 'operation_pending'}
    cause = getattr(error, '__cause__', None)
    if isinstance(error, (TimeoutError, ConnectionError, socket.timeout, urllib.error.URLError, OSError)):
        return 'transport', True
    # requests exceptions are a dependency of the existing LLM worker, but not
    # of the standalone ledger/client. Inspect type ancestry, not error text.
    if any(cls.__name__ in {'Timeout', 'ConnectionError', 'RemoteDisconnected'} for cls in type(error).__mro__):
        return 'transport', True
    if cause is not None:
        return classify_error(cause)
    return 'internal_error', False


@dataclass
class RetryPolicy:
    max_attempts: int = 4
    request_timeout: float = 300
    unit_deadline_s: float = 900
    max_calls: int = 128  # Native API requests/polls, not LLM requests.
    max_llm_calls: int | None = 16  # None disables only the LLM count limit.
    base_delay: float = 10
    max_delay: float = 60
    defer_delay: float = 60
    poll_s: float = 5
    clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[], float] = random.random
    calls_used: int = field(default=0, init=False)
    llm_calls_used: int = field(default=0, init=False)

    def __post_init__(self):
        if (type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 4
                or type(self.max_calls) is not int or self.max_calls < 1
                or (self.max_llm_calls is not None
                    and (type(self.max_llm_calls) is not int or self.max_llm_calls < 1))):
            raise ValueError('retry attempts/call cap invalid')
        if not 0 < self.request_timeout <= 300 or not 0 < self.unit_deadline_s <= 900:
            raise ValueError('retry timeout/deadline invalid')
        if any(not math.isfinite(v) or v < 0 for v in (self.base_delay, self.max_delay, self.defer_delay, self.poll_s)):
            raise ValueError('retry delays invalid')

    @property
    def api_calls_used(self):
        return self.calls_used - self.llm_calls_used


class RetryBudget:
    """One unit deadline; actual counts and kind-specific caps span the run."""
    def __init__(self, policy: RetryPolicy):
        self.policy = policy
        self.deadline = policy.clock() + policy.unit_deadline_s
        self.calls = 0
        self._last_kind = 'api'

    def _at_call_cap(self, kind):
        if kind == 'api':
            return self.policy.api_calls_used >= self.policy.max_calls
        return (self.policy.max_llm_calls is not None
                and self.policy.llm_calls_used >= self.policy.max_llm_calls)

    def defer(self, category, *, delay=None, attempts=0):
        delay = self.policy.defer_delay if delay is None else delay
        raise RetryDeferred(category, next_retry_at=self.policy.wall_clock() + delay, attempts=attempts)

    def charge(self, *, kind='api') -> float:
        if kind not in {'api', 'llm'}:
            raise ValueError('invalid request budget category')
        remaining = self.deadline - self.policy.clock()
        if remaining <= 0:
            self.defer('deadline')
        if self._at_call_cap(kind):
            self.defer('llm_call_cap' if kind == 'llm' else 'call_cap')
        self.policy.calls_used += 1
        if kind == 'llm':
            self.policy.llm_calls_used += 1
        self.calls += 1
        self._last_kind = kind
        return min(self.policy.request_timeout, remaining)

    def wait(self, category, attempt, *, retry_after=None):
        delay = retry_after
        if delay is None:
            base = min(self.policy.max_delay, self.policy.base_delay * 2 ** max(0, attempt - 1))
            delay = base * (1 + 0.5 * self.policy.jitter())
        # A valid async pending response is progress waiting, not a failed
        # HTTP attempt. Its polling still obeys the API cap/deadline.
        if category != 'operation_pending' and attempt >= self.policy.max_attempts:
            self.defer(category, delay=max(self.policy.defer_delay, delay), attempts=attempt)
        if self._at_call_cap(self._last_kind):
            self.defer(category, delay=max(self.policy.defer_delay, delay), attempts=attempt)
        if self.policy.clock() + delay >= self.deadline:
            self.defer(category, delay=delay, attempts=attempt)
        self.policy.sleep(delay)


def budget_for(policy=None) -> RetryBudget:
    if isinstance(policy, RetryBudget):
        return policy
    return RetryBudget(policy or RetryPolicy())
