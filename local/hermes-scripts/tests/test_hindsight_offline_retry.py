"""Bounded single-provider LLM requests, including format negotiation."""
import importlib
from email.utils import formatdate
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def llm(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    module = importlib.import_module('offline_hindsight_reflect_consolidate')
    monkeypatch.setattr(module, 'get_llm_key', lambda _: 'fake-key')
    return module


class Response:
    def __init__(self, status, text='', headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}
    def json(self):
        return {'choices': [{'message': {'content': self.text or '{"summary":["ok"]}'}}]}


def invoke(module, **kwargs):
    unit = module.ReflectUnit('daily', 'test', 'topic', 0, 'scratch fact', 1, ['a'], 'test', 'test')
    return module.call_llm(unit, model='fake', base_url='http://invalid.test/v1', api_key_env='FAKE', label='fake', **kwargs)


def make_policy(**kwargs):
    retry = importlib.import_module('hindsight_offline_retry')
    clock = [1000.0]
    sleeps = []
    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay
    return retry.RetryPolicy(clock=lambda: clock[0], wall_clock=lambda: clock[0], sleep=sleep,
                             jitter=lambda: 0, **kwargs), sleeps


@pytest.mark.parametrize('status', [400, 401, 403])
def test_permanent_http_is_one_actual_call(llm, monkeypatch, status):
    calls = []
    monkeypatch.setattr(llm.requests, 'post', lambda *a, **k: calls.append(k) or Response(status, 'secret prompt'))
    p, sleeps = make_policy()
    with pytest.raises(RuntimeError) as exc:
        invoke(llm, retry_policy=p)
    assert len(calls) == 1
    assert p.calls_used == 1
    assert not sleeps
    assert 'secret prompt' not in str(exc.value)


def test_429_propagates_without_broad_exception_swallow(llm, monkeypatch):
    calls = []
    monkeypatch.setattr(llm.requests, 'post', lambda *a, **k: calls.append(k) or Response(429, headers={'Retry-After': '7'}))
    p, sleeps = make_policy()
    with pytest.raises(llm.RateLimitError) as exc:
        invoke(llm, raise_on_429=True, retry_policy=p)
    assert len(calls) == 1
    assert exc.value.retry_after == 7
    assert not sleeps


@pytest.mark.parametrize('header', ['7', formatdate(1007, usegmt=True)])
def test_429_retry_after_seconds_and_date(llm, monkeypatch, header):
    calls = []
    responses = iter([Response(429, headers={'Retry-After': header}), Response(200)])
    monkeypatch.setattr(llm.requests, 'post', lambda *a, **k: calls.append(k) or next(responses))
    p, sleeps = make_policy()
    assert invoke(llm, retry_policy=p)[0]['summary'] == ['ok']
    assert sleeps == [7]
    assert p.calls_used == 2
    assert all(c['timeout'] <= 300 for c in calls)


def test_retry_after_beyond_deadline_defers_without_sleep(llm, monkeypatch):
    retry = importlib.import_module('hindsight_offline_retry')
    monkeypatch.setattr(llm.requests, 'post', lambda *a, **k: Response(429, headers={'Retry-After': '9999'}))
    p, sleeps = make_policy(unit_deadline_s=20)
    with pytest.raises(retry.RetryDeferred) as exc:
        invoke(llm, retry_policy=p)
    assert exc.value.next_retry_at == 10999
    assert p.calls_used == 1
    assert not sleeps


def test_continuous_502_and_format_retry_share_explicit_llm_call_cap(llm, monkeypatch):
    retry = importlib.import_module('hindsight_offline_retry')
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return Response(400, 'response_format unsupported') if len(calls) == 1 else Response(502, 'upstream secret')
    monkeypatch.setattr(llm.requests, 'post', post)
    p, _ = make_policy(max_calls=1, max_llm_calls=3)
    with pytest.raises(retry.RetryDeferred):
        invoke(llm, retry_policy=p)
    assert p.calls_used == len(calls) == 3
    assert p.llm_calls_used == 3 and p.api_calls_used == 0
    assert 'response_format' not in calls[-1]['json']


@pytest.mark.parametrize('raw', ['', 'not json', '{}', 'null'])
def test_invalid_output_is_not_success(llm, monkeypatch, raw):
    class RawResponse(Response):
        def json(self):
            return {'choices': [{'message': {'content': raw}}]}
    calls = []
    monkeypatch.setattr(llm.requests, 'post', lambda *a, **k: calls.append(1) or RawResponse(200))
    p, sleeps = make_policy()
    with pytest.raises(RuntimeError):
        invoke(llm, retry_policy=p)
    assert len(calls) == 1
    assert not sleeps


def test_valid_async_pending_waits_past_error_retry_count_within_unchanged_caps(llm):
    retry = importlib.import_module('hindsight_offline_retry')
    policy, sleeps = make_policy(max_calls=8, poll_s=1)
    budget = retry.RetryBudget(policy)
    for attempt in range(1, 6):
        budget.charge()
        budget.wait('operation_pending', attempt, retry_after=policy.poll_s)
    assert policy.calls_used == 5 and len(sleeps) == 5
    assert policy.llm_calls_used == 0


def test_unlimited_llm_requests_and_retries_do_not_spend_api_capacity(llm, monkeypatch):
    retry = importlib.import_module('hindsight_offline_retry')
    p, sleeps = make_policy(max_calls=1, max_llm_calls=None)
    budget = retry.RetryBudget(p)
    budget.charge()
    with pytest.raises(retry.RetryDeferred, match='operation_pending'):
        budget.wait('operation_pending', 1, retry_after=1)
    assert not sleeps
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        assert p.calls_used == len(calls) + 1
        assert p.llm_calls_used == len(calls) and p.api_calls_used == 1
        return Response(429, headers={'Retry-After': '1'}) if len(calls) % 2 else Response(200)
    monkeypatch.setattr(llm.requests, 'post', post)
    for _ in range(20):
        assert invoke(llm, retry_policy=p)[0]['summary'] == ['ok']
    assert p.calls_used == 41 and p.llm_calls_used == len(calls) == 40
    assert p.api_calls_used == p.max_calls == 1 and sleeps == [1] * 20
    with pytest.raises(retry.RetryDeferred, match='call_cap'):
        budget.charge()
    assert p.calls_used == 41 and budget.calls == 1


@pytest.mark.parametrize('boundary', ['attempts', 'deadline', 'default_limit'])
def test_llm_limits_still_stop_before_extra_io(llm, monkeypatch, boundary):
    retry = importlib.import_module('hindsight_offline_retry')
    settings = {} if boundary == 'default_limit' else {'max_llm_calls': None}
    if boundary == 'deadline':
        settings['unit_deadline_s'] = 5
    p, sleeps = make_policy(max_calls=1, **settings)
    retry.RetryBudget(p).charge()
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        if boundary == 'default_limit':
            return Response(200)
        return Response(502, headers={'Retry-After': '5'})
    monkeypatch.setattr(llm.requests, 'post', post)
    if boundary == 'default_limit':
        for _ in range(16):
            invoke(llm, retry_policy=p)
    with pytest.raises(retry.RetryDeferred) as caught:
        invoke(llm, retry_policy=p)
    expected = {'attempts': 4, 'deadline': 1, 'default_limit': 16}[boundary]
    assert p.calls_used == len(calls) + 1 == expected + 1
    assert p.llm_calls_used == expected and p.api_calls_used == 1
    assert len(sleeps) == (3 if boundary == 'attempts' else 0)
    assert caught.value.error_class == ('llm_call_cap' if boundary == 'default_limit' else 'http_502')
    assert all(c['timeout'] <= min(p.request_timeout, p.unit_deadline_s) for c in calls)


@pytest.mark.parametrize('field', ['max_calls', 'max_llm_calls', 'max_attempts'])
@pytest.mark.parametrize('value', [True, False, 0, -1, 1.5, float('inf'), float('nan'), '16'])
def test_call_limits_require_positive_integers(llm, field, value):
    with pytest.raises(ValueError):
        make_policy(**{field: value})


def test_only_llm_call_limit_can_be_null(llm):
    with pytest.raises(ValueError):
        make_policy(max_calls=None)
    with pytest.raises(ValueError):
        make_policy(max_attempts=None)
