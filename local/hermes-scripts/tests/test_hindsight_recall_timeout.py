"""Deterministic request-timeout tests; never contact the Hindsight service."""

import argparse
import importlib.util
import io
import sys
from pathlib import Path

import pytest


@pytest.fixture
def layered():
    path = Path(__file__).resolve().parents[1] / "hindsight_recall_layered.py"
    spec = importlib.util.spec_from_file_location("layered_timeout_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def timeouts(layered, monkeypatch):
    seen = []

    def urlopen(request, *, timeout):
        seen.append(timeout)
        return io.BytesIO(b'{"results": []}')

    monkeypatch.setattr(layered.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(layered, "query_variants", lambda query, mode: ["one", "two", "three"])
    return seen


@pytest.mark.parametrize("entrypoint", ["http_json", "recall", "layered_recall"])
def test_legacy_calls_receive_120_second_default(layered, timeouts, entrypoint):
    if entrypoint == "http_json":
        layered.http_json("http://unit.test", {})
    elif entrypoint == "recall":
        layered.recall("http://unit.test", "bank", "query", 2)
    else:
        layered.layered_recall("http://unit.test", "bank", "query", "mixed", 2, 2)
    assert timeouts
    assert all(timeout == 120 for timeout in timeouts)


def test_explicit_http_timeout_remains_positional_and_keyword_compatible(layered, timeouts):
    layered.http_json("http://unit.test", {}, 45)
    layered.http_json("http://unit.test", {}, timeout=60)
    assert timeouts == [45, 60]


@pytest.mark.parametrize("timeout", [0.5, 120, 180])
def test_custom_timeout_reaches_every_serial_request(layered, timeouts, timeout):
    layered.layered_recall(
        "http://unit.test", "bank", "query", "mixed", 2, 2,
        request_timeout=timeout,
    )
    assert timeouts == [timeout, timeout, timeout]


@pytest.mark.parametrize("extra, expected", [([], 120), (["--request-timeout", "185.5"], 185.5)])
def test_cli_forwards_default_and_custom_timeout_to_http(
    layered, timeouts, monkeypatch, capsys, extra, expected,
):
    monkeypatch.setattr(sys, "argv", [
        "hindsight_recall_layered.py", "query", "--api", "http://unit.test",
        "--no-local-cards", "--no-repair-sidecar", "--json", *extra,
    ])
    layered.main()
    assert timeouts == [expected, expected, expected]
    assert '"results": []' in capsys.readouterr().out


@pytest.mark.parametrize("invalid", ["0", "-1", "nan", "inf", "invalid"])
def test_cli_rejects_nonpositive_or_nonfinite_timeout_before_http(
    layered, timeouts, monkeypatch, capsys, invalid,
):
    monkeypatch.setattr(sys, "argv", [
        "hindsight_recall_layered.py", "query", "--request-timeout", invalid,
        "--no-local-cards", "--no-repair-sidecar",
    ])
    with pytest.raises(SystemExit) as exc:
        layered.main()
    assert exc.value.code == 2
    assert "positive" in capsys.readouterr().err
    assert timeouts == []


def test_direct_invalid_timeout_fails_before_variants(layered, timeouts):
    with pytest.raises((ValueError, argparse.ArgumentTypeError), match="positive"):
        layered.layered_recall(
            "http://unit.test", "bank", "query", "mixed", 2, 2,
            request_timeout=-1,
        )
    assert timeouts == []


def test_default_preserves_existing_recall_substitutions(layered, monkeypatch):
    queries = []

    def legacy_recall(api, bank, query, limit, *, include_observations=True):
        queries.append(query)
        return []

    monkeypatch.setattr(layered, "recall", legacy_recall)
    monkeypatch.setattr(layered, "query_variants", lambda query, mode: ["one", "two"])
    assert layered.layered_recall("api", "bank", "query", "mixed", 2, 2) == []
    assert queries == ["one", "two"]


def test_network_timeout_keeps_existing_error_results(layered, monkeypatch):
    def timed_out(request, *, timeout):
        assert timeout == 180
        raise TimeoutError("deterministic timeout")

    monkeypatch.setattr(layered.urllib.request, "urlopen", timed_out)
    monkeypatch.setattr(layered, "query_variants", lambda query, mode: ["one"])
    result = layered.layered_recall(
        "http://unit.test", "bank", "query", "mixed", 2, 2, request_timeout=180,
    )
    assert len(result) == 1
    assert result[0]["layer"] == "error"
    assert "deterministic timeout" in result[0]["error"]
