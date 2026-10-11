import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stalled_consolidation_fails_closed_after_trigger_budget(monkeypatch, capsys):
    module = load_module("hindsight_wait_native_consolidation_test", "hindsight_wait_native_consolidation.py")
    monkeypatch.setattr(module, "snapshot", lambda args: {
        "bank": "hermes",
        "pending_consolidation": 5,
        "failed_consolidation": 0,
        "operations": {"pending": 0, "processing": 0, "failed_historical": 0, "active_or_pending": 0},
        "ready": False,
        "last_consolidated_at": None,
    })
    monkeypatch.setattr(module, "trigger_consolidation", lambda *args, **kwargs: {"operation_id": "op-test"})
    monkeypatch.setattr(module.time, "sleep", lambda *_args: None)

    rc = module.main([
        "--api-url", "http://unused",
        "--bank", "hermes",
        "--stall-cycles", "1",
        "--max-triggers", "1",
        "--poll-s", "1",
        "--json",
    ])

    assert rc == 2
    final_report = json.loads(capsys.readouterr().out)
    assert final_report["ready"] is False
    assert final_report["trigger_stale_exit"] is True
    assert final_report["pending_consolidation"] == 5



def test_reduce_parse_errors_are_hard_gate_input():
    module = load_module("hindsight_offline_v2_rebuild_test", "hindsight_offline_v2_rebuild.py")
    assert module.reduce_parse_error_count({"stats": {"parse_errors": 0}}) == 0
    assert module.reduce_parse_error_count({"stats": {"parse_errors": 1}}) == 1
    for report in ({}, {"stats": {}}, {"stats": {"parse_errors": "bad"}}, {"stats": {"parse_errors": -1}}, {"stats": {"parse_errors": True}}):
        try:
            module.reduce_parse_error_count(report)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"unknown/malformed parse_errors must fail closed: {report!r}")


def test_local_reducer_does_not_replace_cards_when_parse_errors_exist(monkeypatch, tmp_path, capsys):
    module = load_module("hindsight_offline_v2_reduce_gate_test", "hindsight_offline_v2_reduce.py")
    output = tmp_path / "cards"
    output.mkdir()
    marker = output / "existing-card.json"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(module, "collect_observations", lambda *_args, **_kwargs: ([], {"parse_errors": 1, "input_files": 1}))
    monkeypatch.setattr(module, "load_alias_map", lambda *_args, **_kwargs: ({}, {}, {}))
    monkeypatch.setattr(module, "normalize_observation_aliases", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "build_cards", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "write_cards", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not write partial cards")))
    monkeypatch.setattr(sys, "argv", [
        "hindsight_offline_v2_reduce.py", "--mode", "local", "--offline-root", str(tmp_path),
        "--output-root", str(output), "--alias-map", "", "--json",
    ])

    rc = module.main()

    report = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert report["writes"] == []
    assert "parse_errors" in report["write_blocked_reason"]
    assert marker.read_text(encoding="utf-8") == "keep"


def test_local_rebuild_gives_conflict_audit_one_hour_budget(monkeypatch, tmp_path):
    rebuild = load_module("hindsight_offline_v2_rebuild_timeout_test", "hindsight_offline_v2_rebuild.py")
    conflict_timeouts = []
    audit_timeouts = []

    def fake_run_json(cmd, **kwargs):
        command = " ".join(map(str, cmd))
        if "hindsight_offline_v2_reduce.py" in command:
            return {"stats": {"parse_errors": 0}, "card_count": 1, "collected_observations": 1}
        if "hindsight_conflict_audit.py" in command:
            conflict_timeouts.append(kwargs.get("timeout"))
            return {"decision": "pass", "json_path": str(tmp_path / "conflict.json")}
        if "hindsight_eval.py" in command:
            output_dir = cmd[cmd.index("--output-dir") + 1]
            return {"json_path": str(Path(output_dir) / "report.json"), "summary": {}}
        if "hindsight_offline_v2_gate.py" in command:
            return {"decision": "blocked_keep_local_only", "json_path": str(tmp_path / "gate.json")}
        if "hindsight_offline_v2_audit.py" in command:
            audit_timeouts.append(kwargs.get("timeout"))
            return {"ok": True}
        raise AssertionError(command)

    monkeypatch.setattr(rebuild, "run_json", fake_run_json)
    monkeypatch.setattr(rebuild, "run_text", lambda *args, **kwargs: "status ok")
    generic = tmp_path / "generic.jsonl"
    local = tmp_path / "local.jsonl"
    generic.write_text("{}\n", encoding="utf-8")
    local.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "hindsight_offline_v2_rebuild.py", "--mode", "local",
        "--output-dir", str(tmp_path / "out"),
        "--cards-root", str(tmp_path / "cards"),
        "--generic-benchmark", str(generic),
        "--local-benchmark", str(local),
        "--json",
    ])

    rebuild.main()

    assert conflict_timeouts == [3600]
    assert audit_timeouts == [3600]
