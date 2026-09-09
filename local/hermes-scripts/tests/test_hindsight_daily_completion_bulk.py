import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    name = "offline_hindsight_reflect_consolidate_daily_bulk"
    spec = importlib.util.spec_from_file_location(name, ROOT / "offline_hindsight_reflect_consolidate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args_for(output_dir: Path):
    return argparse.Namespace(
        bank="hermes",
        api="http://127.0.0.1:8888",
        output_dir=str(output_dir),
        max_input_chars=60000,
        llm_model="test-model",
        llm_label="xunfei-coding",
        llm_base_url="http://127.0.0.1:3000/v1",
        no_response_format=False,
        emit_observations=True,
        output_language="zh",
    )


def unit(module, day: str, content: str = "body"):
    return module.ReflectUnit(
        "daily",
        day,
        "hermes",
        0,
        content,
        1,
        [f"fact-{day}"],
        f"{day}T00:00:00",
        f"{day}T23:59:59",
    )


def test_bulk_daily_units_query_all_days_once(tmp_path, monkeypatch):
    module = load_module()
    calls = []

    def fake_query(bank, days):
        calls.append((bank, list(days)))
        return [
            module.FactRecord("f1", "hermes-session::20260507_a", "one", None, None, None, "hermes"),
            module.FactRecord("f2", "hermes-session::20260508_b", "two", None, None, None, "hermes"),
        ]

    monkeypatch.setattr(module, "query_facts_for_days", fake_query)

    result = module.build_daily_fact_units_for_days(args_for(tmp_path), ["2026-05-07", "2026-05-08"])

    assert calls == [("hermes", ["2026-05-07", "2026-05-08"])]
    assert [item.source_ids for item in result["2026-05-07"]] == [["f1"]]
    assert [item.source_ids for item in result["2026-05-08"]] == [["f2"]]


def test_bulk_daily_units_include_profile_qualified_session_ids(tmp_path, monkeypatch):
    module = load_module()
    captured = {}

    def fake_psql(sql):
        captured["sql"] = sql
        return [
            {
                "fact_id": "f-profile",
                "document_id": "hermes-session::planner::20260507_120000_abc",
                "text": "profile fact",
                "fact_type": "technical_lesson",
                "event_date": "",
                "created_at": "",
            }
        ]

    monkeypatch.setattr(module, "psql_json", fake_psql)
    facts = module.query_facts_for_days("hermes", ["2026-05-07"])

    assert "hermes-session::%::20260507%" in captured["sql"]
    assert module.parse_doc_day_topic(facts[0].document_id)[0] == "2026-05-07"
    monkeypatch.setattr(module, "query_facts_for_days", lambda bank, days: facts)
    result = module.build_daily_fact_units_for_days(args_for(tmp_path), ["2026-05-07"])
    assert [item.source_ids for item in result["2026-05-07"]] == [["f-profile"]]


def test_completion_uses_exact_progress_key_and_live_artifact(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "current content")
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda args, days: {"2026-05-07": [expected]})
    output = tmp_path / "daily" / "2026-05-07" / "hermes__00__hash.md"
    output.parent.mkdir(parents=True)
    output.write_text("result", encoding="utf-8")
    key = module.unit_progress_key(expected, args_for(tmp_path))
    progress_entry = module.progress_entry(expected, args_for(tmp_path), output_markdown=str(output))
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "processed_units_v2": {
                    key: progress_entry
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)

    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["details"]["2026-05-07"]["matched_units"] == 1

    different_model_args = args_for(tmp_path)
    different_model_args.llm_model = "different-runtime-model"
    report = module.daily_completion_report(different_model_args, ["2026-05-07"])
    assert report["details"]["2026-05-07"]["matched_units"] == 1

    output.unlink()
    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]

    changed_header_like = unit(module, "2026-05-07", "period: changed fact content")
    assert module.unit_completion_signature(changed_header_like) != module.unit_completion_signature(expected)

    changed = unit(module, "2026-05-07", "changed content with same IDs")
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda args, days: {"2026-05-07": [changed]})
    output.write_text("result", encoding="utf-8")
    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]

def test_completion_accepts_exact_json_hash_and_rejects_legacy_markdown(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "current content")
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda args, days: {"2026-05-07": [expected]})
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "missing-progress.json")
    day_dir = tmp_path / "daily" / "2026-05-07"
    day_dir.mkdir(parents=True)
    legacy = day_dir / "hermes__00__legacy.md"
    legacy.write_text("scope: daily\nperiod: 2026-05-07\ntopic: hermes\n## Source IDs\n- fact-2026-05-07\n", encoding="utf-8")

    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]

    markdown = day_dir / "hermes__00__exact.md"
    markdown.write_text("result", encoding="utf-8")
    sidecar = day_dir / "hermes__00__exact.json"
    sidecar.write_text(
        json.dumps(
            {
                "unit": {
                    "scope": "daily",
                    "period": "2026-05-07",
                    "topic": "hermes",
                    "index": 0,
                    "source_ids": ["fact-2026-05-07"],
                    "input_content_hash": module.short_hash(expected.content, 24),
                },
                "markdown_path": str(markdown),
            }
        ),
        encoding="utf-8",
    )

    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["details"]["2026-05-07"]["matched_units"] == 1

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["unit"]["input_content_hash"] = "stale"
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]


def test_completion_rejects_live_artifact_outside_exact_day_directory(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "current content")
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda args, days: {"2026-05-07": [expected]})
    external = tmp_path / "old-root" / "result.md"
    external.parent.mkdir()
    external.write_text("result", encoding="utf-8")
    key = module.unit_progress_key(expected, args_for(tmp_path))
    progress_entry = module.progress_entry(expected, args_for(tmp_path), output_markdown=str(external))
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "processed_units_v2": {
                    key: progress_entry
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)

    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])

    assert report["missing_days"] == ["2026-05-07"]


def test_run_units_rebuilds_stale_cached_daily_artifact(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "current content")
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    key = module.unit_progress_key(expected, args)
    progress_entry = module.progress_entry(
        expected,
        args,
        output_markdown=str(tmp_path / "daily" / "2026-05-07" / "missing.md"),
    )
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "processed_units_v2": {
                    key: progress_entry
                },
                "processed_document_ids": [],
                "processed_unit_keys": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    calls = []

    def fake_call(args, reflect_unit):
        calls.append(reflect_unit)
        return {"executive_summary": ["rebuilt"]}, '{"executive_summary":["rebuilt"]}'

    monkeypatch.setattr(module, "call_llm_for_unit", fake_call)
    monkeypatch.setattr(module, "post_to_hindsight", lambda *args, **kwargs: (True, None))

    assert module.run_units(args, [expected]) == 0
    assert calls == [expected]
    saved = list((tmp_path / "daily" / "2026-05-07").glob("*.md"))
    assert saved


def test_run_units_rebuilds_daily_entry_when_exact_content_hash_changed(tmp_path, monkeypatch):
    module = load_module()
    cached = unit(module, "2026-05-07", "period: old fact")
    current = unit(module, "2026-05-07", "period: new fact")
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    assert module.unit_progress_key(cached, args) == module.unit_progress_key(current, args)
    existing = tmp_path / "daily" / "2026-05-07" / "cached.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("old", encoding="utf-8")
    key = module.unit_progress_key(cached, args)
    entry = module.progress_entry(cached, args, output_markdown=str(existing))
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "processed_units_v2": {key: entry},
                "processed_document_ids": [],
                "processed_unit_keys": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    calls = []
    monkeypatch.setattr(
        module,
        "call_llm_for_unit",
        lambda args, reflect_unit: (calls.append(reflect_unit) or ({"executive_summary": ["new"]}, "{}")),
    )
    monkeypatch.setattr(module, "post_to_hindsight", lambda *args, **kwargs: (True, None))

    assert module.run_units(args, [current]) == 0
    assert calls == [current]
