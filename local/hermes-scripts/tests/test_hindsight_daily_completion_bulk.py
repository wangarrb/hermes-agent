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
        llm_label="oneapi-auto-cheep",
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


def test_bulk_and_single_day_partition_respect_explicit_event_date(tmp_path, monkeypatch):
    module = load_module()
    args = args_for(tmp_path)
    fact = module.FactRecord(
        "fact-1",
        "hermes-session::session_20260330_source",
        "body",
        "world",
        "2026-09-01T12:00:00+08:00",
        "2026-09-01T12:00:01+08:00",
        "hermes",
    )
    monkeypatch.setattr(module, "query_facts_for_days", lambda bank, days: [fact])

    bulk = module.build_daily_fact_units_for_days(args, ["2026-03-30", "2026-09-01"])
    single_old = module.build_daily_fact_units_for_days(args, ["2026-03-30"])
    single_event = module.build_daily_fact_units_for_days(args, ["2026-09-01"])

    assert bulk["2026-03-30"] == []
    assert [u.source_ids for u in bulk["2026-09-01"]] == [["fact-1"]]
    assert single_old["2026-03-30"] == []
    assert [u.source_ids for u in single_event["2026-09-01"]] == [["fact-1"]]


def test_bulk_and_single_day_partition_use_document_date_only_without_event_date(tmp_path, monkeypatch):
    module = load_module()
    args = args_for(tmp_path)
    fact = module.FactRecord(
        "fact-legacy",
        "hermes-session::planner::20260330_source",
        "body",
        "world",
        None,
        None,
        "hermes",
    )
    monkeypatch.setattr(module, "query_facts_for_days", lambda bank, days: [fact])

    bulk = module.build_daily_fact_units_for_days(args, ["2026-03-30", "2026-09-01"])
    single = module.build_daily_fact_units_for_days(args, ["2026-03-30"])

    assert [u.source_ids for u in bulk["2026-03-30"]] == [["fact-legacy"]]
    assert [u.source_ids for u in single["2026-03-30"]] == [["fact-legacy"]]
    assert bulk["2026-09-01"] == []


def test_completion_rejects_progress_only_without_current_sidecar(tmp_path, monkeypatch):
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
    assert report["details"]["2026-05-07"]["matched_units"] == 0
    assert report["missing_days"] == ["2026-05-07"]

    different_model_args = args_for(tmp_path)
    different_model_args.llm_model = "different-runtime-model"
    assert report["details"]["2026-05-07"]["matched_units"] == 0

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
    markdown.write_text(module.render_markdown(expected, {"executive_summary": ["result"]}, "", model="test-model"), encoding="utf-8")
    sidecar = day_dir / "hermes__00__exact.json"
    sidecar.write_text(
        json.dumps(
            {
                "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
                "unit": {
                    "scope": "daily",
                    "period": "2026-05-07",
                    "topic": "hermes",
                    "index": 0,
                    "source_count": expected.source_count,
                    "source_ids": ["fact-2026-05-07"],
                    "date_range_start": expected.date_range_start,
                    "date_range_end": expected.date_range_end,
                    "input_content_hash": module.short_hash(expected.content, 24),
                },
                "model": "test-model",
                "llm_json": {"executive_summary": ["result"]},
                "raw_text": "",
                "markdown_path": str(markdown),
            }
        ),
        encoding="utf-8",
    )

    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["details"]["2026-05-07"]["matched_units"] == 1

    valid_markdown = markdown.read_text(encoding="utf-8")
    for old, new in (
        ("period: 2026-05-07", "period: 2026-05-08"),
        ("- fact-2026-05-07", "- wrong-source"),
    ):
        markdown.write_text(valid_markdown.replace(old, new, 1), encoding="utf-8")
        report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
        assert report["missing_days"] == ["2026-05-07"]
    markdown.write_text(valid_markdown, encoding="utf-8")

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["unit"]["input_content_hash"] = "stale"
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]


def test_completion_manifest_freezes_unit_signatures_without_raw_text(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "current content")
    args = args_for(tmp_path)
    monkeypatch.setattr(
        module,
        "build_daily_fact_units_for_days",
        lambda current_args, days: {"2026-05-07": [expected]},
    )

    units_by_day, completion = module.build_daily_completion_snapshot(args, ["2026-05-07"])
    manifest = module.write_daily_completion_manifest(
        tmp_path,
        args,
        ["2026-05-07"],
        units_by_day,
        completion,
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["schema_version"] == "daily-completion-manifest-v1"
    assert data["units"]["2026-05-07"][0]["source_ids"] == ["fact-2026-05-07"]
    assert data["units"]["2026-05-07"][0]["input_content_hash"] == module.short_hash(expected.content, 24)
    assert "current content" not in manifest.read_text(encoding="utf-8")


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


def test_run_units_rebuilds_legacy_sidecar_even_when_progress_markdown_is_live(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "current content")
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    output_dir = tmp_path / "daily" / "2026-05-07"
    output_dir.mkdir(parents=True)
    legacy_md = output_dir / "hermes__00__legacy.md"
    legacy_json = output_dir / "hermes__00__legacy.json"
    legacy_md.write_text("scope: daily\nperiod: 2026-05-07\ntopic: hermes\n", encoding="utf-8")
    legacy_json.write_text(
        json.dumps({
            "unit": {
                "scope": "daily",
                "period": "2026-05-07",
                "topic": "hermes",
                "index": 0,
                "source_ids": ["fact-2026-05-07"],
            },
            "markdown_path": str(legacy_md),
        }),
        encoding="utf-8",
    )
    key = module.unit_progress_key(expected, args)
    entry = module.progress_entry(expected, args, output_markdown=str(legacy_md))
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps({
            "processed_units_v2": {key: entry},
            "processed_document_ids": [],
            "processed_unit_keys": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    calls = []
    monkeypatch.setattr(
        module,
        "call_llm_for_unit",
        lambda args, reflect_unit: (calls.append(reflect_unit) or ({"executive_summary": ["rebuilt"]}, "{}")),
    )
    monkeypatch.setattr(module, "post_to_hindsight", lambda *args, **kwargs: (True, None))

    assert module.run_units(args, [expected]) == 0
    assert calls == [expected]


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


def test_weekly_refuses_legacy_daily_gaps_when_backfill_is_not_explicit(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-03-15"
    expected = unit(module, day, "source facts")
    monkeypatch.setattr(module, "retained_sqlite_days", lambda _bank: [day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda _args, days: (
        {day: [expected]},
        {"missing_days": [day], "details": {day: {"expected_units": 1, "missing_units": 1}}},
    ))
    monkeypatch.setattr(module, "write_daily_completion_manifest", lambda *_args: tmp_path / "completion.json")
    run_calls = []
    monkeypatch.setattr(module, "run_units", lambda *args: run_calls.append(args) or 0)
    args = argparse.Namespace(
        weekly_source="daily",
        weekly_window="all-history",
        output_dir=str(tmp_path),
        bank="hermes",
        backfill_missing_daily=False,
        scope="weekly",
        mode="submit",
        date=day,
        daily_source="facts",
        group_by="topic",
    )

    try:
        module.resolve_missing_daily_args(args)
    except SystemExit as exc:
        assert "backfill" in str(exc).lower() or "missing" in str(exc).lower()
    else:
        raise AssertionError("weekly must fail closed instead of processing or backfilling incomplete daily inputs")
    assert run_calls == []


def test_weekly_backfill_runs_only_after_explicit_opt_in(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-03-15"
    expected = unit(module, day, "source facts")
    monkeypatch.setattr(module, "retained_sqlite_days", lambda _bank: [day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda _args, days: (
        {day: [expected]},
        {"missing_days": [day], "details": {day: {"expected_units": 1, "missing_units": 1}}},
    ))
    monkeypatch.setattr(module, "write_daily_completion_manifest", lambda *_args: tmp_path / "completion.json")
    run_calls = []
    monkeypatch.setattr(module, "run_units", lambda _args, units: run_calls.extend(units) or 0)
    monkeypatch.setattr(module, "evaluate_daily_completion", lambda *_args: {"missing_days": []})
    args = argparse.Namespace(
        weekly_source="daily",
        weekly_window="all-history",
        output_dir=str(tmp_path),
        bank="hermes",
        backfill_missing_daily=True,
        scope="weekly",
        mode="submit",
        date=day,
        daily_source="facts",
        group_by="topic",
    )

    module.resolve_missing_daily_args(args)

    assert run_calls == [expected]
