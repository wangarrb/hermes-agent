import argparse
import hashlib
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


def test_retained_daily_fact_days_includes_event_dates_and_only_legacy_null_event_dates(monkeypatch):
    module = load_module()
    captured = []

    def fake_psql(sql):
        captured.append(sql)
        if "event_date IS NULL" in sql:
            return ["2026-03-30"]
        if "event_date::date" in sql:
            return ["2026-10-09"]
        return []

    monkeypatch.setattr(module, "psql_json", fake_psql)
    days = module.retained_daily_fact_days("hermes")

    assert days == ["2026-03-30", "2026-10-09"]
    legacy_sql = next(sql for sql in captured if "event_date IS NULL" in sql)
    event_sql = next(sql for sql in captured if "event_date::date" in sql)
    assert "FROM memory_units" in legacy_sql
    assert "fact_type IN ('world', 'experience')" in legacy_sql
    assert "hermes-offline-consolidation%" in legacy_sql
    assert "hermes-offline-canonical%" in legacy_sql
    assert "event_date IS NOT NULL" in event_sql
    assert "fact_type IN ('world', 'experience')" in event_sql
    assert "hermes-offline-consolidation%" in event_sql
    assert "hermes-offline-canonical%" in event_sql


def test_all_history_weekly_span_includes_latest_daily_output_day(tmp_path, monkeypatch):
    module = load_module()
    output_days = ["1950-01-01", "2026-03-30", "2026-10-10"]
    captured = {}
    args = argparse.Namespace(
        bank="hermes",
        output_dir=str(tmp_path),
        weekly_window="all-history",
        weekly_source="daily",
        weekly_group_by="topic",
        max_input_chars=60000,
        _daily_candidate_days=["1950-01-01", "2026-09-24", "2026-10-10"],
    )
    monkeypatch.setattr(module, "existing_daily_output_days", lambda _output_dir: output_days)

    def fake_build(output_dir, period, start, end, max_input_chars, group_by, *, allowed_days=None):
        captured.update(period=period, start=start, end=end, group_by=group_by, allowed_days=allowed_days)
        return []

    monkeypatch.setattr(module, "build_weekly_units_from_daily", fake_build)
    _units, period, start, end, _skipped = module.build_weekly_for_args(args)

    assert period == module.history_period_for(["2026-03-30", "2026-09-24", "2026-10-10"])
    assert start.date().isoformat() == "2026-03-30"
    assert end.date().isoformat() == "2026-10-11"
    assert captured["period"] == period
    assert captured["allowed_days"] == ["2026-03-30", "2026-09-24", "2026-10-10"]


def test_weekly_daily_aggregation_reads_only_allowed_days_inside_span(tmp_path, monkeypatch):
    module = load_module()
    dates = ["1960-01-01", "1988-01-01", "2000-01-01"]
    markers = {"1960-01-01": "KEEP_1960", "1988-01-01": "ANOMALY_1988", "2000-01-01": "KEEP_2000"}
    for day in dates:
        day_dir = tmp_path / "daily" / day
        day_dir.mkdir(parents=True)
        (day_dir / "topic__00__daily.md").write_text(markers[day], encoding="utf-8")
    monkeypatch.setattr(module, "existing_daily_output_days", lambda _output_dir: list(dates))
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        weekly_window="all-history",
        weekly_source="daily",
        weekly_group_by="all",
        max_input_chars=60000,
        _daily_candidate_days=[],
    )

    units, _period, start, _end, _skipped = module.build_weekly_for_args(args)

    content = "\n".join(unit.content for unit in units)
    assert start.date().isoformat() == "1960-01-01"
    assert "KEEP_1960" in content and "KEEP_2000" in content
    assert "ANOMALY_1988" not in content


def test_all_history_facts_weekly_scope_respects_checkpoint_watermark(tmp_path, monkeypatch):
    module = load_module()
    baseline, post, latest = "2026-09-23", "2026-09-24", "2026-10-10"
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: ["1950-01-01", baseline, post, latest])
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    captured = {}
    monkeypatch.setattr(module, "query_facts_for_days", lambda bank, days: captured.update({"bank": bank, "days": list(days)}) or [])
    monkeypatch.setattr(module, "build_fact_units", lambda *_args, **_kwargs: [])
    args = argparse.Namespace(
        bank="hermes",
        output_dir=str(tmp_path),
        weekly_window="all-history",
        weekly_source="facts",
        weekly_group_by="topic",
        max_input_chars=60000,
        backfill_missing_daily=False,
        scope="weekly",
        mode="submit",
        date="2026-10-10",
        week="2026-W41",
        daily_source="facts",
        group_by="topic",
    )
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda *_args: (_ for _ in ()).throw(AssertionError("facts source must not run daily catch-up")))

    module.resolve_missing_daily_args(args)
    checkpoint = json.loads((tmp_path / "daily_completion_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["last_reconciliation"]["post_checkpoint_dates"] == [post, latest]
    assert checkpoint["last_reconciliation"]["excluded_anomalous_dates"] == ["1950-01-01"]
    _units, _period, start, end, _skipped = module.build_weekly_for_args(args)

    assert captured["days"] == [post, latest]
    assert start.date().isoformat() == post
    assert end.date().isoformat() == "2026-10-11"


def test_all_history_facts_source_fails_closed_without_a_checkpoint(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: ["1950-01-01", "2026-03-15"])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda *_args: (_ for _ in ()).throw(AssertionError("must fail before scanning facts")))
    args = _checkpoint_args(tmp_path)
    args.weekly_source = "facts"

    try:
        module.resolve_missing_daily_args(args)
    except SystemExit as exc:
        assert "no verified all-history daily checkpoint" in str(exc).lower()
    else:
        raise AssertionError("direct facts mode must not replay unbounded history without an explicit opt-in")


def test_explicit_all_history_scopes_exclude_known_anomaly_dates(tmp_path, monkeypatch):
    module = load_module()
    anomalies = ["1950-01-01", "1988-01-01"]
    baseline, post = "2026-03-15", "2026-03-16"
    raw_inventory = anomalies + [baseline, post]
    expected_days = [baseline, post]
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: list(raw_inventory))
    monkeypatch.setattr(module, "checkpointed_daily_artifact_gaps", lambda *_args, **_kwargs: [])

    daily_root = tmp_path / "daily-source"
    _write_checkpoint(module, daily_root, baseline_through=baseline)
    captured_daily_days = []
    monkeypatch.setattr(
        module,
        "build_daily_completion_snapshot",
        lambda _args, days: (captured_daily_days.extend(days) or {day: [] for day in days}, {"missing_days": [], "details": {}}),
    )
    daily_args = args_for(daily_root)
    daily_args.weekly_source = "daily"
    daily_args.weekly_window = "all-history"
    daily_args.backfill_missing_daily = True
    daily_args.scope = "weekly"
    daily_args.mode = "submit"
    daily_args.budget_json = False
    module.resolve_missing_daily_args(daily_args)
    daily_checkpoint = json.loads((daily_root / module.DAILY_CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
    assert captured_daily_days == expected_days
    assert not set(anomalies) & set(daily_checkpoint["completed_days"])
    daily_audit = daily_checkpoint["last_reconciliation"]
    assert daily_audit["discovered_dates_hash"] == module.date_list_hash(raw_inventory)
    assert daily_audit["post_checkpoint_dates"] == expected_days
    assert daily_audit["excluded_anomalous_dates"] == anomalies
    assert module.load_daily_completion_checkpoint(daily_root, "hermes")[0]["last_reconciliation"] == daily_audit

    facts_root = tmp_path / "facts-source"
    _write_checkpoint(module, facts_root, baseline_through=baseline)
    facts_args = args_for(facts_root)
    facts_args.weekly_source = "facts"
    facts_args.weekly_window = "all-history"
    facts_args.backfill_missing_daily = True
    facts_args.scope = "weekly"
    facts_args.mode = "submit"
    facts_args.budget_json = False
    module.resolve_missing_daily_args(facts_args)
    facts_checkpoint = json.loads((facts_root / module.DAILY_CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
    facts_audit = facts_checkpoint["last_reconciliation"]
    assert facts_args._weekly_fact_days == expected_days
    assert facts_audit["discovered_dates_hash"] == module.date_list_hash(raw_inventory)
    assert facts_audit["excluded_anomalous_dates"] == anomalies
    assert module.load_daily_completion_checkpoint(facts_root, "hermes")[0]["last_reconciliation"] == facts_audit


def test_daily_reconciliation_audits_anomaly_only_in_existing_outputs(tmp_path, monkeypatch):
    module = load_module()
    baseline, post, anomaly = "2026-03-15", "2026-03-16", "1988-01-01"
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [baseline, post])
    monkeypatch.setattr(module, "existing_daily_output_days", lambda _output_dir: [anomaly, baseline])
    monkeypatch.setattr(module, "checkpointed_daily_artifact_gaps", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        module,
        "build_daily_completion_snapshot",
        lambda _args, days: ({day: [] for day in days}, {"missing_days": [], "details": {}}),
    )
    args = args_for(tmp_path)
    args.weekly_source = "daily"
    args.weekly_window = "all-history"
    args.scope = "weekly"
    args.mode = "submit"
    args.budget_json = False

    module.resolve_missing_daily_args(args)

    checkpoint = json.loads((tmp_path / module.DAILY_CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
    audit = checkpoint["last_reconciliation"]
    assert audit["excluded_anomalous_dates"] == [anomaly]
    assert audit["excluded_anomalous_dates_hash"] == module.date_list_hash([anomaly])


def test_direct_daily_facts_scope_rejects_known_anomalous_dates(tmp_path, monkeypatch):
    module = load_module()
    queried = []
    monkeypatch.setattr(module, "query_facts_for_days", lambda _bank, days: queried.extend(days) or [])
    for day in ("1950-01-01", "1988-01-01"):
        args = args_for(tmp_path)
        args.date = day
        args.daily_source = "facts"
        args.group_by = "topic"
        try:
            module.build_daily_for_args(args)
        except SystemExit as exc:
            assert "anomalous" in str(exc).lower()
        else:
            raise AssertionError(f"direct daily facts processing must reject known anomalous date {day}")
    assert queried == []


def test_direct_daily_raw_scope_rejects_known_anomalous_dates_before_reading_sessions(tmp_path, monkeypatch):
    module = load_module()
    def unexpected_raw_read(*_args, **_kwargs):
        raise AssertionError("anomalous dates must be rejected before reading raw sessions")
    monkeypatch.setattr(module, "build_records_between", unexpected_raw_read)

    for day in ("1950-01-01", "1988-01-01"):
        args = args_for(tmp_path)
        args.date = day
        args.daily_source = "raw"
        args.group_by = "topic"
        args.db = str(tmp_path / "state.db")
        args.no_main = False
        args.prefilter = "safe"
        args.prefilter_threshold = None
        try:
            module.build_daily_for_args(args)
        except SystemExit as exc:
            assert "anomalous" in str(exc).lower()
        else:
            raise AssertionError(f"direct daily raw processing must reject known anomalous date {day}")


def test_facts_empty_increment_persists_date_reconciliation(tmp_path, monkeypatch):
    module = load_module()
    baseline, uncovered, anomaly = "2026-03-15", "2004-09-01", "1950-01-01"
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [anomaly, uncovered, baseline])
    monkeypatch.setattr(module, "checkpointed_daily_artifact_gaps", lambda *_args, **_kwargs: [])
    args = _checkpoint_args(tmp_path)
    args.weekly_source = "facts"
    args.weekly_window = "all-history"
    args.mode = "submit"
    args.budget_json = False

    module.resolve_missing_daily_args(args)

    checkpoint, _ = module.load_daily_completion_checkpoint(tmp_path, "hermes")
    audit = checkpoint["last_reconciliation"]
    assert audit["discovered_date_count"] == 3
    assert audit["discovered_dates_hash"] == module.date_list_hash([anomaly, uncovered, baseline])
    assert audit["post_checkpoint_dates"] == []
    assert audit["post_checkpoint_dates_hash"] == module.date_list_hash([])
    assert audit["uncovered_pre_checkpoint_dates"] == [uncovered]
    assert audit["excluded_anomalous_dates"] == [anomaly]


def test_explicit_facts_repair_provisionalizes_broken_daily_baseline(tmp_path, monkeypatch):
    module = load_module()
    baseline, post = "2026-03-15", "2026-03-16"
    checkpoint = module.load_daily_completion_checkpoint(tmp_path, "hermes")[0]
    checkpoint = module.record_completed_days(
        checkpoint,
        {baseline: [unit(module, baseline, "archived facts")]},
        source="verified_frozen_completion_scope",
        advance_through=baseline,
    )
    checkpoint_path = tmp_path / module.DAILY_CHECKPOINT_FILENAME
    module.save_json(checkpoint_path, checkpoint)
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [baseline, post])
    monkeypatch.setattr(module, "checkpointed_daily_artifact_gaps", lambda *_args, **_kwargs: [baseline])

    args = args_for(tmp_path)
    args.weekly_source = "facts"
    args.weekly_window = "all-history"
    args.weekly_group_by = "topic"
    args.backfill_missing_daily = True
    args.scope = "weekly"
    args.mode = "submit"
    args.budget_json = False
    module.resolve_missing_daily_args(args)

    persisted, _ = module.load_daily_completion_checkpoint(tmp_path, "hermes")
    assert persisted["baseline_from"] is None
    assert persisted["baseline_through"] is None
    assert baseline not in persisted["completed_days"]
    audit = persisted["last_reconciliation"]
    assert audit["scope"] == "explicit-history-repair"
    assert audit["prior_baseline_through"] == baseline
    assert audit["provisionalized_checkpoint_days"] == [baseline]
    assert args._weekly_fact_days == [baseline, post]


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

def test_daily_post_progress_is_scoped_by_bank(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "same daily facts across banks")
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    posts = []
    monkeypatch.setattr(module, "call_llm_for_unit", lambda *_args: ({"executive_summary": ["result"]}, "raw"))
    monkeypatch.setattr(module, "post_to_hindsight", lambda _api, bank, _result, **_kwargs: posts.append(bank) or (True, None))

    def args_for_bank(bank):
        args = args_for(tmp_path)
        args.bank = bank
        args.mode = "submit"
        args.force_repost = False
        args.concurrency = 1
        args.min_concurrency = 1
        args.rate_limit_backoff_seconds = 0
        args.delay = 0
        return args

    hermes_args = args_for_bank("hermes")
    other_args = args_for_bank("other-bank")
    assert module.unit_progress_key(expected, hermes_args) != module.unit_progress_key(expected, other_args)
    assert module.run_units(hermes_args, [expected]) == 0
    assert module.run_units(other_args, [expected]) == 0
    assert posts == ["hermes", "other-bank"]
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert set(progress["processed_document_ids_by_bank"]) == {"hermes", "other-bank"}
    assert {entry["bank"] for entry in progress["processed_units_v2"].values()} == {"hermes", "other-bank"}


def test_unbanked_legacy_daily_progress_is_only_compatible_with_default_bank(tmp_path):
    module = load_module()
    day = "2026-05-07"
    content = f"period: {day}\ndate_range: {day}T00:00:00 to {day}T23:59:59\nlegacy body"
    expected = unit(module, day, content)
    args = args_for(tmp_path)
    result = module.save_result(tmp_path, expected, {"executive_summary": ["legacy"]}, "raw", model=args.llm_model)
    entry = module.progress_entry(expected, args, document_id=result.document_id, output_markdown=str(result.output_md_path))
    entry.pop("bank", None)
    sidecar_path = result.output_json_path
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.pop("artifact_schema_version", None)
    sidecar["unit"].pop("input_content_hash", None)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    entries = {"legacy-unbanked-entry": entry}

    assert not module.legacy_daily_progress_entry_is_reusable(tmp_path, expected, entries, bank="hermes")
    assert module.legacy_daily_progress_entry_is_reusable(
        tmp_path,
        expected,
        entries,
        bank="hermes",
        allow_unbanked_default_bank_migration=True,
    )
    assert not module.legacy_daily_progress_entry_is_reusable(
        tmp_path,
        expected,
        entries,
        bank="other-bank",
        allow_unbanked_default_bank_migration=True,
    )


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
    doc_id = "hermes-offline-consolidation::daily::2026-05-07::hermes::00::verified"
    markdown.write_text(module.render_markdown(expected, {"executive_summary": ["result"]}, "", model="test-model"), encoding="utf-8")
    sidecar = day_dir / "hermes__00__exact.json"
    sidecar.write_text(
        json.dumps(
            {
                "document_id": doc_id,
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
    assert report["missing_days"] == ["2026-05-07"]

    args = args_for(tmp_path)
    key = module.unit_progress_key(expected, args)
    progress_path = tmp_path / "missing-progress.json"
    progress_path.write_text(
        json.dumps({"processed_units_v2": {key: module.progress_entry(expected, args, output_markdown=str(markdown))}}),
        encoding="utf-8",
    )
    report = module.daily_completion_report(args, ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]
    progress_path.write_text(
        json.dumps({"processed_units_v2": {key: module.progress_entry(expected, args, document_id=doc_id, output_markdown=str(markdown))}}),
        encoding="utf-8",
    )
    report = module.daily_completion_report(args, ["2026-05-07"])
    assert report["details"]["2026-05-07"]["matched_units"] == 1
    signature = {
        "topic": expected.topic,
        "index": expected.index,
        "source_ids": expected.source_ids,
        "input_content_hash": module.short_hash(expected.content, 24),
    }
    assert module.checkpointed_daily_artifact_pair_is_valid(tmp_path, "2026-05-07", signature, expected_posted_artifacts={(doc_id, str(markdown.resolve()))})
    current_data = json.loads(sidecar.read_text(encoding="utf-8"))
    del current_data["unit"]["input_content_hash"]
    sidecar.write_text(json.dumps(current_data), encoding="utf-8")
    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]
    assert not module.checkpointed_daily_artifact_pair_is_valid(tmp_path, "2026-05-07", signature, expected_posted_artifacts={(doc_id, str(markdown.resolve()))})
    current_data["unit"]["input_content_hash"] = module.short_hash(expected.content, 24)
    sidecar.write_text(json.dumps(current_data), encoding="utf-8")
    report = module.daily_completion_report(args_for(tmp_path), ["2026-05-07"])
    assert report["details"]["2026-05-07"]["matched_units"] == 1
    missing_sidecar_document_id = json.loads(sidecar.read_text(encoding="utf-8"))
    missing_sidecar_document_id.pop("document_id")
    sidecar.write_text(json.dumps(missing_sidecar_document_id), encoding="utf-8")
    wrong_document_id = module.progress_entry(expected, args, document_id="different-document", output_markdown=str(markdown))
    progress_path.write_text(json.dumps({"processed_units_v2": {key: wrong_document_id}}), encoding="utf-8")
    report = module.daily_completion_report(args, ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]
    assert not module.checkpointed_daily_artifact_pair_is_valid(tmp_path, "2026-05-07", signature, expected_posted_artifacts={(doc_id, str(markdown.resolve()))})
    missing_sidecar_document_id["document_id"] = doc_id
    sidecar.write_text(json.dumps(missing_sidecar_document_id), encoding="utf-8")
    wrong_document_id = module.progress_entry(expected, args, document_id="different-document", output_markdown=str(markdown))
    progress_path.write_text(json.dumps({"processed_units_v2": {key: wrong_document_id}}), encoding="utf-8")
    report = module.daily_completion_report(args, ["2026-05-07"])
    assert report["missing_days"] == ["2026-05-07"]
    assert not module.checkpointed_daily_artifact_pair_is_valid(
        tmp_path,
        "2026-05-07",
        signature,
        expected_posted_artifacts={("different-document", str(markdown.resolve()))},
    )
    progress_path.write_text(
        json.dumps({"processed_units_v2": {key: module.progress_entry(expected, args, document_id=doc_id, output_markdown=str(markdown))}}),
        encoding="utf-8",
    )

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


def test_checkpointed_current_schema_reuse_requires_exact_progress_document_id(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-05-07"
    expected = unit(module, day, "checkpointed current schema fact")
    args = args_for(tmp_path)
    saved = module.save_result(tmp_path, expected, {"executive_summary": ["posted"]}, "raw", model=args.llm_model)
    original_sidecar_text = saved.output_json_path.read_text(encoding="utf-8")
    checkpoint = module.record_completed_days(
        module.load_daily_completion_checkpoint(tmp_path, "hermes")[0],
        {day: [expected]},
        source="successful_daily_run",
    )
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    key = module.unit_progress_key(expected, args)
    exact_entry = module.progress_entry(expected, args, document_id=saved.document_id, output_markdown=str(saved.output_md_path))
    progress_path.write_text(json.dumps({"processed_units_v2": {key: exact_entry}}), encoding="utf-8")

    assert module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)
    exact_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert module.checkpointed_daily_artifact_gaps(
        tmp_path,
        checkpoint,
        args=args,
        progress=exact_progress,
    ) == []

    unbanked_entry = dict(exact_entry)
    unbanked_entry.pop("bank")
    unbanked_progress = {"processed_units_v2": {key: unbanked_entry}}
    progress_path.write_text(json.dumps(unbanked_progress), encoding="utf-8")
    assert not module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)
    assert module.checkpointed_daily_artifact_gaps(
        tmp_path,
        checkpoint,
        args=args,
        progress=unbanked_progress,
    ) == [day]
    progress_path.write_text(json.dumps(exact_progress), encoding="utf-8")

    sidecar = json.loads(saved.output_json_path.read_text(encoding="utf-8"))
    sidecar["document_id"] = "unrelated-document"
    saved.output_json_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert not module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)
    assert module.checkpointed_daily_artifact_gaps(
        tmp_path,
        checkpoint,
        args=args,
        progress=exact_progress,
    ) == [day]

    sidecar.pop("artifact_schema_version")
    sidecar.pop("document_id")
    saved.output_json_path.write_text(json.dumps(sidecar), encoding="utf-8")
    empty_progress = {"processed_units_v2": {}}
    progress_path.write_text(json.dumps(empty_progress), encoding="utf-8")
    assert not module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)
    assert module.checkpointed_daily_artifact_gaps(tmp_path, checkpoint, args=args, progress=empty_progress) == [day]

    hashless_sidecar = json.loads(saved.output_json_path.read_text(encoding="utf-8"))
    hashless_sidecar.pop("unit")["input_content_hash"] = None
    hashless_sidecar["document_id"] = saved.document_id
    saved.output_json_path.write_text(json.dumps(hashless_sidecar), encoding="utf-8")
    hashless_entry = dict(exact_entry, artifact_schema_version=None, input_content_hash=None)
    hashless_progress = {"processed_units_v2": {key: hashless_entry}}
    progress_path.write_text(json.dumps(hashless_progress), encoding="utf-8")
    assert not module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)
    assert module.checkpointed_daily_artifact_gaps(tmp_path, checkpoint, args=args, progress=hashless_progress) == [day]

    wrong_bank_entry = dict(exact_entry, bank="other-bank")
    wrong_bank_progress = {"processed_units_v2": {key: wrong_bank_entry}}
    progress_path.write_text(json.dumps(wrong_bank_progress), encoding="utf-8")
    assert not module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)

    wrong_path_entry = dict(exact_entry, output_markdown=str(tmp_path / "other.md"))
    wrong_path_progress = {"processed_units_v2": {key: wrong_path_entry}}
    progress_path.write_text(json.dumps(wrong_path_progress), encoding="utf-8")
    assert not module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)

    saved.output_json_path.write_text(original_sidecar_text, encoding="utf-8")
    progress_path.write_text(json.dumps(exact_progress), encoding="utf-8")
    assert module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)


def test_hashless_checkpoint_artifact_requires_trusted_legacy_provenance_even_with_post_progress(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-05-07"
    expected = unit(module, day, "hashless checkpoint evidence")
    args = args_for(tmp_path)
    saved = module.save_result(tmp_path, expected, {"executive_summary": ["posted"]}, "raw", model=args.llm_model)
    sidecar = json.loads(saved.output_json_path.read_text(encoding="utf-8"))
    sidecar.pop("artifact_schema_version")
    sidecar["unit"]["input_content_hash"] = None
    saved.output_json_path.write_text(json.dumps(sidecar), encoding="utf-8")
    entry = module.progress_entry(expected, args, document_id=saved.document_id, output_markdown=str(saved.output_md_path))
    entry["artifact_schema_version"] = None
    entry["input_content_hash"] = None
    progress = {"processed_units_v2": {module.unit_progress_key(expected, args): entry}}
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")
    module.DEFAULT_PROGRESS_FILE.write_text(json.dumps(progress), encoding="utf-8")

    checkpoint = module.record_completed_days(
        module.load_daily_completion_checkpoint(tmp_path, "hermes")[0],
        {day: [expected]},
        source="successful_daily_run",
    )
    assert not module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint)
    assert module.checkpointed_daily_artifact_gaps(tmp_path, checkpoint, args=args, progress=progress) == [day]

    trusted_legacy_checkpoint = module.record_completed_days(
        module.load_daily_completion_checkpoint(tmp_path, "hermes")[0],
        {day: [expected]},
        source="verified_frozen_completion_scope",
        advance_through=day,
    )
    assert module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, trusted_legacy_checkpoint, args=args, progress=progress)


def test_hashless_daily_progress_reuse_requires_exact_stable_content_hash(tmp_path):
    module = load_module()
    expected = unit(module, "2026-05-07", "hashless progress input")
    args = args_for(tmp_path)
    saved = module.save_result(tmp_path, expected, {"executive_summary": ["posted"]}, "raw", model=args.llm_model)
    sidecar = json.loads(saved.output_json_path.read_text(encoding="utf-8"))
    sidecar.pop("artifact_schema_version")
    sidecar["unit"].pop("input_content_hash")
    saved.output_json_path.write_text(json.dumps(sidecar), encoding="utf-8")
    entry = module.progress_entry(expected, args, document_id=saved.document_id, output_markdown=str(saved.output_md_path))
    entry["artifact_schema_version"] = None
    entry["input_content_hash"] = None

    assert module.v2_progress_entry_is_reusable(tmp_path, expected, entry, args=args)
    entry["stable_content_hash"] = "stale-stable-hash"
    assert not module.v2_progress_entry_is_reusable(tmp_path, expected, entry, args=args)


def test_verified_default_bank_progress_migration_requires_an_exact_artifact(tmp_path):
    module = load_module()
    day = "2026-05-07"
    expected = unit(module, day, "verified default-bank migration")
    args = args_for(tmp_path)
    saved = module.save_result(tmp_path, expected, {"executive_summary": ["posted"]}, "raw", model=args.llm_model)
    entry = module.progress_entry(expected, args, document_id=saved.document_id, output_markdown=str(saved.output_md_path))
    entry.pop("bank")
    unbanked_template = dict(entry)
    key = module.unit_progress_key(expected, args)
    progress = {"processed_units_v2": {key: entry}}
    checkpoint = module.record_completed_days(
        module.load_daily_completion_checkpoint(tmp_path, "hermes")[0],
        {day: [expected]},
        source="successful_daily_run",
    )

    assert not module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint, progress=progress)
    assert module.migrate_verified_default_bank_progress_entries(progress, tmp_path, {day: [expected]}, args) == 1
    assert progress["processed_units_v2"][key]["bank"] == "hermes"
    assert module.checkpointed_daily_artifact_is_reusable(tmp_path, expected, checkpoint, progress=progress)

    invalid_entry = dict(unbanked_template, output_markdown=str(tmp_path / "outside.md"))
    invalid_progress = {"processed_units_v2": {key: invalid_entry}}
    assert module.migrate_verified_default_bank_progress_entries(invalid_progress, tmp_path, {day: [expected]}, args) == 0
    assert "bank" not in invalid_progress["processed_units_v2"][key]

    other_args = args_for(tmp_path)
    other_args.bank = "other-bank"
    other_progress = {"processed_units_v2": {key: dict(unbanked_template)}}
    assert module.migrate_verified_default_bank_progress_entries(other_progress, tmp_path, {day: [expected]}, other_args) == 0
    assert "bank" not in other_progress["processed_units_v2"][key]


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


def test_run_units_reuses_legacy_pair_when_progress_key_and_render_match(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "current content")
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    day_dir = tmp_path / "daily" / expected.period
    day_dir.mkdir(parents=True)
    markdown = day_dir / "hermes__00__legacy.md"
    sidecar_path = markdown.with_suffix(".json")
    llm_json = {"executive_summary": ["previously completed"]}
    doc_id = "hermes-offline-consolidation::daily::2026-05-07::hermes::00::legacy"
    sidecar = {
        "document_id": doc_id,
        "unit": {
            "scope": "daily",
            "period": expected.period,
            "topic": expected.topic,
            "index": expected.index,
            "source_count": expected.source_count,
            "source_ids": expected.source_ids,
            "date_range_start": expected.date_range_start,
            "date_range_end": expected.date_range_end,
        },
        "model": args.llm_model,
        "llm_json": llm_json,
        "raw_text": "",
        "markdown_path": str(markdown),
    }
    markdown.write_text(
        module.render_markdown(expected, llm_json, "", model=args.llm_model),
        encoding="utf-8",
    )
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    key = module.unit_progress_key(expected, args)
    entry = module.progress_entry(expected, args, document_id=doc_id, output_markdown=str(markdown))
    entry.pop("artifact_schema_version")
    entry.pop("input_content_hash")
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps({"processed_units_v2": {key: entry}, "processed_document_ids": [], "processed_unit_keys": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    calls = []
    monkeypatch.setattr(module, "call_llm_for_unit", lambda *args: calls.append(args) or ({}, "{}"))

    assert module.run_units(args, [expected]) == 0
    assert calls == []
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda _args, days: {expected.period: [expected]})
    report = module.daily_completion_report(args, [expected.period])
    assert report["details"][expected.period]["matched_units"] == 1


def test_unposted_daily_artifact_pair_alone_is_not_success(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-05-07"
    expected = unit(module, day, "archived source facts")
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    module.save_result(
        tmp_path,
        expected,
        {"executive_summary": ["artifact exists, post not confirmed"]},
        "unposted raw response",
        model=args.llm_model,
    )
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    checkpoint, _ = module.load_daily_completion_checkpoint(tmp_path, "hermes")
    completion = module.evaluate_daily_completion(
        tmp_path,
        {day: [expected]},
        args=args,
        progress={"processed_units_v2": {}},
        checkpoint=checkpoint,
    )
    budget = module.build_budget_report([expected], {"processed_units_v2": {}}, args)
    llm_calls = []
    posts = []
    monkeypatch.setattr(module, "call_llm_for_unit", lambda _args, item: llm_calls.append(item) or ({"executive_summary": ["recomputed"]}, "new response"))
    monkeypatch.setattr(module, "post_to_hindsight", lambda *_args, **_kwargs: posts.append(True) or (True, None))

    result = module.run_units(args, [expected])

    assert completion["missing_days"] == [day]
    assert budget["pending_units"] == 1
    assert result == 0
    assert llm_calls == [expected]
    assert posts == [True]


def test_run_units_reuses_checkpointed_pair_without_progress_cache(tmp_path, monkeypatch):
    module = load_module()
    expected = unit(module, "2026-05-07", "previously completed facts")
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    day_dir = tmp_path / "daily" / expected.period
    day_dir.mkdir(parents=True)
    markdown = day_dir / "hermes__00__baseline.md"
    llm_json = {"executive_summary": ["prior result"]}
    sidecar = {
        "unit": {
            "scope": "daily",
            "period": expected.period,
            "topic": expected.topic,
            "index": expected.index,
            "source_count": expected.source_count,
            "source_ids": expected.source_ids,
            "date_range_start": expected.date_range_start,
            "date_range_end": expected.date_range_end,
        },
        "model": args.llm_model,
        "llm_json": llm_json,
        "raw_text": "",
        "markdown_path": str(markdown),
    }
    markdown.write_text(module.render_markdown(expected, llm_json, "", model=args.llm_model), encoding="utf-8")
    markdown.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    checkpoint, _ = module.load_daily_completion_checkpoint(tmp_path, "hermes")
    checkpoint = module.record_completed_days(
        checkpoint,
        {expected.period: [expected]},
        source="verified_frozen_completion_scope",
        advance_through=expected.period,
    )
    assert module.checkpoint_day_allows_trusted_legacy_pair(checkpoint, expected.period)
    assert not module.checkpoint_day_allows_trusted_legacy_pair(
        {**checkpoint, "baseline_provenance": None}, expected.period
    )
    module.save_json(tmp_path / "daily_completion_checkpoint.json", checkpoint)
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    calls = []
    monkeypatch.setattr(module, "call_llm_for_unit", lambda *llm_args: calls.append(llm_args) or ({}, "{}"))
    monkeypatch.setattr(module, "post_to_hindsight", lambda *args, **kwargs: (True, None))
    budget = module.build_budget_report([expected], {"processed_units_v2": {}}, args)
    assert budget["pending_units"] == 0

    assert module.run_units(args, [expected]) == 0
    assert calls == []


def test_submit_enforce_budget_blocks_over_limit_units_before_llm_or_post(tmp_path, monkeypatch):
    module = load_module()
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    args.budget_json = False
    args.enforce_budget = True
    args.budget_max_pending_units = 12
    args.budget_max_pending_chars = 500000
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    llm_calls = []
    post_calls = []
    monkeypatch.setattr(module, "call_llm_for_unit", lambda *call_args: llm_calls.append(call_args) or ({"executive_summary": ["result"]}, "raw"))
    monkeypatch.setattr(module, "post_to_hindsight", lambda *post_args, **kwargs: post_calls.append(post_args) or (True, None))
    units = [unit(module, f"2026-04-{index:02d}", f"new post-checkpoint unit {index}") for index in range(1, 14)]

    code = module.run_units(args, units)

    assert code == 2
    assert llm_calls == []
    assert post_calls == []


def test_submit_enforce_budget_without_finite_cap_fails_closed(tmp_path, monkeypatch):
    module = load_module()
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    args.budget_json = False
    args.enforce_budget = True
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")
    llm_calls = []
    monkeypatch.setattr(module, "call_llm_for_unit", lambda *call_args: llm_calls.append(call_args) or ({}, "raw"))
    weekly = module.ReflectUnit(
        "weekly", "all-history", "hermes", 0, "pending work", 1, ["daily-document"],
        "2026-03-16T00:00:00", "2026-03-16T23:59:59",
    )

    code = module.run_units(args, [weekly])

    assert code == 2
    assert llm_calls == []


def test_weekly_catchup_budget_aggregates_missing_units_before_any_daily_submit(tmp_path, monkeypatch):
    module = load_module()
    baseline = "2026-03-15"
    days = [f"2026-03-{day:02d}" for day in range(16, 29)]
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    units_by_day = {day: [unit(module, day, f"missing post-checkpoint unit {day}")] for day in days}
    completion = {
        "missing_days": days,
        "details": {day: {"expected_units": 1, "missing_units": 1} for day in days},
        "missing_unit_signatures_by_day": {
            day: [module.unit_completion_signature(units_by_day[day][0])] for day in days
        },
    }
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [baseline, *days])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(module, "checkpointed_daily_artifact_gaps", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda _args, _days: (units_by_day, completion))
    monkeypatch.setattr(module, "write_daily_completion_manifest", lambda *_args: tmp_path / "frozen.json")
    monkeypatch.setattr(module, "evaluate_daily_completion", lambda *_args, **_kwargs: {"missing_days": []})
    monkeypatch.setattr(module, "update_daily_completion_checkpoint", lambda *_args, **_kwargs: None)
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    run_calls = []
    monkeypatch.setattr(module, "run_units", lambda _args, missing: run_calls.append(list(missing)) or 0)
    args = _checkpoint_args(tmp_path)
    args.max_input_chars = 60000
    args.enforce_budget = True
    args.budget_json = False
    args.budget_max_pending_units = 12
    args.budget_max_pending_chars = 500000

    try:
        module.resolve_missing_daily_args(args)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("the aggregate catch-up budget must block before the first daily submit")

    assert run_calls == []


def test_weekly_submit_budget_uses_remaining_catchup_unit_and_char_capacity(tmp_path, monkeypatch):
    module = load_module()
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    args.budget_json = False
    args.enforce_budget = True
    args.budget_max_pending_units = 12
    args.budget_max_pending_chars = 500000
    args._budget_phase = "weekly"
    args._budget_spent_units = 10
    args._budget_spent_chars = 490000
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")
    llm_calls = []
    monkeypatch.setattr(module, "call_llm_for_unit", lambda *call_args: llm_calls.append(call_args) or ({}, "raw"))
    weekly = module.ReflectUnit(
        "weekly", "all-history", "hermes", 0, "x" * 20000, 1, ["daily-document"],
        "2026-03-16T00:00:00", "2026-03-16T23:59:59",
    )

    code = module.run_units(args, [weekly])

    assert code == 2
    assert llm_calls == []


def test_run_units_reuses_verified_legacy_progress_across_model_key_changes(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-05-07"
    content = f"period: {day}\ndate_range: {day}T00:00:00 to {day}T23:59:59\nbody"
    expected = unit(module, day, content)
    args = args_for(tmp_path)
    args.mode = "submit"
    args.force_repost = False
    args.concurrency = 1
    args.min_concurrency = 1
    args.rate_limit_backoff_seconds = 0
    args.delay = 0
    old_args = args_for(tmp_path)
    old_args.llm_model = "old-model"
    day_dir = tmp_path / "daily" / day
    day_dir.mkdir(parents=True)
    markdown = day_dir / "hermes__00__legacy-old-model.md"
    llm_json = {"executive_summary": ["previous result"]}
    doc_id = "hermes-offline-consolidation::daily::2026-05-07::hermes::00::legacy-old-model"
    sidecar = {
        "unit": {
            "scope": "daily",
            "period": day,
            "topic": expected.topic,
            "index": expected.index,
            "source_count": expected.source_count,
            "source_ids": expected.source_ids,
            "date_range_start": expected.date_range_start,
            "date_range_end": expected.date_range_end,
        },
        "model": old_args.llm_model,
        "llm_json": llm_json,
        "raw_text": "",
        "markdown_path": str(markdown),
    }
    markdown.write_text(module.render_markdown(expected, llm_json, "", model=old_args.llm_model), encoding="utf-8")
    markdown.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    old_key = module.unit_progress_key(expected, old_args)
    entry = module.progress_entry(expected, old_args, document_id=doc_id, output_markdown=str(markdown))
    entry.pop("artifact_schema_version")
    entry.pop("input_content_hash")
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({"processed_units_v2": {old_key: entry}}), encoding="utf-8")
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    calls = []
    monkeypatch.setattr(module, "call_llm_for_unit", lambda *llm_args: calls.append(llm_args) or ({}, "{}"))
    monkeypatch.setattr(module, "post_to_hindsight", lambda *args, **kwargs: (True, None))

    assert module.run_units(args, [expected]) == 0
    assert calls == []


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


def _checkpoint_args(tmp_path: Path, *, backfill: bool = False):
    return argparse.Namespace(
        weekly_source="daily",
        weekly_window="all-history",
        output_dir=str(tmp_path),
        bank="hermes",
        backfill_missing_daily=backfill,
        scope="weekly",
        mode="submit",
        date="2026-03-16",
        daily_source="facts",
        group_by="topic",
    )


def test_checkpoint_rejects_markerless_manifest_that_does_not_cover_legacy_inventory(tmp_path):
    module = load_module()
    manifest_dir = tmp_path / "completion_manifests"
    manifest_dir.mkdir()
    day = "2026-03-14"
    manifest = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "days": [day],
        "units": {day: []},
        "missing_before": [],
        "details_before": {day: {"expected_units": 0, "missing_units": 0}},
    }
    (manifest_dir / "partial-daily-completion.json").write_text(json.dumps(manifest), encoding="utf-8")

    checkpoint, bootstrapped = module.load_daily_completion_checkpoint(
        tmp_path,
        "hermes",
        legacy_history_days={day, "2026-03-15"},
    )

    assert bootstrapped is False
    assert checkpoint["baseline_through"] is None
    assert checkpoint["bootstrap_required"] is True


def test_checkpoint_requires_consistent_all_history_coverage_marker(tmp_path):
    module = load_module()
    manifest_dir = tmp_path / "completion_manifests"
    manifest_dir.mkdir()
    day = "2026-03-14"
    manifest = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "coverage_scope": "all-history",
        "coverage_days_hash": "not-the-date-list-hash",
        "days": [day],
        "units": {day: []},
        "missing_before": [],
        "details_before": {day: {"expected_units": 0, "missing_units": 0}},
    }
    path = manifest_dir / "all-history-daily-completion.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    checkpoint, bootstrapped = module.load_daily_completion_checkpoint(tmp_path, "hermes")
    assert bootstrapped is False
    assert checkpoint["baseline_through"] is None

    manifest["coverage_days_hash"] = sys.modules["hindsight_daily_checkpoint"].date_list_hash([day])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint, bootstrapped = module.load_daily_completion_checkpoint(tmp_path, "hermes")

    assert bootstrapped is True
    assert checkpoint["baseline_through"] == day


def test_legacy_manifest_bootstrap_requires_exact_posted_progress(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-03-14"
    expected = unit(module, day, f"period: {day}\ndate_range: {day}T00:00:00..{day}T23:59:59\ncurrent legacy facts")
    manifest_dir = tmp_path / "completion_manifests"
    manifest_dir.mkdir()
    manifest = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "days": [day],
        "units": {day: [{"topic": expected.topic, "index": expected.index, "source_ids": expected.source_ids, "input_content_hash": module.short_hash(expected.content, 24)}]},
        "missing_before": [],
        "details_before": {day: {"expected_units": 1, "missing_units": 0}},
    }
    (manifest_dir / "legacy-complete.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "retained_sqlite_days", lambda _bank: [day])
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda _args, _days: {day: [expected]})
    args = args_for(tmp_path)
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)

    checkpoint, bootstrapped = module.load_daily_checkpoint_for_run(tmp_path, "hermes", args=args)
    assert bootstrapped is False
    assert checkpoint["baseline_through"] is None
    assert checkpoint["completed_days"] == {}

    result = module.save_result(tmp_path, expected, {"executive_summary": ["posted"]}, "raw", model=args.llm_model)
    key = module.legacy_unit_progress_key(expected)
    entry = module.progress_entry(
        expected,
        args,
        document_id=result.document_id,
        output_markdown=str(result.output_md_path),
    )
    entry.pop("bank")
    progress_path.write_text(json.dumps({"processed_units_v2": {key: entry}}), encoding="utf-8")

    args.mode = "dry-run"
    checkpoint, bootstrapped = module.load_daily_checkpoint_for_run(tmp_path, "hermes", args=args)
    assert bootstrapped is True
    assert "bank" not in json.loads(progress_path.read_text(encoding="utf-8"))["processed_units_v2"][key]
    assert not hasattr(args, "_bank_scoped_progress")
    assert module.checkpointed_daily_artifact_gaps(tmp_path, checkpoint, args=args) == [day]

    args.mode = "submit"
    args.budget_json = True
    checkpoint, bootstrapped = module.load_daily_checkpoint_for_run(tmp_path, "hermes", args=args)
    assert bootstrapped is True
    budget_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert "bank" not in budget_progress["processed_units_v2"][key]
    assert module.build_budget_report([expected], budget_progress, args)["pending_units"] == 1

    args.budget_json = False
    checkpoint, bootstrapped = module.load_daily_checkpoint_for_run(tmp_path, "hermes", args=args)
    assert bootstrapped is True
    assert checkpoint["baseline_through"] == day
    migrated_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert migrated_progress["processed_units_v2"][key]["bank"] == "hermes"
    assert module.checkpointed_daily_artifact_gaps(tmp_path, checkpoint, args=args, progress=migrated_progress) == []


def test_marked_all_history_manifest_requires_posted_progress(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-03-14"
    expected = unit(module, day, "marked manifest source")
    manifest_dir = tmp_path / "completion_manifests"
    manifest_dir.mkdir()
    manifest = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "coverage_scope": "all-history",
        "coverage_days_hash": module.date_list_hash([day]),
        "days": [day],
        "units": {day: [{"topic": expected.topic, "index": expected.index, "source_ids": expected.source_ids, "input_content_hash": module.short_hash(expected.content, 24)}]},
        "missing_before": [],
        "details_before": {day: {"expected_units": 1, "missing_units": 0}},
    }
    (manifest_dir / "marked-complete.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda _args, _days: {day: [expected]})
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "empty-progress.json")

    checkpoint, bootstrapped = module.load_daily_checkpoint_for_run(tmp_path, "hermes", args=args_for(tmp_path))

    assert bootstrapped is False
    assert checkpoint["baseline_from"] is None
    assert checkpoint["baseline_through"] is None
    assert checkpoint["completed_days"] == {}

    args = args_for(tmp_path)
    result = module.save_result(tmp_path, expected, {"executive_summary": ["posted"]}, "raw", model=args.llm_model)
    key = module.unit_progress_key(expected, args)
    entry = module.progress_entry(expected, args, document_id=result.document_id, output_markdown=str(result.output_md_path))
    progress_path = tmp_path / "empty-progress.json"
    progress_path.write_text(json.dumps({"processed_units_v2": {key: entry}}), encoding="utf-8")
    checkpoint, bootstrapped = module.load_daily_checkpoint_for_run(tmp_path, "hermes", args=args)
    assert bootstrapped is True
    assert checkpoint["baseline_through"] == day


def test_marked_manifest_requires_explicit_unit_input_hash(tmp_path):
    module = load_module()
    day = "2026-03-14"
    manifest_dir = tmp_path / "completion_manifests"
    manifest_dir.mkdir()
    manifest = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "coverage_scope": "all-history",
        "coverage_days_hash": module.date_list_hash([day]),
        "days": [day],
        "units": {day: [{"topic": "hermes", "index": 0, "source_ids": ["fact-1"]}]},
        "missing_before": [],
        "details_before": {day: {"expected_units": 1, "missing_units": 0}},
    }
    (manifest_dir / "marked-hashless.json").write_text(json.dumps(manifest), encoding="utf-8")

    checkpoint, bootstrapped = module.load_daily_completion_checkpoint(tmp_path, "hermes")

    assert bootstrapped is False
    assert checkpoint["baseline_through"] is None
    assert checkpoint["completed_days"] == {}

    manifest["units"][day][0]["input_content_hash"] = 123456789012345678901234
    (manifest_dir / "marked-hashless.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint, bootstrapped = module.load_daily_completion_checkpoint(tmp_path, "hermes")

    assert bootstrapped is False
    assert checkpoint["baseline_through"] is None
    assert checkpoint["completed_days"] == {}


def test_checkpoint_bootstraps_from_latest_verified_completion_manifest(tmp_path):
    module = load_module()
    manifests = tmp_path / "completion_manifests"
    manifests.mkdir()
    good = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "days": ["2026-03-14", "2026-03-15"],
        "units": {
            "2026-03-14": [{"topic": "hermes", "index": 0, "source_ids": ["f1"], "input_content_hash": module.short_hash("fact", 24)}],
            "2026-03-15": [],
        },
        "missing_before": [],
        "details_before": {
            "2026-03-14": {"expected_units": 1, "missing_units": 0},
            "2026-03-15": {"expected_units": 0, "missing_units": 0},
        },
    }
    bad_newer = {**good, "generated_at": "2026-03-17T12:00:00", "missing_before": ["2026-03-15"]}
    (manifests / "20260316-good-daily-completion.json").write_text(json.dumps(good), encoding="utf-8")
    (manifests / "20260317-incomplete-daily-completion.json").write_text(json.dumps(bad_newer), encoding="utf-8")

    checkpoint, bootstrapped = module.load_daily_completion_checkpoint(
        tmp_path,
        "hermes",
        legacy_history_days={"2026-03-14", "2026-03-15"},
    )

    assert bootstrapped is True
    assert checkpoint["baseline_from"] == "2026-03-14"
    assert checkpoint["baseline_through"] == "2026-03-15"
    assert checkpoint["completed_days"]["2026-03-14"]["unit_count"] == 1
    assert checkpoint["completed_days"]["2026-03-15"]["unit_count"] == 0


def test_weekly_submit_persists_manifest_bootstrap_and_advances_only_after_verification(tmp_path, monkeypatch):
    module = load_module()
    manifests = tmp_path / "completion_manifests"
    manifests.mkdir()
    baseline = "2026-03-15"
    day = "2026-03-16"
    prior = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "days": [baseline],
        "units": {baseline: []},
        "missing_before": [],
        "details_before": {baseline: {"expected_units": 0, "missing_units": 0}},
    }
    (manifests / "verified-daily-completion.json").write_text(json.dumps(prior), encoding="utf-8")
    expected = unit(module, day, "newer facts")
    monkeypatch.setattr(module, "retained_sqlite_days", lambda _bank: [baseline])
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda _args, days: {d: [] for d in days})
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [baseline, day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    seen_days = []
    monkeypatch.setattr(
        module,
        "build_daily_completion_snapshot",
        lambda _args, days: (
            seen_days.extend(days) or {day: [expected]},
            {"missing_days": [], "details": {day: {"expected_units": 1, "missing_units": 0}}},
        ),
    )
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")

    module.resolve_missing_daily_args(_checkpoint_args(tmp_path))

    checkpoint = json.loads((tmp_path / "daily_completion_checkpoint.json").read_text(encoding="utf-8"))
    assert seen_days == [day]
    assert checkpoint["baseline_through"] == day
    assert checkpoint["completed_days"][day]["unit_count"] == 1


def test_daily_checkpoint_bootstrap_downgrades_unverified_manifest_artifacts(tmp_path, monkeypatch):
    module = load_module()
    day, next_day = "2026-03-15", "2026-03-16"
    expected = unit(module, day, "archived source facts")
    manifest_dir = tmp_path / "completion_manifests"
    manifest_dir.mkdir()
    manifest = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "days": [day],
        "units": {
            day: [{
                "topic": expected.topic,
                "index": expected.index,
                "source_ids": sorted(expected.source_ids),
                "input_content_hash": module.short_hash(expected.content, 24),
            }],
        },
        "missing_before": [],
        "details_before": {day: {"expected_units": 1, "missing_units": 0}},
    }
    (manifest_dir / "verified-daily-completion.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "retained_sqlite_days", lambda _bank: [day])
    monkeypatch.setattr(module, "build_daily_fact_units_for_days", lambda _args, _days: {day: [expected]})
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")

    module.update_daily_completion_checkpoint(
        args_for(tmp_path),
        {next_day: [unit(module, next_day, "new daily facts")]},
        source="successful_daily_run",
    )

    checkpoint = json.loads((tmp_path / "daily_completion_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["baseline_from"] is None
    assert checkpoint["baseline_through"] is None
    assert set(checkpoint["completed_days"]) == {next_day}
    assert checkpoint["latest_successful_date"] == next_day
    assert checkpoint["bootstrap_required"] is True
    assert checkpoint["baseline_manifest"] is None


def test_successful_daily_checkpoint_persists_exact_unit_signature(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-03-16"
    expected = unit(module, day, "source facts")
    args = args_for(tmp_path)
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)

    module.update_daily_completion_checkpoint(
        args,
        {day: [expected]},
        source="successful_daily_run",
    )

    checkpoint = json.loads((tmp_path / "daily_completion_checkpoint.json").read_text(encoding="utf-8"))
    assert not progress_path.exists()
    assert checkpoint["baseline_through"] is None
    assert checkpoint["latest_successful_date"] == day
    assert checkpoint["bootstrap_required"] is True
    record = checkpoint["completed_days"][day]
    assert record["unit_count"] == 1
    assert record["unit_signatures"][0]["source_ids"] == [f"fact-{day}"]
    assert record["unit_signatures"][0]["input_content_hash"] == module.short_hash(expected.content, 24)


def test_watermark_requires_exact_baseline_records_and_provenance(tmp_path):
    module = load_module()
    output_dir = tmp_path / "watermark-output"
    output_dir.mkdir()
    day = "2026-03-15"
    state = {
        "schema_version": module.DAILY_CHECKPOINT_SCHEMA_VERSION,
        "bank": "hermes",
        "output_dir": str(output_dir.resolve()),
        "baseline_from": day,
        "baseline_through": day,
        "latest_successful_date": day,
        "completed_days": {},
    }
    checkpoint_path = output_dir / module.DAILY_CHECKPOINT_FILENAME
    checkpoint_path.write_text(json.dumps(state), encoding="utf-8")
    try:
        module.load_daily_completion_checkpoint(output_dir, "hermes")
    except ValueError as exc:
        assert "watermark" in str(exc).lower() or "baseline" in str(exc).lower()
    else:
        raise AssertionError("a bare watermark must not suppress incremental discovery")

    empty_signature = hashlib.sha256(b"[]").hexdigest()
    state["completed_days"] = {
        day: {
            "unit_count": 0,
            "unit_set_hash": empty_signature,
            "unit_signatures": [],
            "completed_at": "2026-03-15T00:00:00",
            "source": "verified_frozen_completion_scope",
        },
    }
    checkpoint_path.write_text(json.dumps(state), encoding="utf-8")
    try:
        module.load_daily_completion_checkpoint(output_dir, "hermes")
    except ValueError as exc:
        assert "provenance" in str(exc).lower()
    else:
        raise AssertionError("exact day records without verified watermark provenance must not be trusted")

    state["baseline_provenance"] = {
        "kind": "verified-frozen-candidate-set",
        "verified_from": day,
        "verified_through": day,
        "verified_days": [day],
        "verified_days_hash": module.date_list_hash([day]),
        "candidate_days": [day],
        "candidate_days_hash": module.date_list_hash([day]),
        "verified_at": "2026-03-15T00:00:00",
    }
    checkpoint_path.write_text(json.dumps(state), encoding="utf-8")
    assert module.load_daily_completion_checkpoint(output_dir, "hermes")[0]["baseline_through"] == day


def test_checkpoint_cannot_be_reused_for_another_output_root(tmp_path):
    module = load_module()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    second_root.mkdir()
    state = {
        "schema_version": module.DAILY_CHECKPOINT_SCHEMA_VERSION,
        "bank": "hermes",
        "output_dir": str(first_root.resolve()),
        "baseline_through": None,
        "latest_successful_date": None,
        "completed_days": {},
    }
    (second_root / "daily_completion_checkpoint.json").write_text(json.dumps(state), encoding="utf-8")

    try:
        module.load_daily_completion_checkpoint(second_root, "hermes")
    except ValueError as exc:
        assert "output" in str(exc).lower()
    else:
        raise AssertionError("checkpoint signatures must not cross output roots")


def _write_checkpoint(module, output_dir: Path, *, baseline_through: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "daily_completion_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_version": module.DAILY_CHECKPOINT_SCHEMA_VERSION,
                "bank": "hermes",
                "output_dir": str(output_dir.resolve()),
                "baseline_from": baseline_through,
                "baseline_through": baseline_through,
                "latest_successful_date": baseline_through,
                "completed_days": {
                    baseline_through: {
                        "unit_count": 0,
                        "unit_set_hash": hashlib.sha256(b"[]").hexdigest(),
                        "unit_signatures": [],
                        "completed_at": f"{baseline_through}T00:00:00",
                        "source": "verified_frozen_completion_scope",
                    },
                },
                "baseline_provenance": {
                    "kind": "verified-frozen-candidate-set",
                    "verified_from": baseline_through,
                    "verified_through": baseline_through,
                    "verified_days": [baseline_through],
                    "verified_days_hash": module.date_list_hash([baseline_through]),
                    "candidate_days": [baseline_through],
                    "candidate_days_hash": module.date_list_hash([baseline_through]),
                    "verified_at": f"{baseline_through}T00:00:00",
                },
            }
        ),
        encoding="utf-8",
    )


def test_weekly_automatically_catches_up_only_after_checkpoint(tmp_path, monkeypatch):
    module = load_module()
    baseline = "2026-03-15"
    day = "2026-03-16"
    expected = unit(module, day, "source facts")
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: ["2026-03-14", baseline, day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    seen_days = []

    def snapshot(_args, days):
        seen_days.extend(days)
        return {day: [expected]}, {"missing_days": [day], "details": {day: {"expected_units": 1, "missing_units": 1}}}

    monkeypatch.setattr(module, "build_daily_completion_snapshot", snapshot)
    monkeypatch.setattr(module, "write_daily_completion_manifest", lambda *_args: tmp_path / "completion.json")
    progress_path = tmp_path / "progress.json"
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    run_calls = []
    monkeypatch.setattr(module, "run_units", lambda args, units: run_calls.extend(units) or 0)
    monkeypatch.setattr(module, "evaluate_daily_completion", lambda *_args, **_kwargs: {"missing_days": []})
    checkpoint_updates = []
    monkeypatch.setattr(module, "update_daily_completion_checkpoint", lambda *args, **kwargs: checkpoint_updates.append((args, kwargs)))

    args = _checkpoint_args(tmp_path)
    module.resolve_missing_daily_args(args)

    assert seen_days == [day]
    assert run_calls == [expected]
    assert checkpoint_updates


def test_weekly_does_not_replay_unit_with_precheckpoint_event_date_by_default(tmp_path, monkeypatch):
    module = load_module()
    day = baseline = "2026-03-15"
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    snapshot_calls = []
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda *_args: snapshot_calls.append(True))
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")
    run_calls = []
    monkeypatch.setattr(module, "run_units", lambda _args, units: run_calls.extend(units) or 0)

    module.resolve_missing_daily_args(_checkpoint_args(tmp_path))

    assert snapshot_calls == []
    assert run_calls == []


def test_weekly_records_uncovered_precheckpoint_dates_without_replaying_them(tmp_path, monkeypatch):
    module = load_module()
    baseline_from, baseline_through = "2026-03-15", "2026-03-30"
    pre_start, uncovered, post = "2026-03-14", "2026-03-20", "2026-03-31"
    checkpoint_path = tmp_path / "daily_completion_checkpoint.json"
    checkpoint = module.load_daily_completion_checkpoint(tmp_path, "hermes")[0]
    checkpoint = module.provisionalize_checkpoint_for_repair(checkpoint, [baseline_through])
    checkpoint = module.record_completed_days(
        checkpoint,
        {baseline_from: [], baseline_through: []},
        source="verified_frozen_completion_scope",
        advance_through=baseline_through,
    )
    module.save_json(checkpoint_path, checkpoint)
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [pre_start, baseline_from, uncovered, baseline_through, post])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    expected = unit(module, post, "post-checkpoint facts")
    seen_days = []
    monkeypatch.setattr(
        module,
        "build_daily_completion_snapshot",
        lambda _args, days: (
            seen_days.extend(days) or {post: [expected]},
            {"missing_days": [post], "details": {post: {"expected_units": 1, "missing_units": 1}}},
        ),
    )
    manifest_record = {}
    monkeypatch.setattr(
        module,
        "write_daily_completion_manifest",
        lambda _out, _args, days, _units, completion: manifest_record.update(
            {"days": list(days), "ignored": list(completion.get("pre_checkpoint_ignored_days") or [])}
        ) or tmp_path / "completion.json",
    )
    args = _checkpoint_args(tmp_path)
    args.mode = "dry-run"

    module.resolve_missing_daily_args(args)

    assert seen_days == [post]
    assert args._budget_missing_daily == [post]
    assert manifest_record == {"days": [post], "ignored": [pre_start, uncovered]}


def test_weekly_does_not_backfill_unknown_history_before_checkpoint_by_default(tmp_path, monkeypatch):
    module = load_module()
    baseline = "2026-03-15"
    day = "2026-03-14"
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    snapshot_calls = []
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda *_args: snapshot_calls.append(True))
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")
    run_calls = []
    monkeypatch.setattr(module, "run_units", lambda _args, units: run_calls.extend(units) or 0)

    module.resolve_missing_daily_args(_checkpoint_args(tmp_path))

    assert snapshot_calls == []
    assert run_calls == []


def test_weekly_persists_uncovered_precheckpoint_dates_without_post_units(tmp_path, monkeypatch):
    module = load_module()
    baseline, uncovered = "2026-03-15", "2004-09-01"
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [uncovered])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(
        module,
        "build_daily_completion_snapshot",
        lambda *_args: (_ for _ in ()).throw(AssertionError("no post-checkpoint day should be queried")),
    )

    module.resolve_missing_daily_args(_checkpoint_args(tmp_path))

    checkpoint = json.loads((tmp_path / "daily_completion_checkpoint.json").read_text(encoding="utf-8"))
    audit = checkpoint["last_reconciliation"]
    assert audit["uncovered_pre_checkpoint_dates"] == [uncovered]
    assert audit["post_checkpoint_dates"] == []
    assert audit["baseline_through"] == baseline
    reloaded, bootstrapped = module.load_daily_completion_checkpoint(tmp_path, "hermes")
    assert bootstrapped is False
    assert reloaded["last_reconciliation"] == audit


def test_weekly_fails_closed_if_checkpointed_prebaseline_artifact_disappears(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-03-15"
    expected = unit(module, day, "previously processed facts")
    checkpoint, _ = module.load_daily_completion_checkpoint(tmp_path, "hermes")
    checkpoint = module.record_completed_days(
        checkpoint,
        {day: [expected]},
        source="verified_frozen_completion_scope",
        advance_through=day,
    )
    module.save_json(tmp_path / "daily_completion_checkpoint.json", checkpoint)
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(
        module,
        "build_daily_completion_snapshot",
        lambda _args, days: (
            {day: [expected]},
            {"missing_days": [day], "details": {day: {"expected_units": 1, "missing_units": 1}}},
        ),
    )
    monkeypatch.setattr(module, "write_daily_completion_manifest", lambda *_args: tmp_path / "completion.json")
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")
    run_calls = []
    monkeypatch.setattr(module, "run_units", lambda _args, units: run_calls.extend(units) or 0)

    try:
        module.resolve_missing_daily_args(_checkpoint_args(tmp_path))
    except SystemExit as exc:
        assert "archived checkpoint artifacts are missing or invalid" in str(exc).lower()
    else:
        raise AssertionError("checkpointed historical artifact loss must not trigger automatic LLM repair")

    assert run_calls == []


def test_weekly_failed_catchup_does_not_advance_checkpoint(tmp_path, monkeypatch):
    module = load_module()
    baseline, day = "2026-03-15", "2026-03-16"
    expected = unit(module, day, "source facts")
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda _args, days: ({day: [expected]}, {"missing_days": [day], "details": {day: {"expected_units": 1, "missing_units": 1}}}))
    monkeypatch.setattr(module, "write_daily_completion_manifest", lambda *_args: tmp_path / "completion.json")
    progress_path = tmp_path / "progress.json"
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    checkpoint_updates = []
    monkeypatch.setattr(module, "update_daily_completion_checkpoint", lambda *args, **kwargs: checkpoint_updates.append((args, kwargs)))
    monkeypatch.setattr(module, "run_units", lambda *_args: 1)

    try:
        module.resolve_missing_daily_args(_checkpoint_args(tmp_path))
    except SystemExit:
        pass
    else:
        raise AssertionError("failed daily catch-up must stop the weekly stage")

    assert checkpoint_updates == []


def test_weekly_refuses_unbounded_history_without_checkpoint_or_explicit_opt_in(tmp_path, monkeypatch):
    module = load_module()
    day = "2026-03-16"
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [day])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(module, "build_daily_completion_snapshot", lambda *_args: (_ for _ in ()).throw(AssertionError("must not scan/replay history without a baseline")))
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "missing-progress.json")
    args = _checkpoint_args(tmp_path)
    module.update_daily_completion_checkpoint(
        args,
        {day: [unit(module, day, "one daily success is not an all-history baseline")]},
        source="successful_daily_run",
    )

    try:
        module.resolve_missing_daily_args(args)
    except SystemExit as exc:
        assert "no verified all-history daily checkpoint" in str(exc).lower()
    else:
        raise AssertionError("weekly must not infer a full-history baseline")


def test_explicit_historical_backfill_includes_pre_checkpoint_dates(tmp_path, monkeypatch):
    module = load_module()
    baseline, day = "2026-03-15", "2026-03-14"
    expected = unit(module, day, "source facts")
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [day, baseline])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    seen_days = []
    monkeypatch.setattr(
        module,
        "build_daily_completion_snapshot",
        lambda _args, days: (
            seen_days.extend(days) or {day: [expected]},
            {"missing_days": [day], "details": {day: {"expected_units": 1, "missing_units": 1}}},
        ),
    )
    monkeypatch.setattr(module, "write_daily_completion_manifest", lambda *_args: tmp_path / "completion.json")
    progress_path = tmp_path / "progress.json"
    _write_checkpoint(module, tmp_path, baseline_through=baseline)
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", progress_path)
    run_calls = []
    monkeypatch.setattr(module, "run_units", lambda _args, units: run_calls.extend(units) or 0)
    monkeypatch.setattr(module, "evaluate_daily_completion", lambda *_args, **_kwargs: {"missing_days": []})
    monkeypatch.setattr(module, "update_daily_completion_checkpoint", lambda *_args, **_kwargs: None)

    module.resolve_missing_daily_args(_checkpoint_args(tmp_path, backfill=True))

    assert seen_days == [day, baseline]
    assert run_calls == [expected]


def test_explicit_backfill_downgrades_invalid_checkpoint_before_repair(tmp_path, monkeypatch):
    module = load_module()
    day, later = "2026-03-14", "2026-03-16"
    expected = unit(module, day, "historical facts")
    manifest_dir = tmp_path / "completion_manifests"
    manifest_dir.mkdir()
    manifest = {
        "schema_version": "daily-completion-manifest-v1",
        "generated_at": "2026-03-16T12:00:00",
        "bank": "hermes",
        "artifact_schema_version": module.DAILY_ARTIFACT_SCHEMA_VERSION,
        "coverage_scope": "all-history",
        "coverage_days_hash": sys.modules["hindsight_daily_checkpoint"].date_list_hash([day]),
        "days": [day],
        "units": {
            day: [{"topic": expected.topic, "index": expected.index, "source_ids": expected.source_ids, "input_content_hash": module.short_hash(expected.content, 24)}]
        },
        "missing_before": [],
        "details_before": {day: {"expected_units": 1, "missing_units": 0}},
    }
    (manifest_dir / "verified-daily-completion.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "retained_sqlite_days", lambda _bank: [day])
    monkeypatch.setattr(module, "retained_daily_fact_days", lambda _bank: [day, later])
    monkeypatch.setattr(module, "history_period_for", lambda _days: "all-history")
    monkeypatch.setattr(
        module,
        "build_daily_completion_snapshot",
        lambda _args, _days: (
            {day: [expected], later: []},
            {"missing_days": [day], "details": {day: {"expected_units": 1, "missing_units": 1}}},
        ),
    )
    monkeypatch.setattr(module, "write_daily_completion_manifest", lambda *_args: tmp_path / "repair.json")
    monkeypatch.setattr(module, "DEFAULT_PROGRESS_FILE", tmp_path / "progress.json")
    states_before_run = []

    def fail_before_any_repair(_args, _units):
        state = json.loads((tmp_path / "daily_completion_checkpoint.json").read_text(encoding="utf-8"))
        states_before_run.append(state)
        return 1

    monkeypatch.setattr(module, "run_units", fail_before_any_repair)
    try:
        module.resolve_missing_daily_args(_checkpoint_args(tmp_path, backfill=True))
    except SystemExit:
        pass
    else:
        raise AssertionError("failed explicit history repair should stop")

    assert len(states_before_run) == 1
    assert states_before_run[0]["baseline_through"] is None
    assert states_before_run[0]["baseline_from"] is None
    assert states_before_run[0]["completed_days"] == {}
    assert states_before_run[0]["bootstrap_required"] is True
    reloaded, bootstrapped = module.load_daily_completion_checkpoint(tmp_path, "hermes")
    assert bootstrapped is False
    assert reloaded["baseline_through"] is None
