"""One official inventory stream must preserve the legacy audit's findings.

All HTTP is served by an ephemeral loopback fixture; no production API or bank
is used. The legacy oracle calls the existing standalone helpers in their old
order. Assertions compare data and request counts, never source text.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
BANK = "single-scan-fixture"
TYPES = ["world", "experience", "observation"]
UUID = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
DOC = "hermes-offline-consolidation::daily::2026-01-01::topic::01::aaaaaaaaaaaa"


def _audit():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("single_scan_audit", ROOT / "hindsight_conflict_audit.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(bounded=False):
    rows = {typ: [] for typ in TYPES}
    if bounded == "outlier-cap":
        # A deterministic synthetic inventory verifies the unchanged top-20
        # cutoff, fact-count secondary key, lexical ties and the 999 boundary.
        for i in range(23):
            typ = TYPES[i % len(TYPES)]
            for j in range(1000 + i % 3):
                text = "PENDING_BREAKDOWN" if j == 0 else "ordinary fact"
                rows[typ].append({"id": f"doc-{i:02d}-{j}", "document_id": f"doc-{i:02d}", "fact_type": typ, "text": text})
        rows["world"].extend({"id": f"below-{i}", "document_id": "below-threshold", "text": "ordinary fact"} for i in range(999))
        return rows
    for typ, doc, count in zip(TYPES, ["doc-z", "doc-a", "doc-b"], [1000, 1000, 1001]):
        for i in range(count):
            text = "Traceback (most recent call last) fixture" if i < (60 if bounded else 14) else "ordinary fact"
            if typ == "observation" and i == count - 1:
                text = "[CONTEXT COMPACTION] fixture" if bounded else "PENDING_BREAKDOWN"
            rows[typ].append({"id": f"{typ}-{i}", "document_id": doc, "fact_type": typ, "text": text})
    rows["world"][100]["id"] = UUID
    # Native deduplication is case-sensitive; the known-ID set is not.
    rows["experience"].insert(0, dict(rows["world"][0]))
    rows["experience"].append({"id": UUID.lower(), "text": "ordinary fact"})
    rows["world"].extend([{"text": "ordinary fact"}, {"id": "no-document", "text": "ordinary fact"}])
    return rows


@contextmanager
def _api(rows, failure=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            typ = query.get("type", [None])[0]
            offset = int(query.get("offset", [0])[0])
            limit = int(query.get("limit", [100])[0])
            requests.append(("GET", url.path, typ, offset, limit))
            suffix = url.path.removeprefix(f"/v1/default/banks/{BANK}/")
            fail = failure is not None and (suffix, typ, offset) == failure[:3]
            if fail:
                assert failure is not None
                status = 503 if failure[3] == "http" else 200
                body = b"unavailable" if status == 503 else b"not-json"
            elif suffix == "documents":
                status = 200
                docs = [{"id": DOC}, {"id": "doc-a"}, {"id": "doc-b"}, {"id": "doc-z"}]
                body = json.dumps({"items": docs[offset:offset + limit], "total": len(docs)}).encode()
            elif suffix == "memories/list" and typ in TYPES:
                status = 200
                page = rows[typ][offset:offset + limit]
                body = json.dumps({"items": page, "total": len(rows[typ]), "limit": limit, "offset": offset}).encode()
            else:
                status, body = 404, b"unexpected endpoint"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _stable(value):
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in {"created_at", "generated_at", "json_path", "markdown_path"}}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _inputs(tmp_path, opaque=False):
    cards = tmp_path / "cards"
    offline = tmp_path / "offline"
    cards.mkdir()
    (offline / "daily").mkdir(parents=True)
    local_uuid = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    local_doc = "hermes-offline-consolidation::daily::2026-01-02::topic::01::bbbbbbbbbbbb"
    (offline / "daily" / "source.json").write_text(json.dumps({"document_id": local_doc, "evidence_ids": [local_uuid]}))
    if opaque:
        import hindsight_incremental_merge as merge
        parent = {"bank": BANK, "seed_sha256": merge.canonical_hash({"fixture": "opaque-seed"})}
        seed = merge.import_legacy_seed([
            {"id": UUID, "bank_id": BANK, "text": "[CONTEXT COMPACTION] opaque fixture", "metadata": {"untouched": [1, 2]}},
            {"id": "opaque-second", "bank_id": BANK, "text": "ordinary opaque fixture"},
        ], bank=BANK, provenance=parent)
        manifest = merge.write_generation(tmp_path / "frozen", bank=BANK, parent=parent,
                                          observations=seed, consumed_batch_ids=[])
        return Path(manifest["root"]), offline
    observations = [
        {"id": "uuid-case", "insight": "traceable fact", "evidence_ids": [UUID]},
        {"id": "alias-case", "insight": "traceable alias", "source_documents": [DOC.replace("aaaaaaaaaaaa", "cccccccccccc")]},
        {"id": "local-case", "insight": "local lineage", "source_documents": [local_doc], "evidence_ids": [local_uuid]},
        {"id": "stale-case", "insight": "PENDING_BREAKDOWN", "source_documents": [DOC, "hermes-missing"], "evidence_ids": ["cccccccc-dddd-eeee-ffff-000000000000"]},
        {"id": "numeric-a", "topic": "benchmark", "type": "risk", "tags": ["alpha", "benchmark", "metric"], "insight": "alpha benchmark metric ATE=0.5m default publish", "source_documents": [DOC]},
        {"id": "numeric-b", "topic": "benchmark", "type": "risk", "tags": ["alpha", "benchmark", "metric"], "insight": "alpha benchmark metric ATE=2.0m not default blocked", "source_documents": [DOC]},
    ]
    observations.append(dict(observations[3]))
    (cards / "observations_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in observations))
    return cards, offline


def _legacy(audit, api, cards, offline, flags):
    manual_only = "--manual-only" in flags
    skip = "--skip-source-scan" in flags or "--skip-db-scan" in flags
    client = audit.HindsightNativeClient(api=api, bank=BANK)
    audit.API_SCAN_ERRORS.clear()
    observations = audit.load_observations(cards)
    docs = audit.known_document_ids(BANK, client=client) if not manual_only else set()
    mems = audit.known_memory_ids(BANK, client=client) if not manual_only else set()
    local_docs, local_mems = audit.known_local_lineage_ids(offline) if not manual_only else (set(), set())
    files = audit.known_files(offline) if not manual_only else set()
    cases = []
    if not manual_only:
        cases.extend(audit.core.build_conflict_cases(observations, known_document_ids=docs | local_docs,
                                                   known_memory_ids=mems | local_mems, known_file_paths=files, max_cases=500))
        if not skip:
            if not audit.API_SCAN_ERRORS:
                cases.extend(audit.db_contamination_cases(BANK, client=client))
                cases.extend(audit.fact_outlier_cases(BANK, client=client))
            cases.extend(audit.source_scan_unavailable_cases())
    manual = []
    if "--manual-claim" in flags:
        manual.append(audit.core.manual_conflict_case(claim=flags[flags.index("--manual-claim") + 1], severity="P1"))
    cases = audit.dedupe_cases(cases + manual)
    severity = flags[flags.index("--block-severity") + 1] if "--block-severity" in flags else "P1"
    summary = audit.summarize(cases, severity)
    return {
        "cases": cases, "summary": summary,
        "decision": "pass" if summary["blocking_cases"] == 0 else "blocked_conflict_review_required",
        "known_counts": {"documents": len(docs | local_docs), "memory_units": len(mems | local_mems),
                         "api_documents": len(docs), "api_memory_units": len(mems),
                         "local_lineage_documents": len(local_docs), "local_lineage_memory_ids": len(local_mems), "files": len(files)},
        "source_scan_errors": list(audit.API_SCAN_ERRORS), "db_scan_errors": [],
        "observation_count": len(observations), "manual_case_count": len(manual), "manual_only": manual_only,
    }


@pytest.mark.parametrize("bounded,flags,failure", [
    (False, [], None),
    (True, [], None),
    (True, ["--block-severity", "P2", "--manual-claim", "fixture review"], None),
    (True, ["--skip-source-scan"], None),
    (True, ["--skip-db-scan", "--manual-claim", "fixture review"], None),
    (True, ["--manual-only"], None),
    (True, ["--manual-only", "--manual-claim", "fixture review"], None),
    (False, [], ("documents", None, 0, "http")),
    (False, [], ("memories/list", "world", 0, "http")),
    (True, [], ("memories/list", "world", 100, "http")),
    (True, [], ("memories/list", "experience", 100, "http")),
    (True, [], ("memories/list", "observation", 1000, "json")),
    (True, ["--skip-source-scan"], ("memories/list", "world", 100, "http")),
    (True, ["--manual-only"], ("documents", None, 0, "http")),
    ("outlier-cap", [], None),
    ("opaque", [], None),
])
def test_cli_one_stream_preserves_legacy_audit(tmp_path, bounded, flags, failure, record_property):
    audit = _audit()
    rows = _rows(bounded)
    cards, offline = _inputs(tmp_path, opaque=bounded == "opaque")
    index_before = (cards / "observations_index.jsonl").read_bytes()
    with _api(rows, failure) as (api, requests):
        legacy = _legacy(audit, api, cards, offline, flags)
        old_requests = list(requests)
        requests.clear()
        output = tmp_path / "audit-output"
        proc = subprocess.run([sys.executable, str(ROOT / "hindsight_conflict_audit.py"),
                               "--cards-root", str(cards), "--offline-root", str(offline),
                               "--output-dir", str(output), "--bank", BANK, "--api", api, "--json", *flags],
                              capture_output=True, text=True, timeout=45,
                              env={**os.environ, "HERMES_HOME": str(tmp_path / "hermes-home"), "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"})
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)
        assert json.loads((output / "latest.json").read_text()) == report
        assert json.loads(Path(report["json_path"]).read_text()) == report
        assert (cards / "observations_index.jsonl").read_bytes() == index_before
        assert Path(report["markdown_path"]).read_text() == (output / "latest.md").read_text()
        assert _stable({key: report[key] for key in legacy}) == _stable(legacy)
        old_starts = sum(path.endswith("memories/list") and typ == "world" and offset == 0 for _, path, typ, offset, _ in old_requests)
        new_starts = sum(path.endswith("memories/list") and typ == "world" and offset == 0 for _, path, typ, offset, _ in requests)
        record_property("legacy_inventory_streams", old_starts)
        record_property("new_inventory_streams", new_starts)
        record_property("legacy_memory_gets", sum(r[1].endswith("memories/list") for r in old_requests))
        record_property("new_memory_gets", sum(r[1].endswith("memories/list") for r in requests))
        record_property("parity", "cases/order/known_counts/document_counts/summary/errors")
        assert new_starts == (0 if "--manual-only" in flags else 1), f"inventory streams: legacy={old_starts}, new={new_starts}"
        if failure is None and not flags:
            assert old_starts == 3
            old_gets = sum(r[1].endswith("memories/list") for r in old_requests)
            new_gets = sum(r[1].endswith("memories/list") for r in requests)
            assert old_gets > new_gets
            if bounded is False or bounded == "outlier-cap":
                assert old_gets == 3 * new_gets
            contamin = [c for c in report["cases"] if c["type"] == "source_fact_contamination"]
            outliers = [c for c in report["cases"] if c["type"] == "source_document_fact_outlier"]
            if bounded == "outlier-cap":
                assert contamin == []  # P3 mentions count, but are not source cases.
                expected = sorted(range(23), key=lambda i: (-(1000 + i % 3), f"doc-{i:02d}"))[:20]
                assert [c["target"]["document_id"] for c in outliers] == [f"doc-{i:02d}" for i in expected]
                assert [c["evidence"]["facts"] for c in outliers] == [1000 + i % 3 for i in expected]
                assert all(c["evidence"]["pollutant_facts"] == 1 and c["severity"] == "P2" for c in outliers)
            else:
                assert len(contamin) == (50 if bounded else 42)
                assert [c["target"]["document_id"] for c in outliers] == ["doc-b", "doc-a", "doc-z"]
                assert [c["evidence"]["facts"] for c in outliers] == [1001, 1000, 1000]
                assert [c["evidence"]["pollutant_facts"] for c in outliers] == ([61, 60, 60] if bounded else [15, 14, 14])
                assert outliers[0]["severity"] == ("P1" if bounded else "P2")
                assert report["decision"] == ("blocked_conflict_review_required" if bounded else "pass")
            if bounded == "opaque":
                assert report["observation_count"] == 2
                assert any(c["type"] == "contamination" and c["target"]["id"] == UUID and c["severity"] == "P1" for c in report["cases"])
        if failure and "--manual-only" not in flags:
            unavailable = [c for c in report["cases"] if c["type"] == "source_scan_unavailable"]
            skip = "--skip-source-scan" in flags or "--skip-db-scan" in flags
            assert len(unavailable) == (0 if skip else 1)
            if not skip:
                assert unavailable[0]["severity"] == "P1"
                assert report["decision"] == "blocked_conflict_review_required"
                assert not [c for c in report["cases"] if c["type"] in {"source_fact_contamination", "source_document_fact_outlier"}]
            if failure[0] == "memories/list":
                assert report["known_counts"]["api_memory_units"] == 0
        assert all(method == "GET" and path.startswith(f"/v1/default/banks/{BANK}/") for method, path, *_ in requests)


@pytest.mark.parametrize("cap,include_source_cases,failure", [
    (107, True, None), (200000, True, None), (200000, False, None),
    (200000, True, ("memories/list", "world", 100, "http")),
    (200000, True, ("memories/list", "observation", 1000, "json")),
])
def test_inventory_real_iterator_bounds_and_errors(tmp_path, cap, include_source_cases, failure):
    audit = _audit()
    with _api(_rows(True), failure) as (api, requests):
        class BoundedClient(audit.HindsightNativeClient):
            def iter_memories(self, **kwargs):
                return super().iter_memories(**{**kwargs, "max_items": min(cap, kwargs.get("max_items", cap))})

        client = BoundedClient(api=api, bank=BANK)
        audit.API_SCAN_ERRORS.clear()
        expected_ids = audit.known_memory_ids(BANK, client=client, max_items=cap)
        errors = list(audit.API_SCAN_ERRORS)
        expected_contamination, expected_outliers = [], []
        if include_source_cases and not errors:
            expected_contamination = audit.db_contamination_cases(BANK, client=client)
            expected_outliers = audit.fact_outlier_cases(BANK, client=client)
        requests.clear()
        audit.API_SCAN_ERRORS.clear()
        actual = audit.scan_memory_inventory(BANK, client=client, max_items=cap, include_source_cases=include_source_cases)
        assert _stable(list(actual)) == _stable([expected_ids, expected_contamination, expected_outliers])
        assert audit.API_SCAN_ERRORS == errors
        assert sum(r[1].endswith("memories/list") and r[2] == "world" and r[3] == 0 for r in requests) == 1
        if cap == 107:
            assert len(actual[0]) == cap
            assert [r[3] for r in requests] == [0, 100]
            assert all(r[2] == "world" for r in requests)
            assert actual[2] == []  # The unchanged cap is not a whole-bank coverage claim.
        if failure:
            assert actual == (set(), [], [])
            assert audit.source_scan_unavailable_cases()[0]["severity"] == "P1"
