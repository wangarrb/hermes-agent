"""Consume already exported App pages locally; this never discovers App history.

The supported App-tool producer remains separate. Consumer success is not API
collection freshness or full-account coverage. Dry-run is the default.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from hindsight_app_local_archive import archive_local, daily_local, private_write
from hindsight_app_sessions import convert_pages, sync_sessions
from hindsight_app_usage import UTC8
from hindsight_incremental_sources import checked_path
from hindsight_pipeline_common import hermes_home_from_env

SINCE = "2026-09-09T00:00:00+08:00"


def _validate_target(archive, home):
    archive, home = checked_path(Path(archive)), checked_path(Path(home))
    blocked = [home / "sessions", home / "profiles", home / "hindsight/incremental"]
    if archive == home or any(archive.is_relative_to(p) or p.is_relative_to(archive) for p in blocked):
        raise ValueError("consumer archive must not overlap a production feed")
    return archive, home


def _record_consumer(archive, result):
    path = archive / "collection-status.json"
    status = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(status, dict):
        raise ValueError("invalid collection-status.json")
    # Only this field belongs to the consumer. In particular, preserve producer
    # last_success_at and all listing/archive/account coverage limitations.
    status["consumer"] = {**result, "last_consumed_at": datetime.now(UTC8).isoformat()}
    private_write(path, json.dumps(status, ensure_ascii=False, indent=2) + "\n")


def consume_pending(archive, hermes_home, *, write=False):
    archive, home = _validate_target(archive, hermes_home)
    files = sorted((archive / "exports").glob("page-*.json"))
    result = {
        "status": "collector_missing" if not files else "dry_run",
        "write": write, "export_pages": len(files), "historical_start": SINCE,
        "api_discovery_automated": False, "producer_required": "supported_app_list_read_tools",
        "production_feed_written": False, "hindsight_api_calls": 0,
        "summary_llm_calls": 0, "embedding_calls": 0,
    }
    if not files:
        result["note"] = "No exported pages; the supported App producer has not supplied input."
        if write:
            _record_consumer(archive, result)
        return result

    scope = json.loads((archive / "scope.json").read_text())
    if not isinstance(scope, dict) or not scope:
        raise ValueError("nonempty explicit archive scope is required")
    pages = [json.loads(path.read_text()) for path in files]
    sessions = convert_pages(pages, scope, since=SINCE, allow_partial=True)
    output = archive / "sessions"
    for path in output.glob("session_*.json"):
        if json.loads(path.read_text()).get("source_session_id") not in scope:
            raise ValueError("archive scope cannot silently shrink")
    overlaps = []
    for session in sessions:
        sid = session["source_session_id"]
        legacy = ([path for path in sorted((home / "sessions").glob(f"session_codex-*{sid}.json"))
                   if not path.name.startswith("session_codex-app-")]
                  if session["source"] == "codex-app" else [])
        if legacy:
            overlaps.append(sid)
            session["legacy_cli_session_overlap"] = True
            session["legacy_cli_export_refs"] = [str(path) for path in legacy]
            session["tags"] = sorted(set(session.get("tags", [])) | {"legacy_cli_session_overlap"})
            session["coverage"]["legacy_cli_session_overlap"] = True
            session["coverage"]["external_processing_allowed"] = False
    planned = sync_sessions(sessions, output, write=False)
    result.update(
        new_messages=sum(row["new_messages"] for row in planned),
        legacy_cli_session_overlap=sorted(overlaps),
        partial_items=sum(len(s["coverage"].get("partial_items", [])) for s in sessions),
        deferred_turns=sum(len(s["coverage"].get("deferred_turns", [])) for s in sessions),
    )
    if not write:
        return result

    sync_sessions(sessions, output, write=True)
    # Reuse the archive's own ledger and scanner. It is deliberately outside
    # all production feeds. The consumer's new_messages above describes this
    # merge; archive_local then seals unseen events from the merged snapshots.
    manifest, stored = archive_local(output, archive, home, scope)
    result.update(status="consumed", new_local_documents=manifest["new_local_documents"],
                  total_messages=manifest["messages"], local_documents=manifest["local_documents"])
    result["daily_outputs"] = daily_local(stored, archive / "evidence/daily", archive / "daily")
    _record_consumer(archive, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", type=Path, default=hermes_home_from_env())
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--write", action="store_true", help="Save local archive/ledger/sidecars only; never call an API or model")
    args = parser.parse_args(argv)
    archive = args.archive_root or args.hermes_home / "hindsight/app_archive/account-visible"
    try:
        result = consume_pending(archive, args.hermes_home, write=args.write)
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Do not echo a rejected page/body or other conversation text to cron.
        print(json.dumps({"status": "consumer_failed", "error_class": type(error).__name__,
                          "note": "Local App export validation or archive capture failed; existing CLI sync may continue."}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
