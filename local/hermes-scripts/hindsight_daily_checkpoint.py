"""Durable, input-signature checkpoints for daily Hindsight artifacts.

The checkpoint records content-free unit identities from verified completion
manifests and successful daily runs. It never contains fact text or LLM output.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

CHECKPOINT_SCHEMA_VERSION = "daily-completion-checkpoint-v1"
CHECKPOINT_API_VERSION = 1
MANIFEST_SCHEMA_VERSION = "daily-completion-manifest-v1"
DEFAULT_ARTIFACT_SCHEMA_VERSION = "daily-artifact-v2"
CHECKPOINT_FILENAME = "daily_completion_checkpoint.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _unit_value(unit: Any, name: str, default: Any = None) -> Any:
    if isinstance(unit, Mapping):
        return unit.get(name, default)
    return getattr(unit, name, default)


def signature_record(unit: Any, *, require_explicit_hash: bool = False) -> dict[str, Any]:
    """Return the exact input identity used by completion manifests/checkpoints."""
    topic = str(_unit_value(unit, "topic") or "")
    index = int(_unit_value(unit, "index", -1))
    raw_ids = _unit_value(unit, "source_ids", [])
    if not isinstance(raw_ids, (list, tuple)):
        raise ValueError("daily checkpoint source_ids must be a list")
    source_ids = sorted(str(value) for value in raw_ids)
    content_hash = _unit_value(unit, "input_content_hash")
    if require_explicit_hash and (not isinstance(content_hash, str) or not content_hash):
        raise ValueError("daily checkpoint manifest/signature must provide input_content_hash")
    if not content_hash:
        content_hash = _sha256_text(str(_unit_value(unit, "content", "")))[:24]
    if not topic or index < 0 or not source_ids or len(str(content_hash)) != 24 or any(char not in "0123456789abcdef" for char in str(content_hash)):
        raise ValueError("daily checkpoint unit signature is incomplete or has an invalid content hash")
    return {
        "topic": topic,
        "index": index,
        "source_ids": source_ids,
        "input_content_hash": str(content_hash),
    }


def signature_records(units: list[Any], *, require_explicit_hash: bool = False) -> list[dict[str, Any]]:
    records = [signature_record(unit, require_explicit_hash=require_explicit_hash) for unit in units]
    return sorted(
        records,
        key=lambda item: (
            item["topic"],
            item["index"],
            tuple(item["source_ids"]),
            item["input_content_hash"],
        ),
    )


def signature_set_hash(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical)


def _valid_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def date_list_hash(days: list[str]) -> str:
    normalized: list[str] = []
    for value in days:
        day = _valid_date(value)
        if day is None:
            raise ValueError(f"invalid daily completion date: {value!r}")
        normalized.append(day)
    if len(normalized) != len(set(normalized)):
        raise ValueError("daily completion date list contains duplicates")
    canonical = json.dumps(sorted(normalized), ensure_ascii=False, separators=(",", ":"))
    return _sha256_text(canonical)


def _verified_manifest_checkpoint(
    path: Path,
    *,
    bank: str,
    artifact_schema_version: str,
    legacy_history_days: set[str] | list[str] | None = None,
) -> dict[str, Any] | None:
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("artifact_schema_version") != artifact_schema_version
        or payload.get("bank") != bank
        or payload.get("missing_before") != []
    ):
        return None
    days_raw = payload.get("days")
    units_by_day = payload.get("units")
    details = payload.get("details_before")
    if not isinstance(days_raw, list) or not days_raw or not isinstance(units_by_day, dict) or not isinstance(details, dict):
        return None

    days: list[str] = []
    completed_days: dict[str, dict[str, Any]] = {}
    for value in days_raw:
        day = _valid_date(value)
        if day is None or day in completed_days:
            return None
        raw_units = units_by_day.get(day)
        detail = details.get(day)
        if not isinstance(raw_units, list) or not isinstance(detail, dict):
            return None
        try:
            if int(detail.get("missing_units", -1)) != 0:
                return None
            if int(detail.get("expected_units", -1)) != len(raw_units):
                return None
            records = signature_records(raw_units, require_explicit_hash=True)
        except (TypeError, ValueError):
            return None
        days.append(day)
        completed_days[day] = {
            "unit_count": len(records),
            "unit_set_hash": signature_set_hash(records),
            "unit_signatures": records,
            "completed_at": str(payload.get("generated_at") or ""),
            "source": "verified_completion_manifest",
        }

    if set(units_by_day) != set(days):
        return None
    coverage_scope = payload.get("coverage_scope")
    if coverage_scope == "all-history":
        if payload.get("coverage_days_hash") != date_list_hash(days):
            return None
        verified_coverage_scope = "all-history"
    elif coverage_scope is None and legacy_history_days is not None:
        legacy_days: set[str] = set()
        for value in legacy_history_days:
            day = _valid_date(value)
            if day is None:
                return None
            legacy_days.add(day)
        if set(days) != legacy_days:
            return None
        verified_coverage_scope = "legacy-document-inventory"
    else:
        return None
    latest = max(days)
    baseline_from = min(days)
    verified_days = sorted(days)
    manifest_sha = hashlib.sha256(raw_bytes).hexdigest()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "bank": bank,
        "baseline_from": baseline_from,
        "baseline_through": latest,
        "latest_successful_date": latest,
        "updated_at": str(payload.get("generated_at") or ""),
        "completed_days": completed_days,
        "baseline_manifest": {
            "path": str(path),
            "sha256": manifest_sha,
            "generated_at": str(payload.get("generated_at") or ""),
            "coverage_scope": verified_coverage_scope,
        },
        "baseline_provenance": {
            "kind": "completion-manifest",
            "verified_from": baseline_from,
            "verified_through": latest,
            "verified_days": verified_days,
            "verified_days_hash": date_list_hash(verified_days),
            "manifest_sha256": manifest_sha,
            "coverage_scope": verified_coverage_scope,
        },
    }


def _validate_checkpoint_state(existing: Any, bank: str, output_dir: Path) -> dict[str, Any]:
    if not isinstance(existing, dict) or existing.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("daily completion checkpoint has an unsupported schema")
    if (
        existing.get("bank") != bank
        or existing.get("output_dir") != str(Path(output_dir).expanduser().resolve())
        or not isinstance(existing.get("completed_days"), dict)
    ):
        raise ValueError("daily completion checkpoint bank/output/state does not match this run")
    baseline_from = existing.get("baseline_from")
    baseline = existing.get("baseline_through")
    if baseline_from is not None and _valid_date(baseline_from) is None:
        raise ValueError("daily completion checkpoint has an invalid baseline start date")
    if baseline is not None and _valid_date(baseline) is None:
        raise ValueError("daily completion checkpoint has an invalid baseline date")
    if (baseline is None) != (baseline_from is None):
        raise ValueError("daily completion checkpoint baseline range is incomplete")
    if baseline is not None and baseline_from > baseline:
        raise ValueError("daily completion checkpoint baseline range is reversed")
    latest_dates: list[str] = []
    completed_dates: set[str] = set()
    if baseline is not None:
        latest_dates.append(str(baseline))
    for raw_day, record in existing["completed_days"].items():
        day = _valid_date(raw_day)
        if day is None or not isinstance(record, dict):
            raise ValueError("daily completion checkpoint contains an invalid day record")
        raw_signatures = record.get("unit_signatures")
        if not isinstance(raw_signatures, list):
            raise ValueError(f"daily completion checkpoint signatures are invalid for {day}")
        try:
            signatures = signature_records(raw_signatures, require_explicit_hash=True)
            count = int(record.get("unit_count", -1))
        except (TypeError, ValueError):
            raise ValueError(f"daily completion checkpoint signatures are invalid for {day}") from None
        if (
            count != len(signatures)
            or signatures != raw_signatures
            or record.get("unit_set_hash") != signature_set_hash(signatures)
        ):
            raise ValueError(f"daily completion checkpoint signature hash is invalid for {day}")
        completed_dates.add(day)
        latest_dates.append(day)
    if baseline is not None and (str(baseline) not in completed_dates or str(baseline_from) not in completed_dates):
        raise ValueError("daily completion checkpoint watermark lacks exact baseline day records")
    if baseline is not None:
        provenance = existing.get("baseline_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("daily completion checkpoint watermark provenance is missing")
        verified_from = _valid_date(provenance.get("verified_from"))
        verified_through = _valid_date(provenance.get("verified_through"))
        verified_days = provenance.get("verified_days")
        if (
            verified_from != baseline_from
            or verified_through != baseline
            or not isinstance(verified_days, list)
            or any(_valid_date(day) is None for day in verified_days)
            or verified_days != sorted(set(verified_days))
            or str(provenance.get("verified_days_hash") or "") != date_list_hash(verified_days)
            or baseline_from not in verified_days
            or baseline not in verified_days
            or not set(verified_days).issubset(completed_dates)
        ):
            raise ValueError("daily completion checkpoint watermark provenance is invalid")
        provenance_kind = provenance.get("kind")
        if provenance_kind == "completion-manifest":
            manifest = existing.get("baseline_manifest")
            manifest_sha = provenance.get("manifest_sha256")
            coverage_scope = provenance.get("coverage_scope")
            if (
                not isinstance(manifest, dict)
                or not isinstance(manifest_sha, str)
                or len(manifest_sha) != 64
                or any(char not in "0123456789abcdef" for char in manifest_sha)
                or manifest.get("sha256") != manifest_sha
                or manifest.get("coverage_scope") != coverage_scope
                or coverage_scope not in {"all-history", "legacy-document-inventory"}
            ):
                raise ValueError("daily completion checkpoint manifest provenance is invalid")
        elif provenance_kind == "verified-frozen-candidate-set":
            candidate_days = provenance.get("candidate_days")
            try:
                candidate_days_valid = (
                    isinstance(candidate_days, list)
                    and candidate_days == sorted(set(candidate_days))
                    and date_list_hash(candidate_days) == provenance.get("candidate_days_hash")
                )
            except (TypeError, ValueError):
                candidate_days_valid = False
            if not candidate_days_valid or baseline not in candidate_days or not set(candidate_days).issubset(set(verified_days)):
                raise ValueError("daily completion checkpoint frozen-scope provenance is invalid")
        else:
            raise ValueError("daily completion checkpoint watermark provenance has an unsupported kind")
    expected_latest = max(latest_dates) if latest_dates else None
    if existing.get("latest_successful_date") != expected_latest:
        raise ValueError("daily completion checkpoint latest-success watermark is inconsistent")
    audit = existing.get("last_reconciliation")
    if audit is not None:
        if not isinstance(audit, dict) or type(audit.get("discovered_date_count")) is not int or audit["discovered_date_count"] < 0:
            raise ValueError("daily completion checkpoint reconciliation audit is invalid")
        discovered_hash = audit.get("discovered_dates_hash")
        if not isinstance(discovered_hash, str) or len(discovered_hash) != 64 or any(c not in "0123456789abcdef" for c in discovered_hash):
            raise ValueError("daily completion checkpoint reconciliation date hash is invalid")
        for dates_key, hash_key in (
            ("post_checkpoint_dates", "post_checkpoint_dates_hash"),
            ("uncovered_pre_checkpoint_dates", "uncovered_pre_checkpoint_dates_hash"),
        ):
            values = audit.get(dates_key)
            try:
                valid_hash = isinstance(values, list) and date_list_hash(values) == audit.get(hash_key)
            except ValueError:
                valid_hash = False
            if not valid_hash:
                raise ValueError(f"daily completion checkpoint reconciliation {dates_key} is invalid")
        provisionalized_days = audit.get("provisionalized_checkpoint_days")
        provisionalized_hash = audit.get("provisionalized_checkpoint_days_hash")
        if provisionalized_days is not None or provisionalized_hash is not None:
            try:
                valid_provisionalized = isinstance(provisionalized_days, list) and date_list_hash(provisionalized_days) == provisionalized_hash
            except ValueError:
                valid_provisionalized = False
            if not valid_provisionalized:
                raise ValueError("daily completion checkpoint reconciliation provisionalized dates are invalid")
        excluded_anomalous = audit.get("excluded_anomalous_dates")
        excluded_anomalous_hash = audit.get("excluded_anomalous_dates_hash")
        if excluded_anomalous is not None or excluded_anomalous_hash is not None:
            try:
                valid_excluded_anomalous = isinstance(excluded_anomalous, list) and date_list_hash(excluded_anomalous) == excluded_anomalous_hash
            except ValueError:
                valid_excluded_anomalous = False
            if not valid_excluded_anomalous:
                raise ValueError("daily completion checkpoint reconciliation anomalous dates are invalid")
        for key in ("baseline_from", "baseline_through", "prior_baseline_from", "prior_baseline_through"):
            value = audit.get(key)
            if value is not None and _valid_date(value) is None:
                raise ValueError(f"daily completion checkpoint reconciliation {key} is invalid")
    return existing


def provisionalize_checkpoint_for_repair(checkpoint: dict[str, Any], invalid_days: list[str]) -> dict[str, Any]:
    """Drop unsafe day claims and remove the watermark before explicit history repair."""
    state = copy.deepcopy(checkpoint)
    completed = state.get("completed_days")
    if not isinstance(completed, dict):
        completed = {}
    if "<invalid-checkpoint>" in invalid_days:
        completed = {}
    else:
        for day in invalid_days:
            completed.pop(str(day), None)
    state["completed_days"] = completed
    state["baseline_from"] = None
    state["baseline_through"] = None
    state["baseline_manifest"] = None
    state["baseline_provenance"] = None
    state["bootstrap_required"] = True
    valid_days = [day for day in completed if _valid_date(day) is not None]
    state["latest_successful_date"] = max(valid_days) if valid_days else None
    state["updated_at"] = datetime.now().isoformat()
    state.pop("last_reconciliation", None)
    return state


def load_daily_completion_checkpoint(
    output_dir: Path,
    bank: str,
    *,
    artifact_schema_version: str = DEFAULT_ARTIFACT_SCHEMA_VERSION,
    legacy_history_days: set[str] | list[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Load the checkpoint, or bootstrap only from a complete all-history manifest.

    A current manifest must declare and hash its all-history date inventory. A
    markerless legacy manifest is accepted only when its dates exactly match the
    caller-supplied legacy inventory. The boolean reports bootstrap provenance.
    An empty state is returned when there is no safe baseline.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    if checkpoint_path.exists():
        try:
            existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"daily completion checkpoint cannot be read: {checkpoint_path}") from exc
        return _validate_checkpoint_state(existing, bank, output_dir), False

    manifest_dir = output_dir / "completion_manifests"
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    if manifest_dir.is_dir():
        for path in manifest_dir.glob("*.json"):
            checkpoint = _verified_manifest_checkpoint(
                path,
                bank=bank,
                artifact_schema_version=artifact_schema_version,
                legacy_history_days=legacy_history_days,
            )
            if checkpoint is not None:
                candidates.append((checkpoint["updated_at"], path, checkpoint))
    if candidates:
        _generated_at, _path, checkpoint = max(candidates, key=lambda item: (item[0], item[1].name))
        checkpoint["output_dir"] = str(output_dir)
        return checkpoint, True

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "bank": bank,
        "output_dir": str(output_dir),
        "baseline_from": None,
        "baseline_through": None,
        "latest_successful_date": None,
        "updated_at": None,
        "completed_days": {},
        "baseline_manifest": None,
        "baseline_provenance": None,
        "bootstrap_required": True,
    }, False


def checkpoint_contains_unit(checkpoint: dict[str, Any] | None, unit: Any) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    day = str(_unit_value(unit, "period") or "")
    day_record = (checkpoint.get("completed_days") or {}).get(day)
    if not isinstance(day_record, dict):
        return False
    try:
        expected = signature_record(unit)
        known = day_record.get("unit_signatures")
        if not isinstance(known, list):
            return False
        return expected in known
    except (TypeError, ValueError):
        return False


def record_completed_days(
    checkpoint: dict[str, Any],
    units_by_day: dict[str, list[Any]],
    *,
    source: str,
    advance_through: str | None = None,
) -> dict[str, Any]:
    """Merge fully verified day snapshots and optionally advance the safe watermark."""
    completed = checkpoint.setdefault("completed_days", {})
    if not isinstance(completed, dict):
        raise ValueError("daily completion checkpoint completed_days must be an object")
    now = datetime.now().isoformat()
    for raw_day, units in units_by_day.items():
        day = _valid_date(raw_day)
        if day is None:
            raise ValueError(f"invalid daily checkpoint date: {raw_day!r}")
        records = signature_records(list(units))
        completed[day] = {
            "unit_count": len(records),
            "unit_set_hash": signature_set_hash(records),
            "unit_signatures": records,
            "completed_at": now,
            "source": source,
        }
    if advance_through is not None:
        normalized = _valid_date(advance_through)
        if normalized is None:
            raise ValueError(f"invalid checkpoint watermark: {advance_through!r}")
        current = checkpoint.get("baseline_through")
        if current is None or normalized > current:
            checkpoint["baseline_through"] = normalized
            if checkpoint.get("baseline_from") is None and completed:
                checkpoint["baseline_from"] = min(day for day in completed if _valid_date(day) is not None)
            baseline_from = _valid_date(checkpoint.get("baseline_from"))
            candidate_days = sorted(_valid_date(day) for day in units_by_day)
            verified_days = sorted(
                day for day in completed
                if _valid_date(day) is not None and baseline_from is not None and baseline_from <= day <= normalized
            )
            if baseline_from is None or normalized not in candidate_days or baseline_from not in verified_days or normalized not in verified_days:
                raise ValueError("daily checkpoint watermark advancement requires a verified frozen candidate set")
            checkpoint["baseline_provenance"] = {
                "kind": "verified-frozen-candidate-set",
                "verified_from": baseline_from,
                "verified_through": normalized,
                "verified_days": verified_days,
                "verified_days_hash": date_list_hash(verified_days),
                "candidate_days": candidate_days,
                "candidate_days_hash": date_list_hash(candidate_days),
                "verified_at": now,
            }
    dates = [day for day in completed if _valid_date(day) is not None]
    baseline = checkpoint.get("baseline_through")
    if baseline is not None and _valid_date(baseline) is not None:
        dates.append(str(baseline))
    checkpoint["latest_successful_date"] = max(dates) if dates else None
    checkpoint["updated_at"] = now
    if checkpoint.get("baseline_through") is None:
        checkpoint["bootstrap_required"] = True
    else:
        checkpoint.pop("bootstrap_required", None)
    return checkpoint
