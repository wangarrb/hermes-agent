#!/usr/bin/env python3
"""
Canonical Egomotion4D pitfall index writer with stable-ID allocation.

This is the sole writer for pitfall_index.json. All other code (pipeline,
cron, daily maintenance) must go through this module to add, update, or
adjudicate pitfalls. No other file is allowed to write pitfall_index.json
or pitfall-catalog.md directly.

Stable ID allocation uses a monotonic counter to assign P-ids that never
recycle. The high-water mark is stored inside the index file as next_p_id
so that IDs survive file rebuilds.

Usage:
  from pitfall_writer import PitfallIndex, PitfallStatus

  idx = PitfallIndex.load('/path/to/pitfall_index.json')
  p_id = idx.add_entry(
      title="...",
      status=PitfallStatus.CANDIDATE,
      trigger="...",
      root_cause="...",
      lesson="...",
      tags=["..."],
      source="...",
      detail_locator="...",
  )
  idx.adjudicate_candidates({...})   # candidate -> current/superseded/rejected
  idx.dedup()                        # merge duplicates, reassign aliases
  idx.recalc_counters()
  idx.save()
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Status enum ──────────────────────────────────────────────────────────────

class PitfallStatus(str, Enum):
    """Legal pitfall status values.

    ALIAS is not stored in the entry itself; it is computed from alias_of.
    """
    CURRENT = "current"
    SUPERSEDED = "superseded"
    REJECTED = "rejected_non_algorithmic"
    CANDIDATE = "candidate"

    @classmethod
    def is_terminal(cls, s: "PitfallStatus") -> bool:
        """Has this entry been adjudicated?"""
        return s in (cls.CURRENT, cls.SUPERSEDED, cls.REJECTED)


# ── Entry data model ─────────────────────────────────────────────────────────

@dataclass
class PitfallEntry:
    p_id: str
    title: str
    status: str                 # PitfallStatus value
    is_algorithm_level: bool
    trigger: str
    root_cause: str
    lesson: str
    tags: List[str]
    date: str                   # YYYY-MM-DD
    source: str
    detail_locator: str         # e.g. pitfalls.md#P42
    alias_of: Optional[str] = None
    superseded_by: Optional[str] = None

    # Stable provenance (preserved from original source)
    source_memory_id: Optional[str] = None       # Hindsight memory ID
    source_content_hash: Optional[str] = None     # SHA-256 of source content
    created_at: Optional[str] = None              # UTC lifecycle timestamp
    updated_at: Optional[str] = None              # UTC lifecycle timestamp

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None values (except alias_of / superseded_by which signal absence)
        for key in (
            "source_memory_id", "source_content_hash", "created_at", "updated_at"
        ):
            if d[key] is None:
                del d[key]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PitfallEntry":
        # Allow extra keys for forward compatibility
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: d[k] for k in known if k in d}
        return cls(**kwargs)


# ── Stable ID allocator ──────────────────────────────────────────────────────

class StableIDAllocator:
    """Monotonic P-id counter; P-ids never recycle."""

    def __init__(self, next_p_id: int):
        self._next = next_p_id

    @classmethod
    def from_entries(cls, entries: List[PitfallEntry]) -> "StableIDAllocator":
        prev = 0
        for e in entries:
            m = re.match(r"P(\d+)", e.p_id)
            if m:
                prev = max(prev, int(m.group(1)))
        return cls(prev + 1)

    def next(self) -> int:
        n = self._next
        self._next += 1
        return n

    @property
    def next_p_id(self) -> int:
        return self._next


# ── Main index ───────────────────────────────────────────────────────────────

SCHEMA_VERSION = 3  # v3: added per-entry lifecycle timestamps
PITFALL_CATALOG_PATH = (
    "/home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/"
    "pitfall-catalog.md"
)
RETIRED_SOURCE_PATH = "/home/wyr/wiki/auto-maintenance/project/egomotion4d/pitfalls.md"

@dataclass
class PitfallIndex:
    schema_version: int = SCHEMA_VERSION
    created: str = ""
    last_updated: str = ""
    source_file: str = PITFALL_CATALOG_PATH
    next_p_id: int = 1
    total_entries: int = 0
    canonical_count: int = 0
    alias_count: int = 0
    rejected_count: int = 0
    status_counts: Dict[str, int] = field(default_factory=dict)
    entries: List[PitfallEntry] = field(default_factory=list)

    # ── static constructors ────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> "PitfallIndex":
        """Load from disk, migrating older schemas without inventing history."""
        with open(path) as f:
            raw = json.load(f)
        raw["entries"] = raw.get("entries", [])

        if raw.get("schema_version", 1) < 2:
            raw = cls._migrate_v1_to_v2(raw)
        if raw.get("schema_version", 1) < 3:
            raw["schema_version"] = SCHEMA_VERSION
            if raw.get("source_file") == RETIRED_SOURCE_PATH:
                raw["source_file"] = PITFALL_CATALOG_PATH

        entries = [PitfallEntry.from_dict(e) for e in raw.get("entries", [])]

        kwargs = {k: raw.get(k) for k in [
            "schema_version", "created", "last_updated",
            "source_file", "next_p_id", "total_entries",
            "canonical_count", "alias_count", "rejected_count", "status_counts",
        ]}
        kwargs.setdefault("schema_version", SCHEMA_VERSION)
        kwargs.setdefault("created", "")
        kwargs.setdefault("last_updated", "")
        kwargs.setdefault("next_p_id", StableIDAllocator.from_entries(entries).next_p_id)

        idx = cls(**kwargs, entries=entries)
        if not idx.status_counts:
            idx.recalc_counters()
        return idx

    @classmethod
    def _migrate_v1_to_v2(cls, raw: dict) -> dict:
        raw["schema_version"] = SCHEMA_VERSION
        raw.setdefault("next_p_id", 1)
        for e in raw.get("entries", []):
            e.pop("topic", None)   # remove deprecated field
        return raw

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Write index to path. Returns SHA-256 of written content."""
        self._stamp()

        d = {
            "schema_version": self.schema_version,
            "created": self.created,
            "last_updated": self.last_updated,
            "source_file": self.source_file,
            "next_p_id": self.next_p_id,
            "total_entries": self.total_entries,
            "canonical_count": self.canonical_count,
            "alias_count": self.alias_count,
            "rejected_count": self.rejected_count,
            "status_counts": self.status_counts,
            "entries": [e.to_dict() for e in self.entries],
        }

        payload = json.dumps(d, ensure_ascii=False, indent=2) + "\n"

        # Atomic write
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(payload)
        os.rename(tmp, path)

        return hashlib.sha256(payload.encode()).hexdigest()

    # ── internal helpers ────────────────────────────────────────────────────

    def _stamp(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.last_updated = now
        if not self.created:
            self.created = now

    # ── ID allocation ───────────────────────────────────────────────────────

    def alloc_id(self) -> str:
        alloc = StableIDAllocator(self.next_p_id)
        n = alloc.next()
        self.next_p_id = alloc.next_p_id
        return f"P{n}"

    # ── entry CRUD ──────────────────────────────────────────────────────────

    def add_entry(
        self,
        title: str,
        status: PitfallStatus,
        is_algorithm_level: bool,
        trigger: str,
        root_cause: str,
        lesson: str,
        tags: List[str],
        source: str,
        detail_locator: str,
        date: Optional[str] = None,
        alias_of: Optional[str] = None,
        superseded_by: Optional[str] = None,
        source_memory_id: Optional[str] = None,
        source_content_hash: Optional[str] = None,
        p_id: Optional[str] = None,
    ) -> str:
        """Add a new pitfall entry. Returns the assigned p_id."""
        if p_id is None:
            p_id = self.alloc_id()
        now = datetime.now(timezone.utc).isoformat()

        # Validate self-reference
        if alias_of and superseded_by:
            raise ValueError(f"{p_id}: cannot have both alias_of and superseded_by")
        if alias_of == p_id or superseded_by == p_id:
            raise ValueError(f"{p_id}: cannot reference itself")

        entry = PitfallEntry(
            p_id=p_id,
            title=title,
            status=status.value,
            is_algorithm_level=is_algorithm_level,
            trigger=trigger,
            root_cause=root_cause,
            lesson=lesson,
            tags=sorted(set(tags)),
            date=date or datetime.utcnow().strftime("%Y-%m-%d"),
            source=source,
            detail_locator=detail_locator,
            alias_of=alias_of,
            superseded_by=superseded_by,
            source_memory_id=source_memory_id,
            source_content_hash=source_content_hash,
            created_at=now,
            updated_at=now,
        )
        self.entries.append(entry)
        self.recalc_counters()
        return p_id

    def update_entry(self, p_id: str, **kwargs) -> PitfallEntry:
        """Update mutable fields of an existing entry."""
        entry = self._get(p_id)
        for k, v in kwargs.items():
            if k not in PitfallEntry.__dataclass_fields__:
                raise KeyError(f"Unknown field: {k}")
            if k == "p_id" and v != p_id:
                raise ValueError("Cannot change p_id")
            setattr(entry, k, v)
        if kwargs:
            entry.updated_at = datetime.now(timezone.utc).isoformat()
        return entry

    def remove_entry(self, p_id: str) -> None:
        self.entries = [e for e in self.entries if e.p_id != p_id]

    # ── adjudication ────────────────────────────────────────────────────────

    def adjudicate_candidates(
        self,
        decisions: Dict[str, Tuple[PitfallStatus, Optional[str], Optional[str]]],
    ) -> int:
        """Adjudicate candidate entries in one atomic step.

        decisions: {p_id: (target_status, superseded_by_or_None, alias_of_or_None)}
        Returns number of entries changed.
        """
        changed = 0
        now = datetime.now(timezone.utc).isoformat()
        for p_id_str, (target_status, superseded_by, alias_of) in decisions.items():
            entry = self._get(p_id_str)

            # Validate self-reference
            if alias_of == p_id_str or superseded_by == p_id_str:
                raise ValueError(f"{p_id_str}: cannot reference itself")
            if alias_of and superseded_by:
                raise ValueError(f"{p_id_str}: cannot have both alias_of and superseded_by")
            # Validate target exists
            if superseded_by:
                self._get(superseded_by)
            if alias_of:
                self._get(alias_of)

            entry.status = target_status.value

            if target_status == PitfallStatus.SUPERSEDED:
                entry.is_algorithm_level = True
            elif target_status == PitfallStatus.REJECTED:
                entry.is_algorithm_level = False

            if superseded_by:
                entry.superseded_by = superseded_by
                entry.alias_of = None
            if alias_of:
                entry.alias_of = alias_of
                entry.superseded_by = None

            entry.updated_at = now
            changed += 1
        if changed:
            self.recalc_counters()
        return changed

    # ── dedup ───────────────────────────────────────────────────────────────

    def dedup(self) -> int:
        """Deduplicate entries across all statuses.

        Two entries are considered duplicates if:
        - Same title (case-insensitive, whitespace normalized)
        - Same root_cause (first 80 chars match)

        The lower-p_id entry wins; higher becomes alias_of the lower.
        Returns number of duplicates resolved.
        """
        resolved = 0
        entries_by_id = {e.p_id: e for e in self.entries}

        for i in range(len(self.entries)):
            a = self.entries[i]
            if not a.p_id:  # already removed
                continue
            for j in range(i + 1, len(self.entries)):
                b = self.entries[j]
                if not b.p_id:
                    continue
                if self._is_dup(a, b):
                    # lower-p_id wins
                    winner, loser = (a, b) if int(a.p_id[1:]) < int(b.p_id[1:]) else (b, a)
                    winner.status = PitfallStatus.CURRENT.value
                    winner.superseded_by = None
                    winner.is_algorithm_level = True
                    loser.status = PitfallStatus.SUPERSEDED.value
                    loser.alias_of = winner.p_id
                    loser.superseded_by = None
                    loser.is_algorithm_level = True
                    resolved += 1
                    if resolved > 1000:  # safety
                        raise RuntimeError("Too many duplicates; possible infinite loop")

        if resolved:
            self.recalc_counters()
        return resolved

    def _is_dup(self, a: PitfallEntry, b: PitfallEntry) -> bool:
        if a.p_id == b.p_id:
            return False
        # Already aliased
        if a.alias_of or b.alias_of:
            return False
        # Title match (normalized)
        ta = " ".join(a.title.strip().lower().split())
        tb = " ".join(b.title.strip().lower().split())
        if ta != tb:
            return False
        # Root cause match (first 80 chars normalized)
        ra = " ".join(a.root_cause[:80].strip().lower().split())
        rb = " ".join(b.root_cause[:80].strip().lower().split())
        return ra == rb

    # ── recalc ──────────────────────────────────────────────────────────────

    def recalc_counters(self) -> None:
        canon = 0
        alias = 0
        reject = 0
        for e in self.entries:
            if e.alias_of:
                alias += 1
            elif e.status == PitfallStatus.CURRENT.value:
                canon += 1
            elif e.status.startswith("rejected"):
                reject += 1
        self.total_entries = len(self.entries)
        self.canonical_count = canon
        self.alias_count = alias
        self.rejected_count = reject
        self.status_counts = {
            status.value: sum(1 for entry in self.entries if entry.status == status.value)
            for status in PitfallStatus
        }

    def export_catalog(
        self, path: os.PathLike[str] | str, *, generated_date: str
    ) -> None:
        """Atomically export the consumer-facing catalog from this canonical writer."""
        self.recalc_counters()
        lines = [
            f"# Pitfall Catalog ({generated_date})",
            "",
            "> 自动生成，由 mental_models 每日维护流程更新。",
            "> 结构化索引：`~/.hermes/mental-models/egomotion4d/pitfall_index.json`",
            "",
            "## 统计",
            f"- Current: {self.status_counts[PitfallStatus.CURRENT.value]}",
            f"- Candidate: {self.status_counts[PitfallStatus.CANDIDATE.value]}",
            f"- Superseded: {self.status_counts[PitfallStatus.SUPERSEDED.value]}",
            f"- Rejected (non-algorithmic): {self.status_counts[PitfallStatus.REJECTED.value]}",
            f"- Alias references: {self.alias_count}",
            "",
            "## Current Pitfalls",
            "",
        ]
        for entry in self.entries:
            if entry.status != PitfallStatus.CURRENT.value:
                continue
            lines.append(f"### {entry.p_id}: {entry.title}")
            if entry.trigger:
                lines.append(f"- **Trigger**: {entry.trigger}")
            if entry.root_cause:
                lines.append(f"- **Root Cause**: {entry.root_cause}")
            if entry.detail_locator:
                lines.append(f"- **Locator**: {entry.detail_locator}")
            if entry.tags:
                lines.append(f"- **Tags**: {', '.join(entry.tags)}")
            lines.append("")
        lines.append("## Non-current entries (hidden; use --all-pitfalls to view)")

        output = os.fspath(path)
        tmp = output + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        os.replace(tmp, output)

    # ── validation ──────────────────────────────────────────────────────────

    def validate(
        self,
        *,
        details_path: Optional[os.PathLike[str] | str] = None,
        require_provenance: bool = False,
    ) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        ids = {e.p_id for e in self.entries}
        valid_statuses = {s.value for s in PitfallStatus}
        detail_text = None
        if details_path is not None:
            path = os.fspath(details_path)
            try:
                with open(path, encoding="utf-8") as handle:
                    detail_text = handle.read()
            except OSError as exc:
                errors.append(f"detail file unreadable: {exc}")

        for e in self.entries:
            if e.status not in valid_statuses:
                errors.append(f"{e.p_id}: illegal status '{e.status}'")
            if e.alias_of and e.alias_of not in ids:
                errors.append(f"{e.p_id}: alias_of={e.alias_of} not in index")
            if e.alias_of == e.p_id:
                errors.append(f"{e.p_id}: self-referencing alias_of")
            if e.superseded_by and e.superseded_by not in ids:
                errors.append(f"{e.p_id}: superseded_by={e.superseded_by} not in index")
            if e.superseded_by == e.p_id:
                errors.append(f"{e.p_id}: self-referencing superseded_by")
            if e.alias_of and e.superseded_by:
                errors.append(f"{e.p_id}: has both alias_of and superseded_by")
            if e.status.startswith("rejected") and e.is_algorithm_level:
                errors.append(f"{e.p_id}: rejected but is_algorithm_level=True")
            if e.status == PitfallStatus.CURRENT.value and not e.is_algorithm_level:
                errors.append(f"{e.p_id}: current but is_algorithm_level=False")
            if e.status == PitfallStatus.CURRENT.value and not e.detail_locator:
                errors.append(f"{e.p_id}: missing detail_locator")
            elif detail_text is not None and e.status == PitfallStatus.CURRENT.value:
                if f"## {e.p_id}:" not in detail_text:
                    errors.append(f"{e.p_id}: locator does not resolve in detail file")
            if require_provenance and e.status == PitfallStatus.CURRENT.value:
                if not e.source_memory_id or not e.source_content_hash:
                    errors.append(f"{e.p_id}: missing provenance")
            if not e.tags:
                errors.append(f"{e.p_id}: empty tags")
            if not e.date:
                errors.append(f"{e.p_id}: missing date")

        # Duplicate p_id check
        seen = set()
        for e in self.entries:
            if e.p_id in seen:
                errors.append(f"Duplicate p_id: {e.p_id}")
            seen.add(e.p_id)

        # Count consistency
        canon = sum(1 for e in self.entries if e.status == PitfallStatus.CURRENT.value and not e.alias_of)
        alias = sum(1 for e in self.entries if e.alias_of)
        reject = sum(1 for e in self.entries if e.status.startswith("rejected"))
        if canon != self.canonical_count:
            errors.append(f"canonical_count mismatch: header={self.canonical_count} actual={canon}")
        if alias != self.alias_count:
            errors.append(f"alias_count mismatch: header={self.alias_count} actual={alias}")
        if reject != self.rejected_count:
            errors.append(f"rejected_count mismatch: header={self.rejected_count} actual={reject}")
        expected_status_counts = {
            status.value: sum(1 for entry in self.entries if entry.status == status.value)
            for status in PitfallStatus
        }
        if self.status_counts != expected_status_counts:
            errors.append(
                f"status_counts mismatch: header={self.status_counts} actual={expected_status_counts}"
            )

        return errors

    def reconcile_details(self, details_path: os.PathLike[str] | str) -> None:
        """Reconcile the index against the canonical Markdown detail records.

        A title mismatch at the same stable P-id means the index imported a
        colliding record; the canonical detail record wins. Consumable entries
        without a real detail section are demoted to candidates.
        """
        path = os.fspath(details_path)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        pattern = re.compile(
            r"(?ms)^## (P\d+):\s*(.*?)\n(.*?)(?=^---\s*$|^## P\d+:|\Z)"
        )
        sections = {}
        for match in pattern.finditer(text):
            p_id, title, body = match.groups()
            full = match.group(0).strip()

            def field(name: str) -> str:
                found = re.search(rf"(?m)^\*\*{name}\*\*:\s*(.*)$", body)
                return found.group(1).strip() if found else ""

            sections[p_id] = {
                "title": title.strip(),
                "trigger": field("坑"),
                "root_cause": field("实际"),
                "lesson": field("教训"),
                "tags": [tag.lstrip("#") for tag in field("标签").split() if tag],
                "date": field("日期"),
                "source": field("来源"),
                "hash": hashlib.sha256((full + "\n").encode()).hexdigest(),
            }

        for entry in self.entries:
            section = sections.get(entry.p_id)
            if section is None:
                if entry.status in {
                    PitfallStatus.CURRENT.value,
                    PitfallStatus.SUPERSEDED.value,
                }:
                    entry.status = PitfallStatus.CANDIDATE.value
                    entry.is_algorithm_level = True
                    entry.alias_of = None
                    entry.superseded_by = None
                entry.detail_locator = ""
                continue

            title_mismatch = " ".join(entry.title.lower().split()) != " ".join(
                section["title"].lower().split()
            )
            if entry.status == PitfallStatus.CURRENT.value or title_mismatch:
                entry.title = section["title"]
                entry.trigger = section["trigger"]
                entry.root_cause = section["root_cause"]
                entry.lesson = section["lesson"]
                entry.tags = sorted(set(section["tags"] or entry.tags))
                entry.date = section["date"] or entry.date
                entry.source = section["source"] or entry.source
                entry.status = PitfallStatus.CURRENT.value
                entry.is_algorithm_level = True
                entry.alias_of = None
                entry.superseded_by = None
                entry.detail_locator = f"pitfalls.md#{entry.p_id}"
                entry.source_memory_id = f"pitfalls.md:{entry.p_id}"
                entry.source_content_hash = section["hash"]

        self.next_p_id = StableIDAllocator.from_entries(self.entries).next_p_id
        self.recalc_counters()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _get(self, p_id: str) -> PitfallEntry:
        for e in self.entries:
            if e.p_id == p_id:
                return e
        raise KeyError(f"Entry not found: {p_id}")

    def find_by_id(self, p_id: str) -> Optional[PitfallEntry]:
        try:
            return self._get(p_id)
        except KeyError:
            return None


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Pitfall index writer")
    ap.add_argument("--load", default="/home/wyr/.hermes/mental-models/egomotion4d/pitfall_index.json",
                    help="Path to pitfall_index.json")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--recalc", action="store_true")
    ap.add_argument("--reconcile-details", metavar="PATH")
    ap.add_argument("--details", metavar="PATH",
                    help="Canonical pitfalls.md used for locator validation")
    ap.add_argument("--require-provenance", action="store_true",
                    help="Require source memory identity and content hash for current entries")
    ap.add_argument("--export-catalog", metavar="PATH")
    ap.add_argument("--generated-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--save", action="store_true", default=True)

    args = ap.parse_args()

    idx = PitfallIndex.load(args.load)

    if args.dedup:
        n = idx.dedup()
        print(f"Dedup resolved {n} duplicates")

    if args.recalc:
        idx.recalc_counters()
        print(f"Recounted: {idx.canonical_count}C / {idx.alias_count}A / {idx.rejected_count}R / {idx.total_entries}T")

    if args.reconcile_details:
        idx.reconcile_details(args.reconcile_details)
        print(
            f"Reconciled: {idx.canonical_count}C / {idx.alias_count}A / "
            f"{idx.rejected_count}R / {idx.total_entries}T"
        )

    if args.validate:
        errors = idx.validate(
            details_path=args.details,
            require_provenance=args.require_provenance,
        )
        if errors:
            print(f"VALIDATION FAILED ({len(errors)} errors):")
            for e in errors:
                print(f"  - {e}")
            raise SystemExit(1)
        print("Validation PASSED")

    if args.export_catalog:
        idx.export_catalog(args.export_catalog, generated_date=args.generated_date)
        print(f"Exported catalog: {args.export_catalog}")

    if args.save:
        sha = idx.save(args.load)
        print(f"Saved. SHA-256: {sha}")


if __name__ == "__main__":
    main()
