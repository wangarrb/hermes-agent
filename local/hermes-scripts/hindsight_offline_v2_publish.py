#!/usr/bin/env python3
"""Publish Offline Hindsight v2 canonical cards into the main Hindsight DB.

Default route is deterministic and LLM-free: cards produced by
hindsight_offline_v2_reduce.py are inserted as `fact_type='observation'` memory
units with stable `hermes-offline-canonical::*` document IDs.

Safety:
- No raw SQLite retain is performed.
- No LLM calls are performed.
- Existing canonical docs are backed up before replacement.
- Only document IDs with prefix hermes-offline-canonical:: are replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
except ModuleNotFoundError:
    alt_python = Path("/home/wyr/miniconda/envs/hindsight/bin/python")
    if alt_python.exists() and Path(sys.executable).resolve() != alt_python.resolve():
        os.execv(str(alt_python), [str(alt_python), __file__, *sys.argv[1:]])
    raise

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_CARDS_ROOT = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_cards"
DEFAULT_BACKUP_DIR = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_publish_backups"
DEFAULT_DB_DSN = "dbname=hindsight user=hindsight host=/tmp port=5432"
CANONICAL_PREFIX = "hermes-offline-canonical::"
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def slugify(text: str, *, prefix: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:80]
    return slug or f"{prefix}-{sha1_short(text, 8)}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cards(cards_root: Path) -> list[dict[str, Any]]:
    manifest = cards_root / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"cards manifest not found: {manifest}")
    paths = sorted(list((cards_root / "topics").glob("*.json")) + list((cards_root / "global").glob("*.json")))
    cards = [load_json(p) for p in paths]
    if not cards:
        raise SystemExit(f"no cards found under {cards_root}")

    # Complete V2 publish uses the detailed observations_index when available,
    # not only the compact card top-N. Cards remain the document grouping layer;
    # all deduped observations become observation memory_units.
    index_path = cards_root / "observations_index.jsonl"
    if index_path.exists():
        index_obs: list[dict[str, Any]] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("insight"):
                index_obs.append(obj)
        if index_obs:
            topic_cards = {str(c.get("topic") or ""): c for c in cards if c.get("scope") == "topic"}
            global_card = next((c for c in cards if c.get("scope") == "global"), None) or cards[-1]
            for c in cards:
                c["compact_card_observation_count"] = len(c.get("canonical_observations") or [])
                c["canonical_observations"] = []
            for obs in index_obs:
                topic = str(obs.get("topic") or "global")
                target = topic_cards.get(topic, global_card)
                target.setdefault("canonical_observations", []).append(obs)
            # Drop empty cards caused by max-topic card selection when all details
            # were assigned elsewhere.
            cards = [c for c in cards if c.get("canonical_observations")]
    return cards


def card_document_id(card: dict[str, Any]) -> str:
    scope = str(card.get("scope") or "topic")
    topic = str(card.get("topic") or "global")
    card_id = str(card.get("card_id") or "")
    suffix = re.sub(r"[^a-f0-9]", "", card_id.lower())[-12:] or sha1_short(topic)
    return f"{CANONICAL_PREFIX}{scope}::{slugify(topic, prefix='topic')}::{suffix}"


def render_card_document(card: dict[str, Any]) -> str:
    topic = str(card.get("topic") or "global")
    lines: list[str] = []
    lines.append(f"# Offline Hindsight v2 Canonical Card: {topic}")
    lines.append("")
    lines.append(f"card_id: {card.get('card_id')}")
    lines.append(f"schema_version: {card.get('schema_version')}")
    lines.append(f"scope: {card.get('scope')}")
    lines.append(f"generated_at: {card.get('generated_at')}")
    lines.append("")
    lines.append("## Executive Summary")
    for item in card.get("executive_summary") or []:
        lines.append(f"- {item}")
    lines.append("")
    aliases = card.get("entity_aliases") or []
    if aliases:
        lines.append("## Entity Aliases")
        for row in aliases:
            lines.append(f"- {row.get('canonical')}: {', '.join(map(str, row.get('aliases') or []))}")
        lines.append("")
    lines.append("## Canonical Observations")
    for obs in card.get("canonical_observations") or []:
        lines.append(f"- id: {obs.get('id')}")
        lines.append(f"  type: {obs.get('type')}")
        lines.append(f"  confidence: {obs.get('confidence')}")
        lines.append(f"  insight: {obs.get('insight')}")
        if obs.get("applicability"):
            lines.append(f"  applicability: {obs.get('applicability')}")
        if obs.get("evidence_ids"):
            lines.append(f"  evidence_ids: {', '.join(map(str, obs.get('evidence_ids')[:12]))}")
        if obs.get("tags"):
            lines.append(f"  tags: {', '.join(map(str, obs.get('tags')[:12]))}")
    lines.append("")
    lines.append("## Evidence Index")
    for e in card.get("evidence_index") or []:
        lines.append(f"- {e}")
    return "\n".join(lines).strip() + "\n"


def observation_text(obs: dict[str, Any], card: dict[str, Any]) -> str:
    parts = [str(obs.get("insight") or "").strip()]
    typ = str(obs.get("type") or "").strip()
    topic = str(obs.get("topic") or card.get("topic") or "").strip()
    applicability = str(obs.get("applicability") or "").strip()
    if typ:
        parts.append(f"type={typ}")
    if topic:
        parts.append(f"topic={topic}")
    if applicability:
        parts.append(f"applicability={applicability}")
    return " | ".join(p for p in parts if p)


def observation_tags(obs: dict[str, Any], card: dict[str, Any]) -> list[str]:
    tags = ["offline-v2", "canonical", "observation"]
    for value in [card.get("scope"), card.get("topic"), obs.get("type"), obs.get("topic")]:
        if value:
            tags.append(str(value))
    for t in obs.get("tags") or []:
        tags.append(str(t))
    seen = set()
    out = []
    for t in tags:
        t = t.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t[:100])
    return out[:24]


def observation_entities(obs: dict[str, Any], card: dict[str, Any]) -> list[str]:
    vals = []
    for value in [card.get("topic"), obs.get("topic")]:
        if value:
            vals.append(str(value))
    for t in obs.get("tags") or []:
        vals.append(str(t))
    aliases = card.get("entity_aliases") or []
    alias_by_canon = {str(a.get("canonical")): [str(x) for x in (a.get("aliases") or [])] for a in aliases if a.get("canonical")}
    expanded = []
    for v in vals:
        expanded.append(v)
        expanded.extend(alias_by_canon.get(v, [])[:6])
    # Extract ASCII technical identifiers from the insight as weak entity hints.
    text = str(obs.get("insight") or "")
    expanded.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", text)[:12])
    seen = set()
    out = []
    for v in expanded:
        v = v.strip(" `,;:|[](){}\"'")
        if not v or len(v) < 2:
            continue
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v[:160])
    return out[:16]


def source_memory_uuid_array(obs: dict[str, Any]) -> list[str]:
    ids = []
    for e in obs.get("evidence_ids") or []:
        ids.extend(UUID_RE.findall(str(e)))
    seen = set()
    return [x for x in ids if not (x.lower() in seen or seen.add(x.lower()))][:40]


def docker_encode(texts: list[str], *, model: str, batch_size: int = 64) -> list[list[float]]:
    code = r'''
import json, os, sys, warnings
payload=json.loads(sys.stdin.read())
texts=payload["texts"]
model=payload.get("model") or os.environ.get("HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL") or "BAAI/bge-m3"
batch_size=int(payload.get("batch_size") or 64)
from sentence_transformers import SentenceTransformer
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    m=SentenceTransformer(model, device="cpu", model_kwargs={"low_cpu_mem_usage": False})
out=[]
for i in range(0, len(texts), batch_size):
    emb=m.encode(texts[i:i+batch_size], convert_to_numpy=True, show_progress_bar=False)
    out.extend(emb.tolist())
print(json.dumps({"dimension": len(out[0]) if out else 0, "embeddings": out}))
'''
    cmd = ["sg", "docker", "-c", "docker exec -i hindsight /app/api/.venv/bin/python3 -c " + shlex.quote(code)]
    payload = json.dumps({"texts": texts, "model": model, "batch_size": batch_size}, ensure_ascii=False)
    proc = subprocess.run(cmd, input=payload, text=True, capture_output=True, timeout=1200)
    if proc.returncode != 0:
        raise RuntimeError(f"docker embedding failed: rc={proc.returncode}\nSTDOUT={proc.stdout[-2000:]}\nSTDERR={proc.stderr[-4000:]}")
    obj = json.loads(proc.stdout)
    embeddings = obj.get("embeddings") or []
    if len(embeddings) != len(texts):
        raise RuntimeError(f"embedding count mismatch: got {len(embeddings)}, expected {len(texts)}")
    return embeddings


def maybe_embeddings(texts: list[str], provider: str, model: str) -> tuple[list[list[float] | None], str]:
    if provider == "none" or not texts:
        return [None for _ in texts], "none"
    if provider in {"auto", "docker"}:
        try:
            return docker_encode(texts, model=model), "docker"
        except Exception as e:
            if provider == "docker":
                raise
            print(f"WARN: docker embeddings unavailable, falling back to NULL embeddings: {e}", file=sys.stderr)
    return [None for _ in texts], "none"


def vector_literal(vec: list[float] | None) -> str | None:
    if vec is None:
        return None
    return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"


def memory_fact_metadata(raw: dict[str, Any]) -> dict[str, str]:
    """Hindsight API's MemoryFact model expects metadata values to be strings.

    Direct DB publish bypasses the normal API serializer, so list/dict/None
    values would later make recall fail with Pydantic validation errors. Keep
    structure by JSON-encoding complex values and omit None values.
    """
    out: dict[str, str] = {}
    for key, value in (raw or {}).items():
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            out[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[str(key)] = str(value)
    return out


def connect(dsn: str):
    return psycopg2.connect(dsn)


def backup_existing(conn, bank: str, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = backup_dir / f"canonical-backup-{ts}.json"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, bank_id, original_text, metadata, created_at::text, updated_at::text, tags
            FROM documents
            WHERE bank_id=%s AND id LIKE %s
            ORDER BY id
        """, (bank, CANONICAL_PREFIX + "%"))
        docs = cur.fetchall()
        cur.execute("""
            SELECT id::text, bank_id, document_id, text, context, fact_type, metadata,
                   created_at::text, updated_at::text, tags, source_memory_ids::text[]
            FROM memory_units
            WHERE bank_id=%s AND document_id LIKE %s
            ORDER BY document_id, created_at, id
        """, (bank, CANONICAL_PREFIX + "%"))
        units = cur.fetchall()
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "bank": bank, "document_count": len(docs), "unit_count": len(units), "documents": docs, "memory_units": units}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def filter_semantic_rows_to_existing_targets(
    semantic_rows: list[tuple[Any, Any, str, float, str]],
    *,
    existing_target_ids: set[str],
) -> dict[str, Any]:
    """Keep memory_links rows whose target memory unit exists.

    Some canonical observations intentionally keep dangling evidence IDs as
    non-blocking P2 lineage issues. `source_memory_ids` may store those IDs as
    metadata, but `memory_links.to_unit_id` has a foreign key and must only link
    to existing memory_units.
    """
    existing = {str(x).lower() for x in existing_target_ids}
    kept = []
    skipped = 0
    for row in semantic_rows:
        target_id = str(row[1]).lower()
        if target_id in existing:
            kept.append(row)
        else:
            skipped += 1
    return {"kept": kept, "skipped_missing_targets": skipped}


def publish(cards: list[dict[str, Any]], *, dsn: str, bank: str, mode: str, replace: bool, embedding_provider: str, embedding_model: str, backup_dir: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    rebuild_id = f"offline-v2-rebuild::{now.strftime('%Y%m%d-%H%M%S')}::{sha1_short(json.dumps([c.get('card_id') for c in cards], ensure_ascii=False), 8)}"
    docs = []
    observations = []
    for card in cards:
        doc_id = card_document_id(card)
        content = render_card_document(card)
        metadata = {
            "schema_version": "offline-hindsight-v2-canonical-publish/0.1",
            "source_card_id": card.get("card_id"),
            "scope": card.get("scope"),
            "topic": card.get("topic"),
            "generated_at": card.get("generated_at"),
            "published_at": now.isoformat(),
            "rebuild_id": rebuild_id,
            "observation_count": len(card.get("canonical_observations") or []),
            "entity_aliases": card.get("entity_aliases") or [],
            "source": "offline_hindsight_v2",
        }
        docs.append({"id": doc_id, "card": card, "content": content, "metadata": metadata, "chunk_id": f"{doc_id}::chunk::0000"})
        for idx, obs in enumerate(card.get("canonical_observations") or []):
            text = observation_text(obs, card)
            observations.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::chunk::0000",
                "card": card,
                "obs": obs,
                "idx": idx,
                "text": text,
                "context": f"Offline Hindsight v2 canonical observation; topic={card.get('topic')}; scope={card.get('scope')}; card_id={card.get('card_id')}",
                "tags": observation_tags(obs, card),
                "entities": observation_entities(obs, card),
                "source_memory_ids": source_memory_uuid_array(obs),
                "text_signals": " ".join(observation_entities(obs, card) + observation_tags(obs, card)),
            })
    report: dict[str, Any] = {
        "generated_at": now.isoformat(),
        "mode": mode,
        "bank": bank,
        "replace": replace,
        "rebuild_id": rebuild_id,
        "card_count": len(cards),
        "document_count": len(docs),
        "observation_count": len(observations),
        "embedding_provider_requested": embedding_provider,
        "embedding_provider_used": None,
        "backup_path": None,
        "deleted_canonical_documents": 0,
        "inserted_documents": 0,
        "inserted_observations": 0,
        "linked_unit_entities": 0,
        "semantic_links": 0,
        "skipped_semantic_links_missing_targets": 0,
    }
    if mode == "dry-run":
        return report

    texts = [o["text"] for o in observations]
    embeddings, used_provider = maybe_embeddings(texts, embedding_provider, embedding_model)
    report["embedding_provider_used"] = used_provider
    if embeddings and embeddings[0] is not None:
        report["embedding_dimension"] = len(embeddings[0])

    conn = connect(dsn)
    try:
        conn.autocommit = False
        backup_path = backup_existing(conn, bank, backup_dir)
        report["backup_path"] = str(backup_path)
        with conn.cursor() as cur:
            if replace:
                cur.execute("DELETE FROM documents WHERE bank_id=%s AND id LIKE %s", (bank, CANONICAL_PREFIX + "%"))
                report["deleted_canonical_documents"] = cur.rowcount
            else:
                cur.execute("DELETE FROM documents WHERE bank_id=%s AND id = ANY(%s)", (bank, [d["id"] for d in docs]))
                report["deleted_canonical_documents"] = cur.rowcount
            doc_rows = []
            chunk_rows = []
            for d in docs:
                content_hash = hashlib.sha256(d["content"].encode("utf-8")).hexdigest()
                tags = ["offline-v2", "canonical", str(d["metadata"].get("scope") or ""), str(d["metadata"].get("topic") or "")]
                tags = [t[:100] for t in tags if t]
                doc_rows.append((d["id"], bank, d["content"], content_hash, json.dumps(d["metadata"], ensure_ascii=False), json.dumps({"source": "offline_v2_publish", "rebuild_id": rebuild_id}, ensure_ascii=False), tags))
                chunk_rows.append((d["chunk_id"], d["id"], bank, 0, d["content"], content_hash))
            execute_values(cur, """
                INSERT INTO documents (id, bank_id, original_text, content_hash, metadata, retain_params, tags)
                VALUES %s
            """, doc_rows, template="(%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)")
            execute_values(cur, """
                INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text, content_hash)
                VALUES %s
            """, chunk_rows)
            report["inserted_documents"] = len(doc_rows)

            unit_rows = []
            for o, emb in zip(observations, embeddings):
                obs = o["obs"]
                metadata = memory_fact_metadata({
                    "schema_version": "offline-hindsight-v2-observation/0.1",
                    "observation_id": obs.get("id"),
                    "source_card_id": o["card"].get("card_id"),
                    "rebuild_id": rebuild_id,
                    "type": obs.get("type"),
                    "topic": obs.get("topic") or o["card"].get("topic"),
                    "confidence": obs.get("confidence"),
                    "applicability": obs.get("applicability"),
                    "evidence_ids": obs.get("evidence_ids") or [],
                    "source_documents": obs.get("source_documents") or [],
                    "supersedes": obs.get("supersedes") or [],
                    "valid_from": obs.get("valid_from"),
                    "valid_until": obs.get("valid_until"),
                    "source": "offline_hindsight_v2",
                })
                unit_rows.append((
                    bank,
                    o["text"],
                    vector_literal(emb),
                    now,
                    None,
                    None,
                    now,
                    o["context"],
                    "observation",
                    json.dumps(metadata, ensure_ascii=False),
                    o["chunk_id"],
                    o["doc_id"],
                    o["tags"],
                    json.dumps({"topic": o["card"].get("topic"), "type": obs.get("type"), "applicability": obs.get("applicability")}, ensure_ascii=False),
                    o["text_signals"],
                    [str(x) for x in o["source_memory_ids"]],
                ))
            # vector literal is nullable; cast text column to vector only when non-null.
            unit_return = execute_values(cur, """
                INSERT INTO memory_units (
                    bank_id, text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                    context, fact_type, metadata, chunk_id, document_id, tags,
                    observation_scopes, text_signals, source_memory_ids
                ) VALUES %s
                RETURNING id::text
            """, unit_rows, template="(%s,%s,NULLIF(%s, '')::vector,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s,%s::uuid[])", fetch=True)
            unit_ids = [r[0] for r in unit_return]
            report["inserted_observations"] = len(unit_ids)

            # Upsert normalized entities and link them to observation units.
            entity_names = sorted({e for o in observations for e in o["entities"] if e})
            entity_id_by_name: dict[str, str] = {}
            for name in entity_names:
                cur.execute("""
                    INSERT INTO entities (canonical_name, bank_id, metadata, mention_count)
                    VALUES (%s, %s, %s::jsonb, 1)
                    ON CONFLICT (bank_id, lower(canonical_name)) DO UPDATE
                    SET mention_count = entities.mention_count + 1,
                        last_seen = now(),
                        metadata = entities.metadata || EXCLUDED.metadata
                    RETURNING id::text
                """, (name, bank, json.dumps({"source": "offline_hindsight_v2", "rebuild_id": rebuild_id}, ensure_ascii=False)))
                entity_id_by_name[name] = cur.fetchone()[0]
            link_rows = []
            for uid, o in zip(unit_ids, observations):
                for e in o["entities"]:
                    eid = entity_id_by_name.get(e)
                    if eid:
                        link_rows.append((uid, eid))
            if link_rows:
                execute_values(cur, """
                    INSERT INTO unit_entities (unit_id, entity_id)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, link_rows, template="(%s::uuid,%s::uuid)")
                report["linked_unit_entities"] = len(link_rows)

            semantic_rows = []
            for uid, o in zip(unit_ids, observations):
                for sid in o["source_memory_ids"]:
                    if sid != uid:
                        semantic_rows.append((uid, sid, "semantic", 0.85, bank))
            if semantic_rows:
                candidate_targets = sorted({str(row[1]) for row in semantic_rows})
                cur.execute("""
                    SELECT id::text
                    FROM memory_units
                    WHERE bank_id=%s AND id = ANY(%s::uuid[])
                """, (bank, candidate_targets))
                existing_targets = {str(row[0]).lower() for row in cur.fetchall()}
                filtered = filter_semantic_rows_to_existing_targets(
                    semantic_rows,
                    existing_target_ids=existing_targets,
                )
                semantic_rows = filtered["kept"]
                report["skipped_semantic_links_missing_targets"] = filtered["skipped_missing_targets"]
            if semantic_rows:
                execute_values(cur, """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, bank_id)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, semantic_rows, template="(%s::uuid,%s::uuid,%s,%s,%s)")
                report["semantic_links"] = len(semantic_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish Offline Hindsight v2 canonical cards to Hindsight DB as observation units")
    ap.add_argument("--cards-root", default=str(DEFAULT_CARDS_ROOT))
    ap.add_argument("--bank", default="hermes")
    ap.add_argument("--dsn", default=os.environ.get("HINDSIGHT_DB_DSN", DEFAULT_DB_DSN))
    ap.add_argument("--mode", choices=["dry-run", "publish"], default="dry-run")
    ap.add_argument("--replace", dest="replace", action="store_true", default=True, help="Replace all existing hermes-offline-canonical::* docs before publish (default)")
    ap.add_argument("--no-replace", dest="replace", action="store_false", help="Replace only the currently generated canonical document IDs")
    ap.add_argument("--embedding-provider", choices=["auto", "docker", "none"], default="auto")
    ap.add_argument("--embedding-model", default=os.environ.get("HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL", "BAAI/bge-m3"))
    ap.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cards_root = Path(args.cards_root).expanduser()
    cards = load_cards(cards_root)
    report = publish(
        cards,
        dsn=args.dsn,
        bank=args.bank,
        mode=args.mode,
        replace=args.replace,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        backup_dir=Path(args.backup_dir).expanduser(),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Offline Hindsight v2 Publish")
        for k in ["mode", "bank", "rebuild_id", "card_count", "document_count", "observation_count", "embedding_provider_used", "backup_path", "deleted_canonical_documents", "inserted_documents", "inserted_observations", "linked_unit_entities", "semantic_links", "skipped_semantic_links_missing_targets"]:
            print(f"{k}: {report.get(k)}")


if __name__ == "__main__":
    main()
