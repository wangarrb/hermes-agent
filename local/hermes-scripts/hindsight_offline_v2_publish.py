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
import math
import re
import shlex
import subprocess
import sys
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
except ModuleNotFoundError:
    alt_python = Path("/home/wyr/miniconda/envs/hindsight/bin/python")
    if __name__ == "__main__" and alt_python.exists() and Path(sys.executable).resolve() != alt_python.resolve():
        os.execv(str(alt_python), [str(alt_python), __file__, *sys.argv[1:]])
    raise

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_CARDS_ROOT = Path("/var/lib/hindsight-offline-reflect") / "v2_cards"
DEFAULT_BACKUP_DIR = Path("/var/lib/hindsight-offline-reflect") / "v2_publish_backups"
DEFAULT_DB_DSN = "dbname=hindsight user=hindsight host=/tmp port=5432"
DEFAULT_EMBEDDING_BASE_URL = os.environ.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
CANONICAL_PREFIX = "hermes-offline-canonical::"
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def slugify(text: str, *, prefix: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:80]
    return slug or f"{prefix}-{sha1_short(text, 8)}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cards(cards_root: Path, *, bank: str | None = None) -> list[dict[str, Any]]:
    # Freeze selection before any embedding/DB call. Legacy compatibility is
    # restricted to a complete, unmixed reducer manifest; never reconstruct a
    # missing/partial declaration from every file lying around the directory.
    from hindsight_incremental_merge import load_manifest_artifacts, publication_observation

    _manifest, cards, index_obs = load_manifest_artifacts(cards_root, bank=bank)
    if not cards:
        raise ValueError(f"no cards declared under {cards_root}")
    if index_obs is not None:
        # The old SQL publisher discards arbitrary legacy fields and renders text
        # with inferred suffixes. It cannot safely migrate opaque rows. Keep the
        # full candidate, but block this route until an authorized lossless DB
        # migration/publisher is supplied rather than quietly rewriting the seed.
        if any(obs.get("kind") == "opaque_legacy" for obs in index_obs):
            raise ValueError("opaque legacy seed requires a lossless publish path; existing V2 publisher is unsafe")
        topic_cards = {str(c.get("topic") or ""): c for c in cards if c.get("scope") == "topic"}
        global_card = next((c for c in cards if c.get("scope") == "global"), None)
        for c in cards:
            c["compact_card_observation_count"] = len(c.get("canonical_observations") or [])
            c["canonical_observations"] = []
        for obs in index_obs:
            topic = str(obs.get("topic") or "global")
            target = topic_cards.get(topic) or global_card
            if target is None:
                raise ValueError(f"index topic has no declared card: {topic}")
            target["canonical_observations"].append(
                publication_observation(obs) if _manifest.get("schema_version") else obs
            )
        cards = [c for c in cards if c.get("canonical_observations")]
        if not cards:
            raise ValueError("complete observation index is empty; refusing canonical replacement")
    return cards


def card_document_id(card: dict[str, Any]) -> str:
    scope = str(card.get("scope") or "topic")
    topic = str(card.get("topic") or "global")
    card_id = str(card.get("card_id") or "")
    suffix = re.sub(r"[^a-f0-9]", "", card_id.lower())[-12:] or sha1_short(topic)
    return f"{CANONICAL_PREFIX}{scope}::{slugify(topic, prefix='topic')}::{suffix}"


def semantic_memory_unit_id(bank: str, document_id: str, observation_id: str) -> str:
    """New incremental rows have identity independent of evidence/generation."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(
        ['hindsight-incremental-unit/1', bank, document_id, observation_id], ensure_ascii=False)))


def render_card_document(card: dict[str, Any], *, lossless: bool = False) -> str:
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
    if lossless:
        lines.extend(["", "## Complete Structured Observations", json.dumps(
            card.get("canonical_observations") or [], ensure_ascii=False, sort_keys=True, allow_nan=False)])
    return "\n".join(lines).strip() + "\n"


def observation_text(obs: dict[str, Any], card: dict[str, Any], *, lossless: bool = False) -> str:
    if lossless:
        return json.dumps(obs, ensure_ascii=False, sort_keys=True, allow_nan=False)
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


def ollama_encode(
    texts: list[str],
    *,
    model: str,
    base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    batch_size: int = 128,
    timeout: int = 300,
) -> list[list[float]]:
    """Encode through the active native OpenAI-compatible embedding service.

    Hindsight now runs as native systemd and its configured embedding backend is
    Ollama at /v1/embeddings. Fail closed on any transport, count, ordering, or
    dimensionality error; NULL embeddings are never a valid publish fallback.
    """
    if not texts:
        return []
    url = base_url.rstrip("/") + "/embeddings"
    output: list[list[float]] = []
    expected_dimension: int | None = None
    for start in range(0, len(texts), max(1, int(batch_size))):
        batch = texts[start:start + max(1, int(batch_size))]
        payload = json.dumps({"model": model, "input": batch}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"native embedding request failed at batch {start}: {exc}") from exc
        try:
            obj = json.loads(body)
            rows = list(obj.get("data") or [])
            indices: list[int] = []
            for row in rows:
                if not isinstance(row, dict) or type(row.get("index")) is not int:
                    raise ValueError("embedding row has a missing/invalid index")
                indices.append(row["index"])
            if sorted(indices) != list(range(len(batch))):
                raise ValueError(f"embedding indices must cover 0..{len(batch) - 1} exactly; got {indices!r}")
            rows_by_index = {row["index"]: row for row in rows}
            rows = [rows_by_index[index] for index in range(len(batch))]
            vectors = [list(row["embedding"]) for row in rows]
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(f"invalid native embedding response at batch {start}: {exc}") from exc
        if len(vectors) != len(batch):
            raise RuntimeError(f"native embedding count mismatch at batch {start}: got {len(vectors)}, expected {len(batch)}")
        for vector in vectors:
            if not vector:
                raise RuntimeError(f"native embedding returned an empty vector at batch {start}")
            dimension = len(vector)
            if expected_dimension is None:
                expected_dimension = dimension
            if dimension != expected_dimension:
                raise RuntimeError(f"native embedding dimension mismatch: got {dimension}, expected {expected_dimension}")
        output.extend(vectors)
    if len(output) != len(texts):
        raise RuntimeError(f"native embedding total mismatch: got {len(output)}, expected {len(texts)}")
    return output


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


def maybe_embeddings(
    texts: list[str],
    provider: str,
    model: str,
    *,
    base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    batch_size: int = 128,
) -> tuple[list[list[float] | None], str]:
    if provider == "none":
        return [None for _ in texts], "none"
    if not texts:
        return [], provider
    if provider in {"auto", "ollama"}:
        return ollama_encode(texts, model=model, base_url=base_url, batch_size=batch_size), "ollama"
    if provider == "docker":
        return docker_encode(texts, model=model), "docker"
    raise ValueError(f"unsupported embedding provider: {provider}")


def validate_embeddings(embeddings: list[list[float] | None], expected_count: int) -> int | None:
    """Require a complete, finite, fixed-width embedding matrix before DB writes."""
    if len(embeddings) != expected_count:
        raise RuntimeError(f"embedding count mismatch: got {len(embeddings)}, expected {expected_count}")
    if expected_count == 0:
        return None
    dimension: int | None = None
    for index, vector in enumerate(embeddings):
        if vector is None:
            raise RuntimeError(f"NULL embedding at observation index {index}; canonical publish requires embeddings")
        if not isinstance(vector, (list, tuple)) or not vector:
            raise RuntimeError(f"missing/empty embedding at observation index {index}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in vector):
            raise RuntimeError(f"invalid non-finite/nonnumeric embedding value at observation index {index}")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise RuntimeError(f"embedding dimension mismatch at observation index {index}: got {len(vector)}, expected {dimension}")
    return dimension


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


def backup_existing(conn, bank: str, backup_dir: Path, *, document_ids: list[str] | None = None, graph: bool = False) -> Path:
    closure = {}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if document_ids is None:
            # Preserve the existing CLI backup schema and default scope.
            cur.execute("""
                SELECT id, bank_id, original_text, metadata, created_at::text, updated_at::text, tags
                FROM documents WHERE bank_id=%s AND id LIKE %s ORDER BY id
            """, (bank, CANONICAL_PREFIX + "%"))
            docs = cur.fetchall()
            cur.execute("""
                SELECT id::text, bank_id, document_id, text, context, fact_type, metadata,
                       created_at::text, updated_at::text, tags, source_memory_ids::text[]
                FROM memory_units WHERE bank_id=%s AND document_id LIKE %s
                ORDER BY document_id, created_at, id
            """, (bank, CANONICAL_PREFIX + "%"))
            units = cur.fetchall()
        else:
            # Lossless rollback copy of the exact delta targets, including vectors.
            cur.execute("""
                SELECT to_jsonb(d) AS row FROM documents d
                WHERE bank_id=%s AND id = ANY(%s) ORDER BY id
            """, (bank, document_ids))
            docs = [row["row"] for row in cur.fetchall()]
            cur.execute("""
                SELECT (to_jsonb(m) - 'embedding') || jsonb_build_object('embedding', embedding::text) AS row
                FROM memory_units m WHERE bank_id=%s AND document_id = ANY(%s)
                ORDER BY document_id, created_at, id
            """, (bank, document_ids))
            units = [row["row"] for row in cur.fetchall()]
            if graph:
                unit_ids = [u['id'] for u in units]
                queries = {
                    'chunks': ("SELECT to_jsonb(c) AS row FROM chunks c WHERE bank_id=%s AND document_id=ANY(%s) ORDER BY chunk_id FOR UPDATE", (bank, document_ids)),
                    'unit_entities': ("SELECT to_jsonb(ue) AS row FROM unit_entities ue WHERE unit_id=ANY(%s::uuid[]) ORDER BY unit_id, entity_id FOR UPDATE", (unit_ids,)),
                    'memory_links': ("SELECT to_jsonb(ml) AS row FROM memory_links ml WHERE bank_id=%s AND (from_unit_id=ANY(%s::uuid[]) OR to_unit_id=ANY(%s::uuid[])) ORDER BY from_unit_id, to_unit_id FOR UPDATE", (bank, unit_ids, unit_ids)),
                    'source_references': ("SELECT (to_jsonb(m) - 'embedding') || jsonb_build_object('embedding', embedding::text) AS row FROM memory_units m WHERE bank_id=%s AND source_memory_ids && %s::uuid[] ORDER BY id", (bank, unit_ids)),
                }
                for name, (sql, params) in queries.items():
                    cur.execute(sql, params)
                    closure[name] = [r['row'] for r in cur.fetchall()]
                entity_ids = sorted({r['entity_id'] for r in closure['unit_entities']} |
                    {r['entity_id'] for r in closure['memory_links'] if r.get('entity_id')})
                cur.execute("SELECT to_jsonb(e) AS row FROM entities e WHERE bank_id=%s AND id=ANY(%s::uuid[]) ORDER BY id", (bank, entity_ids))
                closure['entities'] = [r['row'] for r in cur.fetchall()]
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "bank": bank, "document_count": len(docs), "unit_count": len(units), "documents": docs, "memory_units": units}
    if graph:
        payload.update(closure, rollback_policy='in_place_no_cascades; restore existing rows and add-only associations by identity')
    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir = backup_dir / f"canonical-backup-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    run_dir.mkdir(mode=0o700)
    os.chmod(run_dir, 0o700)
    path = run_dir / "backup.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            stream.write("\n")
    except BaseException:
        # fdopen owns/closes fd after entering its context; close only if the
        # context was never entered (e.g. an exceptional fdopen setup failure).
        try:
            os.close(fd)
        except OSError:
            pass
        raise
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


def _upsert_entities(cur, bank, names, rebuild_id, *, incremental=False):
    identities = {}
    for name in names:
        if incremental:
            # Shared entity metadata/counts stay intact; links are add-only.
            cur.execute("""INSERT INTO entities (canonical_name, bank_id, metadata, mention_count)
                VALUES (%s,%s,%s::jsonb,1) ON CONFLICT (bank_id, lower(canonical_name)) DO NOTHING
            """, (name, bank, json.dumps({"source": "offline_hindsight_v2"})))
            cur.execute("SELECT id::text FROM entities WHERE bank_id=%s AND lower(canonical_name)=lower(%s)", (bank, name))
        else:
            cur.execute("""
                INSERT INTO entities (canonical_name, bank_id, metadata, mention_count)
                VALUES (%s, %s, %s::jsonb, 1)
                ON CONFLICT (bank_id, lower(canonical_name)) DO UPDATE
                SET mention_count = entities.mention_count + 1,
                    last_seen = now(), metadata = entities.metadata || EXCLUDED.metadata
                RETURNING id::text
            """, (name, bank, json.dumps({"source": "offline_hindsight_v2", "rebuild_id": rebuild_id}, ensure_ascii=False)))
        identities[name] = cur.fetchone()[0]
    return identities


def publish(
    cards: list[dict[str, Any]],
    *,
    dsn: str,
    bank: str,
    mode: str,
    replace: bool,
    embedding_provider: str,
    embedding_model: str,
    embedding_base_url: str,
    embedding_batch_size: int,
    backup_dir: Path,
    generation_sha256: str | None = None,
    lossless: bool = False,
    backup_document_ids: list[str] | None = None,
    required_embedding_dimension: int | None = None,
    connect_fn=None,
    embedding_fn=None,
    execute_values_fn=None,
    prewrite_check=None,
    precommit_check=None,
    incremental: bool = False,
    incremental_unit_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    if incremental and (replace or not lossless or not generation_sha256 or backup_document_ids is None
            or prewrite_check is None or precommit_check is None):
        raise ValueError('incremental publish requires exact fenced lossless generation')
    sql_values = execute_values_fn or execute_values
    now = datetime.now(timezone.utc)
    rebuild_id = f"offline-v2-rebuild::{now.strftime('%Y%m%d-%H%M%S')}::{sha1_short(json.dumps([c.get('card_id') for c in cards], ensure_ascii=False), 8)}"
    docs = []
    observations = []
    for card in cards:
        doc_id = card_document_id(card)
        if incremental and (not doc_id.startswith(CANONICAL_PREFIX + 'incremental::')
                or len(card.get('canonical_observations') or []) != 1):
            raise ValueError('incremental publish requires one semantic node per safe document')
        content = render_card_document(card, lossless=True) if lossless else render_card_document(card)
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
        if generation_sha256 is not None:
            metadata["generation_sha256"] = generation_sha256
        docs.append({"id": doc_id, "card": card, "content": content, "metadata": metadata, "chunk_id": f"{doc_id}::chunk::0000"})
        for idx, obs in enumerate(card.get("canonical_observations") or []):
            text = observation_text(obs, card, lossless=True) if lossless else observation_text(obs, card)
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
    if incremental:
        if sorted(backup_document_ids) != sorted(d['id'] for d in docs):
            raise ValueError('incremental backup scope differs from exact targets')
        if set(incremental_unit_ids or {}) - {d['id'] for d in docs}:
            raise ValueError('incremental unit identity outside target scope')
    if mode == "dry-run":
        return report

    texts = [o["text"] for o in observations]
    embeddings, used_provider = (embedding_fn or maybe_embeddings)(
        texts,
        embedding_provider,
        embedding_model,
        base_url=embedding_base_url,
        batch_size=embedding_batch_size,
    )
    report["embedding_provider_used"] = used_provider
    embedding_dimension = validate_embeddings(embeddings, len(observations))
    if required_embedding_dimension is not None and embedding_dimension != required_embedding_dimension:
        raise RuntimeError(f"embedding dimension mismatch: required {required_embedding_dimension}, got {embedding_dimension}")
    if embedding_dimension is not None:
        report["embedding_dimension"] = embedding_dimension

    conn = (connect_fn or connect)(dsn)
    try:
        conn.autocommit = False
        if incremental:
            conn.set_session(isolation_level='READ COMMITTED', readonly=False, autocommit=False)
        if prewrite_check is not None:
            prewrite_check(conn)
        backup_path = backup_existing(conn, bank, backup_dir, document_ids=backup_document_ids, graph=incremental) if backup_document_ids is not None else backup_existing(conn, bank, backup_dir)
        report["backup_path"] = str(backup_path)
        with conn.cursor() as cur:
            if incremental:
                pass  # Existing documents/chunks/units are never deleted.
            elif replace:
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
            doc_sql = """
                INSERT INTO documents (id, bank_id, original_text, content_hash, metadata, retain_params, tags)
                VALUES %s
            """
            if incremental:
                old_rows = [r for r in doc_rows if r[0] in (incremental_unit_ids or {})]
                new_rows = [r for r in doc_rows if r[0] not in (incremental_unit_ids or {})]
                if old_rows:
                    sql_values(cur, doc_sql + """ ON CONFLICT (id, bank_id) DO UPDATE SET
                        original_text=EXCLUDED.original_text, content_hash=EXCLUDED.content_hash,
                        metadata=documents.metadata || EXCLUDED.metadata, updated_at=now()
                    """, old_rows, template="(%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)")
                if new_rows:
                    # Absent-target conflicts fail; never overwrite another writer.
                    sql_values(cur, doc_sql, new_rows, template="(%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)")
            else:
                sql_values(cur, doc_sql, doc_rows, template="(%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)")
            sql_values(cur, """
                INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text, content_hash)
                VALUES %s
            """ + (" ON CONFLICT (chunk_id) DO UPDATE SET chunk_text=EXCLUDED.chunk_text, content_hash=EXCLUDED.content_hash" if incremental else ""), chunk_rows)
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
                if lossless:
                    from hindsight_incremental_merge import observation_content_hash
                    metadata.update(memory_fact_metadata({
                        "structured_observation": obs,
                        "observation_content_sha256": observation_content_hash(obs),
                        "generation_sha256": generation_sha256,
                    }))
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
            unit_sql = """
                INSERT INTO memory_units (
                    bank_id, text, embedding, event_date, occurred_start, occurred_end, mentioned_at,
                    context, fact_type, metadata, chunk_id, document_id, tags,
                    observation_scopes, text_signals, source_memory_ids
                ) VALUES %s
            """
            template = "(%s,%s,NULLIF(%s, '')::vector,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s,%s::uuid[])"
            if incremental:
                unit_ids = [(incremental_unit_ids or {}).get(o['doc_id']) or semantic_memory_unit_id(bank, o['doc_id'], o['obs']['id']) for o in observations]
                unit_rows = [(uid, *r) for uid, r in zip(unit_ids, unit_rows)]
                unit_sql = unit_sql.replace('bank_id, text, embedding', 'id, bank_id, text, embedding', 1)
                template = '(%s::uuid,' + template[1:]
                existing = set((incremental_unit_ids or {}).values())
                for updating in (False, True):
                    group = [r for r in unit_rows if (r[0] in existing) == updating]
                    if not group:
                        continue
                    suffix = """ ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding,
                        context=EXCLUDED.context, metadata=memory_units.metadata || EXCLUDED.metadata,
                        tags=ARRAY(SELECT DISTINCT unnest(memory_units.tags || EXCLUDED.tags)),
                        observation_scopes=EXCLUDED.observation_scopes, text_signals=EXCLUDED.text_signals,
                        source_memory_ids=ARRAY(SELECT DISTINCT unnest(COALESCE(memory_units.source_memory_ids, '{}'::uuid[]) || EXCLUDED.source_memory_ids)),
                        updated_at=now()
                        WHERE memory_units.bank_id=EXCLUDED.bank_id AND memory_units.document_id=EXCLUDED.document_id
                    """ if updating else ''
                    returned = sql_values(cur, unit_sql + suffix + ' RETURNING id::text', group, template=template, fetch=True)
                    if sorted(str(r[0]) for r in returned) != sorted(r[0] for r in group):
                        raise ValueError('incremental memory identity CAS failed')
            else:
                unit_return = sql_values(cur, unit_sql + ' RETURNING id::text', unit_rows, template=template, fetch=True)
                unit_ids = [r[0] for r in unit_return]
            report["inserted_observations"] = len(unit_ids)

            # Upsert normalized entities and link them to observation units.
            entity_names = sorted({e for o in observations for e in o["entities"] if e})
            entity_id_by_name = _upsert_entities(cur, bank, entity_names, rebuild_id, incremental=incremental)
            link_rows = []
            for uid, o in zip(unit_ids, observations):
                for e in o["entities"]:
                    eid = entity_id_by_name.get(e)
                    if eid:
                        link_rows.append((uid, eid))
            if link_rows:
                sql_values(cur, """
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
                sql_values(cur, """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, bank_id)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, semantic_rows, template="(%s::uuid,%s::uuid,%s,%s,%s)")
                report["semantic_links"] = len(semantic_rows)
        if precommit_check is not None:
            precommit_check(conn)
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
    ap.add_argument("--embedding-provider", choices=["ollama", "auto", "docker", "none"], default="ollama")
    ap.add_argument("--embedding-model", default=os.environ.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL", "bge-m3"))
    ap.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    ap.add_argument("--embedding-batch-size", type=int, default=128)
    ap.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cards_root = Path(args.cards_root).expanduser()
    cards = load_cards(cards_root, bank=args.bank)
    report = publish(
        cards,
        dsn=args.dsn,
        bank=args.bank,
        mode=args.mode,
        replace=args.replace,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
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
