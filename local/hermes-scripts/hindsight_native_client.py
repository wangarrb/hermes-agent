#!/usr/bin/env python3
"""Official-interface-first Hindsight client helpers.

This module is intentionally small and boring: it centralizes all normal
Hindsight access behind the public SDK/REST API. Direct PostgreSQL access should
live outside business logic and be used only for explicitly documented fallback
or forensic work when the public API cannot expose a required view.

No secrets are logged here. Destructive operations are guarded by explicit
confirm tokens and support dry-run previews.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

# Bypass HTTP proxy for localhost API calls.
# Unset ALL proxy env vars to prevent broken proxies (e.g. dead clash on :7890)
# from causing 502 Bad Gateway on Hindsight localhost calls.
# no_proxy/NO_PROXY alone is insufficient when the proxy itself returns errors.
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_proxy_var, None)
# Also ensure no_proxy covers localhost as a safety net for any re-set vars
_np = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
if "127.0.0.1" not in _np and "localhost" not in _np:
    os.environ["no_proxy"] = f"127.0.0.1,localhost,{_np}".rstrip(",")
    os.environ["NO_PROXY"] = os.environ["no_proxy"]

DEFAULT_API = "http://127.0.0.1:8888"
DEFAULT_BANK = "hermes"
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
DELETE_OPERATION_CONFIRM = "delete-hindsight-operation"
RETRY_OPERATION_CONFIRM = "retry-hindsight-operation"
DELETE_DOCUMENT_CONFIRM = "delete-hindsight-document"
CLEAR_OBSERVATIONS_CONFIRM = "clear-hindsight-observations"
IMPORT_BANK_CONFIRM = "import-hindsight-bank-template"
REPROCESS_DOCUMENT_CONFIRM = "reprocess-hindsight-document"
REGENERATE_ENTITY_CONFIRM = "regenerate-hindsight-entity"
REFRESH_MENTAL_MODEL_CONFIRM = "refresh-hindsight-mental-model"
RECOVER_CONSOLIDATION_CONFIRM = "recover-hindsight-consolidation"


class HindsightApiError(RuntimeError):
    """Raised when the public Hindsight API call fails."""


class HindsightUnsafeOperation(RuntimeError):
    """Raised when a destructive operation lacks explicit confirmation."""


Transport = Callable[[str, str], Any]


def clean_json_text(text: str) -> str:
    return CONTROL_CHARS.sub("", text or "")


def _page_items(data: dict[str, Any], preferred: str) -> list[dict[str, Any]]:
    items = data.get(preferred)
    if items is None:
        items = data.get("items")
    if items is None and preferred != "operations":
        items = data.get("operations")
    if items is None:
        items = []
    return [x for x in items if isinstance(x, dict)]


@dataclass
class HindsightNativeClient:
    """Small REST client for stable Hindsight high-level endpoints.

    A fake `transport` can be injected by tests. The transport receives
    `(method, path, payload=..., params=...)` and returns already-decoded JSON.
    """

    api: str = DEFAULT_API
    bank: str = DEFAULT_BANK
    timeout: int = 30
    transport: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        self.api = (self.api or DEFAULT_API).rstrip("/")
        self.bank = self.bank or DEFAULT_BANK

    def bank_path(self, suffix: str = "") -> str:
        suffix = suffix or ""
        if suffix and not suffix.startswith("/"):
            suffix = "/" + suffix
        return f"/v1/default/banks/{urllib.parse.quote(self.bank, safe='')}{suffix}"

    def request(self, method: str, path: str, *, payload: dict[str, Any] | None = None, params: dict[str, Any] | None = None, timeout: int | None = None) -> Any:
        method = method.upper()
        if not path.startswith("/"):
            path = "/" + path
        clean_params: dict[str, Any] = {}
        for key, value in (params or {}).items():
            if value is None:
                continue
            clean_params[key] = value
        if self.transport is not None:
            return self.transport(method, path, payload=payload, params=clean_params)

        query = ""
        if clean_params:
            query = "?" + urllib.parse.urlencode(clean_params, doseq=True)
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.api + path + query, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise HindsightApiError(f"{method} {path} failed HTTP {e.code}: {clean_json_text(body)[:500]}") from e
        except Exception as e:
            raise HindsightApiError(f"{method} {path} failed: {type(e).__name__}: {e}") from e
        raw = clean_json_text(raw).strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception as e:
            raise HindsightApiError(f"{method} {path} returned non-JSON: {raw[:500]}") from e

    # Health / config / stats -------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def stats(self) -> dict[str, Any]:
        return self.request("GET", self.bank_path("stats"))

    def get_config(self) -> dict[str, Any]:
        return self.request("GET", self.bank_path("config"))

    def patch_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        return self.request("PATCH", self.bank_path("config"), payload={"updates": updates})

    # Documents ---------------------------------------------------------------
    def list_documents(self, *, q: str | None = None, tags: list[str] | None = None, tags_match: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        params = {"q": q, "tags": tags, "tags_match": tags_match, "limit": limit, "offset": offset}
        return self.request("GET", self.bank_path("documents"), params=params)

    def iter_documents(self, *, q: str | None = None, tags: list[str] | None = None, tags_match: str | None = None, limit: int = 100, max_items: int | None = None) -> Iterator[dict[str, Any]]:
        offset = 0
        yielded = 0
        while True:
            page = self.list_documents(q=q, tags=tags, tags_match=tags_match, limit=limit, offset=offset)
            items = _page_items(page, "items")
            if not items:
                break
            for item in items:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            offset += len(items)
            total = page.get("total")
            if isinstance(total, int) and offset >= total:
                break
            if len(items) < limit:
                break

    def list_all_documents(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.iter_documents(**kwargs))

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self.request("GET", self.bank_path("documents/" + urllib.parse.quote(document_id, safe="")))

    def patch_document_tags(self, document_id: str, tags: list[str]) -> dict[str, Any]:
        return self.request("PATCH", self.bank_path("documents/" + urllib.parse.quote(document_id, safe="")), payload={"tags": tags})

    def delete_document(self, document_id: str, *, dry_run: bool = True, confirm: str | None = None) -> dict[str, Any]:
        if dry_run:
            return {"dry_run": True, "operation": "delete_document", "document_id": document_id, "required_confirm": DELETE_DOCUMENT_CONFIRM}
        if confirm != DELETE_DOCUMENT_CONFIRM:
            raise HindsightUnsafeOperation(f"delete_document requires confirm={DELETE_DOCUMENT_CONFIRM}")
        return self.request("DELETE", self.bank_path("documents/" + urllib.parse.quote(document_id, safe="")))

    # Memories ----------------------------------------------------------------
    def list_memories(self, *, type: str | None = None, search_query: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        params = {"type": type, "q": search_query, "limit": limit, "offset": offset}
        return self.request("GET", self.bank_path("memories/list"), params=params)

    def iter_memories(self, *, types: Iterable[str] | None = None, search_query: str | None = None, limit: int = 100, max_items: int | None = None) -> Iterator[dict[str, Any]]:
        type_list = list(types) if types is not None else [None]
        seen: set[str] = set()
        yielded = 0
        for typ in type_list:
            offset = 0
            while True:
                page = self.list_memories(type=typ, search_query=search_query, limit=limit, offset=offset)
                items = _page_items(page, "items")
                if not items:
                    break
                for item in items:
                    item_id = str(item.get("id") or "")
                    if item_id and item_id in seen:
                        continue
                    if item_id:
                        seen.add(item_id)
                    yield item
                    yielded += 1
                    if max_items is not None and yielded >= max_items:
                        return
                offset += len(items)
                total = page.get("total")
                if isinstance(total, int) and offset >= total:
                    break
                if len(items) < limit:
                    break

    def list_all_memories(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.iter_memories(**kwargs))

    def get_memory(self, memory_id: str) -> dict[str, Any]:
        return self.request("GET", self.bank_path("memories/" + urllib.parse.quote(memory_id, safe="")))

    def get_memory_history(self, memory_id: str) -> dict[str, Any]:
        return self.request("GET", self.bank_path("memories/" + urllib.parse.quote(memory_id, safe="") + "/history"))

    def recall(self, query: str, *, types: list[str] | None = None, limit: int | None = None, budget: str = "mid", max_tokens: int = 4096, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "types": types, "budget": budget, "max_tokens": max_tokens}
        if limit is not None:
            # Hindsight's recall schema is token-budget centric; current server still
            # accepts limit in practice. Keep it optional for compatibility.
            payload["limit"] = limit
        payload.update({k: v for k, v in extra.items() if v is not None})
        return self.request("POST", self.bank_path("memories/recall"), payload=payload, timeout=max(self.timeout, 45))

    def retain_items(self, items: list[dict[str, Any]], *, async_mode: bool = True) -> dict[str, Any]:
        """Submit MemoryItem payloads through the official retain endpoint."""
        payload = {"async": async_mode, "items": items}
        return self.request("POST", self.bank_path("memories"), payload=payload, timeout=max(self.timeout, 120))

    def retain_batch(self, items: list[dict[str, Any]], *, retain_async: bool = False, document_tags: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"items": items, "async": bool(retain_async)}
        if document_tags is not None:
            payload["document_tags"] = document_tags
        return self.request("POST", self.bank_path("memories"), payload=payload, timeout=max(self.timeout, 60))

    # Operations / official cleaning -----------------------------------------
    def list_operations(
        self,
        *,
        status: str | None = None,
        type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        exclude_parents: bool | None = None,
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            self.bank_path("operations"),
            params={"status": status, "type": type, "limit": limit, "offset": offset, "exclude_parents": exclude_parents},
        )

    def iter_operations(
        self,
        *,
        status: str | None = None,
        type: str | None = None,
        limit: int = 100,
        max_items: int | None = None,
        exclude_parents: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        offset = 0
        yielded = 0
        while True:
            page = self.list_operations(status=status, type=type, limit=limit, offset=offset, exclude_parents=exclude_parents)
            items = _page_items(page, "operations")
            if not items:
                break
            for item in items:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            offset += len(items)
            total = page.get("total")
            if isinstance(total, int) and offset >= total:
                break
            if len(items) < limit:
                break

    def get_operation(self, operation_id: str, *, include_payload: bool = False) -> dict[str, Any]:
        return self.request(
            "GET",
            self.bank_path("operations/" + urllib.parse.quote(operation_id, safe="")),
            params={"include_payload": include_payload},
        )

    def delete_operation(self, operation_id: str, *, dry_run: bool = True, confirm: str | None = None) -> dict[str, Any]:
        if dry_run:
            return {"dry_run": True, "operation": "delete_operation", "operation_id": operation_id, "required_confirm": DELETE_OPERATION_CONFIRM}
        if confirm != DELETE_OPERATION_CONFIRM:
            raise HindsightUnsafeOperation(f"delete_operation requires confirm={DELETE_OPERATION_CONFIRM}")
        return self.request("DELETE", self.bank_path("operations/" + urllib.parse.quote(operation_id, safe="")))

    def retry_operation(self, operation_id: str, *, dry_run: bool = True, confirm: str | None = None) -> dict[str, Any]:
        if dry_run:
            return {"dry_run": True, "operation": "retry_operation", "operation_id": operation_id, "required_confirm": RETRY_OPERATION_CONFIRM}
        if confirm != RETRY_OPERATION_CONFIRM:
            raise HindsightUnsafeOperation(f"retry_operation requires confirm={RETRY_OPERATION_CONFIRM}")
        return self.request("POST", self.bank_path("operations/" + urllib.parse.quote(operation_id, safe="") + "/retry"))

    # v0.6.1 observability / reversible movement / targeted repair -----------
    def memories_timeseries(self, *, period: str = "7d", time_field: str = "created_at") -> dict[str, Any]:
        return self.request("GET", self.bank_path("stats/memories-timeseries"), params={"period": period, "time_field": time_field})

    def audit_log_stats(self, *, period: str = "7d", action: str | None = None) -> dict[str, Any]:
        return self.request("GET", self.bank_path("audit-logs/stats"), params={"period": period, "action": action})

    def list_audit_logs(
        self,
        *,
        action: str | None = None,
        transport: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            self.bank_path("audit-logs"),
            params={"action": action, "transport": transport, "start_date": start_date, "end_date": end_date, "limit": limit, "offset": offset},
        )

    def export_bank_template(self) -> dict[str, Any]:
        return self.request("GET", self.bank_path("export"), timeout=max(self.timeout, 60))

    def bank_template_schema(self) -> dict[str, Any]:
        return self.request("GET", "/v1/bank-template-schema")

    def import_bank_template(self, template: dict[str, Any], *, dry_run: bool = True, confirm: str | None = None) -> dict[str, Any]:
        if not dry_run and confirm != IMPORT_BANK_CONFIRM:
            raise HindsightUnsafeOperation(f"import_bank_template requires confirm={IMPORT_BANK_CONFIRM}")
        return self.request("POST", self.bank_path("import"), payload=template, params={"dry_run": dry_run}, timeout=max(self.timeout, 120))

    def reprocess_document(self, document_id: str, *, dry_run: bool = True, confirm: str | None = None) -> dict[str, Any]:
        if dry_run:
            return {"dry_run": True, "operation": "reprocess_document", "document_id": document_id, "required_confirm": REPROCESS_DOCUMENT_CONFIRM}
        if confirm != REPROCESS_DOCUMENT_CONFIRM:
            raise HindsightUnsafeOperation(f"reprocess_document requires confirm={REPROCESS_DOCUMENT_CONFIRM}")
        return self.request("POST", self.bank_path("documents/" + urllib.parse.quote(document_id, safe="") + "/reprocess"), timeout=max(self.timeout, 120))

    def regenerate_entity(self, entity_id: str, *, dry_run: bool = True, confirm: str | None = None) -> dict[str, Any]:
        if dry_run:
            return {"dry_run": True, "operation": "regenerate_entity", "entity_id": entity_id, "required_confirm": REGENERATE_ENTITY_CONFIRM}
        if confirm != REGENERATE_ENTITY_CONFIRM:
            raise HindsightUnsafeOperation(f"regenerate_entity requires confirm={REGENERATE_ENTITY_CONFIRM}")
        return self.request("POST", self.bank_path("entities/" + urllib.parse.quote(entity_id, safe="") + "/regenerate"), timeout=max(self.timeout, 120))

    def refresh_mental_model(self, mental_model_id: str, *, dry_run: bool = True, confirm: str | None = None) -> dict[str, Any]:
        if dry_run:
            return {"dry_run": True, "operation": "refresh_mental_model", "mental_model_id": mental_model_id, "required_confirm": REFRESH_MENTAL_MODEL_CONFIRM}
        if confirm != REFRESH_MENTAL_MODEL_CONFIRM:
            raise HindsightUnsafeOperation(f"refresh_mental_model requires confirm={REFRESH_MENTAL_MODEL_CONFIRM}")
        return self.request("POST", self.bank_path("mental-models/" + urllib.parse.quote(mental_model_id, safe="") + "/refresh"), timeout=max(self.timeout, 120))

    def recover_consolidation(self, *, dry_run: bool = True, confirm: str | None = None) -> dict[str, Any]:
        if dry_run:
            return {"dry_run": True, "operation": "recover_consolidation", "required_confirm": RECOVER_CONSOLIDATION_CONFIRM}
        if confirm != RECOVER_CONSOLIDATION_CONFIRM:
            raise HindsightUnsafeOperation(f"recover_consolidation requires confirm={RECOVER_CONSOLIDATION_CONFIRM}")
        return self.request("POST", self.bank_path("consolidation/recover"), timeout=max(self.timeout, 120))

    # Other high-level endpoints ---------------------------------------------
    def list_entities(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        return self.request("GET", self.bank_path("entities"), params={"limit": limit, "offset": offset})

    def iter_entities(self, *, limit: int = 100, max_items: int | None = None) -> Iterator[dict[str, Any]]:
        offset = 0
        yielded = 0
        while True:
            page = self.list_entities(limit=limit, offset=offset)
            items = _page_items(page, "items")
            if not items:
                break
            for item in items:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            offset += len(items)
            total = page.get("total")
            if isinstance(total, int) and offset >= total:
                break
            if len(items) < limit:
                break

    def list_all_entities(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.iter_entities(**kwargs))

    def graph(self) -> dict[str, Any]:
        return self.request("GET", self.bank_path("graph"), timeout=max(self.timeout, 60))

    def trigger_consolidation(self) -> dict[str, Any]:
        return self.request("POST", self.bank_path("consolidate"), payload={})

    def reflect(self, query: str, *, response_schema: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = {"query": query, **{k: v for k, v in kwargs.items() if v is not None}}
        if response_schema is not None:
            payload["response_schema"] = response_schema
        return self.request("POST", self.bank_path("reflect"), payload=payload, timeout=max(self.timeout, 300))


def client_from_args(api: str | None = None, bank: str | None = None, timeout: int = 30) -> HindsightNativeClient:
    return HindsightNativeClient(api=api or DEFAULT_API, bank=bank or DEFAULT_BANK, timeout=timeout)
