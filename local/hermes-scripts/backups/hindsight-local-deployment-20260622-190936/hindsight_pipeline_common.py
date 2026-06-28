#!/usr/bin/env python3
"""Shared configuration helpers for the Hindsight memory pipeline.

The defaults are profile/home relative so the pipeline can be packaged as a
reusable skill instead of being tied to a specific home directory.  A JSON config can override
paths, runtime names, and consolidation tuning without touching .env.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "hindsight-memory-pipeline-config-v1"
DEFAULT_CONFIG_REL = "hindsight/pipeline_config.json"


def hermes_home_from_env() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()


def discover_default_psql() -> str:
    root = Path.home() / ".hindsight-docker" / "installation"
    candidates = sorted(root.glob("*/bin/psql"), key=lambda p: p.as_posix(), reverse=True)
    for p in candidates:
        if p.exists():
            return str(p)
    return "psql"


def default_config(hermes_home: str | Path | None = None) -> dict[str, Any]:
    home = Path(hermes_home).expanduser() if hermes_home else hermes_home_from_env()
    scripts_dir = home / "scripts"
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "hermes_home": str(home),
        "scripts_dir": str(scripts_dir),
        "api_url": os.environ.get("HINDSIGHT_API_URL", "http://127.0.0.1:8888"),
        "bank": os.environ.get("HINDSIGHT_BANK", "hermes"),
        "llm_profile": os.environ.get("HINDSIGHT_OFFLINE_LLM_PROFILE", "minimax"),
        "docker": {
            "container": os.environ.get("HINDSIGHT_DOCKER_CONTAINER", "hindsight"),
            "required": True,
            "tuning_container_path": "/home/hindsight/.hindsight-consolidation-tuning.json",
        },
        "psql": {
            "path": os.environ.get("HINDSIGHT_PSQL", discover_default_psql()),
            "required": False,
        },
        "paths": {
            "sessions_dir": str(home / "sessions"),
            "manifest_dir": str(home / "hindsight" / "session_ingest" / "manifests"),
            "approved_root": str(home / "hindsight" / "review_repair" / "approved"),
            "proposal_root": str(home / "hindsight" / "review_repair" / "proposals"),
            "review_root": str(home / "hindsight" / "review_repair" / "reviews"),
            "status_root": str(home / "hindsight" / "pipeline_runs"),
            "tuning_file": str(scripts_dir / "hindsight-consolidation-tuning-default-20x3.json"),
        },
        "consolidation": {
            "batch_size": int(os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_BATCH_SIZE", "20")),
            "llm_batch_size": int(os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_LLM_BATCH_SIZE", "20")),
            # Hindsight >=0.5.3 / 0.6.x uses the official per-round knob.
            # Keep the old *_MAX_MEMORIES_PER_JOB alias only for local legacy wrappers.
            "max_memories_per_round": int(os.environ.get(
                "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_ROUND",
                os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_JOB", "60"),
            )),
            # Not an upstream Hindsight API setting. This is the external/offline pipeline fanout budget.
            "parallel_batches": int(os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_PARALLEL_BATCHES", "3")),
            "recall_budget": os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_RECALL_BUDGET", "low"),
            "source_facts_max_tokens": int(os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS", "4096")),
            "source_facts_max_tokens_per_observation": int(os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION", "256")),
        },
        "review": {
            "require_llm_review": True,
            "require_human_approval": True,
            "proposal_review": {
                "max_llm_calls": int(os.environ.get("HINDSIGHT_PROPOSAL_REVIEW_MAX_LLM_CALLS", "10")),
                "confirm_token": "review-hindsight-proposals",
                "llm_model": os.environ.get("HINDSIGHT_PROPOSAL_REVIEW_LLM_MODEL", "MiniMax-M2.7"),
                "llm_base_url": os.environ.get("HINDSIGHT_PROPOSAL_REVIEW_LLM_BASE_URL", "https://api.minimaxi.com/v1"),
                "llm_api_key_env": os.environ.get("HINDSIGHT_PROPOSAL_REVIEW_LLM_API_KEY_ENV", "MINIMAX_API_KEY"),
            },
            "notify": {
                "enabled": False,
                "method": "hermes_cron_or_origin_report",
                "note": "Pipeline writes review packets and emits a notification block; when run by Hermes cron, include the notification text in the final delivered response.",
            },
        },
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def expand_value(value: Any, *, home: Path) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value.replace("~", str(Path.home())))
    if isinstance(value, dict):
        return {k: expand_value(v, home=home) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_value(v, home=home) for v in value]
    return value


def config_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    if os.environ.get("HINDSIGHT_PIPELINE_CONFIG"):
        return Path(os.environ["HINDSIGHT_PIPELINE_CONFIG"]).expanduser()
    return hermes_home_from_env() / DEFAULT_CONFIG_REL


def load_config(path: str | Path | None = None, *, require_exists: bool = False) -> dict[str, Any]:
    cfg_path = config_path(path)
    cfg = default_config()
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"pipeline config must be a JSON object: {cfg_path}")
        cfg = deep_merge(cfg, data)
    elif require_exists:
        raise FileNotFoundError(str(cfg_path))
    home = Path(str(cfg.get("hermes_home") or hermes_home_from_env())).expanduser()
    cfg = expand_value(cfg, home=home)
    cfg["config_path"] = str(cfg_path)
    return cfg


def write_default_config(path: str | Path | None = None, *, overwrite: bool = False) -> Path:
    p = config_path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_config()
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def path_from_config(cfg: dict[str, Any], key: str) -> Path:
    return Path(str((cfg.get("paths") or {})[key])).expanduser()


def script_path(cfg: dict[str, Any], script: str) -> str:
    return str(Path(str(cfg.get("scripts_dir") or "")).expanduser() / script)


def ensure_local_dirs(cfg: dict[str, Any]) -> None:
    for key in ["manifest_dir", "proposal_root", "review_root", "status_root"]:
        path_from_config(cfg, key).mkdir(parents=True, exist_ok=True)


def consolidation_tuning(cfg: dict[str, Any]) -> dict[str, Any]:
    c = cfg.get("consolidation") or {}
    max_per_round = int(c.get("max_memories_per_round", c.get("max_memories_per_job", 60)))
    return {
        "consolidation_batch_size": int(c.get("batch_size", 20)),
        "consolidation_llm_batch_size": int(c.get("llm_batch_size", 20)),
        "max_memories_per_round": max_per_round,
        # Legacy alias for pre-v0.6 local wrappers; do not treat as official upstream config.
        "max_memories_per_job": max_per_round,
        "parallel_batches": int(c.get("parallel_batches", 3)),
        "recall_budget": str(c.get("recall_budget", "low")),
        "source_facts_max_tokens": int(c.get("source_facts_max_tokens", 4096)),
        "source_facts_max_tokens_per_observation": int(c.get("source_facts_max_tokens_per_observation", 256)),
    }
