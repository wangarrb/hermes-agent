#!/usr/bin/env python3
"""Controlled Hindsight paid-LLM mode manager.

Current production ingestion route is session/json manifest retain via native
Hindsight APIs. The old SQLite day-topic import route is deprecated and blocked
by default because it breaks session boundaries and creates bad tag/scope
pollution (`hermes/sqlite/incremental`).

不会把 MiniMax key 写入脚本；运行时从 ~/.hermes/.env 读取。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_native_client import (
    DELETE_DOCUMENT_CONFIRM,
    DELETE_OPERATION_CONFIRM,
    HindsightNativeClient,
)

HOME = Path(os.environ.get("HOME", str(Path.home()))).expanduser()
HERMES_HOME = Path(os.environ.get("HERMES_HOME", HOME / ".hermes")).expanduser()
ENV_PATH = HERMES_HOME / ".env"
HINDSIGHT_CONFIG = HERMES_HOME / "hindsight" / "config.json"
IMPORT_SCRIPT = HERMES_HOME / "scripts" / "import_sqlite_to_hindsight.py"
OFFLINE_REFLECT_SCRIPT = HERMES_HOME / "scripts" / "offline_hindsight_reflect_consolidate.py"
SESSION_RETAIN_RUNNER = HERMES_HOME / "scripts" / "hindsight_session_retain_runner.py"
JSON_PARSER_PATCH_SCRIPT = HERMES_HOME / "scripts" / "patch_hindsight_minimax_json_parser.py"
CONSOLIDATION_BUDGET_PATCH_SCRIPT = HERMES_HOME / "scripts" / "patch_hindsight_native_consolidation_budget.py"
RETAIN_TEMPORAL_FK_PATCH_SCRIPT = HERMES_HOME / "scripts" / "patch_hindsight_retain_temporal_fk_guard.py"
CONSOLIDATOR_PARALLEL_PATCH_SCRIPT = HERMES_HOME / "scripts" / "patch_hindsight_consolidator_parallel.py"
RETAIN_CONFIRM = "retain-hindsight-session-manifest"
HALF_DOWNGRADE_CONFIRM = "halve-hindsight-consolidation-concurrency"
DEPRECATED_SQLITE_IMPORT_MESSAGE = (
    "SQLite day-topic import is deprecated/blocked. Use session/json native route instead: "
    "hindsight_session_manifest.py -> hindsight_session_retain_runner.py, or wrapper "
    "hindsight_minimax_import.py session-manifest-retain-llm."
)

API = "http://127.0.0.1:8888"
BANK = "hermes"
# Pin the default to the verified v0.6.1 image. Relying on :latest caused
# accidental rollback to v0.5.2 when wrapper-driven container recreates happened.
IMAGE = os.environ.get("HINDSIGHT_IMAGE", "ghcr.io/vectorize-io/hindsight:0.6.1")
PG0_INSTANCE = Path(
    os.environ.get(
        "HINDSIGHT_PG0_INSTANCE",
        str(HOME / ".hindsight-docker" / "instances" / "hindsight" / "instance.json"),
    )
)


def pg0_root() -> Path:
    # <home>/.hindsight-docker/instances/hindsight/instance.json -> <home>/.hindsight-docker
    return PG0_INSTANCE.resolve().parents[2]


def pg0_metadata() -> dict[str, Any]:
    return json.loads(PG0_INSTANCE.read_text(encoding="utf-8"))


def pg0_data_dir() -> Path:
    return PG0_INSTANCE.parent / "data"


def pg0_bin(name: str) -> Path:
    try:
        version = str(pg0_metadata().get("version") or "18.1.0")
    except Exception:
        version = "18.1.0"
    return pg0_root() / "installation" / version / "bin" / name


def ensure_pg0_running() -> None:
    """Start the host pg0 PostgreSQL instance that backs Hindsight if needed."""
    data_dir = pg0_data_dir()
    pg_ctl = pg0_bin("pg_ctl")
    if not data_dir.exists() or not pg_ctl.exists():
        print(f"WARNING: pg0 paths missing; data_dir={data_dir} pg_ctl={pg_ctl}")
        return
    status = subprocess.run([str(pg_ctl), "-D", str(data_dir), "status"], text=True, capture_output=True)
    if status.returncode == 0:
        return
    log_path = data_dir / "start.log"
    print(f"starting pg0 for Hindsight: {data_dir}")
    subprocess.run(
        [str(pg_ctl), "-D", str(data_dir), "-l", str(log_path), "-w", "-t", "60", "start"],
        check=True,
    )


def database_url() -> str:
    """Build the Hindsight DB URL from pg0 metadata without printing secrets."""
    override = os.environ.get("HINDSIGHT_API_DATABASE_URL")
    if override:
        return override
    try:
        cfg = pg0_metadata()
        user = quote(str(cfg.get("username") or "hindsight"), safe="")
        use_password = os.environ.get("HINDSIGHT_PG_USE_PASSWORD") == "1"
        password = str(cfg.get("password") or "") if use_password else ""
        auth = f"{user}:{quote(password, safe='')}@" if password else f"{user}@"
        host = os.environ.get("HINDSIGHT_PGHOST_TCP", "127.0.0.1")
        port = int(cfg.get("port") or 5432)
        db = quote(str(cfg.get("database") or "hindsight"), safe="")
        return f"postgresql://{auth}{host}:{port}/{db}"
    except Exception:
        return "postgresql://hindsight@127.0.0.1:5432/hindsight"


DATABASE_URL = database_url()
OLLAMA_BASE_URL = os.environ.get("HINDSIGHT_LOCAL_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL = os.environ.get("HINDSIGHT_LOCAL_OLLAMA_MODEL", "qwen3.5:9b-local")
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_MODEL = "MiniMax-M2.7"
# Precision-first default: keep Hindsight on the configured remote LLM instead
# of restoring to a cheap/local model after offline jobs.
# User-selected production default: MiniMax provider with MiniMax-M2.7.
DEFAULT_OFFLINE_LLM_PROFILE = "minimax"
DEFAULT_PAID_LLM_CONCURRENCY = 8


def paid_llm_concurrency() -> int:
    raw = os.environ.get("HINDSIGHT_OFFLINE_LLM_CONCURRENCY", str(DEFAULT_PAID_LLM_CONCURRENCY))
    try:
        # Cap at 8 so the retained .env value does not silently override the
        # user's precision-first default during scheduled/background runs.
        return min(8, max(1, int(raw)))
    except Exception:
        return DEFAULT_PAID_LLM_CONCURRENCY


def int_env(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return max(min_value, int(raw))
    except Exception:
        return default


BUILTIN_LLM_PROFILES: dict[str, dict[str, Any]] = {
    "minimax": {
        "label": "minimax",
        "hindsight_provider": "minimax",
        "model": MINIMAX_MODEL,
        "base_url": MINIMAX_BASE_URL,
        "api_key_envs": ["MINIMAX_CN_API_KEY"],
        "response_format": True,
    },
    "glm": {
        "label": "glm",
        "hindsight_provider": "openai",
        "model": "glm-5",
        "base_url": "https://coding.dashscope.aliyuncs.com/v1",
        "api_key_envs": ["BAILIAN_API_KEY", "DASHSCOPE_API_KEY", "GLM_API_KEY"],
        "response_format": True,
    },
    "deepseek": {
        "label": "deepseek",
        "hindsight_provider": "openai",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_envs": ["DEEPSEEK_API_KEY"],
        "response_format": True,
        "strict_schema": False,
    },
    "deepseek-v4-flash": {
        "label": "deepseek-v4-flash",
        "hindsight_provider": "openai",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_envs": ["DEEPSEEK_API_KEY"],
        "response_format": True,
        "strict_schema": False,
    },
    "deepseek-v4-pro": {
        "label": "deepseek-v4-pro",
        "hindsight_provider": "openai",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_envs": ["DEEPSEEK_API_KEY"],
        "response_format": True,
        "strict_schema": False,
    },
    "opencode-go-deepseek-v4-flash": {
        "label": "opencode-go-deepseek-v4-flash",
        "hindsight_provider": "openai",
        "model": "deepseek-v4-flash",
        "base_url": "https://opencode.ai/zen/go/v1",
        "api_key_envs": ["OPENCODE_GO_API_KEY"],
        "response_format": True,
        "strict_schema": False,
    },
}

PROFILE_ALIASES = {
    "bailian": "glm",
    "dashscope": "glm",
    "zai": "glm",
    "z.ai": "glm",
    "opencode-go": "opencode-go-deepseek-v4-flash",
}


def read_dotenv(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def env_or_dotenv(key: str, dotenv: dict[str, str] | None = None) -> str:
    if key in os.environ and os.environ[key]:
        return os.environ[key].strip()
    data = dotenv if dotenv is not None else read_dotenv()
    return str(data.get(key, "")).strip()


def select_api_key_env(profile: dict[str, Any], dotenv: dict[str, str]) -> str:
    envs = list(profile.get("api_key_envs") or [])
    if profile.get("api_key_env"):
        envs.insert(0, str(profile["api_key_env"]))
    for key_env in envs:
        value = env_or_dotenv(key_env, dotenv)
        if value and value not in {"***", "[REDACTED]"}:
            return key_env
    label = profile.get("label") or "llm"
    raise SystemExit(f"No usable API key found for profile={label}; tried {envs}")


def get_llm_profile(name: str | None = None) -> dict[str, Any]:
    dotenv = read_dotenv()
    requested = (
        name
        or os.environ.get("HINDSIGHT_OFFLINE_LLM_PROFILE")
        or dotenv.get("HINDSIGHT_OFFLINE_LLM_PROFILE")
        or DEFAULT_OFFLINE_LLM_PROFILE
    ).strip().lower()
    requested = PROFILE_ALIASES.get(requested, requested)

    if requested == "custom":
        profile = {
            "label": env_or_dotenv("HINDSIGHT_OFFLINE_LLM_LABEL", dotenv) or "custom",
            "hindsight_provider": env_or_dotenv("HINDSIGHT_OFFLINE_HINDSIGHT_PROVIDER", dotenv) or "openai",
            "model": env_or_dotenv("HINDSIGHT_OFFLINE_LLM_MODEL", dotenv),
            "base_url": env_or_dotenv("HINDSIGHT_OFFLINE_LLM_BASE_URL", dotenv),
            "api_key_envs": [env_or_dotenv("HINDSIGHT_OFFLINE_LLM_API_KEY_ENV", dotenv) or "HINDSIGHT_OFFLINE_LLM_API_KEY"],
            "response_format": (env_or_dotenv("HINDSIGHT_OFFLINE_RESPONSE_FORMAT", dotenv) or "true").lower() not in {"0", "false", "no"},
        }
    else:
        if requested not in BUILTIN_LLM_PROFILES:
            raise SystemExit(f"Unknown --llm-profile={requested!r}; available={sorted(BUILTIN_LLM_PROFILES)} plus custom")
        profile = dict(BUILTIN_LLM_PROFILES[requested])

    # Optional env/.env overrides let the scheduled job switch model/base_url without editing code.
    for env_key, field in [
        ("HINDSIGHT_OFFLINE_LLM_LABEL", "label"),
        ("HINDSIGHT_OFFLINE_HINDSIGHT_PROVIDER", "hindsight_provider"),
        ("HINDSIGHT_OFFLINE_LLM_MODEL", "model"),
        ("HINDSIGHT_OFFLINE_LLM_BASE_URL", "base_url"),
    ]:
        value = env_or_dotenv(env_key, dotenv)
        if value:
            profile[field] = value
    api_key_env_override = env_or_dotenv("HINDSIGHT_OFFLINE_LLM_API_KEY_ENV", dotenv)
    if api_key_env_override:
        profile["api_key_envs"] = [api_key_env_override]
    response_format_override = env_or_dotenv("HINDSIGHT_OFFLINE_RESPONSE_FORMAT", dotenv)
    if response_format_override:
        profile["response_format"] = response_format_override.lower() not in {"0", "false", "no"}

    if not profile.get("model") or not profile.get("base_url"):
        raise SystemExit(f"Incomplete LLM profile: {profile}")
    profile["api_key_env"] = select_api_key_env(profile, dotenv)
    profile["api_key"] = env_or_dotenv(profile["api_key_env"], dotenv)
    if "strict_schema" not in profile:
        profile["strict_schema"] = True
    return profile


def get_minimax_key() -> str:
    profile = get_llm_profile("minimax")
    return str(profile["api_key"])


def docker_shell(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run docker through sg docker because this login shell lacks active docker group."""
    proc = subprocess.run(["sg", "docker", "-c", command], text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker command failed: {command}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc

def arg_present(args: list[str], name: str) -> bool:
    """Return True for either '--name value' or '--name=value'."""
    return name in args or any(x.startswith(name + "=") for x in args)


def arg_value(args: list[str], name: str, default: str | None = None) -> str | None:
    """Read a CLI option from either '--name value' or '--name=value'."""
    prefix = name + "="
    for idx, item in enumerate(args):
        if item == name and idx + 1 < len(args):
            return args[idx + 1]
        if item.startswith(prefix):
            return item[len(prefix):]
    return default


def prepend_arg(args: list[str], name: str, value: str) -> list[str]:
    return [name, value] + args


def write_env_file(env: dict[str, str]) -> str:
    fd, path = tempfile.mkstemp(prefix="hindsight-env-", suffix=".env")
    os.close(fd)
    os.chmod(path, 0o600)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in env.items():
            f.write(f"{k}={v}\n")
    return path


def base_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "NODE_ENV": "production",
        "HF_HUB_VERBOSITY": "error",
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        # Hindsight runs as /home/hindsight inside the container. Keep all HF
        # libraries on the bind-mounted host cache instead of the container
        # writable layer, otherwise bge-m3 may be re-downloaded as *.incomplete.
        "HF_HOME": "/home/hindsight/.cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/home/hindsight/.cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/home/hindsight/.cache/huggingface/hub",
        # bge-m3 is already cached on the bind-mounted host cache. Force
        # HuggingFace/Transformers offline during API startup so optional
        # metadata probes (for example adapter_config.json / PEFT checks) do
        # not hit hf.co and crash the Hindsight lifespan with
        # "Cannot send a request, as the client has been closed".
        # Override to 0 only when intentionally re-downloading the embedding model.
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        # bge-m3 first-load/download can exceed the Hindsight image's default
        # 300s startup health window. Keep this high so start-all.sh does not
        # kill/restart the API while SentenceTransformer is still loading.
        "HINDSIGHT_API_STARTUP_WAIT_SECONDS": os.environ.get("HINDSIGHT_API_STARTUP_WAIT_SECONDS", "1200"),
        "TOKENIZERS_PARALLELISM": "false",
        # Do not force a global proxy inside the Hindsight container. Domestic
        # providers used here (DeepSeek / MiniMax / DashScope) are directly
        # reachable, while a stale 127.0.0.1:7890 proxy stalls LLM calls. If a
        # specific provider (for example opencode-go) needs a proxy, opt in via
        # HINDSIGHT_DOCKER_HTTP_PROXY rather than enabling it globally.
        **(
            {
                "HTTP_PROXY": os.environ["HINDSIGHT_DOCKER_HTTP_PROXY"],
                "HTTPS_PROXY": os.environ.get("HINDSIGHT_DOCKER_HTTPS_PROXY", os.environ["HINDSIGHT_DOCKER_HTTP_PROXY"]),
                "NO_PROXY": os.environ.get("HINDSIGHT_DOCKER_NO_PROXY", "127.0.0.1,localhost"),
            }
            if os.environ.get("HINDSIGHT_DOCKER_HTTP_PROXY")
            else {}
        ),
        "HINDSIGHT_ENABLE_API": "true",
        "HINDSIGHT_ENABLE_CP": "false",
        "HINDSIGHT_API_PORT": "8888",
        "HINDSIGHT_API_DATABASE_URL": DATABASE_URL,
        "HINDSIGHT_CP_DATAPLANE_API_URL": API,
        "HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP": "false",
        "HINDSIGHT_API_AUTO_RETAIN": "false",
        "HINDSIGHT_API_WORKER_ENABLED": "true",
        "HINDSIGHT_API_WORKER_MAX_SLOTS": "1",
        "HINDSIGHT_API_RETAIN_MAX_CONCURRENT": "1",
        "HINDSIGHT_API_LLM_MAX_CONCURRENT": "1",
        "HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT": "1",
        "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT": "1",
        "HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT": "1",
        "HINDSIGHT_API_LLM_TIMEOUT": "300",
        "HINDSIGHT_API_RETAIN_LLM_TIMEOUT": "300",
        "HINDSIGHT_API_CONSOLIDATION_LLM_TIMEOUT": "300",
        "HINDSIGHT_API_REFLECT_LLM_TIMEOUT": "300",
        # Paid-provider safety defaults. Hindsight has nested retry loops
        # (operation attempts + LLM retries); keep them low and use long
        # rate-limit backoff to avoid call explosions on 429/quota windows.
        "HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS": "300",
        "HINDSIGHT_API_LLM_MAX_RETRIES": "2",
        "HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES": "2",
        "HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES": "2",
        "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES": "1",
        "HINDSIGHT_API_CONSOLIDATION_MAX_ATTEMPTS": "1",
        # Native consolidation quality/cost knobs. Hindsight >=0.5.3 / 0.6.x
        # uses max_memories_per_round as the official cap; keep the old
        # max_memories_per_job env only for legacy local wrappers.
        "HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_ROUND",
            os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_JOB", "64"),
        ),
        "HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_JOB",
            os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_ROUND", "64"),
        ),
        # `consolidation_batch_size` is the fetch round size;
        # `consolidation_llm_batch_size` is facts per LLM call.
        # Default 64x8 verified 2026-05-14 (0 429s, ~$6 full drain, 8-way parallel).
        "HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_BATCH_SIZE", "64"
        ),
        "HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_LLM_BATCH_SIZE", "8"
        ),
        "HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_RECALL_BUDGET", "low"
        ),
        "HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS", "4096"
        ),
        "HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION", "256"
        ),
        # Not an upstream Hindsight knob; retained as a local/offline wrapper hint.
        # Default 8-way parallel since 2026-05-14 verification (0 429s, ~$6 full drain).
        "HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_PARALLEL_BATCHES", "8"
        ),
        "HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_RECALL_MAX_CONCURRENT", "60"
        ),
        "HINDSIGHT_API_DB_STATEMENT_TIMEOUT": os.environ.get("HINDSIGHT_API_DB_STATEMENT_TIMEOUT", "600"),
        "HINDSIGHT_API_LLM_MAX_BACKOFF": "30",
        "HINDSIGHT_API_RETAIN_LLM_MAX_BACKOFF": "30",
        "HINDSIGHT_API_REFLECT_LLM_MAX_BACKOFF": "30",
        "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_BACKOFF": "30",
        "HINDSIGHT_API_RETAIN_CHUNK_SIZE": os.environ.get("HINDSIGHT_NATIVE_RETAIN_CHUNK_SIZE", "8000"),
        "HINDSIGHT_API_RETAIN_EXTRACTION_MODE": os.environ.get("HINDSIGHT_NATIVE_RETAIN_EXTRACTION_MODE", "custom"),
        "HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS": os.environ.get(
            "HINDSIGHT_NATIVE_RETAIN_CUSTOM_INSTRUCTIONS",
            "ONLY extract durable user/project facts, decisions, results, preferences, stable environment facts. "
            "Skip tool logs, file listings, raw command output, process chatter, greetings. "
            "Max 3-5 facts per chunk."
        ),
        "HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS": os.environ.get("HINDSIGHT_NATIVE_RETAIN_EXTRACT_CAUSAL_LINKS", "false"),
        # Keep embeddings local. MiniMax embo-01 is not OpenAI-compatible for
        # Hindsight embeddings, and current Hindsight images do not support the
        # old "openai_compatible" embeddings provider name.
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "local",
        "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL": "BAAI/bge-m3",
        "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
        # Hindsight v0.6.x default is false; set explicitly to keep RSS bounded if FlashRank is enabled later.
        "HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA": "false",
        # 避免启动健康检查本身多打一笔 LLM 调用；真正可用性由导入任务验证。
        "HINDSIGHT_API_SKIP_LLM_VERIFICATION": "true",
    }


def ollama_env(*, model: str | None = None, base_url: str | None = None, consolidation_slots: int = 0) -> dict[str, str]:
    env = base_env()
    model = model or OLLAMA_MODEL
    base_url = base_url or OLLAMA_BASE_URL
    consolidation_slots = max(0, int(consolidation_slots))
    worker_slots = max(1, 1 + consolidation_slots)
    env.update({
        "HINDSIGHT_API_ENABLE_OBSERVATIONS": "false",
        "HINDSIGHT_API_WORKER_MAX_SLOTS": str(worker_slots),
        "HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS": str(consolidation_slots),
        "HINDSIGHT_API_LLM_PROVIDER": "ollama",
        "HINDSIGHT_API_LLM_MODEL": model,
        "HINDSIGHT_API_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_LLM_API_KEY": "ollama",
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER": "ollama",
        "HINDSIGHT_API_RETAIN_LLM_MODEL": model,
        "HINDSIGHT_API_RETAIN_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_RETAIN_LLM_API_KEY": "ollama",
        "HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER": "ollama",
        "HINDSIGHT_API_CONSOLIDATION_LLM_MODEL": model,
        "HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY": "ollama",
        "HINDSIGHT_API_REFLECT_LLM_PROVIDER": "ollama",
        "HINDSIGHT_API_REFLECT_LLM_MODEL": model,
        "HINDSIGHT_API_REFLECT_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_REFLECT_LLM_API_KEY": "ollama",
    })
    return env


def paid_llm_env(profile: dict[str, Any], *, enable_observations: bool = False) -> dict[str, str]:
    key = str(profile["api_key"])
    provider = str(profile["hindsight_provider"])
    model = str(profile["model"])
    base_url = str(profile["base_url"])
    concurrency = paid_llm_concurrency()
    consolidation_llm_concurrency = int_env(
        "HINDSIGHT_NATIVE_CONSOLIDATION_LLM_MAX_CONCURRENT",
        concurrency,
    )
    # WorkerPoller reserves consolidation slots out of worker_max_slots.
    # Keep retain slots at the paid-LLM concurrency and reserve explicit
    # consolidation slots when observations/consolidation are enabled.
    consolidation_slots = int(os.environ.get("HINDSIGHT_NATIVE_WORKER_CONSOLIDATION_MAX_SLOTS", "2" if enable_observations else "0"))
    worker_slots = concurrency + consolidation_slots
    env = base_env()
    obs = "true" if enable_observations else "false"
    env.update({
        "HINDSIGHT_API_ENABLE_OBSERVATIONS": obs,
        "HINDSIGHT_API_WORKER_MAX_SLOTS": str(worker_slots),
        "HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS": str(consolidation_slots),
        "HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_ROUND",
            os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_JOB", "64" if enable_observations else "0"),
        ),
        "HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB": os.environ.get(
            "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_JOB",
            os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_ROUND", "64" if enable_observations else "0"),
        ),
        "HINDSIGHT_API_RETAIN_MAX_CONCURRENT": str(concurrency),
        "HINDSIGHT_API_LLM_MAX_CONCURRENT": str(concurrency),
        "HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT": str(concurrency),
        "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT": str(consolidation_llm_concurrency),
        "HINDSIGHT_API_REFLECT_LLM_MAX_CONCURRENT": str(concurrency),
        "HINDSIGHT_API_LLM_PROVIDER": provider,
        "HINDSIGHT_API_LLM_MODEL": model,
        "HINDSIGHT_API_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_LLM_API_KEY": key,
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER": provider,
        "HINDSIGHT_API_RETAIN_LLM_MODEL": model,
        "HINDSIGHT_API_RETAIN_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_RETAIN_LLM_API_KEY": key,
        "HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER": provider,
        "HINDSIGHT_API_CONSOLIDATION_LLM_MODEL": model,
        "HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY": key,
        "HINDSIGHT_API_REFLECT_LLM_PROVIDER": provider,
        "HINDSIGHT_API_REFLECT_LLM_MODEL": model,
        "HINDSIGHT_API_REFLECT_LLM_BASE_URL": base_url,
        "HINDSIGHT_API_REFLECT_LLM_API_KEY": key,
    })
    return env


def minimax_env(*, enable_observations: bool = False) -> dict[str, str]:
    return paid_llm_env(get_llm_profile("minimax"), enable_observations=enable_observations)


def wait_health(timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{API}/health", timeout=5)
            last = r.text[:300]
            if r.ok:
                print(f"health OK: {last}")
                return
        except Exception as e:
            last = repr(e)
        time.sleep(3)
    raise RuntimeError(f"Hindsight did not become healthy within {timeout_s}s; last={last}")


def ensure_bank_exists(bank: str) -> None:
    try:
        r = requests.get(f"{API}/v1/default/banks", timeout=20)
        r.raise_for_status()
        banks = r.json().get("banks") or []
        if any(str(b.get("bank_id") or "") == bank for b in banks if isinstance(b, dict)):
            return
    except Exception as e:
        print(f"WARNING: could not list banks before creating {bank}: {e}")
    r = requests.put(f"{API}/v1/default/banks/{bank}", json={"name": bank}, timeout=20)
    r.raise_for_status()
    print(f"created/verified bank: {bank}")


def patch_bank_config(*, enable_observations: bool, bank: str = BANK) -> None:
    current_chunk_size = int_env("HINDSIGHT_NATIVE_RETAIN_CHUNK_SIZE", 8000)
    retain_extraction_mode = os.environ.get("HINDSIGHT_NATIVE_RETAIN_EXTRACTION_MODE", "custom")
    retain_custom_instructions = os.environ.get(
        "HINDSIGHT_NATIVE_RETAIN_CUSTOM_INSTRUCTIONS",
        "ONLY extract durable user/project facts, decisions, results, preferences, stable environment facts. "
        "Skip tool logs, file listings, raw command output, process chatter, greetings. "
        "Max 3-5 facts per chunk."
    )
    consolidation_llm_batch_size = int_env("HINDSIGHT_NATIVE_CONSOLIDATION_LLM_BATCH_SIZE", 20)
    consolidation_max_memories_per_round = int_env(
        "HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_ROUND",
        int(os.environ.get("HINDSIGHT_NATIVE_CONSOLIDATION_MAX_MEMORIES_PER_JOB", "60")),
    )
    # v0.6.1 treats recall_budget as an env/runtime knob, not an exposed bank
    # config key. Do not PATCH it into /banks/{bank}/config; doing so just emits
    # noisy "key not supported" warnings while the container env remains correct.
    consolidation_source_facts_max_tokens = int_env("HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS", 4096)
    consolidation_source_facts_max_tokens_per_observation = int_env(
        "HINDSIGHT_NATIVE_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION", 256
    )
    updates: dict[str, Any] = {
        "retain_chunk_size": current_chunk_size,
        "retain_extraction_mode": retain_extraction_mode,
        "retain_custom_instructions": retain_custom_instructions,
        "enable_observations": enable_observations,
        "recall_max_tokens": 4096,
        "recall_chunks_max_tokens": 4096,
        "consolidation_llm_batch_size": consolidation_llm_batch_size,
        "consolidation_max_memories_per_round": consolidation_max_memories_per_round,
        # 控制 import-mode observations 的单次 prompt，避免 MiniMax 调用失控。
        "consolidation_source_facts_max_tokens": consolidation_source_facts_max_tokens,
        "consolidation_source_facts_max_tokens_per_observation": consolidation_source_facts_max_tokens_per_observation,
    }
    # Hindsight v0.6.x moved some consolidation knobs to env-only config.
    # Avoid failing the whole mode switch when a bank-config key is not exposed
    # by the running image; keep the env knobs in base_env() as the source of
    # truth for those settings.
    try:
        current = requests.get(f"{API}/v1/default/banks/{bank}/config", timeout=20)
        current.raise_for_status()
        current_cfg = current.json().get("config") or {}
        supported_keys = set(current_cfg.keys())
    except Exception as e:
        print(f"WARNING: could not pre-read bank config keys for {bank}: {e}", file=sys.stderr)
        supported_keys = set()
    if supported_keys:
        skipped = sorted(k for k in updates if k not in supported_keys)
        updates = {k: v for k, v in updates.items() if k in supported_keys}
        if skipped:
            print(
                "bank config keys not supported by current Hindsight image; skipped: "
                + ",".join(skipped),
                file=sys.stderr,
            )
    r = requests.patch(f"{API}/v1/default/banks/{bank}/config", json={"updates": updates}, timeout=20)
    if r.status_code >= 400:
        print(f"bank config patch failed: status={r.status_code} body={r.text[:1000]}", file=sys.stderr)
    r.raise_for_status()
    cfg = r.json().get("config", {})
    print(
        "bank config:",
        json.dumps(
            {
                "retain_chunk_size": cfg.get("retain_chunk_size"),
                "retain_extraction_mode": cfg.get("retain_extraction_mode"),
                "enable_observations": cfg.get("enable_observations"),
                "recall_max_tokens": cfg.get("recall_max_tokens"),
                "consolidation_llm_batch_size": cfg.get("consolidation_llm_batch_size"),
                "consolidation_max_memories_per_round": cfg.get("consolidation_max_memories_per_round"),
                "consolidation_recall_budget": cfg.get("consolidation_recall_budget"),
                "consolidation_source_facts_max_tokens": cfg.get("consolidation_source_facts_max_tokens"),
                "consolidation_source_facts_max_tokens_per_observation": cfg.get("consolidation_source_facts_max_tokens_per_observation"),
            },
            ensure_ascii=False,
        ),
    )


def ensure_hermes_hindsight_idle_config() -> None:
    cfg = {}
    if HINDSIGHT_CONFIG.exists():
        cfg = json.loads(HINDSIGHT_CONFIG.read_text(encoding="utf-8"))
    cfg.update(
        {
            "mode": "local_external",
            "api_url": API,
            "bank_id": BANK,
            "auto_recall": True,
            "auto_retain": False,
            "retain_every_n_turns": 50,
            "memory_mode": "hybrid",
            "recall_budget": "high",
            "recall_max_tokens": 4096,
            "recall_max_input_chars": 800,
            "recall_prefetch_method": "reflect",
            "recall_cache_enabled": False,
        }
    )
    HINDSIGHT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    HINDSIGHT_CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def recreate_container(env: dict[str, str]) -> None:
    ensure_pg0_running()
    env_file = write_env_file(env)
    try:
        print("recreating Docker container: hindsight")
        docker_shell("docker rm -f hindsight >/dev/null 2>&1 || true", check=True)
        cmd = shlex.join([
            "docker",
            "run",
            "-d",
            "--name",
            "hindsight",
            "--network",
            "host",
            "--restart",
            "unless-stopped",
            "-v",
            str(HOME / ".cache" / "huggingface") + ":/home/hindsight/.cache/huggingface",
            "-v",
            str(HOME / ".cache" / "huggingface") + f":{HOME}/.cache/huggingface",
            "-v",
            str(HOME / ".cache" / "torch") + ":/home/hindsight/.cache/torch",
            "-v",
            str(HOME / ".cache" / "torch") + f":{HOME}/.cache/torch",
            "--env-file",
            env_file,
            IMAGE,
        ])
        out = docker_shell(cmd, check=True).stdout.strip()
        print(f"container started: {out[:12]}")
        tuning_file = HERMES_HOME / "scripts" / "hindsight-consolidation-tuning-default-20x3.json"
        if tuning_file.exists():
            cp = docker_shell(
                f"docker cp {shlex.quote(str(tuning_file))} hindsight:/home/hindsight/.hindsight-consolidation-tuning.json",
                check=False,
            )
            if cp.returncode == 0:
                print("copied runtime consolidation tuning into container")
            else:
                print(f"WARNING: failed to copy runtime consolidation tuning: {cp.stderr.strip() or cp.stdout.strip()}", file=sys.stderr)
    finally:
        try:
            os.remove(env_file)
        except FileNotFoundError:
            pass


def patch_hindsight_container_and_restart() -> None:
    """Patch local Hindsight container hardening hooks, then restart once.

    Paid/native workflow containers are ephemeral; without reapplying these,
    MiniMax/GLM can hit fenced-JSON retry loops, and upstream native
    consolidation can process an entire unconsolidated bank in one operation.
    """
    patch_scripts = [
        ("JSON parser / 429 backoff", JSON_PARSER_PATCH_SCRIPT),
        ("native consolidation per-job budget", CONSOLIDATION_BUDGET_PATCH_SCRIPT),
        ("retain temporal FK guard", RETAIN_TEMPORAL_FK_PATCH_SCRIPT),
        ("parallel consolidator", CONSOLIDATOR_PARALLEL_PATCH_SCRIPT),
    ]
    for label, script in patch_scripts:
        if not script.exists():
            print(f"WARNING: {label} patch script missing: {script}")
            continue
        proc = subprocess.run(
            [sys.executable, str(script)],
            text=True,
            capture_output=True,
        )
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        if proc.returncode != 0:
            # Hindsight >=0.6.x already contains several former local fixes
            # (official max_memories_per_round, bounded source facts, improved
            # provider parsing).  Patch scripts are best-effort compatibility
            # hooks; do not block an upgrade if their exact old-code anchors
            # are gone.
            print(f"WARNING: skipped/failed Hindsight {label} patch on current image: exit={proc.returncode}", file=sys.stderr)
    print("restarting Hindsight after local patches")
    docker_shell("docker restart hindsight >/dev/null", check=True)


def patch_json_parser_and_restart() -> None:
    """Backward-compatible alias for older callers."""
    patch_hindsight_container_and_restart()


def existing_sqlite_doc_count(bank: str, *, client: HindsightNativeClient | None = None) -> int:
    client = client or HindsightNativeClient(api=API, bank=bank)
    return sum(1 for d in client.list_all_documents(max_items=100000) if str(d.get("id") or "").startswith("hermes-sqlite::"))


def _operation_refs_sqlite_doc(op: dict[str, Any]) -> bool:
    haystack = json.dumps(op.get("task_payload") or op.get("payload") or op, ensure_ascii=False, default=str)
    return "hermes-sqlite::" in haystack


def purge_sqlite_documents(
    bank: str,
    *,
    client: HindsightNativeClient | None = None,
    dry_run: bool = True,
    confirm_documents: str | None = None,
    confirm_operations: str | None = None,
) -> dict[str, Any]:
    """Clean SQLite-import documents through official Hindsight APIs.

    Destructive execution is intentionally opt-in. `--purge-sqlite` in the CLI
    now only previews by default; callers must pass explicit execute + confirm
    tokens to delete documents/operations.
    """
    client = client or HindsightNativeClient(api=API, bank=bank)
    docs = [d for d in client.list_all_documents(max_items=100000) if str(d.get("id") or "").startswith("hermes-sqlite::")]
    ops: list[dict[str, Any]] = []
    for status in ["pending", "processing", "failed"]:
        try:
            for op in client.iter_operations(status=status, max_items=100000):
                if _operation_refs_sqlite_doc(op):
                    ops.append(op)
        except Exception as e:
            return {
                "dry_run": dry_run,
                "error": f"operations API scan failed: {repr(e)[:500]}",
                "documents_matched": len(docs),
                "operations_matched": len(ops),
            }
    seen_ops: set[str] = set()
    unique_ops = []
    for op in ops:
        op_id = str(op.get("id") or op.get("operation_id") or "")
        if op_id and op_id not in seen_ops:
            seen_ops.add(op_id)
            unique_ops.append(op)

    report: dict[str, Any] = {
        "dry_run": dry_run,
        "bank": bank,
        "documents_matched": len(docs),
        "operations_matched": len(unique_ops),
        "document_ids": [str(d.get("id") or "") for d in docs[:50]],
        "operation_ids": [str(o.get("id") or o.get("operation_id") or "") for o in unique_ops[:50]],
        "required_document_confirm": DELETE_DOCUMENT_CONFIRM,
        "required_operation_confirm": DELETE_OPERATION_CONFIRM,
        "method": "official_api_documents_and_operations",
    }
    if dry_run:
        return report
    deleted_ops = 0
    deleted_docs = 0
    for op in unique_ops:
        op_id = str(op.get("id") or op.get("operation_id") or "")
        if not op_id:
            continue
        client.delete_operation(op_id, dry_run=False, confirm=confirm_operations)
        deleted_ops += 1
    for doc in docs:
        doc_id = str(doc.get("id") or "")
        if not doc_id:
            continue
        client.delete_document(doc_id, dry_run=False, confirm=confirm_documents)
        deleted_docs += 1
    report["operations_deleted"] = deleted_ops
    report["documents_deleted"] = deleted_docs
    return report


def docker_runtime_env(container: str = "hindsight") -> dict[str, str]:
    proc = subprocess.run(
        ["docker", "inspect", container, "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "docker inspect failed").strip()[:500])
    env: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def _int_from_env(env: dict[str, str], key: str, default: int, *, min_value: int = 1) -> int:
    try:
        return max(min_value, int(env.get(key) or default))
    except Exception:
        return max(min_value, default)


def _half(value: int, *, min_value: int = 1) -> int:
    return max(min_value, value // 2 if value > 1 else 1)


def build_half_consolidation_overrides(runtime_env: dict[str, str]) -> dict[str, str]:
    """Build the previously agreed half-downgrade profile without touching live state."""
    current_llm = _int_from_env(runtime_env, "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT", paid_llm_concurrency())
    current_parallel = _int_from_env(runtime_env, "HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES", 3)
    current_recall = _int_from_env(runtime_env, "HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT", 60, min_value=10)
    return {
        # This override is consolidation-only: retain/reflect can stay at the normal paid-LLM concurrency.
        "HINDSIGHT_NATIVE_CONSOLIDATION_LLM_MAX_CONCURRENT": str(_half(current_llm, min_value=1)),
        "HINDSIGHT_NATIVE_CONSOLIDATION_PARALLEL_BATCHES": str(_half(current_parallel, min_value=1)),
        "HINDSIGHT_NATIVE_CONSOLIDATION_RECALL_MAX_CONCURRENT": str(_half(current_recall, min_value=10)),
    }


def run_consolidation_half_downgrade(args: argparse.Namespace) -> int:
    runtime_env = docker_runtime_env()
    overrides = build_half_consolidation_overrides(runtime_env)
    before = {
        key: runtime_env.get(key)
        for key in [
            "HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT",
            "HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES",
            "HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT",
            "HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE",
            "HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE",
            "HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND",
        ]
    }
    report = {
        "schema_version": "hindsight-consolidation-half-downgrade-v1",
        "dry_run": not bool(args.execute),
        "confirm_required": HALF_DOWNGRADE_CONFIRM,
        "current_runtime": before,
        "overrides": overrides,
        "notes": [
            "Does not change current runtime unless --execute and confirm token are provided.",
            "The profile halves native-consolidation concurrency only: LLM_MAX_CONCURRENT, PARALLEL_BATCHES, RECALL_MAX_CONCURRENT.",
            "Apply only while queue is idle unless --allow-existing-queue is explicitly accepted.",
        ],
    }
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.confirm != HALF_DOWNGRADE_CONFIRM:
        raise SystemExit(f"--execute requires --confirm {HALF_DOWNGRADE_CONFIRM}")
    if not args.allow_existing_queue:
        pending, processing, q = queue_counts(BANK)
        if pending or processing:
            raise SystemExit(
                f"refusing half-downgrade while queue active bank={BANK} pending={pending} processing={processing} "
                f"source={q.get('queue_counts_source')}; wait for idle or rerun with --allow-existing-queue"
            )
    os.environ.update(overrides)
    switch_mode("normal-local", allow_existing_queue=True, health_timeout_s=args.health_timeout_s)
    print(json.dumps({**report, "dry_run": False, "applied": True}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def stats(bank: str = BANK) -> dict[str, Any]:
    r = requests.get(f"{API}/v1/default/banks/{bank}/stats", timeout=20)
    r.raise_for_status()
    return r.json()


def _operation_count_from_response(data: dict[str, Any]) -> int:
    for key in ("total", "total_operations", "count"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    for key in ("operations", "items", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def operation_status_count(bank: str, status: str) -> int:
    r = requests.get(
        f"{API}/v1/default/banks/{bank}/operations",
        params={"status": status, "limit": 1, "offset": 0, "exclude_parents": "true"},
        timeout=20,
    )
    r.raise_for_status()
    return _operation_count_from_response(r.json())


def queue_counts(bank: str = BANK) -> tuple[int, int, dict[str, Any]]:
    data = stats(bank)
    try:
        pending = operation_status_count(bank, "pending")
        processing = operation_status_count(bank, "processing")
        data["queue_counts_source"] = "operations_api_exclude_parents"
        return pending, processing, data
    except requests.RequestException as e:
        # Fallback for old servers / transient operations API failures. Stats can
        # include non-claimable parent batch rows, so keep the source explicit.
        data["queue_counts_source"] = f"bank_stats_fallback:{type(e).__name__}"
        ops = data.get("operations_by_status") or {}
        pending = int(data.get("pending_operations") or ops.get("pending") or 0)
        processing = int(ops.get("processing") or 0)
        return pending, processing, data


def assert_no_existing_queue_for_switch(label: str, *, queue_bank: str = BANK) -> None:
    """Fail before provider/container restart when claimable async work is active.

    This is intentionally separate from switch_mode() so orchestrated wrappers can
    run the guard before entering their restore/finally block. Otherwise a guard
    refusal could still trigger a disruptive "restore normal" container recreate
    even though no paid-LLM retain/import had started.
    """
    check_banks = [queue_bank]
    if queue_bank != BANK:
        check_banks.append(BANK)
    for check_bank in check_banks:
        try:
            pending, processing, q = queue_counts(check_bank)
        except requests.RequestException:
            # If current service is down, switching mode is still allowed.
            continue
        if pending or processing:
            raise SystemExit(
                f"refusing to switch to {label} with existing queue in bank={check_bank} "
                f"pending={pending}, processing={processing}, source={q.get('queue_counts_source')}; "
                "wait for idle or rerun with --allow-existing-queue if intentional"
            )


def print_status() -> None:
    try:
        print("health:", requests.get(f"{API}/health", timeout=5).text)
    except Exception as e:
        print("health error:", repr(e))
    try:
        p, pr, s = queue_counts()
        print(
            "stats:",
            json.dumps(
                {
                    "pending_operations": p,
                    "processing_operations": pr,
                    "failed_operations": s.get("failed_operations"),
                    "total_documents": s.get("total_documents"),
                    "total_nodes": s.get("total_nodes"),
                    "total_observations": s.get("total_observations"),
                    "operations_by_status": s.get("operations_by_status"),
                },
                ensure_ascii=False,
            ),
        )
    except Exception as e:
        print("stats error:", repr(e))
    proc = docker_shell(
        "docker inspect hindsight --format '{{range .Config.Env}}{{println .}}{{end}}' | "
        "grep -E 'HINDSIGHT_API_(LLM|RETAIN_LLM|CONSOLIDATION_LLM|REFLECT_LLM)_(PROVIDER|MODEL|BASE_URL|MAX_CONCURRENT)|HINDSIGHT_API_ENABLE_OBSERVATIONS|HINDSIGHT_API_WORKER_(MAX_SLOTS|CONSOLIDATION_MAX_SLOTS)|HINDSIGHT_API_RETAIN_MAX_CONCURRENT|HINDSIGHT_API_CONSOLIDATION_(PARALLEL_BATCHES|RECALL_MAX_CONCURRENT|BATCH_SIZE|LLM_BATCH_SIZE|MAX_MEMORIES_PER_ROUND)' | sort",
        check=False,
    )
    if proc.stdout.strip():
        print("provider env:\n" + proc.stdout.strip())


def try_disable_observations_before_restart(reason: str, *, bank: str = BANK) -> None:
    """Persistently disable Hindsight observations before restarting the server.

    If a previous mode left bank.enable_observations=true, starting the
    normal-local container and only then patching the bank leaves a small
    startup window where a local consolidation can be enqueued. Patch the
    running API first whenever possible; startup env still disables workers
    as a second guard.
    """
    try:
        health = requests.get(f"{API}/health", timeout=5)
        health.raise_for_status()
        patch_bank_config(enable_observations=False, bank=bank)
        print(f"pre-disabled bank={bank} observations/consolidation before {reason}")
    except Exception as e:
        print(f"WARNING: could not pre-disable observations before {reason}: {e}")


def switch_mode(
    mode: str,
    *,
    allow_existing_queue: bool = False,
    enable_observations: bool = False,
    llm_profile: dict[str, Any] | None = None,
    health_timeout_s: int = 300,
    queue_bank: str = BANK,
) -> None:
    ensure_hermes_hindsight_idle_config()
    if mode == "import-minimax":
        profile = llm_profile or get_llm_profile(DEFAULT_OFFLINE_LLM_PROFILE)
        label = str(profile.get("label") or "llm")
        if not allow_existing_queue:
            assert_no_existing_queue_for_switch(label, queue_bank=queue_bank)
        if queue_bank != BANK:
            # Paid/import mode should be scoped to the requested bank. Normal mode
            # may leave the default bank with observations enabled; disable it
            # before the container restart so a non-default smoke/import run does
            # not accidentally spend paid LLM calls on the production backlog.
            try_disable_observations_before_restart(f"import-{label} scoped restart", bank=BANK)
        if not enable_observations:
            try_disable_observations_before_restart(f"import-{label} restart", bank=queue_bank)
        recreate_container(paid_llm_env(profile, enable_observations=enable_observations))
        wait_health(health_timeout_s)
        patch_json_parser_and_restart()
        wait_health(health_timeout_s)
        patch_bank_config(enable_observations=enable_observations, bank=queue_bank)
        print(
            f"mode=import-{label}: {label}/{profile.get('model')} is active for retain; "
            f"observations={'enabled' if enable_observations else 'disabled'} until restored"
        )
    elif mode == "normal-local":
        recreate_container(paid_llm_env(get_llm_profile(DEFAULT_OFFLINE_LLM_PROFILE), enable_observations=True))
        patch_hindsight_container_and_restart()
        wait_health(health_timeout_s)
        patch_bank_config(enable_observations=True)
        print(f"mode=normal-local: remote precision mode active using {DEFAULT_OFFLINE_LLM_PROFILE}; observations enabled")
    else:
        raise SystemExit(f"unknown mode: {mode}")


def wait_queue_drained(poll_s: int, timeout_s: int, *, bank: str = BANK) -> None:
    started = time.time()
    while True:
        try:
            pending, processing, s = queue_counts(bank)
        except requests.RequestException as e:
            print(
                json.dumps(
                    {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "queue_poll_error": repr(e)[:300],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if timeout_s and time.time() - started > timeout_s:
                raise TimeoutError(f"queue did not drain within {timeout_s}s") from e
            time.sleep(poll_s)
            continue
        print(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "bank": bank,
                    "pending_operations": pending,
                    "processing_operations": processing,
                    "queue_counts_source": s.get("queue_counts_source"),
                    "total_documents": s.get("total_documents"),
                    "total_nodes": s.get("total_nodes"),
                    "total_observations": s.get("total_observations"),
                    "operations_by_status": s.get("operations_by_status"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if pending == 0 and processing == 0:
            print("queue drained")
            return
        if timeout_s and time.time() - started > timeout_s:
            raise TimeoutError(f"queue did not drain within {timeout_s}s")
        time.sleep(poll_s)


def run_sqlite_import_minimax(args: argparse.Namespace) -> int:
    if not getattr(args, "allow_deprecated_sqlite_import", False) and os.environ.get("HINDSIGHT_ALLOW_DEPRECATED_SQLITE_IMPORT") != "1":
        raise SystemExit(DEPRECATED_SQLITE_IMPORT_MESSAGE)
    profile = get_llm_profile(getattr(args, "llm_profile", None))
    label = str(profile.get("label") or "llm")
    import_args = list(args.import_args or [])
    if import_args and import_args[0] == "--":
        import_args = import_args[1:]
    if not arg_present(import_args, "--mode"):
        import_args = prepend_arg(import_args, "--mode", "submit")
    if not arg_present(import_args, "--group-by"):
        import_args = prepend_arg(import_args, "--group-by", "day-topic")
    if not arg_present(import_args, "--api"):
        import_args = prepend_arg(import_args, "--api", API)
    # 用户要求：后续增量 retain 默认使用 safe 模式，避免心跳/压缩 handoff/短确认等低价值内容进入付费 retain。
    if not arg_present(import_args, "--prefilter"):
        import_args = prepend_arg(import_args, "--prefilter", "safe")
    if args.enable_observations and "--enable-observations" not in import_args:
        import_args = ["--enable-observations"] + import_args

    if args.purge_sqlite:
        purge_report = purge_sqlite_documents(
            BANK,
            dry_run=not args.execute_purge,
            confirm_documents=args.confirm_purge_documents,
            confirm_operations=args.confirm_purge_operations,
        )
        print("sqlite purge report:", json.dumps(purge_report, ensure_ascii=False))
        if not args.execute_purge:
            raise SystemExit(
                "--purge-sqlite is dry-run by default; rerun with --execute-purge "
                f"--confirm-purge-documents {DELETE_DOCUMENT_CONFIRM} "
                f"--confirm-purge-operations {DELETE_OPERATION_CONFIRM} to delete through official APIs"
            )
        count = int(purge_report.get("documents_matched") or 0)
        # full rebuild should not reuse old document_id progress after purging documents.
        if "--full" in import_args:
            progress = HERMES_HOME / "hindsight" / "sqlite_import_progress.json"
            if progress.exists():
                backup = progress.with_suffix(progress.suffix + f".pre-purge-{int(time.time())}.bak")
                progress.rename(backup)
                print(f"renamed old progress: {backup}")

    exit_code = 1
    if not args.allow_existing_queue:
        assert_no_existing_queue_for_switch(label, queue_bank=BANK)
    try:
        switch_mode(
            "import-minimax",
            allow_existing_queue=True,
            enable_observations=args.enable_observations,
            llm_profile=profile,
            health_timeout_s=getattr(args, "health_timeout_s", 300),
        )
        cmd = [sys.executable, str(IMPORT_SCRIPT)] + import_args
        print("running import:", " ".join(shlex.quote(x) for x in cmd))
        proc = subprocess.run(cmd)
        exit_code = proc.returncode
        if proc.returncode != 0:
            print(f"import script failed with code {proc.returncode}; restoring local mode", file=sys.stderr)
            return proc.returncode
        if not args.no_wait:
            wait_queue_drained(args.poll, args.timeout, bank=BANK)
        else:
            print(f"WARNING: --no-wait used; remaining queue will be processed after restore by the default remote profile ({DEFAULT_OFFLINE_LLM_PROFILE}), not {label}")
        return proc.returncode
    finally:
        print("restoring precision remote mode...")
        try:
            switch_mode("normal-local", allow_existing_queue=True, health_timeout_s=getattr(args, "health_timeout_s", 300))
        except Exception as e:
            print(f"ERROR: failed to restore normal-local mode: {e}", file=sys.stderr)
            if exit_code == 0:
                exit_code = 1


def build_session_manifest_retain_cmd(args: argparse.Namespace) -> list[str]:
    runner_args = list(getattr(args, "runner_args", None) or [])
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    cmd = [
        sys.executable,
        str(SESSION_RETAIN_RUNNER),
        "--manifest",
        str(args.manifest),
        "--bank",
        str(args.bank),
        "--batch-size",
        str(args.batch_size),
        "--wait-timeout-s",
        str(args.wait_timeout_s),
        "--poll-s",
        str(args.poll_s),
        "--json",
    ]
    if getattr(args, "limit", None) is not None:
        cmd.extend(["--limit", str(args.limit)])
    if getattr(args, "ignore_submit_state", False):
        cmd.append("--ignore-submit-state")
    elif getattr(args, "submit_state", None):
        cmd.extend(["--submit-state", str(args.submit_state)])
    if getattr(args, "scan_state", None):
        cmd.extend(["--scan-state", str(args.scan_state)])
    if getattr(args, "no_wait", False):
        cmd.append("--no-wait")
    if getattr(args, "execute", False):
        cmd.extend(["--execute", "--confirm", str(args.confirm)])
    cmd.extend(runner_args)
    return cmd


def run_session_manifest_retain_llm(args: argparse.Namespace) -> int:
    """Switch to paid LLM, retain session/json manifest records, wait, restore local.

    Dry-run intentionally does not switch provider: it only exercises manifest
    filtering/rehydration and cost preview. Real execution requires the same
    retain confirm token as the underlying runner.
    """
    if args.execute and args.confirm != RETAIN_CONFIRM:
        raise SystemExit(f"--execute requires --confirm {RETAIN_CONFIRM}")

    cmd = build_session_manifest_retain_cmd(args)
    if not args.execute:
        print("running session manifest retain dry-run:", " ".join(shlex.quote(str(x)) for x in cmd))
        return subprocess.run(cmd).returncode

    profile = get_llm_profile(getattr(args, "llm_profile", None))
    label = str(profile.get("label") or "llm")
    target_bank = str(args.bank)
    if not args.allow_existing_queue:
        assert_no_existing_queue_for_switch(label, queue_bank=target_bank)
    exit_code = 1
    try:
        switch_mode(
            "import-minimax",
            allow_existing_queue=True,
            enable_observations=args.enable_observations,
            llm_profile=profile,
            health_timeout_s=getattr(args, "health_timeout_s", 300),
            queue_bank=target_bank,
        )
        ensure_bank_exists(target_bank)
        patch_bank_config(enable_observations=args.enable_observations, bank=target_bank)
        print("running session manifest retain:", " ".join(shlex.quote(str(x)) for x in cmd))
        proc = subprocess.run(cmd)
        exit_code = proc.returncode
        if proc.returncode != 0:
            print(f"session manifest retain failed with code {proc.returncode}; restoring local mode", file=sys.stderr)
        elif not args.no_wait:
            print(
                "session manifest retain runner already waited for submitted retain operation ids; "
                "skipping global queue drain before restore so native consolidation requeue does not drain the whole bank backlog"
            )
        else:
            print(f"WARNING: --no-wait used; remaining queue will be processed after restore by the default remote profile ({DEFAULT_OFFLINE_LLM_PROFILE}), not {label}")
    finally:
        print("restoring precision remote mode...")
        # Disable observations on the target bank before returning to daily mode.
        # switch_mode('normal-local') also protects the default BANK, but
        # session/json smoke runs usually target hermes_v3 or another candidate bank.
        try:
            patch_bank_config(enable_observations=False, bank=target_bank)
        except Exception as e:
            print(f"WARNING: failed to pre-disable target bank observations for {target_bank}: {e}", file=sys.stderr)
        try:
            switch_mode("normal-local", allow_existing_queue=True, health_timeout_s=getattr(args, "health_timeout_s", 300))
            patch_bank_config(enable_observations=False, bank=target_bank)
        except Exception as e:
            print(f"ERROR: failed to restore normal-local mode: {e}", file=sys.stderr)
            if exit_code == 0:
                exit_code = 1
        if not args.no_wait:
            # Restoring normal-local recreates the container and briefly enables
            # observations before the bank config is patched back. That can leave
            # a few native retain/observation operations processing after the
            # first wait has already passed. Drain a bounded tail so the next
            # provider switch does not trip the existing-queue guard, but do not
            # turn this cleanup into an unbounded whole-bank wait.
            try:
                post_restore_timeout = int(
                    getattr(args, "timeout", 0)
                    or os.environ.get("HINDSIGHT_POST_RESTORE_QUEUE_DRAIN_TIMEOUT_S", "600")
                )
                wait_queue_drained(args.poll, post_restore_timeout, bank=target_bank)
            except Exception as e:
                print(f"ERROR: post-restore queue wait failed for {target_bank}: {e}", file=sys.stderr)
                if exit_code == 0:
                    exit_code = 1
    return exit_code


def run_offline_reflect_minimax(args: argparse.Namespace) -> int:
    """Switch to selected paid LLM, run offline daily/weekly reflect/consolidation, wait retain queue, restore local."""
    profile = get_llm_profile(getattr(args, "llm_profile", None))
    label = str(profile.get("label") or "llm")
    reflect_args = list(args.reflect_args or [])
    if reflect_args and reflect_args[0] == "--":
        reflect_args = reflect_args[1:]
    if not arg_present(reflect_args, "--mode"):
        reflect_args = prepend_arg(reflect_args, "--mode", "submit")
    if not arg_present(reflect_args, "--api"):
        reflect_args = prepend_arg(reflect_args, "--api", API)
    if not arg_present(reflect_args, "--bank"):
        reflect_args = prepend_arg(reflect_args, "--bank", BANK)
    target_bank = str(arg_value(reflect_args, "--bank", BANK) or BANK)
    if not arg_present(reflect_args, "--prefilter"):
        reflect_args = prepend_arg(reflect_args, "--prefilter", "safe")
    if not arg_present(reflect_args, "--llm-model"):
        reflect_args = prepend_arg(reflect_args, "--llm-model", str(profile["model"]))
    if not arg_present(reflect_args, "--llm-base-url"):
        reflect_args = prepend_arg(reflect_args, "--llm-base-url", str(profile["base_url"]))
    if not arg_present(reflect_args, "--llm-api-key-env"):
        reflect_args = prepend_arg(reflect_args, "--llm-api-key-env", str(profile["api_key_env"]))
    if not arg_present(reflect_args, "--llm-label"):
        reflect_args = prepend_arg(reflect_args, "--llm-label", label)
    if not profile.get("response_format", True) and "--no-response-format" not in reflect_args:
        reflect_args = ["--no-response-format"] + reflect_args

    exit_code = 1
    if not args.allow_existing_queue:
        assert_no_existing_queue_for_switch(label, queue_bank=target_bank)
    try:
        # 这里默认不启用 Hindsight 内置 observations；daily/weekly 脚本本身就是可控的离线 reflect/consolidation。
        switch_mode(
            "import-minimax",
            allow_existing_queue=True,
            enable_observations=args.enable_hindsight_observations,
            llm_profile=profile,
            health_timeout_s=getattr(args, "health_timeout_s", 300),
            queue_bank=target_bank,
        )
        cmd = [sys.executable, str(OFFLINE_REFLECT_SCRIPT)] + reflect_args
        print("running offline reflect:", " ".join(shlex.quote(x) for x in cmd))
        proc = subprocess.run(cmd)
        exit_code = proc.returncode
        if proc.returncode != 0:
            print(f"offline reflect failed with code {proc.returncode}; waiting any submitted queue before restore", file=sys.stderr)
            if not args.no_wait:
                try:
                    wait_queue_drained(args.poll, args.timeout, bank=target_bank)
                except Exception as e:
                    print(f"ERROR: queue wait after failed offline reflect also failed: {e}", file=sys.stderr)
            return proc.returncode
        if not args.no_wait:
            wait_queue_drained(args.poll, args.timeout, bank=target_bank)
        else:
            print(f"WARNING: --no-wait used; posted reflect docs may be retained after restore by the default remote profile ({DEFAULT_OFFLINE_LLM_PROFILE}), not {label}")
        return proc.returncode
    finally:
        print("restoring precision remote mode...")
        try:
            switch_mode("normal-local", allow_existing_queue=True, health_timeout_s=getattr(args, "health_timeout_s", 300))
        except Exception as e:
            print(f"ERROR: failed to restore normal-local mode: {e}", file=sys.stderr)
            if exit_code == 0:
                exit_code = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Hindsight paid-LLM import mode manager (MiniMax/GLM/DeepSeek/custom)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Show Hindsight health/stats/provider mode")
    p_status.set_defaults(func=lambda _a: print_status())

    p_normal = sub.add_parser("normal-local", help="Switch 8888 to precision remote mode (legacy command name)")
    p_normal.set_defaults(func=lambda a: switch_mode("normal-local"))

    p_mini = sub.add_parser("import-minimax", aliases=["import-llm"], help="Switch 8888 to paid-LLM import mode without running import")
    p_mini.add_argument("--llm-profile", help="minimax, bailian/glm, deepseek, or custom; default is bailian/glm")
    p_mini.add_argument("--allow-existing-queue", action="store_true")
    p_mini.add_argument("--enable-observations", action="store_true", help="Also enable observations/consolidation during import mode")
    p_mini.set_defaults(
        func=lambda a: switch_mode(
            "import-minimax",
            allow_existing_queue=a.allow_existing_queue,
            enable_observations=a.enable_observations,
            llm_profile=get_llm_profile(a.llm_profile),
        )
    )

    p_wait = sub.add_parser("wait-queue", help="Wait until Hindsight async queue drains")
    p_wait.add_argument("--bank", default=BANK)
    p_wait.add_argument("--poll", type=int, default=60)
    p_wait.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    p_wait.set_defaults(func=lambda a: wait_queue_drained(a.poll, a.timeout, bank=a.bank))

    p_half = sub.add_parser("consolidation-half-downgrade", help="Preview/apply the safe half-downgrade profile for native consolidation concurrency")
    p_half.add_argument("--execute", action="store_true", help=f"Actually recreate Hindsight with half native-consolidation concurrency; requires --confirm {HALF_DOWNGRADE_CONFIRM}")
    p_half.add_argument("--confirm")
    p_half.add_argument("--allow-existing-queue", action="store_true", help="Allow disruptive recreate while Hindsight queue is active; avoid unless explicitly accepted")
    p_half.add_argument("--health-timeout-s", type=int, default=300)
    p_half.set_defaults(func=run_consolidation_half_downgrade)

    p_import = sub.add_parser("sqlite-import-minimax", aliases=["sqlite-import-llm"], help="DEPRECATED/BLOCKED: old SQLite day-topic import; use session-manifest-retain-llm")
    p_import.add_argument("--llm-profile", help="minimax, bailian/glm, deepseek, or custom; default is bailian/glm")
    p_import.add_argument("--allow-existing-queue", action="store_true")
    p_import.add_argument("--purge-sqlite", action="store_true", help="Preview deletion of old hermes-sqlite::* documents and related ops before a full rebuild; dry-run unless --execute-purge + confirm tokens are provided")
    p_import.add_argument("--execute-purge", action="store_true", help="Actually execute --purge-sqlite through official Hindsight documents/operations APIs")
    p_import.add_argument("--confirm-purge-documents", help=f"Required with --execute-purge: {DELETE_DOCUMENT_CONFIRM}")
    p_import.add_argument("--confirm-purge-operations", help=f"Required with --execute-purge: {DELETE_OPERATION_CONFIRM}")
    p_import.add_argument("--enable-observations", action="store_true", help="Opt in to observations/consolidation; default off for safe SQLite rebuilds")
    p_import.add_argument("--no-wait", action="store_true")
    p_import.add_argument("--poll", type=int, default=60)
    p_import.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    p_import.add_argument("--health-timeout-s", type=int, default=300, help="Seconds to wait for Hindsight health after provider/container switches")
    p_import.add_argument("--allow-deprecated-sqlite-import", action="store_true", help=argparse.SUPPRESS)
    p_import.add_argument("import_args", nargs=argparse.REMAINDER, help="Arguments passed to import_sqlite_to_hindsight.py; prefix with -- if needed")
    p_import.set_defaults(func=run_sqlite_import_minimax)

    p_session = sub.add_parser("session-manifest-retain-minimax", aliases=["session-manifest-retain-llm"], help="Switch to paid LLM, retain reviewed session/json manifest records, wait queue, restore precision remote")
    p_session.add_argument("--llm-profile", help="minimax, bailian/glm, deepseek, or custom; default is bailian/glm")
    p_session.add_argument("--allow-existing-queue", action="store_true")
    p_session.add_argument("--enable-observations", action="store_true", help="Opt in to Hindsight native observations/consolidation during this retain window; default off")
    p_session.add_argument("--no-wait", action="store_true")
    p_session.add_argument("--poll", type=int, default=60)
    p_session.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    p_session.add_argument("--health-timeout-s", type=int, default=300, help="Seconds to wait for Hindsight health after provider/container switches")
    p_session.add_argument("--manifest", required=True, type=Path)
    p_session.add_argument("--bank", default="hermes_v3")
    p_session.add_argument("--limit", type=int, default=None)
    p_session.add_argument("--batch-size", type=int, default=5)
    p_session.add_argument("--submit-state", type=Path, default=HERMES_HOME / "hindsight" / "session_ingest" / "submit_state.json")
    p_session.add_argument("--scan-state", type=Path, default=None, help="Per-source manifest candidate scan state updated after successful retain")
    p_session.add_argument("--ignore-submit-state", action="store_true")
    p_session.add_argument("--execute", action="store_true", help=f"Actually submit retain. Requires --confirm {RETAIN_CONFIRM}. Without this, only dry-runs the runner and does not switch provider.")
    p_session.add_argument("--confirm")
    p_session.add_argument("--wait-timeout-s", type=int, default=1800, help="Seconds to wait for submitted retain operation ids; paid MiniMax + observations can exceed 600s even for small batches")
    p_session.add_argument("--poll-s", type=float, default=5.0)
    p_session.add_argument("runner_args", nargs=argparse.REMAINDER, help="Extra arguments passed to hindsight_session_retain_runner.py; prefix with -- if needed")
    p_session.set_defaults(func=run_session_manifest_retain_llm)

    p_reflect = sub.add_parser("offline-reflect-minimax", aliases=["offline-reflect-llm"], help="Switch to paid LLM, run offline daily/weekly reflect/consolidation, wait queue, restore precision remote")
    p_reflect.add_argument("--llm-profile", help="minimax, bailian/glm, deepseek, or custom; default is bailian/glm")
    p_reflect.add_argument("--allow-existing-queue", action="store_true")
    p_reflect.add_argument("--enable-hindsight-observations", action="store_true", help="Also enable Hindsight built-in observations; default off because offline script performs controlled consolidation")
    p_reflect.add_argument("--no-wait", action="store_true")
    p_reflect.add_argument("--poll", type=int, default=60)
    p_reflect.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    p_reflect.add_argument("--health-timeout-s", type=int, default=300, help="Seconds to wait for Hindsight health after provider/container switches")
    p_reflect.add_argument("reflect_args", nargs=argparse.REMAINDER, help="Arguments passed to offline_hindsight_reflect_consolidate.py; prefix with -- if needed")
    p_reflect.set_defaults(func=run_offline_reflect_minimax)

    args = parser.parse_args()
    result = args.func(args)
    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == "__main__":
    main()
