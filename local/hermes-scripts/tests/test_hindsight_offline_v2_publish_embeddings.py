import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    name = "hindsight_offline_v2_publish_embeddings"
    fake_psycopg2 = types.SimpleNamespace(
        connect=lambda *args, **kwargs: None,
        extras=types.SimpleNamespace(RealDictCursor=object, execute_values=lambda *args, **kwargs: None),
    )
    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extras"] = fake_psycopg2.extras
    spec = importlib.util.spec_from_file_location(name, ROOT / "hindsight_offline_v2_publish.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_encode_preserves_order_and_dimension(monkeypatch):
    module = load_module()
    calls = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append((request.full_url, payload, timeout))
        rows = [
            {"index": i, "embedding": [float(i), float(i + 1), float(i + 2)]}
            for i in reversed(range(len(payload["input"])))
        ]
        return FakeResponse({"data": rows})

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    result = module.ollama_encode(
        ["a", "b", "c"],
        model="bge-m3",
        base_url="http://127.0.0.1:11434/v1",
        batch_size=2,
    )

    assert result == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.0, 1.0, 2.0]]
    assert [call[0] for call in calls] == ["http://127.0.0.1:11434/v1/embeddings"] * 2
    assert all(call[1]["model"] == "bge-m3" for call in calls)


def test_auto_provider_is_native_ollama_and_does_not_fallback_to_null(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "ollama_encode", lambda texts, **kwargs: [[0.0, 1.0] for _ in texts])

    embeddings, provider = module.maybe_embeddings(
        ["a", "b"],
        "auto",
        "bge-m3",
        base_url="http://127.0.0.1:11434/v1",
    )

    assert provider == "ollama"
    assert embeddings == [[0.0, 1.0], [0.0, 1.0]]


def test_native_embedding_failure_is_not_converted_to_null(monkeypatch):
    module = load_module()

    def fail(*args, **kwargs):
        raise RuntimeError("embedding backend unavailable")

    monkeypatch.setattr(module, "ollama_encode", fail)
    try:
        module.maybe_embeddings(["a"], "ollama", "bge-m3")
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("native embedding failure must fail closed")


def test_ollama_encode_rejects_invalid_response_indices(monkeypatch):
    module = load_module()
    vector = [0.0, 1.0, 2.0]
    malformed_responses = [
        [{"index": 0, "embedding": vector}, {"embedding": vector}],
        [{"index": 0, "embedding": vector}, {"index": 0, "embedding": vector}],
        [{"index": 0, "embedding": vector}, {"index": 2, "embedding": vector}],
    ]
    for rows in malformed_responses:
        monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, rows=rows, **_kwargs: FakeResponse({"data": rows}))
        try:
            module.ollama_encode(["a", "b"], model="bge-m3", batch_size=2)
        except RuntimeError as exc:
            assert "index" in str(exc).lower() or "indice" in str(exc).lower()
        else:
            raise AssertionError(f"invalid embedding indices must fail closed: {rows!r}")


def test_canonical_backup_uses_owner_only_directory_and_file(tmp_path):
    import stat

    module = load_module()

    class Cursor:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def execute(self, *_args, **_kwargs):
            pass
        def fetchall(self):
            return []

    class Connection:
        def cursor(self, **_kwargs):
            return Cursor()

    path = module.backup_existing(Connection(), "hermes", tmp_path / "backups")
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_publish_rejects_null_embeddings_before_opening_database(monkeypatch, tmp_path):
    module = load_module()
    monkeypatch.setattr(module, "card_document_id", lambda _card: "doc-1")
    monkeypatch.setattr(module, "render_card_document", lambda _card: "document")
    monkeypatch.setattr(module, "observation_text", lambda _obs, _card: "observation")
    monkeypatch.setattr(module, "observation_tags", lambda _obs, _card: [])
    monkeypatch.setattr(module, "observation_entities", lambda _obs, _card: [])
    monkeypatch.setattr(module, "source_memory_uuid_array", lambda _obs: [])
    monkeypatch.setattr(module, "connect", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("DB opened before embedding validation")))
    card = {"card_id": "card-1", "scope": "topic", "topic": "test", "canonical_observations": [{"id": "obs-1", "text": "observation"}]}

    try:
        module.publish(
            [card], dsn="unused", bank="hermes", mode="publish", replace=True,
            embedding_provider="none", embedding_model="bge-m3",
            embedding_base_url="http://127.0.0.1:11434/v1", embedding_batch_size=8,
            backup_dir=tmp_path,
        )
    except RuntimeError as exc:
        assert "null" in str(exc).lower() or "missing" in str(exc).lower()
    else:
        raise AssertionError("canonical publish must reject NULL embeddings")
