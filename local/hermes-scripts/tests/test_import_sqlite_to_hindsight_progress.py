import importlib.util
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_full_rebuild_ignores_historical_processed_progress(tmp_path):
    mod = load_module('import_sqlite_to_hindsight')
    progress_path = tmp_path / 'sqlite_import_progress.json'
    progress_path.write_text(
        '{"processed":["old-doc"],"last_imported_timestamp":123,"last_imported_iso":"2026-01-01T00:00:00","total_sessions_imported":9,"total_bundles_imported":8}',
        encoding='utf-8',
    )

    progress = mod.progress_for_run(full=True, path=progress_path)

    assert progress['processed'] == []
    assert progress['last_imported_timestamp'] == 0.0
    assert progress['last_imported_iso'] is None
    assert progress['total_sessions_imported'] == 0
    assert progress['total_bundles_imported'] == 0


def test_incremental_uses_historical_processed_progress(tmp_path):
    mod = load_module('import_sqlite_to_hindsight')
    progress_path = tmp_path / 'sqlite_import_progress.json'
    progress_path.write_text(
        '{"processed":["old-doc"],"last_imported_timestamp":123,"last_imported_iso":"2026-01-01T00:00:00","total_sessions_imported":9,"total_bundles_imported":8}',
        encoding='utf-8',
    )

    progress = mod.progress_for_run(full=False, path=progress_path)

    assert progress['processed'] == ['old-doc']
    assert progress['last_imported_timestamp'] == 123
