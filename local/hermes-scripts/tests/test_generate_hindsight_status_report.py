import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'generate_hindsight_status_report.py'


def load_module():
    spec = importlib.util.spec_from_file_location('generate_hindsight_status_report_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_archive_old_reports_keeps_bounded_active_set(tmp_path):
    module = load_module()
    reports = tmp_path / 'reports'
    archive = tmp_path / 'archive'
    reports.mkdir()
    for index in range(16):
        (reports / f'hindsight-status-202607{index + 1:02d}-000000.md').write_text(
            str(index), encoding='utf-8'
        )

    moved = module.archive_old_reports(reports, archive, keep=14)

    assert moved == 2
    assert len(list(reports.glob('hindsight-status-*.md'))) == 14
    assert len(list(archive.glob('hindsight-status-*.md'))) == 2
