"""Thin normal-entry routing. Legacy all-history flags never reach a writer."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def alias_arguments(*, mode, execute, config=None, confirm=None, include_wiki=False,
                    resume_only=False, bank=None, llm_profile=None, llm_overrides=None,
                    max_units=None, max_chars=None):
    if mode not in {'daily', 'weekly', 'full'}:
        raise ValueError('invalid_incremental_mode')
    if execute and confirm != 'run-hindsight-pipeline':
        raise ValueError('explicit_run_confirmation_required')
    argv = ['run' if execute else 'plan', '--mode', mode]
    if config is not None:
        argv += ['--config', str(config)]
    if execute:
        argv += ['--confirm', confirm]
    if include_wiki:
        argv.append('--include-wiki')
    if resume_only:
        argv.append('--resume-only')
    if bank is not None:
        argv += ['--bank', bank]
    if llm_profile is not None:
        argv += ['--llm-profile', llm_profile]
    overrides = llm_overrides or {}
    for key, flag in {'model': '--llm-model', 'base_url': '--llm-base-url',
                      'api_key_env': '--llm-api-key-env', 'provider': '--llm-provider'}.items():
        if overrides.get(key) is not None:
            argv += [flag, overrides[key]]
    if max_units is not None:
        argv += ['--max-units', str(max_units)]
    if max_chars is not None:
        argv += ['--max-chars', str(max_chars)]
    return argv


def run_alias(*, mode, execute, config=None, confirm=None, include_wiki=False,
              resume_only=False, bank=None, llm_profile=None, llm_overrides=None,
              max_units=None, max_chars=None):
    argv = alias_arguments(mode=mode, execute=execute, config=config, confirm=confirm,
        include_wiki=include_wiki, resume_only=resume_only, bank=bank, llm_profile=llm_profile,
        llm_overrides=llm_overrides, max_units=max_units, max_chars=max_chars)
    from hindsight_pipeline_common import load_config
    cfg = load_config(config)
    incremental = cfg.get('incremental', {})
    if not isinstance(incremental, dict):
        raise ValueError('invalid_incremental_configuration')
    nested_publication = incremental.get('publication', {})
    top_publication = cfg.get('publication', {})
    if not isinstance(nested_publication, dict) or not isinstance(top_publication, dict):
        raise ValueError('invalid_publication_configuration')
    publication = dict(nested_publication)
    publication.update(top_publication)
    if 'publisher_python' in publication:
        target = publication['publisher_python']
        if (not isinstance(target, str) or not target.strip()
                or any(ord(c) < 32 or ord(c) == 127 for c in target)
                or not Path(target).is_absolute() or not Path(target).is_file()):
            raise ValueError('invalid_runtime_interpreter')
        if Path(target).resolve() != Path(sys.executable).resolve():
            # Cron ignores Python shebangs. This is an explicitly configured
            # whole-pipeline interpreter, not an import/provider fallback.
            return subprocess.run([target, '-u', str(script_path('hindsight_incremental_pipeline.py')),
                                   *argv], check=False).returncode
    from hindsight_incremental_pipeline import main
    return main(argv)


def script_path(name):
    return Path(__file__).resolve().with_name(name)
