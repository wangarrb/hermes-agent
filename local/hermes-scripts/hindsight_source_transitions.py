"""Explicitly approved SQLite header transitions for the incremental reader.

The reader never looks these up: trusted callers (cron wrappers, CLI users)
supply the certificate files and the independently pinned digests explicitly.
Each pinned digest must match exactly one certificate; a partial or mismatched
set fails closed before any scan. Absent directory means no approved
transitions on this deployment.
"""
from __future__ import annotations

import json
from pathlib import Path

TRANSITIONS_DIRNAME = 'source-transitions'

# Independently reviewed canonical SHA256 digests, pinned per profile after
# review of each certificate file. Empty means no approved transitions.
APPROVED_SOURCE_TRANSITION_SHA256 = {
    'default': '63e9af5b3545751e0783573453f3d6f1872300f5994747c56009eb5510215ab3',
    'coordinator': '85c02f7ec7636fe9b344b0c9e3eff2576452bfbc253bbabbdb0a1786a4dfe3b0',
    'critic': 'bc1726f65b620109a4ac49693289193df45a8153ed80f77442bd05b5bbe7c376',
    'designer': 'e22b621473e7f81963c0d4641f914ad7f265c8eb72af5071fc244709df8c7248',
    'implementer': '92a1bd92698b804aced8b3974fe4764bc8abc46e9d55b012513990b74e8fd575',
    'lj': 'ddbebfe828215d9add8974ab3fee6b4a07e764ef5e48ea561d8581e65b859778',
    'planner': '22fb825eeaa10e0faaaa3b4a9d62b3a4bbd1fc1a1e73d13eee8ef21c81778cd4',
    'reviewer': 'f8b3d2270aa83a8041fca7058c141e1451fb2186fb0161db9df98fe5d0db05fc',
}


def load_source_transition_arguments(hermes_home):
    """Return (certificate_paths, approved_sha256s) for an explicit caller.

    Absent transitions directory: (None, None). Present directory: every
    pinned profile must have its certificate file, each file must reproduce
    its pinned digest, and nothing else is adopted implicitly.
    """
    home = Path(hermes_home)
    directory = home / 'hindsight' / TRANSITIONS_DIRNAME
    if not directory.is_dir() or not APPROVED_SOURCE_TRANSITION_SHA256:
        return None, None
    from hindsight_incremental_merge import canonical_hash

    paths, digests = [], []
    for profile in sorted(APPROVED_SOURCE_TRANSITION_SHA256):
        path = directory / f'certificate-{profile}.json'
        if not path.is_file():
            raise ValueError(f'source transition certificate missing: {profile}')
        certificate = json.loads(path.read_text())
        if canonical_hash(certificate) != APPROVED_SOURCE_TRANSITION_SHA256[profile]:
            raise ValueError(f'source transition certificate digest mismatch: {profile}')
        paths.append(path)
        digests.append(APPROVED_SOURCE_TRANSITION_SHA256[profile])
    return paths, digests
