"""The openWakeWord base models are verified against pinned SHA-256 hashes."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wakeword_trainer import models_check  # noqa: E402


def test_pinned_hashes_cover_the_three_base_models():
    files = models_check.expected()
    assert len(files) == 3 and all(len(h) == 64 for h in files.values())


def test_verify_reports_missing_and_tampered(tmp_path, monkeypatch):
    good = b'genuine'
    names = list(models_check.expected())
    monkeypatch.setattr(models_check, 'expected',
                        lambda: {n: hashlib.sha256(good).hexdigest() for n in names})
    assert len(models_check.verify(str(tmp_path))) == 3                      # all missing
    for n in names:
        (tmp_path / n).write_bytes(good)
    assert models_check.verify(str(tmp_path)) == []
    (tmp_path / names[0]).write_bytes(b'tampered')
    problems = models_check.verify(str(tmp_path))
    assert len(problems) == 1 and 'does not match' in problems[0]


def test_ensure_refuses_unverified_models(monkeypatch):
    monkeypatch.setattr(models_check, 'verify', lambda directory=None: ['x: missing'])
    monkeypatch.setattr(models_check, 'models_dir', lambda: 'somewhere')
    with pytest.raises(RuntimeError):
        models_check.ensure(download=False)
