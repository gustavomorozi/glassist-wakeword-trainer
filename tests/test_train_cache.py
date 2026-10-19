"""The feature cache is a plain .npz (no pickle: loading it can never run code)."""
import os
import sys

import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('torch')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wakeword_trainer import train  # noqa: E402


def test_cache_roundtrip_and_refuses_pickles(tmp_path):
    data = {'train': (np.ones((3, 16, 96), np.float32), np.zeros((5, 16, 96), np.float32)),
            'hold': (np.ones((1, 16, 96), np.float32), np.zeros((2, 16, 96), np.float32))}
    path = str(tmp_path / 'c.npz')
    train._save_cache(path, data)
    back = train._load_cache(path)
    assert set(back) == {'train', 'hold'}
    assert back['train'][0].shape == (3, 16, 96) and back['hold'][1].shape == (2, 16, 96)
    obj = tmp_path / 'evil.npz'
    np.savez(str(obj), train_pos=np.array([object()], dtype=object), train_neg=np.zeros(1),
             hold_pos=np.zeros(1), hold_neg=np.zeros(1))
    with pytest.raises(ValueError):
        train._load_cache(str(obj))
