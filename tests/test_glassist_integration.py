"""Putting a model into a GLaSSIST folder: never overwrite silently, keep the default wake word."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wakeword_trainer.integrations import glassist  # noqa: E402


@pytest.fixture
def gl(tmp_path):
    d = tmp_path / 'GLaSSIST'
    d.mkdir()
    (d / 'wake_word_detector.py').write_text('')
    return d


@pytest.fixture
def model(tmp_path):
    m = tmp_path / 'out' / 'bruninha.onnx'
    m.parent.mkdir()
    m.write_bytes(b'onnx')
    return m


def test_rejects_a_folder_that_is_not_glassist(tmp_path, model):
    assert glassist.check_dir(str(tmp_path / 'missing'))
    assert glassist.check_dir(str(tmp_path))
    with pytest.raises(ValueError):
        glassist.install_model(str(model), str(tmp_path), 'bruninha')


def test_adds_to_the_list_and_keeps_the_default_when_unset(gl, model):
    result = glassist.install_model(str(model), str(gl), 'bruninha')
    assert result['models'] == ['alexa', 'bruninha']            # GLaSSIST's default is not dropped
    assert (gl / 'models' / 'bruninha.onnx').read_bytes() == b'onnx'
    env = (gl / '.env').read_text(encoding='utf-8')
    assert 'HA_WAKE_WORD_MODELS=alexa,bruninha' in env and 'HA_WAKE_WORD_ENABLED=true' in env


def test_keeps_other_settings_and_is_idempotent(gl, model):
    (gl / '.env').write_text('HA_HOST=x\nHA_WAKE_WORD_MODELS=bruna\n', encoding='utf-8')
    assert glassist.install_model(str(model), str(gl), 'bruninha')['models'] == ['bruna', 'bruninha']
    assert glassist.install_model(str(model), str(gl), 'bruninha')['models'] == ['bruna', 'bruninha']
    assert 'HA_HOST=x' in (gl / '.env').read_text(encoding='utf-8')


def test_replace_is_the_only_way_to_reduce_the_list(gl, model):
    (gl / '.env').write_text('HA_WAKE_WORD_MODELS=bruna,alexa\n', encoding='utf-8')
    assert glassist.install_model(str(model), str(gl), 'bruninha', replace=True)['models'] == ['bruninha']


def test_never_overwrites_a_different_model_silently(gl, model):
    (gl / 'models').mkdir()
    (gl / 'models' / 'bruninha.onnx').write_bytes(b'mine')
    with pytest.raises(glassist.ModelExistsError):
        glassist.install_model(str(model), str(gl), 'bruninha')
    assert (gl / 'models' / 'bruninha.onnx').read_bytes() == b'mine'
    assert not (gl / '.env').exists()
    result = glassist.install_model(str(model), str(gl), 'bruninha', force=True)
    assert result['overwrote'] and (gl / 'models' / 'bruninha.onnx').read_bytes() == b'onnx'
