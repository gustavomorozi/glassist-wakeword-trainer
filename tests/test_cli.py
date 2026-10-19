"""The command line: consent for the online TTS, JSON events, model install."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wakeword_trainer import cli  # noqa: E402


def _init(tmp_path, capsys):
    ws = str(tmp_path / 'p')
    assert cli.main(['--json', 'init', '--workspace', ws, '--word', 'Jarvis', '--language', 'en']) == 0
    capsys.readouterr()
    return ws


def test_generate_refuses_without_consent(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv('WAKEWORD_TRAINER_ALLOW_ONLINE_TTS', raising=False)
    ws = _init(tmp_path, capsys)
    assert cli.main(['--json', 'generate', '--workspace', ws]) == 2
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event['event'] == 'error' and event['code'] == 'consent_required'
    assert 'Microsoft' in event['msg']


def test_install_needs_a_trained_model(tmp_path, capsys):
    ws = _init(tmp_path, capsys)
    assert cli.main(['--json', 'install', '--workspace', ws, '--glassist', str(tmp_path)]) == 1


def test_install_does_not_overwrite_without_force(tmp_path, capsys):
    ws = _init(tmp_path, capsys)
    from wakeword_trainer.workspace import Workspace
    w = Workspace(ws)
    os.makedirs(os.path.dirname(w.model_path), exist_ok=True)
    open(w.model_path, 'wb').write(b'new')
    gl = tmp_path / 'g'
    (gl / 'models').mkdir(parents=True)
    (gl / 'wake_word_detector.py').write_text('')
    (gl / 'models' / 'jarvis.onnx').write_bytes(b'old')
    assert cli.main(['--json', 'install', '--workspace', ws, '--glassist', str(gl)]) == 3
    assert (gl / 'models' / 'jarvis.onnx').read_bytes() == b'old'
    assert cli.main(['--json', 'install', '--workspace', ws, '--glassist', str(gl), '--force']) == 0
    assert (gl / 'models' / 'jarvis.onnx').read_bytes() == b'new'
