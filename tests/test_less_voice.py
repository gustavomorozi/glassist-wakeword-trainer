"""Synthetic transients, labelled detection import, speed perturbation and threshold advice."""
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np = pytest.importorskip('numpy')
sf = pytest.importorskip('soundfile')
pytest.importorskip('scipy')

from wakeword_trainer import detections, features, transients  # noqa: E402
from wakeword_trainer.evaluate import recommend_threshold  # noqa: E402


class Ws:
    def __init__(self, root):
        self.neg = str(root / 'neg'); self.real_neg = str(root / 'real_neg'); self.real_pos = str(root / 'real_pos')
        for d in (self.neg, self.real_neg, self.real_pos):
            os.makedirs(d)


def test_transients_are_three_second_clips_with_a_real_event(tmp_path):
    ws = Ws(tmp_path)
    out = transients.generate(ws, count=len(transients.KINDS) * 2)
    assert out['created'] == len(transients.KINDS) * 2
    names = sorted(os.listdir(ws.neg))
    assert {n.split('_')[2] for n in names} == set(transients.KINDS)          # every kind is made
    for n in names:
        a, sr = sf.read(os.path.join(ws.neg, n), dtype='float32')
        assert sr == 16000 and abs(len(a) / sr - 3.0) < 0.01
        assert np.isfinite(a).all() and np.abs(a).max() > 0.1                 # there is an actual sound
    assert transients.generate(ws, count=len(transients.KINDS) * 2)['created'] == 0   # idempotent


def test_transients_are_reproducible(tmp_path):
    one = transients.make_clip('click', np.random.default_rng(7))
    two = transients.make_clip('click', np.random.default_rng(7))
    assert np.array_equal(one, two)


def _write_clip(folder, name):
    sf.write(os.path.join(folder, name), np.zeros(48000, np.float32) + 0.01, 16000)


def test_import_labels_by_what_the_pipeline_did_next(tmp_path):
    ws = Ws(tmp_path)
    det = tmp_path / 'det'; det.mkdir()
    for name in ('trig_20261021_135416_bruninha_098.wav',   # nothing recognised -> false
                 'trig_20261021_140000_bruninha_090.wav',   # command understood -> true
                 'trig_20261021_150000_bruninha_080.wav',   # no log line -> unknown
                 'not_a_clip.wav'):
        _write_clip(str(det), name)
    log = tmp_path / 'g.log'
    log.write_text(
        "2026-10-21 13:54:16,100 - haassist - INFO - Wake word detected: 'bruninha' (confidence: 0.98)\n"
        "2026-10-21 13:54:20,000 - satellite_protocol - ERROR - Pipeline error: stt-no-text-recognized\n"
        "2026-10-21 14:00:03,000 - satellite_protocol - INFO - Voice event: VOICE_ASSISTANT_INTENT_END data={}\n",
        encoding='utf-8')
    out = detections.import_detections(ws, str(det), [str(log)])
    assert out['false_to_real_neg'] == 1 and out['true_to_real_pos'] == 0 and out['unknown'] == 2
    assert os.listdir(ws.real_neg) == ['real_env_det_trig_20261021_135416_bruninha_098.wav']
    assert os.listdir(ws.real_pos) == []                                      # positives are opt-in
    out = detections.import_detections(ws, str(det), [str(log)], include_positives=True)
    assert out['true_to_real_pos'] == 1 and out['already_there'] == 1
    assert len(os.listdir(ws.real_pos)) == 1


def test_label_uses_only_events_shortly_after_the_clip():
    t = dt.datetime(2026, 10, 21, 13, 54, 16)
    events = [(t - dt.timedelta(seconds=40), 'no_text'), (t + dt.timedelta(seconds=90), 'no_text')]
    assert detections.label(t, events) is None                                # too old / too late


def test_speed_perturb_changes_length_but_not_loudness_much():
    a = np.sin(np.linspace(0, 2000, 16000)).astype(np.float32)
    lengths = {len(features.speed_perturb(a)) for _ in range(20)}
    assert len(lengths) > 1 and all(14000 < n < 18500 for n in lengths)


def test_threshold_advice_picks_the_lowest_threshold_with_no_false_clip():
    advice = recommend_threshold([0.99, 0.95, 0.9, 0.4], [0.2, 0.55, 0.1] + [0.05] * 70)
    assert advice['threshold'] == 0.55 and advice['recall'] == 0.75 and advice['evidence'] == 'ok'
    assert recommend_threshold([], [0.1]) is None
    weak = recommend_threshold([0.9], [0.1] * 10)
    assert weak['evidence'] == 'weak'


def test_consensus_output_is_the_lowest_head_and_keeps_the_interface():
    torch = pytest.importorskip('torch')
    from wakeword_trainer.train import Consensus, Head
    torch.manual_seed(0)
    heads = [Head(8).eval() for _ in range(3)]
    group = Consensus(heads).eval()
    x = torch.randn(5, 16, 96)
    with torch.no_grad():
        each = torch.stack([h(x) for h in heads])
        out = group(x)
    assert out.shape == (5, 1)
    assert torch.allclose(out, each.min(0).values)
    assert bool((out <= each.max(0).values + 1e-9).all())


def test_mining_candidates_are_valid_three_second_clips():
    from wakeword_trainer import mine
    rng = np.random.default_rng(5)
    kinds = set()
    for _ in range(40):
        c = mine.candidate(rng)
        assert c.dtype == np.float32 and len(c) == 48000 and np.isfinite(c).all() and np.abs(c).max() <= 1.0
        kinds.add(float(np.abs(c).max()) > 0)
    assert kinds == {True}


def test_a_vowel_is_voiced_not_silent():
    from wakeword_trainer import mine
    v = mine._vowel(np.random.default_rng(1), 0.2, 150)
    assert len(v) == int(16000 * 0.2) and np.abs(v).max() > 0.9
