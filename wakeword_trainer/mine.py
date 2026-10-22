"""Adversarial negative mining: make lots of random candidate sounds, let the CURRENT model score them, and keep
only the ones that fool it. Training on those teaches the model exactly what it still gets wrong, instead of
sounds it already ignores (random clicks alone turned out to be too easy: no model reacted to them).

Candidates cover sharp sounds (the transients) and speech-like ones (syllable bursts through vowel formants,
noise with a syllable rhythm), because wake-word models mostly confuse the word with speech-like sounds.
"""
import os

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from . import SAMPLE_RATE as SR
from . import transients

FORMANTS = {'a': (800, 1200), 'e': (500, 1800), 'i': (300, 2300), 'o': (500, 900), 'u': (350, 700)}


def _vowel(rng, dur, f0):
    """A voiced vowel: harmonic stack shaped by two formant resonances."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    f0 = f0 * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(3, 7) * t))              # vibrato-ish pitch movement
    phase = 2 * np.pi * np.cumsum(f0) / SR
    src = sum(np.sin(h * phase) / h for h in range(1, 30))
    out = src
    for fc in FORMANTS[rng.choice(list(FORMANTS))]:
        r = 0.97
        w = 2 * np.pi * fc * rng.uniform(0.9, 1.1) / SR
        out = lfilter([1 - r], [1, -2 * r * np.cos(w), r * r], out)
    return out / (np.abs(out).max() + 1e-9)


def _syllables(rng):
    """Mumbled speech-like clip: 2-6 vowel+consonant bursts with a natural rhythm, no real words."""
    n = int(SR * 3.0)
    clip = np.zeros(n)
    t = rng.uniform(0.2, 0.8)
    f0 = rng.uniform(85, 260)
    for _ in range(int(rng.integers(2, 7))):
        d = rng.uniform(0.08, 0.28)
        s = int(t * SR)
        if s + int(SR * (d + 0.06)) >= n:
            break
        cons = transients._noise_burst(rng, 40, 1500, 6500, 25) * rng.uniform(0.2, 0.7)      # consonant-like hiss
        vow = _vowel(rng, d, f0 * rng.uniform(0.85, 1.2))
        vow = vow * np.hanning(len(vow))
        seg = np.concatenate([cons, vow])
        clip[s:s + len(seg)] += seg[:n - s]
        t += d + rng.uniform(0.03, 0.18)
    return clip


def candidate(rng):
    """One random 3 s candidate (float32, peak-normalised, with a little background)."""
    kind = rng.choice(['transient', 'syllables', 'syllables', 'noise_rhythm'])
    if kind == 'transient':
        clip = transients.make_clip(str(rng.choice(transients.KINDS)), rng)
    elif kind == 'syllables':
        clip = _syllables(rng).astype(np.float32)
    else:
        n = int(SR * 3.0)
        env = np.clip(np.sin(2 * np.pi * rng.uniform(2, 7) * np.arange(n) / SR), 0, None)
        clip = (rng.standard_normal(n) * env).astype(np.float32)
        clip = lfilter([0.5], [1, -rng.uniform(0.2, 0.9)], clip).astype(np.float32)
    peak = np.abs(clip).max()
    if peak > 0:
        clip = clip / peak * rng.uniform(0.15, 0.9)
    clip = clip + rng.standard_normal(len(clip)).astype(np.float32) * rng.uniform(0.0005, 0.01)
    return np.clip(clip, -1, 1).astype(np.float32)


def mine(ws, model_path, candidates=2000, keep=150, min_score=0.2, seed=777, progress=None):
    """Score `candidates` random sounds with the model and save the `keep` worst offenders (score >= min_score)
    into ws.neg as neg_hard_<n>.wav. Returns counts and the highest scores found."""
    from .evaluate import OnnxHead, clip_max_score
    head = OnnxHead(model_path)
    rng = np.random.default_rng(seed)
    scored = []
    tmp = os.path.join(ws.out, '_mine_tmp.wav')
    for i in range(candidates):
        clip = candidate(rng)
        sf.write(tmp, clip, SR, subtype='PCM_16')
        scored.append((clip_max_score(head, tmp), i, clip))
        if progress:
            progress(i + 1, candidates)
    if os.path.exists(tmp):
        os.remove(tmp)
    scored.sort(key=lambda x: -x[0])
    chosen = [s for s in scored if s[0] >= min_score][:keep]
    for rank, (score, i, clip) in enumerate(chosen):
        sf.write(os.path.join(ws.neg, f'neg_hard_{seed}_{rank:03d}.wav'), clip, SR, subtype='PCM_16')
    return {'candidates': candidates, 'fooled_the_model': sum(1 for s in scored if s[0] >= 0.5),
            'above_min_score': sum(1 for s in scored if s[0] >= min_score), 'saved': len(chosen),
            'top_scores': [round(s[0], 3) for s in scored[:5]]}
