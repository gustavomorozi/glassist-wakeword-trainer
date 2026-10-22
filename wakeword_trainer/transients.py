"""Synthetic short, sharp sounds (clicks, knocks, claps, key presses, thumps, glass rings) used as NEGATIVES.

A wake word model hears a lot of these all day (keyboard, mouse, desk, door, cup) and a model trained
only on speech and music tends to confuse them with the word. Recording them takes your time; making
them from scratch takes none. Every clip is 3 s long: quiet background plus one or more events at a
random place, optionally with a little room reverb. Output goes to data/neg as neg_tr_<kind>_<n>.wav
so it is split into train / hold-out the same way as the other synthetic negatives.
"""
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter

from . import SAMPLE_RATE as SR

KINDS = ('click', 'knock', 'clap', 'keys', 'thump', 'ring')
CLIP_SECONDS = 3.0


def _band(x, lo, hi):
    hi = min(hi, SR / 2 - 100)
    b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype='band')
    return lfilter(b, a, x)


def _env(n, decay_s):
    return np.exp(-np.arange(n) / (SR * decay_s))


def _noise_burst(rng, ms, lo, hi, decay_ms):
    n = max(8, int(SR * ms / 1000))
    return _band(rng.standard_normal(n), lo, hi) * _env(n, decay_ms / 1000)


def _tone_burst(rng, freq, ms, decay_ms):
    n = max(8, int(SR * ms / 1000))
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28)) * _env(n, decay_ms / 1000)


def _mix(*parts):
    """Sum arrays of different lengths (shorter ones are padded with silence)."""
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[:len(p)] += p
    return out


def _event(rng, kind):
    """One event (a short array) of the given kind."""
    if kind == 'click':
        return _noise_burst(rng, rng.uniform(3, 12), rng.uniform(1200, 3000), rng.uniform(4000, 7500), rng.uniform(1.5, 6))
    if kind == 'knock':
        f = rng.uniform(90, 420)
        return _mix(_tone_burst(rng, f, 180, rng.uniform(25, 90)), 0.4 * _noise_burst(rng, 40, 200, 2500, 12))
    if kind == 'clap':
        return _noise_burst(rng, rng.uniform(25, 70), 500, 7000, rng.uniform(12, 35))
    if kind == 'keys':
        return _mix(_noise_burst(rng, rng.uniform(4, 14), rng.uniform(800, 2500), rng.uniform(3500, 7000), rng.uniform(2, 7)),
                    0.5 * _tone_burst(rng, rng.uniform(150, 300), 40, 15))
    if kind == 'thump':
        return _mix(_tone_burst(rng, rng.uniform(45, 160), 350, rng.uniform(60, 160)), 0.3 * _noise_burst(rng, 60, 60, 900, 25))
    return _tone_burst(rng, rng.uniform(1500, 5200), 500, rng.uniform(80, 280))      # ring


def make_clip(kind, rng):
    """A 3 s clip: background, then 1-12 events of `kind` at random times, sometimes with room reverb."""
    n = int(SR * CLIP_SECONDS)
    clip = np.zeros(n, np.float32)
    count = {'click': rng.integers(1, 4), 'keys': rng.integers(4, 13), 'knock': rng.integers(1, 5),
             'clap': rng.integers(1, 4), 'thump': 1, 'ring': 1}[kind]
    t = rng.uniform(0.2, 1.4)
    for _ in range(int(count)):
        ev = _event(rng, kind) * rng.uniform(0.3, 1.0)
        s = int(t * SR)
        if s + len(ev) >= n:
            break
        clip[s:s + len(ev)] += ev.astype(np.float32)
        t += rng.uniform(0.05, 0.45) if kind in ('keys', 'click') else rng.uniform(0.15, 0.6)
    if rng.random() < 0.4:                                      # a little room
        ir = rng.standard_normal(int(SR * rng.uniform(0.1, 0.5)))
        ir = ir * np.exp(-6.9 * np.arange(len(ir)) / len(ir))
        ir[0] = 1.0
        clip = clip + fftconvolve(clip, ir * rng.uniform(0.05, 0.3), mode='full')[:n].astype(np.float32)
    peak = np.abs(clip).max()
    if peak > 0:
        clip = clip / peak * rng.uniform(0.15, 0.95)
    hiss = rng.standard_normal(n).astype(np.float32) * rng.uniform(0.0005, 0.01)   # quiet background
    return np.clip(clip + hiss, -1, 1)


def generate(ws, count=300, seed=2026, progress=None):
    """Write `count` clips into ws.neg (skips the ones that already exist). Returns {'created', 'existing'}."""
    created = existing = 0
    for i in range(count):
        kind = KINDS[i % len(KINDS)]
        path = os.path.join(ws.neg, f'neg_tr_{kind}_{i:04d}.wav')
        if os.path.exists(path):
            existing += 1
        else:
            clip = make_clip(kind, np.random.default_rng(seed + i))
            sf.write(path, clip, SR, subtype='PCM_16')
            created += 1
        if progress:
            progress(i + 1, count)
    return {'created': created, 'existing': existing, 'folder': ws.neg}
