"""Render composed note specs to audio from one-shot samples, with labels.

Why a sampler instead of rendering MIDI through a plugin: we control every
detail that matters for the labels. The label Hz of each note is the
MEASURED f0 of the sample (manifest), times the tuning shift we apply, so
it is exact by construction.

Augmentations (all applied per piece, all reflected in the labels):
* whole-ensemble tuning shift of +-40 cents, plus +-6 cents per instrument
  (every real gamelan set has its own tuning; the model must not memorise
  one set of absolute frequencies)
* velocity-layer choice and gain, damping fade for damped instruments
* mic EQ, synthetic room reverb (or your real impulse responses), noise
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from gamelan_amt.synth.compose import Keys, NoteSpec
from gamelan_amt.tokenize import Note
from gamelan_amt.tuning import SampleInfo

SR = 16000


@dataclass
class LoadedSample:
    info: SampleInfo
    audio: np.ndarray       # mono float32 at 16 kHz
    ring_seconds: float     # time until -40 dB below peak


def _ring_time(y: np.ndarray, sr: int, drop_db: float = 40.0) -> float:
    env = np.abs(y)
    win = max(1, int(0.01 * sr))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    thr = env.max() * 10 ** (-drop_db / 20)
    above = np.nonzero(env > thr)[0]
    return float(above[-1] / sr) if len(above) else len(y) / sr


class SampleBank:
    def __init__(self, samples: list[SampleInfo]):
        import librosa

        self.by_key: dict[tuple[str, str, int, int], list[LoadedSample]] = {}
        for s in samples:
            if math.isnan(s.f0_hz):
                continue
            y, sr = sf.read(s.path, dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            y = y / (np.abs(y).max() + 1e-9)
            ls = LoadedSample(s, y.astype(np.float32), _ring_time(y, SR))
            self.by_key.setdefault((s.laras, s.instrument, s.degree, s.octave), []).append(ls)
        for v in self.by_key.values():
            v.sort(key=lambda x: x.info.layer)

    def laras_list(self) -> list[str]:
        return sorted({k[0] for k in self.by_key})

    def keys_for(self, laras: str) -> dict[str, Keys]:
        out: dict[str, set] = {}
        for (la, inst, d, o) in self.by_key:
            if la == laras:
                out.setdefault(inst, set()).add((d, o))
        return {i: Keys(k) for i, k in out.items()}

    def pick(self, laras, inst, degree, octave, velocity) -> LoadedSample:
        layers = self.by_key[(laras, inst, degree, octave)]
        return layers[min(int(velocity * len(layers)), len(layers) - 1)]


def _shift(y: np.ndarray, ratio: float) -> np.ndarray:
    """Pitch-shift by resampling (also changes duration, like a real retune)."""
    if abs(ratio - 1.0) < 1e-6:
        return y
    return signal.resample(y, max(1, int(round(len(y) / ratio)))).astype(np.float32)


def _synthetic_rir(rng: random.Random, rt60: float) -> np.ndarray:
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    rir = nrng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0 / rng.uniform(0.3, 1.0)        # direct path
    return (rir / np.abs(rir).sum() * 4).astype(np.float32)


def render_piece(specs: list[NoteSpec], laras: str, bank: SampleBank, seed: int,
                 rir_files: list[Path] | None = None) -> tuple[np.ndarray, list[Note], dict]:
    rng = random.Random(seed)
    global_c = rng.uniform(-40, 40)
    inst_c = {i: rng.uniform(-6, 6) for i in {s.instrument for s in specs}}
    inst_gain = {i: rng.uniform(0.4, 1.0) for i in inst_c}

    end = max((s.onset for s in specs), default=0.0) + 4.0
    out = np.zeros(int(end * SR) + SR * 8, dtype=np.float32)
    labels: list[Note] = []
    cache: dict[tuple, np.ndarray] = {}

    for s in specs:
        ls = bank.pick(laras, s.instrument, s.degree, s.octave, s.velocity)
        ratio = 2 ** ((global_c + inst_c[s.instrument]) / 1200)
        key = (id(ls), ratio)
        if key not in cache:
            cache[key] = _shift(ls.audio, ratio)
        y = cache[key]
        ring = ls.ring_seconds / ratio
        if s.offset is not None and s.offset - s.onset < ring:
            dur = s.offset - s.onset
            fade = int(rng.uniform(0.03, 0.08) * SR)       # hand damping
            n = int(dur * SR)
            y = y[: n + fade].copy()
            y[n:] *= np.linspace(1.0, 0.0, len(y) - n, dtype=np.float32)
            off = s.offset
        else:
            off = s.onset + ring
        start = int(s.onset * SR)
        gain = (s.velocity ** 1.5) * inst_gain[s.instrument]
        out[start:start + len(y)] += gain * y[: len(out) - start]
        labels.append(Note(s.instrument, s.onset, off, ls.info.f0_hz * ratio,
                           extra={"degree": s.degree, "octave": s.octave}))

    out = out[: int((max(n.offset for n in labels) + 0.5) * SR)] if labels else out

    # --- room + mic ---------------------------------------------------------
    if rng.random() < 0.8:
        if rir_files:
            rir, sr = sf.read(str(rng.choice(rir_files)), dtype="float32", always_2d=True)
            rir = rir.mean(axis=1)
            if sr != SR:
                rir = signal.resample(rir, int(len(rir) * SR / sr)).astype(np.float32)
            rir /= np.abs(rir).max() + 1e-9
        else:
            rir = _synthetic_rir(rng, rng.uniform(0.2, 1.5))
        wet = signal.fftconvolve(out, rir)[: len(out)]
        mix = rng.uniform(0.1, 0.6)
        out = (1 - mix) * out + mix * wet / (np.abs(wet).max() + 1e-9) * np.abs(out).max()
    if rng.random() < 0.5:
        sos = signal.butter(2, rng.uniform(30, 120), "highpass", fs=SR, output="sos")
        out = signal.sosfilt(sos, out)
    if rng.random() < 0.5:
        sos = signal.butter(2, rng.uniform(4000, 7800), "lowpass", fs=SR, output="sos")
        out = signal.sosfilt(sos, out)
    if rng.random() < 0.6:
        snr_db = rng.uniform(15, 50)
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        noise = nrng.standard_normal(len(out)).astype(np.float32)
        if rng.random() < 0.5:                                 # pinkish
            noise = signal.lfilter([0.049922, -0.095993, 0.050612, -0.004408],
                                   [1, -2.494956, 2.017265, -0.522189], noise)
        p_sig = np.mean(out ** 2) + 1e-12
        noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / (np.mean(noise ** 2) + 1e-12))
        out = out + noise
    out = out / (np.abs(out).max() + 1e-9) * rng.uniform(0.3, 0.95)

    meta = {"seed": seed, "laras": laras, "global_cents": global_c,
            "instrument_cents": inst_c}
    return out.astype(np.float32), labels, meta
