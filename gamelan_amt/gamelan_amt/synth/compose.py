"""Generate gamelan-style multi-instrument note patterns (symbolic only).

This is a simplified, rule-based model of Central Javanese practice, good
enough to give the transcriber realistic textures: a balungan (skeleton
melody), its elaborations, and the colotomic (punctuating) instruments.

* balungan: saron_barung / saron_demung / slenthem, one note per beat
* saron_peking: each balungan note twice, anticipating the beat
* bonang_barung: "mipil" — alternates pairs of balungan notes, 2 or 4 per beat
* bonang_penerus: same idea at double density, higher register
* gender / gambang: dense figuration that lands on the next balungan note
  (gambang often in parallel octaves)
* kenong / kempul / gong / kethuk: structural strokes on fixed beats

It deliberately randomises tempo, irama density, instrument subset and
timing so the model sees a wide range of textures, including solo lines.
Musicologists will find it crude; that is fine for pre-training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SCALES = {
    "slendro": [[1, 2, 3, 5, 6]],
    "pelog": [[1, 2, 3, 5, 6], [2, 3, 5, 6, 7], [1, 2, 4, 5, 6]],  # common pathet subsets
}
GONG_TONES = {"slendro": [2, 5, 6], "pelog": [2, 5, 6]}

FORMS = {
    # beats per gong cycle, kenong beats, kempul beats, kethuk beats (1-indexed)
    "lancaran": (16, [4, 8, 12, 16], [6, 10, 14], [2, 6, 10, 14]),
    "ladrang": (32, [8, 16, 24, 32], [12, 20, 28], [2, 6, 10, 14, 18, 22, 26, 30]),
}

BALUNGAN = ["saron_barung", "saron_demung", "slenthem"]
DAMPED = {"saron_barung", "saron_demung", "slenthem", "saron_peking", "gender"}
REGISTER = {
    "slenthem": "low", "saron_demung": "mid", "saron_barung": "mid",
    "saron_peking": "high", "bonang_barung": "mid", "bonang_penerus": "high",
    "gender": "mid", "gambang": "mid", "kenong": "mid", "kempul": "mid",
    "gong": "low", "kethuk": "low",
}


@dataclass
class NoteSpec:
    instrument: str
    degree: int
    octave: int
    onset: float
    offset: float | None   # None = let it ring (renderer measures decay)
    velocity: float        # 0..1


class Keys:
    """Available (degree, octave) keys of one instrument in one laras."""

    def __init__(self, keys: set[tuple[int, int]]):
        self.keys = keys

    def octave_for(self, degree: int, register: str) -> int | None:
        octs = sorted(o for d, o in self.keys if d == degree)
        if not octs:
            return None
        return {"low": octs[0], "high": octs[-1]}.get(register, octs[len(octs) // 2])

    def octaves_for(self, degree: int) -> list[int]:
        return sorted(o for d, o in self.keys if d == degree)

    def lowest(self) -> tuple[int, int]:
        return min(self.keys, key=lambda k: (k[1], k[0]))


def _beat_times(n_beats: int, period: float, rng: random.Random) -> list[float]:
    times, t, p = [], rng.uniform(0.2, 1.0), period
    rit_from = n_beats - 8 if rng.random() < 0.4 else n_beats  # final slowdown
    for k in range(n_beats):
        times.append(t)
        p *= 1.0 + rng.gauss(0, 0.005)                          # tempo drift
        if k >= rit_from:
            p *= rng.uniform(1.02, 1.07)
        t += p
    return times


def _balungan(n_beats: int, cycle: int, scale: list[int], gong_tone: int,
              rng: random.Random) -> list[int | None]:
    """Scale-step random walk; the last beat of every cycle is the gong tone."""
    n = len(scale)
    pos = scale.index(gong_tone) + n
    melody: list[int | None] = []
    for k in range(n_beats):
        if (k + 1) % cycle == 0:
            target = scale.index(gong_tone) + n * round(pos / n)
            pos = target
        else:
            pos += rng.choice([-2, -1, -1, 1, 1, 2, 0])
            pos = max(0, min(3 * n - 1, pos))
        is_rest = rng.random() < 0.08 and (k + 1) % 4 != 0
        melody.append(None if is_rest else scale[pos % n])
    return melody


def _jit(t: float, rng: random.Random, sd: float = 0.006) -> float:
    return t + rng.gauss(0.0, sd)


def compose_piece(keys: dict[str, Keys], laras: str, seed: int,
                  min_seconds: float = 30.0, max_seconds: float = 90.0) -> list[NoteSpec]:
    """keys: instrument -> available keys for this laras. Returns note specs."""
    rng = random.Random(seed)
    scale = rng.choice(SCALES[laras])
    gong_tone = rng.choice([g for g in GONG_TONES[laras] if g in scale])
    form = rng.choice(list(FORMS))
    cycle, kenong_b, kempul_b, kethuk_b = FORMS[form]

    period = rng.uniform(0.45, 1.3)                       # seconds per beat
    n_beats = max(cycle, int(rng.uniform(min_seconds, max_seconds) / period))
    n_beats = (n_beats // cycle) * cycle
    beats = _beat_times(n_beats, period, rng)
    melody = _balungan(n_beats, cycle, scale, gong_tone, rng)
    density = 4 if period > 0.8 else 2                    # crude irama switch

    # Which instruments play: sometimes solo, usually an ensemble subset.
    usable = [i for i in keys if keys[i].keys]
    if rng.random() < 0.15:
        players = [rng.choice(usable)]
    else:
        k = rng.randint(min(2, len(usable)), len(usable))
        players = rng.sample(usable, k)
        bal = [b for b in BALUNGAN if b in usable]
        if bal and not any(p in BALUNGAN for p in players):
            players.append(rng.choice(bal))

    notes: list[NoteSpec] = []
    level = {p: rng.uniform(0.45, 1.0) for p in players}

    def add(inst: str, degree: int | None, t: float, register: str | None = None,
            octave: int | None = None, accent: float = 0.0) -> None:
        if degree is None:
            return
        if octave is None:
            octave = keys[inst].octave_for(degree, register or REGISTER.get(inst, "mid"))
            if octave is None:
                return
        vel = max(0.05, min(1.0, level[inst] + accent + rng.gauss(0, 0.08)))
        notes.append(NoteSpec(inst, degree, octave, max(0.0, _jit(t, rng)), None, vel))

    for k, (t, deg) in enumerate(zip(beats, melody)):
        beat_no = k % cycle + 1
        prev_t = beats[k - 1] if k else t - period
        nxt = next((m for m in melody[k + 1:] if m is not None), deg)
        for inst in players:
            if inst in BALUNGAN:
                add(inst, deg, t, accent=0.1 if beat_no % 4 == 0 else 0.0)
            elif inst == "saron_peking" and deg is not None:
                add(inst, deg, (prev_t + t) / 2)
                add(inst, deg, t)
            elif inst in ("bonang_barung", "bonang_penerus") and k % 2 == 1:
                # Two-beat span ending on this beat: a b a b ... ending on b.
                d = density * (2 if inst == "bonang_penerus" else 1)
                a = melody[k - 1] or deg
                b = deg or a
                t0 = beats[k - 2] if k >= 2 else beats[k - 1] - period
                for j in range(1, 2 * d + 1):
                    add(inst, a if j % 2 else b, t0 + j * (t - t0) / (2 * d))
            elif inst in ("gender", "gambang") and deg is not None:
                d = density * 2
                cur = rng.choice(keys[inst].octaves_for(deg) or [0])
                for j in range(1, d + 1):
                    tt = prev_t + j * (t - prev_t) / d
                    g = deg if j == d else rng.choice(scale)
                    if j == d - 1:
                        g = nxt if nxt in scale else g
                    octs = keys[inst].octaves_for(g)
                    if not octs:
                        continue
                    o = min(octs, key=lambda x: abs(x - cur))
                    add(inst, g, tt, octave=o)
                    if inst == "gambang" and rng.random() < 0.7 and len(octs) > 1:
                        o2 = o + 1 if o + 1 in octs else o - 1
                        if o2 in octs:
                            add(inst, g, tt, octave=o2)
            elif inst == "kenong" and beat_no in kenong_b:
                add(inst, deg or gong_tone, t, accent=0.1)
            elif inst == "kempul" and beat_no in kempul_b:
                add(inst, deg or gong_tone, t)
            elif inst == "kethuk" and beat_no in kethuk_b:
                d0, o0 = keys[inst].lowest()
                add(inst, d0, t, octave=o0)
            elif inst == "gong" and beat_no == cycle:
                d0 = gong_tone if keys[inst].octaves_for(gong_tone) else keys[inst].lowest()[0]
                add(inst, d0, t, register="low", accent=0.2)

    # Damped instruments: the player mutes the previous key as the next is struck.
    notes.sort(key=lambda n: n.onset)
    last: dict[str, NoteSpec] = {}
    for n in notes:
        if n.instrument in DAMPED:
            p = last.get(n.instrument)
            if p is not None and p.offset is None:
                p.offset = n.onset + rng.uniform(0.0, 0.04)
            last[n.instrument] = n
    return notes
