"""Measure the real pitch of gamelan samples; map Hz back to scale degrees.

Why this exists: a sample library maps gamelan keys onto MIDI note numbers,
but the MIDI number is NOT the sounding frequency. Every training label's
Hz must come from measuring the audio. The same table, measured on a real
ensemble, turns annotators' "instrument + degree" labels into Hz and turns
decoded Hz back into degrees after transcription.

Sample folder convention (one-shot strikes, any sample rate):

    samples/<laras>/<instrument>/<degree>_<octave>[_v<layer>].wav
    e.g. samples/slendro/saron_barung/6_1_v2.wav
         laras slendro, degree 6, octave 1, velocity layer 2

<laras> is "slendro" or "pelog" (real ensembles have one set of each).

`degree` is the gamelan number (slendro 1 2 3 5 6, pelog 1..7); `octave` is
the instrument's register index (0 = lowest key range of that instrument).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FNAME_RE = re.compile(r"^(?P<degree>\d)_(?P<octave>-?\d+)(?:_v(?P<layer>\d+))?$")


@dataclass
class SampleInfo:
    laras: str
    instrument: str
    path: str
    degree: int
    octave: int
    layer: int
    f0_hz: float = float("nan")
    peak_hz: float = float("nan")
    flag: str = ""


def scan_samples(root: Path) -> list[SampleInfo]:
    out = []
    for laras_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for inst_dir in sorted(p for p in laras_dir.iterdir() if p.is_dir()):
            for wav in sorted(inst_dir.glob("*.wav")):
                m = FNAME_RE.match(wav.stem)
                if not m:
                    print(f"skip (name not degree_octave[_vN]): {wav}")
                    continue
                out.append(SampleInfo(laras_dir.name, inst_dir.name, str(wav),
                                      int(m["degree"]), int(m["octave"]),
                                      int(m["layer"] or 1)))
    return out


def measure_f0(y: np.ndarray, sr: int, fmin: float = 40.0,
               fmax: float = 4000.0, within_db: float = 20.0) -> tuple[float, float]:
    """-> (fundamental partial, strongest partial), both in Hz.

    Do NOT use pYIN/YIN here: they look for a common period, which
    inharmonic metallophone partials do not have, and return octave (or
    worse) errors. Tested on synthetic bars/kettles: pYIN was off by
    1200-3600 cents on most instruments; this method was within 1 cent.

    Method: magnitude spectrum of 60-600 ms after the attack (past the strike
    transient), zero-padded x8, parabolic peak interpolation. The fundamental
    is the LOWEST prominent peak within `within_db` of the strongest one.
    """
    from scipy.signal import find_peaks

    y = y.astype(np.float64)
    if y.ndim > 1:
        y = y.mean(axis=0)
    attack = int(np.argmax(np.abs(y) > 0.1 * np.abs(y).max()))
    seg = y[attack + int(0.06 * sr): attack + int(0.6 * sr)]
    if len(seg) < int(0.1 * sr):
        seg = y[attack:]

    n = 1 << int(math.ceil(math.log2(len(seg) * 8)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=n))
    db = 20 * np.log10(spec + 1e-12)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lo, hi = np.searchsorted(freqs, [fmin, fmax])
    peaks, _ = find_peaks(db[lo:hi], prominence=10.0)
    if not len(peaks):
        return float("nan"), float("nan")
    peaks = peaks + lo

    def interp(k: int) -> float:              # parabolic interpolation on dB
        a, b, c = db[k - 1], db[k], db[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        return float((k + d) * sr / n)

    strongest = peaks[np.argmax(db[peaks])]
    strong = peaks[db[peaks] >= db[strongest] - within_db]
    return interp(int(strong.min())), interp(int(strongest))


def cents(a: float, b: float) -> float:
    return 1200.0 * math.log2(a / b)


def compare_estimates(f0: float, peak: float) -> str:
    """Flag samples whose perceived pitch may not be the lowest partial."""
    if math.isnan(f0):
        return "NO_F0"
    d = cents(peak, f0)
    if abs(d) <= 30:
        return ""
    ratio = peak / f0
    if abs(1200 * math.log2(ratio / round(ratio))) <= 30 and round(ratio) == 2:
        return "STRONGEST_IS_OCTAVE"          # usually fine; check by ear
    return f"STRONGEST_IS_PARTIAL_x{ratio:.2f}"  # listen: which one is the pitch?


def write_manifest(samples: list[SampleInfo], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["laras", "instrument", "path", "degree", "octave", "layer",
                    "f0_hz", "peak_hz", "flag", "f0_hz_override"])
        for s in samples:
            w.writerow([s.laras, s.instrument, s.path, s.degree, s.octave, s.layer,
                        f"{s.f0_hz:.3f}", f"{s.peak_hz:.3f}", s.flag, ""])


def read_manifest(path: Path) -> list[SampleInfo]:
    """Reads the CSV; a filled-in f0_hz_override column wins."""
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            f0 = float(r["f0_hz_override"]) if r.get("f0_hz_override") else float(r["f0_hz"])
            out.append(SampleInfo(r["laras"], r["instrument"], r["path"], int(r["degree"]),
                                  int(r["octave"]), int(r["layer"]), f0,
                                  float(r["peak_hz"]), r["flag"]))
    return out


# ---------------------------------------------------------------------------
# Tuning tables (one per ensemble)
# ---------------------------------------------------------------------------

def tuning_table(samples: list[SampleInfo]) -> dict[str, dict[str, float]]:
    """{"laras/instrument": {"degree:octave": Hz}} — median over layers."""
    acc: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        if math.isnan(s.f0_hz):
            continue
        acc.setdefault(f"{s.laras}/{s.instrument}", {}).setdefault(f"{s.degree}:{s.octave}", []).append(s.f0_hz)
    return {i: {k: float(np.median(v)) for k, v in d.items()} for i, d in acc.items()}


def hz_to_degree(hz: float, table: dict[str, float]) -> tuple[str, float]:
    """Nearest key of one instrument -> ("degree:octave", cents error)."""
    best = min(table.items(), key=lambda kv: abs(cents(hz, kv[1])))
    return best[0], cents(hz, best[1])


def hz_to_degree_any(hz: float, tables: dict[str, dict[str, float]]) -> tuple[str, str, float]:
    """Nearest key over all instruments -> (instrument, "degree:octave", cents)."""
    best = None
    for inst, t in tables.items():
        key, c = hz_to_degree(hz, t)
        if best is None or abs(c) < abs(best[2]):
            best = (inst, key, c)
    return best
