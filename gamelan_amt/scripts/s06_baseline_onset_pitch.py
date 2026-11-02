"""Non-neural baseline: onset detection + "new energy" pitch picking.

    python scripts/s06_baseline_onset_pitch.py --data data/synth/test --out results/synth_test

Metallophones have sharp attacks, so a signal-processing baseline can be
surprisingly strong on onsets. For each detected onset, we compare a
10-cent-resolution CQT just after vs. just before the onset; peaks in the
increase are newly struck pitches. Instrument-agnostic, fixed 0.3-s notes,
so compare it on onset_f1 / note_f1_* / cent_mae only.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from gamelan_amt.data import list_pieces, load_piece_labels
from gamelan_amt.metrics import aggregate, evaluate
from gamelan_amt.tokenize import Note

SR = 16000
FMIN = librosa.note_to_hz("C2")
BPO = 120                       # 10 cents per bin
N_BINS = BPO * 6


def transcribe_dsp(y: np.ndarray, max_new: int = 4, rel_db: float = 12.0) -> list[Note]:
    hop = 160
    onsets = librosa.onset.onset_detect(y=y, sr=SR, hop_length=hop, units="frames",
                                        backtrack=False, delta=0.07)
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=hop, fmin=FMIN, n_bins=N_BINS,
                           bins_per_octave=BPO))
    D = librosa.amplitude_to_db(C, ref=np.max)
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BPO)
    notes = []
    for f in onsets:
        before = D[:, max(0, f - 4):max(1, f - 1)].max(axis=1)
        after = D[:, f + 2:f + 8].mean(axis=1) if f + 8 < D.shape[1] else D[:, f:].mean(axis=1)
        gain = after - before
        peaks, props = find_peaks(gain, height=rel_db, distance=BPO // 6)
        if not len(peaks):
            continue
        order = np.argsort(props["peak_heights"])[::-1][:max_new]
        t = f * hop / SR
        for p in peaks[order]:
            if after[p] < -50:              # too quiet in absolute terms
                continue
            notes.append(Note("any", t, t + 0.3, float(freqs[p])))
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = args.out / "dsp_baseline"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for wav, js in list_pieces(args.data):
        y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        est = transcribe_dsp(y.mean(axis=1))
        m = evaluate(load_piece_labels(js), est)
        rows.append(m)
        print(f"{wav.stem}: onset_f1={m['onset_f1']:.3f} note_f1_50c={m['note_f1_50c']:.3f}")
    (d / "summary.json").write_text(json.dumps(aggregate(rows), indent=2))


if __name__ == "__main__":
    main()
