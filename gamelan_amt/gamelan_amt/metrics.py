"""Transcription metrics in Hz (so microtonal pitch is scored directly).

Main metric: onset F1 with pitch (50 ms, 50 cents) — the mir_eval default.
Because 50 cents is as loose as rounding to the nearest semitone, we also
report a 20-cent tolerance and the mean cent error of matched notes; only
those show whether cent tokens help. Offsets are secondary: gamelan notes
often decay or are hand-damped without a clear end.
"""

from __future__ import annotations

import numpy as np
import mir_eval

from gamelan_amt.tokenize import Note


def _arrays(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n.onset, max(n.offset, n.onset + 0.01)] for n in notes])
    hz = np.array([n.hz for n in notes])
    return iv, hz


def _prf(tp: int, n_est: int, n_ref: int) -> tuple[float, float, float]:
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(ref: list[Note], est: list[Note], onset_tol: float = 0.05,
             instrument_map: dict[str, str] | None = None) -> dict[str, float]:
    ri, rp = _arrays(ref)
    ei, ep = _arrays(est)
    out: dict[str, float] = {"n_ref": len(ref), "n_est": len(est)}

    if len(ref) == 0 or len(est) == 0:
        for k in ("onset_f1", "note_f1_50c", "note_f1_20c", "note_offset_f1_50c",
                  "multi_f1_50c"):
            out[k] = 0.0
        out["cent_mae"] = float("nan")
        out["octave_error_rate"] = float("nan")
        return out

    # 1) onset only, pitch ignored
    out["onset_p"], out["onset_r"], out["onset_f1"] = \
        mir_eval.transcription.onset_precision_recall_f1(ri, ei, onset_tolerance=onset_tol)

    # 2) onset + pitch, two tolerances
    for c in (50, 20):
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ri, rp, ei, ep, onset_tolerance=onset_tol, pitch_tolerance=float(c),
            offset_ratio=None)
        out[f"note_p_{c}c"], out[f"note_r_{c}c"], out[f"note_f1_{c}c"] = p, r, f

    # 3) with offsets (secondary)
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=onset_tol, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05)
    out["note_offset_f1_50c"] = f

    # 4) cent error on matched notes
    matches = mir_eval.transcription.match_notes(
        ri, rp, ei, ep, onset_tolerance=onset_tol, pitch_tolerance=50.0,
        offset_ratio=None)
    if matches:
        errs = [abs(1200 * np.log2(ep[j] / rp[i])) for i, j in matches]
        out["cent_mae"] = float(np.mean(errs))
        out["cent_median"] = float(np.median(errs))
    else:
        out["cent_mae"] = out["cent_median"] = float("nan")

    # 5) octave errors: unmatched estimates that WOULD match an unmatched
    #    reference note one octave away. Real octave doublings (e.g. peking
    #    over saron) are in the reference, so they are matched, not counted.
    mi = {i for i, _ in matches}
    mj = {j for _, j in matches}
    ur = [i for i in range(len(ref)) if i not in mi]
    ue = [j for j in range(len(est)) if j not in mj]
    oct_err = 0
    if ur and ue:
        for factor in (2.0, 0.5):
            m2 = mir_eval.transcription.match_notes(
                ri[ur], rp[ur], ei[ue], ep[ue] * factor, onset_tolerance=onset_tol,
                pitch_tolerance=50.0, offset_ratio=None)
            oct_err += len(m2)
    out["octave_error_rate"] = oct_err / len(est)

    # 6) instrument-aware ("multi") F1: a note counts only with the right instrument
    imap = instrument_map or {}
    name = lambda n: imap.get(n.instrument, n.instrument)  # noqa: E731
    tp = 0
    for inst in {name(n) for n in ref} | {name(n) for n in est}:
        r_i = [n for n in ref if name(n) == inst]
        e_i = [n for n in est if name(n) == inst]
        if r_i and e_i:
            a, b = _arrays(r_i)
            c, d = _arrays(e_i)
            tp += len(mir_eval.transcription.match_notes(
                a, b, c, d, onset_tolerance=onset_tol, pitch_tolerance=50.0,
                offset_ratio=None))
    out["multi_p_50c"], out["multi_r_50c"], out["multi_f1_50c"] = _prf(tp, len(est), len(ref))
    return out


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean over pieces (macro average), ignoring NaNs."""
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out
