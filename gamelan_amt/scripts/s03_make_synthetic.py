"""Step 4: compose + render the synthetic training set.

    python scripts/s03_make_synthetic.py --manifest samples/manifest.csv \
        --out data/synth --pieces 800 --workers 8

Each piece is 30-90 s, so 800 pieces is roughly 13 hours. Splits are by
PIECE (train 90 % / val 5 % / test 5 %), never by chunk, so neighbouring
chunks of one piece never end up in different splits.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

from gamelan_amt.synth.compose import compose_piece
from gamelan_amt.synth.render import SR, SampleBank, render_piece
from gamelan_amt.tuning import read_manifest
from gamelan_amt.vocab import GAMELAN_INSTRUMENTS

_BANK = None


def _init(manifest: str):
    global _BANK
    _BANK = SampleBank(read_manifest(Path(manifest)))
    unknown = {k[1] for k in _BANK.by_key} - set(GAMELAN_INSTRUMENTS)
    if unknown:
        raise SystemExit(f"instrument folders not in vocab.GAMELAN_INSTRUMENTS: {unknown}")


def split_of(i: int, n: int) -> str:
    f = i / n
    return "train" if f < 0.90 else ("val" if f < 0.95 else "test")


def _make(job):
    i, n, out, rir_dir = job
    seed = 10_000 + i
    laras_list = _BANK.laras_list()
    laras = laras_list[i % len(laras_list)]
    specs = compose_piece(_BANK.keys_for(laras), laras, seed)
    rirs = sorted(Path(rir_dir).glob("*.wav")) if rir_dir else None
    audio, notes, meta = render_piece(specs, laras, _BANK, seed, rirs)
    d = Path(out) / split_of(i, n)
    d.mkdir(parents=True, exist_ok=True)
    pid = f"synth_{i:05d}"
    sf.write(str(d / f"{pid}.wav"), audio, SR, subtype="PCM_16")
    (d / f"{pid}.json").write_text(json.dumps(
        {"notes": [x.to_json() | {"degree": x.extra["degree"], "octave": x.extra["octave"]}
                   for x in notes], "meta": meta}))
    return len(audio) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pieces", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rir-dir", default=None, help="folder of real room impulse responses")
    args = ap.parse_args()
    jobs = [(i, args.pieces, args.out, args.rir_dir) for i in range(args.pieces)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.manifest,)) as ex:
        secs = sum(ex.map(_make, jobs, chunksize=4))
    print(f"rendered {args.pieces} pieces, {secs/3600:.1f} h of audio -> {args.out}")


if __name__ == "__main__":
    main()
