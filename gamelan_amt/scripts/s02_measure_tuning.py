"""Step 3: measure the true pitch of every one-shot sample.

    python scripts/s02_measure_tuning.py samples/ --out samples/manifest.csv

Reads samples/<laras>/<instrument>/<degree>_<octave>[_v<layer>].wav, writes
manifest.csv with the measured f0 and a flag column, plus tuning.json.

Then OPEN manifest.csv and listen to every row with a non-empty flag
(STRONGEST_IS_PARTIAL / NO_F0 matter most). Type the correct Hz in f0_hz_override if needed, and
re-run with --tuning-only to rebuild tuning.json from your corrections.
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

from gamelan_amt.tuning import (compare_estimates, measure_f0, read_manifest,
                                scan_samples, tuning_table, write_manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tuning-only", action="store_true")
    args = ap.parse_args()
    out = args.out or args.samples / "manifest.csv"

    if args.tuning_only:
        samples = read_manifest(out)
    else:
        samples = scan_samples(args.samples)
        for s in samples:
            y, sr = sf.read(s.path, dtype="float32", always_2d=True)
            s.f0_hz, s.peak_hz = measure_f0(y.mean(axis=1), sr)
            s.flag = compare_estimates(s.f0_hz, s.peak_hz)
            print(f"{s.laras:8s} {s.instrument:16s} {s.degree}:{s.octave} v{s.layer} "
                  f"f0={s.f0_hz:8.2f}  strongest={s.peak_hz:8.2f}  {s.flag}")
        write_manifest(samples, out)
        print(f"\nwrote {out}  ({sum(bool(s.flag) for s in samples)} rows to check by ear)")

    table = tuning_table(samples)
    tpath = out.with_name("tuning.json")
    tpath.write_text(json.dumps(table, indent=2))
    print(f"wrote {tpath}")


if __name__ == "__main__":
    main()
