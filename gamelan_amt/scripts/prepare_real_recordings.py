"""Turn real recordings + manual annotations into the dataset format.

Annotation protocol (much cheaper than labelling Hz by hand):
  1. Record one clean strike of every key of the ensemble, name the files
     samples_<ensemble>/<laras>/<instrument>/<degree>_<octave>.wav, and run
     s02_measure_tuning.py on that folder -> tuning.json for THIS ensemble.
  2. In Sonic Visualiser, add a Time Instants layer per recording and place
     a point at every onset, labelled  instrument:degree:octave
     (e.g. "bonang_barung:5:1"). Export the layer as CSV (time,label).
     Optional 3rd column = duration in seconds, if you mark damping.
  3. Run:
       python scripts/prepare_real_recordings.py --audio raw/utm/ \
           --annotations ann/utm/ --tuning samples_utm/tuning.json \
           --laras slendro --out data/utm_malay/test

Annotators only decide WHEN and WHICH KEY; the Hz comes from the measured
tuning, so microtonal labels need no special ear. Have a second person
annotate ~20 % of clips and report onset agreement (inter-annotator F1).
Audio is converted to 16 kHz mono.
"""

import argparse
import csv
import json
from pathlib import Path

import librosa
import soundfile as sf

from gamelan_amt.tokenize import Note

RING = 1.5   # assumed duration when no duration is annotated (onset metrics ignore it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--tuning", type=Path, required=True)
    ap.add_argument("--laras", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tuning = json.loads(args.tuning.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    for ann in sorted(args.annotations.glob("*.csv")):
        audio = next((p for p in args.audio.glob(ann.stem + ".*")), None)
        if audio is None:
            print(f"no audio for {ann.name}; skipped")
            continue
        notes, problems = [], 0
        with ann.open() as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                t = float(row[0])
                label = row[-1].strip() if len(row) >= 2 else ""
                dur = float(row[2]) if len(row) >= 4 and row[2] else None
                try:
                    inst, deg, octv = label.split(":")
                    hz = tuning[f"{args.laras}/{inst}"][f"{deg}:{octv}"]
                except (ValueError, KeyError):
                    problems += 1
                    print(f"  {ann.name}: bad label {label!r} at {t:.3f}s")
                    continue
                notes.append(Note(inst, t, t + (dur or RING), hz,
                                  extra={"degree": int(deg), "octave": int(octv)}))
        y, _ = librosa.load(str(audio), sr=16000, mono=True)
        sf.write(str(args.out / f"{ann.stem}.wav"), y, 16000, subtype="PCM_16")
        (args.out / f"{ann.stem}.json").write_text(json.dumps({
            "notes": [n.to_json() | n.extra for n in sorted(notes, key=lambda n: n.onset)],
            "meta": {"source": str(audio), "laras": args.laras}}))
        print(f"{ann.stem}: {len(notes)} notes, {problems} bad labels")


if __name__ == "__main__":
    main()
