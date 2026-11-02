"""Step 6: evaluate models on a test split.

    python scripts/s05_evaluate.py --data data/synth/test \
        --system vanilla:medium \
        --system lora_cents:runs/lora_cents/best \
        --system lora_12tet:runs/lora_12tet/best \
        --out results/synth_test

    # zero-shot on the Malay recordings: collapse instrument names to families
    python scripts/s05_evaluate.py --data data/utm_malay/test \
        --system lora_cents:runs/lora_cents/best \
        --instrument-map configs/malay_instrument_map.json --out results/utm_zeroshot

Writes per-piece CSV + summary.json per system, and each system's notes
(JSON) so results can be re-scored without re-running the model.
Instrument-agnostic metrics (onset_f1, note_f1_*, cent_mae) are the fair
comparison against vanilla, whose instrument names are Western.
"""

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf
import torch

from gamelan_amt.data import list_pieces, load_piece_labels
from gamelan_amt.metrics import aggregate, evaluate
from gamelan_amt.model import load_adapter, load_base
from gamelan_amt.transcribe import transcribe_adapted, transcribe_vanilla

KEYS = ["onset_f1", "note_f1_50c", "note_f1_20c", "cent_mae", "octave_error_rate",
        "multi_f1_50c", "note_offset_f1_50c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--system", action="append", required=True,
                    help="name:vanilla-size  or  name:adapter_dir  (vanilla if name starts with 'vanilla')")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instrument-map", type=Path, default=None)
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    imap = json.loads(args.instrument_map.read_text()) if args.instrument_map else None
    pieces = list_pieces(args.data)
    summary = {}

    for spec in args.system:
        name, where = spec.split(":", 1)
        if name.startswith("vanilla"):
            tm = load_base(where, args.device)
            run = lambda audio: transcribe_vanilla(tm, audio)  # noqa: E731
        else:
            tm, lm, vocab, _ = load_adapter(Path(where), args.device)
            run = lambda audio: transcribe_adapted(tm, vocab, audio)  # noqa: E731

        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for wav, js in pieces:
            ref = load_piece_labels(js)
            y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            est = run((torch.from_numpy(y.mean(axis=1)), sr))
            m = evaluate(ref, est, onset_tol=args.onset_tol, instrument_map=imap)
            m["piece"] = wav.stem
            rows.append(m)
            (d / f"{wav.stem}.notes.json").write_text(
                json.dumps({"notes": [n.to_json() for n in est]}))
            print(f"[{name}] {wav.stem}: " + "  ".join(f"{k}={m[k]:.3f}" for k in KEYS))
        with (d / "per_piece.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["piece"] + sorted(k for k in rows[0] if k != "piece"))
            w.writeheader()
            w.writerows(rows)
        agg = aggregate([{k: v for k, v in r.items() if k != "piece"} for r in rows])
        (d / "summary.json").write_text(json.dumps(agg, indent=2))
        summary[name] = agg
        del tm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "system".ljust(20) + "".join(k[:12].rjust(13) for k in KEYS))
    for name, agg in summary.items():
        print(name.ljust(20) + "".join(f"{agg.get(k, float('nan')):13.3f}" for k in KEYS))
    (args.out / "summary_all.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
