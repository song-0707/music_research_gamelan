"""Step 1: print MuScriptor's real structure (for the Methodology section).

    python scripts/s00_inspect_model.py --model medium

Prints module names, the fused attention weight shapes, vocabulary size,
the audio front-end settings, and the prefix length of a 5-s chunk.
"""

import argparse
import contextlib
import io

import torch

from gamelan_amt.model import condition_tensors, load_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tm = load_base(args.model, args.device)
    lm = tm._model
    total = sum(p.numel() for p in lm.parameters())
    print(f"parameters: {total/1e6:.1f}M   card (vocab incl. reserved): {lm.card}")

    print("\n-- top-level modules")
    for n, m in lm.named_children():
        print(f"  {n:22s} {type(m).__name__}")

    print("\n-- layer 0 parameters (LoRA targets live here)")
    for n, p in lm.transformer.layers[0].named_parameters():
        print(f"  transformer.layers.0.{n:28s} {tuple(p.shape)}")
    print(f"  number of layers: {len(lm.transformer.layers)}")

    mel = lm.condition_provider.conditioners["self_wav"]
    t = mel.mel_spec_transform
    print("\n-- audio front-end (use exactly this; do not re-define it)")
    print(f"  sample rate {mel.sample_rate} Hz, n_fft {t.n_fft} "
          f"({1000*t.n_fft/mel.sample_rate:.0f} ms window), hop {t.hop_length} "
          f"({1000*t.hop_length/mel.sample_rate:.0f} ms), mel bins {mel.dim}, "
          f"power {t.power}, log={mel.log_scale}")

    wav = torch.zeros(1, 1, 80000, device=args.device)
    with contextlib.redirect_stdout(io.StringIO()):
        cond = condition_tensors(lm, wav)
    print("\n-- prefix prepended to the token sequence for one 5-s chunk")
    for k, (emb, _) in cond.items():
        print(f"  {k:18s} {tuple(emb.shape)}")


if __name__ == "__main__":
    main()
