"""Make fake metallophone one-shots so the pipeline can be dry-run today.

    python scripts/dev_make_fake_samples.py --out samples_fake

Each "key" is a decaying sum of inharmonic partials (ratios typical of
bars/kettles) at a made-up slendro/pelog tuning. NOT for real training —
only to test s02 -> s03 -> s04 -> s05 end to end before the real samples exist.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
# rough, invented tunings (cents above degree 1)
TUNING = {"slendro": {1: 0, 2: 231, 3: 474, 5: 717, 6: 955},
          "pelog": {1: 0, 2: 120, 3: 270, 4: 540, 5: 670, 6: 785, 7: 950}}
INSTR = {  # instrument: (base Hz of degree 1 octave 0, octaves, partial ratios, decay s)
    "saron_barung": (262.0, [0, 1], [1, 2.76, 5.40], 1.8),
    "saron_peking": (524.0, [0, 1], [1, 2.76, 5.40], 1.2),
    "slenthem": (131.0, [0], [1, 2.9, 5.8], 3.0),
    "bonang_barung": (196.0, [0, 1], [1, 1.52, 2.03, 3.1], 1.5),
    "bonang_penerus": (392.0, [0, 1], [1, 1.52, 2.03, 3.1], 1.0),
    "gambang": (196.0, [0, 1, 2], [1, 3.9, 9.2], 0.5),
    "kenong": (175.0, [0], [1, 1.48, 2.1], 3.0),
    "kempul": (110.0, [0], [1, 1.47, 2.05], 4.0),
    "gong": (45.0, [0], [1, 1.42, 2.6], 7.0),
    "kethuk": (147.0, [0], [1, 1.5], 0.4),
}


def strike(f0, ratios, decay, vel, rng):
    n = int(min(decay * 2.5, 8.0) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k, r in enumerate(ratios):
        y += (0.6 ** k) * np.sin(2 * np.pi * f0 * r * t + rng.uniform(0, 6.28)) * np.exp(-t / (decay / (1 + k)))
    y += 0.3 * vel * rng.standard_normal(n) * np.exp(-t / 0.004)   # mallet click
    return (y / np.abs(y).max() * 0.9).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("samples_fake"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    n = 0
    for laras, tun in TUNING.items():
        for inst, (base, octs, ratios, decay) in INSTR.items():
            d = args.out / laras / inst
            d.mkdir(parents=True, exist_ok=True)
            for deg, c in tun.items():
                for o in octs:
                    f0 = base * 2 ** (o + (c + rng.uniform(-8, 8)) / 1200)
                    for layer, vel in ((1, 0.5), (2, 1.0)):
                        sf.write(str(d / f"{deg}_{o}_v{layer}.wav"),
                                 strike(f0, ratios, decay, vel, rng), SR)
                        n += 1
    print(f"wrote {n} fake samples to {args.out}")


if __name__ == "__main__":
    main()
