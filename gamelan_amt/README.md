# gamelan_amt: implementation plan and code

LoRA adaptation of MuScriptor for gamelan transcription with microtonal
(cent) output. Each step below is one task. Do them in order: each one
depends on the step before.

Everything in `tests/` and the "dry run" at the end works **without**
MuScriptor weights or a GPU, so you can check the pipeline on a laptop first.

---

## Step 0. Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10-3.12
pip install -e ".[dev]"
python -m pytest -q tests/                              # 6 tests should pass
```

Then accept the weights licence (CC BY-NC 4.0, non-commercial) on the
Hugging Face pages of `MuScriptor/muscriptor-small` and `-medium`, and log in:

```bash
hf auth login
```

---

## Step 1. Look inside the real model

```bash
python scripts/s00_inspect_model.py --model medium
```

**Why:** the Methodology section must describe what actually exists.
What you will see (muscriptor 0.4.0):

| Item | Value |
|---|---|
| Architecture | decoder-only Transformer; **no cross-attention** |
| Audio input | log-mel prefix prepended to the token sequence |
| Front-end | 16 kHz, n_fft 2048 (128 ms), hop 160 (10 ms), **512** HTK mel bins, 5.0 s chunks → 501 frames |
| Attention weights | one fused `self_attn.in_proj_weight` [3D × D] per layer (q, k, v stacked) + `self_attn.out_proj` |
| Vocabulary | 1393 tokens (card 1395 on medium/large, 2 reserved) |
| Sizes | small 103M/14 layers/D768 · medium 307M/24/D1024 · large 1.4B/48/D1536 |

There is no `q_proj` or `v_proj` module, so the PEFT library cannot target
them. `gamelan_amt/model.py` adds LoRA to the q and v row-blocks of the fused
weight with `torch.nn.utils.parametrize`.

---

## Step 2. Confirm the token format

MuScriptor ships no training code. `gamelan_amt/tokenize.py` writes target
sequences in the format its decoder reads. Check that against the real model
once, on any Western recording:

```bash
python scripts/s01_check_token_format.py some_song.wav --model medium
```

You want overall agreement above ~0.95. If a convention differs (the script
prints the first difference per chunk), adjust `encode_chunk()` to match.

---

## Step 3. Baseline: vanilla MuScriptor on gamelan

Put 10–20 real gamelan clips in a folder and listen to the output:

```bash
muscriptor transcribe clip.wav --model medium
```

Write down the failure types: wrong pitch, octave errors, missed onsets,
wrong instrument names. These are the problems the thesis claims to fix.

---

## Step 4. Get one-shot samples and measure their real pitch

Folder layout (one strike per key, one or more velocity layers):

```
samples/slendro/saron_barung/6_1_v1.wav     # degree 6, octave 1, layer 1
samples/pelog/bonang_barung/5_0_v2.wav
```

Allowed instrument names are in `gamelan_amt/vocab.py` (`GAMELAN_INSTRUMENTS`).
Sources: render each key once from your sample library (check its licence
allows publishing rendered audio), or record a real Javanese set.

**Do not use the UTM Malay ensemble's samples for training.** Its recordings
are the zero-shot test set.

```bash
python scripts/s02_measure_tuning.py samples/
```

This writes `samples/manifest.csv` and `samples/tuning.json`. The measured
Hz, not the MIDI key, becomes every training label. Listen to each row that
has a flag. If the perceived pitch is not the measured one, type the right Hz
into `f0_hz_override` and run again with `--tuning-only`.

The measurement uses the lowest strong spectral peak. pYIN was tested and
rejected: on inharmonic bars and kettles it was wrong by 1200–3600 cents,
while the spectral method was within 0.01 cents.

---

## Step 5. Generate synthetic training data

```bash
python scripts/s03_make_synthetic.py --manifest samples/manifest.csv \
    --out data/synth --pieces 800 --workers 8      # ~13 h of audio
```

- `synth/compose.py` writes gamelan-style parts: balungan, peking, bonang
  *mipil*, gender/gambang figuration, and kenong/kempul/gong/kethuk.
- `synth/render.py` mixes the samples and applies augmentation that is
  reflected in the labels: a whole-ensemble tuning shift of ±40 cents, a
  per-instrument shift of ±6 cents, hand damping, room reverb, mic EQ and
  noise.
- Splits are made **by piece** (90/5/5), so chunks of one piece never leak
  between train and test.
- Pass real room impulse responses with `--rir-dir` if you have them.

---

## Step 6. Annotate the real test sets

Follow the protocol at the top of `scripts/prepare_real_recordings.py`:

1. Record one strike of every key of each real ensemble, then run Step 4 on
   those strikes to get that ensemble's `tuning.json`.
2. In Sonic Visualiser, mark onsets labelled `instrument:degree:octave`.
3. Convert the annotations:

```bash
python scripts/prepare_real_recordings.py --audio raw/utm --annotations ann/utm \
    --tuning samples_utm/tuning.json --laras pelog --out data/utm_malay/test
    # --laras: whichever set the ensemble is tuned to
```

Annotators only decide *when* a note starts and *which key* was struck. The
Hz comes from the measured tuning. Budget about 20–30 clips of ~30 s, and
have a second person annotate 20 % of them to report agreement.

The Zenodo field recordings have no note labels, so they cannot train a
supervised model. Use them for listening checks only.

---

## Step 7. Train

Debug with `base_model: small` first, then switch to medium.

```bash
python scripts/s04_train.py --config configs/lora_cents.yaml
sbatch slurm/train.sbatch configs/lora_cents.yaml      # on the HPC
```

- Training resumes automatically from `<out_dir>/last/`.
- It stops itself at `max_hours: 23.5` so a 24-hour job limit never kills it mid-step.
- It prints the real trainable-parameter percentage. Report that number;
  the new embedding and output rows are counted in it.

Each ablation config changes exactly one thing compared with `lora_cents.yaml`:

| Config | Question it answers |
|---|---|
| `lora_cents.yaml` | proposed: LoRA (q, v) + cent tokens + gamelan instrument tokens |
| `lora_12tet.yaml` | do cent tokens help? (same, without cent tokens) |
| `frozen_newtokens.yaml` | is LoRA needed? (only the new token rows train) |
| `lora_cents_melproj.yaml` | does adapting the mel→D projection help on inharmonic spectra? |
| `lora_cents_small.yaml` | LoRA on small (fair comparison for the next row) |
| `full_small.yaml` | LoRA vs full fine-tuning (small model, all weights) |

---

## Step 8. Evaluate

```bash
python scripts/s05_evaluate.py --data data/synth/test --out results/synth \
    --system vanilla:medium \
    --system lora_cents:runs/lora_cents/best \
    --system lora_12tet:runs/lora_12tet/best \
    --system frozen:runs/frozen_newtokens/best
python scripts/s06_baseline_onset_pitch.py --data data/synth/test --out results/synth

# zero-shot Malay test (instrument names collapsed to families)
python scripts/s05_evaluate.py --data data/utm_malay/test --out results/utm \
    --system vanilla:medium --system lora_cents:runs/lora_cents/best \
    --instrument-map configs/malay_instrument_map.json
```

Metrics (`gamelan_amt/metrics.py`, all computed in Hz):

| Metric | Role |
|---|---|
| `onset_f1` | onsets only, pitch ignored |
| `note_f1_50c` | **main**: onset + pitch within 50 cents (mir_eval default) |
| `note_f1_20c` | onset + pitch within 20 cents; 50 cents is as loose as semitone rounding, so this is where cent tokens should show |
| `cent_mae` | mean absolute cent error of matched notes |
| `octave_error_rate` | wrong-octave notes; real octave doublings (e.g. peking over saron) are in the reference and are **not** counted |
| `multi_f1_50c` | instrument must also be right (not fair for vanilla) |
| `note_offset_f1_50c` | secondary: gamelan offsets are often undefined |

Convert decoded Hz back to scale degrees with `tuning.hz_to_degree_any()`
and the ensemble's `tuning.json`.

---

## Dry run (no weights, CPU, about a minute)

```bash
python scripts/dev_make_fake_samples.py --out /tmp/fake
python scripts/s02_measure_tuning.py /tmp/fake
python scripts/s03_make_synthetic.py --manifest /tmp/fake/manifest.csv --out /tmp/synth --pieces 40
# copy configs/lora_cents.yaml, set base_model: tiny-random, data_dir: /tmp/synth, steps: 30
python scripts/s04_train.py --config /tmp/dry.yaml
python scripts/s05_evaluate.py --data /tmp/synth/test --system dry:<out_dir>/best --out /tmp/res
```

The tiny random model will score 0. The point is that every stage runs.

---

## Code map

| File | What it does |
|---|---|
| `gamelan_amt/vocab.py` | extended vocabulary: +11 cent tokens, +1 program token per gamelan instrument; no scale-degree tokens |
| `gamelan_amt/tokenize.py` | notes ↔ tokens (tie prologue per 5-s chunk); `CentAwareDecoder` |
| `gamelan_amt/model.py` | vocabulary surgery (BOS row moved, logits mask fixed), LoRA via parametrization, adapter save/load, training loss |
| `gamelan_amt/tuning.py` | f0 measurement, tuning tables, Hz → degree |
| `gamelan_amt/synth/` | composer and sample renderer |
| `gamelan_amt/data.py` | lazy 5-s chunk dataset |
| `gamelan_amt/transcribe.py` | full-recording inference (reuses MuScriptor's chunk loop) |
| `gamelan_amt/metrics.py` | mir_eval wrappers in Hz |
