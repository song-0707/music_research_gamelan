"""Step 2: check that OUR encoder writes tokens the way the pretrained model does.

    python scripts/s01_check_token_format.py some_western_song.wav --model medium

MuScriptor releases no training code, so tokenize.encode_chunk() was written
from the decoder. This script:
  1. runs the stock model and records its raw token stream per chunk,
  2. decodes it to notes,
  3. re-encodes those notes with encode_chunk(),
  4. compares token by token.
High agreement (> 95 %) means the encoder matches the model's native format.
Disagreements are printed so you can see the convention that differs
(e.g. whether `program` is repeated after every `shift`).
"""

import argparse
import contextlib
import difflib
import io
import math

import torch
import torch.nn.functional as F

from muscriptor.events import ChunkBoundary

from gamelan_amt.model import load_base
from gamelan_amt.tokenize import SEGMENT_SECONDS, Note, CentAwareDecoder, encode_chunk, trim_overlaps
from gamelan_amt.vocab import build_extended_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tm = load_base(args.model, args.device)
    vocab = build_extended_vocab(tm._model.card, use_cents=False)
    wav = tm._load_wav(args.audio, None)
    seg = int(SEGMENT_SECONDS * 16000)
    n = min(args.chunks, math.ceil(wav.shape[-1] / seg))
    conds, seeks = [], []
    for i in range(n):
        c = wav[:, i * seg:(i + 1) * seg]
        c = F.pad(c, (0, seg - c.shape[-1]))
        conds.append(tm._build_conditions(c, None)[0])
        seeks.append(i * SEGMENT_SECONDS)

    raw: list[list[int]] = []
    dec = CentAwareDecoder(vocab)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for item in tm._generate_token_stream(conds, seeks, 1, 2000, False, 1.0, 1.0,
                                              True, True, 1, None):
            if isinstance(item, ChunkBoundary):
                raw.append([])
            elif isinstance(item, int):
                raw[-1].append(item)
            dec.feed(item)
    notes = dec.finish()
    # program numbers are stock GM programs here: keep them as-is
    notes = [Note(n.instrument, n.onset, n.offset, n.hz, program=n.program,
                  is_drum=n.is_drum) for n in notes]
    notes = trim_overlaps(notes, vocab)

    name = lambda t: f"{vocab.events[t].type}:{vocab.events[t].value}"  # noqa: E731
    total = same = 0
    for i, s in enumerate(seeks):
        ours = encode_chunk(notes, s, vocab)[:-1]          # drop EOS (stream has none)
        sm = difflib.SequenceMatcher(a=raw[i], b=ours, autojunk=False)
        same += sum(b.size for b in sm.get_matching_blocks())
        total += max(len(raw[i]), len(ours))
        print(f"chunk {i}: model {len(raw[i])} tokens, ours {len(ours)}, "
              f"similarity {sm.ratio():.3f}")
        for op, a0, a1, b0, b1 in sm.get_opcodes()[:200]:
            if op != "equal":
                print(f"   {op:8s} model={[name(t) for t in raw[i][a0:a1]][:8]} "
                      f"ours={[name(t) for t in ours[b0:b1]][:8]}")
                break
    print(f"\noverall token agreement: {same / max(total, 1):.3f}")


if __name__ == "__main__":
    main()
