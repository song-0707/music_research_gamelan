"""Run: python -m pytest -q tests/   (no weights or GPU needed)

Uses MuScriptor's own model builder with a tiny random config, so every
code path (vocab surgery, LoRA, loss, generation, decoding) is exercised.
"""

import random

import numpy as np
import torch

from muscriptor.events import ChunkBoundary
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.transcription_model import TranscriptionModel, _build_model, _ModelConfig

from gamelan_amt.data import make_collate
from gamelan_amt.metrics import evaluate
from gamelan_amt.model import (AdaptConfig, build_adapted, count_parameters, lm_loss,
                               merge_lora, enable_gradient_checkpointing)
from gamelan_amt.tokenize import (CentAwareDecoder, Note, chunk_starts, decode_chunks,
                                  encode_chunk, trim_overlaps)
from gamelan_amt.transcribe import transcribe_adapted
from gamelan_amt.vocab import (GAMELAN_INSTRUMENTS, build_extended_vocab,
                               hz_to_pitch_cents, pitch_cents_to_hz, quantize_cents)


def tiny_tm(card=1395) -> TranscriptionModel:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    lm = _build_model(dev, _ModelConfig(dim=64, num_heads=4, num_layers=2, card=card))
    lm.eval()
    tok = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
    return TranscriptionModel(lm, tok, dev)


def random_notes(seed=0, dur=17.0, n=120):
    rng = random.Random(seed)
    notes = []
    for _ in range(n):
        on = round(rng.uniform(0, dur - 0.5), 3)
        inst = rng.choice(GAMELAN_INSTRUMENTS)
        hz = 440 * 2 ** (rng.uniform(-24, 24) / 12)
        notes.append(Note(inst, on, on + rng.uniform(0.05, 3.0), hz))
    return notes


def _quantized(n: Note):
    # +50 c on pitch p and -50 c on p+1 are the same frequency, so compare
    # on one grid: pitch*100 + cents (10-cent steps), plus nearest semitone.
    p, c = hz_to_pitch_cents(n.hz)
    return (n.instrument, round(n.onset * 100), round((p * 100 + quantize_cents(c)) / 100),
            p * 100 + quantize_cents(c))


def test_encode_decode_roundtrip():
    """Every onset survives chunking + tie prologues with its pitch & cents."""
    for use_cents in (True, False):
        vocab = build_extended_vocab(1395, use_cents=use_cents)
        notes = trim_overlaps(random_notes(), vocab)
        starts = chunk_starts(17.0)
        chunks = [encode_chunk(notes, s, vocab) for s in starts]
        dec = decode_chunks(chunks, starts, vocab)
        if use_cents:
            want = sorted(_quantized(n)[:2] + _quantized(n)[3:] for n in notes)
            got = sorted(_quantized(n)[:2] + _quantized(n)[3:] for n in dec)
        else:  # 12-TET: only the nearest semitone survives
            want = sorted((n.instrument, round(n.onset * 100), hz_to_pitch_cents(n.hz)[0])
                          for n in notes)
            got = sorted((n.instrument, round(n.onset * 100), hz_to_pitch_cents(n.hz)[0])
                         for n in dec)
        assert got == want
        # offsets inside the recording are reproduced to the 10 ms tick
        ref_off = {(n.instrument, round(n.onset * 100), hz_to_pitch_cents(n.hz)[0]):
                   round(n.offset * 100) for n in notes if n.offset < 17.0}
        for n in dec:
            k = (n.instrument, round(n.onset * 100), hz_to_pitch_cents(n.hz)[0])
            if k in ref_off:
                assert round(n.offset * 100) == ref_off[k]


def test_encoder_state_matches_model_output():
    """Conventions seen in the pretrained model's own output (scripts/s01_*)."""
    vocab = build_extended_vocab(1395, use_cents=False)
    notes = [Note("guitar", 4.0, 5.5, 440.0, program=24),
             Note("drums", 5.5, 5.51, 440.0, program=128, is_drum=True),
             Note("guitar", 5.6, 6.0, 440.0, program=24)]
    tokens = encode_chunk(notes, 5.0, vocab)[:-1]
    assert [f"{vocab.events[t].type}:{vocab.events[t].value}" for t in tokens] == [
        "program:24", "pitch:69", "tie:0",
        "shift:50", "velocity:0", "pitch:69",   # program state survives the tie
        "velocity:1", "drum:69",                # a drum sets velocity to 1...
        "shift:60", "pitch:69",                 # ...which then needs no repeat
        "shift:100", "velocity:0", "pitch:69",
    ]


def test_hz_math():
    for hz in (98.0, 261.6, 300.0, 1234.5):
        p, c = hz_to_pitch_cents(hz)
        assert abs(pitch_cents_to_hz(p, c) - hz) < 1e-6
        assert -50 <= c <= 50


def test_surgery_lora_loss_generate():
    tm = tiny_tm()
    lm = tm._model
    base_out = lm.linear.weight.detach().clone()
    cfg = AdaptConfig(base_model="tiny", lora_rank=4, lora_alpha=8,
                      lora_targets=["q", "v", "out"])
    tm, lm, vocab = build_adapted(cfg, "cpu", tm=tm)
    assert lm.card == vocab.num_tokens and lm.initial_token_id == vocab.bos_id
    assert torch.equal(lm.linear.weight[: base_out.shape[0]], base_out)
    tr, tot = count_parameters(lm)
    assert 0 < tr < tot

    # LoRA starts as an exact no-op (B = 0)
    attn = lm.transformer.layers[0].self_attn
    W0 = attn.parametrizations.in_proj_weight.original
    assert torch.allclose(attn.in_proj_weight, W0)

    # one training step: loss finite, grads only on trainable params
    notes = trim_overlaps(random_notes(dur=5.0, n=15), vocab)
    toks = torch.tensor(encode_chunk(notes, 0.0, vocab))
    wavs, inp, tgt = make_collate(vocab)([(torch.randn(1, 80000) * 0.1, toks)] * 2)
    lm.train()
    enable_gradient_checkpointing(lm)
    loss = lm_loss(lm, wavs, inp, tgt)
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in lm.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n
    assert lm.emb.new.grad is not None and lm.emb.new.grad.abs().sum() > 0
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-2)
    opt.step()
    # frozen weights untouched, LoRA is now active
    assert torch.equal(attn.parametrizations.in_proj_weight.original, W0)
    assert not torch.allclose(attn.in_proj_weight, W0)

    # generation through MuScriptor's chunk loop with the extended vocab
    lm.eval()
    est = transcribe_adapted(tm, vocab, (torch.randn(1, 16000 * 7) * 0.1, 16000),
                             max_gen_len=60)
    assert isinstance(est, list)

    # merging LoRA gives the same logits
    cond_wav = torch.randn(1, 1, 80000) * 0.1
    from gamelan_amt.model import condition_tensors
    x = inp[:1]
    with torch.no_grad():
        a = lm(x, condition_tensors(lm, cond_wav), first_step=True)
        merge_lora(lm)
        b = lm(x, condition_tensors(lm, cond_wav), first_step=True)
    assert torch.allclose(a, b, atol=1e-4)


def test_small_card_mask():
    """small has card=1393: new tokens start at 1393 and must NOT be masked."""
    tm = tiny_tm(card=1393)
    tm, lm, vocab = build_adapted(AdaptConfig(base_model="tiny", lora_rank=2), "cpu", tm=tm)
    assert lm._masked_token_ids is None
    assert vocab.new_token_ids()[0] == 1393


def test_metrics():
    ref = [Note("saron_barung", 1.0, 1.5, 300.0),
           Note("saron_peking", 1.0, 1.2, 600.0),    # real octave doubling
           Note("kenong", 2.0, 4.0, 200.0),
           Note("saron_barung", 3.0, 3.5, 250.0)]
    est = [Note("saron_barung", 1.01, 1.5, 300.0 * 2 ** (30 / 1200)),  # 30 c sharp
           Note("saron_peking", 1.0, 1.2, 600.0),
           Note("kenong", 2.02, 4.0, 200.0),
           Note("saron_barung", 3.0, 3.5, 500.0)]                       # octave error
    m = evaluate(ref, est)
    assert abs(m["note_f1_50c"] - 3 / 4) < 1e-9        # 30 c passes at 50 c ...
    assert abs(m["note_f1_20c"] - 2 / 4) < 1e-9        # ... but not at 20 c
    assert abs(m["cent_mae"] - 10.0) < 0.1             # (30 + 0 + 0) / 3
    assert abs(m["octave_error_rate"] - 1 / 4) < 1e-9  # doubling is NOT an error
