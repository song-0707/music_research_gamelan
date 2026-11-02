"""Full-recording transcription with the adapted (or vanilla) model.

Reuses MuScriptor's own chunk loop (`TranscriptionModel._generate_token_stream`:
5-s chunks, tie prologue teacher-forced from the previous chunk), and swaps
the decoder for CentAwareDecoder so cent tokens become Hz. Pinned to
muscriptor==0.4.0 because it calls that private method.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.transcription_model import TranscriptionModel

from gamelan_amt.tokenize import SEGMENT_SECONDS, CentAwareDecoder, Note
from gamelan_amt.vocab import GAMELAN_PROGRAM_BASE, ExtendedVocab, pitch_cents_to_hz

SR = 16000


def _to_wav(tm: TranscriptionModel, audio) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        audio = (torch.from_numpy(audio), SR)
    if isinstance(audio, tuple):
        return tm._load_wav(audio[0], audio[1])
    return tm._load_wav(audio, None)


def gamelan_only_forbidden(vocab: ExtendedVocab) -> list[int]:
    """Forbid stock instruments and drums so output uses gamelan programs only."""
    return [i for i, e in enumerate(vocab.events)
            if (e.type == "program" and e.value < GAMELAN_PROGRAM_BASE)
            or e.type == "drum"]


@torch.inference_mode()
def transcribe_adapted(tm: TranscriptionModel, vocab: ExtendedVocab, audio,
                       gamelan_only: bool = True, max_gen_len: int = 2000,
                       quiet: bool = True) -> list[Note]:
    wav = _to_wav(tm, audio)
    seg = int(SEGMENT_SECONDS * SR)
    n = math.ceil(wav.shape[-1] / seg)
    conds, seeks = [], []
    for i in range(n):
        chunk = wav[:, i * seg:(i + 1) * seg]
        if chunk.shape[-1] < seg:
            chunk = F.pad(chunk, (0, seg - chunk.shape[-1]))
        conds.append(tm._build_conditions(chunk, None)[0])
        seeks.append(i * SEGMENT_SECONDS)
    forbidden = None
    if gamelan_only:
        forbidden = torch.tensor(gamelan_only_forbidden(vocab), device=tm._device)

    dec = CentAwareDecoder(vocab)
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    err = contextlib.redirect_stderr(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx, err:
        for item in tm._generate_token_stream(
                conds, seeks, batch_size=1, max_gen_len=max_gen_len,
                use_sampling=False, temperature=1.0, cfg_coef=1.0,
                no_eos_is_ok=True, prelude_forcing=True, beam_size=1,
                forbidden_tokens=forbidden):
            dec.feed(item)
    return dec.finish()


@torch.inference_mode()
def transcribe_vanilla(tm: TranscriptionModel, audio, quiet: bool = True) -> list[Note]:
    """Stock MuScriptor: 12-TET pitches, stock instrument names."""
    wav = _to_wav(tm, audio)
    notes: list[Note] = []
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    err = contextlib.redirect_stderr(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx, err:
        for ev in tm.transcribe((wav.cpu(), SR)):
            if isinstance(ev, NoteEndEvent):
                s: NoteStartEvent = ev.start_event
                notes.append(Note(s.instrument, s.start_time,
                                  max(ev.end_time, s.start_time + 0.01),
                                  pitch_cents_to_hz(s.pitch, 0.0)))
    notes.sort(key=lambda n: (n.onset, n.hz))
    return notes
