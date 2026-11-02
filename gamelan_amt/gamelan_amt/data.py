"""Chunk dataset: 5-s audio windows + target token sequences.

On-disk format (synthetic and real data alike):

    <root>/<split>/<piece_id>.wav    mono, 16 kHz (PCM_16 is fine)
    <root>/<split>/<piece_id>.json   {"notes": [Note.to_json(), ...], "meta": {...}}

Audio is read lazily per chunk with soundfile (start/stop), so RAM use does
not grow with dataset size. This is ordinary practice, not a research
contribution.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from gamelan_amt.tokenize import SEGMENT_SECONDS, Note, chunk_starts, encode_chunk, trim_overlaps
from gamelan_amt.vocab import ExtendedVocab

SR = 16000
SEG = int(SEGMENT_SECONDS * SR)


def load_piece_labels(json_path: Path) -> list[Note]:
    d = json.loads(json_path.read_text())
    return [Note.from_json(n) for n in d["notes"]]


def list_pieces(split_dir: Path) -> list[tuple[Path, Path]]:
    out = []
    for wav in sorted(split_dir.glob("*.wav")):
        js = wav.with_suffix(".json")
        if js.exists():
            out.append((wav, js))
    return out


class ChunkDataset(Dataset):
    def __init__(self, split_dir: Path, vocab: ExtendedVocab, train: bool,
                 max_tokens: int = 1800, jitter_seconds: float = 2.5,
                 instruments_allowed: set[str] | None = None):
        self.vocab = vocab
        self.train = train
        self.max_tokens = max_tokens
        self.jitter = jitter_seconds if train else 0.0
        self.items: list[tuple[int, float]] = []
        self.pieces: list[tuple[Path, list[Note], float]] = []
        for wav, js in list_pieces(split_dir):
            notes = load_piece_labels(js)
            if instruments_allowed is not None:
                notes = [n for n in notes if n.instrument in instruments_allowed]
            notes = trim_overlaps(notes, vocab)
            dur = sf.info(str(wav)).duration
            pid = len(self.pieces)
            self.pieces.append((wav, notes, dur))
            for s in chunk_starts(dur):
                self.items.append((pid, s))
        if not self.items:
            raise FileNotFoundError(f"no .wav/.json pairs in {split_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        pid, start = self.items[i]
        wav_path, notes, dur = self.pieces[pid]
        if self.jitter:
            start = min(max(0.0, start + random.uniform(-self.jitter, self.jitter)),
                        max(0.0, dur - 1.0))
            start = round(start * 100) / 100          # stay on the 10 ms grid
        a = int(start * SR)
        y, sr = sf.read(str(wav_path), start=a, stop=a + SEG, dtype="float32",
                        always_2d=True)
        assert sr == SR, f"{wav_path}: resample to 16 kHz first"
        y = y.mean(axis=1)
        if len(y) < SEG:
            y = np.pad(y, (0, SEG - len(y)))
        if self.train:
            y = y * random.uniform(0.5, 1.2)
        tokens = encode_chunk(notes, start, self.vocab)
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]      # rare; no EOS -> model learns to continue
        return torch.from_numpy(y).unsqueeze(0), torch.tensor(tokens, dtype=torch.long)


def make_collate(vocab: ExtendedVocab):
    def collate(batch):
        wavs = torch.stack([b[0] for b in batch])               # [B, 1, SEG]
        seqs = [torch.cat([torch.tensor([vocab.bos_id]), b[1]]) for b in batch]
        L = max(len(s) for s in seqs) - 1
        inp = torch.full((len(seqs), L), vocab.pad_id, dtype=torch.long)
        tgt = torch.full((len(seqs), L), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            inp[i, : len(s) - 1] = s[:-1]
            tgt[i, : len(s) - 1] = s[1:]
        return wavs, inp, tgt
    return collate
