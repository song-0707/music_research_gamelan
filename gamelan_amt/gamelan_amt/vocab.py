"""Extended token vocabulary: MuScriptor's 1393 tokens + gamelan tokens.

MuScriptor's decode table (muscriptor.tokenizer.notes.build_event_vocab) is:

    PAD, EOS, UNK | shift 0..1000 | pitch 0..127 | velocity 0..1 | tie
    | program 0..129 | drum 0..127                      -> 1393 tokens

The medium/large checkpoints have card=1395 (two reserved, always-masked
slots); small has card=1393. We APPEND new tokens after `card`, so every
original token id keeps its meaning, and the BOS token (id == card in
LMModel) moves to the new last index (handled in model.py).

New tokens
----------
* ("cent", -50..+50 step 10)   11 tokens, emitted right after every ONSET
                                pitch token of a pitched instrument.
* ("program", 200 + i)          one per gamelan instrument. Reusing the
                                "program" type means MuScriptor's own decoder
                                state machine (OpenNoteTracker) handles them
                                with no changes.

There are deliberately NO scale-degree tokens: a degree is a function of
(pitch, cents, the ensemble's measured tuning) and is derived after
decoding (see tuning.py). Putting it in the vocabulary would duplicate
information and tie the model to one ensemble's tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

from muscriptor.tokenizer.notes import Event, build_event_vocab

MAX_SHIFT_STEPS = 1001  # same as MuScriptor (10 ms ticks, 10 s max)
FRAME_RATE = 100        # ticks per second

CENT_STEP = 10
CENT_VALUES = list(range(-50, 51, CENT_STEP))  # 11 values

GAMELAN_PROGRAM_BASE = 200

# Edit this list for your ensemble. Order fixes the token ids, so do not
# reorder after training has started.
GAMELAN_INSTRUMENTS: list[str] = [
    "saron_demung",
    "saron_barung",
    "saron_peking",
    "slenthem",
    "gender",
    "bonang_barung",
    "bonang_penerus",
    "gambang",
    "kenong",
    "kethuk",
    "kempul",
    "gong",
]

# MuScriptor program token used to initialise each new instrument token
# (program 8 = representative of the "chromatic_percussion" group, i.e.
# celesta/glockenspiel/vibraphone/marimba/xylophone/tubular bells).
INIT_PROGRAM_FOR_NEW_INSTRUMENTS = 8


@dataclass(frozen=True)
class ExtendedVocab:
    base_card: int                 # card of the loaded checkpoint (1393/1395)
    events: list[Event]            # full decode table, index -> Event
    use_cents: bool

    @property
    def num_tokens(self) -> int:   # new "card" (excludes BOS)
        return len(self.events)

    @property
    def bos_id(self) -> int:
        return self.num_tokens

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 1

    def index(self) -> dict[tuple[str, int], int]:
        return {(e.type, e.value): i for i, e in enumerate(self.events)}

    def new_token_ids(self) -> list[int]:
        return list(range(self.base_card, self.num_tokens))

    def program_of(self, instrument: str) -> int:
        return GAMELAN_PROGRAM_BASE + GAMELAN_INSTRUMENTS.index(instrument)

    def instrument_of(self, program: int) -> str:
        i = program - GAMELAN_PROGRAM_BASE
        if 0 <= i < len(GAMELAN_INSTRUMENTS):
            return GAMELAN_INSTRUMENTS[i]
        return f"program_{program}"


def build_extended_vocab(base_card: int, use_cents: bool = True) -> ExtendedVocab:
    """Return the extended decode table.

    `use_cents=False` builds the 12-TET ablation vocabulary (new instrument
    tokens only), so the two models differ in exactly one thing.
    """
    events = build_event_vocab(MAX_SHIFT_STEPS)
    assert len(events) == 1393, len(events)
    # Pad the reserved slots of the checkpoint (1393..base_card-1) so ids align.
    for k in range(base_card - len(events)):
        events.append(Event(type="reserved", value=k))
    if use_cents:
        events += [Event("cent", c) for c in CENT_VALUES]
    events += [
        Event("program", GAMELAN_PROGRAM_BASE + i)
        for i in range(len(GAMELAN_INSTRUMENTS))
    ]
    return ExtendedVocab(base_card=base_card, events=events, use_cents=use_cents)


def quantize_cents(cents: float) -> int:
    """Snap a deviation in [-50, 50] to the nearest 10-cent token value."""
    q = int(round(cents / CENT_STEP)) * CENT_STEP
    return max(-50, min(50, q))


def hz_to_pitch_cents(hz: float) -> tuple[int, float]:
    """Hz -> (nearest MIDI pitch, deviation in cents, in [-50, 50])."""
    import math

    midi = 69.0 + 12.0 * math.log2(hz / 440.0)
    p = int(round(midi))
    return p, (midi - p) * 100.0


def pitch_cents_to_hz(pitch: int, cents: float) -> float:
    """Equation (3) of the proposal."""
    return 440.0 * 2.0 ** ((pitch - 69 + cents / 100.0) / 12.0)
