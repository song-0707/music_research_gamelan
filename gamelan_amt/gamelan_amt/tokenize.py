"""Notes <-> MuScriptor token sequences, with optional cent tokens.

MuScriptor ships an inference-only package, so there is no training-side
encoder to import. This encoder is written to produce exactly what
`muscriptor.events.OpenNoteTracker` (the shipped decoder) expects:

    chunk := tie_prologue  events  EOS
    tie_prologue := (program pitch+)*  tie
        notes still sounding at the chunk start, sorted by (program, pitch),
        one program token per run of pitches.
    events := for each tick (10 ms) in time order:
        shift(t)                     absolute tick inside the chunk, only when
                                     the time changes (no shift for tick 0)
        then per note event, sorted (is_drum, program, velocity, pitch):
            [velocity(1)] drum(p)    drum onset (no program; velocity if not 1)
            [program(k)]             only if it changed (state survives the tie)
            [velocity(v)]            only if it changed (1 = onset, 0 = offset)
            pitch(p)
            [cent(c)]                NEW: after every onset pitch, if use_cents

Notes still sounding at the chunk end get no offset event; the next chunk's
tie prologue carries them over. Run scripts/s01_check_token_format.py once
with real weights to confirm this matches the pretrained model's own output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from muscriptor.events import (
    ChunkBoundary,
    OpenNoteTracker,
    ProgressEvent,
    _DrumHit,
    _EndNote,
    _StartNote,
)

from gamelan_amt.vocab import (
    FRAME_RATE,
    ExtendedVocab,
    hz_to_pitch_cents,
    pitch_cents_to_hz,
    quantize_cents,
)

SEGMENT_SECONDS = 5.0
MIN_DURATION = 0.01


@dataclass
class Note:
    instrument: str
    onset: float          # seconds
    offset: float         # seconds
    hz: float             # true (measured) fundamental frequency
    program: int | None = None   # set for stock MuScriptor programs
    is_drum: bool = False
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {"instrument": self.instrument, "onset": self.onset,
             "offset": self.offset, "hz": self.hz}
        if self.program is not None:
            d["program"] = self.program
        if self.is_drum:
            d["is_drum"] = True
        return d

    @staticmethod
    def from_json(d: dict) -> "Note":
        return Note(d["instrument"], float(d["onset"]), float(d["offset"]),
                    float(d["hz"]), d.get("program"), bool(d.get("is_drum", False)))


def _tick(t: float) -> int:
    return int(round(t * FRAME_RATE))


def _note_fields(n: Note, vocab: ExtendedVocab) -> tuple[int, int, int]:
    """-> (program, pitch, cent_token_value)."""
    program = n.program if n.program is not None else vocab.program_of(n.instrument)
    pitch, cents = hz_to_pitch_cents(n.hz)
    pitch = max(0, min(127, pitch))
    return program, pitch, quantize_cents(cents)


def trim_overlaps(notes: list[Note], vocab: ExtendedVocab) -> list[Note]:
    """Same (program, pitch) cannot overlap: the decoder treats a re-strike
    as "close the previous note". Cut the earlier note at the later onset."""
    by_key: dict[tuple[int, int], list[Note]] = {}
    for n in notes:
        if n.is_drum:
            continue
        prog, pitch, _ = _note_fields(n, vocab)
        by_key.setdefault((prog, pitch), []).append(n)
    for group in by_key.values():
        group.sort(key=lambda n: n.onset)
        for a, b in zip(group, group[1:]):
            if a.offset > b.onset:
                a.offset = b.onset
    return [n for n in notes if n.is_drum or _tick(n.offset) > _tick(n.onset)]


def encode_chunk(
    notes: list[Note],
    start: float,
    vocab: ExtendedVocab,
    duration: float = SEGMENT_SECONDS,
) -> list[int]:
    """Token ids for the chunk [start, start+duration), ending with EOS.

    `notes` should already have gone through trim_overlaps().
    """
    idx = vocab.index()
    s_tick = _tick(start)
    e_tick = _tick(start + duration)
    tokens: list[int] = []

    # --- tie prologue -------------------------------------------------------
    ties = sorted({
        _note_fields(n, vocab)[:2]
        for n in notes
        if not n.is_drum and _tick(n.onset) < s_tick < _tick(n.offset)
    })
    prog_state = None
    for prog, pitch in ties:
        if prog != prog_state:
            tokens.append(idx[("program", prog)])
            prog_state = prog
        tokens.append(idx[("pitch", pitch)])
        # A tied note's onset happened in an earlier chunk, so no cent token.
    tokens.append(idx[("tie", 0)])

    # --- events -------------------------------------------------------------
    # (tick, is_drum, program, velocity, pitch, cents) — same sort order as
    # muscriptor.tokenizer.notes.sort_note_events, so offsets (velocity 0)
    # precede onsets (velocity 1) at the same tick.
    events = []
    for n in notes:
        on, off = _tick(n.onset), _tick(n.offset)
        if n.is_drum:
            if s_tick <= on < e_tick:
                pitch = max(0, min(127, hz_to_pitch_cents(n.hz)[0]))
                events.append((on, 1, 128, 1, pitch, 0))
            continue
        prog, pitch, cents = _note_fields(n, vocab)
        if s_tick <= on < e_tick:
            events.append((on, 0, prog, 1, pitch, cents))
        if s_tick < off < e_tick:
            events.append((off, 0, prog, 0, pitch, 0))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3], e[4]))

    # The program state carries over from the tie prologue; velocity does not
    # (OpenNoteTracker resets only velocity at the tie token).
    cur_tick, cur_prog, cur_vel = s_tick, prog_state, None
    for tick, is_drum, prog, vel, pitch, cents in events:
        if tick != cur_tick:
            tokens.append(idx[("shift", tick - s_tick)])
            cur_tick = tick
        if is_drum:
            # Drums count as onsets: the model writes velocity(1) first if the
            # state is 0, and the state stays 1 afterwards.
            if cur_vel != 1:
                tokens.append(idx[("velocity", 1)])
                cur_vel = 1
            tokens.append(idx[("drum", pitch)])
            continue
        if prog != cur_prog:
            tokens.append(idx[("program", prog)])
            cur_prog = prog
        if vel != cur_vel:
            tokens.append(idx[("velocity", vel)])
            cur_vel = vel
        tokens.append(idx[("pitch", pitch)])
        if vel == 1 and vocab.use_cents:
            tokens.append(idx[("cent", cents)])

    tokens.append(vocab.eos_id)
    return tokens


def chunk_starts(total_seconds: float, duration: float = SEGMENT_SECONDS) -> list[float]:
    import math

    n = max(1, math.ceil(total_seconds / duration - 1e-9))
    return [i * duration for i in range(n)]


class CentAwareDecoder:
    """Token stream -> Notes with Hz, reusing MuScriptor's OpenNoteTracker.

    The tracker already understands the extended vocabulary: new gamelan
    instruments are ordinary "program" events, and it ignores unknown event
    types. We only add one rule: a "cent" token right after an onset sets
    the cents of the note that just started.
    """

    def __init__(self, vocab: ExtendedVocab, instrument_name=None):
        self.vocab = vocab
        self.tracker = OpenNoteTracker(vocab.events, FRAME_RATE)
        self.instrument_name = instrument_name or vocab.instrument_of
        self.open: dict[tuple[int, int], dict] = {}
        self.notes: list[Note] = []
        self._last_started: tuple[int, int] | None = None

    def _apply(self, actions) -> None:
        for a in actions:
            if isinstance(a, _StartNote):
                key = (a.program, a.pitch)
                self.open[key] = {"onset": a.time, "cents": 0}
                self._last_started = key
            elif isinstance(a, _EndNote):
                rec = self.open.pop((a.program, a.pitch), None)
                if rec is None:
                    continue
                self.notes.append(Note(
                    instrument=self.instrument_name(a.program),
                    onset=rec["onset"],
                    offset=max(a.time, rec["onset"] + MIN_DURATION),
                    hz=pitch_cents_to_hz(a.pitch, rec["cents"]),
                    program=a.program,
                ))
            elif isinstance(a, _DrumHit):
                self.notes.append(Note("drums", a.time, a.time + MIN_DURATION,
                                       pitch_cents_to_hz(a.pitch, 0), 128, True))

    def feed(self, item) -> None:
        if isinstance(item, ProgressEvent):
            return
        if isinstance(item, ChunkBoundary):
            self._last_started = None
            self._apply(self.tracker.feed(item))
            return
        event = self.vocab.events[item]
        if event.type == "cent":
            if self._last_started in self.open:
                self.open[self._last_started]["cents"] = event.value
            self._last_started = None
            return
        if event.type == "reserved":
            return
        self._last_started = None
        self._apply(self.tracker.feed(item))

    def finish(self) -> list[Note]:
        self._apply(self.tracker.finish())
        self.notes.sort(key=lambda n: (n.onset, n.hz))
        return self.notes


def decode_chunks(token_chunks: list[list[int]], starts: list[float],
                  vocab: ExtendedVocab) -> list[Note]:
    """Decode a list of per-chunk token lists (EOS optional) into notes."""
    dec = CentAwareDecoder(vocab)
    for i, (toks, s) in enumerate(zip(token_chunks, starts)):
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        dec.feed(ChunkBoundary(s, nxt))
        for t in toks:
            if t == vocab.eos_id:
                break
            dec.feed(t)
    return dec.finish()
