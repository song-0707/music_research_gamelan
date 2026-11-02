"""Load MuScriptor, extend its vocabulary, add LoRA, save/load adapters.

Facts about the architecture this file relies on (muscriptor 0.4.0):

* LMModel is decoder-only. Audio enters as a PREFIX: the log-mel
  conditioner (`condition_provider.conditioners.self_wav`, 512 HTK mel bins,
  n_fft 2048, hop 160 at 16 kHz -> 501 frames per 5 s chunk) is projected by
  `output_proj` and concatenated in front of the token embeddings, together
  with two class tokens (instrument_group, dataset_name). There is no
  cross-attention.
* Each layer `transformer.layers.{i}` has
      self_attn.in_proj_weight  [3D, D]   fused q|k|v (a raw Parameter used
                                          with F.linear, so PEFT cannot wrap it)
      self_attn.out_proj        Linear(D, D)
      linear1 / linear2         FFN (D -> 4D -> D)
  We add LoRA with torch.nn.utils.parametrize, which works on raw Parameters.
* `emb` has card+1 rows; row `card` is the BOS ("initial") token.
  `linear` (output head) has card rows.
* `LMModel._compute_logits` hard-codes `logits[:, 1393:] = -inf`, which
  would hide every new token. We replace it with a mask of only the
  checkpoint's reserved slots.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

from muscriptor.models.lm import LMModel
from muscriptor.modules.conditioners import ConditioningAttributes, WavCondition
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.tokenizer.mt3 import MT3Tokenizer

from gamelan_amt.vocab import (
    GAMELAN_INSTRUMENTS,
    GAMELAN_PROGRAM_BASE,
    INIT_PROGRAM_FOR_NEW_INSTRUMENTS,
    ExtendedVocab,
    build_extended_vocab,
)

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AdaptConfig:
    base_model: str = "medium"            # small | medium | large | path
    use_cents: bool = True                # False = 12-TET ablation
    mode: str = "lora"                    # lora | full | frozen
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_targets: list[str] = field(default_factory=lambda: ["q", "v"])
    # allowed targets: q k v (slices of in_proj_weight), out (out_proj),
    #                  ffn1 ffn2 (linear1/linear2)
    train_mel_proj: bool = False          # also adapt the mel->D projection
    instruments: list[str] = field(default_factory=lambda: list(GAMELAN_INSTRUMENTS))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: Path) -> "AdaptConfig":
        return AdaptConfig(**json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Vocabulary surgery
# ---------------------------------------------------------------------------

class ExtendedEmbedding(nn.Module):
    """Frozen original rows + trainable new rows + frozen BOS row.

    Index layout matches LMModel: ids 0..card-1 tokens, id card = BOS,
    id -1 = zero vector (ScaledEmbedding semantics).
    """

    def __init__(self, old: nn.Embedding, base_card: int, init_ids: list[int],
                 noise: float = 0.02):
        super().__init__()
        W = old.weight.detach()
        assert W.shape[0] == base_card + 1
        self.old = nn.Parameter(W[:base_card].clone(), requires_grad=False)
        self.bos = nn.Parameter(W[base_card:].clone(), requires_grad=False)
        new = W[torch.tensor(init_ids, dtype=torch.long)].clone()
        new = new + noise * W.std() * torch.randn_like(new)
        self.new = nn.Parameter(new)

    @property
    def weight(self) -> torch.Tensor:  # LMModel.generate reads .weight.device
        return torch.cat([self.old, self.new, self.bos], dim=0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        is_zero = input == -1
        y = F.embedding(input.clamp(min=0), self.weight)
        return torch.where(is_zero[..., None], torch.zeros_like(y), y)


class ExtendedLinear(nn.Module):
    """Output head: frozen original rows + trainable new rows."""

    def __init__(self, old: nn.Linear, init_ids: list[int], noise: float = 0.02):
        super().__init__()
        W = old.weight.detach()
        self.old = nn.Parameter(W.clone(), requires_grad=False)
        new = W[torch.tensor(init_ids, dtype=torch.long)].clone()
        new = new + noise * W.std() * torch.randn_like(new)
        self.new = nn.Parameter(new)

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([self.old, self.new], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


def _init_ids_for_new_tokens(vocab: ExtendedVocab, mt3_index: dict) -> list[int]:
    """Which existing row each new token starts from."""
    ids = []
    for ev in vocab.events[vocab.base_card:]:
        if ev.type == "cent":
            # Start from velocity-1 (onset) — a token that also follows the
            # pitch/onset context; noise breaks the symmetry between cents.
            ids.append(mt3_index[("velocity", 1)])
        elif ev.type == "program":
            ids.append(mt3_index[("program", INIT_PROGRAM_FOR_NEW_INSTRUMENTS)])
        else:
            raise ValueError(ev)
    return ids


def _patched_compute_logits(self, sequence, cfg_conditions, model_state, first_step,
                            cfg_coef=None, forbidden_tokens=None):
    """Copy of LMModel._compute_logits with the hard-coded 1393 mask replaced."""
    B = sequence.shape[0]
    cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
    if cfg_coef == 1.0:
        logits = self(sequence, cfg_conditions, first_step=first_step,
                      model_state=model_state)
    else:
        doubled = torch.cat([sequence, sequence], dim=0)
        all_logits = self(doubled, cfg_conditions, first_step=first_step,
                          model_state=model_state)
        cond_logits, uncond_logits = all_logits.split(B, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
    logits = logits[:, -1, :].float()
    if self._masked_token_ids is not None:
        logits[:, self._masked_token_ids] = -torch.inf
    if forbidden_tokens is not None:
        logits[:, forbidden_tokens] = -torch.inf
    return logits


def extend_vocabulary(lm: LMModel, vocab: ExtendedVocab) -> None:
    """In-place: grow emb/linear, move BOS, fix the logits mask."""
    base_card = lm.card
    assert base_card == vocab.base_card, (base_card, vocab.base_card)
    mt3_index = {(e.type, e.value): i for i, e in enumerate(vocab.events[:1393])}
    init_ids = _init_ids_for_new_tokens(vocab, mt3_index)

    lm.emb = ExtendedEmbedding(lm.emb, base_card, init_ids)
    lm.linear = ExtendedLinear(lm.linear, init_ids)
    lm.card = vocab.num_tokens  # moves initial_token_id (BOS) to the new end

    reserved = list(range(1393, base_card))
    device = lm.emb.weight.device
    lm._masked_token_ids = (
        torch.tensor(reserved, dtype=torch.long, device=device) if reserved else None
    )
    lm._compute_logits = types.MethodType(_patched_compute_logits, lm)


# ---------------------------------------------------------------------------
# LoRA via parametrization
# ---------------------------------------------------------------------------

class LoRADelta(nn.Module):
    """W -> W + (alpha/r) * B @ A, optionally only on row blocks of W.

    For the fused in_proj_weight [3D, D], `blocks` picks which of q/k/v get
    an adapter; the others get exactly zero update.
    """

    def __init__(self, out_features: int, in_features: int, rank: int,
                 alpha: float, blocks: list[int] | None = None, n_blocks: int = 1):
        super().__init__()
        self.scale = alpha / rank
        self.n_blocks = n_blocks
        self.block_out = out_features // n_blocks
        self.blocks = blocks if blocks is not None else list(range(n_blocks))
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        for _ in self.blocks:
            A = torch.empty(rank, in_features)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.A.append(nn.Parameter(A))
            self.B.append(nn.Parameter(torch.zeros(self.block_out, rank)))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        parts = []
        k = 0
        for b in range(self.n_blocks):
            if b in self.blocks:
                d = (self.B[k] @ self.A[k]) * self.scale
                parts.append(d.to(W.dtype))
                k += 1
            else:
                parts.append(W.new_zeros(self.block_out, W.shape[1]))
        return W + torch.cat(parts, dim=0)


def add_lora(lm: LMModel, rank: int, alpha: float, targets: list[str]) -> None:
    qkv = [i for i, t in enumerate("qkv") if t in targets]
    for layer in lm.transformer.layers:
        attn = layer.self_attn
        D = attn.embed_dim
        if qkv:
            parametrize.register_parametrization(
                attn, "in_proj_weight",
                LoRADelta(3 * D, D, rank, alpha, blocks=qkv, n_blocks=3))
        for name, mod in (("out", attn.out_proj), ("ffn1", layer.linear1),
                          ("ffn2", layer.linear2)):
            if name in targets:
                out_f, in_f = mod.weight.shape
                parametrize.register_parametrization(
                    mod, "weight", LoRADelta(out_f, in_f, rank, alpha))


def merge_lora(lm: LMModel) -> None:
    """Fold adapters into the base weights (faster inference, same output)."""
    for module in lm.modules():
        if parametrize.is_parametrized(module):
            for name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)


# ---------------------------------------------------------------------------
# Trainable-parameter selection
# ---------------------------------------------------------------------------

def set_trainable(lm: LMModel, cfg: AdaptConfig) -> None:
    for p in lm.parameters():
        p.requires_grad = False
    if cfg.mode == "full":
        for p in lm.transformer.parameters():
            p.requires_grad = True
        for p in lm.out_norm.parameters():
            p.requires_grad = True
    for n, p in lm.named_parameters():
        if ".parametrizations." in n and not n.endswith(".original"):
            p.requires_grad = True          # LoRA A/B
    lm.emb.new.requires_grad = True
    lm.linear.new.requires_grad = True
    if cfg.train_mel_proj:
        for p in lm.condition_provider.conditioners["self_wav"].output_proj.parameters():
            p.requires_grad = True


def count_parameters(lm: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in lm.parameters())
    trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    return trainable, total


def enable_gradient_checkpointing(lm: LMModel) -> None:
    """Recompute each transformer layer in backward (saves activation memory).

    Overrides `forward` on each layer instance so parameter names (and hence
    saved adapters) are unchanged.
    """
    for layer in lm.transformer.layers:
        orig = layer.forward

        def fwd(x, model_state=None, _orig=orig):
            if model_state is None and torch.is_grad_enabled():
                return checkpoint(_orig, x, use_reentrant=False)
            return _orig(x, model_state=model_state)

        layer.forward = fwd


# ---------------------------------------------------------------------------
# Building / loading
# ---------------------------------------------------------------------------

def adapt_lm(lm: LMModel, cfg: AdaptConfig) -> ExtendedVocab:
    vocab = build_extended_vocab(lm.card, use_cents=cfg.use_cents)
    extend_vocabulary(lm, vocab)
    if cfg.mode == "lora":
        add_lora(lm, cfg.lora_rank, cfg.lora_alpha, cfg.lora_targets)
    set_trainable(lm, cfg)
    return vocab


def patch_tokenizer(tm: TranscriptionModel, vocab: ExtendedVocab) -> None:
    """Make the wrapper's tokenizer use the extended decode table, so
    MuScriptor's chunk loop (tie prologue forcing) works unchanged."""
    tok: MT3Tokenizer = tm._tokenizer
    tok._vocab = vocab.events
    tok._token_index = vocab.index()
    tok.num_tokens = vocab.num_tokens


def load_base(base_model: str, device: str | torch.device) -> TranscriptionModel:
    """Pretrained MuScriptor in fp32 (autocast handles bf16 in training).

    base_model="tiny-random" builds a 2-layer random model with the real
    architecture: use it to dry-run the pipeline without downloading weights.
    """
    if base_model == "tiny-random":
        from muscriptor.transcription_model import _build_model, _ModelConfig

        device = torch.device(device)
        torch.manual_seed(0)
        lm = _build_model(device, _ModelConfig(dim=64, num_heads=4, num_layers=2, card=1395))
        lm.eval()
        tok = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
        return TranscriptionModel(lm, tok, device)
    return TranscriptionModel.load_model(base_model, device=device, dtype="float32")


def build_adapted(cfg: AdaptConfig, device, tm: TranscriptionModel | None = None):
    """-> (TranscriptionModel wrapper, LMModel, ExtendedVocab)."""
    tm = tm or load_base(cfg.base_model, device)
    lm = tm._model
    vocab = adapt_lm(lm, cfg)
    lm.to(device)
    patch_tokenizer(tm, vocab)
    return tm, lm, vocab


def trainable_state_dict(lm: LMModel) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in lm.named_parameters() if p.requires_grad}


def save_adapter(lm: LMModel, cfg: AdaptConfig, out_dir: Path) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state_dict(lm), str(out_dir / "adapter.safetensors"))
    cfg.save(out_dir / "adapt_config.json")


def load_adapter(adapter_dir: Path, device, tm: TranscriptionModel | None = None):
    """Rebuild base + surgery + LoRA, then load the trained weights."""
    from safetensors.torch import load_file

    cfg = AdaptConfig.load(adapter_dir / "adapt_config.json")
    tm, lm, vocab = build_adapted(cfg, device, tm=tm)
    sd = load_file(str(adapter_dir / "adapter.safetensors"), device=str(device))
    missing, unexpected = lm.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    expected = set(trainable_state_dict(lm))
    not_loaded = expected - set(sd)
    assert not not_loaded, f"adapter is missing {sorted(not_loaded)[:5]}"
    lm.eval()
    return tm, lm, vocab, cfg


# ---------------------------------------------------------------------------
# Training forward
# ---------------------------------------------------------------------------

def condition_tensors(lm: LMModel, wav: torch.Tensor) -> dict:
    """wav [B, 1, T] at 16 kHz -> the same prefix tensors inference uses.

    Uses the model's own ConditioningProvider, so the front-end can never
    drift from what the checkpoint was trained with. instrument_group and
    dataset_name are left unconditional (None), as in default inference.
    """
    B, _, T = wav.shape
    attrs = [
        ConditioningAttributes(
            wav={"self_wav": WavCondition(
                wav=wav[i:i + 1], length=torch.tensor([T], device=wav.device),
                sample_rate=[SAMPLE_RATE], path=[None], seek_time=[0.0])},
            text={"instrument_group": None, "dataset_name": None},
        )
        for i in range(B)
    ]
    with contextlib.redirect_stdout(io.StringIO()):  # the conditioner prints timings
        tokenized = lm.condition_provider.tokenize(attrs)
        return lm.condition_provider(tokenized)


def lm_loss(lm: LMModel, wav: torch.Tensor, inp: torch.Tensor,
            tgt: torch.Tensor) -> torch.Tensor:
    """Teacher-forced next-token cross-entropy. tgt uses -100 for padding."""
    cond = condition_tensors(lm, wav)
    logits = lm(inp, cond, first_step=True, model_state=None)  # [B, S, V]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           tgt.reshape(-1), ignore_index=-100)
