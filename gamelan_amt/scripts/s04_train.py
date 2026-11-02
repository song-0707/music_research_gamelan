"""Step 5: train an adapter.

    python scripts/s04_train.py --config configs/lora_cents.yaml

Resumable: re-running the same command continues from <out_dir>/last/.
Stops cleanly before `max_hours` so a 24-h job never dies mid-step.
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from gamelan_amt.data import ChunkDataset, make_collate
from gamelan_amt.model import (AdaptConfig, build_adapted, count_parameters,
                               enable_gradient_checkpointing, lm_loss, save_adapter)


def cosine_lr(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def validate(lm, loader, device, amp, max_batches):
    lm.eval()
    tot, n = 0.0, 0
    for i, (wav, inp, tgt) in enumerate(loader):
        if i >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            tot += lm_loss(lm, wav.to(device), inp.to(device), tgt.to(device)).item()
        n += 1
    lm.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    C = yaml.safe_load(Path(args.config).read_text())
    out = Path(C["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    seed = C.get("seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    cfg = AdaptConfig(**C["adapt"])
    tm, lm, vocab = build_adapted(cfg, device)
    if C.get("grad_checkpointing", True):
        enable_gradient_checkpointing(lm)
    tr, tot = count_parameters(lm)
    print(f"trainable {tr:,} / {tot:,} parameters ({100*tr/tot:.2f} %)")

    data = Path(C["data_dir"])
    collate = make_collate(vocab)
    train_ds = ChunkDataset(data / "train", vocab, train=True)
    val_ds = ChunkDataset(data / "val", vocab, train=False)
    print(f"train chunks {len(train_ds)}, val chunks {len(val_ds)}")
    train_dl = DataLoader(train_ds, C["batch_size"], shuffle=True, drop_last=True,
                          num_workers=C.get("workers", 4), collate_fn=collate,
                          persistent_workers=C.get("workers", 4) > 0)
    val_dl = DataLoader(val_ds, C["batch_size"], shuffle=False,
                        num_workers=C.get("workers", 4), collate_fn=collate)

    # Separate learning rates: new token rows start from scratch-ish and
    # usually want a higher LR than the LoRA matrices.
    new_rows = [lm.emb.new, lm.linear.new]
    others = [p for p in lm.parameters() if p.requires_grad
              and all(p is not q for q in new_rows)]
    opt = torch.optim.AdamW([
        {"params": others, "lr": C["lr"], "weight_decay": C.get("weight_decay", 0.0)},
        {"params": new_rows, "lr": C["lr"] * C.get("new_token_lr_mult", 5.0),
         "weight_decay": 0.0},
    ], betas=(0.9, 0.98))
    base_lrs = [g["lr"] for g in opt.param_groups]

    step, best = 0, float("inf")
    last = out / "last"
    if (last / "trainer.pt").exists():
        ck = torch.load(last / "trainer.pt", map_location=device, weights_only=False)
        lm.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"resumed at step {step}")

    accum = C.get("grad_accum", 1)
    total_steps, warmup = C["steps"], C.get("warmup", 500)
    t_start = time.time()
    log = (out / "log.csv").open("a", newline="")
    w = csv.writer(log)

    def save(tag):
        d = out / tag
        save_adapter(lm, cfg, d)
        torch.save({"model": {n: p for n, p in lm.state_dict().items()
                              if n in dict(lm.named_parameters())
                              and dict(lm.named_parameters())[n].requires_grad},
                    "opt": opt.state_dict(), "step": step, "best": best},
                   d / "trainer.pt")

    lm.train()
    it = iter(train_dl)
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            try:
                wav, inp, tgt = next(it)
            except StopIteration:
                it = iter(train_dl)
                wav, inp, tgt = next(it)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = lm_loss(lm, wav.to(device), inp.to(device), tgt.to(device)) / accum
            loss.backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0)
        f = cosine_lr(step, warmup, total_steps)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        opt.step()
        step += 1

        if step % C.get("log_every", 50) == 0:
            el = (time.time() - t_start) / 3600
            print(f"step {step:6d}  loss {loss_acc:.4f}  lr {base_lrs[0]*f:.2e}  {el:.2f} h")
            w.writerow([step, "train", f"{loss_acc:.5f}"]); log.flush()
        if step % C.get("eval_every", 1000) == 0 or step == total_steps:
            vl = validate(lm, val_dl, device, amp, C.get("val_batches", 50))
            print(f"           val loss {vl:.4f}")
            w.writerow([step, "val", f"{vl:.5f}"]); log.flush()
            if vl < best:
                best = vl
                save("best")
            save("last")
        if (time.time() - t_start) / 3600 > C.get("max_hours", 23.5):
            print("time budget reached; saving and exiting (re-run to resume)")
            save("last")
            break
    log.close()


if __name__ == "__main__":
    main()
