"""Laya-FR: fine-tune our Laya v2 cross-encoder (ModernBERT encoder + mean pooling + linear head) on map-labelled
French pairs from the TEST set (organiser-approved self-training), then score all French test candidate pairs.

Model = identical to the main pipeline's er.crossencoder.CrossEncoder: tokenizer(text_a, text_b, max_length=128),
last_hidden_state mean-pooled over the attention mask, Linear(768 -> 1) head (model/ce_head.pt) -> logit.
Loss = BCE-with-logits. AdamW, linear warm-up 300 steps then linear decay, 1 epoch, grad-clip 1.0.

usage:  python train_score.py --dtype bf16            (Blackwell / Ampere / Hopper)
out:    out/scores.parquet (key, ce)  +  out/train_meta.json  [+ out/model/ with --save_model]
"""
import argparse
import json
import os
import time

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MAX_LEN = 128


class CrossEncoder(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.enc = AutoModel.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa")
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(os.path.join(path, "ce_head.pt"), map_location="cpu"))

    def forward(self, ids, am):
        h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def batches(tok, A, B, order, bs):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([A[j] for j in idx], [B[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        yield idx, enc["input_ids"].cuda(non_blocking=True), enc["attention_mask"].cuda(non_blocking=True)


@torch.no_grad()
def score(model, A, B, bs, dtype):
    model.eval()
    order = np.argsort([len(a) + len(b) for a, b in zip(A, B)], kind="stable")
    out = np.empty(len(A), np.float32)
    t0 = time.time()
    for n, (idx, ids, am) in enumerate(batches(model.tok, A, B, order, bs)):
        with torch.autocast("cuda", dtype=dtype):
            out[idx] = model(ids, am).float().cpu().numpy()
        if n % 500 == 0:
            print(f"  scored {min(len(A), (n + 1) * bs):,}/{len(A):,} {min(len(A), (n + 1) * bs) / (time.time() - t0):.0f} pairs/s", flush=True)
    model.train()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--score_bs", type=int, default=1024)
    ap.add_argument("--save_model", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    torch.manual_seed(2026)
    t0 = time.time()
    model = CrossEncoder(a.model).cuda().train()
    tr = pl.read_parquet(os.path.join(a.data, "train_pairs.parquet"))
    dev = pl.read_parquet(os.path.join(a.data, "dev_pairs.parquet"))
    A, B, Y = tr["text_a"].to_list(), tr["text_b"].to_list(), tr["y"].to_numpy().astype(np.float32)
    DA, DB, DY = dev["text_a"].to_list(), dev["text_b"].to_list(), dev["y"].to_numpy()
    hist = [{"step": 0, "dev_auc": float(roc_auc_score(DY, score(model, DA, DB, a.score_bs, dtype)))}]
    print(f"train {len(Y):,} pairs (pos {Y.mean():.3f}) | dev AUC before: {hist[-1]['dev_auc']:.5f}", flush=True)
    order = np.random.default_rng(2026).permutation(len(Y))
    total = len(order) // a.bs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda", enabled=(a.dtype == "fp16"))
    lossf = torch.nn.BCEWithLogitsLoss()
    t1, run = time.time(), 0.0
    for step, (idx, ids, am) in enumerate(batches(model.tok, A, B, order[: total * a.bs], a.bs), 1):
        with torch.autocast("cuda", dtype=dtype):
            logit = model(ids, am)
        loss = lossf(logit.float(), torch.from_numpy(Y[idx]).cuda())
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run += loss.item()
        if step % 200 == 0:
            el = time.time() - t1
            print(f"step {step}/{total} loss {run / 200:.4f} {step * a.bs / el:.0f} pairs/s eta {(total - step) * el / step / 60:.1f}m", flush=True)
            run = 0.0
        if step % a.eval_every == 0:
            hist.append({"step": step, "dev_auc": float(roc_auc_score(DY, score(model, DA, DB, a.score_bs, dtype)))})
            print("dev:", hist[-1], flush=True)
    hist.append({"step": total, "dev_auc": float(roc_auc_score(DY, score(model, DA, DB, a.score_bs, dtype)))})
    print("dev final:", hist[-1], f"| train {(time.time() - t1) / 60:.1f} min", flush=True)
    if a.save_model:
        md = os.path.join(a.out, "model")
        model.enc.save_pretrained(md); model.tok.save_pretrained(md); torch.save(model.head.state_dict(), os.path.join(md, "ce_head.pt"))
    S = pl.read_parquet(os.path.join(a.data, "score_pairs.parquet"))
    ce = score(model, S["text_a"].to_list(), S["text_b"].to_list(), a.score_bs, dtype)
    print(f"non-finite scores: {int((~np.isfinite(ce)).sum())}", flush=True)
    S.select("key").with_columns(pl.Series("ce", ce)).write_parquet(os.path.join(a.out, "scores.parquet"))
    json.dump({"init": "Laya v2 (models/ce_laya_v2)", "train_pairs": len(Y), "bs": a.bs, "lr": a.lr, "dtype": a.dtype, "steps": total,
               "minutes": round((time.time() - t0) / 60, 1), "dev_history": hist}, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)
    print(f"wrote {a.out}/scores.parquet ({len(ce):,}) | total {(time.time() - t0) / 60:.1f} min", flush=True)
