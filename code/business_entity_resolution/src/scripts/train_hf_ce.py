"""Fine-tune a native 1-logit HF reranker (bge-reranker-v2-m3) on the V100: fp16 autocast + GradScaler (no bf16 on sm_70).
Data = real India/US pairs (S1 folds 1-4 x candidate top-k, same construction as train_crossencoder.py)
     + optional organiser-approved France pseudo-labels (cache/<pseudo>.parquet: text_a, text_b, y).
Saves an fp16 HF folder usable by ce_fill.py --hf.
usage: python scripts/train_hf_ce.py --init models/ce_bge_m3_fp16 --out models/ce_bge_p --pseudo pseudo_fr --ns1 60000 --k 10
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from er.crossencoder import pair_text  # noqa: E402
from er.io import CACHE, ROOT, load_gt_pairs, load_norm  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--ns1", type=int, default=60_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cands", default="cands_train_scann")
    ap.add_argument("--s1_seed", type=int, default=17)
    ap.add_argument("--pseudo", default="")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=128)
    a = ap.parse_args()
    print(vars(a), flush=True)
    t0 = time.time()
    tr = load_norm("train").with_row_index("rid")
    split = pl.read_parquet(CACHE / "split.parquet")
    idmap = tr.select("entity_id", pl.col("rid").cast(pl.UInt32))
    truth = load_gt_pairs().join(idmap, left_on="id1", right_on="entity_id").rename({"rid": "r1"}) \
                           .join(idmap, left_on="id2", right_on="entity_id").rename({"rid": "r2"}).select("r1", "r2")
    folds = [int(f) for f in a.folds.split(",")]
    s1 = tr.filter(pl.col("src") == 1).join(split.filter(pl.col("fold").is_in(folds)), on="entity_id") \
           .select(pl.col("rid").cast(pl.UInt32).alias("r1")).sample(a.ns1, seed=a.s1_seed)
    C = pl.read_parquet(CACHE / f"{a.cands}.parquet").filter(pl.col("rank") < a.k).join(s1, on="r1")
    C = C.join(truth.with_columns(pl.lit(1.0).alias("y")), on=["r1", "r2"], how="left").with_columns(pl.col("y").fill_null(0.0))
    recs = tr.select(pl.col("rid").cast(pl.UInt32), "business_name", "business_address")
    TA = [pair_text(n, s) for n, s in C.join(recs, left_on="r1", right_on="rid", how="left", maintain_order="left").select("business_name", "business_address").rows()]
    TB = [pair_text(n, s) for n, s in C.join(recs, left_on="r2", right_on="rid", how="left", maintain_order="left").select("business_name", "business_address").rows()]
    y = C["y"].to_numpy().astype(np.float32)
    if a.pseudo:
        P = pl.read_parquet(CACHE / f"{a.pseudo}.parquet")
        TA += P["text_a"].to_list(); TB += P["text_b"].to_list(); y = np.concatenate([y, P["y"].to_numpy().astype(np.float32)])
        print(f"+ pseudo-labelled pairs {P.height:,} (pos {P['y'].mean():.3f})", flush=True)
    perm = np.random.default_rng(a.s1_seed).permutation(len(y))
    TA = [TA[i] for i in perm]; TB = [TB[i] for i in perm]; y = y[perm]
    print(f"pairs {len(y):,} pos {y.mean():.3f} | {time.time()-t0:.0f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(a.init)
    model = AutoModelForSequenceClassification.from_pretrained(a.init, torch_dtype=torch.float32).cuda().train()

    def collate(idx):
        enc = tok([TA[i] for i in idx], [TB[i] for i in idx], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(y[idx])

    dl = DataLoader(list(range(len(y))), batch_size=a.bs, shuffle=False, num_workers=8, collate_fn=collate, prefetch_factor=4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    t1, run = time.time(), 0.0
    for step, (ids, am, yy) in enumerate(dl):
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(input_ids=ids.cuda(non_blocking=True), attention_mask=am.cuda(non_blocking=True)).logits.squeeze(-1)
        loss = lossf(logit.float(), yy.cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 200 == 0:
            rate = (step + 1) * a.bs / (time.time() - t1)
            print(f"step {step}/{total} loss {run:.4f} {rate:.0f} pairs/s eta {(total - step) * a.bs / rate / 60:.1f}m", flush=True)
    out = ROOT / a.out
    model.half().save_pretrained(out); tok.save_pretrained(out)
    print(f"saved {out} | {(time.time()-t0)/60:.1f} min", flush=True)
