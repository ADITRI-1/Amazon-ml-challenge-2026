"""Stage-3 cross-encoder: reads (S1 record, candidate record) jointly -> match logit.

- trains on S1 from folds --folds (default 1-4) so its scores are out-of-sample for the matcher (folds 5-9) and val (0)
- pairs = stage-1 top-`k` candidates (hard negatives come for free), BCE loss
- init from the fine-tuned bi-encoder (domain-adapted); mean pooling + linear head; fp32 weights + fp16 autocast

usage: python scripts/train_crossencoder.py --init models/biencoder_e5s_v1 --out models/ce_e5s_v1 --ns1 250000 --k 12
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

from er.crossencoder import CE_MAX_LEN, CrossEncoder, PairDS, pair_text  # noqa: E402
from er.io import CACHE, ROOT, load_gt_pairs, load_norm  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--ns1", type=int, default=250_000)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--cands", default="cands_train_v1", help="candidate lists the training pairs come from")
    ap.add_argument("--s1_seed", type=int, default=0)
    ap.add_argument("--synth_fr", type=int, default=0, help="add synthetic French pairs from N generated entities (er.synth_fr)")
    ap.add_argument("--synth_v3", default="", help="path to fr_vocab.json -> use generate_v3 (test-harvested vocabulary)")
    ap.add_argument("--pseudo", default="", help="cache parquet (text_a, text_b, y) of extra pseudo-labelled pairs (organiser-approved France self-training)")
    ap.add_argument("--fold_latin", action="store_true", help="strip Latin diacritics from CE inputs (saved with the model)")
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
    C = C.join(truth.with_columns(pl.lit(1.0).alias("y")), on=["r1", "r2"], how="left").with_columns(pl.col("y").fill_null(0.0)).sample(fraction=1.0, shuffle=True, seed=1)
    recs = tr.select(pl.col("rid").cast(pl.UInt32), "business_name", "business_address")
    TA = C.join(recs, left_on="r1", right_on="rid", how="left", maintain_order="left").select("business_name", "business_address").rows()
    TB = C.join(recs, left_on="r2", right_on="rid", how="left", maintain_order="left").select("business_name", "business_address").rows()
    TA = [pair_text(n, s) for n, s in TA]; TB = [pair_text(n, s) for n, s in TB]
    y = C["y"].to_numpy().astype(np.float32)
    if a.synth_fr:
        from er.synth_fr import generate, generate_v3
        S = generate_v3(a.synth_fr, seed=a.s1_seed, vocab_path=a.synth_v3) if a.synth_v3 else generate(a.synth_fr, seed=a.s1_seed)
        TA += [x[0] for x in S]; TB += [x[1] for x in S]
        y = np.concatenate([y, np.array([x[2] for x in S], np.float32)])
        perm = np.random.default_rng(a.s1_seed).permutation(len(y))
        TA = [TA[i] for i in perm]; TB = [TB[i] for i in perm]; y = y[perm]
        print(f"+ synthetic French pairs {len(S):,} (pos {np.mean([x[2] for x in S]):.3f})", flush=True)
    if a.pseudo:
        Ps = pl.read_parquet(CACHE / f"{a.pseudo}.parquet")
        TA += Ps["text_a"].to_list(); TB += Ps["text_b"].to_list()
        y = np.concatenate([y, Ps["y"].to_numpy().astype(np.float32)])
        perm = np.random.default_rng(a.s1_seed + 1).permutation(len(y))
        TA = [TA[i] for i in perm]; TB = [TB[i] for i in perm]; y = y[perm]
        print(f"+ pseudo-labelled pairs {Ps.height:,} (pos {Ps['y'].mean():.3f})", flush=True)
    print(f"pairs {len(y):,} pos {y.mean():.3f} | {time.time()-t0:.0f}s", flush=True)

    model = CrossEncoder(a.init, train=True)
    if a.fold_latin:
        model.fold = True
    tok = model.tok
    ds = PairDS(TA, TB, y, tok, fold=model.fold)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=12, collate_fn=ds.collate, prefetch_factor=8)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(dl) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    step, t1, run = 0, time.time(), 0.0
    out = ROOT / a.out
    for ep in range(a.epochs):
        for ids, am, yy in dl:
            ids, am, yy = ids.cuda(non_blocking=True), am.cuda(non_blocking=True), yy.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(ids, am)
            loss = lossf(logit.float(), yy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1; run += loss.item()
            if step % 200 == 0:
                el = time.time() - t1
                print(f"ep {ep} step {step}/{total} loss {run/200:.4f} {step*a.bs/el:.0f} pairs/s eta {(total-step)*el/step/60:.1f}m", flush=True)
                run = 0.0
                if not np.isfinite(loss.item()):
                    raise SystemExit("non-finite loss")
            if step % 3000 == 0:
                model.save(out)
    model.save(out)
    print(f"done {step} steps {(time.time()-t0)/60:.1f} min -> {out}")
