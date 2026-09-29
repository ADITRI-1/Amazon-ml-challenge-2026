"""Blocking stage 2: prune each S1's ScaNN top-30 with the cheap m2 model (pair features + competitor features,
no cross-encoder). Survivors (p >= eps) are the final candidate set = candidate_pairs.tsv, and the only pairs the
cross-encoder and the final matcher score.

usage:
  python scripts/filter_candidates.py --split test  --queries all --tag test
  python scripts/filter_candidates.py --split train --queries ce_tr59,fold0 --tag train
out: cache/filt_<tag>.parquet (r1, r2, rank, score, p2)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from er.features import FIELDS  # noqa: E402
from er.io import CACHE, ROOT, load_norm  # noqa: E402
from train_matcher import build  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--queries", default="all", help="'all' or comma list of: ce_<tag> (S1 in that CE file), fold<n>")
    ap.add_argument("--cands", default="")
    ap.add_argument("--model", default="models/matcher_m2/lgb.txt")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--ctx_k", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chunk", type=int, default=4_000_000)
    a = ap.parse_args()
    t0 = time.time()
    df = load_norm(a.split).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    recs = df.select("rid", *FIELDS)
    Cf = pl.read_parquet(CACHE / f"{a.cands or 'cands_' + a.split + '_scann'}.parquet")
    C_all = Cf.filter(pl.col("rank") < a.ctx_k).select("r1", "r2", "score")
    C = Cf.filter(pl.col("rank") < a.k)
    del Cf
    if a.queries != "all":
        parts = []
        split = pl.read_parquet(CACHE / "split.parquet")
        s1 = df.filter(pl.col("src") == 1).select(pl.col("rid").alias("r1"), "entity_id").join(split, on="entity_id")
        for qd in a.queries.split(","):
            if qd.startswith("ce_"):
                parts.append(pl.read_parquet(CACHE / f"{qd}.parquet").select("r1").unique())
            elif qd.startswith("fold"):
                parts.append(s1.filter(pl.col("fold") == int(qd[4:])).select("r1"))
        C = C.join(pl.concat(parts).unique(), on="r1")
    print(f"{a.split}: scoring {C.height:,} pairs of {C['r1'].n_unique():,} S1 with {a.model}", flush=True)
    mdl = lgb.Booster(model_file=str(ROOT / a.model))
    no_truth = pl.DataFrame({"r1": [], "r2": []}, schema={"r1": pl.UInt32, "r2": pl.UInt32})
    out = []
    for i in range(0, C.height, a.chunk):
        c = C.slice(i, a.chunk)
        c2, X = build(c, recs, no_truth, C_all)
        assert mdl.num_feature() == X.shape[1], (mdl.num_feature(), X.shape[1])
        out.append(c2.select("r1", "r2", "rank", "score").with_columns(pl.Series("p2", mdl.predict(X, num_threads=36), dtype=pl.Float32)))
        print(f"  {min(i + a.chunk, C.height):,}/{C.height:,}  {time.time()-t0:.0f}s", flush=True)
    R = pl.concat(out)
    K = R.filter(pl.col("p2") >= a.eps)
    K.write_parquet(CACHE / f"filt_{a.tag}.parquet")
    print(f"kept {K.height:,} / {R.height:,} pairs ({K.height / R['r1'].n_unique():.2f} per S1) at p2>={a.eps} | {time.time()-t0:.0f}s", flush=True)
