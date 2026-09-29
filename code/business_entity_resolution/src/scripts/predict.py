"""Test inference: stage-1 candidates -> matcher -> output/matching_results.tsv + output/candidate_pairs.tsv

candidate_pairs.tsv = the exact top-K list the matcher scores (last filtering stage), per the README.

usage: python scripts/predict.py --matcher models/matcher_m1/lgb.txt --k 30 --th 0.675 [--out output]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from er.context import competitor_features  # noqa: E402
from er.features import FIELDS, pair_features  # noqa: E402
from er.io import CACHE, ROOT, load_norm  # noqa: E402


def write_lists(df, s1_ids, col, path):
    """one row per S1 (all of them), comma-joined unique ids, empty when none."""
    agg = df.group_by("source1_entity_id").agg(pl.col("eid").unique(maintain_order=True).str.join(",").alias(col))
    out = s1_ids.join(agg, on="source1_entity_id", how="left").fill_null("")
    out.write_csv(path, separator="\t", quote_style="never")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", required=True)
    ap.add_argument("--cands", default="cands_test_v1")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--th", type=float, required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--ctx", action="store_true", help="matcher uses competitor features (m2+)")
    ap.add_argument("--ctx_k", type=int, default=50)
    ap.add_argument("--ce", default="", help="cross-encoder score tag for test pairs (cache/ce_<tag>.parquet)")
    ap.add_argument("--cand_file", default="", help="pre-filtered candidates = the final candidate set (no top-k cut)")
    ap.add_argument("--ctx_file", default="", help="candidate lists for competitor features (default: --cands)")
    a = ap.parse_args()
    t0 = time.time()
    te = load_norm("test").with_row_index("rid")
    recs = te.select(pl.col("rid").cast(pl.UInt32), *FIELDS)
    ids = te.select(pl.col("rid").cast(pl.UInt32), "entity_id")
    C_full = pl.read_parquet(CACHE / f"{a.ctx_file or a.cands}.parquet")
    C = pl.read_parquet(CACHE / f"{a.cand_file}.parquet").select("r1", "r2", "rank", "score") if a.cand_file else C_full.filter(pl.col("rank") < a.k)
    C_all = C_full.filter(pl.col("rank") < a.ctx_k).select("r1", "r2", "score") if a.ctx else None
    del C_full
    CE = pl.read_parquet(CACHE / f"ce_{a.ce}.parquet") if a.ce else None
    print(f"candidates {C.height:,}", flush=True)

    mdl = lgb.Booster(model_file=a.matcher)
    fj = Path(a.matcher).with_name("features.json")
    keep = None
    if fj.exists():
        meta = json.load(open(fj))
        keep = [meta["all_features"].index(n) for n in meta["kept"]]
    P = []
    step = 5_000_000  # feature building in chunks to bound memory
    for i in range(0, C.height, step):
        c = C.slice(i, step)
        A = c.join(recs, left_on="r1", right_on="rid", how="left", maintain_order="left").select(FIELDS).rows()
        B = c.join(recs, left_on="r2", right_on="rid", how="left", maintain_order="left").select(FIELDS).rows()
        X = np.hstack([pair_features(A, B, procs=36), c["score"].cast(pl.Float32).to_numpy()[:, None]])
        if C_all is not None:
            X = np.hstack([X, competitor_features(c, C_all, recs.select("rid", "name_core", "addr_n"), procs=36)])
        if CE is not None:
            j = c.select("r1", "r2").with_row_index("_i").join(CE, on=["r1", "r2"], how="left").sort("_i")
            assert j.height == c.height and j["ce"].null_count() == 0, "missing CE scores"
            X = np.hstack([X, j["ce"].to_numpy().astype(np.float32)[:, None]])
        if keep is not None:
            X = X[:, keep]
        P.append(mdl.predict(X, num_threads=36))
        print(f"  scored {min(i + step, C.height):,} / {C.height:,}  {time.time()-t0:.0f}s", flush=True)
    C = C.with_columns(pl.Series("p", np.concatenate(P)))
    C.write_parquet(CACHE / f"testpred_{Path(a.matcher).parent.name}.parquet")
    assert mdl.num_feature() == X.shape[1], (mdl.num_feature(), X.shape[1])

    C = C.join(ids.rename({"rid": "r1", "entity_id": "source1_entity_id"}), on="r1") \
         .join(ids.rename({"rid": "r2", "entity_id": "eid"}), on="r2")
    s1_ids = te.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("source1_entity_id"))
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    cp = write_lists(C.sort("r1", "rank"), s1_ids, "candidate_entity_ids", out / "candidate_pairs.tsv")
    mr = write_lists(C.filter(pl.col("p") >= a.th).sort("r1", "rank"), s1_ids, "matched_entity_ids", out / "matching_results.tsv")
    n_match = (mr["matched_entity_ids"] != "").sum()
    print(f"S1 rows {mr.height:,} | with >=1 match {n_match:,} ({n_match / mr.height:.3f}) | "
          f"pairs predicted {C.filter(pl.col('p') >= a.th).height:,} | {time.time()-t0:.0f}s")
    by_c = C.join(te.select(pl.col("rid").cast(pl.UInt32).alias("r1"), "country"), on="r1") \
            .group_by("country").agg((pl.col("p") >= a.th).sum().alias("pred_pairs"), pl.col("r1").n_unique().alias("s1"))
    print(by_c.with_columns((pl.col("pred_pairs") / pl.col("s1")).round(2).alias("pred_per_S1")))
