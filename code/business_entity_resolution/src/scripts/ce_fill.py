"""Cross-encoder logits for every pair of a filtered candidate file, reusing already-scored pairs.

usage: python scripts/ce_fill.py --split test --filt filt_test --reuse ce_test --tag test_v2
out:   cache/ce_<tag>.parquet (r1, r2, ce) covering all pairs of cache/<filt>.parquet
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from er.crossencoder import CrossEncoder, HFCrossEncoder, ce_score, pair_text  # noqa: E402
from er.io import CACHE, ROOT, load_norm  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--filt", required=True)
    ap.add_argument("--reuse", default="", help="comma list of existing ce_* files to reuse")
    ap.add_argument("--model", default="models/ce_e5s_v1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hf", action="store_true", help="--model is a native HF sequence-classification reranker (bge kit)")
    ap.add_argument("--bs", type=int, default=1024)
    a = ap.parse_args()
    t0 = time.time()
    need = pl.read_parquet(CACHE / f"{a.filt}.parquet").select("r1", "r2").unique()
    have = pl.concat([pl.read_parquet(CACHE / f"{t}.parquet").select("r1", "r2", "ce") for t in a.reuse.split(",") if t]) \
        if a.reuse else pl.DataFrame(schema={"r1": pl.UInt32, "r2": pl.UInt32, "ce": pl.Float32})
    have = have.with_columns(pl.col("r1").cast(pl.UInt32), pl.col("r2").cast(pl.UInt32)).unique(["r1", "r2"])
    got = need.join(have, on=["r1", "r2"])
    miss = need.join(have, on=["r1", "r2"], how="anti")
    print(f"{a.split}: need {need.height:,} | reused {got.height:,} | to score {miss.height:,}", flush=True)
    parts = [got]
    if miss.height:
        df = load_norm(a.split).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).select("rid", "business_name", "business_address")
        TA = [pair_text(n, s) for n, s in miss.join(df, left_on="r1", right_on="rid", how="left", maintain_order="left").select("business_name", "business_address").rows()]
        TB = [pair_text(n, s) for n, s in miss.join(df, left_on="r2", right_on="rid", how="left", maintain_order="left").select("business_name", "business_address").rows()]
        parts.append(miss.with_columns(pl.Series("ce", ce_score((HFCrossEncoder if a.hf else CrossEncoder)(str(ROOT / a.model)), TA, TB, bs=a.bs), dtype=pl.Float32)))
    R = pl.concat(parts)
    assert R.height == need.height
    R.write_parquet(CACHE / f"ce_{a.tag}.parquet")
    print(f"saved cache/ce_{a.tag}.parquet {R.height:,} rows in {time.time()-t0:.0f}s", flush=True)
