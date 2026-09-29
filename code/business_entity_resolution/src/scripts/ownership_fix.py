"""Ownership post-processing: every S2/S3 record belongs to at most one S1, so when a record is predicted for several
S1, keep only the claim with the highest probability (the others are certain errors whenever the record is owned).

usage: python scripts/ownership_fix.py --pred testpred_m6 --th 0.725 --src submission/output_m6 --out submission/output_m6o
Keeps candidate_pairs.tsv identical to --src (blocking unchanged).
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from er.io import CACHE, ROOT, load_norm  # noqa: E402
from predict import write_lists  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--th", type=float, required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    te = load_norm("test").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    P = pl.read_parquet(CACHE / f"{a.pred}.parquet").filter(pl.col("p") >= a.th)
    before = P.height
    P = P.sort("p", descending=True).unique(subset="r2", keep="first", maintain_order=True)
    removed = before - P.height
    cty = te.select(pl.col("rid").alias("r1"), "country")
    print(f"predicted pairs {before:,} → {P.height:,} (removed {removed:,} lower-probability duplicate claims)")
    print("remaining pairs by country:", P.join(cty, on="r1").group_by("country").len().sort("country").rows())
    ids = te.select("rid", "entity_id")
    P = P.join(ids.rename({"rid": "r1", "entity_id": "source1_entity_id"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "eid"}), on="r2")
    s1_ids = te.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("source1_entity_id"))
    o = ROOT / a.out
    o.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / a.src / "candidate_pairs.tsv", o / "candidate_pairs.tsv")
    mr = write_lists(P.sort("r1", "rank"), s1_ids, "matched_entity_ids", o / "matching_results.tsv")
    print(f"wrote {o}: {(mr['matched_entity_ids'] != '').sum():,} S1 with a match")
