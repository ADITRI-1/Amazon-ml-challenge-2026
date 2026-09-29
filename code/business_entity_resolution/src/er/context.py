"""Competitor features: does candidate c fit ANOTHER S1 better than the query S1?

Uses the stage-1 candidate lists of *all* S1 in the split (reverse index on the pool record).
Every S2/S3 record has at most one owner, so a strong competing S1 is evidence against the query S1.
Only the single best competitor is used (no counts), to limit sensitivity to S1 density.
"""
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz

CTX_NAMES = ["comp_score", "comp_margin", "comp_name_tset", "comp_addr_tset", "comp_nums_eq",
             "comp_name_margin", "comp_addr_margin", "has_comp"]

_NUMRE = r"\d+"


def _strings(args):
    import re
    rows = args
    out = np.empty((len(rows), 5), np.float32)
    for i, (nc, ac, n1, a1, no, ao) in enumerate(rows):
        own_n = fuzz.token_set_ratio(nc, n1)
        own_a = fuzz.token_set_ratio(ac, a1) if ac and a1 else np.nan
        if no is None:
            out[i] = (np.nan, np.nan, np.nan, np.nan, np.nan)
            continue
        cn = fuzz.token_set_ratio(nc, no)
        ca = fuzz.token_set_ratio(ac, ao) if ac and ao else np.nan
        na, nb = set(re.findall(_NUMRE, ac)), set(re.findall(_NUMRE, ao))
        out[i] = (cn, ca, float(na == nb) if na and nb else np.nan, own_n - cn, own_a - ca)
    return out


def competitor_features(pairs: pl.DataFrame, cands_all: pl.DataFrame, recs: pl.DataFrame, procs=32) -> np.ndarray:
    """pairs: (r1, r2, score) rows to featurise (order preserved). cands_all: (r1, r2, score) for ALL S1 of the split.
    recs: (rid, name_core, addr_n). Returns (len(pairs), len(CTX_NAMES)) float32."""
    p = pairs.select("r1", "r2", pl.col("score").cast(pl.Float32)).with_row_index("i")
    rev = cands_all.select(pl.col("r1").alias("r1o"), "r2", pl.col("score").cast(pl.Float32).alias("so")) \
                   .join(p.select("r2").unique(), on="r2")
    comp = p.select("i", "r1", "r2").join(rev, on="r2").filter(pl.col("r1o") != pl.col("r1")) \
            .sort("so", descending=True).group_by("i").head(1).select("i", "r1o", "so")
    p = p.join(comp, on="i", how="left").sort("i")
    r = recs.select("rid", "name_core", "addr_n")
    p = p.join(r.rename({"name_core": "nc", "addr_n": "ac"}), left_on="r2", right_on="rid", how="left") \
         .join(r.rename({"name_core": "n1", "addr_n": "a1"}), left_on="r1", right_on="rid", how="left") \
         .join(r.rename({"name_core": "no", "addr_n": "ao"}), left_on="r1o", right_on="rid", how="left").sort("i")
    rows = p.select("nc", "ac", "n1", "a1", "no", "ao").rows()
    chunk = 50_000
    with Pool(procs) as pool:
        S = np.vstack(pool.map(_strings, [rows[i:i + chunk] for i in range(0, len(rows), chunk)]))
    so = p["so"].to_numpy().astype(np.float32)
    sc = p["score"].to_numpy().astype(np.float32)
    has = (~np.isnan(so)).astype(np.float32)
    margin = np.where(has > 0, sc - so, np.nan).astype(np.float32)
    return np.column_stack([so, margin, S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], has]).astype(np.float32)
