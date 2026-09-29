"""TF-IDF pair features (organiser-allowed unsupervised statistics). IDF is computed per country from the SAME split's
records (train split for India/US, test split for France), over the normalised name / address tokens of all sources.
  tf_name_cos   : cosine of binary TF-IDF name-token vectors
  tf_name_wjac  : IDF-weighted Jaccard of name tokens
  tf_miss_max   : largest IDF among name tokens present in only one of the two names (0 if none)
  tf_miss_sum   : sum of IDF of those mismatched name tokens
  tf_addr_cos   : cosine of binary TF-IDF address-token vectors
"""
import math
from multiprocessing import Pool

import numpy as np
import polars as pl

TF_NAMES = ["tf_name_cos", "tf_name_wjac", "tf_miss_max", "tf_miss_sum", "tf_addr_cos"]
_G = {}


def _tok(s):
    return frozenset(t for t in (s or "").split() if t)


def build_idf(df):
    """df: normalised frame with country, name_n, addr_n (all sources). Returns {(country, field): {token: idf}}."""
    out = {}
    for field in ["name_n", "addr_n"]:
        t = df.select("country", pl.col(field).fill_null("").str.split(" ").list.unique().alias("t")).with_row_index("i")
        n = t.group_by("country").len()
        dfq = t.explode("t").filter(pl.col("t") != "").group_by("country", "t").len().rename({"len": "df"})
        dfq = dfq.join(n, on="country").with_columns((pl.col("len") / pl.col("df")).log().alias("idf"))
        for c in dfq["country"].unique().to_list():
            s = dfq.filter(pl.col("country") == c)
            out[(c, field)] = dict(zip(s["t"].to_list(), s["idf"].to_list()))
    return out


def _cos(a, b, w, dflt):
    if not a or not b:
        return -1.0
    na = math.sqrt(sum(w.get(t, dflt) ** 2 for t in a)); nb = math.sqrt(sum(w.get(t, dflt) ** 2 for t in b))
    return sum(w.get(t, dflt) ** 2 for t in a & b) / (na * nb) if na and nb else 0.0


def _chunk(rows):
    idf = _G["idf"]; out = np.zeros((len(rows), len(TF_NAMES)), np.float32)
    for i, (c, n1, n2, a1, a2) in enumerate(rows):
        wn = idf.get((c, "name_n"), {}); wa = idf.get((c, "addr_n"), {})
        dn = _G["dflt"][(c, "name_n")]; da = _G["dflt"][(c, "addr_n")]
        A, B = _tok(n1), _tok(n2)
        un = A | B; inter = A & B; miss = un - inter
        su = sum(wn.get(t, dn) for t in un)
        out[i, 0] = _cos(A, B, wn, dn)
        out[i, 1] = sum(wn.get(t, dn) for t in inter) / su if su else 0.0
        out[i, 2] = max((wn.get(t, dn) for t in miss), default=0.0)
        out[i, 3] = sum(wn.get(t, dn) for t in miss)
        out[i, 4] = _cos(_tok(a1), _tok(a2), wa, da)
    return out


def tfidf_features(pairs, recs, idf, workers=24):
    """pairs: (r1, r2) in row order; recs: (rid, country, name_n, addr_n). Returns float32 matrix aligned with pairs."""
    _G["idf"] = idf
    _G["dflt"] = {k: (max(v.values()) if v else 1.0) for k, v in idf.items()}   # unseen token = rarest
    P = pairs.select("r1", "r2").join(recs.rename({"rid": "r1", "name_n": "n1", "addr_n": "a1"}), on="r1", how="left", maintain_order="left") \
             .join(recs.drop("country").rename({"rid": "r2", "name_n": "n2", "addr_n": "a2"}), on="r2", how="left", maintain_order="left")
    rows = P.select("country", "n1", "n2", "a1", "a2").rows()
    step = 50_000
    with Pool(workers) as pool:
        parts = pool.map(_chunk, [rows[i:i + step] for i in range(0, len(rows), step)])
    return np.vstack(parts)
