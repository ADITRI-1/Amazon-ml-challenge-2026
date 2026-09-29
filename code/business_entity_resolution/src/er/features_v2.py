"""Address/name features v2 — targets the unseen-country (France) gaps, country-agnostic.

Per record (cached per split in cache/<split>_v2.parquet):
  a_canon   : normalised address tokens with street-type abbreviations mapped to one form (r→rue, av/ave→avenue,
              bd/blvd→boulevard, pl→place, ch/chem→chemin, rte→route, imp→impase, rd→road, dr→drive, ln→lane, ...)
  a_noadmin : a_canon minus every comma component that is very frequent in the split and has no digits
              (regions / départements / states / big cities: the "admin tail" that differs between sources)
  n_core2   : core name with extra legal forms stripped (selarl, scop, ets, cie, sca, etablissements)
Pair features (V2_NAMES): token-set / jaccard on a_canon and a_noadmin, and core-name token-set.
Names/addresses are already ascii-folded, lowercased and repeat-collapsed by norm.basic (street→stret, allee→ale).
"""
from collections import Counter
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from er.norm import basic

STREET_MAP = {
    "r": "rue", "av": "avenue", "ave": "avenue", "avn": "avenue", "bd": "boulevard", "blvd": "boulevard", "boul": "boulevard",
    "al": "ale", "pl": "place", "imp": "impase", "ch": "chemin", "chem": "chemin", "rte": "route", "q": "quai",
    "res": "residence", "sq": "square", "fg": "faubourg", "fbg": "faubourg",
    "rd": "road", "dr": "drive", "ln": "lane", "ct": "court", "hwy": "highway", "pkwy": "parkway", "cir": "circle",
    "trl": "trail", "ter": "terace", "pk": "park", "nr": "near", "opp": "oposite", "apt": "apartment", "bldg": "building",
}
EXTRA_LEGAL = {"selarl", "scop", "ets", "cie", "sca", "etablisements", "etablisement"}
V2_NAMES = ["a2_canon_tset", "a2_canon_jacc", "a2_noadmin_tset", "a2_noadmin_contain_b", "n2_core_tset"]


def _canon(tokens):
    return [STREET_MAP.get(t, t) for t in tokens]


def _rec(args):
    raw_addr, name_core, admin = args
    comps = [basic(c) for c in raw_addr.split(",") if c.strip()]
    toks = _canon(" ".join(comps).split())
    keep = [c for c in comps if not (c in admin and not any(ch.isdigit() for ch in c))]
    na = _canon(" ".join(keep).split())
    core2 = " ".join(t for t in name_core.split() if t not in EXTRA_LEGAL)
    return " ".join(toks), " ".join(na), core2


def build_v2(df: pl.DataFrame, min_count=5000, procs=36) -> pl.DataFrame:
    """df: norm cache frame (with rid, business_address, name_core). Admin vocabulary = frequent digit-free components."""
    cnt = Counter()
    for a in df["business_address"].to_list():
        for c in a.split(","):
            c = c.strip()
            if c:
                cnt[c.lower()] += 1
    admin = {basic(c) for c, n in cnt.items() if n >= min_count and not any(ch.isdigit() for ch in c)}
    items = [(a, n, admin) for a, n in zip(df["business_address"].to_list(), df["name_core"].to_list())]
    with Pool(procs) as p:
        rows = p.map(_rec, items, chunksize=20000)
    a, na, c2 = zip(*rows)
    return df.select("rid").with_columns(pl.Series("a_canon", a), pl.Series("a_noadmin", na), pl.Series("n_core2", c2)), len(admin)


def _pair(args):
    out = np.empty((len(args), len(V2_NAMES)), np.float32)
    for i, (ac, bc, an, bn, n1, n2) in enumerate(args):
        sa, sb = set(ac.split()), set(bc.split())
        na, nb = set(an.split()), set(bn.split())
        out[i, 0] = fuzz.token_set_ratio(ac, bc) if ac and bc else np.nan
        out[i, 1] = len(sa & sb) / len(sa | sb) if sa and sb else np.nan
        out[i, 2] = fuzz.token_set_ratio(an, bn) if an and bn else np.nan
        out[i, 3] = len(na & nb) / len(nb) if na and nb else np.nan
        out[i, 4] = fuzz.token_set_ratio(n1, n2)
    return out


def pair_features_v2(C: pl.DataFrame, V: pl.DataFrame, procs=32) -> np.ndarray:
    """C: (r1, r2) rows in order; V: per-record v2 frame (rid, a_canon, a_noadmin, n_core2)."""
    j = C.select("r1", "r2").with_row_index("_i") \
         .join(V.rename({"rid": "r1", "a_canon": "ac", "a_noadmin": "an", "n_core2": "n1"}), on="r1", how="left") \
         .join(V.rename({"rid": "r2", "a_canon": "bc", "a_noadmin": "bn", "n_core2": "n2"}), on="r2", how="left").sort("_i")
    rows = j.select("ac", "bc", "an", "bn", "n1", "n2").fill_null("").rows()
    ch = 50_000
    with Pool(procs) as p:
        return np.vstack(p.map(_pair, [rows[i:i + ch] for i in range(0, len(rows), ch)]))
