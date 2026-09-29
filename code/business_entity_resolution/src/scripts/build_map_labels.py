"""Map-labelled FRANCE training pairs, from the TEST set only (organiser-approved self-training; logs/france_map.md).
Real French candidate pairs (filt_test_exact) get a label only when their map class is near-deterministic:
  1: clean copy @ same address (exact/case/accent/format/reorder/web/legal-drop/stopword/typo/acronym),
     legal-form add @ same (not SNC), filler swap / filler add @ same (Fils, Services, Associés, France, Groupe,
     Développement, Cie)
  0: category-word swap @ same address; name edit + house-number shift in the decoy set {1,2,3,4,5,7,9,11,13,21};
     any SNC legal-form change
  other classes (brand, name-only, other names, number noise) are left out.
+ the full generator output (cache/synth_fr_v4_all.parquet, 727k pairs from all 259k French S1 anchors).
Output: cache/map_train_fr.parquet (text_a, text_b, y, src)
"""
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

import france_synth_v4 as fs  # noqa: E402
from er.crossencoder import pair_text  # noqa: E402
from er.io import CACHE, load_norm  # noqa: E402

FILLER = {"groupe", "france", "services", "developpement", "fils", "cie", "associes"}
CLEAN = {"exact", "case", "accent", "format", "reorder", "web", "legal_drop", "stopword", "typo", "acronym"}
DELTA = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}

te = load_norm("test").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
fr = te.filter(pl.col("country") == "France")
s1 = fr.filter(pl.col("src") == 1)
tc = Counter(t for n in s1["business_name"].to_list() for t in set(fs.toks(n)))
CAT = {t for t, c in tc.items() if c >= 150 and t not in fs.LEG_ALL and len(t) > 2 and not t.isdigit()}
stc = Counter()
for a in s1["business_address"].to_list():
    m = re.match(r"^\s*\d+\s*(?:bis|ter|quater|[A-Da-d])?\s+(\S+)", fs.street_part(a))
    if m:
        stc[fs.fold(m.group(1)).lower().strip(".")] += 1
stypes = {t for t, c in stc.items() if c >= fs.STREET_WORDS_MIN}
T = pl.read_parquet(CACHE / "filt_test_exact.parquet").select("r1", "r2").join(fr.select(pl.col("rid").alias("r1")), on="r1")
R = te.select("rid", "business_name", "business_address")
T = T.join(R.rename({"rid": "r1", "business_name": "n1", "business_address": "a1"}), on="r1").join(R.rename({"rid": "r2", "business_name": "n2", "business_address": "a2"}), on="r2")
items = [(n1 or "", a1 or "", n2 or "", a2 or "", stypes) for n1, a1, n2, a2 in T.select("n1", "a1", "n2", "a2").iter_rows()]
with Pool(24) as pool:
    res = pool.map(fs._classify, items, chunksize=5000)


def label(op, det, ar, na, nb):
    if "snc" in (det or "").split(",") and op in ("legal_add", "legal_swap"):
        return 0, "snc"
    if ar in ("numdiff", "numtypo") and na is not None and nb is not None and abs(nb - na) in DELTA and op not in CLEAN:
        return 0, "numshift_decoy"
    if ar != "same":
        return None, None
    if op in CLEAN:
        return 1, "clean"
    if op == "legal_add":
        return 1, "legal_add"
    if op == "swap":
        s, tg = det.split(">")
        if tg in FILLER:
            return 1, "M_swap"
        if s in CAT and tg in CAT:
            return 0, "catswap"
        return None, None
    if op == "add" and set(det.split()) & FILLER:
        return 1, "M_add"
    return None, None


lab = [label(*r) for r in res]
T = T.with_columns(pl.Series("y", [x[0] for x in lab], dtype=pl.Float32), pl.Series("cls", [x[1] for x in lab]))
real = T.filter(pl.col("y").is_not_null())
print("real map-labelled:", real.height, dict(real.group_by("cls").len().rows()), "pos", round(real["y"].mean(), 3))
real = real.select(pl.struct("n1", "a1").map_elements(lambda r: pair_text(r["n1"], r["a1"]), return_dtype=pl.Utf8).alias("text_a"),
                   pl.struct("n2", "a2").map_elements(lambda r: pair_text(r["n2"], r["a2"]), return_dtype=pl.Utf8).alias("text_b"),
                   "y", pl.lit("real_map").alias("src"))
syn = pl.read_parquet(CACHE / "synth_fr_v4_all.parquet").select("text_a", "text_b", pl.col("y").cast(pl.Float32), pl.lit("synth_v4").alias("src"))
out = pl.concat([real, syn]).unique(subset=["text_a", "text_b"], keep="first").sample(fraction=1.0, shuffle=True, seed=2026)
out.write_parquet(CACHE / "map_train_fr.parquet")
print(f"map_train_fr: {out.height:,} pairs | pos {out['y'].mean():.3f} | by src {dict(out.group_by('src').len().rows())}")
