"""Map-guided France variant (label-free; logs/france_map.md): starting from the 0.984 France predictions (output_m6xo),
  - REMOVE same-address category-word swaps (catswap; map P(match) ~0.1, model accepts ~65%)
  - ADD same-address filler swaps / filler adds (M_swap / M_add; map P ~0.95-1.0) that the model rejected,
    only if the record is not already claimed by another S1 (ownership)
India/US unchanged (= output_m8xhc). Writes submission/output_m8xhcMAP + a per-class table.
Classifier = scripts/france_synth_v4.py name_op/addr_op (model-free); CAT = tokens in >=150 French S1 names.
"""
import argparse
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

import france_synth_v4 as fs  # noqa: E402
from er.io import CACHE, load_norm  # noqa: E402

FILLER = {"groupe", "france", "services", "developpement", "fils", "cie", "associes"}
ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="testpred_m6x", help="France probabilities of m6 on the chosen candidate set")
ap.add_argument("--base", default="output_m8xhc", help="submission folder that supplies India/US rows + candidate_pairs.tsv")
ap.add_argument("--out", default="output_m8xhcMAP")
ap.add_argument("--cls_out", default="france_map_cls")
A = ap.parse_args()

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
T = pl.read_parquet(CACHE / f"{A.pred}.parquet").join(fr.select(pl.col("rid").alias("r1")), on="r1").select("r1", "r2", "p")
R = te.select("rid", "business_name", "business_address")
T = T.join(R.rename({"rid": "r1", "business_name": "n1", "business_address": "a1"}), on="r1").join(R.rename({"rid": "r2", "business_name": "n2", "business_address": "a2"}), on="r2")
items = [(n1 or "", a1 or "", n2 or "", a2 or "", stypes) for n1, a1, n2, a2 in T.select("n1", "a1", "n2", "a2").iter_rows()]
with Pool(24) as pool:
    res = pool.map(fs._classify, items, chunksize=5000)


def cls(op, det, ar):
    if ar == "empty":
        return "name-only"
    if ar != "same":
        return "other"
    if op == "swap":
        s, tg = det.split(">")
        if tg in FILLER:
            return "M_swap@same"
        if s in CAT and tg in CAT:
            return "catswap@same"
        return "swap_other@same"
    if op == "add" and set(det.split()) & FILLER:
        return "M_add@same"
    return "other@same"


T = T.with_columns(pl.Series("cls", [cls(o, d, a) for o, d, a, _, _ in res]))
T = T.with_columns((pl.col("p") >= 0.725).alias("pred"))
print(T.group_by("cls").agg(pl.len(), pl.col("pred").mean().round(3).alias("model_accepts"), pl.col("pred").sum().alias("n_accepted")).sort("len", descending=True))
ids = te.select("rid", "entity_id")
import build_france_variants as bfv  # noqa: E402
cur = T.filter(pl.col("pred")).sort("p", descending=True).unique(subset="r2", keep="first")     # = 0.984 France (m6xo)
drop = cur.filter(pl.col("cls") == "catswap@same")
keep = cur.filter(pl.col("cls") != "catswap@same")
adds = T.filter(~pl.col("pred") & pl.col("cls").is_in(["M_swap@same", "M_add@same"])).join(keep.select("r2"), on="r2", how="anti") \
        .sort("p", descending=True).unique(subset="r2", keep="first")
print(f"0.984 France {cur.height:,} | remove catswap@same {drop.height:,} | add filler@same {adds.height:,}")
F = pl.concat([keep.select("r1", "r2"), adds.select("r1", "r2")])
P = F.join(ids.rename({"rid": "r1", "entity_id": "s1"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "id"}), on="r2").select("s1", "id")
bfv.write(P, A.out, A.base)
# also the two halves separately (to attribute a leaderboard change)
P1 = keep.join(ids.rename({"rid": "r1", "entity_id": "s1"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "id"}), on="r2").select("s1", "id")
if A.out == "output_m8xhcMAP":                       # historical attribution upload
    bfv.write(P1, "output_m8xhcMAPdrop")
T.select("r1", "r2", "p", "cls").write_parquet(CACHE / f"{A.cls_out}.parquet")
