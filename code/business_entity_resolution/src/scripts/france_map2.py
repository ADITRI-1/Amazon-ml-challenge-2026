"""MAP2 France (label-free map rules, logs/france_map.md), on top of the MAP France (LB 0.987):
  base  = 0.984 France (m6 on exact blocking) - same-address category swaps + rejected same-address filler variants
  + ADD rejected clean-name pairs whose street is written differently (map P 0.9-1.0) or whose house number carries
        non-decoy noise (map P 0.8)                                  [record not claimed by another S1]
  - DROP accepted different-name @ same address, multi-word edits @ same address, different name + number dropped
        (map P 0.1-0.2)
India/US unchanged (= output_m8xhc). Writes submission/output_m8xhcMAP2.
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
from er.io import CACHE, load_norm  # noqa: E402

FILLER = {"groupe", "france", "services", "developpement", "fils", "cie", "associes"}
CLEAN = {"exact", "case", "accent", "format", "reorder", "web", "legal_drop", "stopword", "typo", "acronym"}
DELTA = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
te = load_norm("test").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
fr = te.filter(pl.col("country") == "France"); s1 = fr.filter(pl.col("src") == 1)
tc = Counter(t for n in s1["business_name"].to_list() for t in set(fs.toks(n)))
CAT = {t for t, c in tc.items() if c >= 150 and t not in fs.LEG_ALL and len(t) > 2 and not t.isdigit()}
stc = Counter()
for a in s1["business_address"].to_list():
    m = re.match(r"^\s*\d+\s*(?:bis|ter|quater|[A-Da-d])?\s+(\S+)", fs.street_part(a))
    if m:
        stc[fs.fold(m.group(1)).lower().strip(".")] += 1
stypes = {t for t, c in stc.items() if c >= fs.STREET_WORDS_MIN}
T = pl.read_parquet(CACHE / "testpred_m6x.parquet").join(fr.select(pl.col("rid").alias("r1")), on="r1").select("r1", "r2", "p")
R = te.select("rid", "business_name", "business_address")
T = T.join(R.rename({"rid": "r1", "business_name": "n1", "business_address": "a1"}), on="r1").join(R.rename({"rid": "r2", "business_name": "n2", "business_address": "a2"}), on="r2")
with Pool(24) as pool:
    res = pool.map(fs._classify, [(a or "", b or "", c or "", d or "", stypes) for a, b, c, d in T.select("n1", "a1", "n2", "a2").iter_rows()], chunksize=5000)


def cls(op, det, ar, na, nb):
    if ar == "empty":
        return "name-only"
    shift = ar in ("numdiff", "numtypo") and na is not None and nb is not None and abs(nb - na) in DELTA
    if op in CLEAN:
        if ar == "same":
            return "clean@same"
        if ar in ("numdiff", "numtypo") and not shift:
            return "ADD:clean+num-noise"
        if ar == "strdiff":
            return "ADD:clean+street-diff"
        return "clean+other-addr"
    if shift:
        return "numshift"
    if ar == "same":
        if op == "swap":
            s, tg = det.split(">")
            if tg in FILLER:
                return "M_swap"
            if s in CAT and tg in CAT:
                return "catswap"
            return "swap_other"
        if op == "add" and set(det.split()) & FILLER:
            return "M_add"
        if op == "diffname" and len([x for x in fs.toks(det) if x not in fs.LEG_ALL]) > 1:
            return "DROP:other-name@same"
        if op == "multi":
            return "DROP:multi-edit@same"
        return "other@same"
    if op == "diffname" and ar == "numdrop":
        return "DROP:other-name+numdrop"
    return "other"


T = T.with_columns(pl.Series("cls", [cls(*r) for r in res]), (pl.col("p") >= 0.725).alias("acc"))
keep = T.filter((pl.col("acc") & ~pl.col("cls").is_in(["catswap", "DROP:other-name@same", "DROP:multi-edit@same", "DROP:other-name+numdrop"]))
                | (~pl.col("acc") & pl.col("cls").is_in(["M_swap", "M_add"])))
keep = keep.sort("p", descending=True).unique(subset="r2", keep="first")
adds = T.filter(~pl.col("acc") & pl.col("cls").is_in(["ADD:clean+street-diff", "ADD:clean+num-noise"])).join(keep.select("r2"), on="r2", how="anti") \
        .sort("p", descending=True).unique(subset="r2", keep="first")
print("MAP base (after ownership):", keep.height, "| rule adds:", adds.height, dict(adds.group_by("cls").len().rows()),
      "| rule drops:", dict(T.filter(pl.col("acc") & pl.col("cls").str.starts_with("DROP")).group_by("cls").len().rows()))
F = pl.concat([keep.select("r1", "r2"), adds.select("r1", "r2")])
ids = te.select("rid", "entity_id")
P = F.join(ids.rename({"rid": "r1", "entity_id": "s1"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "id"}), on="r2").select("s1", "id")
import build_france_variants as bfv  # noqa: E402
bfv.write(P, "output_m8xhcMAP2")
