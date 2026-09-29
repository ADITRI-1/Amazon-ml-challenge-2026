"""Integrate Laya-FR scores (laya_fr_kit/out/scores.parquet; model trained on map-labelled France TEST data) into France.
  A  "stack": saved m6 matcher with the Laya v2 slot filled by Laya-FR logits (France pairs), th 0.725, ownership
  B  "direct": Laya-FR logit > 0 decides every French pair, ownership by logit
Both written as full submissions (India/US = output_m8xhc), plus a class-wise diff against the 0.987 MAP France.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb  # noqa: E402
import polars as pl  # noqa: E402

import build_france_variants as bfv  # noqa: E402
from ensemble_matcher import matrices  # noqa: E402
from er.io import CACHE, ROOT, load_norm  # noqa: E402

meta = json.load(open(ROOT / "laya_fr_kit/out/train_meta.json"))
print("Laya-FR dev AUC history:", [(h["step"], round(h["dev_auc"], 5)) for h in meta["dev_history"]], "| minutes", meta.get("minutes"))
K = pl.read_parquet(CACHE / "layafr_score_keys.parquet").join(pl.read_parquet(ROOT / "laya_fr_kit/out/scores.parquet"), on="key")
print(f"scores {K.height:,} | non-finite {K['ce'].is_nan().sum()}")
base = pl.read_parquet(CACHE / "ce_laya2_test_exact.parquet")
base.join(K.select("r1", "r2", pl.col("ce").alias("ce_new")), on=["r1", "r2"], how="left") \
    .with_columns(pl.coalesce("ce_new", "ce").alias("ce")).drop("ce_new").write_parquet(CACHE / "ce_layafr_test_exact.parquet")
te = load_norm("test").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
ids = te.select("rid", "entity_id")
none = pl.DataFrame(schema={"r1": pl.UInt32, "r2": pl.UInt32})
Cs, Xs = matrices("test", "filt_test_exact_fr", "cands_test_exact", "test_v2_exact", ["laya_test_exact", "layafr_test_exact"], none)
m6 = lgb.Booster(model_file=str(ROOT / "models/matcher_m6/lgb_seed0.txt"))
A = Cs.select("r1", "r2").with_columns(pl.Series("p", m6.predict(Xs, num_threads=24))).filter(pl.col("p") >= 0.725)
B = K.select("r1", "r2", pl.col("ce").alias("p")).filter(pl.col("p") > 0)
C = pl.read_parquet(CACHE / "france_map_cls.parquet").select("r1", "r2", "cls")
mapfr = bfv.load("output_m8xhcMAP").join(bfv.cty, on="s1").filter(pl.col("country") == "France").select("s1", "id")
for tag, T in [("output_m8xhcLFa", A), ("output_m8xhcLFb", B)]:
    T = T.sort("p", descending=True).unique(subset="r2", keep="first")
    P = T.join(ids.rename({"rid": "r1", "entity_id": "s1"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "id"}), on="r2").select("s1", "id")
    bfv.write(P, tag)
    add = P.join(mapfr, on=["s1", "id"], how="anti"); drop = mapfr.join(P, on=["s1", "id"], how="anti")
    back = lambda D: D.join(ids.rename({"entity_id": "s1", "rid": "r1"}), on="s1").join(ids.rename({"entity_id": "id", "rid": "r2"}), on="id").join(C, on=["r1", "r2"], how="left")
    print(f"{tag}: France {P.height:,} vs MAP {mapfr.height:,} | +{add.height:,} {dict(back(add).group_by('cls').len().sort('len', descending=True).rows())}")
    print(f"      -{drop.height:,} {dict(back(drop).group_by('cls').len().sort('len', descending=True).rows())}")
