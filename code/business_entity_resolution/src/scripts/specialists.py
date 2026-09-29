"""Two teammate experiments on top of m8 (val only; test only if they win):
 C  empty-address specialist: LightGBM trained ONLY on pairs whose S2/S3 record has no address, on m8's 51 features
    + same_name_s1 (number of S1 in the country with the identical normalised name = ambiguity). Own threshold.
    Address-present pairs keep m8 unchanged.
 B  singleton / no-match model: per-S1 classifier "has no true match" from m8 probabilities of its candidates
    (trained on 5-fold OOF m8 probs). S1 predicted singleton above a cut get an empty list.
usage: python scripts/specialists.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ensemble_matcher import matrices  # noqa: E402
from er.io import CACHE, ROOT, load_gt_pairs, load_norm  # noqa: E402
from er.metrics import f05_per_entity  # noqa: E402

EXTRA = ["laya", "laya2", "bge"]
TH8 = 0.75
t0 = time.time()
P8 = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200, "feature_fraction": 0.8,
      "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 24, "seed": 0}

tr = load_norm("train").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
idmap = tr.select("entity_id", "rid")
truth = load_gt_pairs().join(idmap, left_on="id1", right_on="entity_id").rename({"rid": "r1"}) \
                       .join(idmap, left_on="id2", right_on="entity_id").rename({"rid": "r2"}).select("r1", "r2")
Ct, Xt = matrices("train", "filt_tr", "cands_train_scann", "tr_v2", [f"{f}_tr" for f in EXTRA], truth)
Cv, Xv = matrices("train", "filt_va", "cands_train_scann", "va_v2", [f"{f}_va" for f in EXTRA], truth)
m8 = lgb.Booster(model_file=str(ROOT / "models/matcher_m8/lgb_seed0.txt"))
pv = m8.predict(Xv, num_threads=24)
print(f"matrices + m8 predict | {time.time()-t0:.0f}s", flush=True)

rec = tr.select(pl.col("rid"), "country", "name_n", (pl.col("addr_n").fill_null("") == "").alias("na"), "src")
same = rec.filter(pl.col("src") == 1).group_by("country", "name_n").len().rename({"len": "same_name_s1"})
side = rec.join(same, on=["country", "name_n"], how="left").with_columns(pl.col("same_name_s1").fill_null(0)) \
          .select(pl.col("rid").alias("r2"), "na", "same_name_s1")
St = Ct.select("r2").join(side, on="r2", how="left", maintain_order="left")
Sv = Cv.select("r2").join(side, on="r2", how="left", maintain_order="left")

split = pl.read_parquet(CACHE / "split.parquet")
vaq = tr.filter(pl.col("src") == 1).join(split.filter(pl.col("fold") == 0), on="entity_id").select(pl.col("rid").alias("r1"), "country")
kq = vaq.join(truth.group_by("r1").len().rename({"len": "k"}), on="r1", how="left").fill_null(0)


def f05(keep):
    """keep: bool array over Cv rows = predicted match."""
    ev = Cv.select("r1", "y").with_columns(pl.Series("m", keep))
    per = ev.group_by("r1").agg(pl.col("m").sum().alias("npred"), (pl.col("m") & (pl.col("y") == 1)).sum().alias("found"))
    per = kq.join(per, on="r1", how="left").fill_null(0).with_columns(f05_per_entity(pl.col("k"), pl.col("found"), pl.col("npred")).fill_nan(0).alias("f"))
    return per["f"].mean(), per.filter(pl.col("country") == "India")["f"].mean(), per.filter(pl.col("country") == "US")["f"].mean()


base = f05(pv >= TH8)
print(f"m8 base: F0.5 {base[0]:.5f} (India {base[1]:.5f}, US {base[2]:.5f})", flush=True)

# ---------------- C: empty-address specialist ----------------
nat, nav = St["na"].to_numpy(), Sv["na"].to_numpy()
XtC = np.hstack([Xt[nat], St["same_name_s1"].to_numpy()[nat, None]])
XvC = np.hstack([Xv[nav], Sv["same_name_s1"].to_numpy()[nav, None]])
yt, yv = Ct["y"].to_numpy(), Cv["y"].to_numpy()
sc = lgb.train(dict(P8, num_leaves=63, min_data_in_leaf=100), lgb.Dataset(XtC, yt[nat]), 3000,
               valid_sets=[lgb.Dataset(XvC, yv[nav])], callbacks=[lgb.early_stopping(100, verbose=False)])
pc = sc.predict(XvC, num_iteration=sc.best_iteration)
print(f"C: specialist trained on {nat.sum():,} name-only train pairs (val {nav.sum():,}), rounds {sc.best_iteration} | {time.time()-t0:.0f}s", flush=True)
bestC = (base[0], TH8, None)
for th in np.arange(0.5, 0.96, 0.025):
    keep = pv >= TH8
    keep[nav] = pc >= th
    r = f05(keep)
    if r[0] > bestC[0]:
        bestC = (r[0], round(float(th), 3), r)
    # also: m8 itself on name-only pairs with its own threshold (is it just a threshold effect?)
thr_only = max(((f05(np.where(nav, pv >= th, pv >= TH8))[0], round(float(th), 3)) for th in np.arange(0.5, 0.96, 0.025)))
print(f"C: best F0.5 {bestC[0]:.5f} @ specialist th {bestC[1]} (gain {bestC[0]-base[0]:+.5f}) | "
      f"control = m8 with its own name-only threshold: {thr_only[0]:.5f} @ {thr_only[1]} (gain {thr_only[0]-base[0]:+.5f})", flush=True)

# ---------------- B: singleton / no-match model ----------------
fold = (Ct["r1"].cast(pl.UInt64).hash(seed=7) % 5).to_numpy()
pt = np.zeros(len(yt))
for f in range(5):
    m = lgb.train(P8, lgb.Dataset(Xt[fold != f], yt[fold != f]), m8.num_trees())
    pt[fold == f] = m.predict(Xt[fold == f], num_threads=24)
print(f"B: OOF m8 probabilities done | {time.time()-t0:.0f}s", flush=True)


def s1_feats(C, p, S):
    d = C.select("r1").with_columns(pl.Series("p", p), S["na"].alias("na"), S["same_name_s1"].alias("sn"))
    g = d.group_by("r1").agg(pl.col("p").max().alias("pmax"), pl.col("p").sort(descending=True).slice(1, 1).first().fill_null(0).alias("p2"),
                             pl.len().alias("ncand"), (pl.col("p") >= 0.5).sum().alias("n50"), pl.col("p").sum().alias("psum"),
                             pl.col("na").filter(pl.col("p") == pl.col("p").max()).first().cast(pl.Int8).alias("best_na"),
                             pl.col("sn").filter(pl.col("p") == pl.col("p").max()).first().alias("best_sn"))
    return g


BF = ["pmax", "p2", "ncand", "n50", "psum", "best_na", "best_sn"]
tr_s1 = Ct.select("r1").unique()
kt = tr_s1.join(truth.group_by("r1").len().rename({"len": "k"}), on="r1", how="left").fill_null(0)
Gt = s1_feats(Ct, pt, St).join(kt, on="r1")
Gv = s1_feats(Cv, pv, Sv).join(kq.select("r1", "k"), on="r1")
sb = lgb.train(dict(P8, num_leaves=31, min_data_in_leaf=200), lgb.Dataset(Gt.select(BF).to_numpy(), (Gt["k"] == 0).to_numpy().astype(int)), 3000,
               valid_sets=[lgb.Dataset(Gv.select(BF).to_numpy(), (Gv["k"] == 0).to_numpy().astype(int))], callbacks=[lgb.early_stopping(100, verbose=False)])
Gv = Gv.with_columns(pl.Series("ps", sb.predict(Gv.select(BF).to_numpy(), num_iteration=sb.best_iteration)))
print(f"B: singleton model on {Gt.height:,} train S1 (singletons {(Gt['k']==0).mean():.3f}) | val S1 with candidates {Gv.height:,} | {time.time()-t0:.0f}s", flush=True)
bestB = (base[0], None, 0)
for cut in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
    empty = set(Gv.filter(pl.col("ps") >= cut)["r1"].to_list())
    keep = (pv >= TH8) & ~Cv["r1"].is_in(list(empty)).to_numpy()
    r = f05(keep)
    print(f"  cut {cut:.2f}: S1 emptied {len(empty):,} (true singletons among them {Gv.filter((pl.col('ps') >= cut) & (pl.col('k') == 0)).height:,}) -> F0.5 {r[0]:.5f} ({r[0]-base[0]:+.5f})", flush=True)
    if r[0] > bestB[0]:
        bestB = (r[0], cut, len(empty))
print(f"B: best F0.5 {bestB[0]:.5f} @ cut {bestB[1]} (gain {bestB[0]-base[0]:+.5f})", flush=True)
print(f"done {time.time()-t0:.0f}s")
