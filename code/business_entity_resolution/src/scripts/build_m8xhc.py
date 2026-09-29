"""Final candidate: exact blocking; India/US = m8 + empty-address specialist C; France = m6 on exact (output_m6xo).
C: LightGBM trained only on name-only (S2/S3 without address) train pairs, m8's 51 features + same_name_s1
(number of S1 in the country with the identical normalised name). It replaces m8's probability on name-only pairs.
Writes submission/output_m8xhc (+ validator run separately).
"""
import shutil
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
from predict import write_lists  # noqa: E402

EXTRA = ["laya", "laya2", "bge"]
TH = 0.75                       # m8 threshold; C's best threshold was also 0.75
P = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
     "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 24, "seed": 0}
t0 = time.time()


def side_info(split):
    df = load_norm(split).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    rec = df.select("rid", "country", "name_n", (pl.col("addr_n").fill_null("") == "").alias("na"), "src")
    same = rec.filter(pl.col("src") == 1).group_by("country", "name_n").len().rename({"len": "same_name_s1"})
    return rec.join(same, on=["country", "name_n"], how="left").with_columns(pl.col("same_name_s1").fill_null(0)) \
              .select(pl.col("rid").alias("r2"), "na", "same_name_s1"), df


def with_c(C, X, side):
    S = C.select("r2").join(side, on="r2", how="left", maintain_order="left")
    na = S["na"].to_numpy()
    return na, np.hstack([X[na], S["same_name_s1"].to_numpy()[na, None]])


tr_side, tr = side_info("train")
idmap = tr.select("entity_id", "rid")
truth = load_gt_pairs().join(idmap, left_on="id1", right_on="entity_id").rename({"rid": "r1"}) \
                       .join(idmap, left_on="id2", right_on="entity_id").rename({"rid": "r2"}).select("r1", "r2")
# 1) train C exactly as in scripts/specialists.py (ScaNN-blocked train/val pairs)
Ct, Xt = matrices("train", "filt_tr", "cands_train_scann", "tr_v2", [f"{f}_tr" for f in EXTRA], truth)
Cv, Xv = matrices("train", "filt_va", "cands_train_scann", "va_v2", [f"{f}_va" for f in EXTRA], truth)
nat, XtC = with_c(Ct, Xt, tr_side); nav, XvC = with_c(Cv, Xv, tr_side)
sc = lgb.train(P, lgb.Dataset(XtC, Ct["y"].to_numpy()[nat]), 3000, valid_sets=[lgb.Dataset(XvC, Cv["y"].to_numpy()[nav])],
               callbacks=[lgb.early_stopping(100, verbose=False)])
out = ROOT / "models" / "matcher_m8c"; out.mkdir(parents=True, exist_ok=True)
sc.save_model(str(out / "spec_c.txt"), num_iteration=sc.best_iteration)
print(f"C trained: {nat.sum():,} name-only pairs, rounds {sc.best_iteration} | {time.time()-t0:.0f}s", flush=True)
del Xt, Xv
m8 = lgb.Booster(model_file=str(ROOT / "models/matcher_m8/lgb_seed0.txt"))

# 2) val on EXACT blocking: m8 alone vs m8 + C
split = pl.read_parquet(CACHE / "split.parquet")
vaq = tr.filter(pl.col("src") == 1).join(split.filter(pl.col("fold") == 0), on="entity_id").select(pl.col("rid").alias("r1"), "country")
kq = vaq.join(truth.group_by("r1").len().rename({"len": "k"}), on="r1", how="left").fill_null(0)
Cx, Xx = matrices("train", "filt_va_exact", "cands_train_exact", "va_v2_exact", [f"{f}_va_exact" for f in EXTRA], truth)
p8 = m8.predict(Xx, num_threads=24)
nax, XxC = with_c(Cx, Xx, tr_side)
pc = p8.copy(); pc[nax] = sc.predict(XxC, num_iteration=sc.best_iteration)


def f05(p):
    ev = Cx.select("r1", "y").with_columns(pl.Series("m", p >= TH))
    per = ev.group_by("r1").agg(pl.col("m").sum().alias("npred"), (pl.col("m") & (pl.col("y") == 1)).sum().alias("found"))
    per = kq.join(per, on="r1", how="left").fill_null(0).with_columns(f05_per_entity(pl.col("k"), pl.col("found"), pl.col("npred")).fill_nan(0).alias("f"))
    return f"{per['f'].mean():.5f} (India {per.filter(pl.col('country') == 'India')['f'].mean():.5f}, US {per.filter(pl.col('country') == 'US')['f'].mean():.5f})"


print(f"exact-blocking val: m8 {f05(p8)} | m8 + C {f05(pc)}", flush=True)
del Xx

# 3) test on exact blocking
te_side, te = side_info("test")
Cs, Xs = matrices("test", "filt_test_exact", "cands_test_exact", "test_v2_exact", [f"{f}_test_exact" for f in EXTRA],
                  pl.DataFrame(schema={"r1": pl.UInt32, "r2": pl.UInt32}))
ps = m8.predict(Xs, num_threads=24)
nas, XsC = with_c(Cs, Xs, te_side)
ps[nas] = sc.predict(XsC, num_iteration=sc.best_iteration, num_threads=24)
T = Cs.select("r1", "r2", "rank").with_columns(pl.Series("p", ps))
T.write_parquet(CACHE / "testpred_m8xc.parquet")
# ownership fix: a record keeps only its highest-probability claim
Pk = T.filter(pl.col("p") >= TH).sort("p", descending=True).unique(subset="r2", keep="first", maintain_order=True)
print(f"test: name-only pairs re-scored by C {nas.sum():,} | predicted {T.filter(pl.col('p') >= TH).height:,} → {Pk.height:,} after ownership", flush=True)
ids = te.select("rid", "entity_id")
Pk = Pk.join(ids.rename({"rid": "r1", "entity_id": "source1_entity_id"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "eid"}), on="r2")
s1_ids = te.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("source1_entity_id"))
tmp = ROOT / "submission" / "_m8xco"; tmp.mkdir(parents=True, exist_ok=True)
write_lists(Pk.sort("r1", "rank"), s1_ids, "matched_entity_ids", tmp / "matching_results.tsv")

# 4) hybrid: France rows from output_m6xo, India/US rows from m8 + C
s1c = pl.read_csv(ROOT / "dataset/test/test_source1.tsv", separator="\t", infer_schema_length=0).select("entity_id", "country")
a = pl.read_csv(ROOT / "submission/output_m6xo/matching_results.tsv", separator="\t", infer_schema_length=0).fill_null("")
b = pl.read_csv(tmp / "matching_results.tsv", separator="\t", infer_schema_length=0).fill_null("")
k = a.columns[0]; assert a.columns == b.columns
fr = set(s1c.filter(pl.col("country") == "France")["entity_id"].to_list())
h = a.select(k).join(pl.concat([a.filter(pl.col(k).is_in(fr)), b.filter(~pl.col(k).is_in(fr))]), on=k, how="left", maintain_order="left")
assert h.height == a.height and h[h.columns[1]].null_count() == 0
o = ROOT / "submission" / "output_m8xhc"; o.mkdir(parents=True, exist_ok=True)
h.write_csv(o / "matching_results.tsv", separator="\t", quote_style="never")
shutil.copy(ROOT / "submission/output_m6xo/candidate_pairs.tsv", o / "candidate_pairs.tsv")
shutil.rmtree(tmp)
print(f"wrote {o}: {(h[h.columns[1]] != '').sum():,} S1 with a match | {time.time()-t0:.0f}s", flush=True)
