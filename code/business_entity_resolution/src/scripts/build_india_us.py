"""India/US assembly for any candidate set (scalable mode): saved matcher (e.g. m8lean = features + Laya v2 + BGE, no e5
cross-encoder) + no-address specialist C retrained on the same features + ownership fix. France rows are written too
(same models) but are replaced afterwards by the France path (france_map_variant.py -> build_final_lfd.py --base <this>).

Files used for candidate suffix S: cache/filt_{va,test}_S, cache/cands_{train,test}_S, cache/ce_<family>_{va,test}_S.
usage: python scripts/build_india_us.py --matcher m8lean --sfx scalable --out output_iu_scalable
"""
import argparse
import json
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
from predict import write_lists  # noqa: E402
from train_matcher import sweep  # noqa: E402

P = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
     "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 24, "seed": 0}


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", default="m8lean")
    ap.add_argument("--sfx", required=True, help="candidate-set suffix, e.g. scalable")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    md = ROOT / "models" / f"matcher_{a.matcher}"
    meta = json.load(open(md / "ensemble.json"))
    fam = [f for f in meta["extra"].split(",") if f]
    drop = tuple(meta.get("drop_base", []))
    e5 = lambda t: None if "ce_logit" in drop else t
    TH = meta["best_th"]
    m = lgb.Booster(model_file=str(md / "lgb_seed0.txt"))
    tr_side, tr = side_info("train")
    idm = tr.select("entity_id", "rid")
    truth = load_gt_pairs().join(idm, left_on="id1", right_on="entity_id").rename({"rid": "r1"}) \
                           .join(idm, left_on="id2", right_on="entity_id").rename({"rid": "r2"}).select("r1", "r2")
    # specialist C on the matcher's own features (training pairs = folds 5-9, as for the matcher)
    Ct, Xt = matrices("train", "filt_tr", "cands_train_scann", e5("tr_v2"), [f"{f}_tr" for f in fam], truth, drop=drop)
    Cv0, Xv0 = matrices("train", "filt_va", "cands_train_scann", e5("va_v2"), [f"{f}_va" for f in fam], truth, drop=drop)
    nat, XtC = with_c(Ct, Xt, tr_side); nav, XvC = with_c(Cv0, Xv0, tr_side)
    sc = lgb.train(P, lgb.Dataset(XtC, Ct["y"].to_numpy()[nat]), 3000, valid_sets=[lgb.Dataset(XvC, Cv0["y"].to_numpy()[nav])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    sc.save_model(str(md / "spec_c.txt"), num_iteration=sc.best_iteration)
    print(f"C ({a.matcher} features): {nat.sum():,} name-only train pairs, rounds {sc.best_iteration} | {time.time()-t0:.0f}s", flush=True)
    del Xt, Xv0
    # validation on the chosen candidate set
    split = pl.read_parquet(CACHE / "split.parquet")
    vaq = tr.filter(pl.col("src") == 1).join(split.filter(pl.col("fold") == 0), on="entity_id").select(pl.col("rid").alias("r1"), "country")
    kq = vaq.join(truth.group_by("r1").len().rename({"len": "k"}), on="r1", how="left").fill_null(0)
    Cv, Xv = matrices("train", f"filt_va_{a.sfx}", f"cands_train_{a.sfx}", e5(f"va_v2_{a.sfx}"), [f"{f}_va_{a.sfx}" for f in fam], truth, drop=drop)
    pv = m.predict(Xv, num_threads=24)
    na, XC = with_c(Cv, Xv, tr_side)
    pc = pv.copy(); pc[na] = sc.predict(XC, num_iteration=sc.best_iteration, num_threads=24)
    for lab, p in [(a.matcher, pv), (f"{a.matcher} + C", pc)]:
        r = sweep(Cv.select("r1", "r2", "y").with_columns(pl.Series("p", p)), kq, np.array([TH])).row(0, named=True)
        print(f"val [{a.sfx}] {lab}: F0.5 {r['F05']:.5f} (India {r['F05_India']:.5f}, US {r['F05_US']:.5f}) | pairs {Cv.height:,}", flush=True)
    del Xv
    # test
    te_side, te = side_info("test")
    none = pl.DataFrame(schema={"r1": pl.UInt32, "r2": pl.UInt32})
    Cs, Xs = matrices("test", f"filt_test_{a.sfx}", f"cands_test_{a.sfx}", e5(f"test_v2_{a.sfx}"), [f"{f}_test_{a.sfx}" for f in fam], none, drop=drop)
    ps = m.predict(Xs, num_threads=24)
    nas, XsC = with_c(Cs, Xs, te_side)
    ps[nas] = sc.predict(XsC, num_iteration=sc.best_iteration, num_threads=24)
    T = Cs.select("r1", "r2", "rank").with_columns(pl.Series("p", ps))
    T.write_parquet(CACHE / f"testpred_{a.matcher}_{a.sfx}.parquet")
    Pk = T.filter(pl.col("p") >= TH).sort("p", descending=True).unique(subset="r2", keep="first")   # ownership
    ids = te.select("rid", "entity_id")
    j = lambda D: D.join(ids.rename({"rid": "r1", "entity_id": "source1_entity_id"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "eid"}), on="r2")
    s1_ids = te.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("source1_entity_id"))
    o = ROOT / "submission" / a.out; o.mkdir(parents=True, exist_ok=True)
    write_lists(j(T).sort("r1", "rank"), s1_ids, "candidate_entity_ids", o / "candidate_pairs.tsv")
    mr = write_lists(j(Pk).sort("r1", "rank"), s1_ids, "matched_entity_ids", o / "matching_results.tsv")
    print(f"wrote {o}: test pairs {T.height:,} ({T.height / s1_ids.height:.2f}/S1), matches {Pk.height:,} | {time.time()-t0:.0f}s", flush=True)
