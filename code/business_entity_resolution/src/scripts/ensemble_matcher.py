"""Seed-ensembled final matcher on the m4 candidate sets (optionally with extra cross-encoder scores = m5).

Builds the m4 feature matrix ONCE for train (filt_tr), val (filt_va) and test (filt_test), then trains N LightGBM
models that differ only in random seed (bagging / feature subsampling), averages their probabilities, and compares
single-model vs ensemble macro F0.5 on the held-out fold. Writes test outputs only when --write is given.

usage:
  python scripts/ensemble_matcher.py --seeds 5 --tag m4e                       # seed ensemble of m4
  python scripts/ensemble_matcher.py --seeds 5 --extra laya --tag m5 --write   # + Laya CE logit as a feature
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from er.features import FIELDS  # noqa: E402
from er.io import CACHE, ROOT, load_gt_pairs, load_norm  # noqa: E402
from train_matcher import attach, build, sweep  # noqa: E402
from er.features_v2 import pair_features_v2  # noqa: E402

M4 = json.load(open(ROOT / "models" / "matcher_m4" / "features.json"))


def matrices(split, cand, ctx, ce_tag, extra_tags, truth, v2=False, drop=()):
    """drop: base feature names to leave out (lean mode: drop=("ce_logit",) with ce_tag=None -> no e5 CE needed)."""
    df = load_norm(split).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    recs = df.select("rid", *FIELDS)
    C_all = pl.read_parquet(CACHE / f"{ctx}.parquet").filter(pl.col("rank") < 50).select("r1", "r2", "score")
    C = pl.read_parquet(CACHE / f"{cand}.parquet").select("r1", "r2", "rank", "score")
    Cb, X = build(C, recs, truth, C_all)
    assert ce_tag is not None or "ce_logit" in drop, "ce_tag=None needs drop=('ce_logit',)"
    ce = attach(Cb, pl.read_parquet(CACHE / f"ce_{ce_tag}.parquet")) if ce_tag is not None else np.zeros((len(X), 1), np.float32)
    X = np.hstack([X, ce.reshape(len(X), -1)])
    keep = [M4["all_features"].index(n) for n in M4["kept"] if n not in drop]
    X = X[:, keep]
    for t in extra_tags:
        X = np.hstack([X, attach(Cb, pl.read_parquet(CACHE / f"ce_{t}.parquet"))])
    if v2:
        X = np.hstack([X, pair_features_v2(Cb, pl.read_parquet(CACHE / f"{split}_v2.parquet"))])
    return Cb, X


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--extra", default="", help="extra CE score family, e.g. 'laya' -> ce_laya_{tr,va,test}")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--write", action="store_true", help="write submission/output_<tag>/ from the ensemble")
    ap.add_argument("--v2", action="store_true", help="add the v2 address/name features (src/er/features_v2.py)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--no_es", action="store_true", help="train exactly --rounds (no early stopping)")
    ap.add_argument("--drop_base", default="", help="comma list of base features to drop, e.g. ce_logit (lean: no e5 CE)")
    a = ap.parse_args()
    t0 = time.time()
    tr = load_norm("train").with_row_index("rid")
    idmap = tr.select("entity_id", pl.col("rid").cast(pl.UInt32))
    truth = load_gt_pairs().join(idmap, left_on="id1", right_on="entity_id").rename({"rid": "r1"}) \
                           .join(idmap, left_on="id2", right_on="entity_id").rename({"rid": "r2"}).select("r1", "r2")
    fam = [f for f in a.extra.split(",") if f]
    drop = tuple(d for d in a.drop_base.split(",") if d)
    e5 = lambda t: None if "ce_logit" in drop else t
    Ct, Xt = matrices("train", "filt_tr", "cands_train_scann", e5("tr_v2"), [f"{f}_tr" for f in fam], truth, a.v2, drop)
    Cv, Xv = matrices("train", "filt_va", "cands_train_scann", e5("va_v2"), [f"{f}_va" for f in fam], truth, a.v2, drop)
    print(f"features {Xt.shape[1]} | train {len(Xt):,} val {len(Xv):,} | {time.time()-t0:.0f}s", flush=True)
    split = pl.read_parquet(CACHE / "split.parquet")
    vaq = tr.filter(pl.col("src") == 1).join(split.filter(pl.col("fold") == 0), on="entity_id") \
            .select(pl.col("rid").cast(pl.UInt32).alias("r1"), "country")
    kq = vaq.join(truth.group_by("r1").len().rename({"len": "k"}), on="r1", how="left").fill_null(0)
    ths = np.arange(0.5, 0.91, 0.025)
    base = {"objective": "binary", "learning_rate": a.lr, "num_leaves": 127, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 24}
    models, pv_all, rows = [], [], []
    for s in range(a.seeds):
        dt = lgb.Dataset(Xt, Ct["y"].to_numpy()); dv = lgb.Dataset(Xv, Cv["y"].to_numpy(), reference=dt)
        m = lgb.train(dict(base, seed=s, bagging_seed=s, feature_fraction_seed=s), dt, a.rounds, valid_sets=[dv],
                      callbacks=[] if a.no_es else [lgb.early_stopping(100, verbose=False)])
        pv = m.predict(Xv, num_iteration=m.best_iteration)
        models.append(m); pv_all.append(pv)
        r = sweep(Cv.select("r1", "r2", "y").with_columns(pl.Series("p", pv)), kq, ths).sort("F05", descending=True).row(0, named=True)
        rows.append({"model": f"seed {s}", "rounds": m.best_iteration, "F05": round(r["F05"], 5), "th": r["th"]})
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    pe = np.mean(pv_all, axis=0)
    res = sweep(Cv.select("r1", "r2", "y").with_columns(pl.Series("p", pe)), kq, ths)
    best = res.sort("F05", descending=True).row(0, named=True)
    rows.append({"model": f"ensemble of {a.seeds}", "rounds": None, "F05": round(best["F05"], 5), "th": best["th"]})
    R = pl.DataFrame(rows)
    print(R)
    print(f"ensemble: F0.5 {best['F05']:.5f} @th {best['th']} (India {best['F05_India']:.5f}, US {best['F05_US']:.5f}) | "
          f"single-model mean {np.mean([r['F05'] for r in rows[:-1]]):.5f} ± {np.std([r['F05'] for r in rows[:-1]]):.5f}", flush=True)
    out = ROOT / "models" / f"matcher_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(str(out / f"lgb_seed{i}.txt"), num_iteration=m.best_iteration)
    json.dump({"base": "m4", "drop_base": list(drop), "extra": a.extra, "seeds": a.seeds, "best_th": best["th"], "val_F05": best["F05"],
               "val_F05_India": best["F05_India"], "val_F05_US": best["F05_US"]}, open(out / "ensemble.json", "w"), indent=1)
    R.write_csv(ROOT / "logs" / f"ensemble_{a.tag}.csv")
    Cv.select("r1", "r2", "y").with_columns(pl.Series("p", pe)).write_parquet(CACHE / f"valpred_{a.tag}.parquet")
    if a.write:
        te = load_norm("test").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
        Cs, Xs = matrices("test", "filt_test", "cands_test_scann", e5("test_v2"), [f"{f}_test" for f in fam], pl.DataFrame(schema={"r1": pl.UInt32, "r2": pl.UInt32}), a.v2, drop)
        ps = np.mean([m.predict(Xs, num_iteration=m.best_iteration, num_threads=24) for m in models], axis=0)
        Cs = Cs.select("r1", "r2", "rank").with_columns(pl.Series("p", ps))
        Cs.write_parquet(CACHE / f"testpred_{a.tag}.parquet")
        ids = te.select("rid", "entity_id")
        Cs = Cs.join(ids.rename({"rid": "r1", "entity_id": "source1_entity_id"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "eid"}), on="r2")
        s1_ids = te.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("source1_entity_id"))
        from predict import write_lists  # noqa: E402
        o = ROOT / "submission" / f"output_{a.tag}"
        o.mkdir(parents=True, exist_ok=True)
        write_lists(Cs.sort("r1", "rank"), s1_ids, "candidate_entity_ids", o / "candidate_pairs.tsv")
        mr = write_lists(Cs.filter(pl.col("p") >= best["th"]).sort("r1", "rank"), s1_ids, "matched_entity_ids", o / "matching_results.tsv")
        print(f"wrote {o}: {mr.height:,} rows, {(mr['matched_entity_ids'] != '').sum():,} with a match", flush=True)
    print(f"done {time.time()-t0:.0f}s")
