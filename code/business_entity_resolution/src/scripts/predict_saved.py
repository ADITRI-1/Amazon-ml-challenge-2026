"""Apply a SAVED matcher (m6 / m8 LightGBM) to another candidate set (e.g. exact-search blocking) without retraining.
usage: python scripts/predict_saved.py --matcher m8 --extra laya,laya2,bge --sfx exact [--test_extra_sfx ...] [--write TAG]
  val : cands filt_va_<sfx>,  ctx cands_train_<sfx>, CE ce_va_v2_<sfx> + ce_<fam>_va_<sfx>
  test: cands filt_test_<sfx>, ctx cands_test_<sfx>,  CE ce_test_v2_<sfx> + ce_<fam>_test_<sfx>
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb, numpy as np, polars as pl  # noqa: E402
from ensemble_matcher import matrices  # noqa: E402
from er.io import CACHE, ROOT, load_gt_pairs, load_norm  # noqa: E402
from train_matcher import sweep  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--matcher", required=True); ap.add_argument("--extra", default="laya,laya2")
ap.add_argument("--sfx", default="exact"); ap.add_argument("--write", default="")
a = ap.parse_args(); t0 = time.time(); S = a.sfx
fam = [f for f in a.extra.split(",") if f]
md = ROOT / "models" / f"matcher_{a.matcher}"; meta = json.load(open(md / "ensemble.json"))
m = lgb.Booster(model_file=str(md / "lgb_seed0.txt"))
drop = tuple(meta.get("drop_base", []))
e5 = lambda t: None if "ce_logit" in drop else t
tr = load_norm("train").with_row_index("rid")
idmap = tr.select("entity_id", pl.col("rid").cast(pl.UInt32))
truth = load_gt_pairs().join(idmap, left_on="id1", right_on="entity_id").rename({"rid": "r1"}) \
                       .join(idmap, left_on="id2", right_on="entity_id").rename({"rid": "r2"}).select("r1", "r2")
Cv, Xv = matrices("train", f"filt_va_{S}", f"cands_train_{S}", e5(f"va_v2_{S}"), [f"{f}_va_{S}" for f in fam], truth, drop=drop)
pv = m.predict(Xv, num_threads=24)
split = pl.read_parquet(CACHE / "split.parquet")
vaq = tr.filter(pl.col("src") == 1).join(split.filter(pl.col("fold") == 0), on="entity_id").select(pl.col("rid").cast(pl.UInt32).alias("r1"), "country")
kq = vaq.join(truth.group_by("r1").len().rename({"len": "k"}), on="r1", how="left").fill_null(0)
ev = Cv.select("r1", "r2", "y").with_columns(pl.Series("p", pv))
res = sweep(ev, kq, np.array([meta["best_th"]]))
r = res.row(0, named=True)
print(f"{a.matcher} on {S} blocking: val F0.5 {r['F05']:.5f} @ saved th {meta['best_th']} (India {r['F05_India']:.5f}, US {r['F05_US']:.5f}) "
      f"| on ScaNN: {meta['val_F05']:.5f} | val pairs {Cv.height:,} ({Cv.height / vaq.height:.2f}/S1) | {time.time()-t0:.0f}s", flush=True)
if a.write:
    te = load_norm("test").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    Cs, Xs = matrices("test", f"filt_test_{S}", f"cands_test_{S}", e5(f"test_v2_{S}"), [f"{f}_test_{S}" for f in fam], pl.DataFrame(schema={"r1": pl.UInt32, "r2": pl.UInt32}), drop=drop)
    Cs = Cs.select("r1", "r2", "rank").with_columns(pl.Series("p", m.predict(Xs, num_threads=24)))
    Cs.write_parquet(CACHE / f"testpred_{a.write}.parquet")
    ids = te.select("rid", "entity_id")
    Cs = Cs.join(ids.rename({"rid": "r1", "entity_id": "source1_entity_id"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "eid"}), on="r2")
    s1_ids = te.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("source1_entity_id"))
    from predict import write_lists  # noqa: E402
    o = ROOT / "submission" / f"output_{a.write}"; o.mkdir(parents=True, exist_ok=True)
    write_lists(Cs.sort("r1", "rank"), s1_ids, "candidate_entity_ids", o / "candidate_pairs.tsv")
    mr = write_lists(Cs.filter(pl.col("p") >= meta["best_th"]).sort("r1", "rank"), s1_ids, "matched_entity_ids", o / "matching_results.tsv")
    print(f"wrote {o}: test pairs {Cs.height:,} ({Cs.height / s1_ids.height:.2f}/S1), {(mr['matched_entity_ids'] != '').sum():,} S1 with a match | {time.time()-t0:.0f}s", flush=True)
