"""France probabilities of the saved m6 matcher (features + e5 CE + Laya v1 + Laya v2) on a France-only candidate set.
In scalable mode only French pairs need e5 / Laya v1 scores (India/US use the lean m8lean stack).
Input files for suffix S: cache/filt_test_S_fr, cache/cands_test_S, cache/ce_{test_v2,laya_test,laya2_test}_S_fr.
Output: cache/testpred_m6_S_fr.parquet (r1, r2, rank, p) -> france_map_variant.py --pred testpred_m6_S_fr
usage: python scripts/france_m6.py --sfx scalable
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb  # noqa: E402
import polars as pl  # noqa: E402

from ensemble_matcher import matrices  # noqa: E402
from er.io import CACHE, ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sfx", required=True)
a = ap.parse_args()
S = a.sfx
none = pl.DataFrame(schema={"r1": pl.UInt32, "r2": pl.UInt32})
Cs, Xs = matrices("test", f"filt_test_{S}_fr", f"cands_test_{S}", f"test_v2_{S}_fr", [f"laya_test_{S}_fr", f"laya2_test_{S}_fr"], none)
m6 = lgb.Booster(model_file=str(ROOT / "models/matcher_m6/lgb_seed0.txt"))
T = Cs.select("r1", "r2", "rank").with_columns(pl.Series("p", m6.predict(Xs, num_threads=24)))
T.write_parquet(CACHE / f"testpred_m6_{S}_fr.parquet")
print(f"m6 France predictions on {S}: {T.height:,} pairs, {(T['p'] >= 0.725).sum():,} above 0.725", flush=True)
