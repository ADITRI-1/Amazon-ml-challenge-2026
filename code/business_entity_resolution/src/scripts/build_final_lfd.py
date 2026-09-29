"""Final France (public LB 0.9873): MAP France (category swaps removed, filler variants added) minus French name-only
copies that Laya-FR rejects (logit < 0) plus Laya-FR's most confident new matches (logit > 3), ownership-safe.
Needs: submission/output_m8xhcMAP (france_map_variant.py), laya_fr_kit/out/scores.parquet (laya_fr/train_score.py),
cache/layafr_score_keys.parquet (keys of the scored French pairs), cache/france_map_cls.parquet. -> submission/output_m8xhcLFd
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import polars as pl  # noqa: E402
import build_france_variants as bfv  # noqa: E402
from er.io import ROOT, load_norm  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--map", default="output_m8xhcMAP"); ap.add_argument("--cls", default="france_map_cls")
ap.add_argument("--base", default="output_m8xhc"); ap.add_argument("--out", default="output_m8xhcLFd")
A = ap.parse_args()
te = load_norm("test").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)); ids = te.select("rid", "entity_id")
K = pl.read_parquet(ROOT / "cache/layafr_score_keys.parquet").join(pl.read_parquet(ROOT / "laya_fr_kit/out/scores.parquet"), on="key")
C = pl.read_parquet(ROOT / f"cache/{A.cls}.parquet").select("r1", "r2", "cls")
K = K.join(C, on=["r1", "r2"], how="left").join(ids.rename({"rid": "r1", "entity_id": "s1"}), on="r1").join(ids.rename({"rid": "r2", "entity_id": "id"}), on="r2")
m = bfv.load(A.map).join(bfv.cty, on="s1").filter(pl.col("country") == "France").select("s1", "id")
drop = m.join(K.filter((pl.col("cls") == "name-only") & (pl.col("ce") < 0)).select("s1", "id"), on=["s1", "id"])
keep = m.join(drop, on=["s1", "id"], how="anti")
adds = K.filter(pl.col("ce") > 3).join(keep, on=["s1", "id"], how="anti").join(keep.select("id"), on="id", how="anti").sort("ce", descending=True).unique(subset="id", keep="first")
bfv.write(pl.concat([keep, adds.select("s1", "id")]), A.out, A.base)
