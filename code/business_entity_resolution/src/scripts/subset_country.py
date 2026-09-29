"""Keep only the pairs whose Source-1 record is in one country (e.g. French pairs for the France-only cross-encoders).
usage: python scripts/subset_country.py --filt filt_test_scalable --country France --out filt_test_scalable_fr"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from er.io import CACHE, load_norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--filt", required=True); ap.add_argument("--country", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
a = ap.parse_args()
te = load_norm(a.split).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
keep = te.filter(pl.col("country") == a.country).select(pl.col("rid").alias("r1"))
F = pl.read_parquet(CACHE / f"{a.filt}.parquet").join(keep, on="r1")
F.write_parquet(CACHE / f"{a.out}.parquet")
print(f"{a.out}: {F.height:,} {a.country} pairs")
