"""Label-free ScaNN operating-point check on test: for each candidate file (top-30 = what the m2 filter sees),
coverage per country of (a) the exact-blocking filtered candidates and (b) the matched pairs of the shipped
0.9873 output (submission/output_m8xhcLFd). Timings are read from each run's log.
usage: python scripts/scann_sweep.py cands_test_scann cands_test_scann030 cands_test_scann050
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from er.io import CACHE, ROOT, load_norm  # noqa: E402

REF = "output_m8xhcLFd"
te = load_norm("test").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
rid = te.select("entity_id", "rid")
cty = te.select(pl.col("rid").alias("r1"), "country")
m = pl.read_csv(ROOT / f"submission/{REF}/matching_results.tsv", separator="\t", infer_schema_length=0).fill_null("")
c0, c1 = m.columns[:2]
M = m.with_columns(pl.col(c1).str.split(",")).explode(c1).filter(pl.col(c1) != "") \
     .join(rid.rename({"entity_id": c0, "rid": "r1"}), on=c0).join(rid.rename({"entity_id": c1, "rid": "r2"}), on=c1).select("r1", "r2").join(cty, on="r1")
X = pl.read_parquet(CACHE / "filt_test_exact.parquet").select("r1", "r2").join(cty, on="r1")
print(f"reference: {REF} matched pairs {M.height:,} | exact filtered candidates {X.height:,}")
for f in sys.argv[1:]:
    C = pl.read_parquet(CACHE / f"{f}.parquet").filter(pl.col("rank") < 30).select("r1", "r2").with_columns(pl.lit(True).alias("hit"))
    row = []
    for c in sorted(M["country"].unique().to_list()):
        mc, xc = M.filter(pl.col("country") == c), X.filter(pl.col("country") == c)
        cm = mc.join(C, on=["r1", "r2"], how="left")["hit"].fill_null(False).mean()
        cx = xc.join(C, on=["r1", "r2"], how="left")["hit"].fill_null(False).mean()
        row.append(f"{c}: matches {cm:.4%} / exact-cands {cx:.4%}")
    tag = f.replace("cands_test_", "")
    log = ROOT / "logs" / f"2026-09-28_sweep_{tag}.log"
    ms = re.findall(r"= ([\d.]+) ms/query", log.read_text()) if log.exists() else []
    print(f"{f:22s} | " + " | ".join(row) + (f" | ms/query {', '.join(ms)}" if ms else ""), flush=True)
