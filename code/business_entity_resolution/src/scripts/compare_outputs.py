"""Set-level comparison of two submission folders (pair sets per country; list order inside a row is irrelevant to scoring).
usage: python scripts/compare_outputs.py output_A output_B"""
import sys
import polars as pl
s1 = pl.read_csv("dataset/test/test_source1.tsv", separator="\t", infer_schema_length=0).select(pl.col("entity_id").alias("s1"), "country")
def pairs(d, f="matching_results.tsv"):
    m = pl.read_csv(f"submission/{d}/{f}", separator="\t", infer_schema_length=0).fill_null("")
    c0, c1 = m.columns[:2]
    return m.with_columns(pl.col(c1).str.split(",")).explode(c1).filter(pl.col(c1) != "").select(pl.col(c0).alias("s1"), pl.col(c1).alias("id")).join(s1, on="s1")
a, b = sys.argv[1], sys.argv[2]
for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
    A, B = pairs(a, f), pairs(b, f)
    print(f"{f}: {a} {A.height:,} vs {b} {B.height:,}")
    for c in ["France", "India", "US"]:
        x = set(A.filter(pl.col("country") == c).select("s1", "id").iter_rows()); y = set(B.filter(pl.col("country") == c).select("s1", "id").iter_rows())
        print(f"   {c:6s} identical {len(x & y) / max(len(x), 1):.4%} of {a} | only {a} {len(x - y):,} | only {b} {len(y - x):,}")
