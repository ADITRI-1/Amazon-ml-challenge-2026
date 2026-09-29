"""France variants from the self-trained model (Laya v2p) on top of output_m8xhc (India/US unchanged):
  full : France = output_m6xpo (m6 + Laya v2p, ownership-fixed)
  safe : France = output_m6xo  minus the pairs v2p drops  plus ONLY the v2p additions whose names differ by
         accents / case / French legal forms alone (leaderboard lesson: those are real matches; filler-word
         additions such as fils, et, france, associés, groupe were decoys in m7o). Added pairs whose record is
         still claimed by another S1 are skipped (ownership).
"""
import re, shutil, unicodedata
import polars as pl
from pathlib import Path

LEGAL = {"sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "s", "a", "r", "l"}   # 's a r l' = S.A.R.L. split
fold = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
words = lambda s: set(re.findall(r"[a-z0-9]+", fold(s)))
R = pl.concat([pl.read_csv(f"dataset/test/test_source{i}.tsv", separator="\t", infer_schema_length=0).select("entity_id", "business_name", "country") for i in (1, 2, 3)])
cty = R.select(pl.col("entity_id").alias("s1"), "country")
name = dict(R.select("entity_id", "business_name").iter_rows())


def load(d):
    m = pl.read_csv(f"submission/{d}/matching_results.tsv", separator="\t", infer_schema_length=0).fill_null("")
    c0, c1 = m.columns[:2]
    return m.with_columns(pl.col(c1).str.split(",")).explode(c1).filter(pl.col(c1) != "").select(pl.col(c0).alias("s1"), pl.col(c1).alias("id"))


def write(pairs_fr, out, base="output_m8xhc"):
    """France rows from pairs_fr, India/US rows + candidate_pairs.tsv from submission/<base>."""
    src = base
    base = pl.read_csv(f"submission/{src}/matching_results.tsv", separator="\t", infer_schema_length=0).fill_null("")
    k, v = base.columns[:2]
    fr_ids = cty.filter(pl.col("country") == "France")["s1"]
    lists = pairs_fr.unique().sort("s1", "id").group_by("s1", maintain_order=True).agg(pl.col("id").str.join(",").alias("new"))   # deterministic order
    h = base.join(lists.rename({"s1": k}), on=k, how="left", maintain_order="left").with_columns(
        pl.when(pl.col(k).is_in(fr_ids.implode())).then(pl.col("new").fill_null("")).otherwise(pl.col(v)).alias(v)).drop("new")
    assert h.height == base.height
    o = Path(f"submission/{out}"); o.mkdir(exist_ok=True)
    h.write_csv(o / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copy(f"submission/{src}/candidate_pairs.tsv", o / "candidate_pairs.tsv")
    print(f"wrote {o}: France pairs {pairs_fr.height:,}")


if __name__ == "__main__":     # historical hcp / hcs variants (not part of the final pipeline)
    a = load("output_m6xo").join(cty, on="s1").filter(pl.col("country") == "France").select("s1", "id")
    b = load("output_m6xpo").join(cty, on="s1").filter(pl.col("country") == "France").select("s1", "id")
    write(b, "output_m8xhcp")
    dropped = a.join(b, on=["s1", "id"], how="anti")
    added = b.join(a, on=["s1", "id"], how="anti")
    safe = added.filter(pl.struct("s1", "id").map_elements(lambda r: (words(name[r["s1"]]) ^ words(name[r["id"]])) <= LEGAL, return_dtype=pl.Boolean))
    kept = a.join(dropped, on=["s1", "id"], how="anti")
    safe = safe.join(kept.select("id"), on="id", how="anti")          # ownership: record not claimed elsewhere
    print(f"France: m6xo {a.height:,} | v2p drops {dropped.height:,} | v2p adds {added.height:,}, safe adds {safe.height:,}")
    write(pl.concat([kept, safe]), "output_m8xhcs")
