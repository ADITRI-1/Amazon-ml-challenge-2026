"""Re-score a cross-encoder's TEST logits with Latin accent folding (é→e, Ç→C) on the input text, only for pairs whose
text contains an accented Latin letter (all other pairs are unchanged). Fixes BGE reading injected French accent noise
("Çlub" vs "Club") as a different business.  usage: python scripts/rescore_folded.py --src ce_bge_test --out ce_bgef_test --model models/ce_bge_m3_fp16 --hf
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl  # noqa: E402
from er.crossencoder import CrossEncoder, HFCrossEncoder, ce_score, fold_latin, pair_text  # noqa: E402
from er.io import CACHE, ROOT, load_norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", required=True); ap.add_argument("--hf", action="store_true"); ap.add_argument("--split", default="test")
a = ap.parse_args(); t0 = time.time()
C = pl.read_parquet(CACHE / f"{a.src}.parquet")
df = load_norm(a.split).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).select("rid", "business_name", "business_address")
T = {r: pair_text(n, s) for r, n, s in df.iter_rows()}
ta = [T[r] for r in C["r1"].to_list()]; tb = [T[r] for r in C["r2"].to_list()]
idx = [i for i, (x, y) in enumerate(zip(ta, tb)) if fold_latin(x) != x or fold_latin(y) != y]
print(f"pairs {C.height:,} | with accented Latin text {len(idx):,} | {time.time()-t0:.0f}s", flush=True)
m = (HFCrossEncoder if a.hf else CrossEncoder)(str(ROOT / a.model)); m.fold = True
z = ce_score(m, [ta[i] for i in idx], [tb[i] for i in idx], bs=512)
ce = C["ce"].to_numpy().copy(); old = ce[idx].copy(); ce[idx] = z
print(f"re-scored {len(idx):,}: mean logit {old.mean():.2f} -> {z.mean():.2f}; flipped to >0: {((old <= 0) & (z > 0)).sum():,}, to <=0: {((old > 0) & (z <= 0)).sum():,}", flush=True)
C.with_columns(pl.Series("ce", ce, dtype=pl.Float32)).write_parquet(CACHE / f"{a.out}.parquet")
print(f"saved cache/{a.out}.parquet | {time.time()-t0:.0f}s")
