"""Stage-1 retrieval at scale: ScaNN (tree partitioning + 4-bit anisotropic quantization + exact re-rank).

For every S1 record: top-K S2/S3 records of the same country by bi-encoder cosine, searching 15% of the partitions
(validated operating point: every test country keeps ≥99.3% of predicted matches, see logs/ann_scann_operating_point.csv).
Countries come from the data (open set: France etc. handled the same way).

usage: python scripts/gen_candidates_scann.py --split train|test --model models/biencoder_e5s_v1 [--search_frac 0.15 --reorder 250 --tag scann]
out:   cache/cands_<split>_<tag>.parquet  (r1, r2, rank, score)   r* = row index into cache/<split>_norm.parquet
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scann  # noqa: E402

from er.biencoder import encode, load_model, texts_of  # noqa: E402
from er.io import CACHE, load_norm  # noqa: E402

LEAVES, SEARCH_FRAC, REORDER = 2000, 0.15, 250


def cached_embeddings(path, df, model):
    if path.exists():
        return np.load(path).astype(np.float32)
    tok, m = model()
    E = encode(texts_of(df), tok, m)
    np.save(path, E)
    return E.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--search_frac", type=float, default=SEARCH_FRAC, help="share of the 2,000 partitions searched per query")
    ap.add_argument("--reorder", type=int, default=REORDER, help="4-bit candidates re-ranked exactly")
    ap.add_argument("--tag", default="scann")
    ap.add_argument("--index_dir", default="", help="persist per-country ScaNN indexes here (reused when settings match)")
    ap.add_argument("--build_only", action="store_true", help="build + save the indexes, no search")
    a = ap.parse_args()
    t0 = time.time()
    df = load_norm(a.split).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    _m = {}

    def model():
        if not _m:
            _m["tm"] = load_model(a.model)
        return _m["tm"]

    pre = "emb_v1_" if a.split == "train" else "emb_v1_test_"
    out = []
    for c in sorted(df["country"].unique().to_list()):
        pool = df.filter((pl.col("src") > 1) & (pl.col("country") == c))
        q = df.filter((pl.col("src") == 1) & (pl.col("country") == c))
        t = time.time()
        Ep = cached_embeddings(CACHE / f"{pre}{c}_pool.npy", pool, model)
        Eq = cached_embeddings(CACHE / (f"{pre}{c}_s1all.npy" if a.split == "train" else f"{pre}{c}_s1.npy"), q, model)
        te = time.time() - t
        t = time.time()
        leaves = min(LEAVES, max(1, len(Ep) // 500))  # 2000 for every real country pool (validated setting)
        cfg = {"split": a.split, "country": c, "pool": len(Ep), "leaves": leaves, "search_frac": a.search_frac,
               "reorder": a.reorder, "k": a.k, "model": a.model}
        idx = (Path(a.index_dir) / f"{a.split}_{c}").resolve() if a.index_dir else None   # absolute: ScaNN stores asset paths
        if idx is not None and (idx / "cfg.json").exists() and json.load(open(idx / "cfg.json")) == cfg:
            searcher = scann.scann_ops_pybind.load_searcher(str(idx))          # persisted index: no rebuild
            how = "loaded"
        else:
            searcher = scann.scann_ops_pybind.builder(Ep, a.k, "dot_product").tree(
                num_leaves=leaves, num_leaves_to_search=max(1, int(leaves * a.search_frac)),
                training_sample_size=min(len(Ep), 250_000)).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(a.reorder).build()
            how = "built"
            if idx is not None:
                idx.mkdir(parents=True, exist_ok=True)
                searcher.serialize(str(idx))
                np.save(idx / "pool_rid.npy", pool["rid"].to_numpy())
                np.save(idx / "pool_entity_id.npy", np.array(pool["entity_id"].to_list()))
                json.dump(cfg, open(idx / "cfg.json", "w"), indent=1)
                how = f"built + saved to {idx}"
        tb = time.time() - t
        if a.build_only:
            print(f"{a.split} {c}: pool {len(Ep):,} | index {how} in {tb:.0f}s", flush=True)
            del Ep, Eq, searcher
            continue
        t = time.time()
        I, D = searcher.search_batched_parallel(Eq, leaves_to_search=max(1, int(leaves * a.search_frac)), pre_reorder_num_neighbors=a.reorder)
        I, D = np.asarray(I), np.asarray(D)
        ts = time.time() - t
        out.append(pl.DataFrame({"r1": np.repeat(q["rid"].to_numpy(), a.k), "r2": pool["rid"].to_numpy()[I].ravel(),
                                 "rank": np.tile(np.arange(a.k, dtype=np.int16), len(q)), "score": D.ravel().astype(np.float16)}))
        print(f"{a.split} {c}: S1 {len(q):,} x pool {len(Ep):,} | embed {te:.0f}s, index {how} {tb:.0f}s ({leaves} leaves), "
              f"search {ts:.0f}s = {ts / len(q) * 1e3:.3f} ms/query", flush=True)
        del Ep, Eq, searcher
    if a.build_only:
        sys.exit(0)
    C = pl.concat(out)
    C.write_parquet(CACHE / f"cands_{a.split}_{a.tag}.parquet")
    print(f"saved {C.height:,} pairs in {time.time()-t0:.0f}s", flush=True)
