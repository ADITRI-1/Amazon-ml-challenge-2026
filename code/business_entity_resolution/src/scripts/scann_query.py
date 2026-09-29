"""Query persisted ScaNN indexes with NEW Source-1 records (no index rebuild): embeds the records with the fine-tuned
bi-encoder and returns their top-k Source-2/3 candidates per country.
Indexes come from: python scripts/gen_candidates_scann.py --split test --index_dir models/scann_index ...
usage: python scripts/scann_query.py --records new_s1.tsv --index_dir models/scann_index --split test --k 50 --out new_candidates.tsv
       (new_s1.tsv: entity_id, business_name, business_address, country — the challenge TSV columns)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scann  # noqa: E402

from er.biencoder import encode, load_model, texts_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--records", required=True)
ap.add_argument("--index_dir", default="models/scann_index")
ap.add_argument("--split", default="test")
ap.add_argument("--model", default="models/biencoder_e5s_v1")
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()
R = pl.read_csv(a.records, separator="\t", infer_schema_length=0).fill_null("")
tok, m = load_model(a.model)
rows = []
for c in sorted(R["country"].unique().to_list()):
    d = (Path(a.index_dir) / f"{a.split}_{c}").resolve()
    if not (d / "cfg.json").exists():
        print(f"{c}: no index in {d} (unseen country) - skipped", flush=True)
        continue
    q = R.filter(pl.col("country") == c)
    s = scann.scann_ops_pybind.load_searcher(str(d))
    ids = np.load(d / "pool_entity_id.npy", allow_pickle=True)
    t = time.time()
    I, _ = s.search_batched(encode(texts_of(q), tok, m).astype(np.float32), final_num_neighbors=a.k)
    rows += [(e, ",".join(ids[np.asarray(i)])) for e, i in zip(q["entity_id"].to_list(), I)]
    print(f"{c}: {q.height:,} new records | loaded index {d} | search {(time.time() - t) / max(q.height, 1) * 1e3:.2f} ms/record", flush=True)
pl.DataFrame(rows, schema=["source1_entity_id", "candidate_entity_ids"], orient="row").write_csv(a.out, separator="\t")
print(f"wrote {a.out} ({len(rows):,} rows) | {time.time() - t0:.0f}s", flush=True)
