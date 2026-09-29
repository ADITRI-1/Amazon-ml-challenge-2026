#!/bin/bash
# m4 end-to-end: ScaNN retrieval -> m2 filter -> CE fill -> train m4 -> predict test -> validate
set -euo pipefail
cd ~/nikhil/experiment
PY=~/nikhil/.venv/bin/python
L=logs/2026-09-26_m4
DROP=b_nonlatin,skel_ratio,core_contain_min,skel_tset,num_b_in_a,num_first_eq,a_jacc
echo "[m4] start $(date)"
$PY scripts/gen_candidates_scann.py --split train --model models/biencoder_e5s_v1 > ${L}_1_scann_train.log 2>&1
$PY scripts/gen_candidates_scann.py --split test  --model models/biencoder_e5s_v1 > ${L}_2_scann_test.log 2>&1
echo "[m4] scann done $(date)"
$PY scripts/filter_candidates.py --split train --queries ce_tr59,fold0 --tag train > ${L}_3_filter_train.log 2>&1
$PY scripts/filter_candidates.py --split test  --queries all --tag test > ${L}_4_filter_test.log 2>&1
echo "[m4] filter done $(date)"
$PY - <<'PY'
import polars as pl
C = "cache/"
f = pl.read_parquet(C + "filt_train.parquet")
tr = pl.read_parquet(C + "ce_tr59.parquet").select("r1").unique()
f.join(tr, on="r1").write_parquet(C + "filt_tr.parquet")
f.join(tr, on="r1", how="anti").write_parquet(C + "filt_va.parquet")
print("split filt_train")
PY
$PY scripts/ce_fill.py --split train --filt filt_tr --reuse ce_tr59 --tag tr_v2 > ${L}_5_ce_tr.log 2>&1
$PY scripts/ce_fill.py --split train --filt filt_va --reuse ce_val  --tag va_v2 > ${L}_6_ce_va.log 2>&1
$PY scripts/ce_fill.py --split test  --filt filt_test --reuse ce_test --tag test_v2 > ${L}_7_ce_test.log 2>&1
echo "[m4] ce done $(date)"
$PY scripts/train_matcher.py --cand_file filt_train --ctx_file cands_train_scann --ctx --ce tr_v2,va_v2 --drop $DROP --rounds 4000 --tag m4 > ${L}_8_train.log 2>&1
TH=$($PY -c "import json;print(json.load(open('models/matcher_m4/features.json'))['best_th'])")
echo "[m4] trained, threshold $TH $(date)"
$PY scripts/predict.py --matcher models/matcher_m4/lgb.txt --cand_file filt_test --ctx_file cands_test_scann --ctx --ce test_v2 --th $TH --out submission/output_m4 > ${L}_9_predict.log 2>&1
python3 utils/validate_submission.py --matching submission/output_m4/matching_results.tsv --candidate submission/output_m4/candidate_pairs.tsv --test-dir dataset/test --check-ids >> ${L}_9_predict.log 2>&1
echo "[m4] done $(date)"
