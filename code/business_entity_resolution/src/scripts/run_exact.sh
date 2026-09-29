#!/bin/bash
# Exact (brute-force) retrieval instead of ScaNN -> same m2 filter -> reuse all CE scores, score only new pairs
# -> apply saved m6 and m8 -> ownership fix -> validator.  Compares scalable vs exact blocking with the newer encoders.
set -euo pipefail
cd ~/nikhil/experiment
PY=~/nikhil/.venv/bin/python
L=logs/2026-09-27_exact
echo "[exact] start $(date)"
$PY scripts/gen_candidates.py --split test  --model models/biencoder_e5s_v1 --k 50 --tag exact > ${L}_1_cands.log 2>&1
$PY scripts/gen_candidates.py --split train --model models/biencoder_e5s_v1 --k 50 --tag exact >> ${L}_1_cands.log 2>&1
echo "[exact] candidates $(date)"
$PY scripts/filter_candidates.py --split test  --cands cands_test_exact  --tag test_exact > ${L}_2_filter.log 2>&1
$PY scripts/filter_candidates.py --split train --cands cands_train_exact --queries fold0 --tag va_exact >> ${L}_2_filter.log 2>&1
echo "[exact] filtered $(date)"
for S in va test; do
  SP=$([ $S = va ] && echo train || echo test)
  $PY scripts/ce_fill.py --split $SP --filt filt_${S}_exact --reuse ce_${S}_v2 --model models/ce_e5s_v1 --tag ${S}_v2_exact >> ${L}_3_ce.log 2>&1
  $PY scripts/ce_fill.py --split $SP --filt filt_${S}_exact --reuse ce_laya_${S}  --model models/ce_laya_v1 --tag laya_${S}_exact  >> ${L}_3_ce.log 2>&1
  $PY scripts/ce_fill.py --split $SP --filt filt_${S}_exact --reuse ce_laya2_${S} --model models/ce_laya_v2 --tag laya2_${S}_exact >> ${L}_3_ce.log 2>&1
  $PY scripts/ce_fill.py --split $SP --filt filt_${S}_exact --reuse ce_bge_${S}   --model models/ce_bge_m3_fp16 --hf --bs 512 --tag bge_${S}_exact >> ${L}_3_ce.log 2>&1
done
echo "[exact] ce scored $(date)"
$PY scripts/predict_saved.py --matcher m6 --extra laya,laya2     --sfx exact --write m6x > ${L}_4_pred.log 2>&1
$PY scripts/predict_saved.py --matcher m8 --extra laya,laya2,bge --sfx exact --write m8x >> ${L}_4_pred.log 2>&1
for T in m6 m8; do
  TH=$($PY -c "import json;print(json.load(open('models/matcher_$T/ensemble.json'))['best_th'])")
  $PY scripts/ownership_fix.py --pred testpred_${T}x --th $TH --src submission/output_${T}x --out submission/output_${T}xo >> ${L}_4_pred.log 2>&1
  python3 utils/validate_submission.py --matching submission/output_${T}xo/matching_results.tsv --candidate submission/output_${T}xo/candidate_pairs.tsv --test-dir dataset/test --check-ids >> ${L}_4_pred.log 2>&1
done
echo "[exact] done $(date)"
