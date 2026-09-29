#!/bin/bash
# BGE reranker (fine-tuned on Blackwell via blackwell_kit, fp16 folder copied to models/ce_bge_v1)
# -> score val/train/test filtered pairs -> m8 = m6 + BGE logit -> ownership fix -> validator.
set -euo pipefail
cd ~/nikhil/experiment
PY=~/nikhil/.venv/bin/python
L=logs/$(date +%F)_bge
M=models/ce_bge_m3_fp16
echo "[bge] start $(date)"
$PY scripts/ce_fill.py --split train --filt filt_va   --model $M --hf --bs 512 --tag bge_va   > ${L}_1_score.log 2>&1
echo "[bge] val scored $(date)"
$PY scripts/ce_fill.py --split train --filt filt_tr   --model $M --hf --bs 512 --tag bge_tr   >> ${L}_1_score.log 2>&1
echo "[bge] train scored $(date)"
$PY scripts/ce_fill.py --split test  --filt filt_test --model $M --hf --bs 512 --tag bge_test >> ${L}_1_score.log 2>&1
echo "[bge] test scored $(date)"
$PY scripts/ensemble_matcher.py --seeds 1 --extra laya,laya2,bge --tag m8 --write > ${L}_2_m8.log 2>&1
TH=$($PY -c "import json;print(json.load(open('models/matcher_m8/ensemble.json'))['best_th'])")
$PY scripts/ownership_fix.py --pred testpred_m8 --th $TH --src submission/output_m8 --out submission/output_m8o >> ${L}_2_m8.log 2>&1
python3 utils/validate_submission.py --matching submission/output_m8o/matching_results.tsv --candidate submission/output_m8o/candidate_pairs.tsv --test-dir dataset/test --check-ids >> ${L}_2_m8.log 2>&1
echo "[bge] done $(date)"
