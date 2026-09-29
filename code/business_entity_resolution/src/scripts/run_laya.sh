#!/bin/bash
# Laya-multilingual encoder (Apache-2.0, 307M) fine-tuned as a 2nd cross-encoder, then scores every filtered pair.
set -euo pipefail
cd ~/nikhil/experiment
PY=~/nikhil/.venv/bin/python
L=logs/2026-09-26_laya
echo "[laya] start $(date)"
$PY scripts/train_crossencoder.py --init models/laya_ml_encoder --out models/ce_laya_v1 --ns1 100000 --k 12 --bs 128 --lr 2e-5 > ${L}_1_train.log 2>&1
echo "[laya] trained $(date)"
$PY scripts/ce_fill.py --split train --filt filt_va  --model models/ce_laya_v1 --tag laya_va   > ${L}_2_score_va.log 2>&1
echo "[laya] val scored $(date)"
$PY scripts/ce_fill.py --split train --filt filt_tr  --model models/ce_laya_v1 --tag laya_tr   > ${L}_3_score_tr.log 2>&1
$PY scripts/ce_fill.py --split test  --filt filt_test --model models/ce_laya_v1 --tag laya_test > ${L}_4_score_test.log 2>&1
echo "[laya] done $(date)"
