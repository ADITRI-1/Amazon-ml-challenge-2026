#!/bin/bash
# Laya v2: continue from ce_laya_v1 on new US/India pairs (ScaNN top-10, filtered-like distribution) + synthetic French.
set -euo pipefail
cd ~/nikhil/experiment
PY=~/nikhil/.venv/bin/python
L=logs/2026-09-26_laya2
echo "[laya2] start $(date)"
$PY scripts/train_crossencoder.py --init models/ce_laya_v1 --out models/ce_laya_v2 --cands cands_train_scann --ns1 100000 --s1_seed 7 --k 10 --synth_fr 60000 --bs 128 --lr 1e-5 > ${L}_1_train.log 2>&1
echo "[laya2] trained $(date)"
$PY scripts/ce_fill.py --split train --filt filt_va   --model models/ce_laya_v2 --tag laya2_va   > ${L}_2_score_va.log 2>&1
echo "[laya2] val scored $(date)"
$PY scripts/ce_fill.py --split train --filt filt_tr   --model models/ce_laya_v2 --tag laya2_tr   > ${L}_3_score_tr.log 2>&1
$PY scripts/ce_fill.py --split test  --filt filt_test --model models/ce_laya_v2 --tag laya2_test > ${L}_4_score_test.log 2>&1
echo "[laya2] done $(date)"
