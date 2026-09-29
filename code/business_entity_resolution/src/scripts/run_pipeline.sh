#!/bin/bash
# Inference pipeline with trained models, two modes (see docs/scalability.md):
#   --mode scalable (DEFAULT)  ScaNN per-country retrieval (sublinear) + lean India/US scoring (Laya v2 + BGE only;
#                              e5 / Laya v1 cross-encoders run on French pairs only) -> submission/output_scalable
#   --mode exact               exact per-country top-50 (brute force; for small data) + full 4-cross-encoder stack
#                              -> submission/output_m8xhcLFd  (the shipped 0.987 output; locked copy in FINAL_0987_LOCKED/)
# Scores already computed for a pair are reused (ce_fill.py --reuse), so re-runs only score new pairs.
set -euo pipefail
cd "$(dirname "$0")/../.."; export PYTHONPATH=src
MODE=scalable
SEARCH_FRAC=${SEARCH_FRAC:-0.50}          # ScaNN share of the 2,000 partitions searched per query (chosen by scripts/scann_sweep.py)
while [ $# -gt 0 ]; do case $1 in --mode) MODE=$2; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done
PY=${PY:-python}
L=logs/$(date +%F)_pipeline_$MODE
log() { echo "[$MODE] $* $(date +%T)" | tee -a ${L}.log; }

if [ "$MODE" = exact ]; then
  [ -w submission/output_m8xhcLFd ] || { echo "output_m8xhcLFd is locked (read-only) - exact mode would overwrite the shipped file; unlock deliberately first"; exit 1; }
  $PY src/scripts/gen_candidates.py --split test --model models/biencoder_e5s_v1 --k 50 --tag exact;          log candidates
  $PY src/scripts/filter_candidates.py --split test --cands cands_test_exact --tag test_exact;              log filtered
  for M in "models/ce_e5s_v1 test_v2" "models/ce_laya_v1 laya_test" "models/ce_laya_v2 laya2_test"; do set -- $M
    $PY src/scripts/ce_fill.py --split test --filt filt_test_exact --reuse ce_$2_exact --model $1 --tag $2_exact; done
  $PY src/scripts/ce_fill.py --split test --filt filt_test_exact --reuse ce_bge_test_exact --model models/ce_bge_m3_fp16 --hf --bs 512 --tag bge_test_exact
  log scored
  $PY src/scripts/predict_saved.py --matcher m6 --extra laya,laya2 --sfx exact --write m6x
  $PY src/scripts/ownership_fix.py --pred testpred_m6x --th 0.725 --src submission/output_m6x --out submission/output_m6xo
  $PY src/scripts/build_m8xhc.py
  $PY src/scripts/france_map_variant.py
  $PY src/scripts/build_final_lfd.py;                                                                       log done
  OUT=output_m8xhcLFd
else
  S=scalable
  for SP in train test; do
    if [ "${RESUME:-0}" = 1 ] && [ -f cache/cands_${SP}_$S.parquet ]; then log "reuse existing cands_${SP}_$S"; continue; fi
    IDX=$([ $SP = test ] && echo "--index_dir models/scann_index" || true)   # persist the serving (test) indexes only
    $PY src/scripts/gen_candidates_scann.py --split $SP --model models/biencoder_e5s_v1 --search_frac $SEARCH_FRAC --tag $S $IDX; done
  log "candidates (ScaNN $SEARCH_FRAC)"
  $PY src/scripts/filter_candidates.py --split test  --cands cands_test_$S  --tag test_$S
  $PY src/scripts/filter_candidates.py --split train --cands cands_train_$S --queries fold0 --tag va_$S;     log filtered
  # lean India/US stack: Laya v2 + BGE on every pair
  for SP in va test; do SPL=$([ $SP = va ] && echo train || echo test)
    $PY src/scripts/ce_fill.py --split $SPL --filt filt_${SP}_$S --reuse ce_laya2_${SP}_exact,ce_laya2_${SP} --model models/ce_laya_v2 --tag laya2_${SP}_$S
    $PY src/scripts/ce_fill.py --split $SPL --filt filt_${SP}_$S --reuse ce_bge_${SP}_exact,ce_bge_${SP} --model models/ce_bge_m3_fp16 --hf --bs 512 --tag bge_${SP}_$S
  done;                                                                                                 log "scored (lean)"
  $PY src/scripts/build_india_us.py --matcher m8lean --sfx $S --out output_iu_$S;                            log india_us
  # France: e5 + Laya v1 + Laya v2 on French pairs only -> m6 -> France map -> Laya-FR
  $PY src/scripts/subset_country.py --filt filt_test_$S --country France --out filt_test_${S}_fr
  $PY src/scripts/ce_fill.py --split test --filt filt_test_${S}_fr --reuse ce_test_v2_exact,ce_test_v2 --model models/ce_e5s_v1 --tag test_v2_${S}_fr
  $PY src/scripts/ce_fill.py --split test --filt filt_test_${S}_fr --reuse ce_laya_test_exact,ce_laya_test --model models/ce_laya_v1 --tag laya_test_${S}_fr
  $PY src/scripts/ce_fill.py --split test --filt filt_test_${S}_fr --reuse ce_laya2_test_$S --model models/ce_laya_v2 --tag laya2_test_${S}_fr
  $PY src/scripts/france_m6.py --sfx $S
  $PY src/scripts/france_map_variant.py --pred testpred_m6_${S}_fr --base output_iu_$S --out output_${S}_MAP --cls_out france_map_cls_$S
  $PY src/scripts/build_final_lfd.py --map output_${S}_MAP --cls france_map_cls_$S --base output_iu_$S --out output_$S;  log done
  OUT=output_$S
fi
python3 src/utils/validate_submission.py --matching submission/$OUT/matching_results.tsv --candidate submission/$OUT/candidate_pairs.tsv --test-dir dataset/test --check-ids | tail -1 | tee -a ${L}.log
