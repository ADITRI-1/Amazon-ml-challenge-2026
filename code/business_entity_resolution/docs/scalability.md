# Scalability: two pipeline modes

The pipeline runs in two modes that share every trained model and differ only in retrieval and in how many
cross-encoders score each pair. **`scalable` is the default.** `exact` is for small data and is the mode that produced
the shipped `output/` (public leaderboard 0.9873).

```
bash src/scripts/run_pipeline.sh                 # --mode scalable (default)
bash src/scripts/run_pipeline.sh --mode exact    # small data / reproduce output/ exactly
```

| | `exact` (small data) | `scalable` (default) |
|---|---|---|
| Retrieval | brute-force top-50 per country on GPU (all S1 × all S2/S3 of that country) | ScaNN per country: 2,000 partitions, 50% searched, 4-bit codes + exact re-rank of 250, CPU |
| Cost grows with | queries × pool (quadratic when both grow) | queries × searched share of the pool; index 8× smaller than float vectors |
| Memory per record (index) | 1.5 KB (float32 vector, must fit on GPU) | ~0.2 KB 4-bit codes searched in RAM + 1.5 KB float re-rank vectors (on disk in the saved index) |
| Candidate filter | m2 LightGBM, p ≥ 0.03 → 4.28 per S1 | same filter → same ~4.3 per S1 |
| Cross-encoders, India/US pairs | e5-small, Laya v1, Laya v2, BGE (4 passes) | **Laya v2 + BGE** (2 passes) |
| Cross-encoders, French pairs | e5-small, Laya v1, Laya v2 (+ Laya-FR) | same (France model unchanged) |
| India/US matcher | m8 (48 features + 4 CE scores) + specialist C | **m8lean** (47 features + Laya v2 + BGE) + specialist C |
| France | m6 → France map → Laya-FR | same, on the ScaNN candidates |

## When to use which
- **exact** when every country's query set × pool fits one GPU pass in reasonable time. This test set does:
  1.73M S1 × 9.97M S2/S3 took 16.6 min on one V100. It keeps every candidate the bi-encoder ranks in the top 50.
- **scalable** otherwise. Exact cost is proportional to queries × pool, so 10× more records on both sides means ~100×
  the work and float vectors that no longer fit on one GPU. ScaNN needs no GPU for retrieval, stores 4-bit codes,
  and exposes a recall/speed knob (`SEARCH_FRAC`).

## Measured on the challenge data (1× Tesla V100 32 GB, 40 CPU cores)

### Retrieval: ScaNN operating points (test set, label-free)
Share of the shipped 0.987 matched pairs that each setting retrieves in the top 30 (what the filter sees):

| ScaNN searched | France | India | US | Search time per query (France / India / US) |
|---|---|---|---|---|
| 15% | 99.20% | 99.85% | 99.96% | 0.15–0.6 ms |
| 30% | 99.77% | 99.95% | 99.98% | 0.31 / 0.89 / 0.70 ms |
| **50% (default)** | **99.93%** | **99.98%** | **99.99%** | 0.45 / 1.85* / 1.46* ms |
| exact | 100% | 100% | 100% | 16.6 min total on GPU (~0.58 ms per query) |

\*measured while another CPU job shared the machine. `scripts/scann_sweep.py` reproduces this table.
On validation (India/US, labelled), ScaNN at 15% cost 0.00025 F0.5 against exact (m8 0.98993 vs 0.99018); at 50% the
full scalable system is within 0.00003 of exact (0.99033 vs 0.99036, measured below).

### Scoring: lean cross-encoder stack
- Ablation (India/US validation, matcher retrained per set): all four cross-encoders 0.98993 · **no Laya v1 and no
  e5 (m8lean) 0.98995** · no Laya v2 0.98987 · BGE only 0.98959. On exact-search validation: all four 0.99020 vs
  BGE + Laya v2 0.99017. The e5 and Laya v1 cross-encoders add nothing once Laya v2 and BGE are present.
- Measured scoring speed (V100, fp16): e5-small ~9,000 pairs/s, Laya ~1,600–2,000 pairs/s, BGE ~1,200 pairs/s.

| Test scoring (7.41M filtered pairs; 1.17M French) | exact mode | scalable mode |
|---|---|---|
| e5-small | 7.41M pairs, ~14 min | French only, ~2 min |
| Laya v1 | 7.41M, ~79 min | French only, ~12 min |
| Laya v2 | 7.41M, ~79 min | 7.41M, ~79 min |
| BGE | 7.41M, ~106 min | India/US only (6.24M), ~90 min |
| **Total GPU time** | **~4.6 h** | **~3.1 h (−34%)** |

### Full scalable run, end to end (2026-09-29, test set)
`bash src/scripts/run_pipeline.sh` (ScaNN 50%, persistent indexes, lean stack) → `submission/output_scalable` (validator PASS).

| | exact (submitted output) | scalable |
|---|---|---|
| Validation F0.5, India + US (matcher + no-address specialist) | 0.99036 | **0.99033** |
| India / US | 0.99147 / 0.98962 | 0.99143 / 0.98961 |
| Test candidate pairs | 7,407,025 (4.28 per S1) | 7,397,955 (4.27 per S1) |
| Matched pairs identical to the submitted output (France / India / US) | 100% | **99.991 / 99.957 / 99.956%** |

Persistent test indexes (`models/scann_index/test_<country>/`): France 2.4 GB, India 7.9 GB, US 6.4 GB, built in
44 / 98 / 81 s; batch search 0.50 / 1.40 / 1.35 ms per query. Each index stores the ScaNN partitioner, 4-bit codes,
the float vectors used for exact re-ranking (the bulk of the size) and the record-ID map. Serving check:
`scripts/scann_query.py` on 3,000 real test records loaded from the saved indexes returned 99.9987% the same top-50
candidates as the batch run.

Wall-clock of the run on 1× V100 + 40 CPU cores: test retrieval 38 min, filter 36 min, India/US 6 min, France 2 min;
the cross-encoders only had to score 852 new pairs (all other scores reused), so a run from scratch adds the ~3.1 GPU-h
of scoring above. Training-set retrieval (needed once, for the matcher's validation and competitor features) took 67 min.

### Expected effect on the score
- India/US: lean matcher equal on validation (0.98995 vs 0.98993); ScaNN 50% keeps ≥ 99.98% of India/US matches.
- France: ScaNN 50% keeps 99.93% of the shipped French matches; French models unchanged. French pairs that only
  ScaNN retrieves have no Laya-FR score and keep the France-map decision.

## Larger GPUs
Cross-encoder scoring and bi-encoder embedding are GPU-bound and scale with the device (bf16, larger batches):
roughly 2–4× faster on A100/H100/Blackwell-class GPUs. ScaNN search, the filter, LightGBM and the France map run on
CPU and scale with cores. Exact search also gets faster on a bigger GPU but keeps its queries × pool growth.
