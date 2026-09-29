# Work Log — ML Challenge 2026: Business Entity Resolution

Chronological record of every command, experiment, and training run.
Training/inference scripts write their full stdout to `logs/<YYYY-MM-DD>_<run-name>.log`; this file holds the summary + pointer.

---

## 2026-09-25 — Session 1: setup + problem read

**Env**
- Python 3.11.9 (`python`, not `python3` — `python3` is a Store stub on this box)
- GPU: RTX 4050 Laptop, 6 GB VRAM | CPU: Ryzen 7 7840HS, 16 threads | RAM: 15.3 GB | Disk free: 530 GB
- Installed: numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, torch 2.12.1
- Missing (likely needed): rapidfuzz, lightgbm, faiss / sentence-transformers

**Commands**
```
mkdir -p logs
wc -l dataset/{train,test}/*.tsv
head -3 dataset/train/train_source{1,2}.tsv dataset/train/train_ground_truth.tsv
```

**Data size (rows incl. header)**
| file | rows |
|---|---|
| train_source1 | 2,206,822 |
| train_source2 | 5,034,617 |
| train_source3 | 5,285,604 |
| train_ground_truth | 2,206,822 (1 row per S1) |
| test_source1 | 1,732,545 |
| test_source2 | 4,887,274 |
| test_source3 | 5,082,317 |

**Problem (from README)**
- Match each S1 record to 0..N records in S2/S3. Countries: US, India in train; **France only in test** (unseen).
- Metric: macro F0.5 per S1 entity; singletons score 1.0 on empty prediction, 0.0 on any prediction.
- Outputs: `output/matching_results.tsv` (scored), `output/candidate_pairs.tsv` (final blocking set; matches must be a subset).
- Constraints: no external data / APIs / geocoding; model must be MIT/Apache-2.0 and ≤ 8B params.
- Validate with `python utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test`.

**Early observations**
- S2 has Hindi (Devanagari) names and noisy prefixes (`-- Holloway Peak Inc Seafood`), uppercase addresses.
- Some S1 IDs are short (`S1-965667`) — IDs are not zero-padded, don't assume format.
- ~12M records per split on 15 GB RAM → memory is the binding constraint, not GPU.

---

## 2026-09-25 — Session 2: moved to server (V100 box), re-read problem

**Env (this box — supersedes the laptop env above)**
- Working dir: `~/nikhil/experiment` (all work stays here)
- GPU: Tesla V100-PCIE-32GB (Volta sm_70 → fp16 + sdpa, never bf16) | CPU: 40 cores | RAM: 345 GB
- Disk: `/home` at 100%, ~56 GB free → keep intermediates small, delete model weights after use
- Python: `~/nikhil/.venv/bin/python` (torch 2.5.1+cu121 — do NOT upgrade torch past cu12x)
- Missing in venv: pandas, rapidfuzz, lightgbm, faiss, sentence-transformers → install before EDA

**Commands**
```
ls -la ~/nikhil/experiment ; cat README.md Documentation_template.md logs/log.md
python -c "count '\n' per TSV"      # row counts identical to Session 1 table
nvidia-smi ; nproc ; free -g ; df -h ~
```

**Notes**
- Row counts verified identical to Session 1 table (python byte count; ignore transient `wc` numbers seen mid-copy).
- Read TSVs with `sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False` (no CRLF present;
  empty addresses exist, so don't let pandas turn them into NaN).
- RAM no longer the constraint (345 GB); 40 cores → CPU-parallel string features are cheap.
- Test adds France (unseen): features must be language/country-agnostic (char n-grams, normalised tokens).

**Deps install** (full output: `logs/2026-09-25_install-deps.log`)
```
uv pip install --python ~/nikhil/.venv/bin/python -c constraints.txt(torch==2.5.1+cu121) \
  pandas pyarrow polars rapidfuzz lightgbm xgboost scikit-learn faiss-cpu unidecode jellyfish tqdm
```
- Got: pandas 2.3.3, polars 1.44.2, pyarrow 25.0.1, rapidfuzz 3.14.5, lightgbm 4.7.0, xgboost 3.2.0,
  scikit-learn 1.7.2, faiss-cpu 1.15.1, unidecode 1.4.0, jellyfish 1.2.1
- torch still 2.5.1+cu121, CUDA available ✔
- Storage: 56 GB free is fine for code/features/logs; logs can be pruned anytime.

---

## 2026-09-25 — Session 2b: EDA + blocking analysis (NO model training yet)

**Layout created**
- `src/er/io.py` (TSV/parquet loaders), `src/er/norm.py` (unidecode→lower→&→and→collapse repeats; `name_core` = legal words stripped; `name_skel` = consonant skeleton for Indic transliteration), `src/er/embed.py` (char 2-4gram TF-IDF → GPU randomized SVD 256d; exact GPU cosine top-k)
- `scripts/build_norm_cache.py` → `cache/{train,test}_norm.parquet` (~1 GB each, 80 s + 70 s on 36 procs)
- `scripts/nb_text.py` — dump notebook outputs as text
- notebooks: `notebooks/01_eda.{py,ipynb}`, `notebooks/02_blocking.{py,ipynb}` (jupytext pairs; kernel "ER (.venv)")
- run: `jupytext --to ipynb X.py && jupyter nbconvert --execute --inplace --allow-errors --ExecutePreprocessor.kernel_name=er X.ipynb`

**EDA findings (01_eda)**
- GT: 2.21M S1, 7.64M true pairs; each S2/S3 id belongs to ≤1 S1 (pure 1:N). Singletons 5.6% (both countries). Mean 3.46 matches/S1 (1.7 S2 + 1.8 S3). ~26% of S2/S3 records are unmatched distractors.
- Country agreement on true pairs = 100% → block within country is free.
- **Test is harder**: S2+S3 per S1 = 5.8 (train 4.7) → more distractors. Test countries: India 810k, US 663k, France 259k S1.
- India names non-Latin in 23% of S2, 13% of S3 (Devanagari dominant, 9 scripts). India addresses: no PIN codes at all; state appears as full name (S1), 2-letter code (S3), or native script (S2/S3).
- US: S2 addresses ~94% uppercase; S3 uses full state names; ~10% of S2/S3 addresses empty/placeholder (`<NULL>`, `N/A`, `NULL`).
- ~3.5% names are domains (`andersoncarranzablumenberg.com`).
- Similarity AUC true vs random-neg: addr_tset 0.999, core_tset 0.981; vs hard-neg (same first name token): addr 0.993, name ~0.86–0.91 → **address is the strongest signal**; names alone are dangerous for precision.
- Some true matches have totally different names (random brand "Xylobelo One", acronyms "ET"←"Easy Tools") but identical address → need address-driven candidate path.
- Hard negatives: identical names at different addresses (~10% of same-first-token negs have core_tset=100); also near-identical name+address non-matches (e.g. "LNB SECURITY OVERSEAS" vs "Lnb Security") → precision risk.
- Consonant skeleton lifts Devanagari name similarity 69→81 (token_set).
- France (test only): S1 last part = region (Hauts-de-France / Nouvelle-Aquitaine / Pays de la Loire); S2/S3 often use département (Nord, Gironde, Loire-Atlantique) or drop it; few big cities → huge city blocks. Legal forms SARL/SAS/EURL/SASU/SCI added to LEGAL stopwords.

**Bugs hit**
- Kernel died: hard-negative join on first token exploded (common tokens) → capped to 3 records per (token, country).

**Blocking findings (02_blocking; Q = 20k S1 [10k US, 10k India] vs FULL train pool; results table: `logs/02_blocking_results.csv`)**
- Note: pair_recall@k is capped at ~k/3.46 for small k (3.46 true matches per S1), so look at F05_ceiling too.
- Exact keys: none usable alone (best cheap one: hnum+core_first 55% recall @ 34 cands/S1; name_core_exact 46%).
- Rare-token blocking: name ≤67% recall even at ~3k cands; addr r3/cap10k 90% @ 4.7k cands. Test-scale volumes explode (France/India name tokens: p50 ~30k cands) → rejected as primary path.
- Dense char-2-4gram TF-IDF→SVD256 (GPU exact top-k, within country):
  | scheme | pair_recall | F05 ceiling | cands/S1 |
  |---|---|---|---|
  | combo(name|addr)@50 | 0.769 | 0.895 | 50 |
  | combo@200 | 0.825 | 0.925 | 200 |
  | name@50 ∪ addr@50 ∪ combo@50 | 0.888 | 0.956 | 136 |
  | name@100 ∪ skel@50 ∪ addr@100 ∪ combo@100 | 0.915 | 0.969 | 293 |
  | all 4 dense@200 | 0.935 | 0.977 | 644 |
  | all dense@200 ∪ all tokens r3/10k | 0.988 | 0.996 | 9012 (too big) |
- India is the weak side everywhere (F05 ceiling ~2–3 pts below US).
- Residual misses (812/69k in widest union): ~60% are **Indic-script names** (Devanagari/Tamil/Kannada/...) with truncated S2/S3 addresses → unidecode transliteration + char n-grams can't bridge scripts.
- 1-hop pool→pool expansion (combo@20 + each cand's 10 pool-NN): 0.801 recall @ 96 cands vs combo@50 0.769 @ 50 → S2/S3 cluster members help find each other; worth keeping.
- Runtime: notebook 720 s, 60 GB RAM peak. GPU OOM fixed (gpu_topk qbs 4096→1024).

**Decision / next**: char-SVD embeddings are the recall bottleneck (compressed, script-blind). Plan: train a **contrastive bi-encoder** (small multilingual Apache/MIT model) on the 7.6M GT pairs for blocking — should read Indic scripts natively and lift recall@50 substantially. Keep char-SVD + token paths as cheap union members.

---

## 2026-09-25 — Session 3: bi-encoder for blocking

**Setup**
- `scripts/make_split.py` → `cache/split.parquet`: S1 fold = hash(entity_id, seed 2026) % 10; **fold 0 = validation for everything** (~88k India + ~132k US S1).
- Model: `intfloat/multilingual-e5-small` (MIT, 117.7M params, 384d), cached in `~/nikhil/hf` (471 MB). Input text `query: {name} | {address}`, max_len 64 (p99 = 61 tokens, 0.7% truncated).
- Encoding speed (fp16, V100, length-sorted): ~23k rec/s.
- New code: `src/er/biencoder.py` (load/encode), `src/er/metrics.py` (macro F0.5 + blocking report), `scripts/eval_blocking_dense.py`, `scripts/train_biencoder.py`.
- Smoke test: 51k pairs, bs 256 → ran clean (766 pairs/s while sharing GPU with eval).

**Train vs test shift check** (KS on lengths per src×country): D ≤ 0.034 everywhere → no length skew.
Real shifts: S2/S1 & S3/S1 ratio 2.3 → 2.85 (+23% records per S1 in test); country mix (test: India 47%, US 38%, France 15%).
Policy: no count/block-size/cluster-size/ID features; per-pair probability threshold (no top-N / match caps); stress-test thresholds on val with extra injected distractors.

**Runs**
```
python scripts/eval_blocking_dense.py --tag e5s_base          # log: logs/2026-09-25_eval_blocking_e5s_base.log
python scripts/train_biencoder.py --out models/biencoder_e5s_v1 --bs 512 --lr 5e-5 --hard_frac 0.3 --save_every 3000
                                                              # log: logs/2026-09-25_train_biencoder_e5s_v1.log
```

**Baseline: e5-small, NO fine-tuning** (val fold 0, 10k US + 10k India queries vs full train pool; 737 s; `logs/eval_blocking_dense_e5s_base.csv`)
| k | pair_recall | F05 ceiling | India | US |
|---|---|---|---|---|
| 5 | 0.853 | 0.963 | 0.953 | 0.974 |
| 10 | 0.941 | 0.981 | 0.972 | 0.990 |
| 50 | 0.965 | 0.989 | 0.983 | 0.994 |
| 200 | 0.977 | 0.992 | 0.989 | 0.996 |
→ already crushes char-SVD (combo@10 0.685 / @200 0.825). Multilingual model reads Indic scripts natively.
- Queued auto-start of training silently died (waiter process vanished, no output) → relaunched with `nohup` (pid 896958).
- Note: raw per-run logs from sessions 2/2b (install, cache build, nb runs) were removed from logs/ externally; summaries above are complete.

**Quick matcher probe** (`scripts/quick_matcher_probe.py 20 e5s_base`; log `logs/2026-09-25_quick_matcher_probe_base_k20.log`)
- e5-base top-20 cands of the 20k val queries (400k pairs), 33 pair-local features (`src/er/features.py`, 6 s / 400k pairs on 24 procs),
  LightGBM trained on 10k queries, scored on the other 10k, threshold swept.
- F0.5: pair-local only **0.9396** (th 0.70) | + dense cosine **0.9411** | + rank & gap-to-best 0.9422 (only +0.001 → dropped: density-dependent).
- Ceiling for these cands (@20) = 0.985 → matcher currently loses ~4.5 pts. Top gains: num_jacc, a_tset, dense_score, a_contain_b.
- Training v1 bi-encoder: 1,785 pairs/s, loss 0.028 by step 500 (random in-batch negs too easy → v2 should mine hard negatives).

**Decoy analysis** (`scripts/_decoy_ops.py`, val top-20 e5 cands)
- Top matcher FPs were letter-doubling copies of an S1 at the same address ("Ferrora"→"ferrrora", "Verrola"→"Verrrola"); those records belong to NO S1 (planted decoys). My norm collapsed repeats → erased the signal.
- Name collisions are common (vocabulary-generated names, e.g. ~50 "Ujjaraj …" S1s): same core name + different address → P(match)=0.36; same name + near-identical address → 0.95.
- With near-identical address: digit↔letter subs P(match)=0.975 (real OCR noise); letter-doubling insert 0.73 (decoy-ish); c↔k swap 0/36.
- Identical name + near-identical address: all address numbers equal → P(match)=1.000 (10,484/10,484); numbers differ & not subset → 0.822 (house/unit shift decoys).
- v1 bi-encoder: in-batch acc ≈1.0, loss ~0.02 → random negatives too easy. Stopping v1 at step 3000; chain `scripts/_chain_v1_to_mining.sh` evals ckpt then mines hard negatives (`scripts/mine_hardneg.py`, top-30 non-matches for 600k train S1). `train_biencoder.py` now takes `--hardneg`.

**Features v2** (45 = 33 + raw-name edit-op signature [ins_dup, del_dedup, sub_digit, …] + address-number agreement [nums_eq, nums_b_sub_a, num_first_eq, num_first_reldiff])
- probe F0.5: pair-local 0.9396→**0.9461**; + dense cosine 0.9411→**0.9477**; + rank/gap 0.9482 (still dropped).

**v1 bi-encoder, step 3000 (1.5M pairs seen)** — `models/biencoder_e5s_v1` (470 MB); eval log `logs/2026-09-25_eval_blocking_e5s_v1_s3000.log`
| k | pair_recall | F05 ceiling | India | US |
|---|---|---|---|---|
| 5 | 0.914 | 0.985 | 0.987 | 0.984 |
| 10 | 0.990 | 0.997 | 0.997 | 0.998 |
| 20 | 0.995 | 0.998 | 0.998 | 0.999 |
| 50 | 0.997 | 0.999 | 0.999 | 0.999 |
| 200 | 0.999 | 0.9997 | 0.9996 | 0.9998 |
- Low in-batch loss was misleading: fine-tuning still lifted recall@10 0.941→0.990 and closed the India gap. Blocking ≈ solved.
- Decision: skip bi-encoder v2 (≤0.1 pt ceiling left). Killed base-model hard-negative mining (by PID). GPU time → matcher + cross-encoder,
  trained on v1 top-k candidates (v1's near-misses = the hard negatives the matcher must reject).
- Gotcha: `pkill -f <pattern>` also matches any shell whose cmdline contains the pattern (it killed the old waiter, then my own shell). Kill by PID.
- Launched `scripts/gen_candidates.py` (v1, k=50) for train then test → `cache/cands_{train,test}_v1.parquet`; logs `logs/2026-09-25_gen_candidates_{train,test}_v1.log`.

- Log cleanup (user request): removed per-run logs already summarised here (chain, make_split, quick probes, eval .log files — tables kept in eval_blocking_dense_*.csv).

---

## 2026-09-25 — Session 4: stage-1 candidates + matcher m1

**Candidates** (`scripts/gen_candidates.py`, v1, k=50): train 110.3M pairs (19 min: India 475 s, US 673 s); test France 117 s, India 529 s, US running.
→ `cache/cands_{train,test}_v1.parquet`

**Matcher m1** (`scripts/train_matcher.py --k 30 --ntrain 300000 --tag m1`; log `logs/2026-09-25_train_matcher_m1.log`; 983 s)
- train: 9.0M pairs (300k fold 1-9 S1, 1.03M pos); val: ALL fold-0 S1 → 6.62M pairs (761k pos)
- LightGBM 127 leaves, lr 0.05, hit 3000-round cap (val logloss still falling 0.0235→0.0233)
- **val F0.5 = 0.9696** @th 0.675 (India 0.9681, US 0.9705); train F0.5 0.9803 (gap 1.1 pt)
- threshold curve flat: 0.969–0.970 over th 0.625–0.75 → robust to shift
- top gain: dense_score ≫ a_tset > nums_eq > core_jw > num_first_reldiff > num_jacc
- ceiling @30 ≈ 0.998 → ~2.9 pts left in the matcher
- saved `models/matcher_m1/lgb.txt`, `cache/valpred_m1.parquet`, `logs/matcher_m1_threshold_sweep.csv`

**Submission 1 (m1)** — `python scripts/predict.py --matcher models/matcher_m1/lgb.txt --k 30 --th 0.675` (52.0M test pairs scored, ~26 min; log `logs/2026-09-25_predict_m1.log`)
- `output/matching_results.tsv` (97 MB) + `output/candidate_pairs.tsv` (692 MB, = the top-30 list the matcher scores)
- validator: **PASS** (also PASS with `--check-ids`). 1,732,544 S1 rows; 97,888 empty (5.65%, train singleton rate 5.6%).
- predicted pairs/S1: France 3.70, India 3.28, US 3.26 (val 3.4). Uncertain share (0.2<p<0.9): France 4.3%, India 1.5%, US 1.3% → France may be over-matching; check against LB.
- one-owner post-proc (each S2/S3 → max-p S1): val-only check +0.0003 (0.9696→0.9699); not applied yet, needs OOF over all train S1.
- Leaderboard: (pending upload by user)

**Submission package** → `neural_ninjas_submission/` (753 MB, not zipped yet)
- output/ (m1 TSVs), code/business_entity_resolution/{src/er, src/scripts, README.md, requirements.txt (pinned), run_all.sh}, Documentation_template.md (filled, m1 state)
- package copies: scripts' sys.path now points at src/; HF_HOME not hard-coded; `train_biencoder.py --stop_step` added (also in experiment copy) to reproduce v1 = step 3000.
- Must re-sync package when the pipeline changes (cross-encoder, post-processing, new threshold).

---

## 2026-09-25 — Session 5: push to 0.98 — error decomposition + competitor features

**Where m1 loses F0.5 (val, th 0.675; `scripts/_err_decomp.py`)** total loss 0.0304:
- FN-only (missed records) **0.0220** on 20% of S1 ← the main problem; FP-only 0.0047; singleton-FP 0.0020; FP+FN 0.0018; blocking misses only 0.0014
- 50k FN pairs sit in candidates with p mostly 0.05–0.6 (matcher uncertain, not wrong)
**Hypotheses tested on the uncertain band (0.05<p<0.9, 250k pairs, pos rate 0.25)**
- sibling similarity (c vs S1's confident matches): AUC name 0.42 / addr 0.60 → weak (decoys are near-copies too)
- **competitor S1** (best OTHER S1 having c in its top-50): e5 margin AUC 0.808; p + competitor feats AUC **0.920 vs 0.817** → big
- manual review: uncertain cases = empty-address name-only records (owned by same-name S1 elsewhere), random brand names at shared addresses, house-number shifts (noise vs decoy; same-source siblings share the noisy number)
**m2** = m1 + 8 competitor features (`src/er/context.py`: comp_score, comp_margin, name/addr tset to competitor, nums_eq, name/addr margins, has_comp)
- run: `train_matcher.py --k 30 --ntrain 300000 --ctx --rounds 4000 --tag m2`, then stress test `--comp_frac 0.5` (tag m2_stress50) to check sensitivity to S1 density (test US has ~half the S1 of train)
- **Cross-encoder** (`scripts/train_crossencoder.py`, `src/er/crossencoder.py`): init from bi-encoder v1, pair input `name | addr` × 2, max_len 128, BCE, trained on folds 1-4 (250k S1 × top-12 = 3M pairs) so it is out-of-sample for matcher (folds 5-9) and val (0). Log `logs/2026-09-25_train_ce_e5s_v1.log`.
- User rule: next-stage models (Qwen3-Reranker-4B/0.6B, Phi-mini, LLM judge 2–8B, LoRA/QLoRA) only after asking; never auto-start. Candidate order if m2+CE < 0.98: Qwen3-Reranker-4B (Apache) LoRA on uncertain band. Qwen2.5-3B excluded (research licence).

**m2 RESULT: val F0.5 = 0.9805** @th 0.70 (India 0.9796, US 0.9808) — **target 0.98 reached on val** (+1.09 over m1)
- train F0.5 0.9885 (gap 0.8 pt); early-stopped at 2454 rounds; flat curve: ≥0.9791 for th 0.55–0.85
- gain: comp_margin (31.9M) ≫ dense_score (9.4M) > num_jacc > num_first_reldiff > comp_name_margin
- log `logs/2026-09-25_train_matcher_m2.log`, model `models/matcher_m2/lgb.txt`, sweep `logs/matcher_m2_threshold_sweep.csv`
- caveat: running stress run (comp_frac 0.5) RETRAINS under lower density (kept 1.36M/2.21M S1 as competitors, incl. all train/val queries). Proper transfer check still needed: apply m2 unchanged to val features rebuilt with fewer competitors.
- predict.py: `--ctx` added (competitor features from all test S1 top-50 lists). Launched m2 test prediction → `output_m2/` (th 0.70), log `logs/2026-09-25_predict_m2.log`. output/ still = m1.

**Density transfer of m2** (`scripts/eval_transfer.py`, m2 unchanged, val features rebuilt with fewer competitor S1)
| competitor S1 kept (random) | F0.5 @0.70 | best (th) |
|---|---|---|
| 100% | 0.9805 | 0.9805 (0.70) |
| 50% | 0.9734 | 0.9747 (0.85) |
| 30% | 0.9669 | 0.9705 (0.90) |
- also retrain-under-stress (comp_frac 0.5): 0.9778.
- pessimistic: random removal also deletes true owners (impossible in test). Realistic owner-preserving run queued (`--keep_owners`).
- threshold drifts up when competitors are sparse → m2 over-matches on sparse data.

**Cross-encoder ce_e5s_v1** (33 min train; val scoring 6.6M pairs in 734 s ≈ 9k pairs/s; `cache/ce_val.parquet`)
- AUC all pairs: m1 0.99885, m2 0.99943, **CE 0.99957**; on m2-uncertain band (0.05<p<0.95, 150k): m2 0.870, **CE 0.945**
- val F0.5: CE alone **0.9785** (India 0.9799, US 0.9775); **avg(m2, CE) 0.9867** @0.55 (India 0.9881, US 0.9858) [th picked on val]
- CE is density-independent → should also reduce the m2 density risk. Next: m3 stack (m2 feats + CE) trained on folds 5-9 (CE out-of-sample).
- Robustness: polars left joins default to maintain_order="none"; pipeline builds A/B feature rows and CE text pairs positionally after left joins → added `maintain_order="left"` to all such joins (train_matcher, predict, score_ce, train_crossencoder, eval_transfer) + order-safe `attach()` for CE feature. Past results were aligned (CE AUC 0.99957 / m2 0.9805 impossible otherwise); fix guards future runs.
- train_matcher `--ce train_tag,val_tag` added for m3 (train S1 = the CE-scored folds 5-9 set).
- Launched **m3** = m2 feats + ce_logit, train S1 = 300k from folds 5-9 (CE out-of-sample): `train_matcher.py --k 30 --ctx --ce tr59,val --rounds 4000 --tag m3`
- Launched CE scoring of 52M test pairs → `cache/ce_test.parquet` (log `logs/2026-09-25_score_ce_test.log`)

**m3 RESULT: val F0.5 = 0.9886** @th 0.725 (train F0.5 0.9901, gap 0.15 pt; early-stopped at 517 rounds)
- = m2 features + ce_logit; trained on 300k S1 from folds 5-9 (CE out-of-sample), val = all fold 0
- gain: ce_logit 30.7M ≫ comp_margin 10.8M > dense_score 7.1M > num_jacc > num_first_reldiff
- progression: m1 0.9696 → m2 0.9805 → **m3 0.9886**

**Submission 2 (m2)** — val 0.9805; `predict.py --matcher models/matcher_m2/lgb.txt --k 30 --th 0.70 --ctx` (2313 s). Validator PASS (--check-ids). 5,837,737 pairs, 94.4% S1 with ≥1 match. Now in `output/` (m1 files moved to `output_m1/`). Leaderboard: pending upload.
- **Realistic density transfer (owners kept)** m2 @50% other-S1: **0.9806** (= full 0.9805), best th stays 0.70 → density risk only came from removing true owners (impossible in test). Log `logs/2026-09-25_eval_transfer_m2_owners.log`.
- owners kept @30% other-S1: m2 **0.9806** again → top-1 competitor is (almost always) the owner, so thinning unrelated S1 does not change it. Design choice (single best competitor, no counts) = density-invariant. Density risk closed for m2.
- **m3 density transfer** (`logs/2026-09-25_eval_transfer_m3.log`): random removal 50% → 0.9850 (best 0.9859 @0.85), 30% → 0.9818 (best 0.9839 @0.90) [m2: 0.9734 / 0.9669 → CE halves the sensitivity];
  **owners kept 50% → 0.9887** (= full 0.9886, best th 0.75). Density risk closed for m3.
- Gotcha repeat: an old self-matching waiter (pid 944899) had kept looping and blocked the m3 transfer queue; killed by PID.
- m3 owners kept 30% → 0.9888. predict.py `--ce` added (order-safe attach, asserts no missing scores). Queued m3 test prediction (th 0.725) → output_m3/ after ce_test saved.

**Submission 3 (m3)** — val 0.9886; `predict.py --matcher models/matcher_m3/lgb.txt --k 30 --th 0.725 --ctx --ce test` (1434 s). Validator PASS (--check-ids). 5,839,281 pairs; 94.3% S1 with ≥1 match. Now in `output/` (m2 in output_m2/, m1 in output_m1/).
- Package `neural_ninjas_submission/` updated to m3: outputs (cmp-identical, validator PASS), code (+context, crossencoder, train_crossencoder, score_ce, eval_transfer), src/run_all.sh with CE steps, README + Documentation rewritten (progression, competitor feats, CE, density table).

**LEADERBOARD (public) m3 = 0.982** (val 0.9886 → gap ~0.7 pt; likely France/unseen shift — to investigate)
- Log cleanup #2 (user request): removed per-run logs for m1/m2/stress/transfer/candidates/CE-scoring (numbers are recorded above). Kept: final-pipeline logs (bi-encoder, CE, m3 train, m3 predict) + result CSVs.

---

## 2026-09-25 — Session 6: m3 failure analysis (public LB 0.982 vs val 0.9886)

**m3 val loss = 0.0114** (`scripts/_err_m3.py`): FN in cands 0.0081, FP 0.0023 (half unowned decoys, half owned by another S1), blocking misses 0.0010.
**66% of all remaining loss = name-only (empty-address) candidates** (FN 0.0060 + FN_block 0.0008 + FP 0.0008). Next largest patterns ~0.0008 each (number-shift, brand name at same address).
- Name-only S2/S3 records: 345k; **97.7% are owned** (vs 74% overall) → not decoys; the question is WHICH S1 owns them.
- Sibling-copy hypothesis rejected (sibling exact-name AUC 0.53–0.57); exact name = own S1 name → only 22% match.
- Owners' core name shared by other S1 in-country: 44% (21% share with ≥6); record name == owner core only 55%.
- `scripts/_probe_assign.py`: name-only records have median **201 claimers**; missed name-only true pairs are **tied-best on name in 91%**, owner is e5-best claimer only 24% → **largely irreducible ambiguity** (same name, no address, many S1).
- With-address candidates: owner is e5-best claimer 99.5%; missed ones (5k) best-by-e5 67% → recoverable-ish, small.
- **Test conflict check (m3 preds, all test S1 scored):** records claimed by >1 S1 above th: France 2,344 records / 4,645 extra pairs (0.54% of France pairs) vs India 0.03%, US 0.02% → **France over-merges ~20×**.
  Examples: "Lille Club (SAS)" (empty addr) matched to 6+ different S1 named "Lille Club SAS" at different streets, p≈0.84–0.89 each; "Bordeaux Club SAS | BORDEAUX"; "dunkerque amicale sas". Generic French names (<City> Club/Amicale SAS) collide massively; name-only/city-only records get high p for every twin.
  In val (only 10% S1 scored) just 23 such conflicts → this failure is invisible to our validation; likely a main driver of the 0.7 pt LB gap.

---

## 2026-09-25 — Session 7: candidate-set size becomes a ranking criterion (organiser note)

Organisers: blocking must scale (billions of records) and a SMALLER candidate set per S1 ranks higher (candidate_pairs.tsv + code reviewed).
Current weaknesses: fixed top-30 (52M test pairs, 30/S1 vs 3.46 true matches avg); retrieval = exact GPU brute force within country (all-pairs per country).

**Budget curve on val** (`scripts/_budget_curve.py`, `logs/candidate_budget_curve.csv`; m3 preds restricted to kept set):
| rule | cands/S1 | p95 | pair recall | F0.5 |
|---|---|---|---|---|
| top-30 (current) | 30.0 | 30 | 0.9965 | 0.9886 |
| top-10 | 10.0 | 10 | 0.9909 | 0.9883 |
| top-5 | 5.0 | 5 | 0.9109 | 0.9778 |
| e5 gap ≤0.25 to top-1 | 5.0 | 10 | 0.9593 | 0.9835 |
| **pre-filter (m1 pair-feature LGB) p≥0.005** | **5.77** | 13 | 0.9950 | **0.9886** |
| pre-filter p≥0.01 | 5.28 | 12 | 0.9938 | 0.9885 |
| pre-filter p≥0.03 | 4.57 | 9 | 0.9898 | 0.9882 |
→ Plan: FAISS ANN retrieval (sublinear) → cheap pair-feature filter → CE/competitor/m3 on survivors. Also cuts CE cost ~5×. Cut-off ε: awaiting user choice.
- Finer grid + filter comparison (`scripts/_budget_m2.py`): **m2 (pair + competitor feats, no CE) as filter dominates m1**:
  m2 p≥0.02 → val 4.42/S1, F0.5 0.9886 | **p≥0.03 → val 4.17/S1, F0.5 0.9885; TEST 4.28/S1 (France 4.51, India 4.21, US 4.26)** | p≥0.05 → 3.89, 0.9884.
  m1 p≥0.03 → test France 7.07/S1 (generic-name twins not pruned).
- **DECISION (user delegated): blocking filter = m2 @ p≥0.03** (~7× smaller candidate set than top-30, F0.5 −0.0001).
- Order agreed with user: feature audit → scalable blocking (FAISS ANN + m2 filter) → only then full-train scoring for ownership fix.

**Feature audit** (`notebooks/03_feature_audit.ipynb`; 100k S1 from folds 5-9 ≈ 3M train pairs, 50k val S1; ablation table `logs/03_feature_audit_ablation.csv`)
- No constant features. 7 redundant pairs |ρ|≥0.95: hnum_eq≡num_first_eq (ρ=1.0, true duplicate), b_script~b_nonlatin, num_jacc~num_b_in_a, core_jacc~core_contain_min, core_tset~skel_tset, core_ratio~skel_ratio, a_jacc~a_contain_b.
- SHAP share: ce_logit 62.6%, comp_margin 15.4%, comp_addr_margin 3.7%, comp_name_margin 2.9%, comp_score 2.6%, core_first_eq 1.3%, dense_score 1.2%; 33 features <0.3% each.
- Ablation (sample baseline 0.9881; seed-to-seed diff 0.0001): drop 7 redundant → 0.9882 (+0.0001); drop 33 low-SHAP → 0.9877 (−0.0004);
  groups: name −0.0002, address 0.0000, decoy-edit −0.0003, addr-numbers −0.0001, dense 0.0000, **competitor −0.0085, cross-encoder −0.0099**.
- Decision: drop the 7 redundant (55→48). Keep the small features: together they are worth ~0.0004 and they carry the filter model (no CE there).

**ANN retrieval check #1** (`scripts/eval_ann.py`, FAISS IVFFlat nlist 4096, val fold-0 queries vs full train pool, top-50; `logs/ann_recall_check.csv`)
| country | nprobe | pool scanned | true-pair recall ANN | exact | ms/query |
|---|---|---|---|---|---|
| US | 8/16/32/64 | 0.2/0.4/0.8/1.6% | 0.954/0.980/0.990/0.994 | 0.998 | 0.43/0.65/1.3/2.4 |
| India | 8/16/32/64 | 0.2/0.4/0.8/1.6% | 0.887/0.935/0.967/0.984 | 0.997 | 0.28/0.49/0.88/1.6 |
- IVF build: US 229 s, India 44 s. India (mixed scripts) much harder for IVF. Next: `scripts/eval_ann2.py` (IVF nprobe 64/128 + HNSW32 efSearch 64/128/256) measured as FINAL F0.5 loss after m2 filter.
- ANN fixes: coarser IVF (1024 lists) worse/equal for India (nprobe32 2.27% lost, nprobe64 1.36%). fp32 exact reproduces fp16 kept set exactly (0 lost) → not a tie artefact; IVF truly misses. Lost India pairs are LESS non-latin/name-only than average. Bidirectional (reverse S1 index) killed: >40 min CPU for India reverse pass alone, impractical. Next: centred-embedding IVF (`scripts/eval_ann4.py`).
- Centred IVF: no gain (|mu|=0.137, e5 fine-tuned space is not strongly anisotropic): India nprobe16 7.80% vs 7.63% lost, nprobe32 4.33% vs 4.27%. Stopped.
- IVF-lost India pairs (nprobe128): 68% are exact rank 0-5 but moderate cosine (median 0.68 vs 0.86 for found) → "diffuse" queries whose neighbours lie in far cells; 1,702 lost are true matches, 1,483 of them m3 p≥0.9 → real F0.5 loss. Next: query-adaptive nprobe (`scripts/eval_ann5.py`).
- Query-adaptive IVF (escalate to nprobe1024 when top-1 < tau), India: tau .75 → 4.7% escalated, 2.17% lost, F .98645 | .80 → 7.2%, 2.09% | .85 → 13.8% (5.0% avg scan), 1.91%, .98723 | .90 → 34.9% (10.3% scan), 1.42%, .98811. Losses are NOT concentrated in low-top1 queries → poor value; stopped. Next: query expansion (`scripts/eval_ann6.py`).
- Query expansion (search from top hits / Rocchio), India: +hit1 1.84% lost @3.1% scan; +hits1,2 1.66% @4.7%; +rocchio 1.95% @3.1%; +hit1+rocchio 1.73% @4.7% → all WORSE than plain nprobe128 (1.35% @3.1%). Stopped.

**Housekeeping (user request, 2026-09-25 late)**
- User moved output folders into `submission/` (output = m3 submitted, output_m2 = fallback). `neural_ninjas_submission/` package no longer exists (not found anywhere) → will be rebuilt at m4 finalisation from code + log.
- Moved analysis-only scripts to `scripts/analysis/` (probes, ANN evals, quick probe, hard-neg miner); paths fixed. Pipeline scripts stay in `scripts/`.
- Deleted: cache (old top-200 lists, analysis parquets, m1/stress preds, reverse-search S1 embeddings), models/matcher_m1 + matcher_m2_stress50, submission/output_m1, submission/output_m3 (byte-identical to output), ANN run logs + nb03 run log (results in CSVs above). Val embeddings (~8 GB) kept until ScaNN test ends.
- Installed `scann==1.4.2` (torch pinned 2.5.1+cu121). Laya (NandhaKishorM, Apache-2.0, 322M/421M) evaluated on paper — needs torch 2.14+ (separate venv, sm_70/fp16 check); awaiting user go-ahead before any download.
- Sanity: IVF nprobe=nlist (exhaustive) → 0 final cands lost (no bug/mismatch); nprobe128 losses have NO identical-text twin in ANN list → genuine IVF misses (India 20k-query sample: 1.30%, 37.5% true).
- **Test agreement (no labels), IVF4096:** France nprobe64 → 94.7% final cands / 96.5% predicted matches found; nprobe128 → 97.0% / 98.2% (≈1.8% of France matches lost > 1% budget).
- **Test agreement complete (IVF4096; India/US on 100k sampled S1 — sample matches full India@64 to 0.01 pt):**
| country | nprobe | final cands found | predicted matches found |
|---|---|---|---|
| US | 64 / 128 | 99.29% / 99.64% | 99.67% / 99.85% |
| India | 64 / 128 | 96.63% / 97.85% | 98.66% / 99.42% |
| France | 64 / 128 | 94.69% / 96.94% | 96.51% / 98.23% |
→ nprobe128 overall ≈ 0.6% of predicted matches lost (weighted by S1: France 1.77%, India 0.58%, US 0.15%) — within user's 1% budget; France is the outlier.
- **ScaNN vs IVF (val, final F0.5; exact: US 0.98775, India 0.98968)** (`scripts/analysis/eval_ann_scann.py`, `logs/ann_scann_check.csv`):
| index | US lost / ΔF | India lost / ΔF | ms/q |
|---|---|---|---|
| IVF4096 nprobe128 (3.1%) | 0.33% / −0.0004 | 1.35% / −0.0017 | 2.6–4.1 |
| ScaNN 2000 leaves, 2% | 1.52% / −0.0021 | 3.81% / −0.0093 | 0.08–0.11 |
| ScaNN 2000 leaves, 4% | 0.62% / −0.0006 | 1.91% / −0.0033 | 0.13–0.19 |
| ScaNN 4000 leaves, 4% | 0.65% / −0.0006 | 2.00% / −0.0035 | 0.14–0.19 |
→ ScaNN ~20× faster per query at equal recall; leaves count irrelevant; needs ~8–16% of leaves to beat IVF128 recall. Current pick: IVF4096 nprobe128 (test within 1% budget). Pending user: extended ScaNN grid (8/12/20%).
- **DECISION (user): retrieval = ScaNN** (scalability: ~20× faster/query, ~8× less memory via 4-bit AH quantization, billion-scale proven). Running operating-point grid (`scripts/analysis/eval_ann_scann2.py`): leaves 2000, search 6/10/15/25%, val F0.5 + test agreement incl. France.
- **ScaNN operating point** (`logs/ann_scann_operating_point.csv`; leaves 2000, AH-4bit + reorder 250):
| leaves searched | test France / India / US (pred matches found) | val ΔF0.5 India / US | ms/q |
|---|---|---|---|
| 6% | 97.50 / 99.43 / 99.90 | −0.0018 / −0.0003 | 0.08–0.29 |
| 10% | 98.69 / 99.73 / 99.94 | −0.0009 / −0.0002 | 0.11–0.41 |
| **15%** | **99.28 / 99.86 / 99.96** | **−0.0005 / −0.0001** | 0.15–0.60 |
| 25% | 99.69 / 99.93 / ~99.97 | −0.0002 / −0.0001 | 0.24–1.05 |
| IVF4096 nprobe128 | 98.23 / 99.42 / 99.85 | −0.0017 / −0.0004 | 2.6–4.1 |
**DECISION: ScaNN, 15% of leaves, single setting for all countries.** Blocking = ScaNN top-50 per S1 (within country) → m2 filter p≥0.03 (~4.3 cands/S1) = candidate_pairs.tsv → CE + competitor feats + final LGB.

**Disk cleanup (2026-09-26 morning):** `uv cache clean` (~/.cache/uv 19 GB, old vLLM/cu130 wheels; venv verified OK after), removed duplicate ~/.cache/huggingface (e5-small, project uses ~/nikhil/hf), deleted cache/testpred_matcher_m2/m3 (old top-30 preds) + val-query embeddings. Kept pool/test embeddings, cands, CE scores, norm cache for the ScaNN blocking build. Free: 30 GB → 44 GB.

---

## 2026-09-26 — Session 8: m4 = ScaNN blocking + m2 filter + retrained matcher
- New: `scripts/gen_candidates_scann.py`, `scripts/filter_candidates.py`, `scripts/ce_fill.py`, `scripts/run_m4.sh`; train_matcher/predict gained `--cand_file`, `--ctx_file`, `--drop` (+ models/<m>/features.json).
- Chain launched (`logs/2026-09-26_m4_*.log`): ScaNN train/test → m2 filter (p≥0.03) → CE fill (reuse) → m4 (48 feats; train = folds 5-9 CE set, val = fold 0) → predict → submission/output_m4 + validator.
- IVF-PQ benchmark queued (`scripts/analysis/eval_ann_ivfpq.py`): IVF4096+PQ48x8 + IndexRefineFlat(k_factor 5), nprobe 64/128/256, 12 threads; val final F0.5 (India/US) + test recall@10 vs exact for IVF-PQ AND ScaNN-15% (France/India/US). Note: m2/m3 test preds were deleted in morning cleanup → test comparison uses recall@10 vs exact top-10 for both methods.

**m4 RESULT** (`scripts/run_m4.sh`, 07:47 → 09:20, ~1.5 h end-to-end)
- ScaNN retrieval: train 110.3M pairs (India 0.41, US 0.58 ms/query; build 78–106 s), test 86.6M (France 0.21, India 0.46, US 0.38 ms/query).
- m2 filter p≥0.03: train 4.16 cands/S1 (2.17M/15.6M), **test 4.25/S1 (7.36M pairs vs 52.0M in m3)**; France 4.46, India 4.18, US 4.25.
- CE fill reused ~99.9%: new pairs scored train 463, val 346, test 4,120.
- m4 LightGBM, 48 feats, train 1.25M pairs (82% pos), val 917k: **val F0.5 0.98826 @th 0.70 (India 0.98914, US 0.98767)**; train 0.9898; 360 rounds.
  vs m3 0.98863 → −0.0004 for a fully sublinear pipeline with 7× fewer candidates. Curve flat: ≥0.9871 over th 0.50–0.90.
- Test: 1,732,544 rows, 94.3% with ≥1 match, 5,833,480 pairs; 99.58% of m3's predicted pairs shared. Predict 262 s (m3: 1434 s).
- `submission/output_m4/` (candidate_pairs 117 MB vs 692 MB) — validator **PASS** (--check-ids).
- **IVF-PQ benchmark** (`logs/ann_ivfpq_check.csv`; IVF4096+PQ48x8+IndexRefineFlat k_factor 5, 12 threads):
  val final F0.5 India @64/128/256: 0.98529 / 0.98765 / 0.98855 (exact 0.98968); US: 0.98671 / 0.98722 / 0.98742 (exact 0.98775)
  test pred-matches found (m3 pairs rebuilt from submission/output) @256: France 98.89, India 99.60, US 99.89 vs **ScaNN-15% 99.26 / 99.86 / 99.96**;
  recall@10 vs exact @256: 97.2 / 93.3 / 96.6 vs ScaNN 97.9 / 95.8 / 97.9. → ScaNN better everywhere; FAISS-GPU IVF-PQ remains a throughput option only.

---

## 2026-09-26 — Session 9: Laya cross-encoder + seed ensemble (user-approved)

- **Generalisation check (m4, fold 0 = 220k S1 never used for training):** F0.5 0.9883, 95% bootstrap CI [0.9880, 0.9885]; 10 disjoint slices 0.9877–0.9886; m4−m3 = −0.00037 [−0.00047, −0.00027].
- Leverage analysis: test conflicts (4,645 extra claims) cap the ownership fix at ≈+0.0005; LB gap (0.0066) most plausibly France accuracy (15% of test at ~0.95 would explain it) → stronger multilingual judge = main lever.
- **Laya-multilingual** (`convaiinnovations/laya-multilingual`, Apache-2.0; langs incl. fr, hi, bn, ta, te, kn, ml): ModernBERT (mmBERT-base) encoder, 22 layers, 307M; extracted `encoder.*` → `models/laya_ml_encoder` (ModernBertModel, loads in existing venv, no torch upgrade). V100 fp16 check: no NaN/Inf, max |fp16−fp32| 0.28 on |h|≤42; inference ~2,775 pairs/s; train ~350 pairs/s (bs 128).
- Launched `scripts/run_laya.sh`: CE fine-tune on 100k fold 1-4 S1 × top-12 (1.2M pairs, lr 2e-5) → score filt_va / filt_tr / filt_test (~9.5M pairs).
- Seed ensemble launched (`scripts/ensemble_matcher.py --seeds 5 --tag m4e --write`, CPU 24 threads): m4 features built once, 5 LGB seeds averaged; → submission/output_m4e/ + validator.
- **Seed ensemble result (m4e):** single seeds 0.98821–0.98830 (mean 0.98826 ± 0.00003); 5-seed average 0.98827 @th 0.675 (India 0.98922, US 0.98765) → **+0.00001, within noise: no gain** (LightGBM variance already negligible). submission/output_m4e written but equivalent to m4 — not recommended for upload.
- **LEADERBOARD (public) m4 = 0.981** (m3 0.982). Expected ≈ −0.001: val −0.0004 (India/US) + France ScaNN loss (0.72% of pred matches × 15% of test ≈ −0.0005…−0.0007). Trade accepted for scalable blocking (4.25 cands/S1, sublinear). France remains the LB lever.
- **Laya CE trained** (55 min, 1.2M pairs, final loss ~0.026 vs e5 0.034). Held-out filtered val pairs (917k): AUC e5 0.99489 → **Laya 0.99583** (avg 0.99608; India 0.99614, US 0.99553); **m4-uncertain band AUC 0.730 → 0.809**; name-only unchanged (0.907/0.906); standalone F0.5 e5 0.9799 → **Laya 0.9816** (`scripts/analysis/_laya_vs_e5.py`).
- **m5 = m4 feats + Laya CE logit (49 feats): val F0.5 0.98947 @th 0.725 (India 0.99021, US 0.98898)** vs m4 0.98826 (+0.0012, seed noise ±0.00003); 270 rounds. `logs/2026-09-26_m5_val.log`.
- **Submission m5** (`scripts/ensemble_matcher.py --seeds 1 --extra laya --tag m5 --write`): Laya test scoring 7.36M pairs in 4,718 s (~1.6k pairs/s real throughput). val 0.98947 (India 0.99021, US 0.98898). `submission/output_m5/` validator **PASS** (--check-ids); 1,633,180 S1 with ≥1 match.
- **LEADERBOARD (public) m5 = 0.983** (m4 0.981, m3 0.982): LB gain +0.002 > val gain +0.0012 → consistent with Laya reducing France over-merging (m5 dropped ~24.7k France pairs, added ~11.2k). Remaining LB–val gap ≈ 0.0065, still most plausibly France.

---

## 2026-09-26 — Session 10: France push (target LB ≥ 0.989)

- **Where we lack (label-free, m5 test preds):** uncertain pairs (0.1<p<0.9) France 12.4% vs India 3.7% / US 3.6% (val India 3.9%, US 5.1%); S1 with ≥1 unsure pair France 34.8% vs ~11%. India/US test ≈ val → LB 0.983 ⇒ France ≈ 0.945–0.96. Reaching 0.989 needs France ≈ 0.985.
- French gaps found: S2/S3 write "R" for Rue in ~25% of records (S1 always "Rue"), Av/Bd/Pl/Ch similar; S2/S3 end with département (Nord/Gironde/…) where S1 ends with région; legal forms SELARL/SCOP/ETS/CIE not in LEGAL.
- **Synthetic French data** (`src/er/synth_fr.py`), user-approved; **no test records used**: hand-written lexicon (legal forms, street types+abbrev, business words, names, 31 major cities w/ département+région — general cities, not the test cities) + training-measured noise ops + French variants (abbrev, département swap, accents drop) + hard negatives (twins, doubled-letter/shifted-number decoys, same-street others, name-only twins). Sample: 13.7k rows / 3k entities, 64% pos.
- **Laya v2** (`scripts/run_laya_v2.sh`): init ce_laya_v1, +100k new US/India S1 × ScaNN top-10 + 60k synthetic French entities, lr 1e-5 → rescore filt_va / filt_tr / filt_test.
- **Expected-F0.5 per-S1 decoding (`scripts/decode_expected_f05.py`, isotonic calibration on one half of fold 0, eval on the other): global th 0.98944/0.98950 vs expected-F 0.98896/0.98926 → −0.00036. Rejected** (candidate independence assumption fails: cluster members/rival claimers are correlated).
- Laya v2 training: 1,313,474 pairs (313,474 synthetic French, 56% pos), ~60 min.

- **Cleanup (user request):** deleted all cached embeddings (~17.7 GB; ScaNN candidates already built, regenerable ~20 min), old exact candidate lists, old CE score files (ce_test/ce_tr59/ce_val), superseded preds (testpred m4/m4e, valpred m2/m3/m4e), models matcher_m4e/matcher_m3/laya_ml_encoder (regenerable from ~/nikhil/hf), submission/output_m2 + output_m4e, per-step run logs already summarised here. Kept: everything the running Laya v2 + m5v2 jobs use, matcher_m2 (= blocking filter), ce_e5s_v1, ce_laya_v1, submissions m3 (output), m4, m5.
- **m5 + v2 features (54 feats): val 0.98946** vs m5 0.98947 → neutral on India/US (expected: their abbreviations are in training). Running on test for the France uncertainty diagnostic.
- **v2 features on test (France diagnostic):** France uncertain 12.37%→12.63%, S1-unsure 34.8%→35.5%; flips ~0.4% of France pairs each way; India/US unchanged → abbreviation/admin mismatches are NOT what drives France uncertainty (CEs already read raw text). v2 neutral on val and test → **dropped from m6** (simplicity). France gap = judge unfamiliarity with French entities → Laya v2 (synthetic French) is the decisive test.
- m6 queued: `ensemble_matcher.py --seeds 1 --extra laya,laya2 --tag m6 --write` (m5 feats + Laya v1 + Laya v2 logits) after Laya v2 test scoring; → submission/output_m6 + validator.
- **Blackwell kit** `blackwell_kit/` (user request): train_ce.py (standalone bf16 fine-tune, native reranker head of BAAI/bge-reranker-v2-m3, dev AUC/logloss monitoring, saves full + fp16 copy), requirements.txt (torch cu128+), README.md, data/: train_pairs 2.5M (250k fold 1-4 S1 × ScaNN top-10, 34% pos, no IDs), synth_fr_pairs 313k, dev_pairs 50k (5k other fold 1-4 S1). No fold-0 / test data. Smoke-tested here with e5-small + fp16 (60 steps: dev AUC 0.47→0.925, saved OK). Model to bring back: out/ce_bge_m3_fp16 → models/ce_bge_m3_fp16.
- **Laya v2 vs v1 (held-out filtered val, 917k pairs):** AUC 0.99583 → **0.99616** (India 0.99614→0.99659, US 0.99553→0.99579); m5-uncertain band AUC 0.7150 → 0.7432; standalone F0.5 0.9816 → **0.9823**. Synthetic French does not hurt India/US. Keep synth in the bge/Blackwell run. m6 (m5 + v1 + v2 logits) builds automatically after Laya v2 test scoring.
- **m6 (m5 feats + Laya v1 + Laya v2 logits, 50 feats) val F0.5 0.98962 @th 0.725 (India 0.99049, US 0.98904)** vs m5 0.98947 (+0.00015). France effect only measurable on test/LB.
- **Submission m6** `submission/output_m6/` (val 0.98962; India 0.99049, US 0.98904): validator **PASS** (--check-ids); 1,633,002 S1 with ≥1 match.
- **m6 + ownership fix** (`scripts/ownership_fix.py`: record predicted for >1 S1 → keep max-p claim): removed 5,180 duplicate claims (France 3,965, India 879, US 336) → `submission/output_m6o/`, validator PASS.
- LightGBM rounds check launched: (a) no early stopping, 1500 rounds; (b) lr 0.02 with early stopping (≤8000 rounds).
- **LightGBM training length (m6 features):** (a) no early stopping, 1500 rounds → 0.98959 @th 0.80 (India 0.99043, US 0.98903): no gain, over-confident; (b) lr 0.02 + early stopping (620 rounds) → **0.98969** @th 0.75 (India 0.99054, US 0.98913): +0.00007 vs m6 0.98962 (≈2× seed noise).
- **LEADERBOARD (public) m6o = 0.98314** (m5 ≈ 0.9828): +0.0003–0.0004 from Laya v2 (synthetic French) + ownership fix.
- **Cleanup #3 (user request):** removed v2 feature caches (1.14 GB, v2 dropped), m5v2/m6val/m6_full1500/m6_lr02 preds+models, matcher_m4, submission/output (m3, 754 MB) + output_m4, their run logs. Kept: submission m5 (≈0.9828), m6, m6o (0.98314); everything needed for Laya v3/m7.

---

## 2026-09-26 — Session 11: France filler-word pattern → Laya v3

- **LB m6o = 0.98314.** France ambiguity is NOT unusual (S1 names shared: France 53.7%, India 55.1%, US 35.3%; name-only 3.0/2.4/2.9%), but only 37% of France uncertain pairs are name-only (India/US ~73%) → France gap is mostly REDUCIBLE (addressed pairs).
- Manual review of uncertain French addressed pairs: same address + same distinctive token, but a generic filler/category word added or swapped (Centre/Primaire, Comite/Groupement, & Fils, Groupe, France…).
- **Training rule (same address, filtered candidates):** 1 word swapped → P(match) India 0.943 / US 0.964; 1 word added → 0.946 / 0.944; EXCEPT decoy words {group 0.00 (n=1300), holdings 0.00, industries 0.00, public 0.00, enterprises 0.00}. Noise words (≥0.9): ltd, center, services, partners, inc, corp, company, the, sri/shri/smt, dr/mr…
- User: organisers green-lit using test vocabulary for synthetic data (keep/drop decided later). `scripts/build_fr_vocab.py` → `cache/fr_vocab.json` (words/structure only; no labels/pseudo-labels/pairs): French added words: sarl/sas/…, france 4035, **groupe 4004**, services 691, développement 683, cie 668, international 559, **holding 493**, participations 339, distribution, associés, fils; swaps mostly accent insertion (club→çlub 2803, sarl→sàrl 4172…); 15 real (city, département, région) triples; 2000 street patterns.
- `src/er/synth_fr.py` **generate_v3**: test vocabulary + filler-added/category-swap operator (label 1) + decoy-filler negatives {Groupe, Groupement, Holding, Industries, Entreprises, Public} (label 0) + accent insertion. Sample: 14.4k rows / 3k entities, 62% pos.
- **Latin-only accent folding** for CE inputs (`crossencoder.fold_latin`: é→e, ç→c, œ→oe; Devanagari/Tamil untouched), stored per model in `text_cfg.json` (older models unaffected). String features already fold via unidecode.
- Launched `scripts/run_laya_v3.sh`: Laya v3 (init v2, fold_latin, synth v3 60k entities, 100k new US/India S1 × ScaNN top-10, lr 1e-5) → rescore → m7 (m6 + laya3 logit) → ownership fix → `submission/output_m7o` + validator.
- **Laya v3 vs v2 (held-out val):** AUC 0.99616 → **0.99629** (India 0.99659→0.99667, US 0.99579→0.99596); uncertain band 0.7205 → 0.7375; standalone F0.5 0.9823 → 0.9824. Training 58.6 min.

- 2026-09-26 23:00 doc: pipeline.html v5 published (m5/m6/m6o scores, Laya CE table, France gap section, ownership step, tried-and-dropped). v4 backup in scratchpad.
- 2026-09-27 00:12 m7 step failed: models/matcher_m4/features.json removed in cleanup #3 → reconstructed from run_m4.sh DROP list (48 kept), resumed via scripts/run_m7_resume.sh
- **m7** (m6 + Laya v3 logit, 51 feats): val F0.5 **0.98971** @0.725 (India 0.99055, US 0.98914) vs m6 0.98962. Ownership fix removed 16,211 claims → `submission/output_m7o` **validator PASS**.
- m7o vs m6o test pairs: France +35,725 / −5,053 (873,659 total); India/US ±~2.5k. Added French pairs are mostly filler additions (fils, et, développement, france, associés, groupe, legal forms), 18% with empty S2 address; eyeball sample mostly plausible matches, a few doubtful (distinctive-word swaps).
- Held-out synthetic French (seed 999; `scripts/analysis/_synth_holdout_eval.py`, log 2026-09-27_synth_holdout_eval.log), AUC synth-v1 / synth-v3: e5 0.792/0.791, Laya v1 0.849/0.812, v2 0.999/0.972, **v3 0.998/0.9995**; decoy-word false positives v1 0.299 → v2 0.155 → v3 0.000. Measures rule-learning only, not real France.
- **LEADERBOARD m7o ≈ 0.982** (user) < m6o 0.98314. The +35.7k French filler-word pairs added by Laya v3 were net WRONG → on test France, "same address + added filler word (Fils/Développement/France/Groupe/legal form)" is often a DISTRACTOR, so the train-measured rule (US/India) does not transfer. Synthetic v3 (test vocabulary) hurts → drop it. Best stays m6o.
- 2026-09-27 00:33 doc: pipeline.html v6 published (m7o LB ≈0.982 + v3 dropped, France-stricter lesson, synthetic holdout table, multi-cross-encoder next steps). v5 backup in scratchpad.
- **FN stage attribution (val, m6, th 0.725; `scripts/analysis/_fn_stage_attribution.py`)**: 764,081 true pairs; missed: not in ScaNN top-50 2,660 (0.35%), rank 30-49 710, cut by m2 filter 4,874 (0.64%), rejected by matcher 13,338 (1.75%); FP 1,509. S2 empty-address share of misses 62% / 77% / 76% / 81% (found: 2.3%). Oracle F0.5 gains: perfect retrieval +0.0011, ranks 30-49 +0.0003, perfect filter +0.0020, matcher accepts all FNs +0.0055, no FPs +0.0017. → encoder/retrieval work has a ceiling of +0.0011 on val; the loss is at the matcher, mostly S2 name-only records.
- 2026-09-27: user is fine-tuning bge-reranker-v2-m3 on the Blackwell box (blackwell_kit). Prepared import: `er.crossencoder.HFCrossEncoder` + `ce_fill.py --hf --bs` (smoke-tested with a stand-in model), chain `scripts/run_bge.sh` (score va/tr/test → m8 = m6 + BGE → ownership → validator). Expected model location: models/ce_bge_v1 (the kit's out_fp16 folder). Disk free 56 GB.
- 2026-09-27 00:41 cleanup #4 (user request): removed dropped m7 artefacts (ce_laya_v3 model, laya3 CE scores, m7 preds/model, output_m7, output_m7o), output_m5 + output_m6 (superseded by m6o; regenerable from testpred_m6), testpred_m5, pycache. Kept: m6o submission, laya v1/v2 + e5 models, all caches for m6 / BGE / stage-2.
- 2026-09-27 00:45 logs cleanup: removed per-run logs of dropped/deleted runs (laya3/m7, m3 train/predict, m4 predict, chain timestamp logs, m1–m3 threshold sweeps); results remain summarised in log.md.
- 2026-09-27 00:47 cache cleanup: removed fr_vocab.json (test-derived vocabulary; synthetic v3 dropped), filt_train (m4-only), valpred_m5 (analysis done). Remaining cache is all needed by m6o / BGE-m8 / stage-2.
- 2026-09-27 00:55 stage 2 (lean, user: keep only necessary features): m6 p + 7 context feats (b_noaddr, s1_best_other, r2_best_rival, sib_n, sib_name_max, rival_sib_max, n_claim); train on 5-fold OOF m6 probs; `scripts/stage2.py --tag s2 --write`, log 2026-09-27_stage2.log
- **Stage 2 result: val 0.98965 @0.75 vs m6 0.98962 → +0.00003 = noise. NOT adopted.** Gain share p_logit 99.5%, context feats ≤0.2% each. Name-only FN 10,755→10,553 but addressed FN 2,583→2,824.
  Diagnosis (name-only S2, val): FN look like TP on the context features: sibling present 98%, sibling-name token_set 100 (too lenient: subset=100), no rival in the filtered set (n_claim 1). What separates them is how ambiguous the name is: 27% of FN have ≥2 S1 records with the identical normalised name (TP 0.6%); FN median m6 p 0.29 (uncertain, not confident-wrong). m6 already captures this via competitor features over the ScaNN top-50.
  Pointer: output_s2 written but not worth uploading.
- output_s2 + testpred_s2 deleted (stage 2 not adopted); code + models/matcher_s2 kept for the record.
- 2026-09-27 01:48 **BGE model arrived** from Blackwell: `models/ce_bge_m3_fp16` (bge-reranker-v2-m3, 2.81M pairs = kit train pairs folds 1–4 + synthetic French v1 (no test data), 1 epoch, bs 128, lr 2e-5, bf16, 21,980 steps, 105 min). Kit dev AUC 0.99843 (step 2k) → 0.99943 (final) — kit dev set, not comparable to our filtered-val AUCs. `run_bge.sh` now points at this folder; auto-starts when the copy stops growing.
- **BGE vs Laya v2 (held-out filtered val 917k; log 2026-09-27_bge_vs_laya.log)**: AUC e5 0.99489 / Laya2 0.99616 / **BGE 0.99652** (India 0.99705, US 0.99606); S2 name-only 0.9071 / 0.9108 / **0.9186**; m6-uncertain band 0.6868 / 0.7205 / **0.7524**; alone F0.5 Laya2 0.9823 / **BGE 0.9827**. Errors at logit 0: Laya 21,844, BGE 21,098, both 17,005 → BGE fixes 4,839 of Laya's; logit corr 0.974. → best single CE so far and partly complementary.
- **m8 early val (m6 + BGE logit, 51 feats; tag m8val, no test write)**: F0.5 **0.98993** @0.75 (India 0.99079, US 0.98936) vs m6 0.98962 → **+0.00031**, largest single gain since Laya v1 (m5). Test scoring running (run_bge.sh).
- **m8** (m6 + BGE): val **0.98993** @0.75 (India 0.99079, US 0.98936). Ownership fix removed 16,289 → `submission/output_m8o` **validator PASS** (1,631,922 S1 with a match). Test 7.36M BGE scoring 106 min.
- m8o vs m6o: France 817,091 pairs (−49,151 / +23,255, net −25.9k = stricter); India/US ±~4k. France dropped: legal-form adds (sci, sas), accent variants (àmicale, çlub, ècole), développement, groupe; added: fils, et, france, club, associés. Direction agrees with the m7o lesson (France stricter). m8val temp artefacts deleted.
- **LEADERBOARD m8o = 0.980** (user) — big drop vs m6o 0.98314 despite val +0.0003. Cause: 47% of the 49k France pairs m8o dropped have names equal after accent folding (injected French accent noise "Çlub"/"Àmicale"); BGE (XLM-R tokens, India/US training) reads accents as different businesses → France recall collapse. France = 15% of test S1.
- Fixes built (no training): `submission/output_m8h` = France from m6o + India/US from m8o (validator PASS). `scripts/rescore_folded.py`: BGE test re-scored with Latin accent folding on the 956,922 accented pairs (38,140 flip to >0, 2,656 to ≤0) → `submission/output_m8fo` (saved m8 + folded BGE; validator PASS).
- User: test exact (brute-force) blocking with the newer encoders → `scripts/run_exact.sh` (exact top-50 → m2 filter → reuse CE scores, score only new pairs → saved m6/m8 via `scripts/predict_saved.py` → m6xo / m8xo).
- **Exact (brute-force) blocking** (`run_exact.sh`, user request): exact top-50 test 16.6 min (ScaNN 0.2–0.6 ms/S1); m2 filter → test 7,407,025 pairs (4.28/S1; ScaNN 4.25), val 919,364. CE reuse: only 3,273 val / 61,179 test new pairs scored.
  Saved models on exact blocking: **m6 val 0.98987** (India 0.99097, US 0.98913; ScaNN 0.98962) | **m8 val 0.99018** (India 0.99127, US 0.98945; ScaNN 0.98993) → exact blocking +0.00025 on val.
  vs m6o on test: m6xo France −586/+7,208, India −268/+4,189, US −246/+910 (exact mostly ADDS matches ScaNN missed).
  Submissions (validator PASS): `output_m6xo` (m6 on exact), `output_m8xh` (exact; France from m6xo, India/US from m8xo). Deleted: output_m8o (LB 0.980), output_m8xo (BGE accent issue in France), non-ownership m6x/m8x.
- 2026-09-27: **User has only 3–4 leaderboard uploads left.** Plan: #1 output_m8xh; #2 = m8xfo (exact + accent-folded BGE everywhere) if #1 > 0.98314, else output_m6xo; #3 final-package candidate; keep 1 reserve. New France changes must pass the m6o France diff check first. Launched `scripts/run_m8xf.sh` (folded BGE on exact val+test → output_m8xfo).
- **Teammate experiments B & C on top of m8 (val, `scripts/specialists.py`, log 2026-09-27_specialists.log)**:
  C empty-address specialist (LightGBM on 182k name-only train pairs, m8's 51 feats + same_name_s1): **0.99011 vs 0.98993 = +0.00018** @ specialist th 0.75; control (m8 with its own name-only threshold) +0.00000 → gain is from the model, not the threshold.
  B singleton model (per-S1 classifier on OOF m8 probs, 291k train S1, 2.8% singletons): best +0.00001 @ cut 0.7 → no gain; the S1 it empties are almost all already predicted empty by m8.
- **output_m8xhc** (`scripts/build_m8xhc.py`; C model saved models/matcher_m8c/spec_c.txt): exact blocking; India/US = m8 + empty-address specialist C; France = m6xo. Exact val: m8 0.99018 → **m8+C 0.99036** (India 0.99147, US 0.98962). Validator PASS. vs m6o: France −586/+7,208 (identical to m6xo), India −4,685/+9,253, US −3,739/+4,536. → **Upload #1 candidate** (replaces m8xh).
- **output_m8xfo** (exact + accent-folded BGE everywhere; validator PASS): val 0.99018 (folding neutral on India/US). France vs m6xo: −27,985 / +25,338. Drops: legal-form adds (sci, sas, sasu, eurl), groupe, developpement; 21% accent-equal (was 47% unfolded). Adds: fils, et, france, club, associes — the same word pattern as m7o's harmful additions → BGE still reshapes France in an unverified way. Low upload priority.
- **LEADERBOARD m8xhc = 0.984** (user) — best so far (+0.0009 vs m6o). Exact blocking + BGE + C specialist (India/US) confirmed; France = m6 on exact.
- Implied France F0.5 ≈ 0.947 (India/US val ≈ 0.9904, 85% of test). France predicts 3.275 matches/S1 vs India/US 3.38 (truth 3.46) and has ~6× denser borderline band → recall-limited.
- **France rule probes (label-checked on India/US val; `scripts/france_rule.py`, log 2026-09-28_france_rules.log)**:
  (1) identical core name (accents + legal forms stripped), any address, m6-rejected: val precision 2–9% (decoys with different house numbers) → reject rule.
  (2) same core name + same house no. + same street + no decoy word: val precision 0.9973 overall (m6 already accepts 99.5%); the few m6 REJECTS are only 2.3% precise (p≥0.3: 47%, n=17) → m6 stays calibrated even on rule pairs. France has 3,967 such m6-rejected pairs (p median 0.45) → accepting them ≈ 50% precision → not worth an upload. Decoy-word pairs (groupe/holding…) val precision 0.858.
  Theory: with calibrated probabilities the F0.5-optimal threshold ≈ F*/1.25 ≈ 0.76 for France, ≈ current 0.725 → France threshold probing expected ≈ 0.
- **2026-09-28 Organisers APPROVED pseudo-labelling (self-training) on test.** Must be documented + shipped in the final package.
  Launched `scripts/run_pseudo.sh`: `build_pseudo_fr.py` (France test pairs of m6 on exact: 250k pos p≥0.98 & ownership-consistent, all 176k neg p<0.02; middle band unused) → Laya v2 fine-tune (`train_crossencoder.py --pseudo pseudo_fr`, + 60k new India/US S1 × top-10 real pairs, lr 1e-5) = ce_laya_v2p → val check → re-score France test pairs → m6 with laya2p (France) → output_m6xpo.
- 2026-09-27 12:44 Claude Doc 'Entity Resolution Pipeline & the France Gap' created (https://claude.ai/code/artifact/7966b3fb-38cf-45d1-b523-f53ba28f4639): pipeline diagram, score table m1→m8xhc, France problem + upload lessons, self-training plan/status, open decisions.
- **France self-training result.** Laya v2p (pseudo-labels) val guard: CE AUC 0.99618 → 0.99633; m6 with v2p on exact val 0.98990 (v2: 0.98987) → India/US OK. France (m6 + v2p, `output_m6xpo`): ownership removed 31,505 claims (m6x: 5,174); vs m6xo France −5,070 / +21,590 (3.275 → 3.338 per S1). Drops look right (different core names: Club/Lycée, Amis/Collectif). Adds: 24% accent-equal, 44% name-only S2, but many carry the m7o filler pattern (fils, et, france, associés, groupe).
  Built (validator PASS, India/US identical to m8xhc): `output_m8xhcp` = full v2p France (866,129 France pairs); `output_m8xhcs` = conservative: m6xo France − v2p drops + only the 11,432 v2p adds whose names differ by accents/case/legal forms alone (855,971 France pairs) — `scripts/build_france_variants.py`.
- 2026-09-27 User: organisers confirm allowed for self-training on test: unsupervised stats, TF-IDF/token frequencies, blocking indexes, pseudo-labelling. (LB-feedback rules like the m8xhcs 'safe adds' filter are not on that list → document/flag.)
- **France variants (label-free diag vs 0.984 France = m6xo, `scripts/analysis/_france_variant_diag.py`)** — all India/US = m8xhc, validator PASS:
  D `output_m8xhcd` (agreement m6xo ∩ m6xpo): 3.255/S1, +0 / −5,070 (other 2,775, filler 1,886, legal/accent 409).
  `output_m8xhcs` (D + legal/accent-only v2p adds; LB-derived filter): 3.299/S1, +11,432 (all legal/accent) / −5,070.
  `output_m8xhcp` (full v2p): 3.338/S1, +21,590 (legal/accent 11,432, filler 6,017, other 4,141) / −5,070.
  B `output_m8xhcb` (France-aware LightGBM with pseudo rows; val 0.98986 neutral): ownership removed 47,683; 3.345/S1, +24,169 (filler 6,657, other 6,087, legal/accent 11,425) / −5,889.
  By the leaderboard lessons: hcs > hcd > hcp > hcb. Compliant alternative to hcs pending: agreement of two self-trained models (v2p + BGE-p, run A).
- 2026-09-27 **qwen_kit/** built for the user's other GPU: LoRA fine-tune of Qwen/Qwen3-Reranker-4B (Apache-2.0) as pair classifier (score = logit yes − logit no), `train_qwen_rr.py` + `score.py` + README + requirements. Data (no IDs): train 150k (75k real India/US folds 1–4 + 75k France pseudo-labels), dev 20k India/US fold-0, score set 341,946 uncertain-band pairs (111,568 val + 230,378 France; key map + val labels stay in cache/qwen_score_keys.parquet). Smoke-tested here with Qwen3-Reranker-0.6B fp16 (peft 0.21.0 installed via uv --no-deps; torch untouched). Plan: blender (m6 logit + qwen) fit on labelled val band → decide uncertain France pairs.
- 2026-09-27 15:17 user: Qwen3-Reranker-4B fine-tune running on the other system (~90 min). Prepared scripts/blend_qwen.py (LR on m6 logit + qwen, fit on labelled val band; OOF val estimate; France band → output_m8xhcQ).
- **BGE-p (BGE self-trained on France pseudo-labels, V100 fp16, 85 min)**: India/US guard: CE AUC 0.99627 vs BGE 0.99654; m8 with Laya v2p + BGE-p exact val 0.99006 vs 0.99018 → slightly worse on India/US (don't use there).
  France candidates (validator PASS): `output_m8xhcA` full stack BGE-p + C: 3.319/S1, +19,507 (legal/accent 6,657, filler 6,563, other 6,287) / −8,087 (incl. 1,975 legal/accent). `output_m8xhcAG` two-model agreement: 3.306/S1, +13,221 (legal/accent 5,942, filler 4,558, other 2,721) / −5,070.
  → Agreement of the two self-trained models does NOT filter the filler pattern; hcs (legal/accent-only adds) remains best by the leaderboard lessons. Waiting for Qwen (hcQ) before choosing upload #1.
- 2026-09-27 16:28 user: 7 uploads remain; decided to wait for Qwen comparison before uploading. Built `output_m8hc` = ScaNN twin of the 0.984 upload (India/US m8 + C on ScaNN, France m6o; validator PASS) for the scalability question. Removed superseded output_m8h.
- **Qwen3-Reranker-4B (LoRA, other GPU, bf16, 138 min train + 54 min scoring)**: dev AUC 0.897 zero-shot → 0.9941. On the India/US uncertain band (111,568): AUC m6 0.9003 vs Qwen 0.8127; blend OOF 0.9010; val F0.5 0.98962 → 0.98968 (noise). `output_m8xhcQ` (blend) ≈ unchanged France (849,344; +3,681/−3,946). Validator PASS.
  France band by pair type: Qwen says match for 81.8% of legal/accent-only pairs (m6 38.8%), 55.3% filler (m6 44.1%), 55.9% other (m6 37.0%) → independent LLM evidence that legal/accent-only pairs are matches, i.e. corroborates hcs's rule. Kit bug fixed (bf16 → numpy needed .float()).
  Qwen independently backs 10,618 of hcs's 11,432 France adds (93%) (Qwen score > 0; 41,865 legal/accent-only m6-rejected band pairs Qwen calls match before ownership).
- **Qwen3-Reranker-4B (LoRA, other GPU, bf16, 138 min train + 54 min scoring)**: dev AUC 0.897 zero-shot → 0.9941. Kit bug (bf16 → numpy) fixed on their side and here (.float()).
  Uncertain val band (111,568): AUC m6 0.9003 | Qwen 0.8127 | LR blend (OOF) 0.9010; val F0.5 0.98962 → 0.98968 (+0.00006, noise).
  France (`output_m8xhcQ`, validator PASS): 3.274/S1, +3,681 (legal/accent 2,168, filler 497, other 1,016) / −3,946 (filler 1,655, other 1,699, legal/accent 592). Qwen says match on 66% of the France band vs m6 39% (corr 0.605); blender (fit on India/US) down-weights it → small, mostly lesson-consistent changes.
- **LEADERBOARD output_m8xhcs ≈ 0.9839** (user: "0.0005 diff", direction vs m8xhc 0.984 to be confirmed) → France self-training tweak within noise.
- `output_m8xhcQ0` (Qwen decides the France uncertain band at its own yes/no boundary, score > 0; validator PASS): 3.342/S1, vs 0.984 France +29,229 (legal/accent 12,563, filler 8,003, other 8,663) / −11,807 (filler 5,003, other 4,885, legal/accent 1,919). Big, clear change → informative upload.
- User asked about Qwen embeddings to "bridge" encoders: proposed Qwen3-Embedding-0.6B France re-retrieval probe (count new confident French candidates e5 missed); awaiting go.
- LEADERBOARD output_m8xhcQ0 = 0.98355 (worse). Pattern: every France change that ADDS pairs hurt (m7o, hcs, hcQ0) → France likely precision-limited. Best stays m8xhc (~0.9844).
- LEADERBOARD output_m8xhcQ0 = 0.98355 (worse than m8xhc ~0.9844). Pattern: every France change that ADDS pairs hurt (m7o −0.0011, hcs −0.0005, hcQ0 −0.0009) → France likely precision-limited (decoys our India/US-trained models miss). User rejected the stricter-threshold probe; instead asked to MAP the French generator with a separate agent, build labelled French data from the map, then train.
- Launched mapping agent → deliverables logs/france_map.md, scripts/france_synth_v4.py (cache/synth_fr_v4.parquet). User: train on the FULL produced data (no subsampling): all French S1 anchors, all pseudo-labels, full real India/US folds 1–4.
- User: French training data must come from the TEST SET ONLY (anchors = French test S1; all inserted/swapped words, streets, places, formats harvested from French test text; no hand lexicons). India/US labels only to measure op→match relations. France-only model (Laya v2 init) trained on test-derived data: synth v4 (all anchors) + all pseudo-labels; used only to score France, so India/US unaffected. Constraint sent to the mapping agent.
- 2026-09-27 cleanup #5 (user request, ~5 GB freed, 53 GB free): removed submissions m6o, m8fo, m8xfo, m8xh, m8xhcA/AG/b/d/p/Q/Q0/T85; models ce_bge_p, ce_laya_v2p, matcher_s2, matcher_m5; caches of BGE-fold/BGE-p/Laya-v2p scores, France-B preds, rule-probe cands, superseded test preds; qwen adapter. Kept: output_m8xhc (best ~0.9844), output_m8xhcs (0.9839), output_m8hc (ScaNN twin, not uploaded), output_m6xo/m6xpo (France references used by scripts); all models of the m8xhc pipeline; pseudo_fr.parquet; Qwen scores; all caches the mapping agent reads.
- 2026-09-27 19:25 TF-IDF test launched (src/er/tfidf.py, scripts/tfidf_test.py): 5 per-country IDF features (train IDF for India/US, test IDF for France); m6/m8 retrained base vs +tfidf on exact val; France variant output_m8xhcTF.
- **Qwen3-Embedding-8B baseline (zero-shot, sentence-transformers fp16 V100; `scripts/analysis/qwen_embed_test.py`)** on 20k labelled val pairs (same pairs for all): AUC Qwen-emb cosine 0.8522 | our fine-tuned e5-small bi-encoder 0.8911 | e5 CE 0.9951 | Laya v2 0.9959 | BGE 0.9962. m6-uncertain (2,420): 0.547 vs bienc 0.556, CEs 0.83–0.86. S2 no-address: 0.620 vs CEs 0.90–0.91. Best pairwise F0.5: Qwen-emb 0.9254 vs Laya v2 0.9893. → embeddings (even 8B) are far weaker than pair models for this task; not pursued. Weights deleted.
- 2026-09-27 19:36 queued: scripts/ce_ablation.py (BGE-only etc.) after TF-IDF, then scripts/gbm_blend.py (LightGBM vs XGBoost vs average, m6/m8 stacks).
- **TF-IDF test**: m6-type base 0.98988 → +tfidf 0.99000 (+0.00012); m8-type 0.99020 → 0.99023 (+0.00003); 5 TF-IDF feats carry ~0.05–0.12% gain each → noise-level. France variant `output_m8xhcTF` (m6+tfidf, test IDF; validator PASS): 3.264/S1, +3,699 (filler 1,366, other 1,793, legal/accent 540) / −6,566 (legal/accent 2,104, filler 2,075, other 2,387) → mixed (drops real accent/legal matches); not upload-worthy.
- 2026-09-27 19:41 removed stray model_metadata.pkl (252 B, foreign baseline metadata: threshold 0.79 + 6 fuzzy features; not from our pipeline) on user request.
- **Mapping agent result** (`logs/france_map.md`, `scripts/france_synth_v4.py`, `cache/synth_fr_v4.parquet` 168k from 60k anchors; `--n all` = 727k): French decoys = name edit + house-number shift δ∈{1,2,3,4,5,7,9,11,13,21} (model already rejects); SNC decoy-only; **same-address category-word swap = decoy (est P≈0.1, model accepts ~65%)**; **same-address filler swap/add = match (est P≈0.95–1.0, model accepts 72%/44%)**; name-only mostly distractors (P≈0.3) — explains hcs (adds were mostly name-only).
- Verified classification (`scripts/france_map_variant.py`, reproduces 0.984 France exactly = 849,609): catswap@same 29,982 (accepted 19,832), M_swap@same 35,397 (acc 25,804), M_add@same 9,770 (acc 4,355), name-only 68,280 (acc 14,165).
  Built (validator PASS): `output_m8xhcMAPdrop` (0.984 France − 19,799 catswaps = 829,810) and `output_m8xhcMAP` (− catswaps + 14,877 rejected filler@same, ownership-safe = 844,687).
- 2026-09-27 19:47 user shared Kaggle notebook swarnabhahalder/amazon-ml-challenge-2026 (same competition). Pulled source via public Kaggle API → refs/kaggle_amazon_ml_2026_nb.py (code only, no outputs). Stage: phases 0–3B (ingestion, metric unit test, 3 normalisations, TF-IDF/sorted-neighbourhood/postal blocking, MiniLM+FAISS blocking); planned LightGBM + ms-marco MiniLM CE rerank. No results beyond ours; useful only as a compliance/documentation checklist (license audit, fair-play statement).
- 2026-09-27 community scan (user request). Reddit not reachable by our tools (site blocks automated access) → used public GitHub/Kaggle. Key: harshgitty58/Amazon_ML_Challenge README states "top of the leaderboard was 0.9906 at handoff (2026-09-26)"; their own 0.984 val / 0.969 LB (TF-IDF blocking, 2-stage LightGBM, sibling expansion — "copies of one business share typos, spacing and unit numbers", per-S1 expected-F0.5 decoding). Other public repos: AvinashMalladi (">0.98", LightGBM 14 feats), Akash-bardia (0.9761), Noyonika16 (val 0.7337). No public write-up found of how ≥0.988 was reached.
- **Shared raw-noise sibling check** (`scripts/analysis/shared_noise_check.py`; public claim "copies share typos/spacing/unit numbers"): India/US val — 0 shared noise tokens P(match) 0.72 vs 2+ shared 0.95 (m6 already tracks it: accepts 0.70/0.95); uncertain band 0 shared 0.19 (m6 acc 0.08) vs 2+ shared 0.53 (acc 0.40); raw address identical to a sibling 0.64 (n=1,964). France: catswap shares noise with siblings as often as filler variants (0.524 vs 0.504–0.518; raw-addr-equal 0.25 vs 0.22–0.23) → French decoys copy a real record's noise then swap a word; signal cannot separate them. Possible small India/US matcher feature only.
- **CE ablation (India/US matcher, exact val)**: all four 0.99020 | BGE only 0.98982 (−0.00038) | **BGE + Laya v2 0.99017 (−0.00003)** | BGE + e5 0.98984 | no BGE (m6) 0.98988. → BGE + Laya v2 matches the full stack: e5 CE and Laya v1 are redundant for India/US (lean option).
- **LightGBM vs XGBoost**: m6 0.98988 / 0.98984 / avg 0.98991; m8 0.99020 / 0.99019 / avg 0.99020; prediction corr 0.9996 → no gain from XGBoost or blending.
- **LEADERBOARD output_m8xhcMAPdrop = 0.986** (user) — +0.002 vs 0.984: removing 19,799 same-address category swaps CONFIRMS the France map. Next: upload output_m8xhcMAP; train France model on map-labelled test data (full).
- **LEADERBOARD output_m8xhcMAP = 0.987** (user) — filler@same adds also confirmed (+0.001 over MAPdrop 0.986). Best = output_m8xhcMAP. Uploads left: 3.
- MAP2 sizing (remaining map classes, m6x France accept counts): add-candidates rejected — clean+street-diff 9,796 (map .9–1), clean+num-noise 6,486 (map .8); drop-candidates accepted — other-name+numdrop 1,104, multi-edit@same 1,045, other-name@same 944 (map .1–.2); other@same 29,434 rejected/133,511 accepted (mixed, uncategorised).
- **laya_fr_kit/** built for the Blackwell (user: faster run): model/ = Laya v2 (1.2 GB), data/ train 1,436,070 map-labelled test-derived French pairs + dev 20k + score 1,169,377 French test pairs (keys in cache/layafr_score_keys.parquet), train_score.py (bf16, 1 epoch, bs 128, lr 1e-5; trains then scores), README, requirements. Smoke-tested here (fp16, 3k pairs): dev AUC vs map labels 0.943 → 0.959 after 93 steps; scores written, no NaN.
- 2026-09-27 21:42 output_m8xhcMAP2 built (scripts/france_map2.py; MAP + 11,297 clean-name street/number-noise adds − 2,982 other-name/multi-edit drops; validator PASS). Deadline midnight: starting final package neural_ninjas_submission/.
- 2026-09-27 21:45 package neural_ninjas_submission/: output/ = output_m8xhcMAP (0.987, placeholder until final choice), code/business_entity_resolution/{src (er + 27 scripts + validator), README.md (ordered steps), requirements.txt (pinned), docs/ (experiment_log.md, france_map.md)}, Documentation_template.md filled.
- **LEADERBOARD output_m8xhcMAP2 = 0.9868** (user) — slightly below MAP (0.987): the clean-name street/number-noise adds and other-name drops do not help. **Final candidate stays output_m8xhcMAP (0.987)** unless Laya-FR beats it.
- 2026-09-27 22:01 user: official organiser approval covers self-training (pasted stricter plan was not the rule) → keep output_m8xhcMAP (0.987) as final; building zip.
- 2026-09-27 22:06 built output_m8xhcMAPno (MAP − 3,766 France name-only accepted with 0.725≤p<0.9; validator PASS). Laya-FR running on Blackwell (~10 min); scripts/build_layafr.py ready (A: m6 with Laya-FR slot, B: Laya-FR logit>0; class-wise diff vs MAP).
- 2026-09-27 22:34 Laya-FR back (Blackwell, 26 min; dev AUC vs map labels 0.946 → 1.0). Variants: LFa (m6 slot) +20.1k/−9.9k (re-adds 1.9k catswaps ✗); LFb (logit>0) +24.3k/−23.5k (weak adds logit 1–1.5 are number/street/brand decoys); LFc (logit>2) −39k/+12k (too strict). **LFd = MAP − name-only with LFR<0 + adds LFR>3 (ownership-safe)**.
- **LEADERBOARD output_m8xhcLFd = 0.9873** (user; +0.0003 over MAP 0.987) → final = LFd. Rebuilding package.
- 2026-09-27 22:40 FINAL zip rebuilt: Neural_Ninjas_submission.zip with output = output_m8xhcLFd (LB 0.9873); docs + README updated with Laya-FR step; validated after unzip.
- 2026-09-28 user: NO uploads left; final = zip. Focus: scalability with least score loss (val-only decisions now). Started stack-trim ablations (ensemble_matcher, no write): drop Laya v1 / drop Laya v2 / BGE only.
- 2026-09-28 plan approved (user): two documented modes — `scalable` (DEFAULT: high-recall ScaNN + lean scoring) and `exact` (small data; reproduces shipped 0.9873 LFd output, kept byte-identical). Plan: ~/.claude/plans/precious-mixing-glade.md.
- Trim ablations (ScaNN val, m8-type matcher retrained): all four 0.98993 | no Laya v1 (e5+Laya v2+BGE) **0.98995** | no Laya v2 0.98987 | BGE only 0.98959. With the earlier exact-val ablation (BGE+Laya v2 0.99017 vs 0.99020) → lean India/US stack = features + Laya v2 + BGE (e5 CE and Laya v1 dropped). Temp models removed.
- 2026-09-28 code for two modes: `gen_candidates_scann.py` (--search_frac/--reorder/--tag, defaults unchanged); `ensemble_matcher.matrices(drop=...)` + `--drop_base` (ce_tag None → e5 CE never loaded), saved in ensemble.json and honoured by `predict_saved.py`; `build_france_variants.write(base=)` + import-safe; `france_map_variant.py` / `build_final_lfd.py` (now in scripts/) take --pred/--base/--out/--cls; `scripts/compare_outputs.py` (set-level per-country diff).
  Regression: saved m8 exact val 0.99018 (unchanged); exact-mode France path reproduces output_m8xhcLFd 100% of pairs (matching + candidates; only 330 list orders differed → write() now sorted/deterministic).
- **m8lean** (features + Laya v2 + BGE; no e5 CE, no Laya v1; ScaNN val): 0.98995 @0.725 (India 0.99084, US 0.98936) vs m8 0.98993 → lean India/US stack confirmed.
- 2026-09-28 22:16 **0.987 output isolated (user request)**: `FINAL_0987_LOCKED/` (read-only) = output/matching_results.tsv (md5 da2e6126b3f2…) + output/candidate_pairs.tsv (4ef85129ecee…) + the 2026-09-27 zip + MD5SUMS + README; `submission/output_m8xhcLFd` also made read-only. Rule: scalable-mode files use new names (output_iu_*, output_scalable*); any new zip's output/ must pass `md5sum -c FINAL_0987_LOCKED/MD5SUMS`.
- 2026-09-28 22:48 ScaNN sweep: 30% → coverage of 0.987 matches France 99.77 / India 99.95 / US 99.98 (15%: 99.20/99.85/99.96); ms/query 0.31/0.89/0.70. France < 99.9 target → testing 50%; started train ScaNN at 50% (tag scalable) in parallel.
- 2026-09-28 23:37 user: skip generating the scalable output (too slow); only measure scalability. Stopped train-split ScaNN (tag scalable). Zip ships locked 0.987 output; scalable mode documented with measured coverage/timings/val.
- **ScaNN operating point (test, label-free; `scripts/scann_sweep.py`)** — share of the 0.987 matched pairs inside the top-30 (FR / IN / US):
  15% 99.20 / 99.85 / 99.96 (0.15–0.6 ms/query) | 30% 99.77 / 99.95 / 99.98 (0.31 / 0.89 / 0.70 ms) | **50% 99.93 / 99.98 / 99.99** (0.45 / 1.85* / 1.46* ms; *CPU shared with another job). Exact-filtered-candidate coverage at 50%: 99.78 / 99.78 / 99.96.
  → scalable-mode default search_frac 0.50 (≥ 99.9% of the 0.987 matches in every country); 0.15 / 0.30 = speed knobs.
- Measured scoring rates (V100 fp16): e5 ~9k pairs/s, Laya ~1.6–2.1k, BGE ~1.2–1.35k. Exact mode 4 CEs × 7.41M pairs ≈ 4.6 GPU-h; scalable/lean (Laya v2 all pairs, BGE India/US only, e5 + Laya v1 France only) ≈ 3.1 GPU-h (−34%).
- 2026-09-28 23:41 **FINAL zip rebuilt** `Neural_Ninjas_submission.zip` (69 files, 93.0 MB): output/ = locked 0.987 (md5 OK before and after unzip; validator PASS); code adds the two modes — `src/scripts/run_pipeline.sh` (scalable default: ScaNN 50% + lean m8lean; `--mode exact` reproduces output/), build_india_us.py, france_m6.py, subset_country.py, scann_sweep.py, compare_outputs.py, parameterised France scripts; README "Two modes" + step 8b (m8lean); docs/scalability.md; Documentation_template.md §3.1 "Scalable and exact modes" + blocking text. Previous zip preserved in FINAL_0987_LOCKED/.
- 2026-09-28 23:44 user: build the FULL scalable system. Added persistent per-country ScaNN indexes (gen_candidates_scann.py --index_dir/--build_only; reload verified identical), scripts/scann_query.py (serve new S1 records from saved indexes). Launched run_pipeline.sh --mode scalable (ScaNN 50%, index_dir models/scann_index) → submission/output_scalable.
- 2026-09-29 01:07 **scalable run crashed: shared disk 100% full** (ScaNN index serialize I/O error at test India). Persisted indexes store the float re-rank vectors (dataset.npy ≈ 1.5 KB/record: train_US 11 GB, train_India 6.9 GB) + re-created embedding caches (18.6 GB). Freed ~32 GB (train embeddings, train indexes, partial test_India index, cands_test_scann030) → 31 GB free. Verified: cands_train_scalable intact (110,341,050 rows), FINAL_0987_LOCKED md5 OK, final zip OK.
  Fix: runner persists test (serving) indexes only; RESUME=1 reuses existing candidate files. Resuming.
- 2026-09-29 01:11 fix: ScaNN index saved/loaded via absolute paths (relative path was doubled on reload); test_France index rebuilt; resumed (RESUME=1).
- 2026-09-29 01:49 **scalable indexes done**: persisted test indexes models/scann_index/test_{France 2.4G, India 7.9G, US 6.4G} (build 44 / 98 / 81 s; search 0.50 / 1.40 / 1.35 ms/query at 50%); cands_test_scalable 86.6M pairs. Train candidates reused (train ScaNN 50%: India 1.38, US 1.88 ms/query).
- **Serving demo** (`scripts/scann_query.py`, 1,000 real test S1 per country loaded from the saved indexes, no rebuild, while the filter ran): 99.9987% of 150,000 served top-50 pairs identical to the batch run; 9 / 22 / 33 ms per record in small single-thread batches (CPU shared).
- 2026-09-29 01:56 organiser request (methodology submission): docs/submission_methodology/Neural_Ninjas_Methodology.{md,pdf (5 pages, DejaVu embedded),docx,html} + build_docs.py (converters in an isolated /tmp env: markdown, xhtml2pdf, python-docx, pymupdf; project venv got only 'markdown').
- **Scalable run (2026-09-29)**: filter test 7,397,955 pairs (4.27/S1; exact 7,407,025), val 918,876; new pairs scored: 53 val + 799 test (rest reused, ~1 min). **Scalable validation (ScaNN 50% + lean m8lean + C): F0.5 0.99033 (India 0.99143, US 0.98961)** vs exact full stack 0.99036 → −0.00003; m8lean alone 0.99016 vs exact m8 0.99018.
- 2026-09-29 02:34 **FULL SCALABLE SYSTEM DONE** (`run_pipeline.sh --mode scalable`, RESUME=1): submission/output_scalable validator PASS; val 0.99033 (exact 0.99036); matched pairs identical to the locked 0.987 output France 99.991% (−76/+85), India 99.957% (−1,193/+1,909), US 99.956% (−979/+2,146); candidates 7,397,955 (4.27/S1), 99.76–99.93% identical. Wall-clock: test ScaNN 38 min, filter 36 min, scoring 1 min (852 new pairs), India/US 6 min, France 2 min. Docs (scalability.md, Documentation_template §3.1, methodology PDF/DOCX) updated with the measured results; package code synced (index persistence, scann_query.py, resumable runner).
