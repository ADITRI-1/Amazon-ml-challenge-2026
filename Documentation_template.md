# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Neural Ninjas  
**Team Members:** Aditri Jain, Snehil Modi, Yatharth Bansal, Nikhil Agrawal  
**Submission Date:** 2026-09-28

---

## 1. Executive Summary
We retrieve candidates with a fine-tuned multilingual bi-encoder (two modes: scalable ScaNN retrieval by default, exact
top-50 per country for small data — the shipped output used exact), prune them to ~4.3 per Source 1 record with a cheap LightGBM filter, and decide matches with a LightGBM
matcher that stacks 48 string/address/competitor features with four out-of-fold cross-encoders (e5-small, Laya v1/v2,
BGE-reranker-v2-m3), plus a specialist model for records without an address and a one-owner-per-record constraint.
France, which has no labels, is handled by a label-free map of how French variants and decoys are generated
(unsupervised test statistics, organiser-approved), which fixed the one error class the India/US-trained model got
wrong; a France cross-encoder self-trained on map-labelled test pairs then refines it: validation (India/US) macro F0.5 **0.9904**, public leaderboard **0.9873**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Precision-weighted, per-entity metric.** Macro F0.5 over Source 1; a singleton scores 1 only with an empty
  prediction. Every design choice therefore prefers a missed match over a wrong merge.
- **Structure (training ground truth).** 3.46 matches per S1 in both countries (S2 ≤ 5, S3 ≤ 6), 5.6% singletons,
  ~26% of S2/S3 records are unowned distractors, and **every S2/S3 record belongs to at most one S1** (1:N).
- **Noise operations.** Case/format/punctuation, one-letter typos and doubled letters, word re-ordering, legal-form
  drop/add/re-spelling (Pvt Ltd, LLC, SARL…), abbreviations, transliteration and native scripts (Devanagari, Tamil…),
  `.com` / `@handle` forms, "X dba Y", address re-ordering, street-type abbreviations, missing components.
- **Decoy operations.** Near-identical names with a *shifted house number*, doubled letters, added decoy words
  ("group", "holdings"), and name-only records (no address) that are mostly distractors.
- **Where the errors are.** 77% of India/US matcher errors involve an S2/S3 record with no address; 55% of the
  remaining loss is S1 entities with a mix of addressed and name-only matches; different scripts are solved (0.6%).
- **France (test only, no labels).** Same generator, French vocabulary: one-character accent injection (Çlub,
  Àmicale), French legal forms (SARL, SAS, SASU, EURL, SCI, SA; SNC appears only in decoys), département/région
  address tails, and two French-specific edits at the *same address*: a **category-word swap** (Club → École) is a
  different business, a **filler word** (& Fils, Services, Cie, Groupe, Développement…) is the same business.

### 2.2 Solution Strategy
**Approach Type:** Blocking + stacked classifier (Hybrid: bi-encoder retrieval, gradient boosting over string features
and cross-encoder scores, structural post-processing).  
**Core Innovation:** (1) out-of-fold stacking of four cross-encoders from different model families into a LightGBM
matcher with *competitor* features (does another S1 claim this record better?); (2) a label-free, test-derived map of
the French generator that corrects the matcher's French decisions, verified step by step on the leaderboard.

**Data discipline.** S1 records are split into 10 folds by a hash of their ID (all matches follow their S1): fold 0 =
validation only; bi-encoder trained on folds 1–9; cross-encoders on folds 1–4; LightGBM matchers on folds 5–9, so the
matcher only ever sees out-of-sample cross-encoder scores.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** a bi-encoder (`intfloat/multilingual-e5-small`, MIT, 118M) fine-tuned contrastively (InfoNCE
  with hard negatives) on training matches, over normalised text `name | address` (unidecode, legal words stripped,
  consonant skeletons for transliteration). Retrieval is **per country** in one of two modes (section 3.1): **scalable (default)** =
  ScaNN (2,000 partitions, 50% searched, 4-bit codes + exact re-rank, 0.3–1.9 ms per S1 on CPU); **exact** = brute-force
  top-50 on GPU (16.6 min for the whole test set), used for the shipped output. A cheap **LightGBM filter** (38 string features + 8
  competitor features, no cross-encoder) then keeps top-30 candidates with p ≥ 0.03.
- **Candidate pairs generated:** 7,407,025 on test = **4.28 per S1** (1,732,544 S1; 1,686,452 with ≥ 1 candidate).
- **How you ensured true matches were not lost:** validation retrieval recall with the fine-tuned bi-encoder 0.990 at
  top-10 (0.941 before fine-tuning); after the filter, ~99% of true matches remain. Error attribution on validation:
  only 0.35% of true pairs are missed by retrieval and 0.64% by the filter; the matcher decides the rest. The filter
  threshold was chosen on a candidate-budget curve (validation F0.5 −0.0001 vs top-30).


### 3.1 Scalable and exact modes
The same trained models run in two modes (`bash code/business_entity_resolution/src/scripts/run_pipeline.sh [--mode exact]`;
full measurements in `code/business_entity_resolution/docs/scalability.md`).

| | exact (small data; shipped output) | scalable (default) |
|---|---|---|
| Retrieval | brute force on GPU, cost ∝ queries × pool | ScaNN on CPU, 4-bit index (~0.2 KB/record), recall/speed knob |
| Share of shipped matches retrieved (France / India / US) | 100 / 100 / 100% | 99.93 / 99.98 / 99.99% |
| Candidates per S1 | 4.28 | ~4.3 |
| Cross-encoder passes | 4 on every pair | India/US: Laya v2 + BGE; France: e5, Laya v1, Laya v2 |
| Test scoring GPU time (V100) | ~4.6 h | ~3.1 h |
| India/US validation F0.5 (full system) | 0.99036 | **0.99033** (measured end to end) |
| Matched pairs identical to the submitted output (France / India / US) | 100% | **99.991 / 99.957 / 99.956%** |
| Retrieval indexes | none | persistent per-country ScaNN indexes (France 2.4 GB, India 7.9 GB, US 6.4 GB), reloaded to serve new records |

**Why the shipped output uses exact mode:** this test set (1.73M × 9.97M records) fits one V100 pass in 16.6 min, and
the leaderboard-verified 0.9873 file was produced that way. For larger data, exact cost grows with queries × pool
(10× records on both sides ≈ 100× work, and float vectors stop fitting on one GPU), so the default is scalable. A full
scalable run on the test set (`submission/output_scalable`, validator PASS) reproduces 99.96–99.99% of the submitted
matched pairs per country and scores 0.99033 on validation (exact 0.99036); the e5 and Laya v1 cross-encoders only run on French pairs, which the France model still uses.

---

## 4. Matching Model

**Features used (48 + cross-encoder scores):**
- Name features: RapidFuzz ratio / partial / token-set / token-sort on raw, core (legal words stripped) and consonant
  skeleton names; **Jaro-Winkler** on core names; **Jaccard** and containment of core tokens; first-token equality;
  acronym and web-domain matches; token counts, length ratio, script of the candidate; a **Levenshtein edit-operation
  signature** on raw names (doubled/undoubled letters, digit substitutions, insertions, deletions) that catches the
  decoy edits.
- Address features: ratio / partial / token-set / token-sort, Jaccard, containment, house-number equality, number
  Jaccard and relative difference, last component equality, empty-address flag, length ratio.
- Other: dense bi-encoder cosine; **8 competitor features** from the reverse index of all S1 candidate lists (best
  rival S1 score, margin, rival name/address similarity) — density-invariant; **cross-encoder logits**:
  `multilingual-e5-small` cross-encoder (MIT), **Laya v1 and v2** (`convaiinnovations/laya-multilingual` encoder,
  Apache-2.0, 307M; v2 also trained on synthetic French built from a hand-written lexicon), **BGE-reranker-v2-m3**
  (Apache-2.0, 568M). An ablation shows BGE + Laya v2 alone match all four (0.99017 vs 0.99020).

**Model type:** LightGBM (MIT) matcher over the stacked features (127 leaves, lr 0.05, early stopping); a second
LightGBM **specialist for S2/S3 records without an address** (+ the number of S1 sharing the exact name) decides those
pairs (+0.00018 validation; a same-threshold control gives +0). **Ownership constraint:** a record predicted for
several S1 keeps only its highest-probability claim. **France:** the matcher without BGE (BGE misreads injected
French accents), then the **France map** rules below.  
**Threshold selection method:** macro F0.5 sweep on validation fold 0 (0.75 for the India/US matcher, 0.725 for the
France stack; the curve is flat within ±0.001 over 0.5–0.9).

**France map (label-free, from the unlabelled test set).** French candidate pairs are classified by a model-free
edit classifier (name op × address op). With copy-budget tests (per-source match caps from the training ground truth),
independence tests and the house-number-shift set {1,2,3,4,5,7,9,11,13,21}, we estimated P(match) per operation:
same-address category swap ≈ 0.1 (the model accepted 66%), same-address filler variant ≈ 0.95–1.0 (model accepted
44–73%). Removing 19,799 accepted category swaps (public LB 0.984 → 0.986) and adding 14,877 filler variants (→ 0.987)
confirmed the map. A second round of map rules (adding clean names whose street or house number is re-written, dropping other names at the same address) scored 0.9868 and was not kept.

**Laya-FR (self-training, organiser-approved).** To generalise the map beyond hand-written classes, Laya v2 was
fine-tuned (1 epoch, bf16) on 1.44M French pairs derived only from the test set: 710k real French candidate pairs
labelled by the map's near-deterministic classes (clean copies, legal-form adds and same-address filler variants = 1;
same-address category swaps, number-shift decoys, SNC changes = 0) plus 725k pairs from a map-driven generator over all
259k French S1 anchors (words, formats and rates harvested from French test records). The final France keeps the map
decisions, removes 13,605 name-only copies that Laya-FR rejects (logit < 0; the map estimates P(match) ≈ 0.3 for this
class) and adds 10,942 pairs that Laya-FR accepts with logit > 3: public leaderboard 0.987 → 0.9873. Using Laya-FR
directly at logit > 0 would also accept weak same-address brand/number-noise pairs (≈ 80% class rate), below the
F0.5 break-even, so only its confident decisions are used. All words, formats and rates come from
French test records; India/US labels were used only to measure which operations mean match or decoy.

**Use of test data (organiser-approved).** Blocking indexes, unsupervised statistics and token frequencies on the test
records, and pseudo-labelling for self-training were confirmed as allowed by the organisers. Leaderboard feedback was
used to choose between candidate versions (5 uploads per day).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation (fold 0, India + US, 220,731 S1) **0.99036** (India 0.99147, US 0.98962) with
  exact blocking; **public leaderboard 0.9873** (France included). Progression: 0.9696 (bi-encoder + features) →
  0.9805 (competitor features) → 0.9886 (+ e5 cross-encoder) → 0.9895 (+ Laya) → 0.9899 (+ BGE) → 0.9904
  (+ specialist, exact blocking); leaderboard 0.982 → 0.98314 → 0.984 → 0.986 → 0.987 → 0.9873.
- **Common false positives (wrong merges):** France: same-address category-word swaps ("Nantes Club SARL" vs
  "Nantes École SARL"), now removed; different multi-word names at the same address; name-only records whose common
  name matches several S1 records.
- **Common false negatives (missed matches):** S2/S3 records without an address whose name is shared by several S1
  (27% of these misses share the exact name with ≥ 2 S1 — genuinely ambiguous, so skipping is the F0.5-optimal call);
  French filler variants before the map fix; clean names whose street or number is re-written.

What did not help (validation or leaderboard): seed ensembles (±0.00003), expected-F0.5 decoding (−0.00036), extra
address features, TF-IDF features (+0.00003), XGBoost / blending (0), sibling-name stage 2 (+0.00003), singleton
model (+0.00001), Qwen3-Reranker-4B (blend +0.00006; France-deciding −0.0005 LB), Qwen3-Embedding-8B (AUC 0.85 vs
0.996 for cross-encoders), synthetic French with test vocabulary and a transferred India/US filler rule (LB −0.001).

---

## 6. Conclusion
A fine-tuned bi-encoder with a cheap filter gives a small candidate set (4.3 per S1) that keeps ~99% of matches, and
a LightGBM matcher over string, competitor and four cross-encoder features reaches 0.990 on India/US. The main lesson
is that the unlabelled country needed its own, carefully verified map of the data generator rather than more model
capacity: fixing one French decoy class moved the leaderboard more than any model change. All models are MIT or
Apache-2.0 and ≤ 568M parameters.
