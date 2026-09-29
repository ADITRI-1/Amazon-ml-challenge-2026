# France test-set generation map (label-free) — 2026-09-27

Goal: reverse-engineer how the synthetic French test data was generated (S1 → S2/S3 variants, decoys,
distractors), estimate P(match) per operation, and build a labelled French generator
(`scripts/france_synth_v4.py`).

Data used: France candidate pairs = `cache/filt_test_exact.parquet` joined to `cache/test_norm.parquet`
(1,169,377 pairs, 254,885 S1 with ≥1 candidate, 259,452 French S1). India/US labelled pairs =
`filt_tr` + `filt_va` (2,165,100 pairs, 82.4 % positive) with `train_ground_truth.tsv`.
`testpred_m6x` (p) is used **only** to describe the current model per class (bands: hi p≥0.98, lo p<0.02,
pred p≥0.725); every P(match) estimate below is model-free.

Op classifier (analysis only, re-implemented inside the generator): name op ∈ {exact, case, accent,
format (punctuation / legal-form spelling), reorder, web (.com/@/#), legal_drop, legal_add, legal_swap,
stopword, typo, add, drop, swap, acronym, diffname, multi} × address op ∈ {same (incl. street typo),
numdrop, numtypo (house number Levenshtein 1), numdiff, strdiff, empty}. Leading zeros are normalised
(`0082` = `82`); without that, "number changes" in US were 74 % matches (all leading-zero noise).

---------------------------------------------------------------------------------------------------
## 1. Ten key findings (all counts are France candidate pairs unless stated)

1. **A French decoy is almost always "name edit + house number + δ".** δ is uniform over
   {1,2,3,4,5,7,9,11,13,21} (8.1–9.4 % each, 86.8 % of the 155,715 name-edit + number-change pairs;
   the rest are ±10/±20 etc.). The edit is legal-form swap (36.1k), category-word swap (38.1k),
   filler/decoy-word add (38.3k), legal-form add (24.3k), other name (10k). 48.6 % of French S1
   have ≥1 such decoy (123,772 / 254,885); one δ per S1 (113k S1 have one shifted number, 9.3k two);
   records sharing the shifted number carry *different* edits (same edit in only 1.4 % of 20,633
   multi-record twins). The current model already rejects them (pred 0.0–0.2 %).
2. **Legal-form swap never happens at the same address** (legal_swap|same 296 vs 36.1k with a number
   shift). **SNC is a decoy-only legal form**: it never occurs in French S1 names and never in
   same-address legal additions (SCI/SAS/EURL/SASU/SA/SARL, 5.4–6.6k each, uniform), but is 17 % of
   decoy legal forms (5,841 SNC in number-shift decoys).
3. **Filler words at the SAME address are MATCH noise (P≈0.95–1.0), not decoys.** The filler list is
   data-driven and uniform: Groupe 1,004 / France 1,002 / Services 982 / Développement 951 /
   Et Fils 938 / Cie 896 / & Associés 708 / & Fils 703 (+ casing variants). Evidence: (a) per-source
   copy-budget ("cap") test — the count declines to ~0 when an S1 already has 5 clean S2 copies
   (S2: 0.032 → 0.0015 for adds, 0.102 → 0.009 for swaps), exactly like known match noise; (b) the
   word is drawn independently per record (two filler records of one S1 share the word 15.9 % vs
   15.4 % expected under independence) → per-copy noise, not a twin entity. The current model accepts
   only 44 % (M_add) / 72 % (M_swap) of them, and treats *services/cie* (≈99 % accepted) differently
   from *fils/groupe/associés/développement* (10–60 %) although they come from one uniform operation.
4. **The same words (France, Groupe, Développement) are also decoy words — but only with a number
   shift.** Decoy adds: France 9,452 / Groupe 8,602 / International 1,381 / Holding 1,150 /
   Participations 951 / Distribution 677 / Développement 395. International/Holding/Participations/
   Distribution appear at the same address only 8–21 times each (≈1 %). ⇒ the *address*, not the word,
   separates match from decoy.
5. **Category-word swap at the same address is a DECOY (est. P≈0.1) — the main precision leak.**
   34,696 pairs ("Nantes Club SARL" → "Nantes Ecole SARL", same address). Cap curve is flat
   (S2 0.068→0.067, S3 0.070→0.077) like decoys; target words are drawn *without* replacement
   (two catswap records of one S1 share the word 0.3 % vs 1.2 % independent); 9.3 % of them equal
   another French S1 (core name + house number) vs 1.6 % for filler swaps. The current model
   (m6x / best upload m8xhc) **accepts 65 %** of them (≈22.7k predicted pairs).
6. **Name-only records (empty address) are mostly distractors**: ~0.20 per S1 (exact 15.5k, format
   11.2k, case 6.6k, accent 5.8k, legal_drop 5.3k …); cap-test estimate 0.3–0.5 with a +0.1–0.2 bias
   → P≈0.3 (US/India labels 0.13–0.30). The LB-tested `m8xhcs` added 11,432 pairs of which ~8k were
   name-only → LB −0.0005, consistent with P<0.76 (the F0.5 break-even).
7. **Accent injection is exactly one character per record** (86,285 of 86,379 accent-only pairs):
   a→à 23,049, c→ç 18,079, o→ô 9,969, a→â 6,860, e→é/è/ë/ê 19,028, i→ï/î 6,583, u→û/ù 2,651;
   35 % on a word's first letter (Çlub, Àmicale, Ècole). Accent-only at same address = match (US 1.0,
   France model 0.94, cap test consistent).
8. **Name-clean + house-number change is match noise, not a decoy** (exact/case/format/accent/
   legal_drop × numdiff: 5.8k pairs). Their δ is *not* from the decoy set (7–11 % vs 95 % for
   legal-swap decoys; e.g. 25→1, 23→1, 4→149, 73→17), cap test ≈1.0, US analogue exact|numdiff
   P=0.96 / numtypo 0.98. The current model accepts only 7–18 %. (Low volume; medium confidence.)
9. **Address rendering** (clean same-address copies): tail = none 33 % / département 30 % /
   région 30 %, other orders (city-first, tail-first…) ~1.2 % each; S2 street always UPPER, S3 Title;
   number prefix none 83 %, N° 5.8 %, No/NO/No. 3.9 %, # 3.9 %, leading zero 3.9 %, Nº 2 %, "n -" 2 %;
   street types: Rue→RUE/R/R. (≈20 % each), Avenue→AV/Av./AVE…, Boulevard→BD/Bd/BLVD, Allée→All,
   Route→Rte, Impasse→Imp, Chemin→Ch, Cours→Crs, Quai→Q.; city→département harvested:
   Bordeaux/Pessac/Mérignac/La Teste-de-Buch/Lège-Cap-Ferret→Gironde, Nantes/Saint-Nazaire/Pornic/
   Saint-Herblain/La Baule-Escoublac→Loire-Atlantique, Lille/Roubaix/Tourcoing/Dunkerque→Nord,
   Calais→Pas-de-Calais. None of this changes P(match).
10. **Model vs map, net**: the France error is not "too many filler adds" but the *wrong split*: the
   model accepts ~22.7k same-address category swaps (decoys) and rejects ~18k same-address filler
   variants (matches) + ~5k clean-name/number-noise matches. Every LB-tested "add" variant so far
   (m7o, hcs, hcQ0) mixed name-only records / catswaps into the additions — consistent with this map.

---------------------------------------------------------------------------------------------------
## 2. India / US operation catalogue (labelled, candidate pairs after blocking filter)

P(match) by name op × address op (n = candidate pairs; sorted by France volume):

| name op | addr op | US n | US P(m) | India n | India P(m) |
|---|---|---|---|---|---|
| exact | same | 42,834 | 1.000 | 10,812 | 1.000 |
| swap (1 word) | same | 101,200 | 0.965 | 41,339 | 0.953 |
| format / case / reorder | same | 200,875 | 1.000 | 75,062 | 1.000 |
| legal_drop | same | 68,600 | 1.000 | 20,532 | 1.000 |
| accent | same | 30,652 | 1.000 | 10,635 | 1.000 |
| web | same | 48,550 | 0.995 | 22,050 | 0.986 |
| legal_add | same | 50,465 | 0.993 | 11,380 | 0.958 |
| add (1+ word) | same | 55,199 | 0.989 | 56,947 | 0.964 |
| diffname (brand/other) | same | 20,133 | 0.782 | 85,183 | 0.943 (transliteration) |
| drop | same | 51,323 | 0.999 | 10,906 | 0.994 |
| typo | same | 39,085 | 0.976 | 15,409 | 0.963 |
| exact | numdiff / numtypo | 1,872 / 1,529 | 0.960 / 0.984 | 1,062 / – | 0.981 / – |
| swap | numdiff / numtypo | 8,678 / 10,645 | 0.471 / 0.684 | 5,094 / 3,575 | 0.750 / 0.670 |
| add | numdiff / numtypo | 8,189 / 7,317 | 0.133 / 0.267 | 6,157 / 4,697 | 0.695 / 0.626 |
| legal_swap | numdiff | 2,522 | 0.199 | 5,512 | 0.935 |
| legal_add | numdiff / numtypo | 16,810 / 13,939 | 0.122 / 0.267 | 1,650 / 1,320 | 0.602 / 0.576 |
| any clean name | empty | ~80k | 0.13–0.38 | ~40k | 0.22–0.30 |

Word lists (US): same-address match words center 0.998, services 0.999, service 0.999, partners 0.91,
association/council/society/foundation/district/board/trust/authority/federation/commission 1.0;
decoy words (added mostly **with** a number change, n_num ≫ n_same): group (5,201 vs 152; 0.001),
holdings (0.002), north/east/west/south/metro/uptown/valley/harbor (0.00–0.03).
India: honorifics m/s, shri, sri, smt, dr, mr = match (0.99); decoys group/public/industries/holdings/
enterprises/exports/ventures/infratech/overseas (0.00–0.19 even at same address), swap decoys
trading/care/solutions/agro/global/infra/impex/finance (0.0).
dba/fka/aka/t/a/"trading as"/"formerly (known as)" connectors: S3 only, P = 1.000 (11,418).
Match-count structure (GT): mean 3.459 (US) / 3.465 (India) per S1, S2 1.67 (max 5) + S3 1.79 (max 6),
5.6 % of S1 have 0 matches — identical in both countries (→ same generator).

---------------------------------------------------------------------------------------------------
## 3. France operation catalogue by model-confidence band

| name op | n | hi (p≥.98) | lo (p<.02) | model pred |  | addr op | n | hi | lo | pred |
|---|---|---|---|---|---|---|---|---|---|---|
| swap | 162,901 | .342 | .278 | .524 | | same | 712,733 | .850 | .008 | .933 |
| exact | 149,284 | .853 | .022 | .878 | | same+street typo | 107,834 | .853 | .011 | .933 |
| format | 115,189 | .879 | .008 | .907 | | num_diff | 84,820 | .003 | .896 | .011 |
| case | 106,310 | .916 | .006 | .941 | | empty | 68,280 | .056 | .013 | .207 |
| legal_drop | 97,955 | .905 | .011 | .926 | | num_typo | 62,497 | .003 | .890 | .015 |
| accent | 86,379 | .841 | .003 | .939 | | num_drop | 47,177 | .611 | .015 | .729 |
| legal_add | 75,178 | .368 | .341 | .581 | | same+street diff | 28,759 | .617 | .137 | .699 |
| add | 74,622 | .251 | .553 | .284 | | | | | | |
| reorder | 71,731 | .926 | .016 | .940 | | | | | | |
| diffname | 61,737 | .130 | .204 | .370 | | | | | | |
| web | 57,132 | .924 | .022 | .944 | | | | | | |
| legal_swap | 42,914 | .003 | .809 | .009 | | | | | | |
| acronym | 17,801 | .816 | .022 | .954 | | | | | | |

Uncertain band (0.02 ≤ p < 0.98): 230,378 pairs; it is dominated by same-address swaps (catswap +
filler swap), legal_add, brand, and name-only records.
Differences vs India/US profile: France has far more swaps at the same address (catswap is new),
more legal-form traffic (70 % of S1 carry SARL/SAS/EURL/SA/SASU/SCI/EI), 8 % of S1 names contain
"(France)", and decoys concentrate on the δ number shift.

---------------------------------------------------------------------------------------------------
## 4. French-specific operations (all harvested from French test records)

Match (S1 → S2/S3 copy) operations:
- **Accent injection**, 1 char, map above; **case**: S2 names upper 22 %, lower 11 %, title 23 %,
  unchanged 20 %; S3 title 33 %, lower 11 %, upper 5 %.
- **Legal form**: drop; re-spell (S.A.R.L., [SARL], (SARL), Sàrl, Sarl, sarl, 5ARL, 5AS …); move to
  front ("SARL X"); **add** one of SCI/SAS/EURL/SASU/SA/SARL (uniform, ~2 % bracketed) — never SNC.
- **Filler** (M list above): appended at the end (96.5 %) or replacing the category word.
- Abbreviations: Frères→Frs (540), Compagnie↔Cie, Saint→St (309), Club→Cb (172).
- Typos incl. scrambles/insertions ("Lycfeee", "Cbdmiet"), digit-for-letter (S→5, O→0).
- web: name.com (84 %), @handle, #hashtag; acronym ("PC", "BTC").
- Rebrand: "<Brand> dba|DBA|t/a|trading as|aka|fka|formerly (known as) <S1 name>" (S3, ~1k each);
  bare brand tokens built from syllables (Deltaevo, Lyraectowex, Korjax) at the same address.
- Address: tail région / département / drop, order shuffle, S2 upper, number formats, street-type
  abbreviations, street-name typos (13 %), number dropped (6 %), digit edit of the number.

Decoy operations: number shift δ ∈ {1,2,3,4,5,7,9,11,13,21} + {legal swap (incl. SNC), catswap,
decoy word add (France/Groupe/International/Holding/Participations/Distribution/Développement,
often fused with a legal form "Groupe SARL", "France S.A.S."), legal add, other name}; same-address
catswap; same-address other business (another S1 / different multi-word name); initials change
("KA Lycee"→"KNA Lycee"); name-only copies (distractors).

---------------------------------------------------------------------------------------------------
## 5. Structural (label-free) tests used

1. **Copy-budget ("cap") test.** Train GT: an S1 has ≤5 S2 matches and ≤6 S3 matches. So per-copy
   *match* noise must fall towards 0 when an S1 already shows 4–5 clean copies in that source, while
   decoys stay flat. Validated on US labels (match add|same S2: 0.046→0.021→0.000; non-match flat
   0.001). Confound found and removed: legal_add is only possible for S1 without a legal form, which
   also have fewer "clean" copies → stratify by "S1 has legal form" and count clean copies without
   legal_drop. Estimator = weighted fit y(nc) = d + m·typo_curve(nc) using heavy-typo copies as the
   pure-match basis. Decoy controls (number-shift classes, model p≈0) come out at 0.06–0.24 → the
   estimator has a +0.1–0.2 floor; known matches (typos, abbreviations, case/format with street
   typo) come out at 1.0. **Limitation**: entity-level match ops (US S3 "dba" records: 100 % match but
   flat curve 0.038→0.043) also look flat ⇒ flat = "decoy OR one-per-entity op"; decline = match.
2. **Independence ("twin") test**: per-copy noise draws words independently; twin decoys share
   attributes; decoy menus draw without replacement.
3. **δ-set test** for number changes (decoy δ set vs arbitrary digit noise).
4. **Ownership**: same-address variants are almost never a clean copy of another S1 (0.1 %);
   catswap equals another French S1 (core name + number) in 9.3 % (filler swaps 1.6 %).
5. **Global count**: with the table below France implies ≈3.30 matches per S1 with candidates
   (3.41 if catswap were matches) vs 3.53 expected from India/US (3.46 per S1 incl. S1 without
   candidates) → the map does not over-count; the gap is probably blocking misses. Weak evidence.
6. **LB diff** (`output_m8xhcs` vs `output_m8xhc`, the only surviving pair): adds 11,432 = name-only
   ~8.0k + legal_add|same 2.2k + …; drops 5,070 = catswap 1,873 + filler swaps/adds 1,337 + … →
   LB −0.0005. Consistent with name-only P<0.76, catswap drops good, filler drops bad.

---------------------------------------------------------------------------------------------------
## 6. Per-operation P(match) for France (map used by the generator)

Model = share of pairs with m6x p ≥ 0.725 (≈ what the 0.984 upload predicts, France = m6xo).

| class (name op @ address) | n France | model | est. P(match) | confidence | evidence |
|---|---|---|---|---|---|
| clean copy (exact/case/accent/format/reorder/web/legal_drop) @ same | 614,896 | .995 | 1.00 | high | US/IN 1.0; cap |
| legal_add (SCI/SAS/EURL/SASU/SA/SARL) @ same | 42,240 | .938 | 1.00 | high | cap 1.0 (stratified), LB (m8o drop hurt), no SNC |
| filler **swap** (category→Fils/Services/…) @ same | 41,435 | .717 | 0.95–1.0 | high | cap 0.98, independence, US services/center 0.997 |
| filler **add** @ same | 11,484 | .444 | 0.95–1.0 | high | cap 1.0 (nc=5: 1 vs ~15 exp.), independence |
| acronym @ same | 17,507 | .989 | 0.95 | high | cap .86, US 1.0 |
| dba/fka/t/a + S1 name / org word add (Society…) @ same | 15,389 | .993 | 1.00 | high | US 1.0 |
| abbreviation / heavy typo / drop word @ same | 27,849 | .99 | 0.95–1.0 | high | cap .8–1.0, US .98–1.0 |
| brand token @ same | 24,602 | .655 | 0.8 | medium | S2 cap .8–.9, S3 flat (entity-level?), US .80/.83 |
| clean name + number noise (not δ-set) | ~5,800 | .07–.18 | 0.8 | medium | δ test, cap 1.0, US .96–.98 |
| clean name, number dropped / street differs | ~60k | .73–.95 | 0.9–1.0 | medium-high | cap, US .97–1.0 |
| **category swap @ same** | 34,696 | **.655** | **0.1** | medium-high | flat cap, without-replacement draws, 9.3 % = other S1 |
| other multi-word name @ same | 9,117 | .249 | 0.1 | medium | flat cap |
| multi-word edit @ same | 3,982 | .304 | 0.2 | low | |
| initials change (2–4 letter token) @ same | 2,066 | .41 | 0.2 | low | flat |
| **name + number shift δ** (legal swap / catswap / France-Groupe-International… / legal add / other) | ~155,700 | .00–.02 | 0.0 | high | δ-set, US .12–.27 (India higher) |
| other name, number dropped | 11,524 | .105 | 0.1 | medium | flat |
| name-only (empty address), clean name | ~50,000 | .24–.35 | 0.3 | low-medium | cap .3–.5 minus floor, US .13–.30, hcs LB |
| name-only + decoy edit (legal swap, D-word) | ~8,500 | .01–.04 | 0.02 | medium | US 0.13–0.23 |

---------------------------------------------------------------------------------------------------
## 7. Examples (S1 ⇒ candidate, m6x p)

Filler at same address (MATCH per map; model mostly rejects):
- Pharmacie Charite | 20 Rue Rosa Bonheur, Bordeaux, Nouvelle-Aquitaine ⇒ Pharmacie Charite Et Fils | 20 Rue Rosa Bonheur, Bordeaux, Gironde (p .19)
- Amicale des Work | 49 Rue Philippe-Laurent Roland, Lille ⇒ Amicale des Work Développement | 49 Rue Philippe-laurent Roland, Lille (p .03)
- Pro Etablissement SAS | 19 BD Montebello, Appartement 9, Lille ⇒ Pro SAS Et Fils | Nord, 19 BD MONTEBELLO, APPARTEMENT 9, LILLE (p .998)

Category swap at same address (DECOY per map; model mostly accepts):
- Est Societe SAS | 34 Avenue du Général Leclerc, Pessac ⇒ Est Ecole | 34 Avenue Du General Leclerc, Pessac (p .86)
- Passerelles Jeunes SARL | 35 Rue des Palmiers, Tourcoing ⇒ Passerelles Primaire SARL | 35 Rue Des Palmiers, Tourcoing (p .81)
- Europeen Comite SARL | 14 Place Rihour, Lille ⇒ Europeen Sante SARL | 14 - PLACE RIHOUR, LILLE (p .99)

Number-shift decoys (model correct):
- Cycliste Ecole SARL | 4 Rue Jean Baptiste Delambre, Nantes ⇒ Cycliste Ecole SNC | 7 R Jaen Baptiste Delambre, Nantes (δ=3, p .000)
- Mecs Amicale EURL | 118 Rue Maurice Sarraut, Tourcoing ⇒ Mecs Amicale SASU | 131 Rue Maurice Sarraut (δ=13, p .00)
- Comité des Larsen | 18 Chemin d'Exploitation, Dunkerque ⇒ Comité des Larsen France | 27 Ch D'exploitation (δ=9, p .00)

Legal add at same address (MATCH): Agricoles Local Union | 2 Allee des Siffleurs, Lège-Cap-Ferret ⇒ Agricoles Local Union (Sarl) | Gironde, 2 Allee Des Siffleurs (p 1.0)

Clean name + number noise (MATCH per map; model rejects): Chur Club SAS | 23 Rue Moriceau Thébault, Nantes ⇒ Chur Club SAS | 1 Rue Moriceau Thebault, Nantes (p .77); Bordeaux Club SAS | 2 RUE de Begles ⇒ Bordeaux Club SAS | 59 Rue De Bègles (p .20)

Name-only (mostly distractor): Verite Ecole SAS | 9 Rue Edouard Hérriot, Lille ⇒ Verite Ecole SAS | (empty) (p .31)

Brand at same address (uncertain): EHPAD de la Dialogos | 67 RUE rue docteur gustave jean rappin, Nantes ⇒ Arcnyla | 67 Rue Rue Docteur … (p .92)

---------------------------------------------------------------------------------------------------
## 8. Generator `scripts/france_synth_v4.py` and ingredient provenance

Run: `python scripts/france_synth_v4.py --n 60000 --out cache/synth_fr_v4.parquet` (≈70 s; `--n all`
= all 259,452 French S1 anchors, ≈85 s, 727,166 pairs). Options: `--decoy_boost` (default 1.5 on
catswap / other-name / initials / name-only-edit rates and on number-shift decoy presence),
`--clean_keep` (default 0.5: keep half of the easy clean copies), `--vocab_out` (dump harvested JSON).
Columns: text_a (S1 "name | address"), text_b, y, op, p_map, src, anchor.

| ingredient | source |
|---|---|
| anchors (name, address) | French S1, dataset/test/test_source1.tsv |
| régions, cities, city→région | French S1 addresses |
| city→département | French S2/S3 address tails |
| op rates per S1, clean-op mix | France candidate pairs (filt_test_exact, model-free classifier) |
| filler list (M), org words, dba connectors, brand tokens | same-address `add`/`diffname` pairs, French S2/S3 text |
| decoy words, decoy legal strings (incl. SNC), δ distribution, decoy edit mix, decoys per S1 | number-change pairs, French S2/S3 text |
| match legal strings | same-address legal_add pairs |
| category words + swap-target distribution | French S1 token frequencies (≥150), same-address swaps |
| abbreviations, accent map, digit-for-letter | same-address swap / accent / typo pairs |
| number formats, street-type variants, part orders, name/address casing | clean same-address pairs |
| P(match) per class | this map (India/US labels used only to relate ops to match / non-match) |
Not used: `src/er/synth_fr.py` lexicons, `cache/fr_vocab.json`, model probabilities, India/US text.

Run on 2026-09-27 (`--n 60000`, seed 0): **168,003 pairs, 66.5 % positive**:
clean 57,525 (y=1) · name-only clean 12,758 (30 % y=1) · catswap 10,654 (10 %) · shift-decoy add 9,860 (0) ·
clean numdrop 9,039 (1) · shift-decoy catswap 8,652 (0) · filler swap 8,361 (1) · legal add 8,194 (1) ·
shift-decoy legal swap 6,252 (0) · brand 5,186 (80 %) · heavy typo 4,705 (1) · acronym 3,673 (95 %) ·
name-only edit 3,627 (2 %) · other name @same 2,985 (9 %) · drop word 2,688 (95 %) · other name numdrop 2,643 (11 %) ·
filler add 2,325 (1) · dba connector 1,975 (1) · shift-decoy legal add 1,931 (0) · clean + number noise 1,874 (79 %) ·
org word 1,488 (1) · initials change 666 (21 %) · abbreviation 656 (1) · shift-decoy other name 286 (0).

Caveats: rendering is an approximation (street-name typos, extras such as "Appartement 9" kept 50 %);
classes with P between 0.1 and 0.9 get Bernoulli labels (column p_map holds the soft label; use it
for a soft-target loss or filter `p_map in {0,1}` for hard-only training). The biggest open
uncertainties are name-only records (0.3) and bare brand tokens (0.8).
