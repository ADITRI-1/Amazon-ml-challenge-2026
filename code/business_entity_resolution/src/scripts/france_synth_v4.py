"""France synthetic pair generator v4 — follows the operation map in logs/france_map.md.

Provenance (user constraint): every string and every frequency used here is harvested at run time
from the FRENCH TEST records (dataset/test/test_source{1,2,3}.tsv, country == France) and from the
label-free blocking index cache/filt_test_exact.parquet (+ cache/test_norm.parquet for row ids).
No model probabilities, no India/US text, no hand-made lexicon.  India/US labels were used only
offline (logs/france_map.md) to decide which operation is a match and which is a decoy; those
decisions are the P_MATCH table below.

Pipeline
  1. harvest(): classify every France S1 x candidate pair with a model-free edit-operation
     classifier (name op x address op), then collect vocabularies / formats / rates from it.
  2. generate(): for each real French S1 anchor, draw records per operation class with the
     harvested per-S1 rates, render them with harvested S2/S3 formatting, label them with P_MATCH.

Usage:  python scripts/france_synth_v4.py --n 60000 --out cache/synth_fr_v4.parquet
        python scripts/france_synth_v4.py --n all  --out cache/synth_fr_v4_all.parquet
Output columns: text_a (S1 "name | address"), text_b (generated "name | address"), y (0/1),
op (operation class), p_map (map's P(match) for the class), src (S2/S3), anchor (S1 entity id).
"""
import argparse
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import polars as pl
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "dataset" / "test"
CACHE = ROOT / "cache"

# ----------------------------------------------------------------------------------------------
# Map decisions (see logs/france_map.md, section "Per-operation P(match) for France").
# 1.0 / 0.0 = deterministic label; values in between = Bernoulli label (uncertain classes).
# ----------------------------------------------------------------------------------------------
P_MATCH = {
    "clean": 1.0,               # exact/case/accent/format/reorder/web/legal-drop/stopword/typo @ same address
    "clean_numdrop": 1.0,       # clean name, house number dropped
    "clean_streettypo": 1.0,    # clean name, street name typo
    "legal_add": 1.0,           # SARL/SAS/SASU/EURL/SA/SCI appended @ same address
    "M_swap": 1.0,              # category word -> filler (Fils/Services/Associés/France/Groupe/Développement/Cie) @ same
    "M_add": 1.0,               # filler appended @ same address
    "acronym": 0.95,
    "connector": 1.0,           # "<Brand> dba|fka|t/a|... <S1 name>" (S3)
    "orgword_add": 1.0,         # Society/Authority/Board/... appended @ same address
    "drop_word": 0.95,
    "heavy_typo": 1.0,          # scrambled / inserted letters in one word
    "abbrev": 1.0,              # Compagnie->Cie, Frères->Frs, Saint->St, ...
    "brand": 0.8,               # invented single-token brand @ same address (uncertain)
    "clean_numtypo": 0.8,       # clean name, one digit of the house number edited (not a decoy shift)
    # decoys
    "numshift_decoy": 0.0,      # name edit + house number + delta in {1,2,3,4,5,7,9,11,13,21}
    "catswap": 0.1,             # category word -> other category word @ same address
    "diffname_other": 0.1,      # different multi-word name @ same address
    "initials_change": 0.2,     # 2-4 letter initials token changed @ same address
    "diffname_numdrop": 0.1,    # different name, number dropped
    "nameonly_clean": 0.3,      # empty address, clean name variant (uncertain)
    "nameonly_edit": 0.02,      # empty address + decoy edit (legal swap / D-word)
}

REGION_MIN = 1000
LEG_ALL = {"sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "ei", "selarl", "scop", "sca", "gie", "eirl", "selas"}
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "and", "a", "au", "aux", "en"}
_NA = re.compile(r"[^a-z0-9]+")
_DIG = re.compile(r"\d+")


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def toks(s):
    s = fold(s).lower()
    s = re.sub(r"(?<![a-z])((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)
    out = []
    for x in _NA.sub(" ", s.replace("&", " and ")).split():
        x2 = x.replace("5", "s").replace("0", "o") if not x.isdigit() else x
        out.append(x2 if x2 in LEG_ALL else x)
    return out


# ---------------------------------------------------------------- model-free pair classifier
def name_op(a, b):
    if not b.strip():
        return "empty", ""
    bl = b.strip()
    if ".com" in bl.lower() or bl.startswith("@") or bl.startswith("#") or (" " not in bl and len(bl) > 12 and bl.islower()):
        return "web", ""
    if a == b:
        return "exact", ""
    if a.lower() == b.lower():
        return "case", ""
    if fold(a).lower() == fold(b).lower():
        return "accent", ""
    ta, tb = toks(a), toks(b)
    if ta == tb:
        return "format", ""
    la = [t for t in ta if t in LEG_ALL]; lb = [t for t in tb if t in LEG_ALL]
    ca = [t for t in ta if t not in LEG_ALL]; cb = [t for t in tb if t not in LEG_ALL]
    if ca == cb or sorted(ca) == sorted(cb):
        if sorted(la) == sorted(lb):
            return "reorder", ""
        if la and not lb:
            return "legal_drop", ""
        if lb and not la:
            return "legal_add", ",".join(lb)
        return "legal_swap", ",".join(lb)
    ra, rb = list(ca), list(cb)
    for t in list(ra):
        if t in rb:
            ra.remove(t); rb.remove(t)
    typo = 0
    for t in list(ra):
        for u in rb:
            if len(t) > 3 and len(u) > 3 and Levenshtein.distance(t, u) <= (1 if len(t) < 7 else 2):
                ra.remove(t); rb.remove(u); typo += 1
                break
    ra2 = [t for t in ra if t not in STOP]; rb2 = [t for t in rb if t not in STOP]
    if not ra2 and not rb2:
        return ("typo" if typo else "stopword"), ""
    if len(ca) - len(ra) == 0:
        ini = "".join(t[0] for t in ca if t not in STOP and t != "france")
        if len(cb) == 1 and len(cb[0]) >= 2 and (cb[0] == ini or cb[0] == ini[: len(cb[0])]):
            return "acronym", cb[0]
        return "diffname", " ".join(cb)
    if rb2 and not ra2:
        return "add", " ".join(rb2)
    if ra2 and not rb2:
        return "drop", " ".join(ra2)
    if len(ra2) == 1 and len(rb2) == 1:
        return "swap", f"{ra2[0]}>{rb2[0]}"
    return "multi", ""


def street_part(addr):
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    for p in parts:
        if _DIG.search(p):
            return p
    return ""


STREET_WORDS_MIN = 300  # harvested street-type tokens (see harvest)


def addr_op(a, b, stypes):
    if not b.strip():
        return "empty", None, None
    sa, sb = street_part(a), street_part(b)
    ma, mb = _DIG.search(sa), _DIG.search(sb)
    na = int(ma.group(0)) if ma else None
    nb = int(mb.group(0)) if mb else None
    wa = [t for t in toks(sa) if not t.isdigit() and t not in stypes]
    wb = [t for t in toks(sb) if not t.isdigit() and t not in stypes]
    if not sb:
        st = "nostreet"
    elif wa == wb:
        st = "same"
    else:
        st = "typo" if Levenshtein.normalized_similarity(" ".join(wa), " ".join(wb)) >= 0.8 else "diff"
    if st == "diff":
        return "strdiff", na, nb
    if na is not None and nb is not None:
        if na == nb:
            return "same", na, nb
        return ("numtypo" if Levenshtein.distance(str(na), str(nb)) == 1 else "numdiff"), na, nb
    if na is not None and nb is None:
        return "numdrop", na, nb
    return "same" if st == "same" else "nostreet", na, nb


def _classify(args):
    n1, a1, n2, a2, stypes = args
    op, det = name_op(n1, n2)
    ar, na, nb = addr_op(a1, a2, stypes)
    return op, det, ar, na, nb


# ---------------------------------------------------------------- harvest
def read_test(i):
    return pl.read_csv(TEST / f"test_source{i}.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")


def subseq(t, s):
    it = iter(s)
    return all(c in it for c in t)


def harvest(procs=36):
    t0 = time.time()
    fr = pl.concat([read_test(i).with_columns(pl.lit(f"S{i}").alias("src")) for i in (1, 2, 3)]).filter(pl.col("country") == "France")
    s1 = fr.filter(pl.col("src") == "S1")
    s23 = fr.filter(pl.col("src") != "S1")
    V = {}
    parts = lambda a: [p.strip() for p in a.split(",") if p.strip()]  # noqa: E731
    # regions / cities (S1)
    last = Counter(parts(a)[-1] for a in s1["business_address"].to_list() if parts(a))
    regions = [r for r, c in last.items() if c >= REGION_MIN and not _DIG.search(r)]
    regions = [r for r in regions if " " in r or "-" in r][:3] if len(regions) > 3 else regions
    cityreg = Counter()
    for a in s1["business_address"].to_list():
        p = parts(a)
        if len(p) >= 3 and p[-1] in regions and not _DIG.search(p[-2]):
            cityreg[(p[-2], p[-1])] += 1
    city2reg = {}
    for (c, r), n in cityreg.most_common():
        if n >= 500 and c not in city2reg:
            city2reg[c] = r
    cityf = {fold(c).lower(): c for c in city2reg}
    # département per city (S2/S3 tail that is neither region nor city)
    cd = Counter()
    for a in s23["business_address"].to_list():
        p = parts(a)
        if len(p) >= 2 and p[-1] not in regions and fold(p[-1]).lower() not in cityf and not _DIG.search(p[-1]):
            for q in p[:-1]:
                if fold(q).lower() in cityf:
                    cd[(cityf[fold(q).lower()], p[-1])] += 1
    city2dept = {}
    for (c, dpt), n in cd.most_common():
        if c not in city2dept and n >= 200:
            city2dept[c] = dpt
    V.update(regions=regions, city2reg=city2reg, city2dept=city2dept)
    # S1 token stats
    s1_names = s1["business_name"].to_list()
    tc = Counter(t for n in s1_names for t in set(toks(n)))
    s1_leg = Counter(t for n in s1_names for t in toks(n) if t in LEG_ALL)
    CAT = {t for t, c in tc.items() if c >= 150 and t not in LEG_ALL and len(t) > 2 and not t.isdigit()}
    # street-type tokens: first alpha token after the number in S1 street parts
    stc = Counter()
    for a in s1["business_address"].to_list():
        m = re.match(r"^\s*\d+\s*(?:bis|ter|quater|[A-Da-d])?\s+(\S+)", street_part(a))
        if m:
            stc[fold(m.group(1)).lower().strip(".")] += 1
    base_types = {t for t, c in stc.items() if c >= STREET_WORDS_MIN}
    print(f"[harvest] {len(s1):,} S1, {len(s23):,} S2/S3 French records; regions {regions}; {len(city2reg)} cities; "
          f"{len(city2dept)} depts; {len(CAT)} category words; street types {sorted(base_types)[:12]}...", flush=True)

    # ---- candidate pairs (label-free blocking index) ----
    tn = pl.read_parquet(CACHE / "test_norm.parquet", columns=["entity_id", "business_name", "business_address", "country"]).with_row_index("rid")
    cp = pl.read_parquet(CACHE / "filt_test_exact.parquet", columns=["r1", "r2"])
    a = tn.rename({"rid": "r1", "entity_id": "e1", "business_name": "n1", "business_address": "a1", "country": "c1"})
    b = tn.rename({"rid": "r2", "entity_id": "e2", "business_name": "n2", "business_address": "a2", "country": "c2"})
    cp = cp.join(a, on="r1").filter(pl.col("c1") == "France").join(b, on="r2").with_columns(pl.col("e2").str.slice(0, 2).alias("src"))
    # street-type variant tokens seen in S2/S3 (harvested below) are also street words for the classifier
    stypes_cls = set(base_types)
    items = [(x, y, z, w, stypes_cls) for x, y, z, w in zip(cp["n1"].to_list(), cp["a1"].to_list(), cp["n2"].to_list(), cp["a2"].to_list())]
    with Pool(procs) as p:
        res = p.map(_classify, items, chunksize=4000)
    ops, dets, ars, nas, nbs = zip(*res)
    cp = cp.with_columns(pl.Series("op", ops), pl.Series("det", dets), pl.Series("ar", ars))
    print(f"[harvest] classified {len(cp):,} France candidate pairs in {time.time() - t0:.0f}s", flush=True)
    n_s1 = cp["r1"].n_unique()
    N1, A1, N2, A2, SRC = (cp[c].to_list() for c in ("n1", "a1", "n2", "a2", "src"))

    def rawdiff(x, y):
        X = [fold(t).lower() for t in x.split()]
        return " ".join(t for t in y.split() if fold(t).lower() not in X)

    # added words at same address -> M list (filler) and org words; connectors + brands
    add_same = Counter(); add_same_raw = Counter(); conn = Counter(); brands = Counter(); orgw = Counter()
    swap_same = Counter(); abbrev = Counter(); catswap_t = Counter(); accent = Counter(); accent_first = Counter()
    decoy_edit = Counter(); decoy_add_raw = Counter(); decoy_leg = Counter(); match_leg = Counter()
    deltas = Counter(); nshift_per_s1 = Counter(); leg_surface = defaultdict(Counter)
    match_leg_raw = Counter(); decoy_leg_raw = Counter()
    cls_count = Counter()
    typo_digit = Counter()
    for i, (op, det, ar) in enumerate(zip(ops, dets, ars)):
        n1, a1, n2, a2, src = N1[i], A1[i], N2[i], A2[i], SRC[i]
        same = ar == "same"
        if op == "add" and same:
            add_same[det] += 1
            m = re.search(r"^(.*?)\s+(dba|DBA|fka|FKA|aka|t/a|trading as|formerly known as|formerly)\s+", n2)
            if m and fold(n1).lower() in fold(n2).lower():
                conn[m.group(2)] += 1; brands[m.group(1)] += 1
            else:
                add_same_raw[rawdiff(n1, n2)] += 1
        elif op == "swap" and same:
            s, tg = det.split(">")
            swap_same[det] += 1
            if len(tg) < len(s) and tg[0] == s[0] and subseq(tg, s):
                src_w = [w for w in n1.split() if fold(w).lower().strip(".,") == s]
                tg_w = [w for w in n2.split() if fold(w).lower().strip(".,") == tg]
                if src_w and tg_w:
                    abbrev[(s, tg_w[0])] += 1
            elif s in CAT and tg in CAT:
                tg_w = [w.strip(".,()[]") for w in n2.split() if fold(w).lower().strip(".,()[]") == tg]
                if tg_w:
                    w = tg_w[0]
                    catswap_t[w if (w[:1].isupper() and w[1:].islower()) else w.capitalize()] += 1
        elif op == "diffname" and same:
            tb = [x for x in toks(n2) if x not in LEG_ALL]
            if len(tb) == 1 and tb[0] not in CAT and tc.get(tb[0], 0) < 5 and not tb[0].endswith("com"):
                brands[n2.strip()] += 1
        elif op == "accent" and len(n1) == len(n2):
            for j, (x, y) in enumerate(zip(n1, n2)):
                if x != y and fold(x) == x and fold(y) != y and fold(y).lower() == x.lower():
                    accent[(x.lower(), y.lower())] += 1
                    accent_first[j == 0 or n1[j - 1] == " "] += 1
        elif op == "typo" and len(n1) == len(n2):
            for x, y in zip(n1, n2):
                if x != y and y.isdigit() and x.isalpha():
                    typo_digit[(x.lower(), y)] += 1
        if op == "legal_add" and (same or ar in ("numdiff", "numtypo")):
            (match_leg if same else decoy_leg)[det] += 1
            raw = rawdiff(n1, n2)
            if raw and len(raw) <= 12:
                (match_leg_raw if same else decoy_leg_raw)[raw] += 1
        if op == "legal_swap" and ar in ("numdiff", "numtypo"):
            raw = rawdiff(n1, n2)
            if raw and len(raw) <= 12:
                decoy_leg_raw[raw] += 1
        if op == "legal_swap" and ar in ("numdiff", "numtypo"):
            decoy_leg[det] += 1
        if ar in ("numdiff", "numtypo") and op not in ("exact", "case", "accent", "format", "reorder", "web", "legal_drop"):
            d_ = nbs[i] - nas[i]
            deltas[d_] += 1
        # legal surface variants in S2/S3 names
        for m in re.finditer(r"[\[\(]?(?:[A-Za-zÀ-ÿ5]\.){2,}[A-Za-z]?\.?[\]\)]?|[\[\(]?\b[A-Za-zÀ-ÿ5]{2,5}\b[\]\)]?", n2):
            tt = toks(m.group(0))
            if len(tt) == 1 and tt[0] in LEG_ALL:
                leg_surface[tt[0]][m.group(0)] += 1
    # decoy (number-shift) edit mix, restricted to the harvested delta set
    top_d = [d_ for d_, c in deltas.most_common() if c >= 0.02 * sum(deltas.values())]
    for i, (op, det, ar) in enumerate(zip(ops, dets, ars)):
        if ar in ("numdiff", "numtypo") and nas[i] is not None and nbs[i] - nas[i] in top_d:
            if op not in ("exact", "case", "accent", "format", "reorder", "web", "legal_drop"):
                decoy_edit[op] += 1
                if op == "add":
                    decoy_add_raw[rawdiff(N1[i], N2[i])] += 1
    shift_s1 = cp.with_columns(pl.Series("na", nas, dtype=pl.Int64), pl.Series("nb", nbs, dtype=pl.Int64)).filter(
        pl.col("ar").is_in(["numdiff", "numtypo"]) & ~pl.col("op").is_in(["exact", "case", "accent", "format", "reorder", "web", "legal_drop"])
        & (pl.col("nb") - pl.col("na")).is_in(top_d)).group_by("r1").len()
    nshift_per_s1 = Counter(shift_s1["len"].to_list())
    nshift_per_s1[0] = n_s1 - shift_s1.height

    Mset = [w for w, c in add_same.most_common() if c >= 500]                       # filler words
    Mraw = Counter({k: v for k, v in add_same_raw.items() if k and all(t in Mset for t in toks(k) if t not in STOP)})
    Oraw = Counter({k: v for k, v in add_same_raw.items() if k and len(toks(k)) == 1 and toks(k)[0] not in Mset
                    and 20 <= v and toks(k)[0].isalpha() and len(toks(k)[0]) > 3})
    Draw = Counter({k: v for k, v in decoy_add_raw.items() if k and len(k) < 40 and v >= 10})
    dwords = {t for k in Draw for t in toks(k)}
    Oraw = Counter({k: v for k, v in Oraw.items() if toks(k)[0] not in dwords})

    # class rates per S1 (drive how many records of each class an anchor gets)
    def cls(op, det, ar):
        if ar == "empty":
            if op in ("exact", "case", "accent", "format", "reorder", "legal_drop", "legal_add", "typo", "stopword"):
                return "nameonly_clean"
            if op in ("legal_swap", "add"):
                return "nameonly_edit"
            return None
        if ar in ("numdiff", "numtypo"):
            if op in ("exact", "case", "accent", "format", "reorder", "legal_drop"):
                return "clean_numtypo"
            return None  # number-shift decoys are drawn per S1 from nshift_per_s1
        if ar == "numdrop":
            if op in ("exact", "case", "accent", "format", "reorder", "legal_drop", "web"):
                return "clean_numdrop"
            if op == "diffname":
                return "diffname_numdrop"
            return None
        if ar != "same":
            return None
        if op in ("exact", "case", "accent", "format", "reorder", "web", "legal_drop", "stopword", "typo"):
            return "clean"
        if op == "legal_add":
            return "legal_add"
        if op == "acronym":
            return "acronym"
        if op == "drop":
            return "drop_word"
        if op == "add":
            if det in Mset:
                return "M_add"
            return "connector" if re.search(r"\b(dba|fka|aka|t a|trading as|formerly)\b", det) else "orgword_add"
        if op == "swap":
            s, tg = det.split(">")
            if tg in Mset:
                return "M_swap"
            if len(s) <= 4 and len(tg) <= 4 and s not in CAT and tg not in CAT:
                return "initials_change"
            if len(tg) < len(s) and tg[0] == s[0] and subseq(tg, s):
                return "abbrev"
            if s in CAT and tg in CAT:
                return "catswap"
            return "heavy_typo"
        if op == "diffname":
            tb = [x for x in det.split() if x not in LEG_ALL]
            return "brand" if len(tb) == 1 and tc.get(tb[0], 0) < 5 else "diffname_other"
        return None
    for op, det, ar in zip(ops, dets, ars):
        c = cls(op, det, ar)
        if c:
            cls_count[c] += 1
    # clean name-op mix and source-specific formats
    clean_mix = Counter(op for op, ar in zip(ops, ars) if ar == "same" and op in ("exact", "case", "accent", "format", "reorder", "web", "legal_drop", "stopword", "typo"))
    web_forms = Counter()
    for op, n2 in zip(ops, N2):
        if op == "web":
            web_forms["@" if n2.startswith("@") else "#" if n2.startswith("#") else ".com" if n2.lower().endswith(".com") else "plain"] += 1
    # address formats from clean same-address pairs
    numfmt = {"S2": Counter(), "S3": Counter()}; stvar = defaultdict(Counter); order = {"S2": Counter(), "S3": Counter()}
    namecase = {"S2": Counter(), "S3": Counter()}; streettypo = Counter()
    for i, (op, ar) in enumerate(zip(ops, ars)):
        if op not in ("exact", "case", "format") or ar != "same":
            continue
        n1, a1, n2, a2, src = N1[i], A1[i], N2[i], A2[i], SRC[i]
        p1, p2 = parts(a1), parts(a2)
        kinds = []
        for q in p2:
            fq = fold(q).lower()
            if q in regions:
                kinds.append("REG")
            elif _DIG.search(q):
                kinds.append("STREET")
            elif q in city2dept.values():
                kinds.append("DEPT")
            elif fq in cityf:
                kinds.append("CITY")
            else:
                kinds.append("OTHER")
        if "OTHER" not in kinds:
            order[src]["-".join(kinds)] += 1
        sp1, sp2 = street_part(a1), street_part(a2)
        m2 = re.match(r"^\s*(N°|Nº|No\.?|NO|#)?\s*(0*)(\d+)(\s*-)?\s*(\S+)", sp2); m1 = re.match(r"^\s*(\d+)\s*(\S+)", sp1)
        if m1 and m2:
            numfmt[src][(m2.group(1) or "") + ("0" * min(len(m2.group(2)), 3)) + ("-" if m2.group(4) else "")] += 1
            stvar[fold(m1.group(2)).lower()][m2.group(5)] += 1
        n_a = [t for t in toks(sp1) if not t.isdigit()]; n_b = [t for t in toks(sp2) if not t.isdigit()]
        streettypo[n_a != n_b] += 1
        namecase[src]["same" if n1 == n2 else "upper" if n2 == n2.upper() else "lower" if n2 == n2.lower() else "title" if n2 == n2.title() else "other"] += 1
    numdrop_rate = sum(1 for op, ar in zip(ops, ars) if ar == "numdrop" and op in ("exact", "case", "format")) / max(1, sum(1 for op, ar in zip(ops, ars) if ar in ("same", "numdrop") and op in ("exact", "case", "format")))
    src_mix = Counter(SRC)
    V.update(
        CAT=sorted(CAT), s1_leg=dict(s1_leg), Mset=Mset, Mraw=dict(Mraw), Oraw=dict(Oraw), Draw=dict(Draw),
        conn=dict(conn), brands=[b for b, _ in brands.most_common(20000)],
        abbrev={f"{s}>{t}": c for (s, t), c in abbrev.items() if c >= 100 and fold(t).lower() not in Mset},
        catswap_t=dict(Counter({k: v for k, v in catswap_t.items() if v >= 20 and toks(k) and toks(k)[0] not in Mset})),
        match_leg_raw=dict(match_leg_raw.most_common(40)), decoy_leg_raw=dict(decoy_leg_raw.most_common(60)),
        accent={f"{x}>{y}": c for (x, y), c in accent.items()}, accent_first=accent_first[True] / max(1, sum(accent_first.values())),
        typo_digit={f"{x}>{y}": c for (x, y), c in typo_digit.items() if c >= 20},
        match_leg=dict(match_leg), decoy_leg=dict(decoy_leg), decoy_edit=dict(decoy_edit),
        deltas={str(d_): deltas[d_] for d_ in top_d}, nshift_per_s1={str(k): v for k, v in nshift_per_s1.items()},
        leg_surface={k: dict(v.most_common(25)) for k, v in leg_surface.items()},
        cls_rate={k: v / n_s1 for k, v in cls_count.items()}, clean_mix=dict(clean_mix), web_forms=dict(web_forms),
        numfmt={k: dict(v) for k, v in numfmt.items()}, stvar={k: dict(v.most_common(12)) for k, v in stvar.items() if sum(v.values()) >= 200},
        order={k: dict(v.most_common(20)) for k, v in order.items()}, namecase={k: dict(v) for k, v in namecase.items()},
        street_typo_rate=streettypo[True] / max(1, sum(streettypo.values())), numdrop_rate=numdrop_rate,
        src_mix={k: v for k, v in src_mix.items()}, n_s1_with_cands=n_s1,
    )
    print(f"[harvest] done in {time.time() - t0:.0f}s: M-list {Mset}; D-adds {list(Draw)[:8]}; decoy deltas {top_d}", flush=True)
    anchors = list(zip(s1["entity_id"].to_list(), s1["business_name"].to_list(), s1["business_address"].to_list()))
    return V, anchors


# ---------------------------------------------------------------- rendering helpers
def wchoice(r, counter):
    ks = list(counter.keys()); ws = list(counter.values())
    return r.choices(ks, weights=ws, k=1)[0]


class Gen:
    def __init__(self, V, seed):
        self.V = V
        self.r = random.Random(seed)
        self.cityf = {fold(c).lower(): c for c in V["city2reg"]}
        self.regions = set(V["regions"])
        acc = defaultdict(Counter)
        for k, c in V["accent"].items():
            x, y = k.split(">"); acc[x][y] += c
        self.acc = acc
        self.abbrev = defaultdict(Counter)
        for k, c in V["abbrev"].items():
            s, t = k.split(">"); self.abbrev[s][t] += c
        self.catset = set(V["CAT"])
        self.mset = set(V["Mset"])
        self.digit = Counter({k: c for k, c in V["typo_digit"].items()})

    # --- address
    def parse_addr(self, a):
        p = [q.strip() for q in a.split(",") if q.strip()]
        reg = next((q for q in p if q in self.regions), None)
        city = next((self.cityf[fold(q).lower()] for q in p if fold(q).lower() in self.cityf), None)
        street = next((q for q in p if _DIG.search(q)), None)
        extras = [q for q in p if q not in (reg, street) and (city is None or fold(q).lower() != fold(city).lower())]
        m = re.match(r"^\s*(\d+)\s*((?:bis|ter|quater|BIS|Bis|[A-Da-d])\b)?\s*(\S+)?\s*(.*)$", street or "")
        num, suf, stype, sname = (m.group(1), m.group(2) or "", m.group(3) or "", m.group(4) or "") if m else (None, "", "", street or "")
        return dict(num=num, suf=suf, stype=stype, sname=sname, city=city, reg=reg or (self.V["city2reg"].get(city) if city else None), extras=extras)

    def typo(self, w):
        r = self.r
        if len(w) < 4:
            return w
        k = r.random()
        i = r.randrange(1, len(w) - 1)
        if k < 0.3:
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if k < 0.55:
            return w[:i] + w[i + 1:]
        if k < 0.8:
            return w[:i] + r.choice("aeioulnrst") + w[i:]
        mid = list(w[1:-1]); r.shuffle(mid)
        return w[0] + "".join(mid) + w[-1]

    def render_addr(self, P, src, num=None, drop_num=False, street_typo=False):
        V, r = self.V, self.r
        n = P["num"] if num is None else str(num)
        stype = P["stype"]
        base = fold(stype).lower().strip(".")
        if base in V["stvar"]:
            stype = wchoice(r, V["stvar"][base])
        sname = P["sname"]
        if street_typo and sname:
            ws = sname.split(); j = r.randrange(len(ws)); ws[j] = self.typo(ws[j]); sname = " ".join(ws)
        if n is None or drop_num:
            street = f"{stype} {sname}".strip()
        else:
            f = wchoice(r, V["numfmt"][src])
            pre = re.sub(r"[0\-]", "", f)
            zeros = f.count("0")
            num_s = "0" * zeros + n + (" " + P["suf"] if P["suf"] and r.random() < 0.5 else P["suf"])
            street = (pre + (" " if pre and r.random() < 0.6 else "") + num_s + (" -" if "-" in f else "") + " " + " ".join(x for x in [stype, sname] if x)).strip()
        city = P["city"] or ""
        if src == "S2":
            street = street.upper(); city = city.upper() if r.random() < 0.5 else city
        else:
            street = street.title() if r.random() < 0.9 else street
        pat = wchoice(r, V["order"][src]).split("-")
        out = []
        for k in pat:
            if k == "STREET":
                out.append(street); out.extend(P["extras"][:1] if P["extras"] and r.random() < 0.5 else [])
            elif k == "CITY" and city:
                out.append(city)
            elif k == "REG" and P["reg"]:
                out.append(P["reg"])
            elif k == "DEPT" and P["city"] in V["city2dept"]:
                out.append(V["city2dept"][P["city"]])
        if "STREET" not in pat and street:
            out.insert(0, street)
        return ", ".join(x for x in out if x)

    # --- name
    def case(self, n, src):
        k = wchoice(self.r, self.V["namecase"][src])
        return n.upper() if k == "upper" else n.lower() if k == "lower" else n.title() if k == "title" else n

    def accent(self, n):
        idx = [i for i, ch in enumerate(n) if ch.lower() in self.acc]
        if not idx:
            return n
        first = [i for i in idx if i == 0 or n[i - 1] == " "]
        i = self.r.choice(first) if first and self.r.random() < self.V["accent_first"] else self.r.choice(idx)
        y = wchoice(self.r, self.acc[n[i].lower()])
        return n[:i] + (y.upper() if n[i].isupper() else y) + n[i + 1:]

    def legal_tokens(self, n):
        return [w for w in n.split() if toks(w) and toks(w)[0] in LEG_ALL and len(toks(w)) == 1]

    def legal_surface(self, form):
        s = self.V["leg_surface"].get(form)
        return wchoice(self.r, s) if s else form.upper()

    def clean_variant(self, n, src):
        r, V = self.r, self.V
        k = wchoice(r, V["clean_mix"])
        legs = self.legal_tokens(n)
        if k == "exact":
            return n
        if k == "case":
            return self.case(n, src) if n != self.case(n, src) else n.upper()
        if k == "accent":
            return self.accent(n)
        if k == "format" and legs:
            w = legs[-1]; return n.replace(w, self.legal_surface(toks(w)[0]), 1)
        if k == "reorder" and legs:
            w = legs[-1]; return (w + " " + " ".join(x for x in n.split() if x != w)).strip()
        if k == "legal_drop" and legs:
            return " ".join(x for x in n.split() if x not in legs)
        if k == "web":
            f = wchoice(r, V["web_forms"]); core = re.sub(r"[^a-z0-9]", "", fold(n).lower())
            return "@" + core if f == "@" else "#" + core if f == "#" else core + ".com" if f == ".com" else core
        if k == "typo":
            ws = n.split(); cand = [i for i, w in enumerate(ws) if len(w) > 4]
            if cand:
                i = r.choice(cand); ws[i] = self.typo(ws[i]); return " ".join(ws)
        return self.case(n, src)

    def noisy(self, n, src):
        """light per-copy noise used on top of any op (case / accent)."""
        r = self.r
        if r.random() < 0.35:
            n = self.case(n, src)
        if r.random() < 0.15:
            n = self.accent(n)
        return n

    def cat_positions(self, n):
        ws = n.split()
        return [i for i, w in enumerate(ws) if fold(w).lower().strip("()[].,") in self.catset and not _DIG.search(w)
                and fold(w).lower().strip("()[].,") not in self.mset
                and fold(w).lower() not in LEG_ALL and fold(w).lower() not in self.V["city2reg"]]

    def replace_word(self, n, new, positions=None):
        ws = n.split()
        pos = positions if positions is not None else self.cat_positions(n)
        if not pos:
            return None
        i = self.r.choice(pos); ws[i] = new
        return " ".join(ws)

    def add_suffix(self, n, s):
        legs = self.legal_tokens(n)
        if legs and self.r.random() < 0.5 and n.split()[-1] == legs[-1]:
            ws = n.split(); return " ".join(ws[:-1] + [s, ws[-1]])
        return f"{n} {s}"


# ---------------------------------------------------------------- per-anchor generation
MATCH_OPS = ["clean", "clean_numdrop", "clean_streettypo", "legal_add", "M_swap", "M_add", "acronym", "connector",
             "orgword_add", "drop_word", "heavy_typo", "abbrev", "brand", "clean_numtypo"]
DECOY_OPS = ["catswap", "diffname_other", "initials_change", "diffname_numdrop", "nameonly_clean", "nameonly_edit"]


def make_record(g, op, name, P, src):
    """returns (name2, addr2) or None if op not applicable to this anchor."""
    V, r = g.V, g.r
    legs = g.legal_tokens(name)
    addr = lambda **kw: g.render_addr(P, src, **kw)  # noqa: E731
    st_typo = r.random() < V["street_typo_rate"]
    if op == "clean":
        return g.clean_variant(name, src), addr(street_typo=st_typo)
    if op == "clean_numdrop":
        return g.clean_variant(name, src), addr(drop_num=True)
    if op == "clean_streettypo":
        return g.clean_variant(name, src), addr(street_typo=True)
    if op == "clean_numtypo":
        if not P["num"]:
            return None
        n = P["num"]; k = r.random()
        if k < 0.4 and len(n) > 1:
            j = r.randrange(len(n)); n2 = n[:j] + n[j + 1:]
        elif k < 0.7:
            j = r.randrange(len(n) + 1); n2 = n[:j] + str(r.randrange(10)) + n[j:]
        else:
            j = r.randrange(len(n)); n2 = n[:j] + str((int(n[j]) + r.randrange(1, 10)) % 10) + n[j + 1:]
        n2 = n2.lstrip("0") or "1"
        return (g.clean_variant(name, src), addr(num=n2)) if n2 != n else None
    if op == "legal_add":
        if legs:
            return None
        return g.noisy(g.add_suffix(name, wchoice(r, V["match_leg_raw"])), src), addr(street_typo=st_typo)
    if op == "M_swap":
        n2 = g.replace_word(name, wchoice(r, V["Mraw"]).split()[-1] if r.random() < 0.5 else wchoice(r, V["Mraw"]))
        return (g.noisy(n2, src), addr(street_typo=st_typo)) if n2 else None
    if op == "M_add":
        return g.noisy(g.add_suffix(name, wchoice(r, V["Mraw"])), src), addr(street_typo=st_typo)
    if op == "acronym":
        ini = "".join(w[0] for w in toks(name) if w not in LEG_ALL and w not in STOP and w != "france").upper()
        return (ini, addr()) if len(ini) >= 2 else None
    if op == "connector":
        return f"{r.choice(V['brands'])} {wchoice(r, V['conn'])} {name}", addr()
    if op == "orgword_add":
        return g.noisy(f"{name} {wchoice(r, V['Oraw'])}", src), addr()
    if op == "drop_word":
        ws = name.split(); cand = [i for i, w in enumerate(ws) if w not in legs and len(ws) > 2]
        if not cand:
            return None
        del ws[r.choice(cand)]
        return g.noisy(" ".join(ws), src), addr()
    if op == "heavy_typo":
        ws = name.split(); cand = [i for i, w in enumerate(ws) if len(w) > 4 and w not in legs]
        if not cand:
            return None
        i = r.choice(cand); w = ws[i]
        for _ in range(r.choice([1, 2, 2, 3])):
            w = g.typo(w)
        ws[i] = w
        return g.noisy(" ".join(ws), src), addr()
    if op == "abbrev":
        ws = name.split()
        pos = [i for i, w in enumerate(ws) if fold(w).lower().strip(".,") in g.abbrev]
        if not pos:
            return None
        i = r.choice(pos); ws[i] = wchoice(r, g.abbrev[fold(ws[i]).lower().strip(".,")])
        return g.noisy(" ".join(ws), src), addr()
    if op == "brand":
        return r.choice(V["brands"]).split()[0], addr()
    # ---- decoys / uncertain
    if op == "catswap":
        cur = {fold(w).lower().strip("()[].,") for w in name.split()}
        opts = Counter({k: v for k, v in V["catswap_t"].items() if fold(k).lower() not in cur})
        n2 = g.replace_word(name, wchoice(r, opts)) if opts else None
        return (g.noisy(n2, src), addr(street_typo=st_typo)) if n2 else None
    if op == "diffname_other":
        return None  # filled by caller with another anchor's name (needs the anchor pool)
    if op == "initials_change":
        ws = name.split(); pos = [i for i, w in enumerate(ws) if 2 <= len(w) <= 4 and w.isalpha() and w.isupper() and fold(w).lower() not in LEG_ALL]
        if not pos:
            return None
        i = r.choice(pos); w = ws[i]; j = r.randrange(len(w) + 1)
        ws[i] = w[:j] + r.choice("ABCDEFGHIJKLMNOPRSTUVWXYZ") + w[j:] if r.random() < 0.6 else w[:max(1, j - 1)] + r.choice("ABCDEFGHIJKLMNOPRSTUVWXYZ") + w[max(1, j - 1) + 1:]
        return (g.noisy(" ".join(ws), src), addr()) if ws[i] != w else None
    if op == "diffname_numdrop":
        return r.choice(V["brands"]).split()[0], addr(drop_num=True)
    if op == "nameonly_clean":
        k = r.random()
        if k < 0.1 and not legs:
            return g.add_suffix(name, wchoice(r, V["match_leg_raw"])), ""
        return g.clean_variant(name, src), ""
    if op == "nameonly_edit":
        if legs and r.random() < 0.5:
            cur = toks(legs[-1])[0]
            opts = Counter({k: v for k, v in V["decoy_leg"].items() if k != cur and "," not in k})
            return name.replace(legs[-1], g.legal_surface(wchoice(r, opts)), 1), ""
        return g.add_suffix(name, wchoice(r, V["Draw"])), ""
    raise ValueError(op)


def decoy_shift_record(g, edit, name, P, src, delta):
    """number-shift decoy: name edit (legal swap / catswap / D-add / legal add / other name) + number + delta."""
    V, r = g.V, g.r
    legs = g.legal_tokens(name)
    n2 = None
    if edit == "legal_swap" and legs:
        cur = toks(legs[-1])[0]
        opts = Counter({k: v for k, v in V["decoy_leg_raw"].items() if toks(k) and toks(k)[0] != cur})
        n2 = name.replace(legs[-1], wchoice(r, opts), 1)
    elif edit == "legal_add" and not legs:
        n2 = g.add_suffix(name, wchoice(r, V["decoy_leg_raw"]))
    elif edit == "swap":
        cur = {fold(w).lower().strip("()[].,") for w in name.split()}
        opts = Counter({k: v for k, v in V["catswap_t"].items() if fold(k).lower() not in cur})
        n2 = g.replace_word(name, wchoice(r, opts)) if opts else None
    elif edit == "add":
        n2 = g.add_suffix(name, wchoice(r, V["Draw"]))
    elif edit in ("diffname", "multi"):
        n2 = r.choice(V["brands"]).split()[0] if edit == "diffname" else None
    if n2 is None:
        return None
    return g.noisy(n2, src), g.render_addr(P, src, num=int(P["num"]) + delta, street_typo=r.random() < V["street_typo_rate"])


DECOY_BOOST_OPS = {"catswap", "diffname_other", "initials_change", "nameonly_edit"}


def feasibility(V, anchors, seed=1):
    """fraction of anchors for which each op can be applied (rates are per S1 overall -> rescale)."""
    g = Gen(V, seed); ok = Counter(); tot = 0
    for eid, name, a in anchors:
        P = g.parse_addr(a); tot += 1
        for op in V["cls_rate"]:
            if op in P_MATCH and op != "diffname_other" and make_record(g, op, name, P, "S2") is not None:
                ok[op] += 1
    return {op: max(0.05, ok[op] / tot) if op != "diffname_other" else 1.0 for op in V["cls_rate"]}


def _gen_chunk(args):
    V, anchors, pool_names, seed, feas, boost, clean_keep = args
    g = Gen(V, seed)
    r = g.r
    rows = []
    rates = {op: min(3.0, v / feas.get(op, 1.0)) * (boost if op in DECOY_BOOST_OPS else 1.0) * (clean_keep if op == "clean" else 1.0)
             for op, v in V["cls_rate"].items()}
    shift_n = Counter({int(k): v for k, v in V["nshift_per_s1"].items()})
    dec_edit = Counter({k: v for k, v in V["decoy_edit"].items() if k in ("legal_swap", "legal_add", "swap", "add", "diffname")})
    deltas = Counter({int(k): v for k, v in V["deltas"].items()})
    srcw = Counter({k: v for k, v in V["src_mix"].items() if k in ("S2", "S3")})
    for eid, name, a in anchors:
        P = g.parse_addr(a)
        text_a = f"{name} | {a}"
        for op, rate in rates.items():
            if op not in P_MATCH:
                continue
            # Poisson draw of how many records of this class the anchor gets
            k = 0; lam = rate; x = r.random(); pk = pow(2.718281828, -lam); cum = pk
            while x > cum and k < 6:
                k += 1; pk *= lam / k; cum += pk
            for _ in range(k):
                src = wchoice(r, srcw)
                if op in ("M_swap", "catswap") and not g.cat_positions(name):
                    break
                if op == "diffname_other":
                    rec = (g.noisy(r.choice(pool_names), src), g.render_addr(P, src))
                else:
                    rec = make_record(g, op, name, P, src)
                if rec is None:
                    continue
                y = 1 if r.random() < P_MATCH[op] else 0
                rows.append((text_a, f"{rec[0]} | {rec[1]}", y, op, P_MATCH[op], src, eid))
        # number-shift decoys: one delta per anchor, k distinct edits
        if P["num"]:
            k = wchoice(r, shift_n)
            if k == 0 and boost > 1 and r.random() < (boost - 1) * (1 - shift_n[0] / sum(shift_n.values())):
                k = 1
            if k:
                delta = wchoice(r, deltas)
                if int(P["num"]) + delta <= 0:
                    delta = abs(delta)
                used = set()
                for _ in range(k):
                    e = wchoice(r, Counter({x: v for x, v in dec_edit.items() if x not in used}) or dec_edit)
                    used.add(e)
                    src = wchoice(r, srcw)
                    rec = decoy_shift_record(g, e, name, P, src, delta)
                    if rec is None:
                        continue
                    rows.append((text_a, f"{rec[0]} | {rec[1]}", 0, "numshift_decoy:" + e, 0.0, src, eid))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="60000", help="number of French S1 anchors, or 'all'")
    ap.add_argument("--out", default=str(CACHE / "synth_fr_v4.parquet"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=36)
    ap.add_argument("--vocab_out", default=None, help="optional: dump harvested vocabulary/rates as JSON")
    ap.add_argument("--decoy_boost", type=float, default=1.5, help="multiplier on French decoy-class rates (1.0 = as observed)")
    ap.add_argument("--clean_keep", type=float, default=0.5, help="fraction of easy clean-copy matches kept (1.0 = as observed)")
    args = ap.parse_args()
    t0 = time.time()
    V, anchors = harvest(args.procs)
    if args.vocab_out:
        json.dump(V, open(args.vocab_out, "w"), ensure_ascii=False, indent=1, default=str)
    r = random.Random(args.seed)
    if args.n != "all":
        anchors = r.sample(anchors, min(int(args.n), len(anchors)))
    pool_names = [n for _, n, _ in r.sample(anchors, min(50000, len(anchors)))]
    feas = feasibility(V, r.sample(anchors, min(3000, len(anchors))))
    print("[gen] op feasibility (share of anchors an op applies to):", {k: round(v, 2) for k, v in sorted(feas.items())}, flush=True)
    chunks = [anchors[i:i + 5000] for i in range(0, len(anchors), 5000)]
    with Pool(args.procs) as p:
        parts = p.map(_gen_chunk, [(V, c, pool_names, args.seed * 100003 + i, feas, args.decoy_boost, args.clean_keep) for i, c in enumerate(chunks)])
    rows = [x for part in parts for x in part]
    df = pl.DataFrame(rows, schema=["text_a", "text_b", "y", "op", "p_map", "src", "anchor"], orient="row")
    df = df.with_columns(pl.col("y").cast(pl.Int8), pl.col("p_map").cast(pl.Float32))
    df.write_parquet(args.out)
    print(f"[gen] {len(anchors):,} anchors -> {len(df):,} pairs, positives {df['y'].mean():.3f}, {time.time() - t0:.0f}s -> {args.out}")
    s = df.group_by("op").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("pos_rate")).sort("n", descending=True)
    with pl.Config(tbl_rows=60):
        print(s)


if __name__ == "__main__":
    main()
