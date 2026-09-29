"""Synthetic French entity-resolution pairs for training the cross-encoder.

No test records are used. Built from:
  - a small hand-written French lexicon (legal forms, street types + abbreviations, business words, first/last names,
    major cities with département and région) — the same kind of domain knowledge as the LEGAL stop-list in norm.py;
  - the noise operators measured on the TRAINING data (case, legal-form swap/drop, word order, brackets, doubled
    letters, digit↔letter swaps, domain-style names, random brand names at the same address, empty addresses,
    house-number shifts, reordered address components);
  - French-specific variants: street-type abbreviations (Rue→R., Avenue→Av., ...) and département instead of région.
Hard negatives: "twins" (same generic name at a different address = different business) and decoys (doubled letter /
shifted number at the same address).
Output: rows (text_a, text_b, label) in the same "name | address" format the cross-encoder trains on.
"""
import random
import string

LEGAL = ["SARL", "SAS", "SASU", "EURL", "SCI", "SA", "SNC", "SELARL", "SCOP"]
LEGAL_LONG = {"SARL": "S.A.R.L.", "SAS": "S.A.S.", "SA": "S.A.", "EURL": "E.U.R.L."}
BUSINESS = ["Club", "Association", "Amicale", "Comité", "Maison", "École", "Collège", "Lycée", "Centre", "Union",
            "Fédération", "Pharmacie", "Boulangerie", "Garage", "Cabinet", "Atelier", "Studio", "Agence", "Institut",
            "Clinique", "Restaurant", "Café", "Hôtel", "Librairie", "Transports", "Services", "Conseil", "Immobilier",
            "Distribution", "Bâtiment", "Menuiserie", "Plomberie", "Électricité", "Coiffure", "Optique", "Boucherie"]
QUAL = ["Sportive", "Culturelle", "Régionale", "Municipale", "Populaire", "Nouvelle", "Moderne", "Générale", "Centrale",
        "Familiale", "Artisanale", "Internationale", "du Centre", "des Amis", "des Parents", "de la Gare", "du Port",
        "Saint-Michel", "Sainte-Anne", "du Lac", "des Lilas", "Bleu", "Soleil", "Horizon", "Avenir", "Harmonie"]
FIRST = ["Jean", "Pierre", "Marie", "Louis", "Paul", "Jacques", "Michel", "Claude", "Anne", "Sophie", "Nicolas",
         "François", "Isabelle", "Philippe", "Julien", "Camille", "Hélène", "Luc", "Thomas", "Élise", "André", "Chloé"]
LAST = ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy", "Moreau", "Simon",
        "Laurent", "Lefèvre", "Michel", "Garcia", "David", "Bertrand", "Roux", "Vincent", "Fournier", "Morel", "Girard",
        "Bonnet", "Dupont", "Lambert", "Fontaine", "Rousseau", "Blanc", "Guérin", "Muller", "Henry", "Roussel"]
STREET_TYPES = {"Rue": ["R.", "R", "RUE"], "Avenue": ["Av.", "Av", "AVE"], "Boulevard": ["Bd", "Bd.", "BLVD"],
                "Allée": ["All.", "Al."], "Place": ["Pl.", "Pl"], "Impasse": ["Imp.", "Imp"], "Chemin": ["Ch.", "Chem."],
                "Route": ["Rte", "Rte."], "Quai": ["Q.", "Quai"], "Résidence": ["Rés.", "Res."]}
STREET_W = [0.55, 0.12, 0.05, 0.04, 0.04, 0.03, 0.05, 0.05, 0.03, 0.04]
STREET_NAMES = ["de la République", "Victor Hugo", "Jean Jaurès", "du Général de Gaulle", "Pasteur", "de la Paix",
                "des Lilas", "de la Gare", "Gambetta", "Voltaire", "Émile Zola", "du Moulin", "des Écoles", "de l'Église",
                "Nationale", "de Paris", "Carnot", "Foch", "des Tilleuls", "du Château", "Jules Ferry", "de la Liberté",
                "Saint-Martin", "des Roses", "du Marché", "Léon Blum", "de Verdun", "Anatole France", "des Acacias"]
# city -> (département, région): major cities, general knowledge
CITIES = [("Paris", "Paris", "Île-de-France"), ("Boulogne-Billancourt", "Hauts-de-Seine", "Île-de-France"),
          ("Lyon", "Rhône", "Auvergne-Rhône-Alpes"), ("Grenoble", "Isère", "Auvergne-Rhône-Alpes"),
          ("Marseille", "Bouches-du-Rhône", "Provence-Alpes-Côte d'Azur"), ("Nice", "Alpes-Maritimes", "Provence-Alpes-Côte d'Azur"),
          ("Toulouse", "Haute-Garonne", "Occitanie"), ("Montpellier", "Hérault", "Occitanie"),
          ("Strasbourg", "Bas-Rhin", "Grand Est"), ("Reims", "Marne", "Grand Est"), ("Metz", "Moselle", "Grand Est"),
          ("Rennes", "Ille-et-Vilaine", "Bretagne"), ("Brest", "Finistère", "Bretagne"),
          ("Rouen", "Seine-Maritime", "Normandie"), ("Caen", "Calvados", "Normandie"),
          ("Dijon", "Côte-d'Or", "Bourgogne-Franche-Comté"), ("Besançon", "Doubs", "Bourgogne-Franche-Comté"),
          ("Tours", "Indre-et-Loire", "Centre-Val de Loire"), ("Orléans", "Loiret", "Centre-Val de Loire"),
          ("Amiens", "Somme", "Hauts-de-France"), ("Lille", "Nord", "Hauts-de-France"),
          ("Bordeaux", "Gironde", "Nouvelle-Aquitaine"), ("Limoges", "Haute-Vienne", "Nouvelle-Aquitaine"),
          ("Nantes", "Loire-Atlantique", "Pays de la Loire"), ("Angers", "Maine-et-Loire", "Pays de la Loire"),
          ("Clermont-Ferrand", "Puy-de-Dôme", "Auvergne-Rhône-Alpes"), ("Le Havre", "Seine-Maritime", "Normandie"),
          ("Saint-Étienne", "Loire", "Auvergne-Rhône-Alpes"), ("Toulon", "Var", "Provence-Alpes-Côte d'Azur"),
          ("Perpignan", "Pyrénées-Orientales", "Occitanie"), ("Poitiers", "Vienne", "Nouvelle-Aquitaine")]
BRAND_SYL = ["zeta", "lyra", "brix", "aria", "nova", "xylo", "cira", "umbra", "delta", "vio", "tavo", "mira", "quo", "lum"]
DIGIT_SWAP = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "b": "8"}
ACCENT = {"é": "e", "è": "e", "ê": "e", "à": "a", "â": "a", "ô": "o", "û": "u", "ç": "c", "î": "i", "É": "E"}


def _name(r):
    kind = r.random()
    if kind < 0.35:
        base = f"{r.choice(BUSINESS)} {r.choice(QUAL)}"
    elif kind < 0.55:
        base = f"{r.choice(CITIES)[0]} {r.choice(BUSINESS)}"          # "<City> Club": generic, collision-prone
    elif kind < 0.75:
        base = f"{r.choice(BUSINESS)} {r.choice(LAST)}"
    elif kind < 0.9:
        base = f"{r.choice(LAST)} & {r.choice(['Fils', 'Frères', 'Associés', r.choice(LAST)])}"
    else:
        base = f"{r.choice(FIRST)} {r.choice(LAST)} {r.choice(BUSINESS)}"
    if r.random() < 0.7:
        base = f"{base} {r.choice(LEGAL)}" if r.random() < 0.8 else f"{r.choice(LEGAL)} {base}"
    return base


def _addr(r):
    num = r.randint(1, 250)
    suffix = r.choice(["", "", "", "", " bis", " ter", " B"])
    stype = r.choices(list(STREET_TYPES), STREET_W)[0]
    city = r.choice(CITIES)
    return {"num": f"{num}{suffix}", "stype": stype, "sname": r.choice(STREET_NAMES), "city": city}


def _fmt_addr(a, r, s1):
    """S1: canonical 'num Type Name, City, Région'. S2/S3: noisy variants."""
    if s1:
        return f"{a['num']} {a['stype']} {a['sname']}, {a['city'][0]}, {a['city'][2]}"
    stype = r.choice(STREET_TYPES[a["stype"]]) if r.random() < 0.3 else a["stype"]
    num = a["num"]
    if r.random() < 0.05:
        num = ""
    elif r.random() < 0.08:
        num = f"N° {num}"
    street = f"{num} {stype} {a['sname']}".strip()
    admin = r.random()
    tail = a["city"][2] if admin < 0.45 else (a["city"][1] if admin < 0.85 else None)   # région / département / none
    parts = [street, a["city"][0]] + ([tail] if tail else [])
    if r.random() < 0.2:
        r.shuffle(parts)
    out = ", ".join(parts)
    if r.random() < 0.35:
        out = out.upper()
    if r.random() < 0.15:
        out = "".join(ACCENT.get(ch, ch) for ch in out)
    return out


def _noisy_name(n, r):
    toks = n.split()
    x = r.random()
    if x < 0.15 and len(toks) > 1:
        r.shuffle(toks)
    n = " ".join(toks)
    for lf in LEGAL:
        if f" {lf}" in f" {n} ":
            y = r.random()
            if y < 0.25:
                n = " ".join(t for t in n.split() if t != lf)
            elif y < 0.4:
                n = n.replace(lf, LEGAL_LONG.get(lf, lf))
            elif y < 0.5:
                n = n.replace(lf, f"({lf})")
            break
    y = r.random()
    if y < 0.08:
        i = r.randrange(len(n))
        n = n[:i] + r.choice(string.ascii_lowercase) + n[i + 1:]
    elif y < 0.14:
        cands = [i for i, ch in enumerate(n) if ch.lower() in DIGIT_SWAP]
        if cands:
            i = r.choice(cands); n = n[:i] + DIGIT_SWAP[n[i].lower()] + n[i + 1:]
    elif y < 0.18:
        n = "".join(ch for ch in n.lower() if ch.isalnum()) + ".fr"
    if r.random() < 0.3:
        n = n.upper()
    elif r.random() < 0.1:
        n = n.lower()
    if r.random() < 0.15:
        n = "".join(ACCENT.get(ch, ch) for ch in n)
    return n


def _brand(r):
    return "".join(r.choice(BRAND_SYL) for _ in range(r.randint(2, 3))).capitalize()


def _decoy_name(n, r):
    letters = [i for i, ch in enumerate(n) if ch.isalpha()]
    i = r.choice(letters)
    return n[:i] + n[i] + n[i:]            # doubled letter


def generate(n_entities=60000, seed=0):
    """Returns list of (text_a, text_b, label). ~3.4 positives / entity, hard + easy negatives."""
    r = random.Random(seed)
    rows = []
    ents = [(_name(r), _addr(r)) for _ in range(n_entities)]
    by_generic = {}
    for i, (nm, a) in enumerate(ents):
        by_generic.setdefault(nm, []).append(i)
    for i, (nm, a) in enumerate(ents):
        s1 = f"{nm} | {_fmt_addr(a, r, True)}"
        k = max(0, min(10, int(r.gauss(3.4, 1.6))))
        for _ in range(k):
            x = r.random()
            if x < 0.06:
                b = f"{_brand(r)} | {_fmt_addr(a, r, False)}"                       # brand name, same address
            elif x < 0.10:
                b = f"{_noisy_name(nm, r)} | "                                       # name only
            else:
                b = f"{_noisy_name(nm, r)} | {_fmt_addr(a, r, False)}"
            rows.append((s1, b, 1.0))
        # hard negatives
        twins = [j for j in by_generic.get(nm, []) if j != i]
        if twins:                                                                   # same name, other business
            j = r.choice(twins)
            rows.append((s1, f"{_noisy_name(ents[j][0], r)} | {_fmt_addr(ents[j][1], r, False)}", 0.0))
        if r.random() < 0.5:                                                        # decoy: doubled letter, same address
            rows.append((s1, f"{_decoy_name(nm, r)} | {_fmt_addr(a, r, False)}", 0.0))
        if r.random() < 0.4:                                                        # decoy: shifted house number
            a2 = dict(a); base = int("".join(ch for ch in a["num"] if ch.isdigit()) or 1)
            a2["num"] = str(max(1, base + r.choice([-4, -2, -1, 1, 2, 3, 6])))
            rows.append((s1, f"{_noisy_name(nm, r)} | {_fmt_addr(a2, r, False)}", 0.0))
        if r.random() < 0.6:                                                        # same street/city, other business
            j = r.randrange(n_entities)
            a3 = dict(ents[j][1]); a3["city"] = a["city"]
            rows.append((s1, f"{_noisy_name(ents[j][0], r)} | {_fmt_addr(a3, r, False)}", 0.0))
        if r.random() < 0.3 and twins:                                              # name-only twin: ambiguous, label 0
            j = r.choice(twins)
            rows.append((s1, f"{_noisy_name(ents[j][0], r)} | ", 0.0))
    r.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# v3: vocabulary harvested from the UNLABELLED test set (cache/fr_vocab.json, organiser-approved, optional) +
#     two noise operators seen in French data: filler words added/swapped at the same address, accent insertion.
# Labels follow TRAINING-measured rules: a filler word added at the same address is the same business (P≈0.95)
# unless it is a decoy word — train decoys {group, holdings, industries, public, enterprises} (P(match)=0.00)
# → French {groupe, groupement, holding, industries, public, publique, entreprises}.
# ---------------------------------------------------------------------------------------------------------------
import json as _json

DECOY_FR = ["Groupe", "Holding", "Industries", "Entreprises", "Public", "Groupement"]
NOISE_FR = ["France", "(France)", "Services", "Développement", "Cie", "& Cie", "International", "Participations",
            "Distribution", "Associés", "& Associés", "Fils", "& Fils", "Frères", "Centre"]
ACCENT_ADD = {"a": "àâ", "c": "ç", "e": "éèê", "o": "ô", "u": "ùû", "i": "î"}
_LEG = {w.lower() for w in LEGAL} | {"s.a.r.l.", "cie", "&"}


def load_vocab(path):
    v = _json.load(open(path))
    words = [w for w, n in v["s1_words"]]
    cat = [w for w in words[:200] if w.isalpha() and w[0].isupper() and w.lower() not in _LEG and len(w) > 2]
    rare = [w for w in words[200:800] if w.isalpha() and len(w) > 2 and w.lower() not in _LEG]
    regions = Counter(r for _, _, r in v["places"])
    top_regions = {r for r, _ in regions.most_common(3)}
    places = [(c, d, r) for c, d, r in v["places"] if r in top_regions and c not in top_regions and d not in top_regions]
    streets = [(t, n) for t, n, c in v["streets"]]
    return {"cat": cat, "rare": rare, "places": places, "streets": streets}


from collections import Counter  # noqa: E402


def _name_v3(r, V):
    x = r.random()
    head = r.choice(V["rare"]) if x < 0.45 else ("".join(r.choice(string.ascii_uppercase) for _ in range(r.randint(2, 3))) if x < 0.7
                                                else r.choice(V["places"])[0])
    parts = [head, r.choice(V["cat"])]
    if r.random() < 0.35:
        parts.append(r.choice(V["cat"]))
    if r.random() < 0.15:
        r.shuffle(parts)
    n = " ".join(parts)
    if r.random() < 0.6:
        n = f"{n} {r.choice(LEGAL)}" if r.random() < 0.85 else f"{r.choice(LEGAL)} {n}"
    return n


def _addr_v3(r, V):
    t, sname = r.choice(V["streets"])
    c = r.choice(V["places"])
    return {"num": f"{r.randint(1, 300)}{r.choice(['', '', '', '', ' bis', ' B', ' ter'])}", "stype": t if t in STREET_TYPES else "Rue",
            "sname": sname, "city": c}


def _accent(n, r):
    idx = [i for i, ch in enumerate(n) if ch.lower() in ACCENT_ADD]
    if not idx:
        return n
    i = r.choice(idx); ch = r.choice(ACCENT_ADD[n[i].lower()])
    return n[:i] + (ch.upper() if n[i].isupper() else ch) + n[i + 1:]


def _noisy_v3(n, r, V):
    n = _noisy_name(n, r)
    x = r.random()
    if x < 0.18:                                                   # filler added (noise → still a match)
        f = r.choice(NOISE_FR); n = f"{n} {f}" if r.random() < 0.8 else f"{f} {n}"
    elif x < 0.26:                                                 # category word swapped (noise → match)
        toks = n.split(); cats = [i for i, t in enumerate(toks) if t in V["cat"]]
        if cats:
            toks[r.choice(cats)] = r.choice(V["cat"]); n = " ".join(toks)
    if r.random() < 0.2:
        n = _accent(n, r)
    return n


def generate_v3(n_entities=60000, seed=0, vocab_path=None):
    """Like generate(), with test-harvested vocabulary + filler/accent operators + decoy-filler negatives."""
    V = load_vocab(vocab_path)
    r = random.Random(seed)
    ents = [(_name_v3(r, V), _addr_v3(r, V)) for _ in range(n_entities)]
    by_name = {}
    for i, (nm, _) in enumerate(ents):
        by_name.setdefault(nm.lower(), []).append(i)
    rows = []
    for i, (nm, a) in enumerate(ents):
        s1 = f"{nm} | {_fmt_addr(a, r, True)}"
        for _ in range(max(0, min(10, int(r.gauss(3.4, 1.6))))):
            x = r.random()
            if x < 0.06:
                b = f"{_brand(r)} | {_fmt_addr(a, r, False)}"
            elif x < 0.10:
                b = f"{_noisy_v3(nm, r, V)} | "
            else:
                b = f"{_noisy_v3(nm, r, V)} | {_fmt_addr(a, r, False)}"
            rows.append((s1, b, 1.0))
        twins = [j for j in by_name.get(nm.lower(), []) if j != i]
        if twins:
            j = r.choice(twins); rows.append((s1, f"{_noisy_v3(ents[j][0], r, V)} | {_fmt_addr(ents[j][1], r, False)}", 0.0))
        if r.random() < 0.45:                                      # decoy filler at the same address (non-match)
            d = r.choice(DECOY_FR); rows.append((s1, f"{nm} {d} | {_fmt_addr(a, r, False)}" if r.random() < 0.8 else f"{d} {nm} | {_fmt_addr(a, r, False)}", 0.0))
        if r.random() < 0.4:
            rows.append((s1, f"{_decoy_name(nm, r)} | {_fmt_addr(a, r, False)}", 0.0))
        if r.random() < 0.4:
            a2 = dict(a); base = int("".join(ch for ch in a["num"] if ch.isdigit()) or 1)
            a2["num"] = str(max(1, base + r.choice([-4, -2, -1, 1, 2, 3, 6]))); rows.append((s1, f"{_noisy_v3(nm, r, V)} | {_fmt_addr(a2, r, False)}", 0.0))
        if r.random() < 0.6:
            j = r.randrange(n_entities); a3 = dict(ents[j][1]); a3["city"] = a["city"]
            rows.append((s1, f"{_noisy_v3(ents[j][0], r, V)} | {_fmt_addr(a3, r, False)}", 0.0))
        if r.random() < 0.3 and twins:
            rows.append((s1, f"{_noisy_v3(ents[r.choice(twins)][0], r, V)} | ", 0.0))
    r.shuffle(rows)
    return rows
