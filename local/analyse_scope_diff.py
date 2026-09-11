"""
Diff scope v10 (_is_global_q) vs v12 (_detect_scope_and_type) — avec commune_detector.
Répartition question_type complète sur les 109 questions.
À lancer depuis c:\These\Données2\fichiers_pour_rag\
"""

import json, re, sys, unicodedata, pathlib, os
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Charger les 109 questions ─────────────────────────────────────────────────

RERUN2 = ROOT / "comparaisons_rag" / "ablations_103q_v43_gpt4o_RERUN2_20260816_130113.json"
with open(RERUN2, encoding="utf-8") as f:
    data = json.load(f)
first_config = next(iter(data.values()))
questions = sorted(set(e["question"] for e in first_config))
print(f"{len(questions)} questions chargées.\n")

# ── commune_detector ───────────────────────────────────────────────────────────

try:
    from commune_detector import detect_communes
    HAS_DETECTOR = True
    print("commune_detector disponible — utilisation réelle.\n")
except ImportError:
    HAS_DETECTOR = False
    detect_communes = lambda q: []
    print("commune_detector ABSENT — communes=[] pour toutes les questions.\n")

# ── Logique v10 _is_global_q (lignes 818-822 reproduites exactement) ──────────

_GLOBAL_KW = (
    "moyen", "moyenne", "general", "global", "ensemble", "niveau",
    "corse entiere", "ile entiere", "l ensemble", "toutes les communes",
    "score global", "score corse", "indicateur corse",
)

def _norm(q):
    return "".join(
        c for c in unicodedata.normalize("NFD", q.lower())
        if unicodedata.category(c) != "Mn"
    )

def is_global_v10(question, communes):
    q_norm = _norm(question)
    return not communes and any(kw in q_norm for kw in _GLOBAL_KW)

# ── Logique v12 ────────────────────────────────────────────────────────────────

_RE_FAISABILITE = re.compile(
    r"peut[- ]on\b|est[- ]il possible de|"
    r"les donn[eé]es permettent[- ]elles|est[- ]ce possible",
    re.IGNORECASE,
)
_RE_EXISTENCE = re.compile(
    r"existe[- ]t[- ]il des donn[eé]es|"
    r"avez[- ]vous des donn[eé]es|"
    r"y a[- ]t[- ]il des (?:donn[eé]es|informations)",
    re.IGNORECASE,
)

def detect_scope_v12(question, communes):
    q_norm = _norm(question)
    if not communes and any(kw in q_norm for kw in _GLOBAL_KW):
        return "globale"
    elif communes:
        return "locale"
    return "thematique"

def detect_type_v12(question):
    if _RE_FAISABILITE.search(question):
        return "faisabilite"
    if _RE_EXISTENCE.search(question):
        return "existence"
    return "factuelle"

# ── Diff scope ────────────────────────────────────────────────────────────────

print("=" * 70)
print("DIFF scope v10 (_is_global_q) vs v12 (_detect_scope_and_type)")
print(f"commune_detector={'actif' if HAS_DETECTOR else 'ABSENT (invalide)'}")
print("=" * 70)

diffs = []
for q in questions:
    communes = detect_communes(q)
    v10_global = is_global_v10(q, communes)
    v12_scope  = detect_scope_v12(q, communes)
    v10_coarse = "globale" if v10_global else "non-globale"
    v12_coarse = "globale" if v12_scope == "globale" else "non-globale"
    if v10_coarse != v12_coarse:
        diffs.append((q, communes, v10_coarse, v12_scope))

if diffs:
    print(f"\n⚠️  {len(diffs)} écart(s) détecté(s) :")
    for q, comm, v10s, v12s in diffs:
        print(f"  communes={comm}")
        print(f"  v10={v10s!r}  v12={v12s!r}")
        print(f"  └ {q}\n")
else:
    print(f"\n✅  Zéro écart sur {len(questions)} questions.")

# ── Détail : questions locale vs non-locale avec communes détectées ───────────
locale_qs = [(q, detect_communes(q)) for q in questions if detect_communes(q)]
print(f"\n{len(locale_qs)} questions avec commune(s) détectée(s) :")
for q, comm in locale_qs[:20]:
    print(f"  {comm} → {q[:80]}")
if len(locale_qs) > 20:
    print(f"  ... et {len(locale_qs)-20} autres")

# ── Répartition question_type ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RÉPARTITION question_type (commune_detector=" + ("actif" if HAS_DETECTOR else "ABSENT") + ")")
print("=" * 70)

counts = Counter(detect_type_v12(q) for q in questions)
for t, n in sorted(counts.items()):
    print(f"  {t:<15} : {n}")

print("\n--- Questions classées 'faisabilite' ---")
for q in questions:
    if detect_type_v12(q) == "faisabilite":
        print(f"  • {q}")

print("\n--- Questions classées 'existence' ---")
for q in questions:
    if detect_type_v12(q) == "existence":
        print(f"  • {q}")

print("\n--- Tableau complet (scope + type) ---")
for q in sorted(questions):
    communes = detect_communes(q)
    scope = detect_scope_v12(q, communes)
    qtype = detect_type_v12(q)
    comm_str = ",".join(communes) if communes else "—"
    print(f"  [{scope:<10}] [{qtype:<12}] [{comm_str:<12}] {q}")
