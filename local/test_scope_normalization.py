"""
test_scope_normalization.py — Mesure la portée du bug de normalisation diacritique.

Objectif :
  1. Combien de communes du gazetteer (360 noms) avaient un nom avec diacritiques
     qui les rendait indétectables dans des textes les mentionnant sans accent ?
  2. Sur les 109 questions d'évaluation, combien changeaient de liste de communes
     détectées avant/après le fix ?

Exécution : python local/test_scope_normalization.py
"""
import sys, os, json, unicodedata, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.chdir(str(ROOT))

# ── Importer les deux versions ────────────────────────────────────────────────
# On importe le module tel quel (après fix).
from commune_detector import detect_communes, _normalize_str, _COMMUNE_NAMES

print(f"Gazetteer : {len(_COMMUNE_NAMES)} communes chargées")


# ── 1. Communes avec diacritiques dans le nom ─────────────────────────────────
def has_diacritic(s: str) -> bool:
    return s != _normalize_str(s)

communes_with_diacritics = [c for c in _COMMUNE_NAMES if has_diacritic(c)]
print(f"\n[1] Communes avec diacritiques : {len(communes_with_diacritics)}/{len(_COMMUNE_NAMES)}")

# Simulation du bug AVANT fix : detection sans normalisation
def detect_communes_buggy(text: str) -> list[str]:
    """Reproduce le comportement avant fix (lowercase only, pas de normalisation)."""
    if not _COMMUNE_NAMES:
        return []
    text_lower = text.lower()
    found, seen = [], set()
    for commune_name in sorted(_COMMUNE_NAMES, key=len, reverse=True):
        if commune_name in seen:
            continue
        commune_lower = commune_name.lower()
        pattern = re.escape(commune_lower)
        pattern = pattern.replace(r'\-', r'[\s\-]?')
        pattern = pattern.replace(r"\'", r"[\s']?")
        if re.search(r'\b' + pattern + r'\b', text_lower):
            seen.add(commune_name)
            found.append(commune_name)
    return found

# Test : chaque commune mentionnée dans un texte simple, avec et sans diacritiques
missed_before = []
for c in communes_with_diacritics:
    text = f"Situation à {c}"                         # texte avec diacritiques
    text_no_acc = f"Situation a {_normalize_str(c)}"  # texte sans diacritiques
    before_with = detect_communes_buggy(text)
    before_without = detect_communes_buggy(text_no_acc)
    after_with = detect_communes(text)
    after_without = detect_communes(text_no_acc)
    # Was the commune missed by the buggy version ?
    if c not in before_with or c not in before_without:
        missed_before.append((c, bool(c in before_with), bool(c in before_without)))

print(f"   → communes indétectables (avec accents dans texte)  avant fix : {sum(1 for _,a,_ in missed_before if not a)}")
print(f"   → communes indétectables (sans accents dans texte)  avant fix : {sum(1 for _,_,b in missed_before if not b)}")
if missed_before:
    print("   Exemples :")
    for c, wa, wo in missed_before[:8]:
        print(f"     {c!r:35s}  avec_acc={wa} sans_acc={wo}")


# ── 2. Impact sur les 109 questions d'évaluation ─────────────────────────────
EVAL_FILE = ROOT / "comparaisons_rag" / "COMPLET.json"
if not EVAL_FILE.exists():
    # Chercher un fichier d'évaluation alternatif
    alts = list((ROOT / "comparaisons_rag").glob("*.json"))
    EVAL_FILE = next((f for f in alts if "COMPLET" in f.name or "109" in f.name), None)

questions_changed = []
if EVAL_FILE and EVAL_FILE.exists():
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    # Extraire les questions
    all_questions = []
    if isinstance(data, list):
        all_questions = [d.get("question", d.get("q", "")) for d in data if isinstance(d, dict)]
    elif isinstance(data, dict):
        # Peut être structuré différemment
        for v in data.values():
            if isinstance(v, dict) and "question" in v:
                all_questions.append(v["question"])
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "question" in item:
                        all_questions.append(item["question"])

    print(f"\n[2] Questions d'évaluation trouvées : {len(all_questions)}")

    for q in all_questions:
        before = detect_communes_buggy(q)
        after  = detect_communes(q)
        if set(before) != set(after):
            questions_changed.append((q[:80], before, after))

    print(f"   → questions avec scope changé après fix : {len(questions_changed)}/{len(all_questions)}")
    if questions_changed:
        for q, b, a in questions_changed[:10]:
            print(f"     [{b!r:20s} → {a!r:20s}]  {q}")
else:
    print(f"\n[2] Fichier d'évaluation non trouvé : {EVAL_FILE}")


# ── 3. Bilan ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Communes avec diacritiques                    : {len(communes_with_diacritics)}")
print(f"Communes que le bug rendait indétectables     : {len(missed_before)}")
print(f"Questions d'éval dont le scope change         : {len(questions_changed)}")
print("=" * 70)
