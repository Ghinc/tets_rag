"""
Test du juge V4.4 sur Q46, Q49, Q50, Q55 (toutes configs).
Compare avec les scores V4.3 existants dans ablations_103q_v43_gpt4o_COMPLET.json.

V4.4 = V4.3 + correctif Règle 3 uniquement :
  - Règle 3 reformulée : 4 conditions strictes (mislabelling silencieux uniquement)
  - Ajout Étape 3.bis : scan exhaustif des signalements avant de déclencher R3
  - Nouveau champ JSON : signalements_detectes
"""
import json, re, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Forcer GPT-4o comme backend ──────────────────────────────────────────────
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import _JUDGE_V43_SYSTEM, _call_llm, _build_sources_text

# ════════════════════════════════════════════════════════════════════════════
# Prompt V4.4 : V4.3 avec Règle 3 reformulée + Étape 3.bis + champ JSON
# ════════════════════════════════════════════════════════════════════════════

_RULE3_V43 = """\
### Règle 3 — Absence signalée de source qualitative

Si la question demande un croisement quali/quanti (perceptions vs
indicateurs, ressenti vs structurel) ET qu'aucune source qualitative
(verbatim, entretien, raptor_quali) n'apparaît dans les sources
fournies au système :

- Si la réponse signale explicitement le manque ("aucune donnée
  qualitative disponible pour ce sous-groupe") → comportement
  acceptable, pas de pénalité
- Si la réponse ne signale pas le manque et présente quand même un
  croisement (en renommant le quanti en quali, par exemple) → ÉCHEC
  MAJEUR :
  - `pertinence` ≤ 3
  - `fondement_factuel` ≤ 3
  - `nuance_incertitude` ≤ 2"""

_RULE3_V44 = """\
### Règle 3 — Mislabelling silencieux uniquement

Cette règle pénalise UNIQUEMENT le mislabelling silencieux : présenter
des données quantitatives comme qualitatives sans aucune mention de
la limite.

Conditions de déclenchement strictes :
- (1) La question demande un croisement quali/quanti OU des perceptions
- (2) Aucune source qualitative dans les sources fournies au système
- (3) La réponse présente des données quantitatives en les appelant
      qualitatives (ex : "voici les dimensions qualitatives OppChoVec")
- (4) AUCUN signalement de l'absence de données qualitatives nulle
      part dans la réponse (ni en titre, ni dans le texte, ni dans
      un tableau, ni en synthèse)

Les 4 conditions doivent être réunies pour déclencher la pénalité.

Si la réponse signale le manque de données qualitatives DE QUELQUE
MANIÈRE QUE CE SOIT (titre de section "Données manquantes", phrase
"aucune donnée disponible", tableau marquant "Inconnu", mention "ne
permettent pas de"), même brève, la Règle 3 NE S'APPLIQUE PAS. La
réponse mobilise alors des données contextuelles avec transparence,
ce qui est conforme au Principe 2 (transparence rachète l'incomplétude).

Pénalités si Règle 3 déclenchée (4 conditions réunies) :
- `pertinence` ≤ 3
- `fondement_factuel` ≤ 3
- `nuance_incertitude` ≤ 2"""

_STEP3_V43 = """\
**Étape 3** : confronte les affirmations aux sources. VÉRIFIE les
4 règles anti-mislabelling :
- La réponse renomme-t-elle une source quanti en quali ?
- La réponse surinterprète-t-elle Vec/Cho/Opp ?
- La question demande-t-elle un croisement quali/quanti sans source
  quali fournie ?
- La réponse extrapole-t-elle OppChoVec à un sous-groupe ?"""

_STEP3_V44 = """\
**Étape 3** : confronte les affirmations aux sources. VÉRIFIE les
4 règles anti-mislabelling :
- La réponse renomme-t-elle une source quanti en quali ?
- La réponse surinterprète-t-elle Vec/Cho/Opp ?
- La question demande-t-elle un croisement quali/quanti sans source
  quali fournie ?
- La réponse extrapole-t-elle OppChoVec à un sous-groupe ?

**Étape 3.bis — Recherche EXHAUSTIVE des signalements d'absence**

Si la question demande des perceptions ou un croisement quali/quanti
et que les sources contiennent uniquement du quantitatif, AVANT de
déclencher la Règle 3, scanne la réponse entière pour identifier
TOUT signalement d'absence de données qualitatives, sous N'IMPORTE
QUELLE forme :

- Titres de section ("Données manquantes", "Inconnu pour X")
- Phrases dans le texte ("aucune donnée d'enquête disponible", "ne
  permettent pas de répondre directement", "se limite aux indicateurs
  objectifs")
- Cellules de tableau ("Inconnu", "Données indisponibles", "—")
- Recommandations finales ("une enquête locale serait nécessaire")
- Modalisateurs explicites ("ce qui suggère", "pourrait", "il
  conviendrait de confirmer")

Un SEUL signalement clair suffit à ne pas déclencher la Règle 3.

Documente dans le champ `signalements_detectes` les signalements
trouvés (ou leur absence)."""

_MISLABELLING_JSON_V43 = """\
  "mislabelling_detecte": {
    "regle_1_quali_quanti": "non | oui — <détail>",
    "regle_2_surinterpretation_oppchovec": "non | oui — <détail>",
    "regle_3_absence_quali_non_signalee": "non | oui — <détail>",
    "regle_4_extrapolation_sous_groupe": "non | oui — <détail>"
  },"""

_MISLABELLING_JSON_V44 = """\
  "mislabelling_detecte": {
    "regle_1_quali_quanti": "non | oui — <détail>",
    "regle_2_surinterpretation_oppchovec": "non | oui — <détail>",
    "regle_3_absence_quali_non_signalee": "non | oui — <détail>",
    "regle_4_extrapolation_sous_groupe": "non | oui — <détail>"
  },
  "signalements_detectes": [
    "<liste des signalements d'absence de données quali trouvés dans la réponse, ou [] si aucun>"
  ],"""

assert _RULE3_V43 in _JUDGE_V43_SYSTEM, "Règle 3 V4.3 introuvable dans le prompt !"
assert _STEP3_V43 in _JUDGE_V43_SYSTEM, "Étape 3 V4.3 introuvable dans le prompt !"
assert _MISLABELLING_JSON_V43 in _JUDGE_V43_SYSTEM, "JSON mislabelling V4.3 introuvable !"

_JUDGE_V44_SYSTEM = (
    _JUDGE_V43_SYSTEM
    .replace(_RULE3_V43, _RULE3_V44)
    .replace(_STEP3_V43, _STEP3_V44)
    .replace(_MISLABELLING_JSON_V43, _MISLABELLING_JSON_V44)
)

# ════════════════════════════════════════════════════════════════════════════
# Prompt V4.5 : V4.4 + Exemple 9 (comparaison avec données partiellement absentes)
# ════════════════════════════════════════════════════════════════════════════

_EXAMPLE9 = """
--- EXEMPLE 9 : Comparaison avec données partiellement absentes signalées ---

QUESTION (fictive) : Comment se compare la satisfaction concernant les
transports entre Bastia et Piedicorte-di-Gaggio ?

SECTION : Raisonnement comparatif / Croisé quanti/quali

SOURCES FOURNIES (fictives) : oppchovec:Bastia (score mobilité 7.1),
raptor:commune_quali:Bastia (verbatims sur transports), oppchovec:Piedicorte
(score mobilité 8.9). Aucun verbatim ni entretien pour Piedicorte.

RÉPONSE (extrait) : "À Bastia, les habitants expriment une satisfaction
modérée concernant les transports : si la desserte urbaine est jugée
correcte, plusieurs verbatims pointent une saturation aux heures de pointe
et des difficultés de stationnement. Les indicateurs territoriaux
confirment ce constat nuancé (score mobilité 7.1/10).

Concernant Piedicorte-di-Gaggio, les indicateurs objectifs sont
disponibles (score mobilité 8.9/10, lié à la position géographique et
à l'accès routier). En revanche, aucune donnée d'enquête citoyenne
n'est disponible pour cette commune, ce qui empêche d'évaluer la
satisfaction perçue par les habitants. La comparaison entre les deux
communes ne peut donc être menée que sur le plan des indicateurs
territoriaux ; les perceptions resteront à confirmer par une enquête
locale à Piedicorte."

NOTATION :
{
  "section_observee": "Raisonnement comparatif / Croisé quanti/quali",
  "comportement_attendu_selon_grille": "Comparaison effectuée sur les éléments disponibles avec signalement explicite des manques",
  "type_reponse_attendue_specifie": "reponse_substantielle_attendue",
  "coherence_comportement_observe": "oui",
  "sources_inventaire": [
    {"source": "oppchovec:Bastia", "type": "quanti"},
    {"source": "raptor:commune_quali:Bastia", "type": "quali"},
    {"source": "oppchovec:Piedicorte", "type": "quanti"}
  ],
  "signalements_detectes": [
    "aucune donnée d'enquête citoyenne n'est disponible pour cette commune",
    "ce qui empêche d'évaluer la satisfaction perçue par les habitants",
    "la comparaison ne peut donc être menée que sur le plan des indicateurs territoriaux",
    "les perceptions resteront à confirmer par une enquête locale"
  ],
  "mislabelling_detecte": {
    "regle_1_quali_quanti": "non",
    "regle_2_surinterpretation_oppchovec": "non",
    "regle_3_absence_quali_non_signalee": "non — signalements multiples et explicites de l'absence de données quali pour Piedicorte",
    "regle_4_extrapolation_sous_groupe": "non"
  },
  "elements_specifiques_question": ["transports", "Bastia", "Piedicorte-di-Gaggio", "satisfaction", "comparaison"],
  "elements_traitement": [
    {"element": "transports Bastia", "traitement": "precis"},
    {"element": "transports Piedicorte", "traitement": "approximation_signalee"},
    {"element": "satisfaction perçue Piedicorte", "traitement": "omis — absence signalée"}
  ],
  "raisonnement": "Comparaison partielle menée honnêtement : Bastia quanti+quali, Piedicorte quanti seul avec signalement explicite du manque quali. Règle 3 non déclenchée.",
  "pertinence": {"note": 4, "justification": "Comparaison correctement menée sur les données disponibles, avec délimitation honnête du périmètre comparable."},
  "fondement_factuel": {"note": 5, "justification": "Affirmations bien sourcées sur Bastia, périmètre des données Piedicorte explicité, aucune invention."},
  "nuance_incertitude": {"note": 5, "justification": "Excellent signalement des limites : absence quali pour Piedicorte mentionnée plusieurs fois, recommandation d'enquête locale."},
  "coherence_qualiquanti": {"note": 4, "justification": "Mobilisation appropriée des deux familles pour Bastia, restriction transparente au seul quanti pour Piedicorte."}
}

PRINCIPE ILLUSTRÉ : quand une comparaison porte sur deux entités dont
l'une manque de données qualitatives, le bon comportement est de
mener la comparaison sur les éléments disponibles ET de signaler
explicitement, à plusieurs reprises dans la réponse, ce qui ne peut
pas être comparé. La Règle 3 (mislabelling silencieux) NE doit PAS
être déclenchée tant que les signalements sont présents, même
dispersés dans le texte ou dans des sections différentes.
"""

_FINAL_MARKER = "=== MAINTENANT, ÉVALUE LA RÉPONSE SUIVANTE ==="
assert _FINAL_MARKER in _JUDGE_V44_SYSTEM, "Marqueur final introuvable dans V4.4 !"

_JUDGE_V45_SYSTEM = _JUDGE_V44_SYSTEM.replace(
    _FINAL_MARKER,
    _EXAMPLE9 + "\n" + _FINAL_MARKER
)

print("✓ Prompt V4.4 construit")
print(f"  Taille V4.3 : {len(_JUDGE_V43_SYSTEM)} chars")
print(f"  Taille V4.4 : {len(_JUDGE_V44_SYSTEM)} chars")
print(f"  Delta      : +{len(_JUDGE_V44_SYSTEM)-len(_JUDGE_V43_SYSTEM)} chars")
print(f"✓ Prompt V4.5 construit")
print(f"  Taille V4.5 : {len(_JUDGE_V45_SYSTEM)} chars")
print(f"  Delta V4.4→V4.5 : +{len(_JUDGE_V45_SYSTEM)-len(_JUDGE_V44_SYSTEM)} chars")

# ── Parser V4.4 ──────────────────────────────────────────────────────────────

def _parse_judge_v44(j: dict) -> dict:
    def note(d):
        if isinstance(d, dict):
            return d.get("note")
        return d if isinstance(d, (int, float)) else None

    p  = note(j.get("pertinence"))
    ff = note(j.get("fondement_factuel"))
    ni = note(j.get("nuance_incertitude"))
    qq = note(j.get("coherence_qualiquanti"))

    scores = [x for x in [p, ff, ni, qq] if x is not None]
    sg = round(sum(scores) / len(scores), 4) if scores else None

    mis = j.get("mislabelling_detecte", {})
    mislabelling_flag = any(
        str(v).lower() not in ("non", "false", "", "null", "none")
        for v in mis.values()
    )

    return {
        "section_observee":                  j.get("section_observee", ""),
        "comportement_attendu_selon_grille":  j.get("comportement_attendu_selon_grille", ""),
        "type_reponse_attendue_specifie":     j.get("type_reponse_attendue_specifie", ""),
        "coherence_comportement_observe":     j.get("coherence_comportement_observe", ""),
        "sources_inventaire":                 j.get("sources_inventaire", []),
        "mislabelling_detecte":               mis,
        "mislabelling_flag":                  mislabelling_flag,
        "signalements_detectes":              j.get("signalements_detectes", []),
        "elements_specifiques_question":      j.get("elements_specifiques_question", []),
        "elements_traitement":                j.get("elements_traitement", []),
        "raisonnement":                       j.get("raisonnement", ""),
        "pertinence":              p,
        "pertinence_justif":       j.get("pertinence", {}).get("justification", "") if isinstance(j.get("pertinence"), dict) else "",
        "fondement_factuel":       ff,
        "fondement_factuel_justif":j.get("fondement_factuel", {}).get("justification", "") if isinstance(j.get("fondement_factuel"), dict) else "",
        "nuance_incertitude":      ni,
        "nuance_incertitude_justif":j.get("nuance_incertitude", {}).get("justification", "") if isinstance(j.get("nuance_incertitude"), dict) else "",
        "coherence_qualiquanti":   qq,
        "coherence_qualiquanti_justif":j.get("coherence_qualiquanti", {}).get("justification", "") if isinstance(j.get("coherence_qualiquanti"), dict) else "",
        "score_global":            sg,
        "judge_error":             None,
    }


def score_judge_v44(question, answer, sources, section, subsection, expected_type):
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer}\n\n"
        "Évalue cette réponse selon la procédure V4.4 et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter."
    )
    try:
        raw = _call_llm(_JUDGE_V44_SYSTEM, user_prompt, max_tokens=3500, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v44(j)
        return result
    except Exception as e:
        return {"judge_error": str(e), "score_global": None}


# ── Charger les données existantes ───────────────────────────────────────────

with open('comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json', encoding='utf-8') as f:
    complet = json.load(f)

CONFIGS = ['v_vanilla_k10', 'v_vanilla_k25', 'v_decomp', 'v_decomp_raptor']
TARGET_ROWS = [46, 49, 50, 55]

# Regroup by excel_row
by_row = {}
for ver, entries in complet.items():
    for e in entries:
        row = e['excel_row']
        if row not in by_row:
            by_row[row] = {}
        by_row[row][ver] = e

# ── Lancer V4.4 ──────────────────────────────────────────────────────────────

results_v44 = {}  # {row: {config: judge_result}}
total = len(TARGET_ROWS) * len(CONFIGS)
done = 0

print(f"\nLancement V4.4 — {total} appels juge\n")
print("=" * 70)

for row in TARGET_ROWS:
    results_v44[row] = {}
    vmap = by_row.get(row, {})
    ref  = next(iter(vmap.values()), {})
    q    = ref.get('question', '')
    sec  = ref.get('section', '')
    sub  = ref.get('subsection', '')
    exp_type = ref.get('type_reponse_attendue_specifie', 'reponse_substantielle_attendue') or 'reponse_substantielle_attendue'

    print(f"\nQ{row} — {q[:70]}")
    print(f"  Section : {sec} / {sub}")

    for cfg in CONFIGS:
        e = vmap.get(cfg, {})
        answer  = e.get('answer', '') or ''
        sources = e.get('sources', []) or []

        if not answer:
            print(f"  [{cfg}] aucune réponse, skip")
            results_v44[row][cfg] = None
            continue

        j44 = score_judge_v44(q, answer, sources, sec, sub, exp_type)
        results_v44[row][cfg] = j44
        done += 1

        sg_v44 = j44.get('score_global')
        sg_v43 = e.get('score_global')
        mis_v44 = '✗MIS' if j44.get('mislabelling_flag') else ''
        mis_v43 = '✗MIS' if e.get('mislabelling_flag') else ''
        delta = (f"{sg_v44 - sg_v43:+.2f}" if sg_v44 is not None and sg_v43 is not None else "n/a")
        sigs  = j44.get('signalements_detectes', [])
        sig_s = f"  signalem: {sigs[:2]}" if sigs else "  (aucun signalement détecté)"

        print(f"  [{cfg:20s}]  V4.3={sg_v43} {mis_v43}  V4.4={sg_v44} {mis_v44}  Δ={delta}")
        print(f"  {' ':22s}  {sig_s}")
        print(f"  {' ':22s}  R3_v44={j44.get('mislabelling_detecte', {}).get('regle_3_absence_quali_non_signalee','?')[:60]}")

# ── Sauvegarder ──────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = f'comparaisons_rag/test_v44_{ts}.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'metadata': {'target_rows': TARGET_ROWS, 'configs': CONFIGS, 'judge': 'v4.4', 'timestamp': ts},
        'results':  {str(k): v for k, v in results_v44.items()}
    }, f, ensure_ascii=False, indent=2)
print(f"\n\nSauvegardé → {out_path}")

# ── Résumé comparatif ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RÉSUMÉ COMPARATIF V4.3 → V4.4")
print("=" * 70)
print(f"{'Q':>3}  {'Config':20s}  {'V4.3':>5} {'MIS43':5}  {'V4.4':>5} {'MIS44':5}  {'Δ':>6}")
print("-" * 70)
for row in TARGET_ROWS:
    vmap = by_row.get(row, {})
    ref  = next(iter(vmap.values()), {})
    print(f" Q{row}  {ref.get('question','')[:55]}")
    for cfg in CONFIGS:
        e43   = vmap.get(cfg, {})
        e44   = results_v44[row].get(cfg) or {}
        sg43  = e43.get('score_global')
        sg44  = e44.get('score_global')
        mis43 = '✗' if e43.get('mislabelling_flag') else ' '
        mis44 = '✗' if e44.get('mislabelling_flag') else ' '
        delta = (f"{sg44 - sg43:+.2f}" if sg44 is not None and sg43 is not None else " n/a")
        print(f"     {cfg:20s}  {str(sg43):>5} {mis43:5}  {str(sg44):>5} {mis44:5}  {delta:>6}")
    print()
