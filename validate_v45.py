"""Validation rapide de la construction du prompt V4.5."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import eval_from_excel as evmod
evmod.JUDGE_MODEL = 'gpt-4o'
evmod._openai_client = None
from eval_from_excel import _JUDGE_V43_SYSTEM

# ── V4.4 changes ─────────────────────────────────────────────────────────────
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

assert _RULE3_V43 in _JUDGE_V43_SYSTEM
assert _STEP3_V43 in _JUDGE_V43_SYSTEM
assert _MISLABELLING_JSON_V43 in _JUDGE_V43_SYSTEM

_JUDGE_V44_SYSTEM = (
    _JUDGE_V43_SYSTEM
    .replace(_RULE3_V43, _RULE3_V44)
    .replace(_STEP3_V43, _STEP3_V44)
    .replace(_MISLABELLING_JSON_V43, _MISLABELLING_JSON_V44)
)

# ── V4.5 : ajout Exemple 9 ───────────────────────────────────────────────────
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

# ── Validations ──────────────────────────────────────────────────────────────
assert "EXEMPLE 9" in _JUDGE_V45_SYSTEM
assert _FINAL_MARKER in _JUDGE_V45_SYSTEM
assert _RULE3_V44 in _JUDGE_V45_SYSTEM
assert _STEP3_V44 in _JUDGE_V45_SYSTEM
assert _RULE3_V43 not in _JUDGE_V45_SYSTEM, "Règle 3 V4.3 ne devrait plus être présente"

print("OK — V4.3 :", len(_JUDGE_V43_SYSTEM), "chars")
print("OK — V4.4 :", len(_JUDGE_V44_SYSTEM), "chars", f"(+{len(_JUDGE_V44_SYSTEM)-len(_JUDGE_V43_SYSTEM)})")
print("OK — V4.5 :", len(_JUDGE_V45_SYSTEM), "chars", f"(+{len(_JUDGE_V45_SYSTEM)-len(_JUDGE_V44_SYSTEM)} vs V4.4)")
print()
print("Vérification Exemple 9 dans V4.5 :")
idx = _JUDGE_V45_SYSTEM.find("EXEMPLE 9")
print(" ", _JUDGE_V45_SYSTEM[idx:idx+80])
print()
print("Vérification position par rapport au marqueur final :")
idx_ex  = _JUDGE_V45_SYSTEM.find("EXEMPLE 9")
idx_fin = _JUDGE_V45_SYSTEM.find(_FINAL_MARKER)
print(f"  EXEMPLE 9 à char {idx_ex}, FINAL_MARKER à char {idx_fin} → exemple avant marqueur : {idx_ex < idx_fin}")
print()
print("Toutes les assertions passées. V4.5 prêt.")
