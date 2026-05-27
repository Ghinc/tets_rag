"""
Ajoute 3 nouvelles questions 'Limites architecturales' (Q104-106) au fichier COMPLET.
Juge avec V4.4.
"""
import json, re, sys, io, time, requests
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Juge V4.4 ────────────────────────────────────────────────────────────────
import importlib
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm

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

_JUDGE_V44_SYSTEM = (
    _JUDGE_V43_SYSTEM
    .replace(_RULE3_V43, _RULE3_V44)
    .replace(_STEP3_V43, _STEP3_V44)
    .replace(_MISLABELLING_JSON_V43, _MISLABELLING_JSON_V44)
)
assert _RULE3_V44 in _JUDGE_V44_SYSTEM
print(f"V4.4 prêt ({len(_JUDGE_V44_SYSTEM)} chars)")

# ── Nouvelles questions ───────────────────────────────────────────────────────
NEW_QUESTIONS = [
    {
        "excel_row": 104,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Pour quelles communes corses peut-on croiser à la fois des données d'enquête citoyenne, des entretiens semi-directifs, et un score OppChoVec complet ?",
    },
    {
        "excel_row": 105,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Classer les 10 communes corses ayant les écarts les plus marqués entre indicateurs objectifs et perceptions subjectives.",
    },
    {
        "excel_row": 106,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Quelles sont les communes corses qui présentent un score Vec inférieur à 3/10 et où les habitants expriment néanmoins une satisfaction élevée concernant leur cadre de vie ?",
    },
    {
        "excel_row": 107,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Existe-t-il une corrélation entre la qualité du logement et la satisfaction des habitants à l'échelle des communes corses ?",
    },
    {
        "excel_row": 108,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Sur l'ensemble du corpus, comment l'âge des répondants influence-t-il leur perception du bien-être territorial ?",
    },
    {
        "excel_row": 109,
        "section": "Limites architecturales",
        "subsection": "",
        "question": "Dans quelle commune les 18-25 ans se sentent-ils le mieux ?",
    },
]

CONFIGS = {
    "v_vanilla_k10":   10,
    "v_vanilla_k25":   25,
    "v_decomp":        5,
    "v_decomp_raptor": 5,
}
BASE = "http://localhost:8000/api/query"


def call_rag(question: str, version: str, k: int) -> dict:
    try:
        r = requests.post(BASE, json={"question": question, "rag_version": version, "k": k},
                          timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "answer": f"ERREUR: {e}", "sources": []}


def judge_v44(question, answer, sources, section, subsection):
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : limite_architecturale\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:4000]}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Réponds UNIQUEMENT avec le JSON demandé, sans texte avant ni après."
    )
    t0 = time.time()
    try:
        raw = _call_llm(_JUDGE_V44_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["judge_error"] = None
        result["judge_elapsed_s"] = round(time.time() - t0, 1)
        result["signalements_detectes"] = j.get("signalements_detectes", [])
        return result
    except Exception as e:
        return {"judge_error": str(e), "score_global": None,
                "judge_elapsed_s": round(time.time() - t0, 1)}


# ── Chargement du fichier existant ───────────────────────────────────────────
COMPLET = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
with open(COMPLET, encoding="utf-8") as f:
    results = json.load(f)

# ── Boucle principale ─────────────────────────────────────────────────────────
# Ne traiter que les questions absentes du fichier
existing_rows = {e['excel_row'] for e in results['v_vanilla_k10']}
questions_to_run = [q for q in NEW_QUESTIONS if q['excel_row'] not in existing_rows]

print(f"\n{'='*65}")
print(f"Lancement — {len(questions_to_run)} nouvelles questions × {len(CONFIGS)} configs")
print(f"(déjà présentes : {sorted(existing_rows & {q['excel_row'] for q in NEW_QUESTIONS})})")
print('='*65)

for q_meta in questions_to_run:
    print(f"\nQ{q_meta['excel_row']} — {q_meta['question'][:70]}")
    for ver, k in CONFIGS.items():
        print(f"  [{ver:<20s}] RAG...", end="", flush=True)
        t0 = time.time()
        rag = call_rag(q_meta["question"], ver, k)
        elapsed_rag = round(time.time() - t0, 1)

        if "error" in rag and "answer" not in rag:
            entry = {**q_meta, "rag_status": "error", "rag_error": rag["error"],
                     "answer": "", "sources": [], "n_sources": 0,
                     "rag_elapsed_s": elapsed_rag, "score_global": None}
            results[ver].append(entry)
            print(f" ERREUR: {rag['error'][:60]}")
            continue

        answer  = rag.get("answer", "")
        sources = rag.get("sources", [])
        print(f" OK ({elapsed_rag}s) — Judge...", end="", flush=True)
        time.sleep(1.0)

        j = judge_v44(q_meta["question"], answer, sources,
                      q_meta["section"], q_meta["subsection"])

        sg = j.get("score_global")
        mis = j.get("mislabelling_detecte", {})
        has_mis = mis and isinstance(mis, dict) and any(
            isinstance(v, str) and v.strip().lower().startswith("oui")
            for v in mis.values()
        )
        r3 = mis.get("regle_3_absence_quali_non_signalee", "non") if mis else "non"
        flag_str = "✗MIS" if has_mis else "✓"

        entry = {
            **q_meta,
            "rag_status": "ok",
            "answer": answer,
            "n_sources": len(sources),
            "n_subquestions": 0,
            "rag_elapsed_s": elapsed_rag,
            "sources": sources,
            **{k2: v for k2, v in j.items()},
            "mislabelling_flag": has_mis,
        }
        results[ver].append(entry)
        print(f" {sg:.2f} {flag_str}  R3={r3[:3]}")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Backup du fichier original
backup = COMPLET.with_name(f"ablations_103q_v43_gpt4o_COMPLET_backup_{ts}.json")
import shutil
shutil.copy(COMPLET, backup)
print(f"\nBackup → {backup.name}")

# Mise à jour in-place
with open(COMPLET, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Mis à jour → {COMPLET.name}")
print(f"  v_vanilla_k10   : {len(results['v_vanilla_k10'])} entrées")
print(f"  v_vanilla_k25   : {len(results['v_vanilla_k25'])} entrées")
print(f"  v_decomp        : {len(results['v_decomp'])} entrées")
print(f"  v_decomp_raptor : {len(results['v_decomp_raptor'])} entrées")
