"""
run_judge_v45_12q.py
=====================
Juge V4.5 : remplace DIMENSION 4 (Cohérence Quali/Quanti) par un score
d'ALIGNEMENT OBJECTIF/SUBJECTIF conditionné au type de la question.

  - Objectif  → la réponse doit être ancrée indicateurs/OppChoVec
  - Subjectif → la réponse doit être ancrée verbatims/scores d'enquête
  - Mixte     → les deux présents et articulés

  Nuance clé : un score sur 5 issu d'une enquête citoyenne (Likert)
  est une donnée SUBJECTIVE même si c'est un chiffre.

Charge les réponses de COMPLET.json (pas de nouveau call RAG).
Cibles : 12 questions × v_decomp + v_decomp_no_typing.
Compare les scores V4.5 aux scores V4.3 existants dans COMPLET.json.

Usage :
  python run_judge_v45_12q.py           # run + HTML
  python run_judge_v45_12q.py --report  # HTML depuis résultats existants
"""
import argparse, json, re, sys, time, datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None
from eval_from_excel import _call_llm, _build_sources_text, _JUDGE_V43_SYSTEM, _parse_judge_v43

# ── Constantes ──────────────────────────────────────────────────────────────
COMPLET     = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
OUT_DIR     = Path("comparaisons_rag/judge_v45_12q")
RESULTS     = OUT_DIR / "judge_v45_results.json"
CONFIGS     = ["v_decomp", "v_decomp_no_typing"]
TARGET_ROWS = [2, 4, 5, 6, 8, 9, 10, 11, 14, 15, 25, 35]

ORIENTATIONS = {
    2:  "mixte",
    4:  "objectif",  5:  "objectif",  6:  "objectif",
    8:  "objectif",  9:  "subjectif", 10: "subjectif", 11: "subjectif",
    14: "subjectif", 15: "objectif",
    25: "mixte",     35: "mixte",
}
ORIENT_LABEL = {
    "objectif":  "Objectif — données OppChoVec/indicateurs",
    "subjectif": "Subjectif — perceptions/verbatims/enquête",
    "mixte":     "Mixte — objectif + subjectif",
}
CONFIG_LABEL = {
    "v_decomp":           "v_decomp (avec typage)",
    "v_decomp_no_typing": "v_decomp_no_typing (sans typage)",
}

# ── Construction du prompt V4.5 ─────────────────────────────────────────────
#
# On part de _JUDGE_V43_SYSTEM et on remplace :
#   1. La définition courte de DIMENSION 4 (barème compact dans la section dimensions)
#   2. L'Étape 5 de la procédure
#
# On garde le reste du prompt V4.3 INTACT :
#   - Les 4 principes cardinaux
#   - Les définitions opérationnelles OppChoVec
#   - Les 4 règles anti-mislabelling (Règle 1 reste pertinente : si on mislabelle
#     une source quanti en quali pour créer un faux équilibre → alignement factice)
#   - La grille par sous-section
#   - Les exemples (restent illustratifs des dimensions 1-3 ; la dim 4 dans les
#     exemples illustre des cas de mauvais registre, ce qui correspond aussi à V4.5)
#
# La clé JSON de sortie reste "coherence_qualiquanti" pour compatibilité avec
# _parse_judge_v43().

_V45_DIM4 = """\
=== DIMENSION 4 : ALIGNEMENT OBJECTIF / SUBJECTIF ===

Définition : le registre DOMINANT de la réponse est-il cohérent avec le
type de données attendu par la question ? Le type attendu est explicitement
indiqué dans le contexte (REGISTRE ATTENDU : OBJECTIF / SUBJECTIF / MIXTE).

DISTINCTIONS FONDAMENTALES :
- Données OBJECTIVES : scores OppChoVec (Vec, Opp, Cho), indicateurs
  territoriaux statistiques, équipements, classements administratifs.
- Données SUBJECTIVES : verbatims de citoyens, scores issus d'enquêtes
  citoyennes (Likert, satisfaction, perception), synthèses d'entretiens.
  ATTENTION : un score sur 5 issu d'une enquête citoyenne est une donnée
  SUBJECTIVE même si c'est un chiffre.
  Un score OppChoVec est OBJECTIF même s'il est cité dans un contexte
  de bien-être.

Barème :
- 1 : Le mauvais registre DOMINE largement. Réponse objectif à une
      question subjectif, ou inversement.
- 2 : Le mauvais type représente plus de 50 % de la réponse.
- 3 : Les deux types se partagent la réponse environ à parts égales.
      Acceptable pour MIXTE, problématique pour OBJECTIF ou SUBJECTIF.
- 4 : Le bon type domine mais l'autre type occupe encore >30 %.
- 5 : Le bon type domine clairement. Un aparté dans l'autre registre
      est acceptable et bien intégré.

Pour une question MIXTE :
- 5 : objectif et subjectif bien intégrés et articulés.
- 3 : un type domine trop, l'autre anecdotique.
- 1 : un seul type présent, l'autre totalement absent."""

_V43_DIM4_OLD = """\
=== DIMENSION 4 : COHÉRENCE QUALI / QUANTI ===

Barème :
- 1 : Déséquilibré ou inapproprié.
- 2 : Données non pertinentes présentées sans signalement.
- 3 : Approximations signalées mais ciblage imparfait. Plafond si
      Règle 1 s'applique.
- 4 : Bonne intégration ciblée.
- 5 : Intégration exemplaire et ciblée."""

_V43_STEP5_OLD = "**Étape 5** : évalue le ciblage des données mobilisées (quali/quanti)."
_V45_STEP5 = (
    "**Étape 5** : identifie le REGISTRE ATTENDU (OBJECTIF/SUBJECTIF/MIXTE) "
    "indiqué dans le contexte de la question. Évalue si le registre DOMINANT "
    "de la réponse correspond à ce registre attendu. "
    "La DOMINANCE compte, pas la simple présence : un aparté dans l'autre "
    "registre est acceptable."
)

_JUDGE_V45_SYSTEM = (
    _JUDGE_V43_SYSTEM
    .replace(_V43_DIM4_OLD, _V45_DIM4)
    .replace(_V43_STEP5_OLD, _V45_STEP5)
)

# Vérification que les remplacements ont eu lieu
assert _V45_DIM4 in _JUDGE_V45_SYSTEM, (
    "ERREUR : remplacement DIMENSION 4 échoué. "
    "Vérifiez que _V43_DIM4_OLD correspond exactement au texte dans eval_from_excel.py"
)
assert _V45_STEP5 in _JUDGE_V45_SYSTEM, (
    "ERREUR : remplacement Étape 5 échoué."
)


# ── Juge V4.5 ───────────────────────────────────────────────────────────────

def judge_v45(question: str, answer: str, sources: list,
              section: str, subsection: str, expected_type: str,
              registre_attendu: str) -> dict:
    """Juge V4.5 : identique à V4.3 sauf DIMENSION 4 → alignement obj/subj."""
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
        f"REGISTRE ATTENDU : {registre_attendu.upper()}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:4000]}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter.\n"
        "Pour DIMENSION 4 : utilise le REGISTRE ATTENDU ci-dessus comme référence."
    )
    try:
        raw = _call_llm(_JUDGE_V45_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["error"] = None
        return result
    except Exception as e:
        return {"error": str(e), "score_global": None}


# ── Chargement COMPLET.json ──────────────────────────────────────────────────

def load_complet() -> dict:
    with open(COMPLET, encoding="utf-8") as f:
        return json.load(f)


def get_entry(complet: dict, config: str, row: int) -> dict | None:
    entries = complet.get(config, [])
    for e in entries:
        if e.get("excel_row") == row:
            return e
    return None


# ── Idempotence ──────────────────────────────────────────────────────────────

def load_results() -> dict:
    if RESULTS.exists():
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def result_key(config: str, row: int) -> str:
    return f"{config}__Q{row:03d}"


def is_complete(entry: dict) -> bool:
    return isinstance(entry.get("score_global_v45"), (int, float))


# ── Boucle principale ────────────────────────────────────────────────────────

def run(force: bool = False) -> dict:
    complet = load_complet()
    results = load_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(CONFIGS) * len(TARGET_ROWS)
    done = 0

    for config in CONFIGS:
        cfg_short = "DECOMP" if config == "v_decomp" else "NO_TYP"
        for row in TARGET_ROWS:
            key = result_key(config, row)
            existing = results.get(key, {})

            if not force and is_complete(existing):
                done += 1
                print(f"  Q{row:03d} [{cfg_short}] déjà complet → "
                      f"V4.5={existing['score_global_v45']:.2f} "
                      f"(V4.3={existing.get('score_global_v43','?')})")
                continue

            src_entry = get_entry(complet, config, row)
            if src_entry is None:
                print(f"  Q{row:03d} [{cfg_short}] ABSENT dans COMPLET.json — ignoré")
                continue

            orient = ORIENTATIONS[row]
            print(f"  Q{row:03d} [{cfg_short}] ({orient:9s}) jugement V4.5...", end=" ", flush=True)
            t0 = time.time()

            res = judge_v45(
                question      = src_entry["question"],
                answer        = src_entry.get("answer", ""),
                sources       = src_entry.get("sources", []),
                section       = src_entry.get("section", ""),
                subsection    = src_entry.get("subsection", ""),
                expected_type = src_entry.get("type_reponse_attendue_specifie",
                                              "reponse_substantielle_attendue"),
                registre_attendu = orient,
            )
            elapsed = time.time() - t0

            record = {
                "config":       config,
                "excel_row":    row,
                "orientation":  orient,
                "question":     src_entry["question"],
                # Scores V4.3 depuis COMPLET.json
                "score_global_v43":          src_entry.get("score_global"),
                "pertinence_v43":            src_entry.get("pertinence"),
                "fondement_factuel_v43":     src_entry.get("fondement_factuel"),
                "nuance_incertitude_v43":    src_entry.get("nuance_incertitude"),
                "coherence_qualiquanti_v43": src_entry.get("coherence_qualiquanti"),
                # Scores V4.5 (nouveau jugement)
                "score_global_v45":           res.get("score_global"),
                "pertinence_v45":             res.get("pertinence"),
                "fondement_factuel_v45":      res.get("fondement_factuel"),
                "nuance_incertitude_v45":     res.get("nuance_incertitude"),
                "alignement_objsubj_v45":     res.get("coherence_qualiquanti"),  # renamed
                # Justifications V4.5
                "pertinence_v45_justif":          res.get("pertinence_justif"),
                "fondement_factuel_v45_justif":   res.get("fondement_factuel_justif"),
                "nuance_incertitude_v45_justif":  res.get("nuance_incertitude_justif"),
                "alignement_objsubj_v45_justif":  res.get("coherence_qualiquanti_justif"),
                "raisonnement_v45":               res.get("raisonnement"),
                "mislabelling_detecte_v45":       res.get("mislabelling_detecte"),
                "judge_error_v45":                res.get("error"),
                "elapsed_s":    round(elapsed, 1),
                "timestamp":    datetime.datetime.now().isoformat(),
            }
            results[key] = record
            save_results(results)

            if res.get("error"):
                print(f"ERREUR → {res['error']}")
            else:
                v43 = record.get("score_global_v43")
                v45 = record.get("score_global_v45")
                d4_v43 = record.get("coherence_qualiquanti_v43")
                d4_v45 = record.get("alignement_objsubj_v45")
                print(f"V4.5={v45:.2f} (V4.3={v43}) | D4: {d4_v43}→{d4_v45} ({elapsed:.1f}s)")
            done += 1

    print(f"\n{done}/{total} terminés. Résultats → {RESULTS}")
    return results


# ── Génération HTML ──────────────────────────────────────────────────────────

def delta_str(a, b):
    """Retourne la chaîne Δ (b-a) avec signe, ou '—' si indéfini."""
    if a is None or b is None:
        return "—"
    d = round(b - a, 2)
    return f"+{d:.2f}" if d > 0 else f"{d:.2f}"


def delta_class(a, b):
    if a is None or b is None:
        return ""
    d = b - a
    if d > 0.1:
        return "up"
    if d < -0.1:
        return "down"
    return "flat"


def score_badge(v, dim=False):
    if v is None:
        return '<span class="na">—</span>'
    cls = "score"
    if dim:
        if v >= 4.5:
            cls = "score hi"
        elif v >= 3:
            cls = "score mid"
        else:
            cls = "score lo"
    return f'<span class="{cls}">{v}</span>'


def make_html(results: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Organise par orientation
    by_orient = {"objectif": [], "subjectif": [], "mixte": []}
    for config in CONFIGS:
        for row in TARGET_ROWS:
            key = result_key(config, row)
            rec = results.get(key)
            if rec:
                by_orient[ORIENTATIONS[row]].append(rec)

    # Stats globales
    def avg(lst):
        lst = [x for x in lst if x is not None]
        return round(sum(lst) / len(lst), 3) if lst else None

    all_recs = [results[result_key(c, r)] for c in CONFIGS for r in TARGET_ROWS
                if result_key(c, r) in results]

    g43  = avg([r.get("score_global_v43") for r in all_recs])
    g45  = avg([r.get("score_global_v45") for r in all_recs])
    d4_43 = avg([r.get("coherence_qualiquanti_v43") for r in all_recs])
    d4_45 = avg([r.get("alignement_objsubj_v45") for r in all_recs])

    # Stats par config
    def config_stats(config):
        recs = [results.get(result_key(config, r)) for r in TARGET_ROWS
                if result_key(config, r) in results]
        return {
            "global_v43":  avg([r.get("score_global_v43") for r in recs]),
            "global_v45":  avg([r.get("score_global_v45") for r in recs]),
            "d4_v43":      avg([r.get("coherence_qualiquanti_v43") for r in recs]),
            "d4_v45":      avg([r.get("alignement_objsubj_v45") for r in recs]),
        }

    stats_decomp = config_stats("v_decomp")
    stats_notyp  = config_stats("v_decomp_no_typing")

    # Stats par orientation × config
    def orient_config_stats(orient, config):
        rows_o = [r for r in TARGET_ROWS if ORIENTATIONS[r] == orient]
        recs = [results.get(result_key(config, r)) for r in rows_o
                if result_key(config, r) in results]
        return {
            "global_v43": avg([r.get("score_global_v43") for r in recs]),
            "global_v45": avg([r.get("score_global_v45") for r in recs]),
            "d4_v43":     avg([r.get("coherence_qualiquanti_v43") for r in recs]),
            "d4_v45":     avg([r.get("alignement_objsubj_v45") for r in recs]),
        }

    def delta_row(label, v43, v45):
        d = delta_str(v43, v45)
        cls = delta_class(v43, v45)
        return (f'<tr><td>{label}</td>'
                f'<td class="num">{(f"{v43:.3f}") if v43 is not None else "—"}</td>'
                f'<td class="num">{(f"{v45:.3f}") if v45 is not None else "—"}</td>'
                f'<td class="num delta {cls}">{d}</td></tr>')

    # ── Table récapitulative par question ──
    def dim_cells(rec):
        dims = [
            ("Pert", "pertinence_v43", "pertinence_v45"),
            ("Fact", "fondement_factuel_v43", "fondement_factuel_v45"),
            ("Nua",  "nuance_incertitude_v43", "nuance_incertitude_v45"),
            ("D4",   "coherence_qualiquanti_v43", "alignement_objsubj_v45"),
        ]
        cells = ""
        for label, k43, k45 in dims:
            v43 = rec.get(k43)
            v45 = rec.get(k45)
            is_d4 = label == "D4"
            d = delta_str(v43, v45)
            dcls = delta_class(v43, v45)
            cells += (
                f'<td class="num">{v43 if v43 is not None else "—"}</td>'
                f'<td class="num {"hl" if is_d4 else ""}">{v45 if v45 is not None else "—"}</td>'
                f'<td class="num delta {dcls} {"hl" if is_d4 else ""}">{d}</td>'
            )
        return cells

    def question_rows(orient):
        rows_o = [r for r in TARGET_ROWS if ORIENTATIONS[r] == orient]
        html = ""
        for row in rows_o:
            for ci, config in enumerate(CONFIGS):
                key = result_key(config, row)
                rec = results.get(key, {})
                q_text = rec.get("question", f"Q{row:03d}")[:90]
                g43 = rec.get("score_global_v43")
                g45 = rec.get("score_global_v45")
                gd  = delta_str(g43, g45)
                gcls = delta_class(g43, g45)
                cfg_short = CONFIG_LABEL[config]

                if ci == 0:
                    html += (
                        f'<tr class="q-first">'
                        f'<td rowspan="2" class="q-num">Q{row:03d}</td>'
                        f'<td rowspan="2" class="q-orient orient-{orient}">{orient[:3].upper()}</td>'
                        f'<td rowspan="2" class="q-text" title="{q_text}">{q_text}…</td>'
                    )
                else:
                    html += '<tr class="q-second">'

                html += (
                    f'<td class="cfg-cell">{cfg_short}</td>'
                    f'<td class="num">{g43 if g43 is not None else "—"}</td>'
                    f'<td class="num">{g45 if g45 is not None else "—"}</td>'
                    f'<td class="num delta {gcls}">{gd}</td>'
                )
                html += dim_cells(rec)
                html += "</tr>\n"
        return html

    # ── Section détail par question ──
    def detail_section(orient):
        rows_o = [r for r in TARGET_ROWS if ORIENTATIONS[r] == orient]
        html = f'<h3 class="orient-{orient}">{ORIENT_LABEL[orient]}</h3>\n'
        for row in rows_o:
            html += f'<div class="q-block">\n'
            # Récupère la question depuis l'un des configs
            rec0 = results.get(result_key(CONFIGS[0], row), {})
            html += (f'<h4>Q{row:03d} — {rec0.get("question", "")}</h4>\n'
                     f'<p class="orient-tag">Registre attendu : '
                     f'<strong>{ORIENT_LABEL[ORIENTATIONS[row]]}</strong></p>\n')

            for config in CONFIGS:
                key = result_key(config, row)
                rec = results.get(key, {})
                cfg_label = CONFIG_LABEL[config]
                d4_v43 = rec.get("coherence_qualiquanti_v43")
                d4_v45 = rec.get("alignement_objsubj_v45")
                g43 = rec.get("score_global_v43")
                g45 = rec.get("score_global_v45")

                html += (f'<details class="config-detail">\n'
                         f'<summary><strong>{cfg_label}</strong> — '
                         f'Global V4.3={g43} → V4.5={g45} | '
                         f'D4 V4.3={d4_v43} → V4.5={d4_v45}'
                         f'</summary>\n<div class="detail-body">\n')

                # Tableau comparatif
                html += '<table class="dim-table"><thead><tr>'
                html += '<th>Dimension</th><th>V4.3</th><th>V4.5</th><th>Δ</th>'
                html += '<th>Justification V4.5</th></tr></thead><tbody>\n'

                dims = [
                    ("Pertinence",         "pertinence_v43",            "pertinence_v45",            "pertinence_v45_justif"),
                    ("Fondement factuel",  "fondement_factuel_v43",     "fondement_factuel_v45",     "fondement_factuel_v45_justif"),
                    ("Nuance/Incertitude", "nuance_incertitude_v43",    "nuance_incertitude_v45",    "nuance_incertitude_v45_justif"),
                    ("Alignement obj/sub [D4]", "coherence_qualiquanti_v43", "alignement_objsubj_v45", "alignement_objsubj_v45_justif"),
                ]
                for dim_label, k43, k45, k45j in dims:
                    v43 = rec.get(k43)
                    v45 = rec.get(k45)
                    justif = rec.get(k45j, "")
                    d = delta_str(v43, v45)
                    dcls = delta_class(v43, v45)
                    is_d4 = "D4" in dim_label
                    row_cls = " class=\"d4-row\"" if is_d4 else ""
                    html += (
                        f'<tr{row_cls}>'
                        f'<td>{dim_label}</td>'
                        f'<td class="num">{v43 if v43 is not None else "—"}</td>'
                        f'<td class="num">{v45 if v45 is not None else "—"}</td>'
                        f'<td class="num delta {dcls}">{d}</td>'
                        f'<td class="justif">{justif or ""}</td>'
                        f'</tr>\n'
                    )
                html += '</tbody></table>\n'

                # Raisonnement V4.5
                rai = rec.get("raisonnement_v45", "")
                if rai:
                    html += f'<p class="raisonnement"><em>Raisonnement V4.5 :</em> {rai}</p>\n'

                html += '</div>\n</details>\n'
            html += '</div>\n'
        return html

    # ── Bloc DÉCISION ──
    def decision_block():
        rows_obj = [r for r in TARGET_ROWS if ORIENTATIONS[r] == "objectif"]
        rows_sub = [r for r in TARGET_ROWS if ORIENTATIONS[r] == "subjectif"]
        rows_mix = [r for r in TARGET_ROWS if ORIENTATIONS[r] == "mixte"]

        def avg_d4(config, rows):
            recs = [results.get(result_key(config, r)) for r in rows
                    if result_key(config, r) in results]
            return avg([r.get("alignement_objsubj_v45") for r in recs if r])

        def avg_d4_v43(config, rows):
            recs = [results.get(result_key(config, r)) for r in rows
                    if result_key(config, r) in results]
            return avg([r.get("coherence_qualiquanti_v43") for r in recs if r])

        def avg_glob(config, rows, version="v45"):
            recs = [results.get(result_key(config, r)) for r in rows
                    if result_key(config, r) in results]
            key = f"score_global_{version}"
            return avg([r.get(key) for r in recs if r])

        html = '<div class="decision-block">\n'
        html += '<h2>DÉCISION — Impact du juge V4.5 sur les 12 questions (v_decomp vs v_decomp_no_typing)</h2>\n'
        html += '<p class="decision-subtitle">Δ = V4.5 − V4.3 (sur la même config). Positif = V4.5 est plus sévère ou plus généreux selon le sens.</p>\n'

        # 3 orientations × 2 configs
        html += '<div class="decision-grid">\n'
        for orient_name, rows_o, orient_cls in [
            ("Objectif (Q4,5,6,8,15)", rows_obj, "objectif"),
            ("Subjectif (Q9,10,11,14)", rows_sub, "subjectif"),
            ("Mixte (Q2,25,35)",        rows_mix, "mixte"),
        ]:
            html += f'<div class="decision-card orient-{orient_cls}">\n'
            html += f'<h3>{orient_name}</h3>\n'
            html += '<table class="decision-table"><thead><tr>'
            html += '<th>Config</th><th>D4 V4.3</th><th>D4 V4.5</th><th>Δ D4</th><th>Global V4.3</th><th>Global V4.5</th><th>Δ Global</th>'
            html += '</tr></thead><tbody>\n'
            for config in CONFIGS:
                d4_43 = avg_d4_v43(config, rows_o)
                d4_45 = avg_d4(config, rows_o)
                g43   = avg_glob(config, rows_o, "v43")
                g45   = avg_glob(config, rows_o, "v45")
                d4d   = delta_str(d4_43, d4_45)
                gd    = delta_str(g43, g45)
                d4dcls = delta_class(d4_43, d4_45)
                gdcls  = delta_class(g43, g45)
                cfg_s = "DECOMP" if config == "v_decomp" else "NO_TYP"
                html += (
                    f'<tr>'
                    f'<td class="cfg-s">{cfg_s}</td>'
                    f'<td class="num">{(f"{d4_43:.2f}") if d4_43 is not None else "—"}</td>'
                    f'<td class="num">{(f"{d4_45:.2f}") if d4_45 is not None else "—"}</td>'
                    f'<td class="num delta {d4dcls}">{d4d}</td>'
                    f'<td class="num">{(f"{g43:.2f}") if g43 is not None else "—"}</td>'
                    f'<td class="num">{(f"{g45:.2f}") if g45 is not None else "—"}</td>'
                    f'<td class="num delta {gdcls}">{gd}</td>'
                    f'</tr>\n'
                )
            html += '</tbody></table>\n</div>\n'
        html += '</div>\n'

        # Δ typage (DECOMP − NO_TYP) sur V4.5
        html += '<h3>Δ typage (v_decomp − v_decomp_no_typing) sur scores V4.5</h3>\n'
        html += '<table class="decision-table"><thead><tr>'
        html += '<th>Orientation</th><th>Δ D4 V4.5</th><th>Δ Global V4.5</th><th>Δ D4 V4.3</th><th>Δ Global V4.3</th>'
        html += '</tr></thead><tbody>\n'
        for orient_name, rows_o in [
            ("Objectif", rows_obj), ("Subjectif", rows_sub), ("Mixte", rows_mix), ("TOTAL", TARGET_ROWS)
        ]:
            d4_45_dec  = avg_d4("v_decomp", rows_o)
            d4_45_nt   = avg_d4("v_decomp_no_typing", rows_o)
            g45_dec    = avg_glob("v_decomp", rows_o)
            g45_nt     = avg_glob("v_decomp_no_typing", rows_o)
            d4_43_dec  = avg_d4_v43("v_decomp", rows_o)
            d4_43_nt   = avg_d4_v43("v_decomp_no_typing", rows_o)
            g43_dec    = avg_glob("v_decomp", rows_o, "v43")
            g43_nt     = avg_glob("v_decomp_no_typing", rows_o, "v43")

            d4d_45  = delta_str(d4_45_nt, d4_45_dec)
            gd_45   = delta_str(g45_nt,   g45_dec)
            d4d_43  = delta_str(d4_43_nt, d4_43_dec)
            gd_43   = delta_str(g43_nt,   g43_dec)

            d4_45_cls = delta_class(d4_45_nt, d4_45_dec)
            g_45_cls  = delta_class(g45_nt, g45_dec)

            bold = " class=\"total-row\"" if orient_name == "TOTAL" else ""
            html += (
                f'<tr{bold}>'
                f'<td>{orient_name}</td>'
                f'<td class="num delta {d4_45_cls}">{d4d_45}</td>'
                f'<td class="num delta {g_45_cls}">{gd_45}</td>'
                f'<td class="num">{d4d_43}</td>'
                f'<td class="num">{gd_43}</td>'
                f'</tr>\n'
            )
        html += '</tbody></table>\n'
        html += '</div>\n'
        return html

    # ── Tableau récapitulatif (12 questions × 2 configs) ──
    def recap_table():
        html = '<table class="recap-table"><thead>\n'
        html += '<tr>'
        html += '<th rowspan="2">Q</th><th rowspan="2">Or.</th><th rowspan="2">Question</th>'
        html += '<th rowspan="2">Config</th>'
        html += '<th colspan="3">Global</th>'
        html += '<th colspan="3">Pertinence</th>'
        html += '<th colspan="3">Factuel</th>'
        html += '<th colspan="3">Nuance</th>'
        html += '<th colspan="3" class="hl-header">D4 Align</th>'
        html += '</tr>\n<tr>'
        for _ in range(5):
            html += '<th>V4.3</th><th>V4.5</th><th>Δ</th>'
        html += '</tr>\n</thead><tbody>\n'
        for orient in ["objectif", "subjectif", "mixte"]:
            html += f'<tr class="orient-sep"><td colspan="21" class="orient-label orient-{orient}">{ORIENT_LABEL[orient]}</td></tr>\n'
            html += question_rows(orient)
        html += '</tbody></table>\n'
        return html

    # ── CSS ──
    css = """
<style>
:root {
    --bg: #f8f9fa; --bg2: #ffffff; --border: #dee2e6; --text: #212529;
    --text-muted: #6c757d; --accent: #0d6efd;
    --up: #198754; --down: #dc3545; --flat: #6c757d;
    --obj: #0d6efd; --sub: #6f42c1; --mix: #0dcaf0; --mix-dark: #0a9ab5;
    --hi: #198754; --mid: #fd7e14; --lo: #dc3545;
    --d4-bg: #fff3cd; --d4-border: #ffc107;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #121416; --bg2: #1e2124; --border: #2d3035; --text: #e9ecef;
        --text-muted: #adb5bd; --accent: #4dabf7;
        --up: #51cf66; --down: #ff6b6b; --flat: #868e96;
        --obj: #74c0fc; --sub: #cc5de8; --mix: #22b8cf;
        --d4-bg: #3d3000; --d4-border: #ffd43b;
    }
}
:root[data-theme="light"] {
    --bg: #f8f9fa; --bg2: #ffffff; --border: #dee2e6; --text: #212529;
    --text-muted: #6c757d; --accent: #0d6efd;
    --up: #198754; --down: #dc3545; --flat: #6c757d;
    --obj: #0d6efd; --sub: #6f42c1; --mix: #0dcaf0; --mix-dark: #0a9ab5;
    --d4-bg: #fff3cd; --d4-border: #ffc107;
}
:root[data-theme="dark"] {
    --bg: #121416; --bg2: #1e2124; --border: #2d3035; --text: #e9ecef;
    --text-muted: #adb5bd; --accent: #4dabf7;
    --up: #51cf66; --down: #ff6b6b; --flat: #868e96;
    --obj: #74c0fc; --sub: #cc5de8; --mix: #22b8cf;
    --d4-bg: #3d3000; --d4-border: #ffd43b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 14px; background: var(--bg); color: var(--text); line-height: 1.5; padding: 24px; }
h1 { font-size: 1.6rem; margin-bottom: 4px; }
h2 { font-size: 1.2rem; margin: 24px 0 12px; border-bottom: 2px solid var(--border); padding-bottom: 4px; }
h3 { font-size: 1rem; margin: 16px 0 8px; }
h4 { font-size: 0.95rem; margin: 12px 0 4px; }
.meta { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 16px; }
table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
th, td { border: 1px solid var(--border); padding: 5px 8px; text-align: left; }
th { background: var(--bg); font-weight: 600; }
.num { text-align: center; font-variant-numeric: tabular-nums; }
.delta { font-weight: 700; }
.up { color: var(--up); }
.down { color: var(--down); }
.flat { color: var(--flat); }
.hl { background: var(--d4-bg); }
.hl-header { background: var(--d4-bg); }
.d4-row { background: var(--d4-bg); }
.orient-objectif { color: var(--obj); font-weight: 600; }
.orient-subjectif { color: var(--sub); font-weight: 600; }
.orient-mixte { color: var(--mix-dark); font-weight: 600; }
.orient-sep td { padding: 6px 8px; font-weight: 600; }
.q-num { font-weight: 700; text-align: center; }
.q-orient { text-align: center; font-size: 0.75rem; }
.q-text { max-width: 240px; font-size: 0.78rem; color: var(--text-muted); }
.cfg-cell { font-size: 0.75rem; color: var(--text-muted); white-space: nowrap; }
.cfg-s { font-size: 0.8rem; font-weight: 600; }
.q-first td, .q-second td { vertical-align: middle; }
.q-second td { border-top: 1px dashed var(--border); }
.decision-block { background: var(--bg2); border: 2px solid var(--accent); border-radius: 8px; padding: 20px; margin-bottom: 28px; }
.decision-subtitle { color: var(--text-muted); margin-bottom: 16px; font-size: 0.88rem; }
.decision-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
.decision-card { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
.decision-table { font-size: 0.82rem; }
.total-row { font-weight: 700; background: var(--bg2); }
.recap-table { font-size: 0.78rem; }
.q-block { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 16px; padding: 16px; }
.config-detail { margin-bottom: 8px; }
.config-detail summary { cursor: pointer; padding: 6px; background: var(--bg); border-radius: 4px; }
.detail-body { padding: 8px 0; }
.dim-table { margin-top: 8px; }
.justif { max-width: 400px; font-size: 0.8rem; color: var(--text-muted); }
.raisonnement { margin-top: 8px; font-size: 0.82rem; color: var(--text-muted); }
.orient-tag { font-size: 0.82rem; margin-bottom: 8px; color: var(--text-muted); }
details summary { user-select: none; }
.na { color: var(--text-muted); }
@media (max-width: 900px) { .decision-grid { grid-template-columns: 1fr; } }
.section-wrap { overflow-x: auto; }
</style>
"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Juge V4.5 — Alignement Obj/Subj — 12 questions</title>
{css}
</head>
<body>
<h1>Juge V4.5 — Dimension 4 : Alignement Objectif/Subjectif</h1>
<p class="meta">12 questions × v_decomp + v_decomp_no_typing | Généré le {ts}<br>
<strong>Dimension 4 V4.3</strong> : Cohérence Quali/Quanti (équilibre des registres)<br>
<strong>Dimension 4 V4.5</strong> : Alignement Objectif/Subjectif (le registre dominant est-il cohérent avec le type de la question ?)</p>

{decision_block()}

<h2>Tableau récapitulatif — 12 questions × 2 configs</h2>
<div class="section-wrap">
{recap_table()}
</div>

<h2>Détail par question</h2>
"""
    for orient in ["objectif", "subjectif", "mixte"]:
        html += detail_section(orient)

    html += f"""
<p class="meta" style="margin-top:24px">Résultats JSON : {RESULTS}</p>
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true",
                        help="Génère uniquement le HTML depuis les résultats existants")
    parser.add_argument("--force", action="store_true",
                        help="Re-juge tout, même les entrées déjà complètes")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = OUT_DIR / f"rapport_judge_v45_{ts}.html"

    if args.report:
        results = load_results()
    else:
        print(f"Juge V4.5 — {len(CONFIGS)} configs × {len(TARGET_ROWS)} questions\n")
        results = run(force=args.force)

    html = make_html(results)
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML → {html_path}")

    # Copie dans Downloads
    dl = Path.home() / "Downloads" / f"rapport_judge_v45_{ts}.html"
    dl.write_text(html, encoding="utf-8")
    print(f"     → {dl}")


if __name__ == "__main__":
    main()
