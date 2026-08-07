"""
run_notyping_v2_12q.py
=======================
Ablation V_decomp_no_typing_v2 : prompt décomposeur réécrit sans numérotation
(supprime l'artefact "Règles (1)(2)(3)" qui pouvait induire le modèle à chercher
une règle 1 manquante correspondant à la contrainte de typage).

Pipeline : décomposition (Mistral Large, nouveau prompt) → retrieval brut (hors
RAPTOR) → synthèse (Mistral Large) — identique à v_decomp_no_typing sauf le prompt.

Juge : V4.3 (GPT-4o, identique au rapport de référence).

Comparaison avec v_decomp (typage, run existant dans COMPLET.json, non refait).

Usage :
  python run_notyping_v2_12q.py           # run pipeline + juge + HTML
  python run_notyping_v2_12q.py --report  # HTML depuis résultats existants
  python run_notyping_v2_12q.py --force   # re-juge tout
"""
import argparse, json, re, sys, time, datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")

# ── Patch le prompt AVANT tout import du module ──────────────────────────────
import rag_v10_raptor_subq as rag_mod
from rag_v10_raptor_subq import _SYSTEM_DECOMPOSER_NO_TYPING_V2
rag_mod._SYSTEM_DECOMPOSER_NO_TYPING = _SYSTEM_DECOMPOSER_NO_TYPING_V2

# ── Imports pipeline ─────────────────────────────────────────────────────────
from rag_ablations import DecompNoTypingRAG

# ── Imports juge ─────────────────────────────────────────────────────────────
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None
from eval_from_excel import (
    _call_llm, _build_sources_text, _JUDGE_V43_SYSTEM, _parse_judge_v43
)

# ── Constantes ────────────────────────────────────────────────────────────────
COMPLET     = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
OUT_DIR     = Path("comparaisons_rag/notyping_v2_12q")
RESULTS     = OUT_DIR / "notyping_v2_results.json"
TARGET_ROWS = [2, 4, 5, 6, 8, 9, 10, 11, 14, 15, 25, 35]
K           = 5
N_SUBQ      = 5

ORIENTATIONS = {
    2:  "mixte",
    4:  "objectif",  5:  "objectif",  6:  "objectif",
    8:  "objectif",  9:  "subjectif", 10: "subjectif", 11: "subjectif",
    14: "subjectif", 15: "objectif",
    25: "mixte",     35: "mixte",
}
ORIENT_LABEL = {
    "objectif":  "Objectif — OppChoVec / indicateurs",
    "subjectif": "Subjectif — verbatims / enquête",
    "mixte":     "Mixte — objectif + subjectif",
}

# ── Juge V4.3 ────────────────────────────────────────────────────────────────

def judge_v43(question: str, answer: str, sources: list,
              section: str, subsection: str, expected_type: str) -> dict:
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:4000]}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter."
    )
    try:
        raw = _call_llm(_JUDGE_V43_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["error"] = None
        return result
    except Exception as e:
        return {"error": str(e), "score_global": None}


# ── Chargement COMPLET.json (v_decomp existant) ───────────────────────────────

def load_complet_decomp() -> Dict[int, dict]:
    """Charge les entrées v_decomp (avec typage) pour les 12 questions cibles."""
    with open(COMPLET, encoding="utf-8") as f:
        data = json.load(f)
    entries = {}
    for e in data.get("v_decomp", []):
        row = e.get("excel_row")
        if row in TARGET_ROWS:
            entries[row] = e
    return entries


# ── Persistance JSON ──────────────────────────────────────────────────────────

def load_results() -> dict:
    if RESULTS.exists():
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def is_complete(entry: dict) -> bool:
    return (entry.get("rag_status") == "ok"
            and isinstance(entry.get("score_global"), (int, float)))


# ── Normalisation sources pour le juge ───────────────────────────────────────

def normalize_sources_for_judge(sources: List[Dict]) -> List[Dict]:
    """Convertit les sources pipeline en format attendu par _build_sources_text."""
    normalized = []
    for s in sources:
        meta = s.get("meta") or s.get("metadata") or {}
        content = s.get("text") or s.get("content") or s.get("extrait") or ""
        label = s.get("label") or s.get("collection") or meta.get("source_type", "")
        normalized.append({
            "content":  content,
            "metadata": {**meta, "source_type": s.get("collection", label)},
            "label":    label,
        })
    return normalized


# ── Boucle principale ─────────────────────────────────────────────────────────

def run(force: bool = False) -> dict:
    complet_decomp = load_complet_decomp()
    results = load_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Questions cibles depuis COMPLET.json (section, subsection, expected_type, question)
    meta = {row: complet_decomp[row] for row in TARGET_ROWS if row in complet_decomp}
    if not meta:
        sys.exit("ERREUR : aucune entrée v_decomp trouvée dans COMPLET.json pour ces 12 questions.")

    # Initialisation pipeline (une seule fois)
    print("Initialisation pipeline DecompNoTypingRAG (v2 prompt)...")
    rag = DecompNoTypingRAG()
    rag.init()
    print("Pipeline prêt.\n")

    total = len(TARGET_ROWS)
    for i, row in enumerate(TARGET_ROWS, 1):
        key = f"Q{row:03d}"
        existing = results.get(key, {})

        if not force and is_complete(existing):
            sg = existing.get("score_global")
            print(f"  [{i:02d}/{total}] Q{row:03d} déjà complet → score={sg}")
            continue

        src = meta.get(row)
        if not src:
            print(f"  [{i:02d}/{total}] Q{row:03d} ABSENT dans COMPLET.json — ignoré")
            continue

        question     = src["question"]
        section      = src.get("section", "")
        subsection   = src.get("subsection", "")
        expected_type = src.get("type_reponse_attendue_specifie",
                                "reponse_substantielle_attendue")
        orient       = ORIENTATIONS[row]

        # ── Phase 1 : RAG (si pas déjà fait) ──
        if not force and existing.get("rag_status") == "ok" and existing.get("answer"):
            print(f"  [{i:02d}/{total}] Q{row:03d} RAG déjà ok — re-juge seulement")
            answer      = existing["answer"]
            sources_raw = existing.get("sources_raw", [])
            sub_qa      = existing.get("sub_questions", [])
        else:
            print(f"  [{i:02d}/{total}] Q{row:03d} [{orient:9s}] RAG...", end=" ", flush=True)
            t0 = time.time()
            try:
                answer, sources_raw, _, sub_qa_list, sources_mob = rag.query(
                    question, k=K, n_subquestions=N_SUBQ
                )
                rag_elapsed = round(time.time() - t0, 1)
                sub_qa = sub_qa_list
                sources_raw_norm = normalize_sources_for_judge(sources_raw)
                print(f"ok ({rag_elapsed}s, {len(sub_qa)} SQ, {len(sources_raw)} src)")
            except Exception as e:
                print(f"ERREUR RAG : {e}")
                results[key] = {**existing, "rag_status": "error", "rag_error": str(e)}
                save_results(results)
                continue

            # Sauvegarde intermédiaire (RAG ok, juge pas encore fait)
            results[key] = {
                "excel_row":    row,
                "orientation":  orient,
                "question":     question,
                "section":      section,
                "subsection":   subsection,
                "expected_type": expected_type,
                "rag_status":   "ok",
                "answer":       answer,
                "sub_questions": sub_qa,
                "sources_raw":  [{"content": s.get("text") or s.get("content", ""),
                                  "metadata": s.get("meta") or s.get("metadata") or {},
                                  "collection": s.get("collection", ""),
                                  "label": s.get("label", "")}
                                 for s in sources_raw],
                "sources_mobilisees": sources_mob,
                "n_sources":    len(sources_raw),
                "n_subquestions": len(sub_qa),
                "rag_elapsed_s": rag_elapsed,
            }
            save_results(results)
            sources_raw_norm = normalize_sources_for_judge(sources_raw)

        # ── Phase 2 : Juge V4.3 ──
        print(f"         juge V4.3...", end=" ", flush=True)
        t1 = time.time()
        j = judge_v43(
            question=question, answer=answer,
            sources=sources_raw_norm,
            section=section, subsection=subsection,
            expected_type=expected_type,
        )
        judge_elapsed = round(time.time() - t1, 1)

        if j.get("error"):
            print(f"ERREUR : {j['error']}")
        else:
            sg = j.get("score_global")
            print(f"score={sg} ({judge_elapsed}s)")

        results[key].update({
            "score_global":           j.get("score_global"),
            "pertinence":             j.get("pertinence"),
            "fondement_factuel":      j.get("fondement_factuel"),
            "nuance_incertitude":     j.get("nuance_incertitude"),
            "coherence_qualiquanti":  j.get("coherence_qualiquanti"),
            "pertinence_justif":          j.get("pertinence_justif"),
            "fondement_factuel_justif":   j.get("fondement_factuel_justif"),
            "nuance_incertitude_justif":  j.get("nuance_incertitude_justif"),
            "coherence_qualiquanti_justif": j.get("coherence_qualiquanti_justif"),
            "raisonnement":           j.get("raisonnement"),
            "mislabelling_detecte":   j.get("mislabelling_detecte"),
            "mislabelling_flag":      j.get("mislabelling_flag"),
            "judge_error":            j.get("error"),
            "judge_elapsed_s":        judge_elapsed,
            "timestamp":              datetime.datetime.now().isoformat(),
        })
        save_results(results)

    print(f"\n{len(results)}/{total} entrées. Résultats → {RESULTS}")
    return results


# ── HTML ──────────────────────────────────────────────────────────────────────

def _avg(lst):
    lst = [x for x in lst if isinstance(x, (int, float))]
    return round(sum(lst) / len(lst), 3) if lst else None


def _sc(v, hi=4.5, mid=3.0):
    if v is None:
        return '<span class="na">—</span>'
    cls = "shi" if v >= hi else ("smid" if v >= mid else "slo")
    return f'<span class="{cls}">{v}</span>'


def _d(a, b):
    if a is None or b is None:
        return "—"
    d = round(b - a, 2)
    return (f'+{d:.2f}' if d > 0 else f'{d:.2f}')


def _dcls(a, b):
    if a is None or b is None:
        return ""
    d = b - a
    return "up" if d > 0.05 else ("dn" if d < -0.05 else "fl")


def sources_html(sources: list) -> str:
    if not sources:
        return '<em class="muted">aucune source</em>'
    items = []
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata") or s.get("meta") or {}
        coll = s.get("collection") or meta.get("source_type") or s.get("label") or "?"
        commune = meta.get("commune") or ""
        label = f"{coll}" + (f" — {commune}" if commune else "")
        content = (s.get("content") or s.get("text") or "")[:600]
        sq_idx = meta.get("sub_question_idx") or s.get("sub_question_idx")
        sq_tag = f' <span class="sq-tag">SQ{sq_idx}</span>' if sq_idx else ""
        items.append(
            f'<details class="src-item"><summary>Src {i} : {label}{sq_tag}</summary>'
            f'<div class="src-body">{content}</div></details>'
        )
    return "".join(items)


def subq_html(sub_qa: list) -> str:
    if not sub_qa:
        return '<em class="muted">sous-questions non disponibles</em>'
    items = []
    for sq in sub_qa:
        idx = sq.get("idx", "·")
        q   = sq.get("question", "")
        a   = sq.get("answer", "").strip()
        a_block = (
            f'<details class="sq-ans"><summary class="sq-ans-tog">Réponse</summary>'
            f'<div class="sq-ans-body">{a}</div></details>'
        ) if a else ""
        items.append(
            f'<li><span class="sq-idx">SQ{idx}</span>'
            f' <span class="sq-q">{q}</span>{a_block}</li>'
        )
    return f'<ol class="sq-list">{"".join(items)}</ol>'


def mis_html(mis: dict) -> str:
    if not mis:
        return '<em class="muted">—</em>'
    lines = []
    for rule, val in mis.items():
        triggered = str(val).lower() not in ("non", "false", "", "null", "none")
        cls = "mis-yes" if triggered else "mis-no"
        label = rule.replace("regle_", "R").replace("_", " ")
        lines.append(f'<span class="{cls}">{label}: {val}</span>')
    return "<br>".join(lines)


def make_html(results: dict, complet_decomp: Dict[int, dict]) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Stats globales ──
    nt2_recs  = [results.get(f"Q{r:03d}") for r in TARGET_ROWS if results.get(f"Q{r:03d}")]
    dec_recs  = [complet_decomp.get(r) for r in TARGET_ROWS if complet_decomp.get(r)]

    def avg_dim(recs, key):
        return _avg([r.get(key) for r in recs if r])

    dims_keys = [
        ("Pertinence",    "pertinence"),
        ("Fact. factuel", "fondement_factuel"),
        ("Nuance",        "nuance_incertitude"),
        ("Coh. Q/Q",      "coherence_qualiquanti"),
    ]

    # ── Bloc DÉCISION ──
    def decision_block():
        rows_obj = [r for r in TARGET_ROWS if ORIENTATIONS[r] == "objectif"]
        rows_sub = [r for r in TARGET_ROWS if ORIENTATIONS[r] == "subjectif"]
        rows_mix = [r for r in TARGET_ROWS if ORIENTATIONS[r] == "mixte"]

        def grp_avg(recs_src, rows_o, key):
            vals = [recs_src.get(r, {}).get(key) if isinstance(recs_src, dict)
                    else next((e.get(key) for e in recs_src if e and e.get("excel_row") == r), None)
                    for r in rows_o]
            return _avg(vals)

        # Pour complet_decomp (dict keyed by row)
        def dec_avg(rows_o, key):
            return _avg([complet_decomp.get(r, {}).get(key) for r in rows_o])

        # Pour results (dict keyed by "Q{row:03d}")
        def nt2_avg(rows_o, key):
            return _avg([results.get(f"Q{r:03d}", {}).get(key) for r in rows_o])

        html = '<div class="dec-block">\n'
        html += '<h2>DÉCISION — v_decomp (typage) vs v_decomp_no_typing_v2 (nouveau prompt)</h2>\n'
        html += '<p class="dec-sub">Δ = no_typing_v2 − v_decomp (positif = no_typing_v2 meilleur)</p>\n'

        html += '<div class="dec-grid">\n'
        for label, rows_o, orient in [
            ("Objectif (Q4,5,6,8,15)", rows_obj, "objectif"),
            ("Subjectif (Q9,10,11,14)", rows_sub, "subjectif"),
            ("Mixte (Q2,25,35)", rows_mix, "mixte"),
            ("TOTAL (12 questions)", TARGET_ROWS, None),
        ]:
            cls = f"dec-card orient-{orient}" if orient else "dec-card dec-total"
            html += f'<div class="{cls}"><h3>{label}</h3>\n'
            html += '<table class="dec-tbl"><thead><tr>'
            html += '<th>Dim.</th><th>DECOMP</th><th>NT_v2</th><th>Δ</th></tr></thead><tbody>\n'
            for dim_label, dim_key in dims_keys + [("GLOBAL", "score_global")]:
                dv  = dec_avg(rows_o, dim_key)
                nv  = nt2_avg(rows_o, dim_key)
                d   = _d(dv, nv)
                cls2 = _dcls(dv, nv)
                bold = ' class="total-dim"' if dim_label == "GLOBAL" else ""
                html += (
                    f'<tr{bold}><td>{dim_label}</td>'
                    f'<td class="num">{(f"{dv:.3f}") if dv is not None else "—"}</td>'
                    f'<td class="num">{(f"{nv:.3f}") if nv is not None else "—"}</td>'
                    f'<td class="num delta {cls2}">{d}</td></tr>\n'
                )
            html += '</tbody></table></div>\n'
        html += '</div>\n</div>\n'
        return html

    # ── Tableau récap ──
    def recap_table():
        html = '<table class="recap"><thead><tr>'
        html += '<th>Q</th><th>Orient.</th>'
        for dim_label, _ in dims_keys:
            html += f'<th colspan="3">{dim_label}</th>'
        html += '<th colspan="3">GLOBAL</th></tr>\n<tr>'
        html += '<th></th><th></th>'
        for _ in range(5):
            html += '<th>DEC</th><th>NT2</th><th>Δ</th>'
        html += '</tr></thead><tbody>\n'

        for orient in ["objectif", "subjectif", "mixte"]:
            rows_o = [r for r in TARGET_ROWS if ORIENTATIONS[r] == orient]
            html += (f'<tr class="orient-sep"><td colspan="17" '
                     f'class="orient-label orient-{orient}">'
                     f'{ORIENT_LABEL[orient]}</td></tr>\n')
            for row in rows_o:
                dec = complet_decomp.get(row, {})
                nt2 = results.get(f"Q{row:03d}", {})
                q_short = (dec.get("question") or nt2.get("question") or "")[:60]
                html += f'<tr title="{q_short}">'
                html += f'<td class="q-num">Q{row:03d}</td>'
                html += f'<td class="orient-cell orient-{orient}">{orient[:3].upper()}</td>'
                for _, dk in dims_keys + [("GLOBAL", "score_global")]:
                    dv = dec.get(dk)
                    nv = nt2.get(dk)
                    d  = _d(dv, nv)
                    dc = _dcls(dv, nv)
                    html += (
                        f'<td class="num">{_sc(dv)}</td>'
                        f'<td class="num">{_sc(nv)}</td>'
                        f'<td class="num delta {dc}">{d}</td>'
                    )
                html += '</tr>\n'
        html += '</tbody></table>\n'
        return html

    # ── Détail par question ──
    def detail_section():
        html = ""
        for orient in ["objectif", "subjectif", "mixte"]:
            html += (f'<h3 class="orient-{orient}">'
                     f'{ORIENT_LABEL[orient]}</h3>\n')
            rows_o = [r for r in TARGET_ROWS if ORIENTATIONS[r] == orient]
            for row in rows_o:
                dec = complet_decomp.get(row, {})
                nt2 = results.get(f"Q{row:03d}", {})
                question = dec.get("question") or nt2.get("question") or f"Q{row:03d}"
                html += f'<div class="q-block">\n<h4>Q{row:03d} — {question}</h4>\n'
                html += f'<p class="orient-tag">Orientation : <strong>{ORIENT_LABEL[ORIENTATIONS[row]]}</strong></p>\n'

                for config_label, rec, is_new in [
                    ("v_decomp — avec typage (run existant)", dec, False),
                    ("v_decomp_no_typing_v2 — sans typage, prompt propre (nouveau run)", nt2, True),
                ]:
                    if not rec:
                        continue
                    sg = rec.get("score_global")
                    html += (f'<details class="cfg-detail {"cfg-new" if is_new else "cfg-old"}">\n'
                             f'<summary><strong>{config_label}</strong> '
                             f'— Global : {_sc(sg)}</summary>\n'
                             f'<div class="cfg-body">\n')

                    # Scores par dimension
                    html += '<table class="dim-tbl"><thead><tr>'
                    html += '<th>Dimension</th><th>Score</th><th>Justification</th></tr></thead><tbody>\n'
                    for dim_label, dk in dims_keys + [("GLOBAL", "score_global")]:
                        v = rec.get(dk)
                        jk = dk + "_justif" if dk != "score_global" else None
                        justif = rec.get(jk, "") if jk else ""
                        bold = ' class="total-dim"' if dim_label == "GLOBAL" else ""
                        html += (f'<tr{bold}><td>{dim_label}</td>'
                                 f'<td class="num">{_sc(v)}</td>'
                                 f'<td class="justif">{justif or ""}</td></tr>\n')
                    html += '</tbody></table>\n'

                    # Mislabelling
                    mis = rec.get("mislabelling_detecte")
                    if mis:
                        html += f'<div class="mis-block">{mis_html(mis)}</div>\n'

                    # Raisonnement juge
                    rai = rec.get("raisonnement")
                    if rai:
                        html += f'<p class="raisonnement"><em>Raisonnement juge :</em> {rai}</p>\n'

                    # Réponse complète
                    answer = rec.get("answer", "")
                    if answer:
                        html += (f'<details class="ans-detail"><summary>Réponse complète</summary>'
                                 f'<div class="ans-body">{answer}</div></details>\n')

                    # Sous-questions (seulement disponibles pour no_typing_v2)
                    sub_qa = rec.get("sub_questions") if is_new else None
                    if is_new:
                        html += (f'<details class="sq-detail"><summary>Sous-questions '
                                 f'({len(sub_qa) if sub_qa else 0})</summary>'
                                 f'<div>{subq_html(sub_qa)}</div></details>\n')

                    # Sources
                    src_list = rec.get("sources_raw") if is_new else rec.get("sources", [])
                    if src_list:
                        html += (f'<details class="src-detail"><summary>Sources '
                                 f'({len(src_list)})</summary>'
                                 f'<div>{sources_html(src_list)}</div></details>\n')

                    html += '</div>\n</details>\n'
                html += '</div>\n'
        return html

    css = """
<style>
:root {
    --bg: #f8f9fa; --bg2: #fff; --border: #dee2e6; --text: #212529;
    --muted: #6c757d; --accent: #0d6efd;
    --up: #198754; --dn: #dc3545; --fl: #6c757d;
    --obj: #0d6efd; --sub: #6f42c1; --mix: #0a9ab5;
    --shi: #198754; --smid: #fd7e14; --slo: #dc3545;
    --new-border: #0d6efd; --old-border: #adb5bd;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #121416; --bg2: #1e2124; --border: #2d3035; --text: #e9ecef;
        --muted: #adb5bd; --accent: #4dabf7;
        --up: #51cf66; --dn: #ff6b6b; --fl: #868e96;
        --obj: #74c0fc; --sub: #cc5de8; --mix: #22b8cf;
        --shi: #51cf66; --smid: #ff922b; --slo: #ff6b6b;
    }
}
:root[data-theme="light"] { --bg: #f8f9fa; --bg2: #fff; --border: #dee2e6; --text: #212529; --muted: #6c757d; --obj: #0d6efd; --sub: #6f42c1; --mix: #0a9ab5; --shi: #198754; --smid: #fd7e14; --slo: #dc3545; --up: #198754; --dn: #dc3545; --fl: #6c757d; }
:root[data-theme="dark"] { --bg: #121416; --bg2: #1e2124; --border: #2d3035; --text: #e9ecef; --muted: #adb5bd; --obj: #74c0fc; --sub: #cc5de8; --mix: #22b8cf; --shi: #51cf66; --smid: #ff922b; --slo: #ff6b6b; --up: #51cf66; --dn: #ff6b6b; --fl: #868e96; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 13px; background: var(--bg); color: var(--text); line-height: 1.5; padding: 24px; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: 4px; }
h2 { font-size: 1.15rem; margin: 24px 0 10px; border-bottom: 2px solid var(--border); padding-bottom: 4px; }
h3 { font-size: 1rem; margin: 16px 0 8px; }
h4 { font-size: 0.9rem; margin: 10px 0 4px; }
.meta { color: var(--muted); font-size: 0.82rem; margin-bottom: 16px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid var(--border); padding: 4px 7px; text-align: left; font-size: 0.8rem; }
th { background: var(--bg); font-weight: 600; }
.num { text-align: center; font-variant-numeric: tabular-nums; }
.delta { font-weight: 700; }
.up { color: var(--up); } .dn { color: var(--dn); } .fl { color: var(--fl); }
.shi { color: var(--shi); font-weight: 700; }
.smid { color: var(--smid); font-weight: 600; }
.slo { color: var(--slo); font-weight: 600; }
.na { color: var(--muted); }
.orient-objectif { color: var(--obj); font-weight: 600; }
.orient-subjectif { color: var(--sub); font-weight: 600; }
.orient-mixte { color: var(--mix); font-weight: 600; }
.orient-sep td { padding: 6px 8px; font-weight: 600; }
.q-num { font-weight: 700; text-align: center; white-space: nowrap; }
.orient-cell { text-align: center; font-size: 0.72rem; }
/* Decision block */
.dec-block { background: var(--bg2); border: 2px solid var(--accent); border-radius: 8px; padding: 18px; margin-bottom: 24px; }
.dec-sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 14px; }
.dec-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 16px; }
.dec-card { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
.dec-total { grid-column: 1 / -1; background: var(--bg2); border: 2px solid var(--border); }
.dec-tbl { font-size: 0.8rem; }
.total-dim { font-weight: 700; }
/* Recap table */
.recap-wrap { overflow-x: auto; margin-bottom: 24px; }
.recap { font-size: 0.75rem; white-space: nowrap; }
/* Q block */
.q-block { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 14px; margin-bottom: 14px; }
.orient-tag { font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; }
.cfg-detail { margin-bottom: 10px; }
.cfg-detail summary { cursor: pointer; padding: 6px 10px; border-radius: 4px; background: var(--bg); }
.cfg-new summary { border-left: 3px solid var(--accent); }
.cfg-old summary { border-left: 3px solid var(--old-border); }
.cfg-body { padding: 10px 0; }
.dim-tbl { margin-bottom: 8px; font-size: 0.8rem; }
.justif { color: var(--muted); max-width: 400px; }
.mis-block { font-size: 0.78rem; margin: 6px 0; }
.mis-yes { color: var(--dn); }
.mis-no { color: var(--muted); }
.raisonnement { font-size: 0.8rem; color: var(--muted); margin: 6px 0; }
.ans-detail summary, .sq-detail summary, .src-detail summary { cursor: pointer; color: var(--accent); font-size: 0.8rem; }
.ans-body { margin-top: 6px; font-size: 0.82rem; white-space: pre-wrap; background: var(--bg); padding: 10px; border-radius: 4px; }
.sq-list { margin: 8px 0 4px 16px; }
.sq-list li { margin-bottom: 6px; }
.sq-idx { font-weight: 700; color: var(--accent); font-size: 0.75rem; }
.sq-q { color: var(--text); }
.sq-ans summary.sq-ans-tog { cursor: pointer; color: var(--muted); font-size: 0.78rem; }
.sq-ans-body { font-size: 0.8rem; white-space: pre-wrap; background: var(--bg); padding: 8px; border-radius: 3px; margin-top: 4px; }
.src-item summary { cursor: pointer; color: var(--muted); font-size: 0.78rem; }
.src-body { font-size: 0.78rem; white-space: pre-wrap; background: var(--bg); padding: 8px; border-radius: 3px; margin-top: 4px; color: var(--muted); }
.sq-tag { background: var(--accent); color: white; font-size: 0.65rem; padding: 1px 4px; border-radius: 3px; }
details summary { user-select: none; }
</style>
"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>No-Typing v2 vs Decomp — 12 questions — Juge V4.3</title>
{css}
</head>
<body>
<h1>v_decomp (typage) vs v_decomp_no_typing_v2 (prompt propre) — 12 questions</h1>
<p class="meta">Juge V4.3 (GPT-4o) · Généré le {ts}<br>
<strong>no_typing_v2</strong> : prompt de décomposeur réécrit sans numérotation ni structure
évoquant une règle manquante — uniquement les 3 contraintes actives (données inexistantes,
hors-domaine, géographique).<br>
<strong>v_decomp</strong> : run existant depuis COMPLET.json, non refait.</p>

{decision_block()}

<h2>Tableau récapitulatif</h2>
<div class="recap-wrap">
{recap_table()}
</div>

<h2>Détail par question</h2>
{detail_section()}

<p class="meta" style="margin-top:20px">
JSON no_typing_v2 → {RESULTS}<br>
Sources v_decomp → {COMPLET}
</p>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = OUT_DIR / f"rapport_notyping_v2_{ts}.html"

    complet_decomp = load_complet_decomp()

    if args.report:
        results = load_results()
    else:
        print(f"No-Typing v2 — {len(TARGET_ROWS)} questions (k={K}, n_subq={N_SUBQ})\n")
        results = run(force=args.force)

    html = make_html(results, complet_decomp)
    html_path.write_text(html, encoding="utf-8")
    print(f"\nHTML → {html_path}")

    dl = Path.home() / "Downloads" / f"rapport_notyping_v2_{ts}.html"
    dl.write_text(html, encoding="utf-8")
    print(f"     → {dl}")
    print(f"\nRelance : python run_notyping_v2_12q.py")


if __name__ == "__main__":
    main()
