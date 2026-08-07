"""
run_register_alignment_12q.py
==============================
Check d'alignement de registre (standalone, GPT-4o-mini) sur 12 questions
× v_decomp vs v_decomp_no_typing, depuis les réponses déjà dans COMPLET.json.

Mesure : le registre DOMINANT de la réponse est-il cohérent avec le type de la question ?
  - Question quanti → réponse ancrée indicateurs/scores
  - Question quali  → réponse ancrée verbatims/perceptions
  - Question mixte  → les deux présents, équilibre attendu
  Note 1 = hors-sujet (mauvais registre dominant) / Note 5 = alignement parfait.

Usage :
  python run_register_alignment_12q.py           # évalue + HTML
  python run_register_alignment_12q.py --report  # HTML depuis résultats existants
"""
import argparse, json, re, sys, time, datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import importlib
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None
from eval_from_excel import _call_llm

# ── Constantes ────────────────────────────────────────────────────────────────
COMPLET  = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
OUT_DIR  = Path("comparaisons_rag/register_alignment_12q")
RESULTS  = OUT_DIR / "register_alignment_results_v2.json"
CONFIGS  = ["v_decomp", "v_decomp_no_typing"]
TARGET_ROWS = [2, 4, 5, 6, 8, 9, 10, 11, 14, 15, 25, 35]

ORIENTATIONS = {
    2: "mixte",     4: "objectif", 5: "objectif", 6: "objectif",
    8: "objectif",  9: "subjectif", 10: "subjectif", 11: "subjectif",
    14: "subjectif", 15: "objectif", 25: "mixte", 35: "mixte",
}
ORIENT_LABEL = {
    "objectif":  "Objectif — données OppChoVec/indicateurs",
    "subjectif": "Perception — données subjectives/enquête",
    "mixte":     "Vaste — objectif + subjectif",
}

# ── Prompt du check ────────────────────────────────────────────────────────────
_SYSTEM_REGISTER = """\
Tu es un évaluateur spécialisé en cohérence de registre pour un système RAG sur le bien-être en Corse.

DISTINCTION FONDAMENTALE — objectif vs subjectif :

Données OBJECTIVES (indépendantes des opinions) :
- Scores OppChoVec (Vec, Opp, Cho) et leurs composantes (revenu, logement, emploi, accès services, droit de vote…)
- Indicateurs territoriaux statistiques (taux de chômage, équipements, densité…)
- Données administratives, effectifs, classements territoriaux

Données SUBJECTIVES (issues des perceptions et expériences des personnes) :
- Verbatims de citoyens (extraits de témoignages, citations directes)
- Scores de satisfaction issus d'enquêtes citoyennes (ex : "3.8/5 sur la sécurité", "note moyenne de 4.1 pour les services")
- Synthèses d'entretiens semi-directifs, perceptions rapportées, ressentis

ATTENTION : un score sur 5 issu d'une enquête citoyenne est une donnée SUBJECTIVE, même si c'est un chiffre.
Un score OppChoVec est une donnée OBJECTIVE, même si on le cite pour parler de "bien-être".

Une question peut être de trois types :
- OBJECTIF : demande des indicateurs territoriaux, classements, scores OppChoVec
- SUBJECTIF : demande des perceptions, ressentis, verbatims — les scores d'enquête citoyenne sont ici bienvenus
- MIXTE    : croise les deux de façon équilibrée

Ta tâche : évaluer si le type DOMINANT des données mobilisées dans la réponse est cohérent avec le type attendu.

CRITÈRES DE NOTATION (1-5) :
- Note 5 : type dominant clairement correct ; aparté dans l'autre type acceptable et bien intégré
- Note 4 : type dominant correct mais l'autre type occupe trop de place (>30%)
- Note 3 : les deux types se partagent — acceptable pour MIXTE, problématique sinon
- Note 2 : le mauvais type domine légèrement (>50%)
- Note 1 : la réponse est principalement dans le mauvais type — hors-sujet

Pour une question MIXTE :
- Note 5 : objectif et subjectif bien intégrés et articulés
- Note 3 : un type domine trop, l'autre anecdotique
- Note 1 : un seul type présent, l'autre absent

Ne juge pas la qualité factuelle ni la richesse — UNIQUEMENT l'alignement objectif/subjectif.

Réponds en JSON :
{
  "type_dominant_observe": "objectif|subjectif|mixte",
  "alignement_score": <1-5>,
  "verdict": "ALIGNE|HORS_SUJET|PARTIEL",
  "proportion_attendu": "<estimation en % du type attendu dans la réponse>",
  "justification": "<1-2 phrases : qu'est-ce qui domine et pourquoi c'est aligné ou non>"
}"""

def check_register(question: str, registre_attendu: str, answer: str) -> dict:
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"REGISTRE ATTENDU : {registre_attendu.upper()}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:3000]}\n\n"
        "Évalue l'alignement de registre selon les critères donnés.\n"
        "Réponds UNIQUEMENT avec le JSON demandé."
    )
    try:
        raw = _call_llm(_SYSTEM_REGISTER, user_prompt,
                        max_tokens=400, light=True, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        return {
            "registre_dominant":  j.get("type_dominant_observe", "?"),
            "alignement_score":   j.get("alignement_score"),
            "verdict":            j.get("verdict", "?"),
            "proportion_attendu": j.get("proportion_attendu", "?"),
            "justification":      j.get("justification", ""),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "alignement_score": None, "verdict": "ERREUR"}

# ── Chargement réponses ───────────────────────────────────────────────────────
def load_answers() -> dict:
    with open(COMPLET, encoding="utf-8") as f:
        data = json.load(f)
    answers = {}  # row → {cfg → {question, answer, section, subsection}}
    for cfg in CONFIGS:
        for e in data.get(cfg, []):
            row = e["excel_row"]
            if row not in TARGET_ROWS:
                continue
            if e.get("rag_status") != "ok":
                continue
            answers.setdefault(row, {})[cfg] = {
                "question":   e["question"],
                "answer":     e.get("answer", ""),
                "section":    e.get("section", ""),
                "subsection": e.get("subsection", ""),
                "score_global_v43": e.get("score_global"),
                "coherence_qualiquanti_v43": e.get("coherence_qualiquanti"),
            }
    return answers

# ── Évaluation ────────────────────────────────────────────────────────────────
def run_all(answers: dict) -> dict:
    results = {}
    for row in TARGET_ROWS:
        reg = ORIENTATIONS[row]
        results[row] = {"registre_attendu": reg, "configs": {}}
        for cfg in CONFIGS:
            entry = answers.get(row, {}).get(cfg)
            if not entry:
                print(f"  Q{row:03d} [{cfg}] : manquant", flush=True)
                continue
            cfg_s = "DECOMP   " if cfg == "v_decomp" else "NO_TYPING"
            print(f"  Q{row:03d} [{cfg_s}] ({reg:6}) check...", end="", flush=True)
            t0 = time.time()
            result = check_register(entry["question"], reg, entry["answer"])
            elapsed = round(time.time() - t0, 1)
            result["elapsed_s"] = elapsed
            result["score_global_v43"] = entry["score_global_v43"]
            result["coherence_qualiquanti_v43"] = entry["coherence_qualiquanti_v43"]
            result["answer"] = entry["answer"]
            result["question"] = entry["question"]
            results[row]["configs"][cfg] = result
            sc = result.get("alignement_score", "?")
            vd = result.get("verdict", "?")
            print(f" score={sc} [{vd}] ({elapsed}s)", flush=True)
        results[row]["question"] = answers.get(row, {}).get("v_decomp", {}).get("question", f"Q{row}")
    return results

# ── HTML ──────────────────────────────────────────────────────────────────────
COLORS = {"v_decomp": "#27ae60", "v_decomp_no_typing": "#8e44ad"}
REG_COLORS = {"objectif": "#1d4ed8", "subjectif": "#9d174d", "mixte": "#4338ca"}
REG_BG     = {"objectif": "#dbeafe", "subjectif": "#fce7f3", "mixte": "#e0e7ff"}

def _sc(s):
    return f"{s:.2f}" if isinstance(s, (int, float)) else "—"

def _avg(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    return round(sum(v)/len(v), 2) if v else None

def verdict_badge(v):
    cls = {"ALIGNE": "vd-ok", "PARTIEL": "vd-part", "HORS_SUJET": "vd-bad"}.get(v, "vd-unk")
    return f'<span class="vbadge {cls}">{v or "?"}</span>'

def score_color(s):
    if s is None:    return "#94a3b8"
    if s >= 4.5:     return "#16a34a"
    if s >= 3.5:     return "#65a30d"
    if s >= 2.5:     return "#ca8a04"
    return "#dc2626"

def make_html(results: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Statistiques globales ──────────────────────────────────────────────
    def cfg_entries(cfg, orient=None):
        return [
            results[r]["configs"].get(cfg, {})
            for r in TARGET_ROWS
            if (orient is None or ORIENTATIONS[r] == orient)
            and results[r]["configs"].get(cfg, {}).get("alignement_score") is not None
        ]

    def summary_stats(cfg, orient=None):
        es = cfg_entries(cfg, orient)
        scores = [e["alignement_score"] for e in es if e.get("alignement_score")]
        hs = sum(1 for e in es if e.get("verdict") == "HORS_SUJET")
        return _avg(scores), hs, len(es)

    # ── Tableau DECISION ──────────────────────────────────────────────────
    def decision_row(label, orient=None):
        sa, ha, na = summary_stats("v_decomp", orient)
        sb, hb, nb = summary_stats("v_decomp_no_typing", orient)
        d = round(sa - sb, 2) if sa and sb else None
        d_mis = ha - hb
        d_str = ("+" if d and d > 0 else "") + (f"{d:.2f}" if d is not None else "—")
        d_clr = "#16a34a" if (d or 0) > 0 else ("#dc2626" if (d or 0) < 0 else "#64748b")
        d_mis_str = ("+" if d_mis > 0 else "") + str(d_mis)
        d_mis_clr = "#16a34a" if d_mis < 0 else ("#dc2626" if d_mis > 0 else "#64748b")
        return (
            f'<tr><td class="dl">{label}</td>'
            f'<td class="dc">{_sc(sa)}</td><td class="dc">{_sc(sb)}</td>'
            f'<td class="dc" style="color:{d_clr};font-weight:800">{d_str}</td>'
            f'<td class="dc">{ha}/{na}</td><td class="dc">{hb}/{nb}</td>'
            f'<td class="dc" style="color:{d_mis_clr};font-weight:700">{d_mis_str}</td>'
            f'</tr>'
        )

    decision_html = f"""
<div class="decision-box">
  <div class="dtitle">ALIGNEMENT REGISTRE — Δ bruts (v_decomp − v_decomp_no_typing)</div>
  <div class="dnote">Score 1–5 : dominance du bon registre dans la réponse · HORS_SUJET = mauvais registre dominant</div>
  <table class="dtable">
    <thead><tr>
      <th>Périmètre</th>
      <th>DECOMP</th><th>NO_TYPING</th><th>Δ score</th>
      <th>H-S DECOMP</th><th>H-S NO_TYP</th><th>Δ H-S</th>
    </tr></thead>
    <tbody>
      {decision_row("Global (n=12)")}
      {decision_row("Objectif — OppChoVec (n=5)", "objectif")}
      {decision_row("Subjectif — Perception (n=4)", "subjectif")}
      {decision_row("Mixte — Vaste (n=3)", "mixte")}
    </tbody>
  </table>
  <div class="dcaption">Δ positif = DECOMP (avec typage) mieux aligné · H-S = HORS_SUJET</div>
</div>"""

    # ── Détail par question ───────────────────────────────────────────────
    def col_html(e, cfg):
        if not e or e.get("error"):
            return f'<div class="col-err">Erreur : {(e or {}).get("error","?")}</div>'
        sc   = e.get("alignement_score")
        vd   = e.get("verdict", "?")
        dom  = e.get("registre_dominant", "?")
        prop = e.get("proportion_attendu", "?")
        just = e.get("justification", "")
        sg43 = e.get("score_global_v43")
        cqq  = e.get("coherence_qualiquanti_v43")
        answer = e.get("answer", "")
        color = COLORS[cfg]
        border = "border-left:3px solid #dc2626" if vd == "HORS_SUJET" else \
                 "border-left:3px solid #ca8a04" if vd == "PARTIEL" else \
                 "border-left:3px solid #16a34a"
        return f"""
<div class="col-inner" style="{border}">
  <div class="col-lbl" style="color:{color}">{'DECOMP (typage)' if cfg=='v_decomp' else 'NO_TYPING'}</div>
  <div class="score-big" style="color:{score_color(sc)}">{_sc(sc)}<span class="score-unit">/5</span></div>
  <div style="margin:6px 0">{verdict_badge(vd)}</div>
  <div class="info-row"><span class="info-lbl">Registre observé :</span> <b>{dom}</b></div>
  <div class="info-row"><span class="info-lbl">Part du bon registre :</span> {prop}</div>
  <div class="info-row just-txt">{just}</div>
  <div class="info-row" style="color:#94a3b8;font-size:11px">V4.3 global={_sc(sg43)} · cqq={_sc(cqq)}</div>
  <details class="ans-det"><summary class="ans-sum">Réponse complète</summary>
    <div class="ans-body">{answer}</div>
  </details>
</div>"""

    detail_sections = []
    for row in TARGET_ROWS:
        rd  = results.get(row, {})
        reg = ORIENTATIONS[row]
        q   = rd.get("question", f"Q{row}")
        typ_e = rd["configs"].get("v_decomp", {})
        nt_e  = rd["configs"].get("v_decomp_no_typing", {})
        sa = typ_e.get("alignement_score")
        sb = nt_e.get("alignement_score")
        dq = round(sa - sb, 2) if sa is not None and sb is not None else None
        dq_s = ("+" if dq and dq > 0 else "") + (f"{dq:.2f}" if dq is not None else "—")
        dq_clr = score_color(dq)
        has_hs = typ_e.get("verdict") == "HORS_SUJET" or nt_e.get("verdict") == "HORS_SUJET"
        hdr_cls = "q-bad" if has_hs else ""

        detail_sections.append(f"""
<div class="q-sec {hdr_cls}">
  <div class="q-hdr">
    <span class="q-num">Q{row}</span>
    <span class="q-reg" style="background:{REG_BG[reg]};color:{REG_COLORS[reg]}">{ORIENT_LABEL[reg]}</span>
    <span class="q-txt">{q}</span>
    <span class="q-delta" style="color:{dq_clr}">Δ = {dq_s}</span>
  </div>
  <div class="q-cols">
    {col_html(typ_e, "v_decomp")}
    {col_html(nt_e, "v_decomp_no_typing")}
  </div>
</div>""")

    detail_html = "\n".join(detail_sections)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Alignement registre — v_decomp vs no_typing — 12q</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;font-size:13px;background:#f8fafc;color:#1e293b;padding:24px;line-height:1.5}}
h2{{font-size:1.2rem;margin-bottom:4px}}
.sub{{color:#64748b;font-size:12px;margin-bottom:20px}}
/* Decision */
.decision-box{{background:#fff;border:2px solid #1e293b;border-radius:8px;
  padding:16px 20px;margin-bottom:28px;max-width:820px}}
.dtitle{{font-weight:800;font-size:0.95rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}}
.dnote{{font-size:11px;color:#64748b;margin-bottom:12px}}
.dtable{{border-collapse:collapse;font-size:13px;width:100%}}
.dtable th{{background:#f1f5f9;padding:6px 10px;text-align:left;border:1px solid #e2e8f0;
  font-size:10px;text-transform:uppercase;letter-spacing:.05em}}
.dtable td{{padding:7px 10px;border:1px solid #e2e8f0}}
.dl{{font-weight:600}}.dc{{text-align:center}}
.dcaption{{font-size:11px;color:#64748b;margin-top:8px}}
/* Sections */
.q-sec{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:16px;overflow:hidden}}
.q-sec.q-bad{{border-color:#fca5a5}}
.q-hdr{{padding:10px 16px;background:#f8fafc;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.q-num{{font-weight:800;color:#2980b9;font-size:1rem}}
.q-reg{{font-size:10px;font-weight:700;padding:2px 9px;border-radius:10px}}
.q-txt{{font-size:13px;font-weight:600;flex:1;min-width:200px}}
.q-delta{{font-size:13px;font-weight:700;white-space:nowrap}}
.q-cols{{display:grid;grid-template-columns:1fr 1fr}}
.col-inner{{padding:14px 16px;border-right:1px solid #e2e8f0}}
.col-inner:last-child{{border-right:none}}
.col-lbl{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px}}
.score-big{{font-size:2rem;font-weight:800;line-height:1;margin-bottom:6px}}
.score-unit{{font-size:1rem;font-weight:400;color:#64748b}}
.info-row{{margin:4px 0;font-size:12px}}
.info-lbl{{color:#64748b}}
.just-txt{{color:#374151;font-style:italic;line-height:1.4;margin:6px 0}}
/* Verdicts */
.vbadge{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:10px;font-weight:700}}
.vd-ok{{background:#dcfce7;color:#15803d}}
.vd-part{{background:#fef9c3;color:#854d0e}}
.vd-bad{{background:#fee2e2;color:#dc2626}}
.vd-unk{{background:#f1f5f9;color:#64748b}}
/* Réponse */
.ans-det{{margin-top:10px}}
.ans-sum{{font-size:11px;color:#3b82f6;cursor:pointer;list-style:none}}
.ans-sum::-webkit-details-marker{{display:none}}
.ans-sum::before{{content:"▶ "}}
details[open]>.ans-sum::before{{content:"▼ "}}
.ans-body{{margin-top:6px;padding:8px 10px;font-size:11px;line-height:1.7;color:#1e293b;
  white-space:pre-wrap;word-break:break-word;background:#f8fafc;
  border-left:3px solid #e2e8f0;max-height:500px;overflow-y:auto}}
.col-err{{padding:14px;color:#dc2626;font-size:12px}}
</style>
</head>
<body>
<h2>Alignement objectif/subjectif — v_decomp vs v_decomp_no_typing — 12 questions</h2>
<div class="sub">Généré le {ts} · Check GPT-4o-mini · Score 1–5 : le bon type de données (objectif vs subjectif) domine-t-il la réponse ?<br>
<span style="color:#64748b">Subjectif = verbatims + scores d'enquête citoyenne · Objectif = OppChoVec + indicateurs territoriaux</span></div>

{decision_html}

<h3 style="margin:0 0 12px;font-size:0.95rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em">Détail par question</h3>
{detail_html}

<p style="margin-top:20px;font-size:11px;color:#94a3b8">
  Réponses issues de ablations_103q_v43_gpt4o_COMPLET.json ·
  Relance : <code>python run_register_alignment_12q.py</code>
</p>
</body>
</html>"""
    return html

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    if args.report:
        with open(RESULTS, encoding="utf-8") as f:
            results = {int(k): v for k, v in json.load(f).items()}
        print("Résultats chargés.", flush=True)
    else:
        answers = load_answers()
        print(f"Check alignement registre — {len(TARGET_ROWS)}×{len(CONFIGS)} réponses (GPT-4o-mini)\n", flush=True)
        results = run_all(answers)
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nRésultats → {RESULTS}", flush=True)

    html = make_html(results)
    ts_f = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = OUT_DIR / f"rapport_register_alignment_{ts_f}.html"
    out.write_text(html, encoding="utf-8")
    print(f"HTML → {out}  ({round(out.stat().st_size/1024)} Ko)", flush=True)

if __name__ == "__main__":
    main()
