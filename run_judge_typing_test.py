"""
Passe le juge classique V4.3 sur les 12×2 réponses du test typing raptor.
Sources non disponibles dans les JSONs → juge sans contexte source (noté dans le rapport).
Usage :
  python run_judge_typing_test.py           # évalue + génère HTML
  python run_judge_typing_test.py --report  # régénère HTML depuis résultats existants
"""
import sys, json, re, time, datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORK_DIR     = Path(__file__).parent
TEST_DIR     = WORK_DIR / "comparaisons_rag" / "test_typing_raptor"
COMPLET_PATH = WORK_DIR / "comparaisons_rag" / "ablations_103q_v43_gpt4o_COMPLET.json"
RESULTS_PATH = TEST_DIR / "judge_typing_results.json"
CONFIGS      = ["v_decomp_raptor", "v_decomp_raptor_no_typing"]
TARGET_ROWS  = [2, 4, 5, 6, 8, 9, 10, 11, 14, 15, 25, 35]

# ── Métadonnées questions ──────────────────────────────────────────────────────
def load_meta():
    with open(COMPLET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    for e in data["v_decomp_raptor"]:
        if e["excel_row"] in TARGET_ROWS:
            meta[e["excel_row"]] = {
                "question":      e["question"],
                "section":       e.get("section", ""),
                "subsection":    e.get("subsection", ""),
                "expected_type": e.get("expected_type", "reponse_substantielle_attendue"),
            }
    return meta

# ── Chargement réponses existantes ───────────────────────────────────────────
def load_answers():
    results = {}  # row → {cfg → entry}
    for cfg in CONFIGS:
        for row in TARGET_ROWS:
            p = TEST_DIR / f"Q{row:03d}_{cfg}.json"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                e = json.load(f)
            results.setdefault(row, {})[cfg] = e
    return results

# ── Évaluation juge ───────────────────────────────────────────────────────────
def run_judge(answers, meta):
    import eval_from_excel as ev

    all_results = {}
    for row in TARGET_ROWS:
        m = meta.get(row, {})
        q  = m.get("question", "")
        sec   = m.get("section", "")
        subsec = m.get("subsection", "")
        etype  = m.get("expected_type", "reponse_substantielle_attendue")
        all_results[row] = {"question": q, "section": sec, "subsection": subsec, "configs": {}}

        for cfg in CONFIGS:
            entry = answers.get(row, {}).get(cfg)
            if not entry:
                print(f"  Q{row:03d} [{cfg}] : MANQUANT — ignoré")
                continue
            answer = entry.get("answer", "")
            cfg_short = "TYP" if "no_typing" not in cfg else "NT "
            print(f"  Q{row:03d} [{cfg_short}] juge...", end="", flush=True)
            t0 = time.time()
            result = ev.score_judge_v43(
                question=q,
                answer=answer,
                sources=[],       # sources non disponibles
                section=sec,
                subsection=subsec,
                expected_type=etype,
            )
            elapsed = round(time.time() - t0, 1)
            result["elapsed_s"] = elapsed
            all_results[row]["configs"][cfg] = result
            score = result.get("score_global")
            mis = result.get("mislabelling_detecte") or result.get("mislabelling_flag")
            print(f" score={score} mis={mis} ({elapsed}s)")

    return all_results

# ── HTML ──────────────────────────────────────────────────────────────────────
def score_color(s):
    if s is None:  return "var(--muted)"
    if s >= 4.5:   return "#16a34a"
    if s >= 3.5:   return "#65a30d"
    if s >= 2.5:   return "#ca8a04"
    return "#dc2626"

def is_mis(entry):
    if entry is None: return False
    if "mislabelling_flag" in entry:
        return bool(entry["mislabelling_flag"])
    md = entry.get("mislabelling_detecte") or {}
    if not md: return False
    return any(str(v).lower() not in ("non","false","","null","none","0") for v in md.values())

ORIENT_LABEL = {
    "objective": "Objectif — QUANTI",
    "perception": "Perception — QUALI",
    "vaste": "Vaste — BOTH",
}

def make_html(all_results, answers_meta):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Tableau de synthèse ───────────────────────────────────────────────────
    rows_typ, rows_nt = [], []
    scores_typ, scores_nt = [], []
    mis_typ, mis_nt = 0, 0

    for row in TARGET_ROWS:
        rd = all_results.get(row, {})
        typ_r = rd.get("configs", {}).get("v_decomp_raptor", {})
        nt_r  = rd.get("configs", {}).get("v_decomp_raptor_no_typing", {})
        st = typ_r.get("score_global")
        sn = nt_r.get("score_global")
        if st is not None: scores_typ.append(st)
        if sn is not None: scores_nt.append(sn)
        if is_mis(typ_r): mis_typ += 1
        if is_mis(nt_r):  mis_nt  += 1

    avg_typ = round(sum(scores_typ)/len(scores_typ), 2) if scores_typ else None
    avg_nt  = round(sum(scores_nt)/len(scores_nt), 2)  if scores_nt  else None

    def synth_row(row):
        rd = all_results.get(row, {})
        q  = rd.get("question", f"Q{row}")[:80]
        sec = rd.get("subsection", "")[:35]
        typ_r = rd.get("configs", {}).get("v_decomp_raptor", {})
        nt_r  = rd.get("configs", {}).get("v_decomp_raptor_no_typing", {})
        st = typ_r.get("score_global"); sn = nt_r.get("score_global")
        mt = is_mis(typ_r); mn = is_mis(nt_r)
        delta = round(sn - st, 2) if st is not None and sn is not None else None
        d_color = "var(--neg)" if (delta is not None and delta < -0.25) else \
                  "var(--pos)" if (delta is not None and delta > 0.25) else "var(--muted)"
        sc = lambda s: f'<span style="font-weight:700;color:{score_color(s)}">{s if s is not None else "—"}</span>'
        mis_badge = lambda m: '<span class="mis-badge">MIS</span>' if m else ""
        return (
            f'<tr>'
            f'<td class="qnum">Q{row}</td>'
            f'<td class="qtext">{q}</td>'
            f'<td class="sec">{sec}</td>'
            f'<td class="score-cell">{sc(st)}{mis_badge(mt)}</td>'
            f'<td class="score-cell">{sc(sn)}{mis_badge(mn)}</td>'
            f'<td class="delta" style="color:{d_color}">{("+" if delta and delta>0 else "")+str(delta) if delta is not None else "—"}</td>'
            f'</tr>'
        )

    synth_html = "".join(synth_row(r) for r in TARGET_ROWS)
    sc_avg = lambda s: f'<span style="font-weight:800;color:{score_color(s)}">{s if s is not None else "—"}</span>'

    # ── Détail par question ───────────────────────────────────────────────────
    def dim_table(res):
        if not res: return "<em>—</em>"
        dims = [("pertinence","Pertinence"),("fondement_factuel","Fond. factuel"),
                ("nuance_incertitude","Nuance"),("coherence_qualiquanti","Cohé. quali/quanti")]
        rows = ""
        for k, lbl in dims:
            note = res.get(k)
            just = (res.get(f"{k}_justification") or (res.get("justifications") or {}).get(k) or "")[:200]
            clr = score_color(note)
            rows += f'<tr><td class="dim-lbl">{lbl}</td><td class="dim-note" style="color:{clr}">{note if note is not None else "—"}</td><td class="dim-just">{just}</td></tr>'
        return f'<table class="dim-table">{rows}</table>'

    def mis_detail(res):
        md = res.get("mislabelling_detecte") or {}
        mf = res.get("mislabelling_flag")
        if mf is not None:
            return f'<span class="{"mis-val-oui" if mf else "mis-val-non"}">{"OUI" if mf else "non"}</span>'
        if not md: return '<span class="mis-val-non">non</span>'
        parts = []
        for k, v in md.items():
            oui = str(v).lower() not in ("non","false","","null","none","0")
            parts.append(f'<span class="{"mis-val-oui" if oui else "mis-val-non"}">{k}: {v}</span>')
        return " ".join(parts)

    def col_block(res, cfg_label):
        if not res:
            return f'<div class="col-empty">Résultat manquant</div>'
        score = res.get("score_global")
        rais  = (res.get("raisonnement") or "")[:400]
        sec_obs = res.get("section_observee") or "—"
        comp_att = res.get("comportement_attendu_selon_grille") or "—"
        is_viol = is_mis(res)
        border = 'border-left:3px solid #dc2626' if is_viol else 'border-left:3px solid #e2e8f0'
        return f"""
<div class="col-block" style="{border}">
  <div class="col-label">{cfg_label}</div>
  <div class="score-big" style="color:{score_color(score)}">{score if score is not None else "—"}<span class="score-big-unit">/5</span></div>
  <div class="mis-row">Mislabelling : {mis_detail(res)}</div>
  <div class="detail-block">
    <div class="detail-lbl">Section observée</div>
    <div class="detail-val">{sec_obs}</div>
  </div>
  <div class="detail-block">
    <div class="detail-lbl">Comportement attendu selon grille</div>
    <div class="detail-val">{comp_att[:200]}</div>
  </div>
  <div class="detail-block">
    <div class="detail-lbl">Notes par dimension</div>
    {dim_table(res)}
  </div>
  <details class="rais-details">
    <summary class="rais-toggle">Raisonnement du juge</summary>
    <div class="rais-body">{rais}</div>
  </details>
</div>"""

    detail_sections = []
    for row in TARGET_ROWS:
        rd = all_results.get(row, {})
        q   = rd.get("question", f"Q{row}")
        sec = rd.get("section", "")
        sub = rd.get("subsection", "")
        typ_r = rd.get("configs", {}).get("v_decomp_raptor", {})
        nt_r  = rd.get("configs", {}).get("v_decomp_raptor_no_typing", {})
        st = typ_r.get("score_global")
        sn = nt_r.get("score_global")
        header_cls = "bad" if (is_mis(typ_r) or is_mis(nt_r)) else ""
        # orientation from original JSON
        orig = answers_meta.get(row, {})
        orient = orig.get("orientation", "")
        orient_lbl = ORIENT_LABEL.get(orient, orient)

        detail_sections.append(f"""
<div class="q-section {header_cls}">
  <div class="q-header">
    <span class="q-num">Q{row}</span>
    <span class="orient-badge orient-{orient}">{orient_lbl}</span>
    <span class="sub-badge">{sub}</span>
    <span class="q-text">{q}</span>
  </div>
  <div class="q-cols">
    {col_block(typ_r, "v_decomp_raptor (TYPAGE)")}
    {col_block(nt_r,  "v_decomp_raptor_no_typing (NO_TYPING)")}
  </div>
</div>""")

    detail_html = "\n".join(detail_sections)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Juge V4.3 — test typing raptor</title>
<style>
:root{{
  --bg:#f8fafc;--surface:#fff;--surface-2:#f1f5f9;--border:#e2e8f0;
  --text:#1e293b;--muted:#64748b;--accent:#3b82f6;
  --pos:#16a34a;--neg:#dc2626;--warn:#ca8a04;
  --h-font:"Georgia",serif;
}}
@media(prefers-color-scheme:dark){{
  :root{{--bg:#0f172a;--surface:#1e293b;--surface-2:#334155;--border:#475569;
    --text:#f1f5f9;--muted:#94a3b8}}
}}
:root[data-theme="light"]{{--bg:#f8fafc;--surface:#fff;--surface-2:#f1f5f9;--border:#e2e8f0;--text:#1e293b;--muted:#64748b}}
:root[data-theme="dark"]{{--bg:#0f172a;--surface:#1e293b;--surface-2:#334155;--border:#475569;--text:#f1f5f9;--muted:#94a3b8}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;padding:24px}}
h1{{font-family:var(--h-font);font-size:1.5rem;margin-bottom:4px}}
.subtitle{{color:var(--muted);font-size:0.85rem;margin-bottom:20px}}
.warning-box{{background:color-mix(in srgb,var(--warn) 10%,var(--surface));border:1px solid var(--warn);
  border-radius:6px;padding:10px 14px;margin-bottom:20px;font-size:13px;color:var(--text)}}
/* Synthèse */
.synth-table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:32px}}
.synth-table th{{background:var(--surface-2);padding:8px 10px;text-align:left;
  font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}
.synth-table td{{padding:7px 10px;border-top:1px solid var(--border)}}
.synth-table tr:hover td{{background:var(--surface-2)}}
.qnum{{font-weight:700;color:var(--accent);white-space:nowrap}}
.qtext{{max-width:360px}}
.sec{{color:var(--muted);font-size:11px;max-width:180px}}
.score-cell{{text-align:center;white-space:nowrap}}
.delta{{text-align:center;font-size:13px;font-weight:600}}
.mis-badge{{display:inline-block;background:#fef2f2;color:#dc2626;border:1px solid #fca5a5;
  border-radius:3px;font-size:9px;padding:1px 4px;margin-left:4px;font-weight:700}}
.avg-row td{{border-top:2px solid var(--border);font-weight:700;padding-top:10px}}
/* Détail */
.q-section{{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  margin-bottom:20px;overflow:hidden}}
.q-section.bad{{border-color:#fca5a5}}
.q-header{{padding:12px 16px;background:var(--surface-2);display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.q-num{{font-weight:800;color:var(--accent);font-size:1rem}}
.orient-badge{{border-radius:12px;padding:2px 10px;font-size:11px;font-weight:600}}
.orient-objective{{background:#dbeafe;color:#1d4ed8}}
.orient-perception{{background:#fce7f3;color:#9d174d}}
.orient-vaste{{background:#e0e7ff;color:#4338ca}}
.sub-badge{{font-size:11px;color:var(--muted);background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:1px 8px}}
.q-text{{font-size:14px;font-weight:600;flex:1;min-width:200px}}
.q-cols{{display:grid;grid-template-columns:1fr 1fr;gap:0}}
.col-block{{padding:16px;border-right:1px solid var(--border)}}
.col-block:last-child{{border-right:none}}
.col-label{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:10px}}
.score-big{{font-size:2.2rem;font-weight:800;line-height:1}}
.score-big-unit{{font-size:1rem;font-weight:400;color:var(--muted)}}
.mis-row{{font-size:12px;color:var(--muted);margin:6px 0 12px}}
.mis-val-oui{{color:#dc2626;font-weight:700}}
.mis-val-non{{color:var(--muted)}}
.detail-block{{margin-bottom:10px}}
.detail-lbl{{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:2px}}
.detail-val{{font-size:12px;color:var(--text)}}
.dim-table{{width:100%;border-collapse:collapse;font-size:11px;margin-top:4px}}
.dim-table td{{padding:3px 6px;border-top:1px solid var(--border)}}
.dim-lbl{{color:var(--muted);width:110px}}
.dim-note{{font-weight:700;width:28px;text-align:center}}
.dim-just{{color:var(--text)}}
.rais-details{{margin-top:10px}}
.rais-toggle{{font-size:11px;color:var(--accent);cursor:pointer;user-select:none}}
.rais-body{{background:var(--surface-2);border-left:2px solid var(--border);
  margin-top:6px;padding:8px 10px;font-size:11px;line-height:1.6;color:var(--text);
  white-space:pre-wrap;word-break:break-word}}
.col-empty{{padding:16px;color:var(--muted);font-size:13px}}
</style>
</head>
<body>
<h1>Juge V4.3 — Test typing raptor</h1>
<div class="subtitle">Généré le {ts} · 12 questions × 2 configs · v_decomp_raptor vs v_decomp_raptor_no_typing</div>

<div class="warning-box">
  ⚠️ <strong>Sources non disponibles</strong> : les sources brutes n'étant pas sauvegardées dans les JSONs du test,
  le juge V4.3 a été exécuté sans contexte source (<code>sources=[]</code>).
  La dimension <em>fondement_factuel</em> peut être sous-estimée ; les autres (pertinence, nuance, cohérence quali/quanti) restent valides.
</div>

<h2 style="margin-bottom:10px;font-size:1rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em">Tableau de synthèse</h2>
<div style="overflow-x:auto">
<table class="synth-table">
  <thead>
    <tr>
      <th>Q</th><th>Question</th><th>Sous-section</th>
      <th style="text-align:center">TYPAGE</th>
      <th style="text-align:center">NO_TYPING</th>
      <th style="text-align:center">Δ</th>
    </tr>
  </thead>
  <tbody>
    {synth_html}
    <tr class="avg-row">
      <td colspan="3" style="text-align:right;color:var(--muted);font-size:12px">Moyenne · Mislabelling</td>
      <td class="score-cell">{sc_avg(avg_typ)} &nbsp;<span style="font-size:11px;color:var(--neg)">{mis_typ}/12 mis</span></td>
      <td class="score-cell">{sc_avg(avg_nt)} &nbsp;<span style="font-size:11px;color:var(--neg)">{mis_nt}/12 mis</span></td>
      <td class="delta"></td>
    </tr>
  </tbody>
</table>
</div>

<h2 style="margin:24px 0 10px;font-size:1rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em">Détail par question</h2>
{detail_html}
</body>
</html>"""
    return html

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    report_only = "--report" in sys.argv

    if report_only:
        if not RESULTS_PATH.exists():
            print("Aucun résultat trouvé — lancez sans --report d'abord.")
            sys.exit(1)
        with open(RESULTS_PATH, encoding="utf-8") as f:
            all_results = {int(k): v for k, v in json.load(f).items()}
        print("Résultats chargés.")
    else:
        meta    = load_meta()
        answers = load_answers()
        print(f"Évaluation juge V4.3 sur {len(TARGET_ROWS)}×2 réponses (sans sources)…")
        all_results = run_judge(answers, meta)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nRésultats → {RESULTS_PATH}")

    # Charger orientations depuis les JSONs bruts
    answers_meta = {}
    for row in TARGET_ROWS:
        p = TEST_DIR / f"Q{row:03d}_v_decomp_raptor.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            answers_meta[row] = {"orientation": d.get("orientation", "")}

    html = make_html(all_results, answers_meta)
    ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = TEST_DIR / f"rapport_juge_typing_{ts_file}.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nHTML → {out}  ({round(out.stat().st_size/1024)} Ko)")

if __name__ == "__main__":
    main()
