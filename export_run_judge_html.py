"""Export un JSON ablations_Nq_run_judge_*.json en HTML consultable.
Supporte les nouveaux fichiers avec juge V2 + V4.1 (champs v41_*).
"""
import json, sys
from pathlib import Path
from datetime import datetime

JSON_FILE = r"comparaisons_rag/ablations_6q_run_judge_20260513_131420.json"
OUT_DIR   = "comparaisons_rag"

if len(sys.argv) > 1:
    JSON_FILE = sys.argv[1]

with open(JSON_FILE, encoding="utf-8") as f:
    data = json.load(f)

versions = list(data.keys())
questions_ordered = [entry.get("excel_row") for entry in data[versions[0]]]
idx = {ver: {e["excel_row"]: e for e in data[ver]} for ver in versions}

# Détecte si le fichier contient des scores V4.1
has_v41 = any(
    e.get("v41_score_global") is not None
    for e in data[versions[0]]
)

def esc(s): return str(s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def stars(n):
    if n is None: return "—"
    return "★"*int(n) + "☆"*(5-int(n))
def fmt_score(v):
    return f"{v:.2f}" if isinstance(v, float) else (str(v) if v is not None else "—")

ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
n_q  = len(questions_ordered)
out_path = Path(OUT_DIR) / f"ablations_{n_q}q_judge_review_{ts}.html"

# ── couleurs par config ──────────────────────────────────────────────────────
_COLORS = {
    "v_vanilla_k10":   "#c0392b",
    "v_vanilla_k25":   "#e67e22",
    "v_decomp":        "#27ae60",
    "v_decomp_raptor": "#2980b9",
}
def color(ver): return _COLORS.get(ver, "#555")

# ── calcul moyennes par config ───────────────────────────────────────────────
def mean_score(ver, key):
    vals = [e[key] for e in data[ver] if e.get(key) is not None]
    return round(sum(vals)/len(vals), 2) if vals else None

summary_v2  = {ver: mean_score(ver, "score_global")       for ver in versions}
summary_v41 = {ver: mean_score(ver, "v41_score_global")   for ver in versions} if has_v41 else {}

html = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Ablations {n_q}q — Juge{'s V2+V4.1' if has_v41 else ' V2'}</title>
<style>
  body{{font-family:Arial,sans-serif;font-size:13px;margin:20px;background:#f4f6f8;color:#222;}}
  h1{{color:#1a1a2e;font-size:17px;margin-bottom:4px;}}
  .subtitle{{color:#888;font-size:12px;margin-bottom:16px;}}
  /* Recap table */
  .recap{{background:white;border-radius:8px;padding:14px 16px;margin-bottom:20px;
          box-shadow:0 1px 3px rgba(0,0,0,.08);}}
  .recap h2{{font-size:13px;color:#1a1a2e;margin:0 0 10px;}}
  .recap table{{border-collapse:collapse;font-size:12px;}}
  .recap th{{background:#1a1a2e;color:white;padding:6px 12px;text-align:center;}}
  .recap th:first-child{{text-align:left;}}
  .recap td{{padding:6px 12px;border-bottom:1px solid #eee;text-align:center;}}
  .recap td:first-child{{text-align:left;}}
  .recap tr:last-child td{{border-bottom:none;}}
  /* TOC */
  .toc{{background:white;padding:10px 16px;border-radius:8px;margin-bottom:20px;
        box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:12px;}}
  .toc a{{color:#2980b9;text-decoration:none;margin:2px 8px 2px 0;display:inline-block;}}
  /* Question block */
  .q-block{{background:white;border-radius:8px;padding:16px;margin-bottom:28px;
            box-shadow:0 1px 4px rgba(0,0,0,.1);}}
  .q-header{{font-weight:bold;font-size:14px;color:#1a1a2e;margin-bottom:4px;}}
  .q-meta{{font-size:11px;color:#888;margin-bottom:14px;}}
  /* Summary table per question */
  .summary-table{{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:12px;}}
  .summary-table th{{background:#1a1a2e;color:white;padding:7px 10px;text-align:center;}}
  .summary-table th:first-child{{text-align:left;}}
  .summary-table td{{padding:6px 10px;border-bottom:1px solid #eee;text-align:center;}}
  .summary-table td:first-child{{text-align:left;}}
  .summary-table tr:last-child td{{border-bottom:none;}}
  .best-cell{{font-weight:bold;color:#27ae60;font-size:13px;}}
  .tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;
        font-weight:bold;text-transform:uppercase;letter-spacing:.4px;color:white;}}
  /* Detail cards */
  .detail-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;}}
  .detail-card{{border:1px solid #e0e0e0;border-radius:6px;padding:12px;background:#fafafa;}}
  .detail-card-header{{display:flex;align-items:center;justify-content:space-between;
                       margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #eee;}}
  .global-badge{{font-size:14px;font-weight:bold;padding:2px 9px;border-radius:5px;
                 color:white;background:#1a1a2e;}}
  .scores-row{{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;}}
  .score-box{{background:#f0f0f0;border-radius:5px;padding:4px 8px;text-align:center;min-width:60px;}}
  .score-lbl{{font-size:9px;color:#888;display:block;}}
  .score-val{{font-size:14px;font-weight:bold;}}
  .score-stars{{font-size:9px;color:#f39c12;}}
  /* V4.1 sub-section */
  .v41-section{{margin-top:8px;padding:8px;background:#f0f7ff;border-radius:5px;
                border-left:3px solid #2980b9;}}
  .v41-title{{font-size:10px;font-weight:bold;color:#2980b9;margin-bottom:6px;}}
  .v41-badge{{font-size:13px;font-weight:bold;color:white;background:#2980b9;
              padding:2px 8px;border-radius:4px;float:right;}}
  .et-row{{font-size:10px;color:#555;margin-top:4px;}}
  .et-precis{{color:#27ae60;font-weight:bold;}}
  .et-approx-ok{{color:#e67e22;}}
  .et-approx-nok{{color:#c0392b;font-weight:bold;}}
  .et-omis{{color:#c0392b;}}
  .raisonnement{{font-style:italic;color:#555;font-size:11px;padding:6px 9px;
                 background:#fffde7;border-left:3px solid #f1c40f;border-radius:4px;margin-bottom:8px;}}
  .answer{{font-size:11.5px;line-height:1.55;white-space:pre-wrap;max-height:380px;
           overflow-y:auto;color:#333;border-top:1px solid #eee;padding-top:8px;}}
  .meta-bar{{font-size:10px;color:#bbb;text-align:right;margin-top:6px;}}
  .error{{color:#c0392b;font-style:italic;}}
</style></head><body>
<h1>Ablations RAG — Juges {'V2 + V4.1' if has_v41 else 'V2'} sur {n_q} questions</h1>
<p class="subtitle">Source : {Path(JSON_FILE).name} &nbsp;|&nbsp; {len(versions)} configs</p>
"""

# ── Tableau récap global ─────────────────────────────────────────────────────
html += '<div class="recap"><h2>Scores moyens par config</h2><table>\n'
html += '<tr><th>Config</th><th>Juge V2 (moy.)</th>'
if has_v41:
    html += '<th>Juge V4.1 (moy.)</th>'
html += '</tr>\n'
best_v2  = max((v for v in summary_v2.values()  if v), default=None)
best_v41 = max((v for v in summary_v41.values() if v), default=None) if has_v41 else None
for ver in versions:
    c = color(ver)
    v2  = summary_v2.get(ver)
    v41 = summary_v41.get(ver) if has_v41 else None
    cls_v2  = ' class="best-cell"' if v2  and v2  == best_v2  else ''
    cls_v41 = ' class="best-cell"' if v41 and v41 == best_v41 else ''
    html += (f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td>'
             f'<td{cls_v2}><b>{fmt_score(v2)}</b></td>')
    if has_v41:
        html += f'<td{cls_v41}><b>{fmt_score(v41)}</b></td>'
    html += '</tr>\n'
html += '</table></div>\n'

# ── TOC ──────────────────────────────────────────────────────────────────────
html += '<div class="toc"><b>Navigation</b> : '
for row in questions_ordered:
    html += f'<a href="#r{row}">R{row}</a> '
html += '</div>\n'

# ── Par question ─────────────────────────────────────────────────────────────
for row in questions_ordered:
    e0       = idx[versions[0]].get(row, {})
    question = e0.get("question", "")
    section  = e0.get("section", "")
    subsec   = e0.get("subsection", "")

    html += f'<div class="q-block" id="r{row}">\n'
    html += f'<div class="q-header">R{row} — {esc(question)}</div>\n'
    html += f'<div class="q-meta">{esc(section)} › {esc(subsec)}</div>\n'

    # Tableau récap par config (V2 + V4.1 si dispo)
    dims       = ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti","score_global"]
    dim_labels = ["Pertinence","Factuel","Nuance","Quali/Q","V2 Global"]
    globals_v2  = [idx[v].get(row,{}).get("score_global")     for v in versions]
    globals_v41 = [idx[v].get(row,{}).get("v41_score_global") for v in versions] if has_v41 else []
    best_g_v2   = max((g for g in globals_v2  if g), default=None)
    best_g_v41  = max((g for g in globals_v41 if g), default=None) if globals_v41 else None

    html += '<table class="summary-table"><tr><th>Config</th>'
    for lbl in dim_labels: html += f'<th>{lbl}</th>'
    if has_v41: html += '<th>V4.1 Global</th>'
    html += '<th>Src</th></tr>\n'

    for ver in versions:
        e   = idx[ver].get(row, {})
        c   = color(ver)
        if e.get("rag_status") != "ok":
            html += f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td><td colspan="{len(dims)+1+(1 if has_v41 else 0)}" class="error">Erreur RAG</td></tr>\n'
            continue
        html += f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td>'
        for d in dims[:-1]:
            v = e.get(d)
            html += f'<td>{v}/5 {stars(v)}</td>' if v else '<td>—</td>'
        sg  = e.get("score_global")
        cls = ' class="best-cell"' if sg and sg == best_g_v2 else ''
        html += f'<td{cls}><b>{fmt_score(sg)}</b></td>'
        if has_v41:
            sg41 = e.get("v41_score_global")
            cls41 = ' class="best-cell"' if sg41 and sg41 == best_g_v41 else ''
            html += f'<td{cls41}><b>{fmt_score(sg41)}</b></td>'
        html += f'<td>{e.get("n_sources","—")}</td></tr>\n'
    html += '</table>\n'

    # Cartes détail
    html += '<div class="detail-grid">\n'
    for ver in versions:
        e  = idx[ver].get(row, {})
        c  = color(ver)
        sg = e.get("score_global")
        html += '<div class="detail-card">\n'
        html += f'<div class="detail-card-header"><span class="tag" style="background:{c}">{esc(ver)}</span>'
        html += f'<span class="global-badge">{fmt_score(sg)} / 5</span></div>\n'

        if e.get("rag_status") != "ok":
            html += f'<div class="error">Erreur : {esc(e.get("rag_error","?"))}</div>\n'
        else:
            # Scores V2
            html += '<div class="scores-row">\n'
            for d, lbl in [("pertinence","Pertinence"),("fondement_factuel","Factuel"),
                           ("nuance_incertitude","Nuance"),("coherence_qualiquanti","Quali/Q")]:
                v = e.get(d)
                html += (f'<div class="score-box"><span class="score-lbl">{lbl}</span>'
                         f'<span class="score-val">{v if v else "?"}</span>'
                         f'<span class="score-stars">{stars(v)}</span></div>\n')
            html += '</div>\n'
            if e.get("raisonnement"):
                html += f'<div class="raisonnement">V2 🤖 {esc(e["raisonnement"])}</div>\n'

            # Scores V4.1
            if has_v41 and e.get("v41_score_global") is not None:
                sg41 = e.get("v41_score_global")
                html += f'<div class="v41-section"><div class="v41-title">Juge V4.1 <span class="v41-badge">{fmt_score(sg41)}/5</span></div>\n'
                html += '<div class="scores-row">\n'
                for d, lbl in [("v41_pertinence","Pertinence"),("v41_fondement_factuel","Factuel"),
                               ("v41_nuance_incertitude","Nuance"),("v41_coherence_qualiquanti","Quali/Q")]:
                    v = e.get(d)
                    html += (f'<div class="score-box"><span class="score-lbl">{lbl}</span>'
                             f'<span class="score-val">{v if v else "?"}</span>'
                             f'<span class="score-stars">{stars(v)}</span></div>\n')
                html += '</div>\n'
                if e.get("v41_raisonnement"):
                    html += f'<div class="raisonnement">V4.1 🤖 {esc(e["v41_raisonnement"])}</div>\n'
                # elements_traitement
                ets = e.get("v41_elements_traitement", [])
                if ets:
                    _cls = {"precis":"et-precis","approximation_signalee":"et-approx-ok",
                            "approximation_non_signalee":"et-approx-nok","omis":"et-omis"}
                    parts = []
                    for et in ets:
                        if isinstance(et, dict):
                            t = et.get("traitement","?")
                            parts.append(f'<span class="{_cls.get(t,"")}">{esc(et["element"])}={t}</span>')
                    html += f'<div class="et-row">Éléments : {" · ".join(parts)}</div>\n'
                html += '</div>\n'

            html += f'<div class="answer">{esc(e.get("answer",""))}</div>\n'
            j_t   = e.get("judge_elapsed_s","?")
            j41_t = e.get("judge_v41_elapsed_s","")
            j41_part = f" | V4.1 {j41_t}s" if j41_t else ""
            html += f'<div class="meta-bar">{e.get("n_sources","?")} src | RAG {e.get("rag_elapsed_s","?")}s | V2 {j_t}s{j41_part}</div>\n'
        html += '</div>\n'
    html += '</div>\n</div>\n\n'

html += "</body></html>\n"

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"HTML : {out_path}")
