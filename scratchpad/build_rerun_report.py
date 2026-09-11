"""
build_rerun_report.py
Rapport HTML complet après re-run synthèse tronquée.
Charge automatiquement le fichier RERUN le plus récent.
Affiche : synthèse globale, par section, et détail par question (P/FF/NI/CQ + réponse).
Ajoute un check de troncature sur toutes les questions de tous les configs.
"""
import json, re, sys, html as H
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# ── Fichiers
BASE_DIR = Path("comparaisons_rag")
REJUDGE  = BASE_DIR / "ablations_103q_v43_gpt4o_REJUDGE_20260814_192707.json"

# Trouver le RERUN le plus récent
# Prendre le fichier RERUN le plus récent (RERUN2 prioritaire sur RERUN)
rerun_files = sorted(
    BASE_DIR.glob("ablations_103q_v43_gpt4o_RERUN*.json"),
    key=lambda p: p.stat().st_mtime
)
if not rerun_files:
    sys.exit("Aucun fichier RERUN trouvé dans comparaisons_rag/")
RERUN = rerun_files[-1]
OUT   = Path(r"C:\Users\comiti_g\Downloads\rerun_rapport_complet.html")

print(f"Base    : {REJUDGE.name}")
print(f"RERUN   : {RERUN.name}")

with open(REJUDGE, encoding="utf-8") as f: old_data = json.load(f)
with open(RERUN,   encoding="utf-8") as f: new_data = json.load(f)

CONFIGS  = ["v_decomp_raptor", "v_decomp", "v_vanilla_k10", "v_vanilla_k25"]
CFG_LBL  = {"v_vanilla_k10": "Vanilla k=10", "v_vanilla_k25": "Vanilla k=25",
             "v_decomp": "Décomp", "v_decomp_raptor": "Décomp+RAPTOR"}
CFG_COL  = {"v_vanilla_k10": "#C0392B", "v_vanilla_k25": "#E67E22",
             "v_decomp": "#27AE60", "v_decomp_raptor": "#2980B9"}
DIMS     = ["pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti"]
DIM_LBL  = ["Pertinence", "Fond. factuel", "Nuance/Inc.", "Cohér. Q/Q"]
SECTIONS = [
    "Retrieval mono-commune",
    "Raisonnement causal et contre-intuitif",
    "Raisonnement comparatif",
    "Gestion d'absence d'information",
    "Gestion de l'incertitude et des biais",
    "Robustesse sémantique",
    "Limites architecturales",
]

def norm_sec(s):
    s = (s or "").strip()
    if "absence" in s.lower() and "information" in s.lower():
        return "Gestion d'absence d'information"
    return s

def trunc_sev(ans):
    s = ans.strip()
    if re.search(r"\*\*[A-Za-zÀ-ÿ0-9\s,]{0,40}$", s):
        if not re.search(r"\*\*[^*]+\*\*\s*$", s):
            return 3
    if not re.search(r"[.!?»)\]]\*{0,2}\s*$|---\s*$|={3,}\s*$", s):
        return 2
    return 0

def mean(vals):
    v = [float(x) for x in vals if x is not None]
    return round(sum(v) / len(v), 3) if v else None

def sc(v):
    if v is None: return "#9E9A90"
    v = float(v)
    if v >= 4.75: return "#1A6B3A"
    if v >= 4.4:  return "#2E8B57"
    if v >= 4.0:  return "#D4A017"
    if v >= 3.5:  return "#D4641A"
    return "#C0392B"

def fmt(v, bold=False):
    if v is None: return "—"
    b = "font-weight:900;" if bold else ""
    c = sc(v)
    return f'<span style="color:{c};{b}">{float(v):.2f}</span>'

def dfmt(old, new):
    if old is None or new is None: return "—"
    d = float(new) - float(old)
    c = "#1A6B3A" if d > 0.02 else ("#C0392B" if d < -0.02 else "#667080")
    w = "font-weight:800;" if abs(d) > 0.15 else ""
    sign = "+" if d >= 0 else ""
    return f'<span style="color:{c};{w}">{sign}{d:.2f}</span>'

# ── Index par question (config → row → entry)
def idx(data):
    out = {}
    for cfg in CONFIGS:
        out[cfg] = {e["excel_row"]: e for e in data.get(cfg, [])}
    return out

old_idx = idx(old_data)
new_idx = idx(new_data)

# ── Scores globaux
def global_scores(data_idx):
    res = {}
    for cfg in CONFIGS:
        entries = list(data_idx[cfg].values())
        res[cfg] = {
            "global": mean([e.get("score_global") for e in entries]),
            **{d: mean([e.get(d) for e in entries]) for d in DIMS},
        }
    return res

old_gs = global_scores(old_idx)
new_gs = global_scores(new_idx)

# ─────────────────────────────────────────────
# HTML SECTIONS
# ─────────────────────────────────────────────

# 1. TABLEAU GLOBAL
global_rows_html = ""
for cfg in CONFIGS:
    og = old_gs[cfg]; ng = new_gs[cfg]
    lbl = CFG_LBL[cfg]; col = CFG_COL[cfg]
    rerun_mark = "✓ re-run" if cfg in ["v_decomp_raptor", "v_decomp"] else "—"
    dim_cells = "".join(
        f'<td>{fmt(og[d])}</td><td>{fmt(ng[d], True)}</td><td>{dfmt(og[d], ng[d])}</td>'
        for d in DIMS
    )
    global_rows_html += (
        f'<tr><td style="color:{col};font-weight:700">{lbl}</td>'
        f'<td>{fmt(og["global"])}</td><td>{fmt(ng["global"], True)}</td>'
        f'<td>{dfmt(og["global"], ng["global"])}</td>'
        f'{dim_cells}'
        f'<td style="color:var(--muted);font-size:11px">{rerun_mark}</td></tr>'
    )

# 2. CHECK TRONCATURE toutes configs + les deux jeux de données
# Grouper par config et sévérité pour un affichage clair
trunc_by_cfg = {}
for cfg in CONFIGS:
    entries = list(new_idx[cfg].values())
    s3 = []; s2 = []
    for e in sorted(entries, key=lambda x: x["excel_row"]):
        ans = e.get("answer", "")
        sv = trunc_sev(ans)
        if sv == 3: s3.append(e)
        elif sv == 2: s2.append(e)
    trunc_by_cfg[cfg] = {"s3": s3, "s2": s2}

trunc_count = sum(len(v["s3"]) + len(v["s2"]) for v in trunc_by_cfg.values())

def trunc_rows(entries, sev_icon):
    html = ""
    for e in entries:
        ans = e.get("answer", "")
        end_repr = H.escape(ans.strip()[-65:])
        rerun_tag = '<span style="color:#2980B9;font-size:10px">re-run</span>' if e.get("rerun_synthesis") else ""
        html += (
            f'<tr><td>{sev_icon}</td>'
            f'<td>Q{e["excel_row"]}</td>'
            f'<td style="font-size:11px">{H.escape(e.get("question","")[:65])}…</td>'
            f'<td>{len(ans)}ch</td>'
            f'{rerun_tag and f"<td>{rerun_tag}</td>" or "<td></td>"}'
            f'<td><code style="font-size:10px">…{end_repr}</code></td></tr>'
        )
    return html

cfg_blocks_html = ""
for cfg in CONFIGS:
    s3 = trunc_by_cfg[cfg]["s3"]
    s2 = trunc_by_cfg[cfg]["s2"]
    total_cfg = len(s3) + len(s2)
    if total_cfg == 0:
        cfg_blocks_html += f'<div style="margin-bottom:12px"><strong style="color:{CFG_COL[cfg]}">{CFG_LBL[cfg]}</strong> <span style="color:#1A6B3A">✓ Aucune troncature</span></div>'
        continue
    note = ""
    if cfg in ["v_vanilla_k10", "v_vanilla_k25"]:
        note = '<span style="color:var(--muted);font-size:11px"> — max_tokens=1000, pipeline vanilla non re-run</span>'
    elif s2 and not any(e.get("rerun_synthesis") for e in s2):
        note = '<span style="color:#D4A017;font-size:11px"> — sev=2 manqués par bug détection (:{0,1}), re-run à faire</span>'
    rows_html = trunc_rows(s3, "❌ sev3") + trunc_rows(s2, "⚠ sev2")
    cfg_blocks_html += f"""
<div style="margin-bottom:16px">
  <strong style="color:{CFG_COL[cfg]}">{CFG_LBL[cfg]}</strong>
  — {len(s3)} sev=3 / {len(s2)} sev=2{note}
  <div style="overflow-x:auto;margin-top:6px">
  <table>
    <thead><tr>
      <th>Sév.</th><th>Q</th><th>Question</th><th>Len</th><th></th><th>Fin</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table></div>
</div>"""

trunc_summary = (
    f'<p style="color:#1A6B3A;font-weight:700;margin-bottom:8px">✓ Aucune troncature dans le backup RERUN.</p>'
    if trunc_count == 0 else
    f'<p style="color:#C0392B;font-weight:700;margin-bottom:8px">⚠ {trunc_count} réponse(s) encore tronquée(s) — détail par config ci-dessous.</p>'
)
trunc_table = cfg_blocks_html

# 3. DÉTAIL PAR QUESTION (configs re-run seulement)
all_rows_for_rowid = {}
for cfg in CONFIGS:
    for e in new_data.get(cfg, []):
        rid = e["excel_row"]
        if rid not in all_rows_for_rowid:
            all_rows_for_rowid[rid] = e  # pour la question text

question_cards_html = ""
for cfg in CONFIGS:
    lbl = CFG_LBL[cfg]; col = CFG_COL[cfg]
    new_entries = sorted(new_data.get(cfg, []), key=lambda x: x["excel_row"])
    question_cards_html += f'<h3 style="color:{col};margin:28px 0 12px">{lbl}</h3>'

    for e_new in new_entries:
        row = e_new["excel_row"]
        e_old = old_idx[cfg].get(row, {})
        is_rerun = e_new.get("rerun_synthesis", False) or e_new.get("rerun_synthesis2", False)

        q_text = H.escape(e_new.get("question", ""))
        sec    = H.escape(norm_sec(e_new.get("section", "")))
        subsec = H.escape(e_new.get("subsection", "") or "")

        old_sg = e_old.get("score_global")
        new_sg = e_new.get("score_global")
        old_ans = e_old.get("answer", "")
        new_ans = e_new.get("answer", "")

        # réponse nouvelle (complète, collapsible)
        new_ans_html = H.escape(new_ans).replace("\n", "<br>")
        old_end_html = H.escape(old_ans.strip()[-100:]) if old_ans else "—"

        rerun_badge = (
            '<span class="badge-rerun">re-run</span>' if is_rerun
            else '<span class="badge-unchanged">inchangé</span>'
        )
        sev = trunc_sev(new_ans)
        sev_html = "" if sev == 0 else (
            '<span class="badge-trunc">⚠ encore tronquée</span>' if sev == 2
            else '<span class="badge-trunc">❌ encore tronquée (sev=3)</span>'
        )

        # Dimensions
        dim_table = '<table class="dim-table"><thead><tr><th>Dim</th><th>Ancien</th><th>Nouveau</th><th>Δ</th></tr></thead><tbody>'
        for d, dl in zip(DIMS, DIM_LBL):
            ov = e_old.get(d); nv = e_new.get(d)
            dim_table += f'<tr><td>{dl}</td><td>{fmt(ov)}</td><td>{fmt(nv, True)}</td><td>{dfmt(ov, nv)}</td></tr>'
        dim_table += f'<tr style="border-top:2px solid var(--rule)"><td><strong>Global</strong></td><td>{fmt(old_sg)}</td><td>{fmt(new_sg, True)}</td><td>{dfmt(old_sg, new_sg)}</td></tr>'
        dim_table += "</tbody></table>"

        uid = f"q{row}_{cfg}"
        question_cards_html += f"""
<div class="qcard" id="{uid}">
  <div class="qcard-header">
    <span class="qnum">Q{row}</span>
    {rerun_badge}{sev_html}
    <span class="qsec">{sec} · {subsec}</span>
    <span class="qscore-pill">{fmt(old_sg)} → {fmt(new_sg, True)} ({dfmt(old_sg, new_sg)})</span>
  </div>
  <p class="qtext">{q_text}</p>
  <div class="qbody">
    <div class="col-dims">{dim_table}</div>
    <div class="col-answer">
      <div class="ans-label">Réponse nouvelle · {len(new_ans)} chars · {'' if sev==0 else '⚠ '}<code style="font-size:10px">…{H.escape(new_ans.strip()[-60:])}</code></div>
      <details>
        <summary>Voir la réponse complète</summary>
        <div class="ans-body">{new_ans_html}</div>
      </details>
      {'<div class="old-end-label">Fin ancienne réponse : <code>' + old_end_html + '</code></div>' if is_rerun else ''}
    </div>
  </div>
</div>"""

# 4. SYNTHÈSE NARRATIVE
total_rerun = sum(
    1 for cfg in CONFIGS for e in new_data.get(cfg, [])
    if e.get("rerun_synthesis") or e.get("rerun_synthesis2")
)
dr_old = old_gs["v_decomp_raptor"]["global"]
dr_new = new_gs["v_decomp_raptor"]["global"]
dc_old = old_gs["v_decomp"]["global"]
dc_new = new_gs["v_decomp"]["global"]

gen_date = datetime.now().strftime("%Y-%m-%d %H:%M")

# ─────────────────────────────────────────────
# HTML FINAL
# ─────────────────────────────────────────────
HTML = f"""<title>Re-run synthèse tronquée</title>
<style>
:root{{
  --paper:#F0EEE9;--card:#FAFAF7;--ink:#1A1E2A;--muted:#667080;
  --rule:#D8D4CB;--sh:0 1px 3px rgba(0,0,0,.06),0 4px 14px rgba(0,0,0,.04);
}}
@media(prefers-color-scheme:dark){{
  :root:not([data-theme=light]){{
    --paper:#111520;--card:#181E30;--ink:#DDE1EC;--muted:#7A8BAA;--rule:#242A40;
  }}
}}
:root[data-theme=dark]{{--paper:#111520;--card:#181E30;--ink:#DDE1EC;--muted:#7A8BAA;--rule:#242A40}}
:root[data-theme=light]{{--paper:#F0EEE9;--card:#FAFAF7;--ink:#1A1E2A;--muted:#667080;--rule:#D8D4CB}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,sans-serif;background:var(--paper);color:var(--ink);font-size:13.5px;line-height:1.6}}

.topbar{{position:sticky;top:0;z-index:100;background:var(--card);border-bottom:1px solid var(--rule);padding:10px 20px;display:flex;align-items:center;gap:12px}}
.tb-title{{font-size:14px;font-weight:800}}
.tb-sub{{font-size:11.5px;color:var(--muted)}}
.theme-btn{{margin-left:auto;border:1px solid var(--rule);background:transparent;border-radius:5px;padding:3px 9px;cursor:pointer;font-size:11px;color:var(--muted)}}

main{{max-width:1300px;margin:0 auto;padding:24px 20px 80px;display:flex;flex-direction:column;gap:28px}}
h2{{font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:12px}}
h3{{font-size:14px;font-weight:700}}

.box{{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:20px;box-shadow:var(--sh)}}
.callout{{background:color-mix(in srgb,#2980B9 8%,var(--card));border-left:3px solid #2980B9;border-radius:0 6px 6px 0;padding:14px 18px;font-size:13px}}

table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12.5px}}
th{{font-size:10.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);padding:6px 10px;text-align:right;border-bottom:2px solid var(--rule)}}
th:first-child{{text-align:left}}
td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--rule)}}
td:first-child{{text-align:left}}
tr:last-child td{{border-bottom:none}}

/* QUESTION CARDS */
.qcard{{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:16px 20px;box-shadow:var(--sh);margin-bottom:12px}}
.qcard-header{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}}
.qnum{{font-size:13px;font-weight:900;color:var(--muted)}}
.qsec{{font-size:11px;color:var(--muted);margin-left:auto}}
.qscore-pill{{font-size:12px;background:color-mix(in srgb,var(--ink) 5%,var(--card));border-radius:20px;padding:2px 10px}}
.qtext{{font-weight:600;margin-bottom:12px;font-size:13.5px}}
.qbody{{display:grid;grid-template-columns:340px 1fr;gap:20px}}
@media(max-width:800px){{.qbody{{grid-template-columns:1fr}}}}

.dim-table{{width:100%;font-size:12px}}
.dim-table th,.dim-table td{{padding:4px 8px}}
.dim-table tr:last-child td{{font-size:13px}}

.col-answer{{min-width:0}}
.ans-label{{font-size:11px;color:var(--muted);margin-bottom:6px}}
.ans-body{{font-size:12.5px;line-height:1.65;background:color-mix(in srgb,var(--ink) 3%,var(--card));border-radius:6px;padding:12px;margin-top:8px;max-height:400px;overflow-y:auto;white-space:pre-wrap}}
.old-end-label{{font-size:10.5px;color:var(--muted);margin-top:6px}}
details summary{{cursor:pointer;font-size:12px;color:var(--muted)}}

.badge-rerun{{background:#2980B9;color:#fff;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px}}
.badge-unchanged{{background:color-mix(in srgb,var(--ink) 12%,var(--card));color:var(--muted);font-size:10px;border-radius:4px;padding:2px 7px}}
.badge-trunc{{background:#C0392B;color:#fff;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px}}

code{{font-family:monospace;font-size:11px;background:color-mix(in srgb,var(--ink) 6%,var(--card));padding:1px 4px;border-radius:3px}}
</style>

<div class="topbar">
  <div>
    <div class="tb-title">Re-run synthèse — correction troncature layer 1</div>
    <div class="tb-sub">{total_rerun} réponses re-générées (max_tokens 2500→6000) · Re-jugées V4.3 gpt-4o · Généré le {gen_date}</div>
  </div>
  <button class="theme-btn" onclick="(()=>{{let r=document.documentElement,t=r.getAttribute('data-theme');r.setAttribute('data-theme',t==='dark'?'light':'dark')}})()">⬤</button>
</div>

<main>

<div class="callout">
  <strong>Contexte :</strong> le synthétiseur Mistral était limité à <code>max_tokens=2500</code>
  (~6 500 chars), tronquant les réponses longues en plein milieu d'une phrase ou d'un titre gras.
  Ce re-run relance la génération avec <code>max_tokens=6000</code> pour les {total_rerun} réponses
  identifiées comme tronquées (sev≥2), puis les re-juge avec le juge V4.3 complet.
</div>

<!-- SCORES GLOBAUX -->
<div class="box">
  <h2>Scores globaux — toutes configs (109 questions)</h2>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>Config</th>
      <th>Ancien global</th><th>Nouveau global</th><th>Δ global</th>
      {''.join(f'<th colspan="3" style="text-align:center">{dl}</th>' for dl in DIM_LBL)}
      <th>Re-run</th>
    </tr>
    <tr>
      <th></th><th></th><th></th><th></th>
      {''.join('<th style="font-size:9px;color:var(--muted)">Anc.</th><th style="font-size:9px;color:var(--muted)">Nouv.</th><th style="font-size:9px;color:var(--muted)">Δ</th>' for _ in DIMS)}
      <th></th>
    </tr></thead>
    <tbody>{global_rows_html}</tbody>
  </table></div>
  <p style="font-size:11px;color:var(--muted);margin-top:8px">
    Vanilla : non re-run (réponses courtes, pas de troncature synthèse).
    Les comparaisons sont par rapport au backup REJUDGE (déjà corrigé pour [:4000]).
  </p>
</div>

<!-- CHECK TRONCATURE -->
<div class="box">
  <h2>Check troncature — toutes réponses, tous configs (backup RERUN)</h2>
  {trunc_summary}
  <div style="background:color-mix(in srgb,#D4A017 8%,var(--card));border-left:3px solid #D4A017;border-radius:0 6px 6px 0;padding:10px 14px;font-size:12px;margin-bottom:14px">
    <strong>Note :</strong> le script de re-run avait un bug de détection (<code>:{0,1}\\s*$</code>)
    qui rendait sev=2 jamais déclenchée. Seules les entrées sev=3 (bold non fermé) ont été re-générées.
    Les sev=2 dans Décomp/DR restent dans l'état du backup REJUDGE (non re-générées). Un second pass est nécessaire.
    Le vanilla (pipeline différent, <code>max_tokens=1000</code>) n'a pas été re-run.
  </div>
  {trunc_table}
</div>

<!-- DETAIL PAR QUESTION -->
<div class="box">
  <h2>Détail par question — Décomp+RAPTOR et Décomp</h2>
  <p style="font-size:11px;color:var(--muted);margin-bottom:16px">
    Badge <span class="badge-rerun">re-run</span> = réponse re-générée.
    Badge <span class="badge-unchanged">inchangé</span> = réponse non tronquée, scores conservés.
  </p>
  {question_cards_html}
</div>

</main>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"\nRapport → {OUT}")
print(f"Re-run count : {total_rerun}")
print(f"Troncatures restantes : {trunc_count}")
