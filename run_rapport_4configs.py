"""
run_rapport_4configs.py — Rapport HTML comparatif 4 configs sur 12 questions.

Configs : v_selfask · v_decomp_raptor · v_vanilla_k10 · v_vanilla_k25
Sources  : comparaisons_rag/selfask_12q/selfask_q{row}.json + COMPLET.json
Pas de nouveau run RAG ni juge.

Relance : python run_rapport_4configs.py
"""

import io, json, sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

# ── Constantes ────────────────────────────────────────────────────────────────

SELFASK_DIR = Path("comparaisons_rag/selfask_12q")
COMPLET     = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
DL_DIR      = Path(r"C:\Users\comiti_g\Downloads")
TS          = datetime.now().strftime("%Y%m%d_%H%M%S")

TARGET_12Q = [
    {"excel_row": 4,  "orientation": "objectif",   "question": "Quel est le score moyen de bien-être à Ajaccio ?"},
    {"excel_row": 5,  "orientation": "objectif",   "question": "Quelle dimension OppChoVec obtient la note la plus faible à Ajaccio ?"},
    {"excel_row": 6,  "orientation": "objectif",   "question": "Quel est le score OppChoVec d'Ajaccio, par catégorie ?"},
    {"excel_row": 8,  "orientation": "objectif",   "question": "Combien d'habitants ont répondu à l'enquête à Ajaccio ?"},
    {"excel_row": 15, "orientation": "objectif",   "question": "De combien de services de proximité dispose la ville d'Ajaccio ?"},
    {"excel_row": 9,  "orientation": "perception", "question": "Comment les 18-25 ans ressentent-ils le bien-être ?"},
    {"excel_row": 10, "orientation": "perception", "question": "Que pensent les entrepreneurs Ajacciens de la qualité de vie ?"},
    {"excel_row": 11, "orientation": "perception", "question": "Que pensent les seniors du bien-être à Ajaccio ?"},
    {"excel_row": 14, "orientation": "perception", "question": "Que révèlent les verbatims sur la sécurité à Corte ?"},
    {"excel_row": 2,  "orientation": "vaste",      "question": "Peut-on considérer Ajaccio comme un territoire favorable au bien-être ?"},
    {"excel_row": 25, "orientation": "vaste",      "question": "Les indicateurs objectifs et les perceptions qualitatives convergent-ils ?"},
    {"excel_row": 35, "orientation": "vaste",      "question": "Observe-t-on un écart significatif entre indicateurs objectifs et perception à Bastia ?"},
]

ORIENT_ROWS  = {"objectif": [4,5,6,8,15], "perception": [9,10,11,14], "vaste": [2,25,35]}
DIMS         = ["pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti"]
DIM_LABELS   = {"pertinence": "Pertinence", "fondement_factuel": "Factuel",
                "nuance_incertitude": "Nuance", "coherence_qualiquanti": "Quali/Quanti"}
ORIENT_LABELS = {"objectif": "Objectif", "perception": "Perception", "vaste": "Vaste"}

CFGS = [
    ("v_selfask",       "Self-Ask",       "sa"),
    ("v_decomp_raptor", "Decomp+Raptor",  "dr"),
    ("v_vanilla_k10",   "Vanilla k=10",   "vk10"),
    ("v_vanilla_k25",   "Vanilla k=25",   "vk25"),
]

# ── Chargement données ─────────────────────────────────────────────────────────

print("Chargement des données...")

# Self-Ask depuis JSONs individuels
sa_by_row = {}
for q in TARGET_12Q:
    p = SELFASK_DIR / f"selfask_q{q['excel_row']:03d}.json"
    if p.exists():
        sa_by_row[q["excel_row"]] = json.loads(p.read_text(encoding="utf-8"))

# COMPLET.json pour les 3 autres configs
complet = json.loads(COMPLET.read_text(encoding="utf-8"))
dr_by_row   = {e["excel_row"]: e for e in complet.get("v_decomp_raptor",[])}
vk10_by_row = {e["excel_row"]: e for e in complet.get("v_vanilla_k10",[])}
vk25_by_row = {e["excel_row"]: e for e in complet.get("v_vanilla_k25",[])}

ALL_DATA = {
    "v_selfask":       sa_by_row,
    "v_decomp_raptor": dr_by_row,
    "v_vanilla_k10":   vk10_by_row,
    "v_vanilla_k25":   vk25_by_row,
}

def _get(cfg_key, row, field):
    return ALL_DATA[cfg_key].get(row, {}).get(field)

def _ml_flag(entry):
    if entry is None: return False
    flag = entry.get("mislabelling_flag")
    if flag is not None: return bool(flag)
    ml = entry.get("mislabelling_detecte") or {}
    return any(str(v).lower() not in ("non","false","","null","none") for v in ml.values())

def _avg(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    return round(sum(v)/len(v), 2) if v else None

def _delta(a, b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)): return round(a-b, 2)
    return None

def _fmt(v, digits=2):
    return f"{v:.{digits}f}" if isinstance(v,(int,float)) else "—"

def _score_cls(v):
    if not isinstance(v,(int,float)): return "na"
    if v >= 4.5: return "hi"
    if v >= 3.5: return "mid"
    return "lo"

def _delta_cls(d):
    if not isinstance(d,(int,float)): return "na"
    if d > 0.25:  return "pos"
    if d < -0.25: return "neg"
    return "neu"

def _sign(v):
    return "+" if isinstance(v,(int,float)) and v > 0 else ""

# ── Calcul stats ───────────────────────────────────────────────────────────────

rows_all = [q["excel_row"] for q in TARGET_12Q]

# Moyennes globales par config × dim
def cfg_avgs(cfg_key):
    avgs = {d: _avg([_get(cfg_key, r, d) for r in rows_all]) for d in DIMS}
    avgs["score_global"] = _avg([_get(cfg_key, r, "score_global") for r in rows_all])
    avgs["ml_n"] = sum(1 for r in rows_all if _ml_flag(ALL_DATA[cfg_key].get(r)))
    return avgs

all_avgs = {k: cfg_avgs(k) for k,_,_ in CFGS}

# Moyennes par orientation
def orient_avgs(cfg_key, orient_key):
    ori_rows = ORIENT_ROWS[orient_key]
    present  = [r for r in ori_rows if r in ALL_DATA[cfg_key]]
    return _avg([_get(cfg_key, r, "score_global") for r in present])

# ── HTML ───────────────────────────────────────────────────────────────────────

def build_html() -> str:
    n = len(rows_all)

    # ── Global summary table ─────────────────────────────────────────────────
    dim_headers = "".join(f"<th>{DIM_LABELS[d]}</th>" for d in DIMS)

    def summary_row(cfg_key, label, css):
        avgs = all_avgs[cfg_key]
        dim_cells = "".join(
            f'<td class="num sc-{_score_cls(avgs[d])}">{_fmt(avgs[d])}</td>'
            for d in DIMS
        )
        g = avgs["score_global"]
        ml_n = avgs["ml_n"]
        return (
            f'<tr>'
            f'<td class="cfg-name cfg-{css}">{label}</td>'
            f'{dim_cells}'
            f'<td class="num bold sc-{_score_cls(g)}">{_fmt(g)}</td>'
            f'<td class="num">{ml_n}/{n}</td>'
            f'</tr>'
        )

    summary_rows = "\n".join(summary_row(k, lbl, css) for k,lbl,css in CFGS)

    # ── Delta rows vs v_decomp_raptor ────────────────────────────────────────
    dr_avgs = all_avgs["v_decomp_raptor"]
    delta_rows_html = ""
    for cfg_key, lbl, css in CFGS:
        if cfg_key == "v_decomp_raptor": continue
        avgs = all_avgs[cfg_key]
        dim_cells = ""
        for d in DIMS:
            dv = _delta(avgs[d], dr_avgs[d])
            dim_cells += f'<td class="num delta-{_delta_cls(dv)}">{_sign(dv)}{_fmt(dv)}</td>'
        sg_d = _delta(avgs["score_global"], dr_avgs["score_global"])
        ml_d = avgs["ml_n"] - dr_avgs["ml_n"]
        ml_s = f"{ml_d:+d}" if isinstance(ml_d, int) else "—"
        delta_rows_html += (
            f'<tr class="delta-row">'
            f'<td class="cfg-name cfg-{css}">Δ {lbl} − Raptor</td>'
            f'{dim_cells}'
            f'<td class="num bold delta-{_delta_cls(sg_d)}">{_sign(sg_d)}{_fmt(sg_d)}</td>'
            f'<td class="num">{ml_s}</td>'
            f'</tr>'
        )

    # ── Orientation table ────────────────────────────────────────────────────
    orient_html = ""
    for ori in ["objectif","perception","vaste"]:
        cells = ""
        ref = orient_avgs("v_decomp_raptor", ori)
        for cfg_key, _, css in CFGS:
            v = orient_avgs(cfg_key, ori)
            dv = _delta(v, ref) if cfg_key != "v_decomp_raptor" else None
            d_str = (f" <span class='small delta-{_delta_cls(dv)}'>{_sign(dv)}{_fmt(dv)}</span>"
                     if dv is not None else "")
            cells += f'<td class="num sc-{_score_cls(v)}">{_fmt(v)}{d_str}</td>'
        q_ids = ", ".join(f"Q{r:03d}" for r in ORIENT_ROWS[ori])
        orient_html += (
            f'<tr>'
            f'<td><span class="badge badge-{ori}">{ORIENT_LABELS[ori]}</span></td>'
            f'<td class="qlist">{q_ids}</td>'
            f'{cells}'
            f'</tr>'
        )

    # ── Per-question big table ────────────────────────────────────────────────
    q_meta = {q["excel_row"]: q for q in TARGET_12Q}
    q_rows_html = ""
    for q in TARGET_12Q:
        row = q["excel_row"]
        ori = q["orientation"]
        cells = ""
        scores = {}
        for cfg_key, _, css in CFGS:
            g = _get(cfg_key, row, "score_global")
            scores[cfg_key] = g
            cells += f'<td class="num sc-{_score_cls(g)}">{_fmt(g)}</td>'
        # delta vs raptor for non-raptor configs
        delta_cells = ""
        dr_g = scores.get("v_decomp_raptor")
        for cfg_key, _, css in CFGS:
            if cfg_key == "v_decomp_raptor":
                delta_cells += "<td></td>"
            else:
                dv = _delta(scores[cfg_key], dr_g)
                delta_cells += f'<td class="num delta-{_delta_cls(dv)}">{_sign(dv)}{_fmt(dv)}</td>'

        q_rows_html += (
            f'<tr>'
            f'<td class="q-num-cell">Q{row:03d}</td>'
            f'<td><span class="badge badge-{ori}">{ORIENT_LABELS[ori]}</span></td>'
            f'<td class="q-text-cell">{q["question"]}</td>'
            f'{cells}'
            f'{delta_cells}'
            f'</tr>'
        )

    # ── Per-question detail (collapsible) ─────────────────────────────────────
    detail_sections = ""
    for q in TARGET_12Q:
        row = q["excel_row"]
        ori = q["orientation"]

        # Dim score grid
        dim_grid = '<table class="dim-grid"><thead><tr><th>Dim</th>'
        for _, lbl, _ in CFGS:
            dim_grid += f'<th>{lbl}</th>'
        dim_grid += '</tr></thead><tbody>'
        for d in DIMS:
            dim_grid += f'<tr><td class="dim-lbl">{DIM_LABELS[d]}</td>'
            for cfg_key,_,_ in CFGS:
                v = _get(cfg_key, row, d)
                dim_grid += f'<td class="num sc-{_score_cls(v)}">{_fmt(v)}</td>'
            dim_grid += '</tr>'
        dim_grid += '<tr class="global-row"><td class="dim-lbl"><strong>Global</strong></td>'
        for cfg_key,_,_ in CFGS:
            g = _get(cfg_key, row, "score_global")
            dim_grid += f'<td class="num bold sc-{_score_cls(g)}">{_fmt(g)}</td>'
        dim_grid += '</tr>'
        dim_grid += '<tr><td class="dim-lbl">Mislabelling</td>'
        for cfg_key,_,_ in CFGS:
            e = ALL_DATA[cfg_key].get(row)
            fl = _ml_flag(e)
            dim_grid += f'<td class="num sc-{"lo" if fl else "hi"}">{"✗" if fl else "✓"}</td>'
        dim_grid += '</tr></tbody></table>'

        # Self-Ask hops
        sa_e = sa_by_row.get(row, {})
        hops = sa_e.get("hops", [])
        hops_html = ""
        for h in hops:
            srcs = ", ".join(s.get("source_type","?") for s in h.get("sources",[]))
            ia = h.get("intermediate_answer","")
            hops_html += (
                f'<div class="hop">'
                f'<div class="hop-hdr">Hop {h["hop"]} — <em>{h["follow_up"]}</em></div>'
                f'<div class="hop-src">Sources : {srcs or "—"}</div>'
                f'<div class="hop-ia">{ia[:350]}{"…" if len(ia)>350 else ""}</div>'
                f'</div>'
            )
        if not hops_html:
            hops_html = '<p class="note">0 hop — réponse directe (aucune retrieval)</p>'

        # Sa score chip for summary line
        sa_g   = _get("v_selfask",       row, "score_global")
        dr_g   = _get("v_decomp_raptor", row, "score_global")
        vk10_g = _get("v_vanilla_k10",   row, "score_global")
        vk25_g = _get("v_vanilla_k25",   row, "score_global")

        detail_sections += f"""
<details class="q-detail">
  <summary class="q-summary">
    <span class="q-num">Q{row:03d}</span>
    <span class="badge badge-{ori}">{ORIENT_LABELS[ori]}</span>
    <span class="q-text">{q['question']}</span>
    <span class="q-chips">
      <span class="chip cfg-sa">{_fmt(sa_g)}</span>
      <span class="chip cfg-dr">{_fmt(dr_g)}</span>
      <span class="chip cfg-vk10">{_fmt(vk10_g)}</span>
      <span class="chip cfg-vk25">{_fmt(vk25_g)}</span>
    </span>
  </summary>
  <div class="q-body">
    {dim_grid}
    <details class="inner-det">
      <summary>▸ Self-Ask — {len(hops)} hop(s)</summary>
      <div class="hops-wrap">{hops_html}</div>
      <div class="ans-box ans-sa">{(sa_e.get('final_answer','') or '')[:700]}{'…' if len(sa_e.get('final_answer','') or '')>700 else ''}</div>
    </details>
    <details class="inner-det">
      <summary>▸ Decomp+Raptor — {dr_by_row.get(row,{}).get('n_subquestions','?')} sous-questions</summary>
      <div class="ans-box ans-dr">{(dr_by_row.get(row,{}).get('answer','') or '')[:700]}{'…' if len(dr_by_row.get(row,{}).get('answer','') or '')>700 else ''}</div>
    </details>
    <details class="inner-det">
      <summary>▸ Vanilla k=10</summary>
      <div class="ans-box ans-vk10">{(vk10_by_row.get(row,{}).get('answer','') or '')[:700]}{'…' if len(vk10_by_row.get(row,{}).get('answer','') or '')>700 else ''}</div>
    </details>
    <details class="inner-det">
      <summary>▸ Vanilla k=25</summary>
      <div class="ans-box ans-vk25">{(vk25_by_row.get(row,{}).get('answer','') or '')[:700]}{'…' if len(vk25_by_row.get(row,{}).get('answer','') or '')>700 else ''}</div>
    </details>
  </div>
</details>
"""

    # ── Full HTML ─────────────────────────────────────────────────────────────
    cfg_legend = " · ".join(f'<span class="cfg-{css} bold">{lbl}</span>' for _,lbl,css in CFGS)

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>4 configs RAG — 12 questions pilote</title>
<style>
:root {{
  --bg:#f4f5f8;--bg2:#fff;--bg3:#ecedf2;--border:#d0d3de;
  --text:#1a1d2e;--text2:#4a4e6b;--text3:#8891b2;
  --hi:#1e7e44;--hi-bg:#d4edda;--mid:#856404;--mid-bg:#fff3cd;
  --lo:#842029;--lo-bg:#f8d7da;
  --pos:#1e7e44;--pos-bg:#d4edda;--neg:#842029;--neg-bg:#f8d7da;--neu:#4a4e6b;
  --sa:#3a7bd5;--dr:#e07b39;--vk10:#2e7d62;--vk25:#7b52ab;
  --r:6px;
}}
@media(prefers-color-scheme:dark){{
  :root{{--bg:#0d1117;--bg2:#161b22;--bg3:#1f2430;--border:#30363d;
  --text:#c9d1d9;--text2:#8b949e;--text3:#484f58;
  --hi:#3fb950;--hi-bg:#0d2818;--mid:#d29922;--mid-bg:#271e00;
  --lo:#f85149;--lo-bg:#2d0d0c;
  --pos:#3fb950;--pos-bg:#0d2818;--neg:#f85149;--neg-bg:#2d0d0c;--neu:#8b949e;
  --sa:#6ea8fe;--dr:#f4a261;--vk10:#4ade80;--vk25:#c084fc;}}
}}
:root[data-theme="dark"]{{--bg:#0d1117;--bg2:#161b22;--bg3:#1f2430;--border:#30363d;--text:#c9d1d9;--text2:#8b949e;--text3:#484f58;--hi:#3fb950;--hi-bg:#0d2818;--mid:#d29922;--mid-bg:#271e00;--lo:#f85149;--lo-bg:#2d0d0c;--pos:#3fb950;--pos-bg:#0d2818;--neg:#f85149;--neg-bg:#2d0d0c;--neu:#8b949e;--sa:#6ea8fe;--dr:#f4a261;--vk10:#4ade80;--vk25:#c084fc;}}
:root[data-theme="light"]{{--bg:#f4f5f8;--bg2:#fff;--bg3:#ecedf2;--border:#d0d3de;--text:#1a1d2e;--text2:#4a4e6b;--text3:#8891b2;--hi:#1e7e44;--hi-bg:#d4edda;--mid:#856404;--mid-bg:#fff3cd;--lo:#842029;--lo-bg:#f8d7da;--pos:#1e7e44;--pos-bg:#d4edda;--neg:#842029;--neg-bg:#f8d7da;--neu:#4a4e6b;--sa:#3a7bd5;--dr:#e07b39;--vk10:#2e7d62;--vk25:#7b52ab;}}

*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,sans-serif;font-size:14px;background:var(--bg);color:var(--text);line-height:1.55}}
.wrap{{max-width:1300px;margin:0 auto;padding:1.5rem 1rem}}
.ph{{margin-bottom:2rem}}
.ph h1{{font-size:1.4rem;font-weight:700;margin-bottom:.3rem}}
.ph .meta{{font-size:.8rem;color:var(--text2)}}
section{{margin-bottom:2.5rem}}
h2{{font-size:.9rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text2);margin-bottom:.9rem;padding-bottom:.4rem;border-bottom:1px solid var(--border)}}
.tscroll{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
th{{background:var(--bg3);color:var(--text2);font-weight:600;text-align:left;padding:.4rem .65rem;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}}
td{{padding:.38rem .65rem;border-bottom:1px solid var(--border);vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
.num{{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace;white-space:nowrap}}
.bold{{font-weight:700}}
.small{{font-size:.78em}}

/* Config colors */
.cfg-name{{font-weight:600;white-space:nowrap}}
.cfg-sa,.bold.cfg-sa{{color:var(--sa)}}
.cfg-dr,.bold.cfg-dr{{color:var(--dr)}}
.cfg-vk10,.bold.cfg-vk10{{color:var(--vk10)}}
.cfg-vk25,.bold.cfg-vk25{{color:var(--vk25)}}

/* Score cells */
.sc-hi{{color:var(--hi);background:var(--hi-bg);border-radius:3px}}
.sc-mid{{color:var(--mid);background:var(--mid-bg);border-radius:3px}}
.sc-lo{{color:var(--lo);background:var(--lo-bg);border-radius:3px}}
.sc-na{{color:var(--text3)}}
.delta-pos{{color:var(--pos);font-weight:600}}
.delta-neg{{color:var(--neg);font-weight:600}}
.delta-neu{{color:var(--neu)}}
.delta-na{{color:var(--text3)}}
.delta-row td{{background:var(--bg3)}}

/* Badges */
.badge{{display:inline-block;padding:.12em .45em;border-radius:3px;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em}}
.badge-objectif{{background:#dbeafe;color:#1d4ed8}}
.badge-perception{{background:#fce7f3;color:#9d174d}}
.badge-vaste{{background:#d1fae5;color:#065f46}}
@media(prefers-color-scheme:dark){{
  .badge-objectif{{background:#1e3a5f;color:#93c5fd}}
  .badge-perception{{background:#4a1942;color:#f9a8d4}}
  .badge-vaste{{background:#064e3b;color:#6ee7b7}}
}}
:root[data-theme="dark"] .badge-objectif{{background:#1e3a5f;color:#93c5fd}}
:root[data-theme="dark"] .badge-perception{{background:#4a1942;color:#f9a8d4}}
:root[data-theme="dark"] .badge-vaste{{background:#064e3b;color:#6ee7b7}}
:root[data-theme="light"] .badge-objectif{{background:#dbeafe;color:#1d4ed8}}
:root[data-theme="light"] .badge-perception{{background:#fce7f3;color:#9d174d}}
:root[data-theme="light"] .badge-vaste{{background:#d1fae5;color:#065f46}}

/* Summary table */
.sum-th-cfg{{width:200px}}

/* Big question table */
.q-num-cell{{font-family:ui-monospace,monospace;font-size:.8rem;color:var(--text2);white-space:nowrap}}
.q-text-cell{{max-width:320px}}
.qlist{{font-family:ui-monospace,monospace;font-size:.75rem;color:var(--text3)}}

/* Detail sections */
.q-detail{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);margin-bottom:.5rem;overflow:hidden}}
.q-summary{{display:flex;align-items:center;gap:.6rem;padding:.65rem .9rem;cursor:pointer;user-select:none;flex-wrap:wrap}}
.q-summary::-webkit-details-marker{{display:none}}
.q-summary::before{{content:"▸";color:var(--text3);flex-shrink:0;transition:transform .15s}}
details[open] .q-summary::before{{transform:rotate(90deg)}}
.q-num{{font-family:ui-monospace,monospace;font-size:.8rem;color:var(--text2);flex-shrink:0}}
.q-text{{flex:1;font-weight:500;font-size:.85rem;min-width:0}}
.q-chips{{display:flex;gap:.25rem;flex-shrink:0;flex-wrap:wrap}}
.chip{{font-family:ui-monospace,monospace;font-size:.73rem;padding:.1em .42em;border-radius:3px;background:var(--bg3);font-weight:700}}
.chip.cfg-sa{{color:var(--sa)}}
.chip.cfg-dr{{color:var(--dr)}}
.chip.cfg-vk10{{color:var(--vk10)}}
.chip.cfg-vk25{{color:var(--vk25)}}
.q-body{{padding:.9rem;border-top:1px solid var(--border)}}

/* Dim grid */
.dim-grid{{width:auto;margin-bottom:.8rem;font-size:.82rem}}
.dim-grid th{{font-size:.72rem}}
.dim-lbl{{color:var(--text2);white-space:nowrap;padding-right:.8rem}}
.global-row td{{border-top:2px solid var(--border)}}

/* Inner details */
.inner-det{{margin-bottom:.5rem;border:1px solid var(--border);border-radius:var(--r);overflow:hidden}}
.inner-det>summary{{padding:.4rem .7rem;cursor:pointer;font-size:.8rem;color:var(--text2);font-weight:600;background:var(--bg3);user-select:none}}
.inner-det>summary::-webkit-details-marker{{display:none}}
.inner-det[open]>summary{{border-bottom:1px solid var(--border)}}

/* Hop chain */
.hops-wrap{{padding:.7rem}}
.hop{{margin-bottom:.7rem;border-left:3px solid var(--sa);padding-left:.75rem}}
.hop-hdr{{font-weight:600;font-size:.82rem;margin-bottom:.2rem}}
.hop-hdr em{{font-weight:400;color:var(--text2)}}
.hop-src{{font-size:.73rem;color:var(--text3);font-family:ui-monospace,monospace;margin-bottom:.2rem}}
.hop-ia{{font-size:.79rem;color:var(--text2);background:var(--bg2);border-radius:3px;padding:.35rem .55rem}}

/* Answer boxes */
.ans-box{{font-size:.8rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;padding:.65rem;border-radius:0 var(--r) var(--r) 0;border-left:3px solid var(--border);margin:.5rem 0;background:var(--bg2)}}
.ans-sa{{border-left-color:var(--sa)}}
.ans-dr{{border-left-color:var(--dr)}}
.ans-vk10{{border-left-color:var(--vk10)}}
.ans-vk25{{border-left-color:var(--vk25)}}

.note{{font-size:.78rem;color:var(--text3);padding:.3rem .6rem;font-style:italic}}
.theme-btn{{position:fixed;top:.8rem;right:.8rem;background:var(--bg2);border:1px solid var(--border);color:var(--text);border-radius:20px;padding:.3rem .8rem;cursor:pointer;font-size:.8rem;z-index:100}}
</style>
</head>
<body>
<button class="theme-btn" onclick="const r=document.documentElement,c=r.getAttribute('data-theme');r.setAttribute('data-theme',c==='dark'?'light':'dark')">◑ Thème</button>

<div class="wrap">

<div class="ph">
  <h1>4 configurations RAG — Pilote 12 questions · Juge V4.3</h1>
  <div class="meta">
    {cfg_legend} ·
    Δ = config − Decomp+Raptor ·
    Généré le {TS[:4]}-{TS[4:6]}-{TS[6:8]} {TS[9:11]}:{TS[11:13]}
  </div>
</div>

<!-- ── RÉCAP GLOBAL ────────────────────────────────────────────── -->
<section>
  <h2>Moyennes globales — 12 questions</h2>
  <div class="tscroll">
  <table>
    <thead>
      <tr>
        <th class="sum-th-cfg">Config</th>
        {dim_headers}
        <th>Global</th>
        <th>Mislabelling</th>
      </tr>
    </thead>
    <tbody>
      {summary_rows}
      {delta_rows_html}
    </tbody>
  </table>
  </div>
</section>

<!-- ── PAR ORIENTATION ─────────────────────────────────────────── -->
<section>
  <h2>Score global moyen par orientation</h2>
  <p style="font-size:.78rem;color:var(--text2);margin-bottom:.6rem">Δ en indice = écart vs Decomp+Raptor</p>
  <div class="tscroll">
  <table>
    <thead>
      <tr>
        <th>Orientation</th>
        <th>Questions</th>
        {"".join(f'<th class="cfg-{css}">{lbl}</th>' for _,lbl,css in CFGS)}
      </tr>
    </thead>
    <tbody>
      {orient_html}
    </tbody>
  </table>
  </div>
</section>

<!-- ── TABLE SYNTHÉTIQUE ──────────────────────────────────────── -->
<section>
  <h2>Scores par question</h2>
  <div class="tscroll">
  <table>
    <thead>
      <tr>
        <th>Q</th>
        <th>Orient.</th>
        <th>Question</th>
        {"".join(f'<th class="cfg-{css}">{lbl}</th>' for _,lbl,css in CFGS)}
        {"".join(f'<th>Δ {lbl[:2]}-DR</th>' for _,lbl,css in CFGS if css!="dr")}
      </tr>
    </thead>
    <tbody>
      {q_rows_html}
    </tbody>
  </table>
  </div>
</section>

<!-- ── DÉTAIL PAR QUESTION ────────────────────────────────────── -->
<section>
  <h2>Détail par question</h2>
  <p style="font-size:.78rem;color:var(--text2);margin-bottom:.7rem">
    Chips dans le résumé : <span class="chip cfg-sa">SA</span>
    <span class="chip cfg-dr">DR</span>
    <span class="chip cfg-vk10">Vk10</span>
    <span class="chip cfg-vk25">Vk25</span> — cliquer pour déplier.
  </p>
  {detail_sections}
</section>

</div>
</body>
</html>"""


html = build_html()

ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path("comparaisons_rag/selfask_12q") / f"rapport_4configs_{ts2}.html"
out.write_text(html, encoding="utf-8")
dl  = DL_DIR / f"rapport_4configs_{ts2}.html"
try:
    dl.write_text(html, encoding="utf-8")
    print(f"HTML → {out}")
    print(f"     → {dl}")
except Exception:
    print(f"HTML → {out}")
print("Relance : python run_rapport_4configs.py")
