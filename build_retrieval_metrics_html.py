"""
Calcule recall/précision/F1 pour les 103 questions × 4 configs
et génère un HTML avec heatmap.
Q104+ (limites architecturales récentes) exclues.
"""
import json, sys, io, openpyxl
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import eval_from_excel as ev
from eval_from_excel import score_retrieval, classify_source

XLSX = r'C:\Users\comiti_g\Downloads\rag_evaluation_with_metrics_full.xlsx'
JSON = 'comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json'
OUT  = Path('comparaisons_rag/retrieval_metrics_103q.html')

CONFIGS = ['v_vanilla_k10', 'v_vanilla_k25', 'v_decomp', 'v_decomp_raptor']
LABELS  = {'v_vanilla_k10': 'Vanilla k10', 'v_vanilla_k25': 'Vanilla k25',
           'v_decomp': 'Decomp', 'v_decomp_raptor': 'Decomp+Raptor'}
COLORS  = {'v_vanilla_k10': '#c0392b', 'v_vanilla_k25': '#e67e22',
           'v_decomp': '#27ae60', 'v_decomp_raptor': '#2980b9'}

# ── Charger GT depuis Excel ───────────────────────────────────────────────────
wb = openpyxl.load_workbook(XLSX)
ws = wb.active
gt_by_row = {}  # 1-based question index → GT text
for r in range(2, ws.max_row + 1):
    q = ws.cell(r, 3).value
    if q and str(q).strip():
        gt_by_row[r - 1] = ws.cell(r, 4).value  # col 4 = Documents à retrieve

print(f'{len(gt_by_row)} lignes GT chargées')

# ── Charger résultats RAG ─────────────────────────────────────────────────────
with open(JSON, encoding='utf-8') as f:
    data = json.load(f)

index = {}
for cfg, entries in data.items():
    for e in entries:
        index[(e['excel_row'], cfg)] = e

# ── Calculer métriques ────────────────────────────────────────────────────────
results = {}  # (row, cfg) → {recall, precision, f1, detail, ...}
skipped_no_gt = 0
skipped_limites = 0

all_rows = sorted({e['excel_row'] for e in data['v_vanilla_k10']})

for row in all_rows:
    if row >= 104:   # limites architecturales récentes
        skipped_limites += 1
        continue
    gt_text = gt_by_row.get(row)
    for cfg in CONFIGS:
        e = index.get((row, cfg), {})
        sources = e.get('sources', [])
        # classify_source attend des dicts avec metadata
        src_dicts = [s.get('metadata', s) if isinstance(s, dict) else {} for s in sources]
        r = score_retrieval(src_dicts, gt_text)
        results[(row, cfg)] = r

print(f'Métriques calculées : {len(results)} entrées')
print(f'Ignorées (limites arch. récentes) : {skipped_limites * len(CONFIGS)}')

# ── Agréger par section × config ─────────────────────────────────────────────
def clean_sec(s):
    s = (s or '').replace('‘', '').replace('’', "'").replace('′', "'")
    if 'absence' in s.lower() and 'information' in s.lower():
        return "Gestion de l'absence d'information"
    return s

def clean_sub(s):
    return (s or '').replace('‘', '').replace('’', "'").replace('′', "'").strip() or '—'

# Index section/subsection par row
sec_info = {}
for e in data['v_vanilla_k10']:
    sec_info[e['excel_row']] = (clean_sec(e.get('section', '')), clean_sub(e.get('subsection', '')))

def mean(lst): return sum(lst) / len(lst) if lst else None

def agg_metrics(rows_cfg_pairs):
    r_vals, p_vals, f_vals = [], [], []
    for row, cfg in rows_cfg_pairs:
        m = results.get((row, cfg), {})
        if m.get('recall') is not None:   r_vals.append(m['recall'])
        if m.get('precision') is not None: p_vals.append(m['precision'])
        if m.get('f1') is not None:        f_vals.append(m['f1'])
    return mean(r_vals), mean(p_vals), mean(f_vals), len(r_vals)

# Grouper par (section, subsection)
by_subsec = defaultdict(list)
for row in all_rows:
    if row >= 104: continue
    sec, sub = sec_info.get(row, ('?', '—'))
    by_subsec[(sec, sub)].append(row)

SEC_ORDER = [
    'Retrieval mono-commune',
    'Raisonnement comparatif',
    'Raisonnement causal et contre-intuitif',
    "Gestion de l'absence d'information",
    "Gestion de l'incertitude et des biais",
    'Robustesse sémantique',
    'Limites architecturales',
]

by_sec_grp = defaultdict(list)
for key in by_subsec:
    by_sec_grp[key[0]].append(key)

ordered_keys = []
for sec in SEC_ORDER:
    if sec in by_sec_grp:
        ordered_keys.extend(sorted(by_sec_grp[sec], key=lambda x: x[1]))
for sec in sorted(by_sec_grp):
    if sec not in SEC_ORDER:
        ordered_keys.extend(sorted(by_sec_grp[sec], key=lambda x: x[1]))

sec_rows_all = defaultdict(list)
for (sec, sub), rows in by_subsec.items():
    sec_rows_all[sec].extend(rows)

# ── Helpers HTML ──────────────────────────────────────────────────────────────
def bg(v):
    if v is None: return '#f5f5f5'
    if v >= 0.85: return '#d5f5e3'
    if v >= 0.70: return '#eafaf1'
    if v >= 0.55: return '#fef9e7'
    if v >= 0.40: return '#fef5e4'
    return '#fdecea'

def fg(v):
    if v is None: return '#aaa'
    if v >= 0.85: return '#1a7a40'
    if v >= 0.70: return '#27ae60'
    if v >= 0.55: return '#b7950b'
    if v >= 0.40: return '#ca6f1e'
    return '#c0392b'

def triple_cell(r, p, f, bold=False, n=None):
    """Génère 3 cellules : recall, précision, F1."""
    cells = ''
    for v in (r, p, f):
        if v is None:
            cells += '<td style="background:#f5f5f5;color:#aaa;text-align:center;font-size:0.82em">—</td>'
        else:
            fw = 'bold' if bold else 'normal'
            cells += (f'<td style="background:{bg(v)};color:{fg(v)};font-weight:{fw};'
                      f'text-align:center;font-size:0.85em">{v:.0%}</td>')
    return cells

# ── Corps du tableau ──────────────────────────────────────────────────────────
rows_html = ''
prev_sec = None
sec_idx = -1

for (sec, sub) in ordered_keys:
    if sec != prev_sec:
        if prev_sec is not None:
            # Ligne moyenne section
            rows_html += '<tr style="border-top:2px solid #bdc3c7">'
            rows_html += (f'<td colspan="2" style="padding:5px 14px;font-style:italic;'
                          f'color:#555;background:#fafafa">Moyenne {prev_sec}</td>')
            for cfg in CONFIGS:
                pairs = [(row, cfg) for row in sec_rows_all[prev_sec] if row < 104]
                r, p, f, n = agg_metrics(pairs)
                rows_html += triple_cell(r, p, f, bold=True)
            rows_html += '</tr><tr><td colspan="14" style="height:6px;background:#f0f2f5;border:none"></td></tr>'

        sec_idx += 1
        rows_html += (f'<tr id="sec{sec_idx}" style="background:#2c3e50">'
                      f'<td colspan="14" style="color:white;font-weight:bold;padding:8px 14px">'
                      f'{sec}</td></tr>')
        prev_sec = sec

    sub_rows = by_subsec[(sec, sub)]
    n_total = len(sub_rows)
    rows_html += (f'<tr><td style="padding:4px 14px 4px 26px">{sub}</td>'
                  f'<td style="color:#888;text-align:center;font-size:0.82em">{n_total}</td>')
    for cfg in CONFIGS:
        pairs = [(row, cfg) for row in sub_rows]
        r, p, f, n = agg_metrics(pairs)
        rows_html += triple_cell(r, p, f)
    rows_html += '</tr>'

# Dernière section
if prev_sec:
    rows_html += '<tr style="border-top:2px solid #bdc3c7">'
    rows_html += (f'<td colspan="2" style="padding:5px 14px;font-style:italic;'
                  f'color:#555;background:#fafafa">Moyenne {prev_sec}</td>')
    for cfg in CONFIGS:
        pairs = [(row, cfg) for row in sec_rows_all[prev_sec] if row < 104]
        r, p, f, n = agg_metrics(pairs)
        rows_html += triple_cell(r, p, f, bold=True)
    rows_html += '</tr><tr><td colspan="14" style="height:6px;background:#f0f2f5;border:none"></td></tr>'

# Total global
rows_html += '<tr style="background:#ecf0f1;border-top:3px solid #7f8c8d">'
rows_html += '<td style="font-weight:bold;padding:7px 14px">TOTAL</td><td style="color:#888;text-align:center">103</td>'
for cfg in CONFIGS:
    pairs = [(row, cfg) for row in all_rows if row < 104]
    r, p, f, n = agg_metrics(pairs)
    rows_html += triple_cell(r, p, f, bold=True)
rows_html += '</tr>'

# ── En-tête tableau ───────────────────────────────────────────────────────────
header_cells = ''
for cfg in CONFIGS:
    c = COLORS[cfg]
    lbl = LABELS[cfg]
    header_cells += (f'<th colspan="3" style="background:{c};color:white">{lbl}</th>')

subheader = '<th style="background:#ecf0f1;text-align:left">Sous-section</th><th style="background:#ecf0f1;color:#888">N</th>'
for _ in CONFIGS:
    subheader += '<th style="background:#ecf0f1;font-size:0.8em">R</th><th style="background:#ecf0f1;font-size:0.8em">P</th><th style="background:#ecf0f1;font-size:0.8em">F1</th>'

nav_html = ' &nbsp;·&nbsp; '.join(
    f'<a href="#sec{i}">{sec}</a>'
    for i, sec in enumerate(SEC_ORDER) if sec in by_sec_grp
)

# ── Imprimer quelques stats console ──────────────────────────────────────────
print()
print(f"{'Config':<18} {'Recall':>7} {'Précision':>10} {'F1':>7}  (sur questions avec GT)")
print('-' * 50)
for cfg in CONFIGS:
    pairs = [(row, cfg) for row in all_rows if row < 104]
    r, p, f, n = agg_metrics(pairs)
    print(f"{LABELS[cfg]:<18} {r:.1%}    {p:.1%}     {f:.1%}  (n={n})")

# ── HTML final ────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Retrieval Recall / Précision / F1 — 103 questions</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; margin: 0; background: #f0f2f5; }}
  .topbar {{ background: #2c3e50; color: white; padding: 12px 24px;
             position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 6px rgba(0,0,0,.3); }}
  .topbar h1 {{ margin: 0 0 5px; font-size: 1.1em; }}
  .topbar nav {{ font-size: 0.82em; opacity: .8; }}
  .topbar nav a {{ color: #aed6f1; text-decoration: none; }}
  .topbar nav a:hover {{ text-decoration: underline; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 20px 24px; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  th, td {{ border: 1px solid #e0e0e0; padding: 5px 7px; }}
  th {{ font-size: 0.82em; position: sticky; top: 58px; z-index: 10; }}
  tr:hover td {{ filter: brightness(0.96); }}
  .legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; font-size: 0.82em; }}
  .leg {{ padding: 3px 10px; border-radius: 4px; }}
  .note {{ color: #666; font-size: 0.85em; margin-bottom: 12px; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>Retrieval — Recall / Précision / F1 — 103 questions × 4 configs</h1>
  <nav>{nav_html}</nav>
</div>
<div class="container">
  <div class="legend">
    <span class="leg" style="background:#d5f5e3;color:#1a7a40">≥ 85%</span>
    <span class="leg" style="background:#eafaf1;color:#27ae60">70 – 84%</span>
    <span class="leg" style="background:#fef9e7;color:#b7950b">55 – 69%</span>
    <span class="leg" style="background:#fef5e4;color:#ca6f1e">40 – 54%</span>
    <span class="leg" style="background:#fdecea;color:#c0392b">&lt; 40%</span>
    <span style="color:#aaa;font-size:0.9em">— = pas de GT défini (question skip)</span>
  </div>
  <p class="note">R = Recall · P = Précision · F1 = F-mesure harmonique. Questions Q104+ exclues (limites architecturales sans GT).</p>
  <table>
    <thead>
      <tr>
        <th style="background:#ecf0f1;text-align:left;min-width:220px"></th>
        <th style="background:#ecf0f1"></th>
        {header_cells}
      </tr>
      <tr>{subheader}</tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>
</body>
</html>"""

OUT.write_text(html, encoding='utf-8')
print(f'\nHTML -> {OUT}')
