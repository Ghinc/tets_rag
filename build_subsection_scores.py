"""Génère scores_par_subsection.html — tableau scores × sous-section × config."""
import json, sys, io
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json', encoding='utf-8') as f:
    data = json.load(f)

CONFIGS = ['v_vanilla_k10', 'v_vanilla_k25', 'v_decomp', 'v_decomp_raptor']
LABELS  = {'v_vanilla_k10': 'Vanilla k10', 'v_vanilla_k25': 'Vanilla k25',
           'v_decomp': 'Decomp', 'v_decomp_raptor': 'Decomp+Raptor'}
COLORS  = {'v_vanilla_k10': '#c0392b', 'v_vanilla_k25': '#e67e22',
           'v_decomp': '#27ae60', 'v_decomp_raptor': '#2980b9'}
SEC_ORDER = [
    'Retrieval mono-commune',
    'Raisonnement comparatif',
    'Raisonnement causal et contre-intuitif',
    "Gestion de l'absence d'information",
    "Gestion de l'incertitude et des biais",
    'Robustesse sémantique',
    'Limites architecturales',
]

def mean(lst): return sum(lst) / len(lst) if lst else None

def clean(s):
    return (s or '').replace('‘', '').replace('’', "'").replace('′', "'")

# Agréger scores par (section normalisée, sous-section) × config
agg = defaultdict(lambda: defaultdict(list))
for cfg, entries in data.items():
    for e in entries:
        sec = clean(e.get('section') or '?')
        if 'absence' in sec.lower() and 'information' in sec.lower():
            sec = "Gestion de l'absence d'information"
        sub = clean(e.get('subsection') or '').strip() or '—'
        sg = e.get('score_global')
        if sg is not None:
            agg[(sec, sub)][cfg].append(float(sg))

# Totaux par section
sec_totals = defaultdict(lambda: defaultdict(list))
for (sec, sub), d in agg.items():
    for cfg, scores in d.items():
        sec_totals[sec][cfg].extend(scores)

# Ordre : sections définies + éventuelles sections inconnues
by_sec = defaultdict(list)
for key in agg:
    by_sec[key[0]].append(key)

ordered = []
for sec in SEC_ORDER:
    if sec in by_sec:
        ordered.extend(sorted(by_sec[sec], key=lambda x: x[1]))
for sec in sorted(by_sec):
    if sec not in SEC_ORDER:
        ordered.extend(sorted(by_sec[sec], key=lambda x: x[1]))

# ── Helpers HTML ──────────────────────────────────────────────────────────────
def sc_bg(v):
    if v is None: return '#f5f5f5'
    if v >= 4.5: return '#d5f5e3'
    if v >= 4.0: return '#eafaf1'
    if v >= 3.5: return '#fef9e7'
    if v >= 3.0: return '#fef5e4'
    return '#fdecea'

def sc_fg(v):
    if v is None: return '#aaa'
    if v >= 4.5: return '#1a7a40'
    if v >= 4.0: return '#27ae60'
    if v >= 3.5: return '#b7950b'
    if v >= 3.0: return '#ca6f1e'
    return '#c0392b'

def cell(v, bold=False):
    if v is None:
        return '<td style="background:#f5f5f5;color:#aaa;text-align:center">—</td>'
    bg, fg = sc_bg(v), sc_fg(v)
    fw = 'bold' if bold else 'normal'
    return f'<td style="background:{bg};color:{fg};font-weight:{fw};text-align:center">{v:.2f}</td>'

# ── Navigation ────────────────────────────────────────────────────────────────
nav_items = []
for i, sec in enumerate(SEC_ORDER):
    if sec in by_sec:
        nav_items.append(f'<a href="#sec{i}">{sec}</a>')
nav_html = ' &nbsp;·&nbsp; '.join(nav_items)

# ── Corps du tableau ──────────────────────────────────────────────────────────
rows_html = ''
prev_sec = None
sec_idx = -1

for (sec, sub) in ordered:
    if sec != prev_sec:
        # Ligne total section précédente
        if prev_sec is not None:
            rows_html += (
                f'<tr style="border-top:2px solid #bdc3c7">'
                f'<td style="padding:5px 14px;font-style:italic;color:#555;background:#fafafa">'
                f'Moyenne {prev_sec}</td><td style="background:#fafafa;color:#888;text-align:center">—</td>'
            )
            for cfg in CONFIGS:
                rows_html += cell(mean(sec_totals[prev_sec][cfg]), bold=True)
            rows_html += '</tr><tr><td colspan="6" style="height:6px;background:#f0f2f5;border:none"></td></tr>'

        sec_idx += 1
        rows_html += (
            f'<tr id="sec{sec_idx}" style="background:#2c3e50">'
            f'<td colspan="6" style="color:white;font-weight:bold;padding:8px 14px;font-size:0.95em">'
            f'{sec}</td></tr>'
        )
        prev_sec = sec

    d = agg[(sec, sub)]
    n = len(d.get('v_vanilla_k10', []))
    rows_html += (
        f'<tr><td style="padding:4px 14px 4px 26px">{sub}</td>'
        f'<td style="color:#888;text-align:center;font-size:0.85em">{n}</td>'
    )
    for cfg in CONFIGS:
        rows_html += cell(mean(d[cfg]))
    rows_html += '</tr>'

# Dernière section total
if prev_sec:
    rows_html += (
        f'<tr style="border-top:2px solid #bdc3c7">'
        f'<td style="padding:5px 14px;font-style:italic;color:#555;background:#fafafa">'
        f'Moyenne {prev_sec}</td><td style="background:#fafafa;color:#888;text-align:center">—</td>'
    )
    for cfg in CONFIGS:
        rows_html += cell(mean(sec_totals[prev_sec][cfg]), bold=True)
    rows_html += '</tr><tr><td colspan="6" style="height:6px;background:#f0f2f5;border:none"></td></tr>'

# Total global
all_scores = {cfg: [float(e['score_global']) for e in data[cfg]
                    if isinstance(e.get('score_global'), (int, float))]
              for cfg in CONFIGS}
rows_html += (
    '<tr style="background:#ecf0f1;border-top:3px solid #7f8c8d">'
    '<td style="font-weight:bold;padding:7px 14px">TOTAL</td>'
    '<td style="color:#888;text-align:center;font-size:0.85em">'
    f'{len(all_scores["v_vanilla_k10"])}</td>'
)
for cfg in CONFIGS:
    rows_html += cell(mean(all_scores[cfg]), bold=True)
rows_html += '</tr>'

# ── HTML final ────────────────────────────────────────────────────────────────
header_cells = ''.join(
    f'<th style="background:{COLORS[cfg]};color:white;min-width:90px">{LABELS[cfg]}</th>'
    for cfg in CONFIGS
)

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Scores par sous-section — RAG ablations</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; margin: 0; background: #f0f2f5; }}
  .topbar {{ background: #2c3e50; color: white; padding: 12px 24px;
             position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 6px rgba(0,0,0,.3); }}
  .topbar h1 {{ margin: 0 0 5px; font-size: 1.1em; }}
  .topbar nav {{ font-size: 0.82em; opacity: .8; }}
  .topbar nav a {{ color: #aed6f1; text-decoration: none; }}
  .topbar nav a:hover {{ text-decoration: underline; }}
  .container {{ max-width: 820px; margin: 0 auto; padding: 20px 24px; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  th, td {{ border: 1px solid #e0e0e0; padding: 5px 8px; }}
  th {{ font-size: 0.85em; position: sticky; top: 58px; z-index: 10; }}
  tr:hover td {{ filter: brightness(0.96); }}
  .legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; font-size: 0.82em; }}
  .leg {{ padding: 3px 10px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>Scores par sous-section — Judge V4.3 (GPT-4o) · 109 questions × 4 configs</h1>
  <nav>{nav_html}</nav>
</div>
<div class="container">
  <div class="legend">
    <span class="leg" style="background:#d5f5e3;color:#1a7a40">≥ 4.50</span>
    <span class="leg" style="background:#eafaf1;color:#27ae60">4.00 – 4.49</span>
    <span class="leg" style="background:#fef9e7;color:#b7950b">3.50 – 3.99</span>
    <span class="leg" style="background:#fef5e4;color:#ca6f1e">3.00 – 3.49</span>
    <span class="leg" style="background:#fdecea;color:#c0392b">&lt; 3.00</span>
  </div>
  <table>
    <thead>
      <tr>
        <th style="background:#ecf0f1;text-align:left;min-width:240px">Sous-section</th>
        <th style="background:#ecf0f1;color:#888;min-width:32px">N</th>
        {header_cells}
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>
</body>
</html>"""

out = Path('comparaisons_rag/scores_par_subsection.html')
out.write_text(html, encoding='utf-8')
print(f'HTML -> {out}')
