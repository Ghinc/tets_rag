"""HTML résumé — section Retrieval mono-commune (Q1–Q25), toutes configs."""
import json, sys, io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json', encoding='utf-8') as f:
    data = json.load(f)

CONFIGS = ['v_vanilla_k10', 'v_vanilla_k25', 'v_decomp', 'v_decomp_raptor']
LABELS  = {'v_vanilla_k10': 'Vanilla k10', 'v_vanilla_k25': 'Vanilla k25',
           'v_decomp': 'Decomp', 'v_decomp_raptor': 'Decomp+Raptor'}
COLORS  = {'v_vanilla_k10': '#c0392b', 'v_vanilla_k25': '#e67e22',
           'v_decomp': '#27ae60', 'v_decomp_raptor': '#2980b9'}

def clean(s):
    return (s or '').replace('‘','').replace('’',"'").replace('′',"'")

# Index global
index = {}
for cfg, entries in data.items():
    for e in entries:
        index[(e['excel_row'], cfg)] = e

# Rows de la section
TARGET_ROWS = sorted({
    e['excel_row'] for e in data['v_vanilla_k10']
    if 'mono' in (e.get('section') or '').lower()
})

# Regrouper par sous-section
by_sub = defaultdict(list)
sub_of = {}
for row in TARGET_ROWS:
    e0 = index.get((row, 'v_vanilla_k10'), {})
    sub = clean(e0.get('subsection') or '').strip() or '—'
    by_sub[sub].append(row)
    sub_of[row] = sub

SUB_ORDER = [
    'Retrieval factuel et interprétation',
    'Retrieval source-spécifique',
    'Retrieval par sous-population',
    'Analyse multi-source et cohérence des données',
    'Retrieval descriptif global',
    'Gestion de l\'incertitude et des limites',
    'Questions normatives / prescriptives',
]
ordered_subs = [s for s in SUB_ORDER if s in by_sub]
for s in sorted(by_sub):
    if s not in ordered_subs:
        ordered_subs.append(s)

# ── Helpers ──────────────────────────────────────────────────────────────────
def sc_color(v):
    if v is None: return '#95a5a6'
    return '#27ae60' if v >= 4 else ('#e67e22' if v >= 3 else '#c0392b')

def score_pill(v):
    if v is None: return '<span class="pill grey">—</span>'
    c = sc_color(v)
    return f'<span class="pill" style="background:{c}">{v:.2f}</span>'

def dim_cell(v):
    if v is None: return '<td class="dc">—</td>'
    c = sc_color(v)
    return f'<td class="dc" style="color:{c};font-weight:bold">{v:.1f}</td>'

def mis_flags(e):
    mis = e.get('mislabelling_detecte', {})
    if not mis or not isinstance(mis, dict): return ''
    short = {'regle_1_quali_quanti':'R1','regle_2_surinterpretation_oppchovec':'R2',
             'regle_3_absence_quali_non_signalee':'R3','regle_4_extrapolation_sous_groupe':'R4'}
    flags = [short.get(k,'?') for k,v in mis.items()
             if isinstance(v,str) and v.strip().lower().startswith('oui')]
    return ''.join(f'<span class="flag">{f}</span>' for f in flags)

def avg(lst):
    vals = [x for x in lst if x is not None]
    return sum(vals)/len(vals) if vals else None

# ── Navigation par sous-section ──────────────────────────────────────────────
nav_html = ' · '.join(
    f'<a href="#sub-{i}">{s}</a>'
    for i, s in enumerate(ordered_subs)
)

# ── Résumé global de la section ───────────────────────────────────────────────
def global_summary():
    rows_h = '<tr><th>Config</th><th>N</th><th>Score global</th><th>Pertinence</th><th>Factuel</th><th>Nuance</th><th>Q/Q</th><th>Flags %</th></tr>'
    for cfg in CONFIGS:
        entries = [index.get((r, cfg), {}) for r in TARGET_ROWS]
        sg   = avg([e.get('score_global')   for e in entries])
        pert = avg([e.get('pertinence')      for e in entries])
        fact = avg([e.get('fondement_factuel') for e in entries])
        nua  = avg([e.get('nuance_incertitude') for e in entries])
        qq   = avg([e.get('coherence_qualiquanti') for e in entries])
        n_flags = sum(1 for e in entries
                      if any(v.strip().lower().startswith('oui')
                             for v in (e.get('mislabelling_detecte') or {}).values()
                             if isinstance(v, str)))
        flag_pct = f'{100*n_flags/len(TARGET_ROWS):.0f}%'
        c = COLORS[cfg]
        rows_h += (f'<tr><td style="color:{c};font-weight:bold">{LABELS[cfg]}</td>'
                   f'<td style="text-align:center">{len(TARGET_ROWS)}</td>'
                   f'<td>{score_pill(sg)}</td>'
                   f'{dim_cell(pert)}{dim_cell(fact)}{dim_cell(nua)}{dim_cell(qq)}'
                   f'<td style="text-align:center;color:#c0392b">{flag_pct}</td></tr>')
    return f'<table class="summary-tbl">{rows_h}</table>'

# ── Corps principal ───────────────────────────────────────────────────────────
body_parts = []

for sub_idx, sub in enumerate(ordered_subs):
    rows_in_sub = by_sub[sub]
    sub_rows_html = ''

    for row in rows_in_sub:
        e0 = index.get((row, 'v_vanilla_k10'), {})
        q_text = e0.get('question', f'Q{row}')

        # Bandeau question
        sub_rows_html += f'''
<div class="q-block" id="q{row}">
  <div class="q-header">
    <span class="q-num">Q{row}</span>
    <span class="q-text">{q_text}</span>
  </div>
  <table class="resp-table">
    <thead>
      <tr>
        <th style="width:110px">Config</th>
        <th style="width:60px">Score</th>
        <th style="width:140px">Dimensions (P/F/N/Q)</th>
        <th style="width:50px">Flags</th>
        <th>Réponse complète</th>
      </tr>
    </thead>
    <tbody>'''

        for cfg in CONFIGS:
            e = index.get((row, cfg), {})
            if not e:
                sub_rows_html += f'<tr><td style="color:{COLORS[cfg]}">{LABELS[cfg]}</td><td colspan="4" style="color:#aaa">—</td></tr>'
                continue
            c    = COLORS[cfg]
            sg   = e.get('score_global')
            pert = e.get('pertinence')
            fact = e.get('fondement_factuel')
            nua  = e.get('nuance_incertitude')
            qq   = e.get('coherence_qualiquanti')
            answer = (e.get('answer') or '').replace('<','&lt;').replace('>','&gt;')
            n_chars = len(e.get('answer') or '')

            dims = ''.join(
                f'<span style="color:{sc_color(v)};font-weight:bold">{v:.1f}</span>'
                if v is not None else '<span style="color:#aaa">—</span>'
                for v in [pert, fact, nua, qq]
            )
            dims_cell = ' / '.join(
                f'<span style="color:{sc_color(v)};font-weight:bold">{v:.1f}</span>'
                if v is not None else '<span style="color:#aaa">—</span>'
                for v in [pert, fact, nua, qq]
            )

            sub_rows_html += f'''
      <tr>
        <td style="color:{c};font-weight:bold;white-space:nowrap">{LABELS[cfg]}</td>
        <td style="text-align:center">{score_pill(sg)}</td>
        <td style="font-size:0.82em;text-align:center">{dims_cell}</td>
        <td style="text-align:center">{mis_flags(e)}</td>
        <td>
          <details>
            <summary>{n_chars} car. — cliquer pour lire</summary>
            <div class="answer-box">{answer}</div>
          </details>
        </td>
      </tr>'''

        sub_rows_html += '</tbody></table></div>'

    body_parts.append(f'''
<div class="sub-block" id="sub-{sub_idx}">
  <h3 class="sub-header">{sub} <span class="sub-count">({len(rows_in_sub)} questions)</span></h3>
  {sub_rows_html}
</div>''')

# ── HTML final ────────────────────────────────────────────────────────────────
html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Retrieval mono-commune — 25 questions × 4 configs</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; margin: 0; background: #f0f2f5; color: #222; }}
  .topbar {{ background: #2c3e50; color: white; padding: 12px 24px;
             position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.3); }}
  .topbar h1 {{ margin: 0 0 5px; font-size: 1.05em; }}
  .topbar nav {{ font-size: 0.8em; opacity:.8; }}
  .topbar nav a {{ color: #aed6f1; text-decoration: none; }}
  .topbar nav a:hover {{ text-decoration: underline; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px 24px; }}

  /* Résumé global */
  .summary-tbl {{ border-collapse: collapse; margin-bottom: 28px; background: white;
                  border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow: hidden; width: auto; }}
  .summary-tbl th, .summary-tbl td {{ border: 1px solid #e0e0e0; padding: 6px 14px; }}
  .summary-tbl th {{ background: #ecf0f1; font-size: 0.85em; }}
  .dc {{ text-align: center; }}

  /* Sous-sections */
  .sub-block {{ background: white; border-radius: 8px; margin-bottom: 28px;
                box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow: hidden; }}
  .sub-header {{ margin: 0; padding: 10px 18px; background: #34495e; color: white;
                 font-size: 0.95em; display: flex; align-items: center; gap: 8px; }}
  .sub-count {{ font-weight: normal; font-size: 0.85em; opacity:.7; }}

  /* Blocs question */
  .q-block {{ border-bottom: 2px solid #ecf0f1; padding: 14px 18px; }}
  .q-block:last-child {{ border-bottom: none; }}
  .q-header {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px; }}
  .q-num {{ background: #34495e; color: white; border-radius: 4px;
            padding: 2px 8px; font-weight: bold; white-space: nowrap; }}
  .q-text {{ font-weight: 600; font-size: 0.98em; }}

  /* Tableau réponses */
  .resp-table {{ width: 100%; border-collapse: collapse; }}
  .resp-table th {{ background: #f5f6fa; padding: 5px 8px; border: 1px solid #ddd;
                    font-size: 0.82em; text-align: left; }}
  .resp-table td {{ border: 1px solid #eee; padding: 5px 8px; vertical-align: top; }}
  .resp-table tr:hover td {{ background: #fafbfc; }}

  /* Pills et flags */
  .pill {{ display: inline-block; color: white; border-radius: 4px;
           padding: 1px 8px; font-size: 0.85em; font-weight: bold; }}
  .pill.grey {{ background: #95a5a6; }}
  .flag {{ background: #c0392b; color: white; border-radius: 3px;
           padding: 1px 5px; font-size: 0.8em; font-weight: bold; margin-right: 2px; }}

  /* Réponse dépliable */
  details summary {{ cursor: pointer; color: #2980b9; font-size: 0.82em;
                     user-select: none; padding: 2px 0; }}
  details summary:hover {{ text-decoration: underline; }}
  details summary::-webkit-details-marker {{ display: none; }}
  .answer-box {{ white-space: pre-wrap; font-family: monospace; font-size: 0.78em;
                 background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;
                 padding: 10px; margin-top: 6px; max-height: 400px; overflow-y: auto; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>Retrieval mono-commune — 25 questions × 4 configurations · Juge V4.3 GPT-4o</h1>
  <nav>{nav_html}</nav>
</div>
<div class="container">
  <h3 style="margin-top:0;color:#555;font-size:0.95em">Résumé par configuration</h3>
  {global_summary()}
  {''.join(body_parts)}
</div>
</body>
</html>'''

out = Path('comparaisons_rag/mono_commune_all_answers.html')
out.write_text(html, encoding='utf-8')
print(f'HTML → {out}  ({out.stat().st_size // 1024} Ko)')
