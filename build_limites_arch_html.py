"""Génère un HTML détaillé des questions Limites architecturales (Q40, Q104-106)."""
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json', encoding='utf-8') as f:
    data = json.load(f)

CONFIGS = ['v_vanilla_k10', 'v_vanilla_k25', 'v_decomp', 'v_decomp_raptor']
COLORS = {
    'v_vanilla_k10':   '#c0392b',
    'v_vanilla_k25':   '#e67e22',
    'v_decomp':        '#27ae60',
    'v_decomp_raptor': '#2980b9',
}
LABELS = {
    'v_vanilla_k10':   'Vanilla k10',
    'v_vanilla_k25':   'Vanilla k25',
    'v_decomp':        'Decomp',
    'v_decomp_raptor': 'Decomp+Raptor',
}
TARGET_ROWS = [40, 104, 105, 106, 107, 108, 109]

# Indexer par (excel_row, config)
index = {}
for cfg, entries in data.items():
    for e in entries:
        index[(e['excel_row'], cfg)] = e

def score_badge(s):
    if s is None: return '<span class="badge grey">?</span>'
    c = '#27ae60' if s >= 4 else ('#e67e22' if s >= 3 else '#c0392b')
    return f'<span class="badge" style="background:{c}">{s:.2f}</span>'

def mis_badge(e):
    mis = e.get('mislabelling_detecte', {})
    if not mis or not isinstance(mis, dict):
        return ''
    triggered = [k[-1] for k, v in mis.items()
                 if isinstance(v, str) and v.strip().lower().startswith('oui')]
    if not triggered:
        return '<span class="badge" style="background:#27ae60">✓ clean</span>'
    return ' '.join(f'<span class="badge" style="background:#c0392b">R{r}</span>' for r in triggered)

def dim_row(label, key, e):
    v = e.get(key)
    j = e.get(key + '_justif', '')
    if v is None: return ''
    c = '#27ae60' if v >= 4 else ('#e67e22' if v >= 3 else '#c0392b')
    return f'<tr><td class="dim-label">{label}</td><td><span style="color:{c};font-weight:bold">{v}/5</span></td><td class="justif">{j}</td></tr>'

def render_mislabelling(e):
    mis = e.get('mislabelling_detecte', {})
    if not mis or not isinstance(mis, dict): return ''
    rows = ''
    short = {'regle_1_quali_quanti':'R1','regle_2_surinterpretation_oppchovec':'R2',
             'regle_3_absence_quali_non_signalee':'R3','regle_4_extrapolation_sous_groupe':'R4'}
    for k, v in mis.items():
        if not isinstance(v, str): continue
        label = short.get(k, k)
        triggered = v.strip().lower().startswith('oui')
        color = '#c0392b' if triggered else '#27ae60'
        rows += f'<tr><td style="color:{color};font-weight:bold">{label}</td><td style="font-size:0.85em">{v}</td></tr>'
    sigs = e.get('signalements_detectes', [])
    sig_html = ''
    if sigs:
        items = ''.join(f'<li>{s}</li>' for s in sigs)
        sig_html = f'<div class="sig-box"><b>Signalements détectés :</b><ul>{items}</ul></div>'
    return f'<table class="mis-table">{rows}</table>{sig_html}'

def render_config_block(cfg, e):
    color = COLORS[cfg]
    label = LABELS[cfg]
    sg = e.get('score_global')
    raisonnement = e.get('raisonnement', '') or ''
    answer = e.get('answer', '') or ''
    n_src = e.get('n_sources', 0)
    elapsed = e.get('rag_elapsed_s', '')
    uid = f"q{e['excel_row']}_{cfg}"

    dims_html = '<table class="dim-table">'
    dims_html += dim_row('Pertinence', 'pertinence', e)
    dims_html += dim_row('Fondement factuel', 'fondement_factuel', e)
    dims_html += dim_row('Nuance/incertitude', 'nuance_incertitude', e)
    dims_html += dim_row('Cohérence quali/quanti', 'coherence_qualiquanti', e)
    dims_html += '</table>'

    return f'''
<div class="config-block" style="border-left:4px solid {color}">
  <div class="config-header">
    <span class="config-name" style="color:{color}">{label}</span>
    {score_badge(sg)}
    {mis_badge(e)}
    <span class="meta">{n_src} sources · {elapsed}s RAG</span>
  </div>

  <div class="two-col">
    <div>
      <div class="section-title">Dimensions</div>
      {dims_html}
    </div>
    <div>
      <div class="section-title">Mislabelling</div>
      {render_mislabelling(e)}
    </div>
  </div>

  <div class="section-title">Raisonnement du juge</div>
  <div class="raisonnement">{raisonnement}</div>

  <details>
    <summary>Réponse complète du système ({len(answer)} car.)</summary>
    <div class="answer">{answer.replace("<","&lt;").replace(">","&gt;")}</div>
  </details>
</div>'''

# ── Construction HTML ─────────────────────────────────────────────────────────
question_blocks = ''
for row in TARGET_ROWS:
    e0 = index.get((row, 'v_vanilla_k10'), {})
    q_text = e0.get('question', f'Q{row}')
    section = e0.get('section', '')
    subsection = e0.get('subsection', '')

    # Résumé scores par config
    summary = '<table class="summary-table"><tr><th>Config</th><th>Score</th><th>Flags</th><th>Raisonnement (extrait)</th></tr>'
    for cfg in CONFIGS:
        e = index.get((row, cfg), {})
        sg = e.get('score_global')
        color = COLORS[cfg]
        rai = (e.get('raisonnement','') or '')[:120]
        summary += f'<tr><td style="color:{color};font-weight:bold">{LABELS[cfg]}</td><td>{score_badge(sg)}</td><td>{mis_badge(e)}</td><td class="justif">{rai}…</td></tr>'
    summary += '</table>'

    config_blocks = ''.join(render_config_block(cfg, index.get((row, cfg), {})) for cfg in CONFIGS)

    question_blocks += f'''
<div class="question-card">
  <div class="q-header">
    <span class="q-num">Q{row}</span>
    <span class="q-section">{section}{" — " + subsection if subsection else ""}</span>
  </div>
  <div class="q-text">{q_text}</div>
  {summary}
  <div class="configs-container">
    {config_blocks}
  </div>
</div>'''

html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Limites architecturales — Q40, Q104-106</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; margin: 0; background: #f4f6f8; color: #222; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 1.4em; color: #333; border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
  .question-card {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
                    margin-bottom: 32px; padding: 20px; }}
  .q-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
  .q-num {{ background: #34495e; color: white; border-radius: 4px; padding: 2px 8px;
            font-weight: bold; font-size: 1em; }}
  .q-section {{ color: #888; font-size: 0.9em; }}
  .q-text {{ font-size: 1.05em; font-weight: 600; margin-bottom: 14px;
             padding: 10px 14px; background: #f8f9fa; border-radius: 6px;
             border-left: 4px solid #34495e; }}
  .summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
  .summary-table th {{ background: #ecf0f1; padding: 6px 10px; text-align: left; font-size: 0.85em; }}
  .summary-table td {{ padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
  .configs-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .config-block {{ background: #fafafa; border-radius: 6px; padding: 14px; }}
  .config-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
  .config-name {{ font-weight: bold; font-size: 1em; }}
  .meta {{ color: #999; font-size: 0.8em; margin-left: auto; }}
  .badge {{ display: inline-block; color: white; border-radius: 4px; padding: 2px 7px;
            font-size: 0.85em; font-weight: bold; }}
  .badge.grey {{ background: #95a5a6; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px; }}
  .section-title {{ font-weight: 600; font-size: 0.8em; text-transform: uppercase;
                    color: #666; margin: 8px 0 4px; letter-spacing: .04em; }}
  .dim-table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  .dim-table td {{ padding: 3px 6px; border-bottom: 1px solid #eee; }}
  .dim-label {{ color: #555; width: 120px; }}
  .justif {{ color: #666; font-size: 0.85em; font-style: italic; }}
  .mis-table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  .mis-table td {{ padding: 3px 6px; border-bottom: 1px solid #eee; vertical-align: top; }}
  .sig-box {{ background: #eafaf1; border-radius: 4px; padding: 6px 10px; margin-top: 6px; font-size: 0.82em; }}
  .sig-box ul {{ margin: 4px 0 0 16px; padding: 0; }}
  .raisonnement {{ background: #fff8e1; border-radius: 4px; padding: 8px 10px; font-size: 0.85em;
                   color: #555; margin-bottom: 8px; border-left: 3px solid #f39c12; }}
  details summary {{ cursor: pointer; color: #2980b9; font-size: 0.85em; padding: 4px 0;
                     user-select: none; }}
  details summary:hover {{ text-decoration: underline; }}
  .answer {{ white-space: pre-wrap; font-family: monospace; font-size: 0.8em; background: #f5f5f5;
             border-radius: 4px; padding: 10px; margin-top: 6px; max-height: 400px;
             overflow-y: auto; border: 1px solid #ddd; }}
  @media (max-width: 900px) {{
    .configs-container {{ grid-template-columns: 1fr; }}
    .two-col {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Limites architecturales — Q40, Q104, Q105, Q106</h1>
  <p style="color:#666;font-size:0.9em">Juge V4.3 (Q40) · Juge V4.4 (Q104–106) · GPT-4o</p>
  {question_blocks}
</div>
</body>
</html>'''

out = Path('comparaisons_rag/limites_architecturales.html')
out.write_text(html, encoding='utf-8')
print(f'HTML → {out}')
