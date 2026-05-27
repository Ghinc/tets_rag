import json, html as html_mod

with open('comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

by_row = {}
for ver, entries in data.items():
    for e in entries:
        row = e['excel_row']
        if row not in by_row:
            by_row[row] = {}
        by_row[row][ver] = e

target_rows = [19, 20, 21, 44, 45, 46, 47, 48, 49, 50, 53, 54, 55]
configs = ['v_vanilla_k10', 'v_vanilla_k25', 'v_decomp', 'v_decomp_raptor']
cfg_labels = {
    'v_vanilla_k10':    'Vanilla k10',
    'v_vanilla_k25':    'Vanilla k25',
    'v_decomp':         'Decomp',
    'v_decomp_raptor':  'Decomp+RAPTOR',
}
cfg_colors = {
    'v_vanilla_k10':   '#e8f4f8',
    'v_vanilla_k25':   '#e8f0e8',
    'v_decomp':        '#f8f0e8',
    'v_decomp_raptor': '#f0e8f8',
}

def score_color(s):
    if s is None: return '#9e9e9e'
    if s >= 4.5:  return '#388e3c'
    if s >= 3.5:  return '#f57c00'
    return '#c62828'

def fmt_score(s):
    if s is None: return '—'
    return f'{s:.2f}'

CSS = """
body { font-family: Arial, sans-serif; font-size: 14px; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1   { color: #333; border-bottom: 3px solid #555; padding-bottom: 10px; }
h2   { color: #444; font-size: 1em; margin: 32px 0 8px; background: #ddd; padding: 6px 14px; border-radius: 4px; }
h3   { color: #555; font-size: 0.92em; margin: 20px 0 6px; font-style: italic; }
.question-block  { background: white; border-radius: 8px; padding: 20px; margin-bottom: 28px; box-shadow: 0 1px 5px rgba(0,0,0,0.12); }
.question-header { font-size: 1.05em; font-weight: bold; color: #222; margin-bottom: 6px; }
.sub-tag  { display:inline-block; background:#555; color:white; font-size:0.72em; padding:2px 8px; border-radius:10px; margin-bottom:12px; }
table     { border-collapse: collapse; width: 100%; margin-bottom: 14px; }
th        { background: #444; color: white; padding: 6px 10px; font-size: 0.83em; text-align: center; }
td        { border: 1px solid #ddd; padding: 5px 10px; font-size: 0.83em; }
.badge    { display:inline-block; padding: 2px 10px; border-radius: 12px; color:white; font-weight:bold; font-size:0.88em; }
.mis-badge{ display:inline-block; background:#c62828; color:white; font-size:0.68em; padding:1px 5px; border-radius:4px; margin-left:4px; vertical-align:middle; }
.cfg-block { border-left: 4px solid #999; padding: 10px 14px; margin: 10px 0; border-radius: 0 6px 6px 0; }
.cfg-label { font-weight: bold; font-size: 0.88em; margin-bottom: 4px; }
.mini      { font-size: 0.76em; color: #777; margin-bottom: 4px; }
.judge-text{ font-size: 0.82em; color: #444; font-style: italic; margin: 5px 0 4px; border-left: 2px solid #ccc; padding-left: 8px; }
.answer-box{ font-size: 0.83em; color: #333; line-height: 1.55; max-height: 320px; overflow-y: auto;
             border: 1px solid #ddd; padding: 8px 10px; background: #fafafa; border-radius: 4px;
             white-space: pre-wrap; margin-top: 6px; }
details summary { cursor: pointer; font-size: 0.84em; color: #1565c0; margin-top: 5px; }
details[open] summary { margin-bottom: 5px; }
.toc  { background: white; padding: 14px 20px; border-radius: 8px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
.toc a{ color: #1565c0; text-decoration: none; display: block; margin: 2px 0; font-size: 0.88em; }
.toc a:hover { text-decoration: underline; }
"""

def score_cell(v, mis=False):
    col = score_color(v)
    mis_s = '<span class="mis-badge">MIS</span>' if mis else ''
    if v is not None:
        return f'<td style="text-align:center"><span class="badge" style="background:{col}">{v:.2f}</span>{mis_s}</td>'
    return '<td style="text-align:center;color:#bbb">—</td>'

lines = [
    '<!DOCTYPE html>',
    '<html lang="fr">',
    '<head>',
    '<meta charset="UTF-8">',
    '<title>Gestion absence — Détail complet</title>',
    f'<style>{CSS}</style>',
    '</head>',
    '<body>',
    '<h1>Gestion de l\'absence d\'information — Détail complet</h1>',
    '<p style="color:#666">13 questions · 4 configurations · Juge GPT-4o + V4.3 · sections <em>Gestion de l\'incertitude</em> &amp; <em>Gestion d\'absence d\'information</em></p>',
]

# TOC
lines.append('<div class="toc"><strong>Table des matières</strong>')
current_sub = None
for row in target_rows:
    vmap = by_row.get(row, {})
    if not vmap: continue
    ref = next(iter(vmap.values()))
    sub = ref.get('subsection', '')
    if sub != current_sub:
        lines.append(f'<br><em style="color:#555;font-size:0.85em">{html_mod.escape(sub)}</em><br>')
        current_sub = sub
    q = ref.get('question', '')
    lines.append(f'<a href="#q{row}">Q{row} — {html_mod.escape(q[:90])}</a>')
lines.append('</div>')

current_sec = None
current_sub = None

for row in target_rows:
    vmap = by_row.get(row, {})
    if not vmap: continue
    ref = next(iter(vmap.values()))
    sec = ref.get('section', '')
    sub = ref.get('subsection', '')
    q   = ref.get('question', '')

    if sec != current_sec:
        lines.append(f'<h2>&#128196; Section : {html_mod.escape(sec)}</h2>')
        current_sec = sec
        current_sub = None

    if sub != current_sub:
        lines.append(f'<h3>{html_mod.escape(sub)}</h3>')
        current_sub = sub

    lines.append(f'<div class="question-block" id="q{row}">')
    lines.append(f'<div class="question-header">Q{row} &mdash; {html_mod.escape(q)}</div>')
    lines.append(f'<span class="sub-tag">{html_mod.escape(sub)}</span>')

    # Scores table
    lines.append('<table>')
    lines.append('<tr><th>Métrique</th>' +
                 ''.join(f'<th>{cfg_labels[c]}</th>' for c in configs) + '</tr>')

    metrics = [
        ('Score global',       'score_global',          True),
        ('Pertinence',         'pertinence',             False),
        ('Factuel',            'fondement_factuel',      False),
        ('Nuance/Incertitude', 'nuance_incertitude',     False),
        ('Cohérence Q/Q',      'coherence_qualiquanti',  False),
    ]
    for label, field, show_mis in metrics:
        cells = []
        for c in configs:
            e = vmap.get(c, {})
            v   = e.get(field)
            mis = e.get('mislabelling_flag', False) if show_mis else False
            cells.append(score_cell(v, mis))
        lines.append(f'<tr><td style="color:#555">{label}</td>{"".join(cells)}</tr>')

    # n_sources
    cells = [
        f'<td style="text-align:center;color:#666">{vmap.get(c,{}).get("n_sources","—")} src</td>'
        for c in configs
    ]
    lines.append(f'<tr><td style="color:#555">Sources utilisées</td>{"".join(cells)}</tr>')

    lines.append('</table>')

    # Per-config detail
    for c in configs:
        e   = vmap.get(c, {})
        sg  = e.get('score_global')
        mis = e.get('mislabelling_flag', False)
        ans = (e.get('answer', '') or '').strip()
        rais= (e.get('raisonnement', '') or '').strip()

        border_col = '#c62828' if mis else '#bdbdbd'
        bg = cfg_colors[c]

        lines.append(f'<div class="cfg-block" style="background:{bg};border-left-color:{border_col}">')

        mis_s = '<span class="mis-badge">MISLABELLING</span>' if mis else ''
        badge = (f'<span class="badge" style="background:{score_color(sg)}">{fmt_score(sg)}</span>'
                 if sg is not None else '<span style="color:#aaa">—</span>')
        lines.append(f'<div class="cfg-label">{cfg_labels[c]} &nbsp; {badge} {mis_s}</div>')

        # mini metrics
        p  = e.get('pertinence')
        ff = e.get('fondement_factuel')
        ni = e.get('nuance_incertitude')
        qq = e.get('coherence_qualiquanti')
        ns = e.get('n_sources', '—')
        lines.append(f'<div class="mini">P={fmt_score(p)} &middot; F={fmt_score(ff)} &middot; N={fmt_score(ni)} &middot; Q/Q={fmt_score(qq)} &middot; {ns} sources</div>')

        if rais:
            lines.append(f'<div class="judge-text">&#129516; {html_mod.escape(rais)}</div>')

        if ans:
            lines.append('<details><summary>Voir la réponse complète du RAG</summary>')
            lines.append(f'<div class="answer-box">{html_mod.escape(ans)}</div>')
            lines.append('</details>')

        lines.append('</div>')  # cfg-block

    lines.append('</div>')  # question-block

lines.append('</body></html>')

out_path = 'comparaisons_rag/gestion_absence_detail.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'OK -> {out_path}')
