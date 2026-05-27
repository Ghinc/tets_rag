"""
Calcule les métriques d'accord humain-juge pour chaque dimension
et génère le tableau LaTeX complet.
"""
import json, sys, io
from scipy import stats
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('comparaisons_rag/judge_scores_v2_gpt4o_20260512_105001.json', encoding='utf-8') as f:
    data = json.load(f)

dims_order = [
    ('pertinence',           'Pertinence'),
    ('fondement_factuel',    'Fondement factuel'),
    ('nuance_incertitude',   'Nuance / incertitude'),
    ('coherence_qualiquanti','Cohérence quali/quanti'),
    ('score_global',         'Score global'),
]

results = {}
for dim, _ in dims_order:
    h_scores, j_scores = [], []
    for e in data:
        h = e['scores_humain'].get(dim)
        j = e['scores_juge_v2'].get(dim)
        if h is not None and j is not None:
            h_scores.append(float(h))
            j_scores.append(float(j))
    h = np.array(h_scores)
    j = np.array(j_scores)
    r,   p_r   = stats.pearsonr(h, j)
    rho, p_rho = stats.spearmanr(h, j)
    mae  = np.mean(np.abs(h - j))
    bias = np.mean(h - j)
    a1   = np.mean(np.abs(h - j) <= 1.0) * 100
    results[dim] = dict(r=r, p_r=p_r, rho=rho, p_rho=p_rho, mae=mae, bias=bias, a1=a1)

# ── Résumé console ──────────────────────────────────────────────────────────
print(f"{'Dimension':25s}  {'r':>7s}  {'p_r':>6s}  {'rho':>7s}  {'p_rho':>6s}  {'MAE':>5s}  {'Biais':>7s}  {'±1':>5s}")
print('-' * 85)
for dim, label in dims_order:
    v = results[dim]
    def sig(p):
        return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{label:25s}  {v['r']:+.3f}{sig(v['p_r']):3s}  {v['p_r']:.3f}  "
          f"{v['rho']:+.3f}{sig(v['p_rho']):3s}  {v['p_rho']:.3f}  "
          f"{v['mae']:.3f}  {v['bias']:+.3f}  {v['a1']:.0f}%")

# ── LaTeX ────────────────────────────────────────────────────────────────────
def num(v, decimals=2):
    """Formatte un float en LaTeX avec virgule décimale."""
    fmt = f'{v:.{decimals}f}'
    return fmt.replace('.', '{,}')

def star_latex(p):
    if p < 0.001: return r'^{***}'
    if p < 0.01:  return r'^{**}'
    if p < 0.05:  return r'^{*}'
    return ''

def r_cell(v, p, bold=False):
    s = star_latex(p)
    inner = f'{num(abs(v))}{s}' if v >= 0 else f'-{num(abs(v))}{s}'
    if v >= 0 and v < 1:
        inner = f'{num(v)}{s}'
    else:
        inner = f'{num(v)}{s}'
    if bold:
        return f'\\textbf{{{inner}}}'
    return inner

def bias_cell(v):
    sign = '+' if v >= 0 else ''
    return f'${sign}{num(v)}$'

def a1_cell(v, bold=False):
    inner = f'{v:.0f}\\%'
    if bold:
        return f'\\textbf{{{inner}}}'
    return inner

latex_lines = [
    r'\begin{table}[t]',
    r'    \centering',
    r'    \small',
    r'    \begin{tabular}{@{}lccccc@{}}',
    r'        \toprule',
    (r'        \textbf{Dimension} & \textbf{Pearson} & \textbf{Spearman}'
     r' & \textbf{MAE} & \textbf{Biais (h$-$j)} & \textbf{Accord $\pm 1$} \\'),
    r'        \midrule',
]

for dim, label in dims_order:
    v = results[dim]
    is_global = (dim == 'score_global')

    if is_global:
        latex_lines.append(r'        \midrule')

    r_s   = r_cell(v['r'],   v['p_r'],   bold=is_global)
    rho_s = r_cell(v['rho'], v['p_rho'], bold=False)
    mae_s = (f'\\textbf{{{num(v["mae"])}}}' if is_global else num(v['mae']))
    bias_s= bias_cell(v['bias'])
    a1_s  = a1_cell(v['a1'], bold=is_global)

    if is_global:
        label_s = f'\\textbf{{{label}}}'
    else:
        label_s = label

    latex_lines.append(
        f'        {label_s} & {r_s} & {rho_s} & {mae_s} & {bias_s} & {a1_s} \\\\'
    )

latex_lines += [
    r'        \bottomrule',
    r'    \end{tabular}',
    r'    \caption{Accord humain--juge (GPT-4o, $n=20$). $^{*}p<0{,}05$.}',
    r'    \label{tab:accord_juge}',
    r'\end{table}',
]

print()
print('=' * 60)
print('TABLEAU LATEX')
print('=' * 60)
print('\n'.join(latex_lines))
