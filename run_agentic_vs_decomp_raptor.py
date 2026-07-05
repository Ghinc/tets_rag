"""
Compare v11 (Agentic) vs v_decomp_raptor sur les 7 questions du test multi-juge.
Étapes :
  1. Charger les résultats v_decomp_raptor existants (avec scores juge)
  2. Lancer v11 sur les 7 questions
  3. Juger v11 avec le juge V4.3 GPT-4o
  4. Générer un HTML de comparaison
"""
import sys, io, json, os, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

TARGET_ROWS = [11, 41, 44, 48, 68, 83, 98]
OUT_JSON = Path('comparaisons_rag/agentic_vs_decomp_raptor.json')
OUT_HTML = Path('comparaisons_rag/agentic_vs_decomp_raptor.html')

# ── 1. Charger résultats decomp+raptor existants ──────────────────────────────
with open('comparaisons_rag/ablations_7q_judge_compare_4juges_20260518_103131.json', encoding='utf-8') as f:
    existing = json.load(f)

decomp_raptor = {}
for e in existing['v_decomp_raptor']:
    if e['excel_row'] in TARGET_ROWS:
        decomp_raptor[e['excel_row']] = e

questions = {e['excel_row']: e for e in existing['v_vanilla_k10'] if e['excel_row'] in TARGET_ROWS}
print(f"decomp_raptor chargé : {len(decomp_raptor)} entrées")

# ── 2. Lancer v11 sur les 7 questions ────────────────────────────────────────
# Charger résultats v11 déjà calculés si disponibles
v11_results = {}
if OUT_JSON.exists():
    saved = json.load(open(OUT_JSON, encoding='utf-8'))
    v11_results = {r['excel_row']: r for r in saved.get('v11', [])}
    print(f"v11 résultats déjà sauvegardés : {len(v11_results)} entrées")

rows_to_run = [r for r in TARGET_ROWS if r not in v11_results]
if rows_to_run:
    print(f"\nLancement v11 sur {len(rows_to_run)} questions : {rows_to_run}")
    from rag_v11_agentic import AgenticRAGPipeline
    pipeline = AgenticRAGPipeline(chroma_path='./chroma_portrait')
    pipeline.init()

    for row in rows_to_run:
        q_text = questions[row]['question']
        print(f"\n  Q{row} : {q_text[:70]}...")
        t0 = time.time()
        answer, sources, sub_qs = pipeline.query(q_text, use_fast_path=False)
        elapsed = round(time.time() - t0, 1)
        v11_results[row] = {
            'excel_row': row,
            'question': q_text,
            'section': questions[row].get('section', ''),
            'subsection': questions[row].get('subsection', ''),
            'answer': answer,
            'n_sources': len(sources) if sources else 0,
            'elapsed': elapsed,
        }
        print(f"  → OK ({elapsed}s, {v11_results[row]['n_sources']} sources)")
        # Sauvegarde intermédiaire
        _save = {'v11': list(v11_results.values())}
        OUT_JSON.write_text(json.dumps(_save, ensure_ascii=False, indent=2), encoding='utf-8')
else:
    print("v11 : tous les runs déjà disponibles")

# ── 3. Juger v11 avec juge V4.3 GPT-4o ───────────────────────────────────────
rows_to_judge = [r for r in TARGET_ROWS if not v11_results.get(r, {}).get('score_global')]
if rows_to_judge:
    print(f"\nJugement V4.3 GPT-4o sur {len(rows_to_judge)} réponses v11...")
    from eval_from_excel import score_judge_v43
    for row in rows_to_judge:
        r = v11_results[row]
        print(f"  Jugement Q{row}...", flush=True)
        t0 = time.time()
        verdict = score_judge_v43(
            question=r['question'],
            answer=r['answer'],
            sources=[],
            section=r.get('section', ''),
            subsection=r.get('subsection', ''),
            expected_type='reponse_substantielle',
        )
        r.update(verdict)
        r['judge_elapsed'] = round(time.time() - t0, 1)
        print(f"  → score={verdict.get('score_global')} ({r['judge_elapsed']}s)")
        OUT_JSON.write_text(json.dumps({'v11': list(v11_results.values())}, ensure_ascii=False, indent=2), encoding='utf-8')
else:
    print("Jugements v11 déjà disponibles")

# ── 4. Générer HTML ───────────────────────────────────────────────────────────
COLORS = {'v_decomp_raptor': '#2980b9', 'v11': '#8e44ad'}
LABELS = {'v_decomp_raptor': 'Decomp+Raptor (v10)', 'v11': 'Agentic RAG (v11)'}

def sc_color(v):
    if v is None: return '#95a5a6'
    return '#27ae60' if v >= 4 else ('#e67e22' if v >= 3 else '#c0392b')

def pill(v):
    if v is None: return '<span class="pill grey">—</span>'
    c = sc_color(v)
    return f'<span class="pill" style="background:{c}">{v:.2f}</span>'

def dim_span(v):
    if v is None: return '<span style="color:#aaa">—</span>'
    c = sc_color(v)
    return f'<span style="color:{c};font-weight:bold">{v:.1f}</span>'

def mis_flags(e):
    mis = e.get('mislabelling_detecte', {})
    if not mis or not isinstance(mis, dict): return ''
    short = {'regle_1_quali_quanti':'R1','regle_2_surinterpretation_oppchovec':'R2',
             'regle_3_absence_quali_non_signalee':'R3','regle_4_extrapolation_sous_groupe':'R4'}
    flags = [short.get(k,'?') for k,v in mis.items()
             if isinstance(v,str) and v.strip().lower().startswith('oui')]
    return ''.join(f'<span class="flag">{f}</span>' for f in flags)

# Résumé global
summary_rows = '<tr><th>Config</th><th>Score moy.</th><th>Pert.</th><th>Fact.</th><th>Nuance</th><th>Q/Q</th><th>Flags</th></tr>'
for cfg, store in [('v_decomp_raptor', decomp_raptor), ('v11', v11_results)]:
    vals = list(store.values())
    def avg(key):
        vs = [v.get(key) for v in vals if v.get(key) is not None]
        return sum(vs)/len(vs) if vs else None
    sg   = avg('score_global')
    pert = avg('pertinence')
    fact = avg('fondement_factuel')
    nua  = avg('nuance_incertitude')
    qq   = avg('coherence_qualiquanti')
    n_flags = sum(1 for v in vals if any(
        str(x).strip().lower().startswith('oui')
        for x in (v.get('mislabelling_detecte') or {}).values() if isinstance(x, str)))
    c = COLORS[cfg]
    summary_rows += (f'<tr>'
        f'<td style="color:{c};font-weight:bold">{LABELS[cfg]}</td>'
        f'<td>{pill(sg)}</td>'
        f'<td>{dim_span(pert)}</td><td>{dim_span(fact)}</td>'
        f'<td>{dim_span(nua)}</td><td>{dim_span(qq)}</td>'
        f'<td style="color:#c0392b;text-align:center">{n_flags}/{len(TARGET_ROWS)}</td>'
        f'</tr>')

# Blocs par question
q_blocks = ''
for row in TARGET_ROWS:
    q_info = questions[row]
    q_text = q_info['question']
    section = q_info.get('section','')
    subsection = q_info.get('subsection','')

    configs_html = ''
    for cfg, store in [('v_decomp_raptor', decomp_raptor), ('v11', v11_results)]:
        e = store.get(row, {})
        c = COLORS[cfg]
        lbl = LABELS[cfg]
        sg = e.get('score_global')
        pert = e.get('pertinence')
        fact = e.get('fondement_factuel')
        nua  = e.get('nuance_incertitude')
        qq   = e.get('coherence_qualiquanti')
        rai  = (e.get('raisonnement','') or '')
        answer = (e.get('answer','') or '').replace('<','&lt;').replace('>','&gt;')
        n_src = e.get('n_sources', e.get('rag_elapsed_s','—'))
        elapsed = e.get('elapsed', e.get('rag_elapsed_s',''))

        dims = ' / '.join(dim_span(v) for v in [pert, fact, nua, qq])

        configs_html += f'''
<div class="config-block" style="border-left:4px solid {c}">
  <div class="config-header">
    <span class="config-name" style="color:{c}">{lbl}</span>
    {pill(sg)}
    {mis_flags(e)}
    <span class="meta">{n_src} src{(" · " + str(elapsed) + "s") if elapsed else ""}</span>
  </div>
  <div class="dims-row">P/F/N/Q : {dims}</div>
  <div class="section-title">Raisonnement juge</div>
  <div class="raisonnement">{rai[:400]}{"…" if len(rai)>400 else ""}</div>
  <details>
    <summary>Réponse complète ({len(e.get("answer",""))} car.)</summary>
    <div class="answer-box">{answer}</div>
  </details>
</div>'''

    q_blocks += f'''
<div class="q-card" id="q{row}">
  <div class="q-header">
    <span class="q-num">Q{row}</span>
    <span class="q-section">{section}{" — " + subsection if subsection else ""}</span>
  </div>
  <div class="q-text">{q_text}</div>
  <div class="configs-grid">{configs_html}</div>
</div>'''

nav = ' · '.join(f'<a href="#q{r}">Q{r}</a>' for r in TARGET_ROWS)

html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Agentic RAG vs Decomp+Raptor — 7 questions</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; margin: 0; background: #f0f2f5; color: #222; }}
  .topbar {{ background: #1a2a3a; color: white; padding: 12px 24px;
             position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.3); }}
  .topbar h1 {{ margin: 0 0 4px; font-size: 1.05em; }}
  .topbar nav {{ font-size: 0.82em; opacity:.8; }}
  .topbar nav a {{ color: #aed6f1; text-decoration: none; }}
  .topbar nav a:hover {{ text-decoration: underline; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 20px 24px; }}
  .summary-tbl {{ border-collapse: collapse; margin-bottom: 28px; background: white;
                  border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow: hidden; }}
  .summary-tbl th, .summary-tbl td {{ border: 1px solid #e0e0e0; padding: 7px 14px; }}
  .summary-tbl th {{ background: #ecf0f1; font-size: 0.85em; }}
  .q-card {{ background: white; border-radius: 8px; margin-bottom: 28px;
             box-shadow: 0 1px 4px rgba(0,0,0,.1); padding: 18px 20px; }}
  .q-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
  .q-num {{ background: #1a2a3a; color: white; border-radius: 4px;
            padding: 2px 10px; font-weight: bold; }}
  .q-section {{ color: #888; font-size: 0.85em; }}
  .q-text {{ font-weight: 600; font-size: 1em; padding: 8px 12px;
             background: #f8f9fa; border-left: 4px solid #1a2a3a;
             border-radius: 4px; margin-bottom: 14px; }}
  .configs-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  .config-block {{ background: #fafafa; border-radius: 6px; padding: 12px 14px; }}
  .config-header {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }}
  .config-name {{ font-weight: bold; font-size: 0.95em; }}
  .meta {{ color: #999; font-size: 0.78em; margin-left: auto; }}
  .dims-row {{ font-size: 0.82em; margin-bottom: 6px; color: #555; }}
  .section-title {{ font-weight: 600; font-size: 0.72em; text-transform: uppercase;
                    color: #888; margin: 6px 0 3px; letter-spacing: .04em; }}
  .raisonnement {{ background: #fff8e1; border-left: 3px solid #f39c12; border-radius: 4px;
                   padding: 7px 10px; font-size: 0.83em; color: #555; margin-bottom: 7px; }}
  .pill {{ display: inline-block; color: white; border-radius: 4px;
           padding: 1px 8px; font-size: 0.85em; font-weight: bold; }}
  .pill.grey {{ background: #95a5a6; }}
  .flag {{ background: #c0392b; color: white; border-radius: 3px;
           padding: 1px 5px; font-size: 0.78em; font-weight: bold; margin-right: 2px; }}
  details summary {{ cursor: pointer; color: #2980b9; font-size: 0.8em; padding: 3px 0; }}
  details summary:hover {{ text-decoration: underline; }}
  .answer-box {{ white-space: pre-wrap; font-family: monospace; font-size: 0.76em;
                 background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;
                 padding: 10px; margin-top: 6px; max-height: 380px; overflow-y: auto; }}
  @media (max-width: 900px) {{ .configs-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="topbar">
  <h1>Agentic RAG (v11) vs Decomp+Raptor (v10) — 7 questions · Juge V4.3 GPT-4o</h1>
  <nav>{nav}</nav>
</div>
<div class="container">
  <h3 style="margin-top:0;color:#555;font-size:0.9em">Résumé global</h3>
  <table class="summary-tbl">{summary_rows}</table>
  {q_blocks}
</div>
</body>
</html>'''

OUT_HTML.write_text(html, encoding='utf-8')
print(f'\nHTML → {OUT_HTML}  ({OUT_HTML.stat().st_size // 1024} Ko)')
print('\nTerminé.')
