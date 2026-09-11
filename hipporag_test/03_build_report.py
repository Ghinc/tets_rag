"""
HippoRAG v2 — Génération du rapport HTML.
Tableau 4 systèmes : Pipeline > GraphRAG > LightRAG > HippoRAG (à confirmer).
Nouveauté : sources HippoRAG = TEXTE COMPLET consultable avec score PPR.
Entrées :
  - comparison_hipporag.json
  - ../lightrag_test/comparison_results.json
  - ../graphrag_test/comparison_graphrag.json
Sortie : rapport_hipporag.html (ouvrir directement dans le navigateur)
"""
import json, pathlib, html

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIPPORAG_CMP = pathlib.Path(__file__).parent / "comparison_hipporag.json"
LIGHTRAG_CMP = ROOT / "lightrag_test"  / "comparison_results.json"
GRAPHRAG_CMP = ROOT / "graphrag_test"  / "comparison_graphrag.json"
OUT          = pathlib.Path(__file__).parent / "rapport_hipporag.html"


# ---------- chargement données ----------

def load_json(p: pathlib.Path) -> dict:
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    print(f"  [WARN] fichier introuvable : {p}")
    return {}

h_data  = load_json(HIPPORAG_CMP)
lr_data = load_json(LIGHTRAG_CMP)
gr_data = load_json(GRAPHRAG_CMP)

def get_score(eval_dict: dict) -> float:
    """Récupère le score depuis eval, gère score_global (float) et note_globale (str avec virgule)."""
    if not eval_dict:
        return 0.0
    raw = eval_dict.get("score_global") or eval_dict.get("note_globale") or 0
    try:
        return float(str(raw).replace(",", "."))
    except (ValueError, TypeError):
        return 0.0

def get_items(data: dict) -> list:
    """Récupère la liste de comparaisons depuis results ou comparisons."""
    return data.get("comparisons") or data.get("results") or []


# Scores globaux par système
h_avg  = h_data.get("systems", {}).get("hipporag",  {}).get("avg_score", 0)
p_avg  = h_data.get("systems", {}).get("pipeline",  {}).get("avg_score", 0)

lr_items = get_items(lr_data)
gr_items = get_items(gr_data)

lr_scores = [get_score(c.get("lightrag", {}).get("eval", {})) for c in lr_items]
gr_scores = [get_score(c.get("graphrag", {}).get("eval", {})) for c in gr_items]
lr_avg = round(sum(lr_scores) / len(lr_scores), 2) if lr_scores else 0.0
gr_avg = round(sum(gr_scores) / len(gr_scores), 2) if gr_scores else 0.0

# Index par qid pour les autres systèmes
lr_by_id = {c["id"]: c for c in lr_items}
gr_by_id = {c["id"]: c for c in gr_items}

# Questions (depuis hipporag)
questions = get_items(h_data)


# ---------- helpers HTML ----------

def bar(score, color):
    pct = int(score / 5 * 100)
    return f'<div class="bar-wrap"><div class="bar" style="width:{pct}%;background:{color}"></div><span class="bar-val">{score}/5</span></div>'

def score_pill(score):
    try:
        s = float(score)
    except (TypeError, ValueError):
        return f'<span class="pill pill-red">{score}</span>'
    if s >= 4.5: cls = "pill-green"
    elif s >= 3.5: cls = "pill-yellow"
    else: cls = "pill-red"
    return f'<span class="pill {cls}">{score}</span>'

def h(text): return html.escape(str(text))

def sources_hipporag(sources):
    if not sources:
        return '<p class="no-src">Aucune source disponible.</p>'
    parts = []
    for s in sources[:5]:
        score_val = round(s.get("score", 0), 4)
        fname = h(s.get("file", "—"))
        text_preview = h(s.get("text", "")[:500])
        if len(s.get("text", "")) > 500:
            text_preview += "…"
        parts.append(f"""
<div class="src-block">
  <div class="src-header">
    <span class="src-rank">#{s.get('rank','')} — {fname}</span>
    <span class="src-score">PPR: {score_val}</span>
  </div>
  <div class="src-text">{text_preview}</div>
</div>""")
    return "".join(parts)

def sources_pipeline(sources):
    if not sources:
        return '<p class="no-src">Aucune source disponible.</p>'
    parts = []
    for s in sources[:5]:
        commune = h(s.get("commune", s.get("vue", "—")))
        excerpt  = h((s.get("excerpt") or s.get("content") or "")[:300])
        if len(excerpt) > 300:
            excerpt += "…"
        parts.append(f'<div class="src-block"><span class="src-rank">{commune}</span><div class="src-text">{excerpt}</div></div>')
    return "".join(parts)


# ---------- génération HTML ----------

PALETTE = {
    "pipeline":  "#2563EB",
    "hipporag":  "#059669",
    "graphrag":  "#D97706",
    "lightrag":  "#7C3AED",
}

def build_question_section(cmp):
    qid      = cmp["id"]
    question = h(cmp["question"])
    h_ans    = h(cmp["hipporag"]["answer"])
    p_ans    = h(cmp["pipeline"]["answer"])
    h_eval   = cmp["hipporag"]["eval"]
    p_eval   = cmp["pipeline"]["eval"]

    # Autres systèmes
    lr = lr_by_id.get(qid, {})
    gr = gr_by_id.get(qid, {})
    lr_score = get_score(lr.get("lightrag", {}).get("eval", {})) if lr else "—"
    gr_score = get_score(gr.get("graphrag", {}).get("eval", {})) if gr else "—"
    lr_ans   = h(lr.get("lightrag", {}).get("answer", "—")[:800] if lr else "—")
    gr_ans   = h(gr.get("graphrag", {}).get("answer", "—")[:800] if gr else "—")

    return f"""
<section class="question-block" id="{qid}">
  <h3 class="q-title">{question}</h3>

  <div class="mini-scoreboard">
    <div class="mini-score" style="border-color:{PALETTE['pipeline']}">
      <span class="sys-name">Pipeline v10</span>
      {score_pill(p_eval.get('score_global', 0))}
    </div>
    <div class="mini-score" style="border-color:{PALETTE['hipporag']}">
      <span class="sys-name">HippoRAG v2</span>
      {score_pill(h_eval.get('score_global', 0))}
    </div>
    <div class="mini-score" style="border-color:{PALETTE['graphrag']}">
      <span class="sys-name">GraphRAG</span>
      {score_pill(gr_score)}
    </div>
    <div class="mini-score" style="border-color:{PALETTE['lightrag']}">
      <span class="sys-name">LightRAG</span>
      {score_pill(lr_score)}
    </div>
  </div>

  <div class="answer-grid">

    <div class="answer-card" style="--accent:{PALETTE['pipeline']}">
      <div class="card-header">Pipeline RAG v10</div>
      <div class="eval-row">
        {bar(p_eval.get('pertinence',0), PALETTE['pipeline'])}
        <span class="eval-label">Pertinence</span>
        {bar(p_eval.get('precision_factuelle',0), PALETTE['pipeline'])}
        <span class="eval-label">Précision</span>
      </div>
      <p class="answer-text">{p_ans[:1200]}{"…" if len(cmp["pipeline"]["answer"]) > 1200 else ""}</p>
      <details class="sources-details">
        <summary>Sources ({len(cmp["pipeline"].get("sources",[]))} passages)</summary>
        {sources_pipeline(cmp["pipeline"].get("sources", []))}
      </details>
      <div class="judge-comment">{h(p_eval.get('commentaire',''))}</div>
    </div>

    <div class="answer-card" style="--accent:{PALETTE['hipporag']}">
      <div class="card-header">HippoRAG v2 <span class="badge-new">sources consultables</span></div>
      <div class="eval-row">
        {bar(h_eval.get('pertinence',0), PALETTE['hipporag'])}
        <span class="eval-label">Pertinence</span>
        {bar(h_eval.get('precision_factuelle',0), PALETTE['hipporag'])}
        <span class="eval-label">Précision</span>
      </div>
      <p class="answer-text">{h_ans[:1200]}{"…" if len(cmp["hipporag"]["answer"]) > 1200 else ""}</p>
      <details class="sources-details">
        <summary>Sources PPR ({len(cmp["hipporag"].get("sources",[]))} passages — texte intégral)</summary>
        {sources_hipporag(cmp["hipporag"].get("sources", []))}
      </details>
      <div class="judge-comment">{h(h_eval.get('commentaire',''))}</div>
    </div>

  </div>

  <div class="other-systems">
    <details>
      <summary>Réponses GraphRAG &amp; LightRAG</summary>
      <div class="other-grid">
        <div class="other-card" style="border-left:3px solid {PALETTE['graphrag']}">
          <strong>GraphRAG 2.7.2</strong> {score_pill(gr_score)}
          <p>{gr_ans}</p>
        </div>
        <div class="other-card" style="border-left:3px solid {PALETTE['lightrag']}">
          <strong>LightRAG 1.5.6</strong> {score_pill(lr_score)}
          <p>{lr_ans}</p>
        </div>
      </div>
    </details>
  </div>
</section>"""


question_sections = "\n".join(build_question_section(c) for c in questions)

# Classement général
systems_sorted = sorted([
    ("Pipeline RAG v10", p_avg,  PALETTE["pipeline"]),
    ("HippoRAG v2",      h_avg,  PALETTE["hipporag"]),
    ("GraphRAG 2.7.2",   gr_avg, PALETTE["graphrag"]),
    ("LightRAG 1.5.6",   lr_avg, PALETTE["lightrag"]),
], key=lambda x: x[1], reverse=True)

podium_rows = ""
for rank, (name, score, color) in enumerate(systems_sorted, 1):
    pct = int(score / 5 * 100)
    medal = ["🥇", "🥈", "🥉", "4e"][rank - 1]
    sources_note = ""
    if name.startswith("HippoRAG") or name.startswith("Pipeline"):
        sources_note = '<span class="src-ok">✓ sources consultables</span>'
    else:
        sources_note = '<span class="src-no">✗ sources non consultables</span>'
    podium_rows += f"""
<tr class="{'rank-1' if rank==1 else ''}">
  <td class="rank-cell">{medal} #{rank}</td>
  <td class="sys-cell" style="color:{color}">{name}</td>
  <td class="score-cell">{score}/5</td>
  <td class="bar-cell"><div class="bar-wrap"><div class="bar" style="width:{pct}%;background:{color}"></div></div></td>
  <td class="note-cell">{sources_note}</td>
</tr>"""

html_out = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Benchmark RAG — 4 systèmes</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {{
    --bg: #f8f9fa; --surface: #ffffff; --text: #1a1a2e;
    --muted: #6b7280; --border: #e5e7eb;
    --pipeline: {PALETTE['pipeline']}; --hipporag: {PALETTE['hipporag']};
    --graphrag: {PALETTE['graphrag']}; --lightrag: {PALETTE['lightrag']};
    --radius: 10px; --shadow: 0 2px 12px rgba(0,0,0,.07);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#111827; --surface:#1f2937; --text:#f3f4f6;
      --muted:#9ca3af; --border:#374151;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#111827; --surface:#1f2937; --text:#f3f4f6;
    --muted:#9ca3af; --border:#374151;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
  a {{ color: inherit; }}

  /* Header */
  header {{ background: #0f172a; color: #f1f5f9; padding: 3rem 2rem 2.5rem; text-align: center; }}
  header h1 {{ font-size: clamp(1.6rem, 4vw, 2.4rem); font-weight: 700; letter-spacing: -.02em; }}
  header p {{ margin-top: .5rem; color: #94a3b8; font-size: .95rem; }}

  .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }}

  /* Tableau podium */
  .scoreboard {{ background: var(--surface); border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; margin-bottom: 2.5rem; }}
  .scoreboard h2 {{ padding: 1.2rem 1.5rem; border-bottom: 1px solid var(--border); font-size: 1.05rem; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: .7rem 1rem; text-align: left; }}
  th {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); border-bottom: 1px solid var(--border); }}
  tr + tr {{ border-top: 1px solid var(--border); }}
  .rank-1 {{ background: color-mix(in srgb, {PALETTE['pipeline']} 5%, transparent); }}
  .rank-cell {{ font-size: 1.1rem; white-space: nowrap; }}
  .sys-cell {{ font-weight: 600; }}
  .score-cell {{ font-size: 1.1rem; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .bar-cell {{ width: 200px; }}
  .bar-wrap {{ display: flex; align-items: center; gap: .5rem; }}
  .bar {{ height: 8px; border-radius: 4px; transition: width .4s; }}
  .bar-val {{ font-size: .8rem; color: var(--muted); white-space: nowrap; }}
  .src-ok {{ background: #d1fae5; color: #065f46; padding: .2rem .5rem; border-radius: 4px; font-size: .75rem; font-weight: 600; }}
  .src-no {{ background: #fee2e2; color: #991b1b; padding: .2rem .5rem; border-radius: 4px; font-size: .75rem; }}

  /* Question blocks */
  .question-block {{ background: var(--surface); border-radius: var(--radius); box-shadow: var(--shadow); padding: 1.5rem; margin-bottom: 2rem; }}
  .q-title {{ font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 1rem; padding-bottom: .6rem; border-bottom: 1px solid var(--border); }}

  .mini-scoreboard {{ display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: 1.2rem; }}
  .mini-score {{ display: flex; align-items: center; gap: .5rem; padding: .4rem .8rem; border: 2px solid; border-radius: 20px; font-size: .85rem; }}
  .sys-name {{ color: var(--muted); }}

  .pill {{ padding: .15rem .5rem; border-radius: 99px; font-size: .8rem; font-weight: 700; }}
  .pill-green {{ background: #d1fae5; color: #065f46; }}
  .pill-yellow {{ background: #fef3c7; color: #92400e; }}
  .pill-red {{ background: #fee2e2; color: #991b1b; }}

  .answer-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }}
  @media (max-width: 750px) {{ .answer-grid {{ grid-template-columns: 1fr; }} }}

  .answer-card {{ border: 2px solid var(--accent, #888); border-radius: 8px; padding: 1rem; display: flex; flex-direction: column; gap: .6rem; }}
  .card-header {{ font-size: .9rem; font-weight: 700; color: var(--accent); display: flex; align-items: center; gap: .5rem; }}
  .badge-new {{ background: #d1fae5; color: #065f46; font-size: .7rem; padding: .1rem .45rem; border-radius: 4px; font-weight: 600; }}

  .eval-row {{ display: flex; flex-direction: column; gap: .3rem; }}
  .eval-label {{ font-size: .75rem; color: var(--muted); }}

  .answer-text {{ font-size: .88rem; line-height: 1.55; color: var(--text); white-space: pre-wrap; }}

  details.sources-details {{ border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
  details.sources-details summary {{ padding: .5rem .75rem; cursor: pointer; font-size: .82rem; font-weight: 600; color: var(--muted); background: var(--bg); }}
  details.sources-details summary:hover {{ background: color-mix(in srgb, var(--accent) 8%, var(--bg)); }}

  .src-block {{ padding: .7rem .75rem; border-top: 1px solid var(--border); }}
  .src-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: .3rem; }}
  .src-rank {{ font-size: .78rem; font-weight: 600; color: var(--muted); }}
  .src-score {{ font-size: .75rem; color: var(--muted); font-variant-numeric: tabular-nums; }}
  .src-text {{ font-size: .82rem; line-height: 1.5; color: var(--text); white-space: pre-wrap; }}
  .no-src {{ color: var(--muted); font-size: .82rem; padding: .5rem; }}

  .judge-comment {{ font-size: .8rem; font-style: italic; color: var(--muted); border-top: 1px solid var(--border); padding-top: .5rem; }}

  .other-systems {{ margin-top: .5rem; }}
  .other-systems > details > summary {{ cursor: pointer; font-size: .82rem; color: var(--muted); padding: .3rem 0; }}
  .other-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; margin-top: .75rem; }}
  @media (max-width: 600px) {{ .other-grid {{ grid-template-columns: 1fr; }} }}
  .other-card {{ background: var(--bg); padding: .75rem; border-radius: 6px; font-size: .82rem; }}
  .other-card p {{ margin-top: .4rem; color: var(--text); }}

  .section-title {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; margin-top: 2rem; }}
  footer {{ text-align: center; padding: 2rem; font-size: .8rem; color: var(--muted); border-top: 1px solid var(--border); margin-top: 2rem; }}
</style>
</head>
<body>
<header>
  <h1>Benchmark RAG — 4 Systèmes</h1>
  <p>Enquête bien-être en Corse &nbsp;·&nbsp; 5 questions &nbsp;·&nbsp; Juge : gpt-4o-mini</p>
</header>

<div class="container">

  <section class="scoreboard">
    <h2>Classement général</h2>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>Rang</th><th>Système</th><th>Score moyen</th><th>Barre</th><th>Sources</th>
        </tr>
      </thead>
      <tbody>
        {podium_rows}
      </tbody>
    </table>
    </div>
  </section>

  <h2 class="section-title">Détail par question</h2>
  {question_sections}

</div>

<footer>Rapport généré par 03_build_report.py &nbsp;·&nbsp; HippoRAG v2.0.0a3 · GraphRAG 2.7.2 · LightRAG 1.5.6 · Pipeline RAG v10</footer>
</body>
</html>"""

OUT.write_text(html_out, encoding="utf-8")
print(f"✓ Rapport généré : {OUT}")
print(f"  Classement : " + " > ".join(f"{n} ({s})" for n, s, _ in systems_sorted))
