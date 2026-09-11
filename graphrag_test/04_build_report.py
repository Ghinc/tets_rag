"""
Génère rapport_graphrag.html depuis comparison_graphrag.json.
Même design que rapport_comparaison.html (LightRAG vs Pipeline).
Inclut tableau 3 systèmes (GraphRAG + LightRAG + Pipeline).
"""
import json, pathlib, html as html_module

GR_JSON  = pathlib.Path(__file__).parent / "comparison_graphrag.json"
LR_JSON  = pathlib.Path(__file__).parent.parent / "lightrag_test" / "comparison_results.json"
OUT_HTML = pathlib.Path(__file__).parent / "rapport_graphrag.html"

QUESTIONS_ORDER = [
    "bien_etre_ajaccio",
    "corte_vs_bastia",
    "etudiants_qov",
    "scores_satisfaction",
    "environnement_bienetre",
]

Q_LABELS = {
    "bien_etre_ajaccio":    "Facteurs de bien-être à Ajaccio",
    "corte_vs_bastia":      "Corte vs Bastia : revenus &amp; logement",
    "etudiants_qov":        "Perception QoV des étudiants",
    "scores_satisfaction":  "Communes aux scores les plus élevés",
    "environnement_bienetre": "Environnement naturel &amp; bien-être",
}


def score_color(s: float, system: str) -> str:
    if system == "gr":
        return "#7C3AED"   # violet GraphRAG
    if system == "lr":
        return "#C05621"   # orange LightRAG
    return "#2563EB"       # bleu Pipeline


def dim_bar(score: float, max_score: int = 5) -> str:
    pct = score / max_score * 100
    return f'<div style="width:100%;background:#E5E7EB;border-radius:3px;height:6px;margin-top:2px"><div style="width:{pct:.0f}%;background:currentColor;height:6px;border-radius:3px"></div></div>'


def render_sources_pipeline(sources: list) -> str:
    if not sources:
        return '<p style="color:var(--muted);font-size:0.78rem;padding:0.5rem 1rem">Aucune source disponible.</p>'
    items = []
    seen = set()
    for s in sources[:30]:
        label = s.get("label") or s.get("type") or "Source"
        commune = s.get("commune", "")
        view = s.get("view", "")
        key = (label, commune)
        if key in seen:
            continue
        seen.add(key)
        excerpt = (s.get("text_excerpt") or "")[:200]
        commune_tag = f' <span style="color:var(--muted);font-size:0.68rem">· {commune}</span>' if commune else ""
        view_tag = f' <span style="color:var(--muted);font-size:0.68rem">· {view}</span>' if view else ""
        excerpt_html = f'<div style="color:var(--muted);font-size:0.72rem;margin-top:2px;font-style:italic">{html_module.escape(excerpt[:180])}{"…" if len(excerpt)>179 else ""}</div>' if excerpt else ""
        items.append(f'<div style="background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:0.35rem 0.6rem;font-size:0.75rem;margin-bottom:0.25rem"><span style="font-weight:600;font-family:monospace">{html_module.escape(label)}</span>{commune_tag}{view_tag}{excerpt_html}</div>')
    return "\n".join(items)


def render_sources_graphrag(sources: list, mode: str) -> str:
    return f'<p style="color:var(--muted);font-size:0.78rem;padding:0.5rem 0;font-style:italic">GraphRAG mode <strong>{mode}</strong> : les sources sont agrégées dans les community reports internes — non exposées directement par le CLI. Pour un audit, consulter l\'index GraphRAG dans <code>graphrag_test/output/</code>.</p>'


def render_answer(text: str) -> str:
    if not text:
        return '<p style="color:var(--muted);font-style:italic">Réponse vide.</p>'
    lines = text.split("\n")
    html_parts = []
    for line in lines:
        line = line.rstrip()
        if line.startswith("### "):
            html_parts.append(f"<h4>{html_module.escape(line[4:])}</h4>")
        elif line.startswith("## "):
            html_parts.append(f"<h3>{html_module.escape(line[3:])}</h3>")
        elif line.startswith("# "):
            html_parts.append(f"<h3>{html_module.escape(line[2:])}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            html_parts.append(f"<li>{html_module.escape(line[2:])}</li>")
        elif line.startswith("**") and line.endswith("**"):
            html_parts.append(f"<p><strong>{html_module.escape(line[2:-2])}</strong></p>")
        elif line == "":
            html_parts.append("<br>")
        else:
            # inline bold
            import re
            line_html = re.sub(r"\*\*(.+?)\*\*", lambda m: f"<strong>{html_module.escape(m.group(1))}</strong>", html_module.escape(line))
            html_parts.append(f"<p>{line_html}</p>")

    # wrap consecutive <li> in <ul>
    out = ""
    in_ul = False
    for part in html_parts:
        if part.startswith("<li>"):
            if not in_ul:
                out += "<ul>"
                in_ul = True
            out += part
        else:
            if in_ul:
                out += "</ul>"
                in_ul = False
            out += part
    if in_ul:
        out += "</ul>"
    return out


def score_card(score: float, sys_color: str, label: str) -> str:
    return f'<div style="text-align:center;min-width:2.8rem"><div style="font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em">{label}</div><div style="font-size:0.82rem;font-weight:600;color:{sys_color}">{score}</div></div>'


def build_html(gr_results: list, lr_by_id: dict) -> str:
    # Compute averages
    gr_avg = round(sum(r["graphrag"]["eval"].get("note_globale", 0) for r in gr_results) / len(gr_results), 2)
    pl_avg = round(sum(r["pipeline"]["eval"].get("note_globale", 0) for r in gr_results) / len(gr_results), 2)
    lr_avg = 4.08  # from previous run

    by_id = {r["id"]: r for r in gr_results}

    # Score table rows
    table_rows = ""
    for qid in QUESTIONS_ORDER:
        r = by_id.get(qid)
        if not r:
            continue
        gr_g = r["graphrag"]["eval"].get("note_globale", 0)
        pl_g = r["pipeline"]["eval"].get("note_globale", 0)
        lr_r = lr_by_id.get(qid)
        lr_g = lr_r["lightrag"]["eval"].get("note_globale", 0) if lr_r else "—"

        winner = "gr" if gr_g >= pl_g and gr_g >= (lr_g if isinstance(lr_g, (int, float)) else 0) else ("pl" if pl_g >= (lr_g if isinstance(lr_g, (int, float)) else 0) else "lr")
        bg = {"gr": "#F5F3FF", "lr": "#FEF0E6", "pl": "#EFF6FF"}.get(winner, "")
        table_rows += f"""<tr style="background:{bg}">
          <td>{Q_LABELS.get(qid, qid)}</td>
          <td style="text-align:center;font-weight:600;color:#7C3AED">{gr_g}</td>
          <td style="text-align:center;font-weight:600;color:#C05621">{lr_g}</td>
          <td style="text-align:center;font-weight:600;color:#2563EB">{pl_g}</td>
        </tr>"""

    # Q cards
    q_cards = ""
    for qid in QUESTIONS_ORDER:
        r = by_id.get(qid)
        if not r:
            continue
        gr = r["graphrag"]
        pl = r["pipeline"]
        question_label = r["question"]
        gr_eval = gr["eval"]
        pl_eval = pl["eval"]
        mode = gr.get("mode", "global")

        def dim_scores_strip(ev, color):
            dims = [
                ("pertinence", "Pertinence"),
                ("precision_factuelle", "Précision"),
                ("exhaustivite", "Exhaustivité"),
                ("tracabilite", "Traçabilité"),
            ]
            cards = "".join(
                f'<div style="text-align:center;min-width:2.8rem"><div style="font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em">{lbl}</div><div style="font-size:0.82rem;font-weight:600;color:{color}">{ev.get(key,{}).get("score","?")}</div></div>'
                for key, lbl in dims
            )
            return cards

        gr_dims = dim_scores_strip(gr_eval, "#7C3AED")
        pl_dims = dim_scores_strip(pl_eval, "#2563EB")

        gr_answer_html = render_answer(gr.get("answer", ""))
        pl_answer_html = render_answer(pl.get("answer", ""))

        gr_src_html = render_sources_graphrag(gr.get("sources", []), mode)
        pl_src_html = render_sources_pipeline(pl.get("sources", []))

        q_cards += f"""
<div style="margin-bottom:2.5rem">
  <div style="display:flex;align-items:flex-start;gap:0.75rem;padding:0.9rem 1.1rem;background:var(--surface);border:1px solid var(--border);border-radius:8px 8px 0 0;border-bottom:none">
    <div style="flex-shrink:0;width:1.7rem;height:1.7rem;background:#1D7874;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:600">{QUESTIONS_ORDER.index(qid)+1}</div>
    <div style="font-family:'Spectral',serif;font-size:1rem;font-weight:600;line-height:1.3">{question_label}</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);border:1px solid var(--border);border-radius:0 0 8px 8px;overflow:hidden">
    <!-- GraphRAG -->
    <div style="background:var(--surface);display:flex;flex-direction:column">
      <div style="padding:0.5rem 1rem;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;background:#F5F3FF;color:#7C3AED;border-bottom:2px solid #7C3AED">
        GraphRAG (Microsoft) · mode {mode}
      </div>
      <div style="display:flex;gap:0.4rem;padding:0.6rem 1rem;background:var(--surface2);border-bottom:1px solid var(--border);flex-wrap:wrap;align-items:center">
        {gr_dims}
        <div style="margin-left:auto;font-family:'Spectral',serif;font-size:1.25rem;font-weight:600;color:#7C3AED">{gr_eval.get("note_globale","?")}</div>
      </div>
      <div style="padding:1rem;flex:1;font-size:0.84rem;line-height:1.65;overflow-wrap:break-word">{gr_answer_html}</div>
      <div style="padding:0.6rem 1rem;font-size:0.78rem;color:var(--muted);border-top:1px dashed var(--border);font-style:italic">« {html_module.escape(gr_eval.get("commentaire_global",""))} »</div>
      <div style="border-top:1px solid var(--border)">
        <details>
          <summary style="cursor:pointer;padding:0.55rem 1rem;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);list-style:none;display:flex;align-items:center;gap:0.4rem">
            ▸ Sources GraphRAG
          </summary>
          <div style="padding:0 1rem 0.75rem">{gr_src_html}</div>
        </details>
      </div>
    </div>
    <!-- Pipeline -->
    <div style="background:var(--surface);display:flex;flex-direction:column">
      <div style="padding:0.5rem 1rem;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;background:#EFF6FF;color:#2563EB;border-bottom:2px solid #2563EB">
        Pipeline RAG v10 · RAPTOR + décomposition
      </div>
      <div style="display:flex;gap:0.4rem;padding:0.6rem 1rem;background:var(--surface2);border-bottom:1px solid var(--border);flex-wrap:wrap;align-items:center">
        {pl_dims}
        <div style="margin-left:auto;font-family:'Spectral',serif;font-size:1.25rem;font-weight:600;color:#2563EB">{pl_eval.get("note_globale","?")}</div>
      </div>
      <div style="padding:1rem;flex:1;font-size:0.84rem;line-height:1.65;overflow-wrap:break-word">{pl_answer_html}</div>
      <div style="padding:0.6rem 1rem;font-size:0.78rem;color:var(--muted);border-top:1px dashed var(--border);font-style:italic">« {html_module.escape(pl_eval.get("commentaire_global",""))} »</div>
      <div style="border-top:1px solid var(--border)">
        <details>
          <summary style="cursor:pointer;padding:0.55rem 1rem;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);list-style:none;display:flex;align-items:center;gap:0.4rem">
            ▸ Sources Pipeline ({len(pl.get("sources",[]))} entrées)
          </summary>
          <div style="padding:0 1rem 0.75rem">{pl_src_html}</div>
        </details>
      </div>
    </div>
  </div>
</div>"""

    return f"""<title>GraphRAG vs Pipeline RAG v10</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=Inter:wght@400;500;600&display=swap">
<style>
:root {{
  --bg: #F3F2EE; --surface: #FFFFFF; --surface2: #F7F6F2;
  --border: #DDE0DA; --text: #1C1C1A; --muted: #6C6C68;
  --accent: #1D7874;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #181A18; --surface: #1F221F; --surface2: #1A1D1A;
    --border: #2D312D; --text: #E4E4DF; --muted: #8E8E85; --accent: #3AA89F;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #181A18; --surface: #1F221F; --surface2: #1A1D1A;
  --border: #2D312D; --text: #E4E4DF; --muted: #8E8E85; --accent: #3AA89F;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', system-ui, sans-serif; font-size: 0.9rem; line-height: 1.65;
        background: var(--bg); color: var(--text); padding: 2.5rem 1.25rem 5rem; }}
h1,h2,h3,h4 {{ font-family: 'Spectral', Georgia, serif; text-wrap: balance; }}
h3 {{ font-size: 0.92rem; font-weight: 600; margin: 0.75rem 0 0.2rem; }}
h4 {{ font-size: 0.86rem; font-weight: 600; margin: 0.5rem 0 0.15rem; }}
p {{ margin-bottom: 0.4rem; }}
ul {{ padding-left: 1.3rem; margin-bottom: 0.4rem; }}
li {{ margin-bottom: 0.15rem; }}
strong {{ font-weight: 600; }}
.page {{ max-width: 1080px; margin: 0 auto; }}
table {{ width:100%; border-collapse:collapse; font-size:0.84rem; }}
th {{ text-align:left; padding:0.6rem 0.75rem; font-weight:600; border-bottom:2px solid var(--border); background:var(--surface2); }}
td {{ padding:0.55rem 0.75rem; border-bottom:1px solid var(--border); }}
details summary::-webkit-details-marker {{ display:none; }}
@media (max-width:720px) {{
  div[style*="grid-template-columns:1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
}}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>
<div class="page">
<header style="border-bottom:2px solid #7C3AED;padding-bottom:1.5rem;margin-bottom:2rem">
  <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#7C3AED;margin-bottom:0.5rem">Évaluation comparative — RAG sur données qualité de vie en Corse</div>
  <h1 style="font-size:1.9rem;font-weight:600">GraphRAG (Microsoft) vs Pipeline RAG v10</h1>
  <p style="color:var(--muted);font-size:0.85rem;margin-top:0.4rem">5 questions identiques · Juge automatique gpt-4o-mini · 4 dimensions · Corpus llm_wiki_data (23 fichiers)</p>
</header>

<!-- Scoreboard 3 systèmes -->
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--border);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:2rem">
  <div style="background:var(--surface);padding:1.5rem;text-align:center">
    <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#7C3AED;margin-bottom:0.5rem">GraphRAG</div>
    <div style="font-family:'Spectral',serif;font-size:3rem;font-weight:600;line-height:1;color:#7C3AED">{gr_avg}</div>
    <div style="font-size:0.78rem;color:var(--muted);margin-top:0.25rem">/ 5 · moyenne 5 questions</div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:0.5rem">Knowledge graph · community reports · gpt-4o-mini</div>
  </div>
  <div style="background:var(--surface);padding:1.5rem;text-align:center">
    <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#C05621;margin-bottom:0.5rem">LightRAG (référence)</div>
    <div style="font-family:'Spectral',serif;font-size:3rem;font-weight:600;line-height:1;color:#C05621">{lr_avg}</div>
    <div style="font-size:0.78rem;color:var(--muted);margin-top:0.25rem">/ 5 · run précédent</div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:0.5rem">Graphe + vecteurs hybrides · gpt-4o-mini</div>
  </div>
  <div style="background:var(--surface);padding:1.5rem;text-align:center">
    <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#2563EB;margin-bottom:0.5rem">Pipeline RAG v10</div>
    <div style="font-family:'Spectral',serif;font-size:3rem;font-weight:600;line-height:1;color:#2563EB">{pl_avg}</div>
    <div style="font-size:0.78rem;color:var(--muted);margin-top:0.25rem">/ 5 · moyenne 5 questions</div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:0.5rem">RAPTOR + décomposition · mistral-large + haiku</div>
  </div>
</div>

<!-- Tableau récapitulatif -->
<div style="overflow-x:auto;margin-bottom:2rem">
  <table>
    <thead>
      <tr>
        <th>Question</th>
        <th style="text-align:center;color:#7C3AED">GraphRAG</th>
        <th style="text-align:center;color:#C05621">LightRAG</th>
        <th style="text-align:center;color:#2563EB">Pipeline v10</th>
      </tr>
    </thead>
    <tbody>{table_rows}
      <tr style="font-weight:600;background:var(--surface2)">
        <td>Moyenne</td>
        <td style="text-align:center;color:#7C3AED">{gr_avg}</td>
        <td style="text-align:center;color:#C05621">{lr_avg}</td>
        <td style="text-align:center;color:#2563EB">{pl_avg}</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- Note méthodologique GraphRAG -->
<div style="background:#F5F3FF;border-left:3px solid #7C3AED;border-radius:0 6px 6px 0;padding:0.9rem 1.1rem;margin-bottom:1.5rem;font-size:0.84rem;color:#5B21B6">
  <strong>GraphRAG — deux modes de requête :</strong>
  <ul style="margin-top:0.4rem;padding-left:1.2rem">
    <li><strong>Global</strong> : synthèse depuis les community reports (résumés de clusters thématiques). Idéal pour les questions d'agrégation et de patterns transversaux. Sources non consultables directement.</li>
    <li><strong>Local</strong> : récupération depuis les entités et leurs relations dans le graphe. Idéal pour les questions ciblées sur une commune ou un groupe. Sources via l'index GraphRAG.</li>
  </ul>
</div>

<div style="font-family:'Spectral',serif;font-size:1.15rem;font-weight:600;margin-bottom:1.25rem;padding-bottom:0.5rem;border-bottom:1px solid var(--border)">Réponses détaillées par question</div>

{q_cards}

<footer style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid var(--border);font-size:0.8rem;color:var(--muted);line-height:1.6">
  <p><strong>GraphRAG v2.7.x</strong> (Microsoft Research, 2024) — indexation par extraction d'entités + relations + community summaries via LLM. Modes global (Leiden clustering) et local (entity retrieval). <strong>Juge</strong> : gpt-4o-mini, 4 dimensions (pertinence, précision factuelle, exhaustivité, traçabilité), note 1–5.</p>
  <p style="margin-top:0.5rem"><strong>Limitation</strong> — GraphRAG et LightRAG ont tous les deux tourné sur 23 fichiers (llm_wiki_data) ; le pipeline dispose du corpus ChromaDB complet. La comparaison est indicative, pas iso-corpus.</p>
</footer>
</div>"""


def main():
    gr_data = json.loads(GR_JSON.read_text(encoding="utf-8"))
    lr_data = json.loads(LR_JSON.read_text(encoding="utf-8"))
    lr_by_id = {r["id"]: r for r in lr_data["results"]}

    html_out = build_html(gr_data["results"], lr_by_id)
    OUT_HTML.write_text(html_out, encoding="utf-8")
    print(f"✓ {OUT_HTML}")


if __name__ == "__main__":
    main()
