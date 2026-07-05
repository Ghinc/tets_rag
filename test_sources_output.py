"""
Test : vérification du bloc ## Sources mobilisées sur 3 questions.
Questions : Q7 (OppChoVec Ajaccio), Q11 (entrepreneurs Ajaccio), Q44 (communes dynamiques).
Sortie : test_sources_output.html
"""
import sys, io, json, re, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# sentence_transformers DOIT être importé avant chromadb (conflit libs natives Windows)
from sentence_transformers import SentenceTransformer  # noqa: F401

TARGET_ROWS = [7, 11, 44]
OUT_HTML = Path("test_sources_output.html")

# ── Charger les questions depuis l'Excel ──────────────────────────────────────
from eval_from_excel import load_questions

qs_all = load_questions(r"C:\Users\comiti_g\Downloads\rag_evaluation_with_metrics_full.xlsx")
questions = {q["excel_row"]: q for q in qs_all if q["excel_row"] in TARGET_ROWS}
print(f"Questions chargées : {list(questions.keys())}")

# ── Lancer le pipeline ────────────────────────────────────────────────────────
from rag_v10_raptor_subq import RaptorSubQuestionPipeline

pipeline = RaptorSubQuestionPipeline(chroma_path="./chroma_portrait")
pipeline.init()

results = []
for row in TARGET_ROWS:
    q_info = questions[row]
    q_text = q_info["question"]
    print(f"\n{'='*60}")
    print(f"Q{row} : {q_text}")
    t0 = time.time()
    answer, sources, scoring, sub_qa = pipeline.query(
        q_text, k=5, n_subquestions=5, use_bilan=False
    )
    elapsed = round(time.time() - t0, 1)

    # Séparer réponse et section sources
    sources_section = ""
    main_answer = answer
    split_match = re.search(r'===SOURCES_MOBILISEES===', answer)
    if split_match:
        main_answer = answer[:split_match.start()].strip()
        sources_section = answer[split_match.start():]
        # Nettoyer le marqueur de fin éventuel pour l'affichage
        sources_section = re.sub(r'===FIN_SOURCES===.*$', '', sources_section, flags=re.DOTALL).strip()
        sources_section = sources_section.replace('===SOURCES_MOBILISEES===', '## Sources mobilisées\n')
    else:
        print(f"  ⚠️  Pas de section '===SOURCES_MOBILISEES===' dans la réponse Q{row}")

    results.append({
        "row": row,
        "section": q_info.get("section", ""),
        "subsection": q_info.get("subsection", ""),
        "question": q_text,
        "answer": main_answer,
        "sources_section": sources_section,
        "sources_raw": sources,
        "sub_qa": sub_qa,
        "elapsed": elapsed,
    })
    print(f"  → OK ({elapsed}s, {len(sources)} sources récupérées)")
    print(f"  → Section sources : {'OUI' if sources_section else 'NON'}")

# ── Générer le HTML ───────────────────────────────────────────────────────────

def md_to_html(text: str) -> str:
    """Conversion markdown minimaliste → HTML."""
    import html as _html
    text = _html.escape(text)
    # Titres
    text = re.sub(r"^## (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^### (.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    # Gras
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Listes
    text = re.sub(r"^[-•] (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"(<li>.*</li>\n?)+", lambda m: f"<ul>{m.group()}</ul>", text, flags=re.DOTALL)
    # Sauts de ligne
    text = text.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p>{text}</p>"


def sources_table(sources: list) -> str:
    if not sources:
        return "<em>Aucune source récupérée.</em>"
    rows = ""
    for s in sources:
        t = s.get("type") or s.get("source_type") or ""
        sq_idx = s.get("sub_question_idx", "—")
        view = s.get("view_name") or s.get("view") or "—"
        commune = s.get("commune") or s.get("dim1_value") or "—"
        extrait = (s.get("extrait") or "")[:200]
        dist = s.get("distance")
        dist_str = f"{dist:.3f}" if dist is not None else "—"
        rows += (
            f"<tr>"
            f"<td>SQ{sq_idx}</td>"
            f"<td><code>{t}</code></td>"
            f"<td>{view}</td>"
            f"<td>{commune}</td>"
            f"<td>{dist_str}</td>"
            f"<td style='font-size:0.78em;color:#555'>{extrait}…</td>"
            f"</tr>"
        )
    return (
        "<table class='src-tbl'>"
        "<thead><tr><th>SQ</th><th>Type</th><th>Vue</th><th>Commune</th><th>Dist.</th><th>Extrait</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


q_blocks = ""
for r in results:
    sources_html = md_to_html(r["sources_section"]) if r["sources_section"] else \
        "<p style='color:#c0392b'><em>⚠ Section « Sources mobilisées » absente de la réponse.</em></p>"

    sq_rows = "".join(
        f"<tr><td>{sq['idx']}</td><td>{sq['question']}</td>"
        f"<td style='font-size:0.8em'>{sq['answer'][:400]}…</td></tr>"
        for sq in r["sub_qa"]
    )
    sq_table = (
        "<table class='src-tbl'><thead><tr><th>#</th><th>Sous-question</th><th>Réponse intermédiaire</th></tr></thead>"
        f"<tbody>{sq_rows}</tbody></table>"
    ) if sq_rows else ""

    q_blocks += f"""
<div class="q-card" id="q{r['row']}">
  <div class="q-header">
    <span class="q-num">Q{r['row']}</span>
    <span class="q-meta">{r['section']} — {r['subsection']}</span>
    <span class="q-time">{r['elapsed']}s · {len(r['sources_raw'])} sources récupérées</span>
  </div>
  <div class="q-text">{r['question']}</div>

  <div class="section-label">Réponse synthétisée</div>
  <div class="answer-box">{md_to_html(r['answer'])}</div>

  <div class="section-label sources-label">Sources mobilisées (déclaré par le LLM)</div>
  <div class="sources-box">{sources_html}</div>

  <details>
    <summary>Toutes les sources récupérées par le retriever ({len(r['sources_raw'])})</summary>
    {sources_table(r['sources_raw'])}
  </details>

  <details>
    <summary>Sous-questions et réponses intermédiaires</summary>
    {sq_table}
  </details>
</div>"""

nav = " · ".join(f'<a href="#q{r["row"]}">Q{r["row"]}</a>' for r in results)

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Test sources mobilisées — V_decomp+raptor</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; margin: 0; background: #f0f2f5; color: #222; }}
  .topbar {{ background: #1a2a3a; color: white; padding: 12px 24px;
             position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.3); }}
  .topbar h1 {{ margin: 0 0 4px; font-size: 1em; }}
  .topbar nav a {{ color: #aed6f1; text-decoration: none; font-size: 0.85em; margin-right: 8px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 20px 24px; }}
  .q-card {{ background: white; border-radius: 8px; margin-bottom: 32px;
             box-shadow: 0 1px 4px rgba(0,0,0,.1); padding: 20px 22px; }}
  .q-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }}
  .q-num {{ background: #1a2a3a; color: white; border-radius: 4px; padding: 2px 10px; font-weight: bold; }}
  .q-meta {{ color: #888; font-size: 0.82em; }}
  .q-time {{ color: #aaa; font-size: 0.78em; margin-left: auto; }}
  .q-text {{ font-weight: 600; font-size: 1em; padding: 8px 12px; background: #f8f9fa;
             border-left: 4px solid #1a2a3a; border-radius: 4px; margin-bottom: 16px; }}
  .section-label {{ font-weight: 700; font-size: 0.72em; text-transform: uppercase;
                    letter-spacing: .06em; color: #2980b9; margin: 14px 0 5px; }}
  .sources-label {{ color: #27ae60; }}
  .answer-box {{ background: #fafafa; border: 1px solid #e8e8e8; border-radius: 6px;
                 padding: 12px 16px; line-height: 1.6; }}
  .answer-box p {{ margin: 0 0 8px; }}
  .answer-box ul {{ margin: 4px 0 8px 18px; padding: 0; }}
  .sources-box {{ background: #f0faf4; border: 1px solid #b2dfdb; border-radius: 6px;
                  padding: 12px 16px; line-height: 1.6; }}
  .sources-box h3 {{ color: #1a7a4a; font-size: 0.95em; margin: 8px 0 4px; }}
  .sources-box h4 {{ color: #27ae60; font-size: 0.88em; margin: 6px 0 3px; }}
  .sources-box p {{ margin: 0 0 6px; }}
  .sources-box ul {{ margin: 2px 0 8px 16px; padding: 0; }}
  details {{ margin-top: 12px; }}
  details summary {{ cursor: pointer; color: #2980b9; font-size: 0.82em;
                     padding: 4px 0; user-select: none; }}
  details summary:hover {{ text-decoration: underline; }}
  .src-tbl {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 0.78em; }}
  .src-tbl th {{ background: #ecf0f1; padding: 5px 8px; text-align: left;
                 border: 1px solid #ddd; font-size: 0.85em; }}
  .src-tbl td {{ padding: 4px 8px; border: 1px solid #eee; vertical-align: top; }}
  .src-tbl tr:nth-child(even) td {{ background: #fafafa; }}
  code {{ background: #f0f0f0; border-radius: 3px; padding: 1px 4px; font-size: 0.9em; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>Test — Sources mobilisées · V_decomp+raptor · 3 questions</h1>
  <nav>{nav}</nav>
</div>
<div class="container">
  {q_blocks}
</div>
</body>
</html>"""

OUT_HTML.write_text(html, encoding="utf-8")
print(f"\nHTML → {OUT_HTML} ({OUT_HTML.stat().st_size // 1024} Ko)")
print("Terminé.")
