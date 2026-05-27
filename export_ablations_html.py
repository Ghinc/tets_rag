"""Export ablations_20q JSON to readable HTML for review."""
import json, sys
from pathlib import Path

JSON_FILE = r"comparaisons_rag/ablations_20q_20260513_101927.json"
OUT_FILE = r"comparaisons_rag/ablations_20q_review.html"

with open(JSON_FILE, encoding="utf-8") as f:
    data = json.load(f)

versions = list(data.keys())
# Collect all questions in order
all_entries = {ver: {e["excel_row"]: e for e in data[ver] if e.get("excel_row")} for ver in versions}
rows = [e for e in data[versions[0]] if e.get("excel_row")]

html = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Ablations 20q — Revue des réponses</title>
<style>
  body { font-family: Arial, sans-serif; font-size: 13px; margin: 20px; background: #f5f5f5; }
  h1 { color: #333; }
  .question-block { background: white; border-radius: 8px; padding: 16px; margin-bottom: 24px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
  .q-header { font-weight: bold; font-size: 14px; color: #1a1a2e; margin-bottom: 4px; }
  .q-meta { color: #666; font-size: 11px; margin-bottom: 12px; }
  .config-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .config-card { border: 1px solid #ddd; border-radius: 6px; padding: 10px; background: #fafafa; }
  .config-name { font-weight: bold; font-size: 11px; color: #555; text-transform: uppercase;
                 margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid #eee; }
  .v-vanilla-k10 .config-name { color: #c0392b; }
  .v-vanilla-k25 .config-name { color: #e67e22; }
  .v-decomp .config-name { color: #27ae60; }
  .v-decomp-raptor .config-name { color: #2980b9; }
  .answer { font-size: 12px; line-height: 1.5; color: #333; white-space: pre-wrap; }
  .meta-bar { font-size: 10px; color: #999; margin-bottom: 6px; }
  .error { color: #c0392b; font-style: italic; }
  .toc { background: white; padding: 12px; border-radius: 8px; margin-bottom: 24px; }
  .toc a { color: #2980b9; text-decoration: none; display: block; margin: 2px 0; font-size: 12px; }
  .toc a:hover { text-decoration: underline; }
</style>
</head>
<body>
<h1>Ablations RAG — 20 questions annotées</h1>
<p style="color:#666; font-size:12px;">Source : ablations_20q_20260513_101927.json &nbsp;|&nbsp; 4 configs × 20 questions</p>

<div class="toc">
<strong>Navigation</strong><br>
"""

for i, entry in enumerate(rows, 1):
    row = entry["excel_row"]
    q = entry["question"][:70]
    html += f'<a href="#r{row}">R{row} — {q}{"..." if len(entry["question"])>70 else ""}</a>\n'

html += "</div>\n\n"

for entry in rows:
    row = entry["excel_row"]
    question = entry["question"]
    section = entry.get("section", "")
    subsection = entry.get("subsection", "")

    html += f'<div class="question-block" id="r{row}">\n'
    html += f'<div class="q-header">R{row} — {question}</div>\n'
    html += f'<div class="q-meta">{section} › {subsection}</div>\n'
    html += '<div class="config-grid">\n'

    for ver in versions:
        css_class = ver.replace("_", "-")
        e = all_entries[ver].get(row, {})
        html += f'<div class="config-card {css_class}">\n'
        html += f'<div class="config-name">{ver}</div>\n'
        if e.get("status") == "ok":
            n_src = e.get("n_sources", 0)
            n_sq = e.get("n_subquestions", 0)
            t = e.get("elapsed_s", "?")
            html += f'<div class="meta-bar">{n_src} sources | {n_sq} SQ | {t}s</div>\n'
            answer = e.get("answer", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f'<div class="answer">{answer}</div>\n'
        else:
            err = e.get("error", "status=" + str(e.get("status","?")))
            html += f'<div class="error">ERREUR : {err[:100]}</div>\n'
        html += '</div>\n'

    html += '</div>\n</div>\n\n'

html += "</body></html>\n"

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(html)
print(f"HTML exporté : {OUT_FILE}")
