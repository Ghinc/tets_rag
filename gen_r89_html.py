"""Génère le HTML R89 à partir des résultats déjà connus."""
from datetime import datetime
from pathlib import Path

QUESTION  = "Le faible nombre de répondants dans certaines communes affecte-t-il la fiabilité des conclusions ?"
EXCEL_ROW = 89
OUT_DIR   = "comparaisons_rag"

# Résultats récupérés du run précédent
results = [
    {"version": "v_vanilla_k10",   "n_sources": 10,  "rag_elapsed_s": 12.9, "judge_elapsed_s": 7.4,
     "pertinence": 3, "fondement_factuel": 4, "nuance_incertitude": 4, "coherence_qualiquanti": 4,
     "score_global": 3.75,
     "raisonnement": None,
     "answer": None},  # on re-fetch juste la réponse
    {"version": "v_vanilla_k25",   "n_sources": 25,  "rag_elapsed_s": 14.0, "judge_elapsed_s": 2.2,
     "pertinence": 5, "fondement_factuel": 5, "nuance_incertitude": 5, "coherence_qualiquanti": 4,
     "score_global": 4.75,
     "raisonnement": None,
     "answer": None},
    {"version": "v_decomp",        "n_sources": 25,  "rag_elapsed_s": 56.7, "judge_elapsed_s": 2.1,
     "pertinence": 5, "fondement_factuel": 5, "nuance_incertitude": 5, "coherence_qualiquanti": 5,
     "score_global": 5.00,
     "raisonnement": None,
     "answer": None},
    {"version": "v_decomp_raptor", "n_sources": 54,  "rag_elapsed_s": 77.1, "judge_elapsed_s": 3.8,
     "pertinence": 5, "fondement_factuel": 4, "nuance_incertitude": 4, "coherence_qualiquanti": 5,
     "score_global": 4.50,
     "raisonnement": None,
     "answer": None},
]

import sys, time, requests
sys.path.insert(0, str(Path(__file__).parent))
BASE    = "http://localhost:8000/api/query"
HEADERS = {"Content-Type": "application/json"}

print("Récupération des réponses...", flush=True)
for r in results:
    ver = r["version"]
    k = 10 if ver == "v_vanilla_k10" else (25 if ver == "v_vanilla_k25" else 5)
    resp = requests.post(BASE, json={"question": QUESTION, "rag_version": ver, "k": k},
                         headers=HEADERS, timeout=300)
    r["answer"] = resp.json().get("answer", "") if resp.status_code == 200 else "ERREUR"
    print(f"  {ver}: OK", flush=True)
    time.sleep(0.5)

def esc(s): return str(s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def stars(n):
    if n is None: return "—"
    return "★" * n + "☆" * (5-n)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
html_path = Path(OUT_DIR) / f"R89_detail_{ts}.html"

html = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>R89 — Détail ablations</title>
<style>
  body{{font-family:Arial,sans-serif;font-size:13px;margin:24px;background:#f4f6f8;}}
  h1{{color:#1a1a2e;font-size:18px;}}
  .q{{background:#1a1a2e;color:white;padding:14px 18px;border-radius:8px;margin-bottom:20px;font-size:15px;}}
  .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:20px;}}
  .card{{background:white;border-radius:8px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.1);}}
  .card-header{{display:flex;align-items:center;justify-content:space-between;padding-bottom:10px;border-bottom:2px solid;margin-bottom:12px;}}
  .card-title{{font-weight:bold;font-size:12px;text-transform:uppercase;letter-spacing:.5px;}}
  .v_vanilla_k10 .card-header{{border-color:#c0392b;}} .v_vanilla_k10 .card-title{{color:#c0392b;}}
  .v_vanilla_k25 .card-header{{border-color:#e67e22;}} .v_vanilla_k25 .card-title{{color:#e67e22;}}
  .v_decomp      .card-header{{border-color:#27ae60;}} .v_decomp .card-title{{color:#27ae60;}}
  .v_decomp_raptor .card-header{{border-color:#2980b9;}} .v_decomp_raptor .card-title{{color:#2980b9;}}
  .global-score{{font-size:20px;font-weight:bold;background:#1a1a2e;color:white;padding:4px 12px;border-radius:6px;}}
  .scores{{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;}}
  .score-box{{background:#f8f9fa;border-radius:6px;padding:6px 10px;text-align:center;min-width:68px;}}
  .score-label{{font-size:10px;color:#888;display:block;}}
  .score-val{{font-size:17px;font-weight:bold;color:#333;}}
  .score-stars{{font-size:10px;color:#f39c12;}}
  .raisonnement{{font-style:italic;color:#555;font-size:11px;margin-bottom:10px;padding:8px;
                 background:#fffde7;border-left:3px solid #f39c12;border-radius:4px;}}
  .answer{{font-size:12px;line-height:1.6;white-space:pre-wrap;color:#222;
           max-height:520px;overflow-y:auto;border-top:1px solid #eee;padding-top:10px;}}
  .meta{{font-size:10px;color:#aaa;margin-top:8px;text-align:right;}}
  table{{width:100%;border-collapse:collapse;background:white;border-radius:8px;
         overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1);}}
  th{{background:#1a1a2e;color:white;padding:10px 14px;text-align:center;font-size:12px;}}
  th:first-child{{text-align:left;}}
  td{{padding:9px 14px;border-bottom:1px solid #eee;text-align:center;font-size:12px;}}
  td:first-child{{text-align:left;font-weight:bold;}}
  tr:last-child td{{border-bottom:none;}}
  .best{{font-weight:bold;color:#27ae60;font-size:14px;}}
</style></head><body>
<div class="q">R{EXCEL_ROW} — {esc(QUESTION)}</div>

<table>
<tr><th>Config</th><th>Pertinence</th><th>Factuel</th><th>Nuance</th><th>Quali/Q</th>
    <th>Global</th><th>Sources</th><th>Temps RAG</th></tr>
"""

best_sg = max(r["score_global"] or 0 for r in results)
for r in results:
    sg = r["score_global"]
    sg_str = f"{sg:.2f}" if sg else "—"
    cls = " class='best'" if sg == best_sg else ""
    html += f"<tr><td>{esc(r['version'])}</td>"
    for d in ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti"]:
        v = r.get(d); html += f"<td>{v}/5 {stars(v)}</td>" if v else "<td>—</td>"
    html += f"<td{cls}><b>{sg_str}</b></td><td>{r['n_sources']}</td><td>{r['rag_elapsed_s']}s</td></tr>\n"

html += "</table>\n<div class='grid'>\n"
for r in results:
    ver = r["version"]; sg = r["score_global"]
    sg_disp = f"{sg:.2f}" if sg else "?"
    html += f'<div class="card {ver}">\n<div class="card-header">'
    html += f'<span class="card-title">{esc(ver)}</span>'
    html += f'<span class="global-score">{sg_disp} / 5</span></div>\n'
    html += '<div class="scores">\n'
    for d, lbl in [("pertinence","Pertinence"),("fondement_factuel","Factuel"),
                   ("nuance_incertitude","Nuance"),("coherence_qualiquanti","Quali/Q")]:
        v = r.get(d)
        html += (f'<div class="score-box"><span class="score-label">{lbl}</span>'
                 f'<span class="score-val">{v if v else "?"}</span>'
                 f'<span class="score-stars">{stars(v)}</span></div>\n')
    html += '</div>\n'
    if r.get("raisonnement"):
        html += f'<div class="raisonnement">🤖 {esc(r["raisonnement"])}</div>\n'
    html += f'<div class="answer">{esc(r["answer"])}</div>\n'
    html += f'<div class="meta">{r["n_sources"]} sources | RAG {r["rag_elapsed_s"]}s | Judge {r["judge_elapsed_s"]}s</div>\n'
    html += '</div>\n'

html += "</div></body></html>\n"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nHTML : {html_path}", flush=True)
