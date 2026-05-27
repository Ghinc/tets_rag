"""Run une seule question sur les 4 configs + judge + export HTML."""
import json, re, sys, time, requests
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_from_excel import _call_llm, _build_sources_text, _JUDGE_V2_SYSTEM

QUESTION = "Le faible nombre de répondants dans certaines communes affecte-t-il la fiabilité des conclusions ?"
SECTION  = "Limites et robustesse"
EXCEL_ROW = 89

BASE     = "http://localhost:8000/api/query"
HEADERS  = {"Content-Type": "application/json"}
VERSIONS = ["v_vanilla_k10", "v_vanilla_k25", "v_decomp", "v_decomp_raptor"]
OUT_DIR  = "comparaisons_rag"

_JUDGE_PROMPT = (
    "QUESTION : {question}\n\nSECTION : {section}\n\n"
    "SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
    "RÉPONSE DU SYSTÈME :\n{answer}\n\n"
    "Évalue cette réponse. Suis la procédure en 5 étapes, résume-la en 1-2 phrases "
    "dans 'raisonnement', puis attribue les 4 notes. Pas de justification par dimension.\n\n"
    "Format JSON strict :\n"
    "{{\n"
    "  \"raisonnement\": \"<1-2 phrases, max 200 chars>\",\n"
    "  \"pertinence\": 1-5,\n  \"fondement_factuel\": 1-5,\n"
    "  \"nuance_incertitude\": 1-5,\n  \"coherence_qualiquanti\": 1-5,\n"
    "  \"applicable_sujet\": true|false,\n  \"note_sujet\": null|1-5,\n"
    "  \"sujet_evalue\": \"libellé court ou null\",\n"
    "  \"reason_non_applicable\": \"methodologique|refus|comparative|factuelle_brute|null\"\n"
    "}}"
)

results = []
for ver in VERSIONS:
    k = 10 if ver == "v_vanilla_k10" else (25 if ver == "v_vanilla_k25" else 5)
    print(f"\n[{ver}] RAG...", flush=True)
    t0 = time.time()
    r_resp = requests.post(BASE, json={"question": QUESTION, "rag_version": ver, "k": k},
                           headers=HEADERS, timeout=300)
    rag_t = time.time() - t0
    data = r_resp.json()
    answer = data.get("answer", "")
    raw_sources = data.get("sources", [])
    sources_for_judge = [
        {"content": s.get("content") or s.get("extrait") or "",
         "metadata": s.get("metadata", {}),
         "label": s.get("label", "")}
        for s in raw_sources
    ]
    print(f"  {len(raw_sources)} sources, {rag_t:.1f}s", flush=True)

    time.sleep(1.5)
    print(f"[{ver}] Judge...", flush=True)
    t0 = time.time()
    sources_text = _build_sources_text(sources_for_judge)
    prompt = _JUDGE_PROMPT.format(question=QUESTION, section=SECTION,
                                  sources_text=sources_text, answer=answer[:4000])
    raw = _call_llm(_JUDGE_V2_SYSTEM, prompt, max_tokens=600, json_mode=True)
    judge_t = time.time() - t0
    m = re.search(r'\{[\s\S]*\}', raw)
    j = json.loads(m.group()) if m else {}
    print(f"  P={j.get('pertinence')} F={j.get('fondement_factuel')} "
          f"N={j.get('nuance_incertitude')} C={j.get('coherence_qualiquanti')} | {judge_t:.1f}s", flush=True)

    results.append({"version": ver, "answer": answer, "n_sources": len(raw_sources),
                    "rag_elapsed_s": round(rag_t,1), "judge_elapsed_s": round(judge_t,1),
                    "pertinence": j.get("pertinence"), "fondement_factuel": j.get("fondement_factuel"),
                    "nuance_incertitude": j.get("nuance_incertitude"),
                    "coherence_qualiquanti": j.get("coherence_qualiquanti"),
                    "raisonnement": j.get("raisonnement"),
                    "score_global": round(sum(filter(None,[j.get("pertinence"),j.get("fondement_factuel"),
                                             j.get("nuance_incertitude"),j.get("coherence_qualiquanti")]))/4,2)
                                   if all(j.get(x) for x in ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti"]) else None})

# --- HTML ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
html_path = Path(OUT_DIR) / f"R89_detail_{ts}.html"

def esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def stars(n):
    if n is None: return "—"
    return "★" * n + "☆" * (5-n)

html = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>R89 — Détail ablations</title>
<style>
  body{{font-family:Arial,sans-serif;font-size:13px;margin:24px;background:#f4f6f8;}}
  h1{{color:#1a1a2e;font-size:18px;}} .q{{background:#1a1a2e;color:white;padding:14px 18px;border-radius:8px;margin-bottom:20px;font-size:15px;}}
  .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;}}
  .card{{background:white;border-radius:8px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.1);}}
  .card-title{{font-weight:bold;font-size:12px;text-transform:uppercase;letter-spacing:.5px;padding-bottom:8px;border-bottom:2px solid;margin-bottom:10px;}}
  .v_vanilla_k10 .card-title{{color:#c0392b;border-color:#c0392b;}}
  .v_vanilla_k25 .card-title{{color:#e67e22;border-color:#e67e22;}}
  .v_decomp .card-title{{color:#27ae60;border-color:#27ae60;}}
  .v_decomp_raptor .card-title{{color:#2980b9;border-color:#2980b9;}}
  .scores{{display:flex;gap:12px;margin-bottom:10px;flex-wrap:wrap;}}
  .score-box{{background:#f8f9fa;border-radius:6px;padding:6px 10px;text-align:center;min-width:70px;}}
  .score-label{{font-size:10px;color:#888;display:block;}}
  .score-val{{font-size:18px;font-weight:bold;color:#333;}}
  .score-stars{{font-size:11px;color:#f39c12;}}
  .global-score{{font-size:22px;font-weight:bold;padding:8px 14px;border-radius:6px;background:#1a1a2e;color:white;}}
  .raisonnement{{font-style:italic;color:#555;font-size:12px;margin-bottom:10px;padding:8px;background:#fffde7;border-left:3px solid #f39c12;border-radius:4px;}}
  .answer{{font-size:12px;line-height:1.6;white-space:pre-wrap;color:#222;max-height:500px;overflow-y:auto;}}
  .meta{{font-size:10px;color:#999;margin-top:8px;}}
  .summary-table{{width:100%;border-collapse:collapse;margin-bottom:20px;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1);}}
  .summary-table th{{background:#1a1a2e;color:white;padding:10px 14px;text-align:left;font-size:12px;}}
  .summary-table td{{padding:10px 14px;border-bottom:1px solid #eee;font-size:12px;}}
  .summary-table tr:last-child td{{border-bottom:none;}}
  .best{{font-weight:bold;color:#27ae60;}}
</style></head><body>
<div class="q">R{EXCEL_ROW} — {esc(QUESTION)}</div>

<table class="summary-table">
<tr><th>Config</th><th>Pertinence</th><th>Factuel</th><th>Nuance</th><th>Quali/Q</th><th>Global</th><th>Sources</th><th>Temps</th></tr>
"""
best_global = max((r["score_global"] or 0) for r in results)
for r in results:
    sg = r["score_global"]
    cls = " class='best'" if sg == best_global else ""
    html += f"<tr><td><b>{esc(r['version'])}</b></td>"
    for dim in ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti"]:
        v = r.get(dim)
        html += f"<td>{v}/5 {stars(v)}</td>" if v else "<td>—</td>"
    sg_str = f"{sg:.2f}" if sg else "—"
    html += f"<td{cls}><b>{sg_str}</b></td>"
    html += f"<td>{r['n_sources']}</td><td>{r['rag_elapsed_s']}s</td></tr>\n"

html += "</table>\n<div class='grid'>\n"

for r in results:
    ver = r["version"]
    sg = r["score_global"]
    html += f'<div class="card {ver}">\n'
    sg_disp = f"{sg:.2f}" if sg else "?"
    html += f'<div class="card-title">{esc(ver)} &nbsp; <span class="global-score">{sg_disp}</span></div>\n'
    html += '<div class="scores">\n'
    for dim, label in [("pertinence","Pertinence"),("fondement_factuel","Factuel"),
                       ("nuance_incertitude","Nuance"),("coherence_qualiquanti","Quali/Q")]:
        v = r.get(dim)
        html += f'<div class="score-box"><span class="score-label">{label}</span><span class="score-val">{v if v else "?"}</span><span class="score-stars">{stars(v)}</span></div>\n'
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
