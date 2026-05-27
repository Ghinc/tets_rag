"""
Exporte le fichier COMPLET en HTML amélioré :
- Questions groupées par catégorie
- Texte complet des questions (pas de troncature)
- Réponse complète dépliable par config
- Colonnes redimensionnables
"""
import json, sys, io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SRC = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
OUT = Path("comparaisons_rag/ablations_103q_v43_gpt4o_20260518_222912.html")

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

CONFIGS = ["v_vanilla_k10", "v_vanilla_k25", "v_decomp", "v_decomp_raptor"]
LABELS  = {"v_vanilla_k10": "Vanilla k10", "v_vanilla_k25": "Vanilla k25",
           "v_decomp": "Decomp", "v_decomp_raptor": "Decomp+Raptor"}
COLORS  = {"v_vanilla_k10": "#c0392b", "v_vanilla_k25": "#e67e22",
           "v_decomp": "#27ae60", "v_decomp_raptor": "#2980b9"}

# Indexer par (excel_row, config)
index = {}
for cfg, entries in data.items():
    for e in entries:
        index[(e["excel_row"], cfg)] = e

# Toutes les lignes uniques, triées par section puis numéro
all_rows = sorted({e["excel_row"] for e in data["v_vanilla_k10"]})
e0_by_row = {r: data["v_vanilla_k10"][i] for i, r in enumerate(
    sorted(e["excel_row"] for e in data["v_vanilla_k10"]))}

def normalize_sec(s):
    return s.replace("’", "'").replace("‘", "'")

# Grouper par section
by_section = defaultdict(list)
for row in all_rows:
    e = index.get((row, "v_vanilla_k10"), {})
    sec = normalize_sec(e.get("section", "?"))
    by_section[sec].append(row)

# Ordre sections
SEC_ORDER = [
    "Retrieval mono-commune",
    "Raisonnement comparatif",
    "Raisonnement causal et contre-intuitif",
    "Gestion d'absence d'information",
    "Gestion de l'absence d'information",
    "Gestion de l'incertitude et des biais",
    "Robustesse sémantique",
    "Limites architecturales",
]
# Fusionner les deux variantes "gestion absence"
merged_sections = {}
for sec, rows in by_section.items():
    key = "Gestion de l'absence d'information" if "absence" in sec.lower() and "information" in sec.lower() else sec
    merged_sections.setdefault(key, [])
    merged_sections[key].extend(rows)

ordered_keys = [k for k in SEC_ORDER if k in merged_sections]
for k in merged_sections:
    if k not in ordered_keys:
        ordered_keys.append(k)

# ── Helpers ──────────────────────────────────────────────────────────────────
def sc_color(v):
    if v is None: return "#95a5a6"
    return "#27ae60" if v >= 4 else ("#e67e22" if v >= 3 else "#c0392b")

def score_cell(v):
    if v is None: return "<td>—</td>"
    c = sc_color(v)
    return f'<td style="color:{c};font-weight:bold;text-align:center">{v:.2f}</td>'

def mis_flags(e):
    mis = e.get("mislabelling_detecte", {})
    if not mis or not isinstance(mis, dict): return ""
    short = {"regle_1_quali_quanti":"R1","regle_2_surinterpretation_oppchovec":"R2",
             "regle_3_absence_quali_non_signalee":"R3","regle_4_extrapolation_sous_groupe":"R4"}
    flags = [short.get(k,"?") for k, v in mis.items()
             if isinstance(v, str) and v.strip().lower().startswith("oui")]
    if not flags: return ""
    return " ".join(f'<span style="background:#c0392b;color:white;border-radius:3px;padding:1px 5px;font-size:0.8em">{f}</span>' for f in flags)

def avg(entries, key):
    vals = [e[key] for e in entries if isinstance(e.get(key), (int, float))]
    return f"{sum(vals)/len(vals):.2f}" if vals else "—"

# ── Résumé global ─────────────────────────────────────────────────────────────
def summary_table():
    dims = ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti","score_global"]
    dlabels = ["Pertinence","Factuel","Nuance","Quali/Q","Global"]
    rows = f'<tr><th>Config</th><th>N OK</th>{"".join(f"<th>{l}</th>" for l in dlabels)}</tr>\n'
    for cfg in CONFIGS:
        entries = [e for e in data[cfg] if not e.get("judge_error") and e.get("rag_status") == "ok"]
        n = len(data[cfg])
        c = COLORS[cfg]
        cells = "".join(f"<td>{avg(entries, d)}</td>" for d in dims)
        rows += f'<tr><td style="color:{c};font-weight:bold">{LABELS[cfg]}</td><td>{len(entries)}/{n}</td>{cells}</tr>\n'
    return f'<table class="summary">{rows}</table>'

# ── Corps principal ───────────────────────────────────────────────────────────
body_parts = []

for sec_key in ordered_keys:
    rows_in_sec = sorted(set(merged_sections[sec_key]))
    n_q = len(rows_in_sec)

    section_rows = ""
    for i, row in enumerate(rows_in_sec):
        e_ref = index.get((row, "v_vanilla_k10"), {})
        q_text = e_ref.get("question", f"Q{row}")
        subsec = normalize_sec(e_ref.get("subsection", "") or "")
        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"

        # Ligne question (span 4 configs + toutes colonnes)
        section_rows += f'''
<tr style="background:{bg}">
  <td colspan="10" style="padding:6px 10px;font-weight:600;font-size:0.95em;border-top:2px solid #ddd">
    <span style="background:#34495e;color:white;border-radius:3px;padding:1px 6px;font-size:0.85em;margin-right:8px">Q{row}</span>
    {f'<span style="color:#888;font-size:0.85em">{subsec} — </span>' if subsec else ""}
    {q_text}
  </td>
</tr>'''

        for cfg in CONFIGS:
            e = index.get((row, cfg), {})
            if not e:
                section_rows += f'<tr style="background:{bg}"><td style="color:{COLORS[cfg]}">{LABELS[cfg]}</td><td colspan="9" style="color:#aaa">—</td></tr>'
                continue
            c = COLORS[cfg]
            sg = e.get("score_global")
            rai = (e.get("raisonnement") or "")[:150]
            answer = (e.get("answer") or "").replace("<","&lt;").replace(">","&gt;")
            n_src = e.get("n_sources", 0)
            uid = f"r{row}_{cfg}"

            section_rows += f'''
<tr style="background:{bg}">
  <td style="color:{c};font-weight:bold;white-space:nowrap;padding:4px 8px">{LABELS[cfg]}</td>
  {score_cell(e.get("pertinence"))}
  {score_cell(e.get("fondement_factuel"))}
  {score_cell(e.get("nuance_incertitude"))}
  {score_cell(e.get("coherence_qualiquanti"))}
  {score_cell(sg)}
  <td style="text-align:center">{mis_flags(e)}</td>
  <td style="font-size:0.82em;color:#555;max-width:320px">{rai}{"…" if len(e.get("raisonnement") or "") > 150 else ""}</td>
  <td style="font-size:0.82em;color:#888;text-align:center">{n_src}</td>
  <td style="padding:2px 6px">
    <details>
      <summary style="cursor:pointer;color:#2980b9;font-size:0.8em;user-select:none">Réponse ({len(e.get("answer",""))} car.)</summary>
      <div style="white-space:pre-wrap;font-family:monospace;font-size:0.78em;background:#f5f5f5;border:1px solid #ddd;border-radius:4px;padding:8px;margin-top:4px;max-height:350px;overflow-y:auto">{answer}</div>
    </details>
  </td>
</tr>'''

    body_parts.append(f'''
<div class="section-block" id="sec-{sec_key.replace(" ","_").replace("'","").replace("/","")}">
  <h3 class="sec-header">{sec_key} <span class="sec-count">({n_q} questions)</span></h3>
  <table class="detail-table">
    <thead>
      <tr>
        <th style="min-width:100px">Config</th>
        <th>Pert.</th><th>Fact.</th><th>Nuance</th><th>Q/Q</th>
        <th>Global</th><th>Flags</th>
        <th style="min-width:260px">Raisonnement (extrait)</th>
        <th>Src</th><th style="min-width:120px">Réponse</th>
      </tr>
    </thead>
    <tbody>
      {section_rows}
    </tbody>
  </table>
</div>''')

# ── Navigation ────────────────────────────────────────────────────────────────
nav_links = " · ".join(
    f'<a href="#sec-{k.replace(" ","_").replace(chr(39),"").replace("/","")}">{k}</a>'
    for k in ordered_keys
)

# ── HTML final ────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Ablations RAG — 103q — V4.3 GPT-4o</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; margin: 0; background: #f0f2f5; color: #222; }}
  .header {{ background: #2c3e50; color: white; padding: 16px 24px; position: sticky;
             top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.3); }}
  .header h1 {{ margin: 0 0 6px; font-size: 1.15em; }}
  .nav {{ font-size: 0.8em; opacity: .8; }}
  .nav a {{ color: #aed6f1; text-decoration: none; }}
  .nav a:hover {{ text-decoration: underline; }}
  .container {{ max-width: 1600px; margin: 0 auto; padding: 20px 24px; }}
  .summary {{ border-collapse: collapse; margin-bottom: 24px; background: white;
              border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow: hidden; }}
  .summary th, .summary td {{ border: 1px solid #e0e0e0; padding: 6px 12px; }}
  .summary th {{ background: #ecf0f1; font-size: 0.85em; }}
  .section-block {{ background: white; border-radius: 8px; margin-bottom: 24px;
                    box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow: hidden; }}
  .sec-header {{ margin: 0; padding: 12px 18px; background: #34495e; color: white;
                 font-size: 1em; display: flex; align-items: center; gap: 8px; }}
  .sec-count {{ font-weight: normal; font-size: 0.85em; opacity: .75; }}
  .detail-table {{ width: 100%; border-collapse: collapse; table-layout: auto; }}
  .detail-table thead th {{ background: #ecf0f1; padding: 6px 8px; border: 1px solid #ddd;
                             font-size: 0.82em; position: sticky; top: 52px; z-index: 10;
                             resize: horizontal; overflow: hidden; white-space: nowrap; }}
  .detail-table td {{ border: 1px solid #eee; padding: 4px 7px; vertical-align: top; }}
  details summary {{ list-style: none; }}
  details summary::-webkit-details-marker {{ display: none; }}
</style>
</head>
<body>
<div class="header">
  <h1>Ablations RAG — 103 questions — Judge V4.3 GPT-4o</h1>
  <div class="nav">Aller à : {nav_links}</div>
</div>
<div class="container">
  <h3 style="margin-top:0">Résumé par config</h3>
  {summary_table()}
  {"".join(body_parts)}
</div>
</body>
</html>"""

OUT.write_text(html, encoding="utf-8")
print(f"HTML → {OUT}  ({OUT.stat().st_size // 1024} Ko)")
