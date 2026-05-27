"""
Ajoute Claude Sonnet comme 4e juge V4.3 aux données existantes.
Ne rappelle PAS Mistral ni GPT-4o — réutilise les scores déjà calculés.
Génère un nouveau HTML de comparaison à 4 juges.
"""
import json, sys, re, time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

EXISTING_JSON = "comparaisons_rag/ablations_7q_judge_compare_20260518_100143.json"
DIAG_CACHE    = "comparaisons_rag/diagnostic_retrieval_cache.json"
OUT_DIR       = Path("comparaisons_rag")
VERSIONS      = ["v_vanilla_k10", "v_vanilla_k25", "v_decomp", "v_decomp_raptor"]
CLAUDE_MODEL  = "claude-sonnet-4-6"
TAG           = "v43cl"

with open(EXISTING_JSON, encoding="utf-8") as f:
    data = json.load(f)
with open(DIAG_CACHE, encoding="utf-8") as f:
    cache = json.load(f)

cache_idx = {ver: {(e.get("row") or e.get("excel_row") or e.get("q_idx")): e.get("sources", [])
                   for e in cache.get(ver, [])}
             for ver in VERSIONS}

selected_rows = [e["excel_row"] for e in data[VERSIONS[0]]]
print(f"Lignes : {selected_rows}\n")

# ── Client Anthropic ──────────────────────────────────────────────────────
import anthropic, dotenv, os
dotenv.load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
_client = anthropic.Anthropic(api_key=api_key)

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text

def call_claude(system: str, prompt: str, max_tokens: int = 3000) -> str:
    for attempt in range(4):
        try:
            msg = _client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            if attempt < 3 and ("529" in str(e) or "overloaded" in str(e).lower()
                                 or "rate" in str(e).lower()):
                wait = 2 ** attempt * 5
                print(f"    [RATE LIMIT] Attente {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise

def score_judge_v43_claude(question, answer, sources, section, subsection, expected_type):
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter.\n"
        "Réponds UNIQUEMENT avec le JSON demandé, sans texte avant ni après."
    )
    try:
        raw = call_claude(_JUDGE_V43_SYSTEM, user_prompt)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["error"] = None
        return result
    except Exception as e:
        return {"error": str(e), "score_global": None}

# ── Jugement Claude ───────────────────────────────────────────────────────
for ver in VERSIONS:
    print(f"\n{'='*55}")
    print(f"Config: {ver}  |  Judge: Claude Sonnet ({CLAUDE_MODEL})")
    print(f"{'='*55}")
    for entry in data[ver]:
        row      = entry["excel_row"]
        sources  = cache_idx[ver].get(row, [])
        question = entry.get("question", "")
        section  = entry.get("section", "")
        subsec   = entry.get("subsection", "") or ""
        answer   = entry.get("answer", "") or ""
        do_ref   = entry.get("do_refusal", False)
        do_rob   = entry.get("do_robust", False)
        expected = ("refus_attendu" if do_ref
                    else "limite_architecturale" if do_rob
                    else "reponse_substantielle_attendue")

        print(f"  R{row:3} ...", end=" ", flush=True)
        t0  = time.time()
        res = score_judge_v43_claude(question, answer, sources, section, subsec, expected)
        elapsed = round(time.time() - t0, 1)

        for k, v in res.items():
            entry[f"{TAG}_{k}"] = v
        entry[f"{TAG}_elapsed_s"] = elapsed

        sg   = res.get("score_global")
        sg41 = entry.get("v41_score_global")
        d    = round(sg - sg41, 2) if sg is not None and sg41 is not None else "N/A"
        print(f"V4.1={sg41}  Claude={sg}  Δ={d}  ({elapsed}s)")

# ── Sauvegarde JSON ───────────────────────────────────────────────────────
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
json_out = OUT_DIR / f"ablations_7q_judge_compare_4juges_{ts}.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"\nJSON → {json_out}")

# ── HTML ──────────────────────────────────────────────────────────────────
_COLORS = {
    "v_vanilla_k10":   "#c0392b",
    "v_vanilla_k25":   "#e67e22",
    "v_decomp":        "#27ae60",
    "v_decomp_raptor": "#2980b9",
}
_JUDGES = [
    ("v43_gpt4o", "V4.3-GPT4o",        "#8e44ad"),
    ("v43ml",     "V4.3-Mistral-Large", "#2471a3"),
    ("v43ms",     "V4.3-Mistral-Small", "#1a8a6e"),
    (TAG,         "V4.3-Claude-Sonnet", "#b7770d"),
]
_MISLAB_LABELS = {
    "regle_1_quali_quanti":               "R1",
    "regle_2_surinterpretation_oppchovec": "R2",
    "regle_3_absence_quali_non_signalee":  "R3",
    "regle_4_extrapolation_sous_groupe":   "R4",
}

def esc(s): return str(s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def stars(n):
    try: return "★"*int(n) + "☆"*(5-int(n))
    except: return "—"
def fmt(v):
    if v is None: return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)
def delta_html(d):
    if d is None: return "<span style='color:#aaa'>—</span>"
    col = "#27ae60" if d >= 0 else "#c0392b"
    return f"<span style='color:{col};font-weight:bold'>{d:+.2f}</span>"

idx = {ver: {e["excel_row"]: e for e in data[ver]} for ver in VERSIONS}

def mean_v(ver, prefix):
    key = f"{prefix}_score_global"
    vals = [e[key] for e in data[ver] if e.get(key) is not None]
    return round(sum(vals)/len(vals), 2) if vals else None

html_out = OUT_DIR / f"ablations_7q_judge_compare_4juges_{ts}.html"
H = []
H.append(f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Comparaison 4 juges V4.3</title>
<style>
  body{{font-family:Arial,sans-serif;font-size:13px;margin:20px;background:#f4f6f8;color:#222;}}
  h1{{color:#1a1a2e;font-size:17px;margin-bottom:4px;}}
  .subtitle{{color:#888;font-size:12px;margin-bottom:16px;}}
  .recap{{background:white;border-radius:8px;padding:14px 16px;margin-bottom:20px;
          box-shadow:0 1px 3px rgba(0,0,0,.08);}}
  .recap h2{{font-size:13px;color:#1a1a2e;margin:0 0 10px;}}
  table{{border-collapse:collapse;font-size:12px;}}
  th{{background:#1a1a2e;color:white;padding:6px 10px;text-align:center;white-space:nowrap;}}
  th:first-child{{text-align:left;}}
  td{{padding:5px 9px;border-bottom:1px solid #eee;text-align:center;}}
  td:first-child{{text-align:left;}}
  tr:last-child td{{border-bottom:none;}}
  .toc{{background:white;padding:10px 16px;border-radius:8px;margin-bottom:20px;
        box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:12px;}}
  .toc a{{color:#2980b9;text-decoration:none;margin:2px 8px 2px 0;display:inline-block;}}
  .q-block{{background:white;border-radius:8px;padding:16px;margin-bottom:28px;
            box-shadow:0 1px 4px rgba(0,0,0,.1);}}
  .q-header{{font-weight:bold;font-size:14px;color:#1a1a2e;margin-bottom:4px;}}
  .q-meta{{font-size:11px;color:#888;margin-bottom:14px;}}
  .tag{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;
        font-weight:bold;text-transform:uppercase;color:white;}}
  .jtag{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:bold;color:white;}}
  .cmp-table{{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:11px;}}
  .cmp-table th{{padding:5px 7px;}}
  .cmp-table td{{padding:4px 7px;}}
  .group-sep{{border-left:2px solid #ddd;}}
  .detail-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:12px;}}
  .detail-card{{border:1px solid #ddd;border-radius:6px;padding:12px;background:#fafafa;}}
  .card-top{{display:flex;justify-content:space-between;align-items:center;
             padding-bottom:6px;border-bottom:1px solid #eee;margin-bottom:8px;}}
  .judge-blocks{{display:flex;flex-direction:column;gap:6px;}}
  .judge-block{{border-left:3px solid #ddd;padding:5px 8px;border-radius:0 4px 4px 0;background:white;}}
  .jb-title{{font-size:10px;font-weight:bold;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;}}
  .scores-row{{display:flex;gap:4px;flex-wrap:wrap;}}
  .score-box{{background:#f0f0f0;border-radius:4px;padding:3px 6px;text-align:center;min-width:48px;}}
  .score-lbl{{font-size:8px;color:#888;display:block;}}
  .score-val{{font-size:12px;font-weight:bold;}}
  .score-stars{{font-size:8px;color:#f39c12;}}
  .mislab-row{{font-size:9px;margin-top:3px;}}
  .mislab-oui{{color:#c0392b;font-weight:bold;}}
  .mislab-non{{color:#bbb;}}
  .raisonnement{{font-style:italic;color:#555;font-size:10px;padding:4px 7px;
                 background:#fffde7;border-left:2px solid #f1c40f;border-radius:3px;margin-top:4px;}}
  .answer{{font-size:11px;line-height:1.5;white-space:pre-wrap;max-height:260px;
           overflow-y:auto;color:#333;border-top:1px solid #eee;padding-top:8px;margin-top:8px;}}
  .meta-bar{{font-size:10px;color:#bbb;text-align:right;margin-top:4px;}}
  .delta-p{{color:#27ae60;font-weight:bold;}} .delta-m{{color:#c0392b;font-weight:bold;}}
  .best-cell{{font-weight:bold;color:#27ae60;}}
  .error{{color:#c0392b;font-style:italic;font-size:11px;}}
</style></head><body>
<h1>Comparaison 4 juges V4.3 — 7 questions × 4 configs</h1>
<p class="subtitle">{ts} &nbsp;|&nbsp; baseline = V4.1 &nbsp;|&nbsp; modèle Claude : {CLAUDE_MODEL}</p>
""")

# Recap global
H.append('<div class="recap"><h2>Scores moyens par config et par juge</h2>')
H.append('<table><tr><th>Config</th><th>V4.1</th>')
for _, jlabel, jcol in _JUDGES:
    H.append(f'<th><span class="jtag" style="background:{jcol}">{esc(jlabel)}</span></th>')
H.append('</tr>\n')
for ver in VERSIONS:
    c   = _COLORS.get(ver, "#555")
    m41 = mean_v(ver, "v41")
    H.append(f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td>')
    H.append(f'<td><b>{fmt(m41)}</b></td>')
    for pfx, _, jcol in _JUDGES:
        mv = mean_v(ver, pfx)
        d  = round(mv - m41, 2) if mv is not None and m41 is not None else None
        cls = "delta-m" if (d is not None and d < -0.1) else ("delta-p" if (d is not None and d > 0.1) else "")
        H.append(f'<td><b>{fmt(mv)}</b> <span class="{cls}" style="font-size:10px">'
                 f'({fmt(d) if d is not None else "?"})</span></td>')
    H.append('</tr>\n')
H.append('</table></div>\n')

# TOC
H.append('<div class="toc"><b>Navigation</b> : ')
for row in selected_rows:
    e0 = idx[VERSIONS[0]].get(row, {})
    H.append(f'<a href="#r{row}">R{row} {esc((e0.get("section","") or "")[:18])}</a> ')
H.append('</div>\n')

# Par question
for row in selected_rows:
    e0 = idx[VERSIONS[0]].get(row, {})
    H.append(f'<div class="q-block" id="r{row}">\n')
    H.append(f'<div class="q-header">R{row} — {esc(e0.get("question",""))}</div>\n')
    H.append(f'<div class="q-meta">{esc(e0.get("section",""))} › {esc(e0.get("subsection","") or "")}</div>\n')

    # Tableau comparatif compact
    H.append('<table class="cmp-table"><tr><th>Config</th><th>V4.1</th>')
    for _, jlabel, jcol in _JUDGES:
        H.append(f'<th class="group-sep"><span class="jtag" style="background:{jcol}">{esc(jlabel)}</span></th>'
                 f'<th>Δ</th>')
    H.append('</tr>\n')
    for ver in VERSIONS:
        e   = idx[ver].get(row, {})
        c   = _COLORS.get(ver, "#555")
        sg41 = e.get("v41_score_global")
        H.append(f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td>')
        H.append(f'<td><b>{fmt(sg41)}</b></td>')
        for pfx, _, jcol in _JUDGES:
            sg  = e.get(f"{pfx}_score_global")
            d   = round(sg - sg41, 2) if sg is not None and sg41 is not None else None
            dcls = "delta-m" if (d is not None and d < -0.1) else ("delta-p" if (d is not None and d > 0.1) else "")
            H.append(f'<td class="group-sep"><b>{fmt(sg)}</b></td>')
            H.append(f'<td class="{dcls}">{fmt(d) if d is not None else "—"}</td>')
        H.append('</tr>\n')
    H.append('</table>\n')

    # Cartes détail (une par config RAG)
    H.append('<div class="detail-grid">\n')
    for ver in VERSIONS:
        e  = idx[ver].get(row, {})
        c  = _COLORS.get(ver, "#555")
        H.append('<div class="detail-card">\n')
        H.append(f'<div class="card-top">'
                 f'<span class="tag" style="background:{c}">{esc(ver)}</span>'
                 f'<span style="font-size:10px;color:#888">{e.get("n_sources","?")} src</span></div>\n')
        if e.get("rag_status") != "ok":
            H.append(f'<div class="error">Erreur RAG</div>\n')
        else:
            H.append('<div class="judge-blocks">\n')
            sg41 = e.get("v41_score_global")
            for pfx, jlabel, jcol in _JUDGES:
                sg = e.get(f"{pfx}_score_global")
                d  = round(sg - sg41, 2) if sg is not None and sg41 is not None else None
                H.append(f'<div class="judge-block" style="border-left-color:{jcol}">\n')
                H.append(f'<div class="jb-title" style="color:{jcol}">'
                         f'{esc(jlabel)}'
                         f'<span style="background:{jcol};color:white;padding:1px 6px;'
                         f'border-radius:3px;font-size:11px;margin-left:4px">{fmt(sg)}</span>'
                         f'&nbsp;{delta_html(d)}</div>\n')
                H.append('<div class="scores-row">\n')
                for dk, lbl in [("pertinence","Pert"),("fondement_factuel","Fact"),
                                 ("nuance_incertitude","Nua"),("coherence_qualiquanti","Q/Q")]:
                    v = e.get(f"{pfx}_{dk}")
                    H.append(f'<div class="score-box"><span class="score-lbl">{lbl}</span>'
                             f'<span class="score-val">{v if v is not None else "—"}</span>'
                             f'<span class="score-stars">{stars(v)}</span></div>\n')
                H.append('</div>\n')
                # Mislabelling
                ml    = e.get(f"{pfx}_mislabelling_detecte", {}) or {}
                flags = []
                for k, lbl in _MISLAB_LABELS.items():
                    val = (ml.get(k, "non") or "non")
                    cls = "mislab-oui" if val.startswith("oui") else "mislab-non"
                    flags.append(f'<span class="{cls}">{lbl}{"✗" if val.startswith("oui") else "✓"}</span>')
                H.append(f'<div class="mislab-row">{" ".join(flags)}</div>\n')
                if e.get(f"{pfx}_raisonnement"):
                    H.append(f'<div class="raisonnement">{esc(e[f"{pfx}_raisonnement"])}</div>\n')
                H.append('</div>\n')  # judge-block
            H.append('</div>\n')  # judge-blocks
            H.append(f'<div class="answer">{esc(e.get("answer",""))}</div>\n')
        H.append('</div>\n')  # detail-card
    H.append('</div>\n</div>\n\n')

H.append('</body></html>\n')

with open(html_out, "w", encoding="utf-8") as f:
    f.write("".join(H))
print(f"HTML → {html_out}")
