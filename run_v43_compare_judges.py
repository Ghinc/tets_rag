"""
Compare 3 variantes du judge V4.3 :
  - v43_gpt4o      : résultats déjà dans le JSON (GPT-4o)
  - v43_mistral_large : mistral-large-latest  (nouveau)
  - v43_mistral_small : mistral-small-latest  (nouveau)

Lance uniquement les 7 questions du dry-run × 4 configs.
Génère un JSON + HTML de comparaison.
"""
import json, sys, time, importlib
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

EXISTING_JSON = "comparaisons_rag/ablations_7q_v43_dryrun_20260517_222601.json"
DIAG_CACHE    = "comparaisons_rag/diagnostic_retrieval_cache.json"
OUT_DIR       = Path("comparaisons_rag")
VERSIONS      = ["v_vanilla_k10", "v_vanilla_k25", "v_decomp", "v_decomp_raptor"]

with open(EXISTING_JSON, encoding="utf-8") as f:
    base = json.load(f)
with open(DIAG_CACHE, encoding="utf-8") as f:
    cache = json.load(f)

cache_idx = {ver: {(e.get("row") or e.get("excel_row") or e.get("q_idx")): e.get("sources", [])
                   for e in cache.get(ver, [])}
             for ver in VERSIONS}

# rows sélectionnés (ordre conservé depuis le dry-run)
selected_rows = [e["excel_row"] for e in base[VERSIONS[0]]]
print(f"Lignes : {selected_rows}\n")

# ── Wrapper pour changer de modèle à la volée ──────────────────────────────
import eval_from_excel as evmod

def run_judge_with_model(model_name, api_key_env, base_url, results_dict, tag):
    """Lance score_judge_v43 pour tous les (ver, row), stocke sous préfixe tag."""
    evmod.JUDGE_MODEL     = model_name
    evmod.JUDGE_API_KEY_ENV = api_key_env
    evmod.JUDGE_BASE_URL  = base_url
    # Force recréation du client
    evmod._openai_client  = None

    for ver in VERSIONS:
        print(f"\n{'='*55}")
        print(f"Config: {ver}  |  Judge: {tag}")
        print(f"{'='*55}")
        for entry in results_dict[ver]:
            row = entry["excel_row"]
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
            v43 = evmod.score_judge_v43(question, answer, sources, section, subsec, expected)
            elapsed = round(time.time() - t0, 1)

            for k, v in v43.items():
                entry[f"{tag}_{k}"] = v
            entry[f"{tag}_elapsed_s"] = elapsed

            sg = v43.get("score_global")
            sg41 = entry.get("v41_score_global")
            d = round(sg - sg41, 2) if sg is not None and sg41 is not None else "N/A"
            print(f"V4.1={sg41}  {tag}={sg}  Δ={d}  ({elapsed}s)")

# ── Run mistral-large ──────────────────────────────────────────────────────
run_judge_with_model(
    model_name   = "mistral-large-latest",
    api_key_env  = "MISTRAL_API_KEY",
    base_url     = "https://api.mistral.ai/v1",
    results_dict = base,
    tag          = "v43ml",
)

# ── Run mistral-small ──────────────────────────────────────────────────────
run_judge_with_model(
    model_name   = "mistral-small-latest",
    api_key_env  = "MISTRAL_API_KEY",
    base_url     = "https://api.mistral.ai/v1",
    results_dict = base,
    tag          = "v43ms",
)

# ── Sauvegarde JSON ────────────────────────────────────────────────────────
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
json_out = OUT_DIR / f"ablations_7q_judge_compare_{ts}.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(base, f, ensure_ascii=False, indent=2)
print(f"\nJSON → {json_out}")

# ── Génération HTML ────────────────────────────────────────────────────────
_COLORS = {
    "v_vanilla_k10":   "#c0392b",
    "v_vanilla_k25":   "#e67e22",
    "v_decomp":        "#27ae60",
    "v_decomp_raptor": "#2980b9",
}
_JUDGES = [
    ("v43_gpt4o",  "V4.3-GPT4o",       "#8e44ad"),
    ("v43ml",      "V4.3-Mistral-Large","#2471a3"),
    ("v43ms",      "V4.3-Mistral-Small","#1a8a6e"),
]
_MISLAB_LABELS = {
    "regle_1_quali_quanti":               "R1",
    "regle_2_surinterpretation_oppchovec": "R2",
    "regle_3_absence_quali_non_signalee":  "R3",
    "regle_4_extrapolation_sous_groupe":   "R4",
}

def esc(s): return str(s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def stars(n):
    if n is None: return "—"
    try: return "★"*int(n) + "☆"*(5-int(n))
    except: return "?"
def fmt(v):
    if v is None: return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)
def delta_html(d):
    if d is None: return "<span style='color:#aaa'>—</span>"
    col = "#27ae60" if d >= 0 else "#c0392b"
    return f"<span style='color:{col};font-weight:bold'>{d:+.2f}</span>"

idx = {ver: {e["excel_row"]: e for e in base[ver]} for ver in VERSIONS}

def mean_v(ver, prefix):
    key = f"{prefix}_score_global"
    vals = [e[key] for e in base[ver] if e.get(key) is not None]
    return round(sum(vals)/len(vals), 2) if vals else None

html_out = OUT_DIR / f"ablations_7q_judge_compare_{ts}.html"
H = []
H.append(f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Comparaison judges V4.3 — GPT4o vs Mistral-Large vs Mistral-Small</title>
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
  td{{padding:5px 10px;border-bottom:1px solid #eee;text-align:center;}}
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
  .jtag{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;
         font-weight:bold;color:white;}}
  /* Tableau comparaison 3 juges */
  .cmp-table{{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:11.5px;}}
  .cmp-table th{{padding:6px 8px;}}
  .cmp-table td{{padding:5px 8px;}}
  .group-sep{{border-left:2px solid #ccc;}}
  .best-cell{{font-weight:bold;color:#27ae60;}}
  .worst-cell{{color:#c0392b;}}
  /* Cartes */
  .detail-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:12px;}}
  .detail-card{{border:1px solid #ddd;border-radius:6px;padding:12px;background:#fafafa;}}
  .card-top{{display:flex;justify-content:space-between;align-items:center;
             padding-bottom:6px;border-bottom:1px solid #eee;margin-bottom:8px;}}
  .global-badge{{font-size:13px;font-weight:bold;padding:2px 8px;border-radius:4px;color:white;background:#555;}}
  .judge-row{{display:flex;gap:8px;margin-bottom:6px;flex-wrap:wrap;align-items:center;}}
  .judge-block{{border:1px solid #e0e0e0;border-radius:5px;padding:6px 8px;flex:1;min-width:160px;}}
  .jb-title{{font-size:10px;font-weight:bold;margin-bottom:4px;}}
  .scores-row{{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px;}}
  .score-box{{background:#f0f0f0;border-radius:4px;padding:3px 6px;text-align:center;min-width:50px;}}
  .score-lbl{{font-size:8px;color:#888;display:block;}}
  .score-val{{font-size:12px;font-weight:bold;}}
  .score-stars{{font-size:8px;color:#f39c12;}}
  .raisonnement{{font-style:italic;color:#555;font-size:10px;padding:4px 8px;
                 background:#fffde7;border-left:2px solid #f1c40f;border-radius:3px;margin-top:4px;}}
  .mislab-row{{font-size:9.5px;margin-top:4px;}}
  .mislab-oui{{color:#c0392b;font-weight:bold;}}
  .mislab-non{{color:#aaa;}}
  .answer{{font-size:11px;line-height:1.5;white-space:pre-wrap;max-height:300px;
           overflow-y:auto;color:#333;border-top:1px solid #eee;padding-top:8px;margin-top:8px;}}
  .meta-bar{{font-size:10px;color:#bbb;text-align:right;margin-top:4px;}}
  .error{{color:#c0392b;font-style:italic;}}
  .delta-p{{color:#27ae60;font-weight:bold;}} .delta-m{{color:#c0392b;font-weight:bold;}}
</style></head><body>
<h1>Comparaison judges V4.3 : GPT-4o vs Mistral-Large vs Mistral-Small</h1>
<p class="subtitle">{ts} &nbsp;|&nbsp; 7 questions × 4 configs &nbsp;|&nbsp; baseline = V4.1</p>
""")

# ── Recap global ──────────────────────────────────────────────────────────
H.append('<div class="recap"><h2>Scores moyens par config et par juge</h2>')
H.append('<table><tr><th>Config</th><th>V4.1</th>')
for _, jlabel, jcol in _JUDGES:
    H.append(f'<th><span class="jtag" style="background:{jcol}">{esc(jlabel)}</span></th>')
H.append('<th>Δ GPT4o</th><th>Δ ML</th><th>Δ MS</th></tr>\n')

for ver in VERSIONS:
    c    = _COLORS.get(ver, "#555")
    m41  = mean_v(ver, "v41")
    vals = {pfx: mean_v(ver, pfx) for pfx, _, _ in _JUDGES}
    H.append(f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td>')
    H.append(f'<td><b>{fmt(m41)}</b></td>')
    for pfx, _, jcol in _JUDGES:
        H.append(f'<td><b>{fmt(vals[pfx])}</b></td>')
    for pfx, _, _ in _JUDGES:
        d = round(vals[pfx] - m41, 2) if vals[pfx] is not None and m41 is not None else None
        cls = "delta-m" if (d is not None and d < 0) else ("delta-p" if (d is not None and d > 0) else "")
        H.append(f'<td class="{cls}"><b>{fmt(d)}</b></td>')
    H.append('</tr>\n')
H.append('</table></div>\n')

# ── TOC ───────────────────────────────────────────────────────────────────
H.append('<div class="toc"><b>Navigation</b> : ')
for row in selected_rows:
    e0 = idx[VERSIONS[0]].get(row, {})
    H.append(f'<a href="#r{row}">R{row} {esc((e0.get("section","") or "")[:20])}</a> ')
H.append('</div>\n')

# ── Par question ──────────────────────────────────────────────────────────
for row in selected_rows:
    e0  = idx[VERSIONS[0]].get(row, {})
    H.append(f'<div class="q-block" id="r{row}">\n')
    H.append(f'<div class="q-header">R{row} — {esc(e0.get("question",""))}</div>\n')
    H.append(f'<div class="q-meta">{esc(e0.get("section",""))} › {esc(e0.get("subsection","") or "")}</div>\n')

    # Tableau compact : 1 ligne par config, colonnes = 4 juges × 4 dims + global
    H.append('<table class="cmp-table"><tr><th>Config</th>')
    H.append('<th>V4.1</th>')
    for _, jlabel, jcol in _JUDGES:
        H.append(f'<th class="group-sep"><span class="jtag" style="background:{jcol}">{esc(jlabel)}</span> Pert/Fact/Nua/QQ</th>'
                 f'<th>Global</th><th>Δ</th>')
    H.append('<th>Mislabelling</th></tr>\n')

    for ver in VERSIONS:
        e  = idx[ver].get(row, {})
        c  = _COLORS.get(ver, "#555")
        if e.get("rag_status") != "ok":
            H.append(f'<tr><td colspan="13" class="error">{esc(ver)} — Erreur RAG</td></tr>\n')
            continue
        sg41 = e.get("v41_score_global")
        H.append(f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td>')
        H.append(f'<td><b>{fmt(sg41)}</b></td>')
        for pfx, _, jcol in _JUDGES:
            sg = e.get(f"{pfx}_score_global")
            d  = round(sg - sg41, 2) if sg is not None and sg41 is not None else None
            dims_str = "/".join(
                str(e.get(f"{pfx}_{k}") or "—")
                for k in ("pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti")
            )
            dcls = "delta-m" if (d is not None and d < 0) else ("delta-p" if (d is not None and d > 0) else "")
            H.append(f'<td class="group-sep">{dims_str}</td>')
            H.append(f'<td><b>{fmt(sg)}</b></td>')
            H.append(f'<td class="{dcls}">{fmt(d) if d is not None else "—"}</td>')
        # Mislabelling synthèse (on prend v43ml comme référence Mistral)
        ml = e.get("v43ml_mislabelling_detecte", {}) or {}
        flags = [lbl for k, lbl in _MISLAB_LABELS.items()
                 if (ml.get(k, "non") or "non").startswith("oui")]
        ml_str = ", ".join(flags) if flags else "—"
        ml_cls = "mislab-oui" if flags else "mislab-non"
        H.append(f'<td class="{ml_cls}" style="font-size:10px">{esc(ml_str)}</td>')
        H.append('</tr>\n')
    H.append('</table>\n')

    # Cartes détail (une par config RAG)
    H.append('<div class="detail-grid">\n')
    for ver in VERSIONS:
        e   = idx[ver].get(row, {})
        c   = _COLORS.get(ver, "#555")
        H.append('<div class="detail-card">\n')
        H.append(f'<div class="card-top"><span class="tag" style="background:{c}">{esc(ver)}</span>'
                 f'<span style="font-size:10px;color:#888">{e.get("n_sources","?")} src | RAG {e.get("rag_elapsed_s","?")}s</span></div>\n')
        if e.get("rag_status") != "ok":
            H.append(f'<div class="error">Erreur : {esc(e.get("rag_error","?"))}</div>\n')
        else:
            # Un bloc par juge
            H.append('<div class="judge-row">\n')
            for pfx, jlabel, jcol in _JUDGES:
                sg  = e.get(f"{pfx}_score_global")
                sg41= e.get("v41_score_global")
                d   = round(sg - sg41, 2) if sg is not None and sg41 is not None else None
                H.append(f'<div class="judge-block">\n')
                H.append(f'<div class="jb-title" style="color:{jcol}">{esc(jlabel)}'
                         f' <span style="background:{jcol};color:white;padding:1px 6px;'
                         f'border-radius:3px;font-size:11px">{fmt(sg)}</span>'
                         f' {delta_html(d)}</div>\n')
                H.append('<div class="scores-row">\n')
                for dk, lbl in [("pertinence","Pert"),("fondement_factuel","Fact"),
                                 ("nuance_incertitude","Nua"),("coherence_qualiquanti","Q/Q")]:
                    v = e.get(f"{pfx}_{dk}")
                    H.append(f'<div class="score-box"><span class="score-lbl">{lbl}</span>'
                             f'<span class="score-val">{v if v is not None else "—"}</span>'
                             f'<span class="score-stars">{stars(v)}</span></div>\n')
                H.append('</div>\n')
                # Mislabelling flags
                ml = e.get(f"{pfx}_mislabelling_detecte", {}) or {}
                flags = []
                for k, lbl in _MISLAB_LABELS.items():
                    val = (ml.get(k, "non") or "non")
                    if val.startswith("oui"):
                        flags.append(f'<span class="mislab-oui">{lbl}✗</span>')
                    else:
                        flags.append(f'<span class="mislab-non">{lbl}✓</span>')
                H.append(f'<div class="mislab-row">{" ".join(flags)}</div>\n')
                if e.get(f"{pfx}_raisonnement"):
                    H.append(f'<div class="raisonnement">{esc(e[f"{pfx}_raisonnement"])}</div>\n')
                H.append('</div>\n')
            H.append('</div>\n')  # judge-row
            H.append(f'<div class="answer">{esc(e.get("answer",""))}</div>\n')
        H.append('</div>\n')  # detail-card
    H.append('</div>\n</div>\n\n')  # detail-grid + q-block

H.append('</body></html>\n')

with open(html_out, "w", encoding="utf-8") as f:
    f.write("".join(H))
print(f"HTML → {html_out}")
