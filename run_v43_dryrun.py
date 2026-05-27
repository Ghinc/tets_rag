"""
Dry-run Judge V4.3 : 1 question par section, 4 configs.
Génère JSON + HTML consultable avec comparaison V4.1 vs V4.3.
"""
import json, sys, random, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

ABLATION_JSON = "comparaisons_rag/ablations_20q_run_judge_merged_20260516.json"
DIAG_CACHE    = "comparaisons_rag/diagnostic_retrieval_cache.json"
OUT_DIR       = Path("comparaisons_rag")

VERSIONS = ["v_vanilla_k10", "v_vanilla_k25", "v_decomp", "v_decomp_raptor"]
SEED     = 42

# ── Chargement ───────────────────────────────────────────────────────────────
with open(ABLATION_JSON, encoding="utf-8") as f:
    ablation = json.load(f)
with open(DIAG_CACHE, encoding="utf-8") as f:
    cache = json.load(f)

# ── Sélection 1 question par section (seed fixe) ─────────────────────────────
random.seed(SEED)
base_entries = ablation[VERSIONS[0]]
by_section = defaultdict(list)
for e in base_entries:
    by_section[e.get("section", "N/A")].append(e)

selected_rows = []
for sec in sorted(by_section.keys()):
    picked = random.choice(by_section[sec])
    selected_rows.append(picked["excel_row"])
    print(f"  Sélectionné R{picked['excel_row']:3} [{sec[:40]}] {picked['question'][:60]}")

print(f"\n{len(selected_rows)} questions : {selected_rows}\n")

# ── Import juge ───────────────────────────────────────────────────────────────
from eval_from_excel import score_judge_v43

# ── Cache sources par config×row ─────────────────────────────────────────────
cache_idx = {}
for ver in VERSIONS:
    cache_idx[ver] = {}
    for e in cache.get(ver, []):
        row = e.get("row") or e.get("excel_row") or e.get("q_idx")
        cache_idx[ver][row] = e.get("sources", [])

# ── Jugement V4.3 ─────────────────────────────────────────────────────────────
results = {}
for ver in VERSIONS:
    print(f"\n{'='*60}")
    print(f"Config: {ver}")
    print(f"{'='*60}")
    ver_entries = {e["excel_row"]: e for e in ablation[ver]}
    results[ver] = []
    for row in selected_rows:
        ae = ver_entries.get(row, {})
        sources = cache_idx[ver].get(row, [])
        question    = ae.get("question", "")
        section     = ae.get("section", "")
        subsection  = ae.get("subsection", "") or ""
        answer      = ae.get("answer", "") or ""
        do_refusal  = ae.get("do_refusal", False)
        do_robust   = ae.get("do_robust", False)
        expected    = ("refus_attendu" if do_refusal
                       else "limite_architecturale" if do_robust
                       else "reponse_substantielle_attendue")

        t0 = time.time()
        v43 = score_judge_v43(question, answer, sources, section, subsection, expected)
        elapsed = round(time.time() - t0, 1)

        sg43 = v43.get("score_global")
        sg41 = ae.get("v41_score_global")
        delta = round(sg43 - sg41, 2) if sg43 is not None and sg41 is not None else None
        print(f"  R{row:3} | V4.1={sg41}  V4.3={sg43}  Δ={delta:+.2f}" if delta is not None
              else f"  R{row:3} | V4.1={sg41}  V4.3={sg43}  Δ=N/A")

        # Fusionne ae + résultats V4.3
        entry = dict(ae)
        for k, v in v43.items():
            entry[f"v43_{k}"] = v
        entry["judge_v43_elapsed_s"] = elapsed
        results[ver].append(entry)

# ── Sauvegarde JSON ───────────────────────────────────────────────────────────
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
json_out = OUT_DIR / f"ablations_7q_v43_dryrun_{ts}.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nJSON → {json_out}")

# ── Génération HTML ───────────────────────────────────────────────────────────
_COLORS = {
    "v_vanilla_k10":   "#c0392b",
    "v_vanilla_k25":   "#e67e22",
    "v_decomp":        "#27ae60",
    "v_decomp_raptor": "#2980b9",
}
_MISLAB_LABELS = {
    "regle_1_quali_quanti":             "R1 quali/quanti",
    "regle_2_surinterpretation_oppchovec": "R2 surinterpréta-tion",
    "regle_3_absence_quali_non_signalee": "R3 absence quali",
    "regle_4_extrapolation_sous_groupe":  "R4 extrapolation",
}

def esc(s): return str(s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def stars(n):
    if n is None: return "—"
    return "★"*int(n) + "☆"*(5-int(n))
def fmt(v): return f"{v:.2f}" if isinstance(v, float) else (str(v) if v is not None else "—")
def delta_badge(d):
    if d is None: return "<span style='color:#aaa'>—</span>"
    col = "#27ae60" if d >= 0 else "#c0392b"
    return f"<span style='color:{col};font-weight:bold'>{d:+.2f}</span>"

idx = {ver: {e["excel_row"]: e for e in results[ver]} for ver in VERSIONS}

def mean_v(ver, key):
    vals = [e[key] for e in results[ver] if e.get(key) is not None]
    return round(sum(vals)/len(vals), 2) if vals else None

html_out = OUT_DIR / f"ablations_7q_v43_dryrun_{ts}.html"
H = []
H.append(f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Dry-run V4.3 — 7 questions</title>
<style>
  body{{font-family:Arial,sans-serif;font-size:13px;margin:20px;background:#f4f6f8;color:#222;}}
  h1{{color:#1a1a2e;font-size:17px;margin-bottom:4px;}}
  .subtitle{{color:#888;font-size:12px;margin-bottom:16px;}}
  .recap{{background:white;border-radius:8px;padding:14px 16px;margin-bottom:20px;
          box-shadow:0 1px 3px rgba(0,0,0,.08);}}
  .recap h2{{font-size:13px;color:#1a1a2e;margin:0 0 10px;}}
  table{{border-collapse:collapse;font-size:12px;}}
  .recap table{{width:auto;}}
  th{{background:#1a1a2e;color:white;padding:6px 12px;text-align:center;}}
  th:first-child{{text-align:left;}}
  td{{padding:6px 12px;border-bottom:1px solid #eee;text-align:center;}}
  td:first-child{{text-align:left;}}
  tr:last-child td{{border-bottom:none;}}
  .toc{{background:white;padding:10px 16px;border-radius:8px;margin-bottom:20px;
        box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:12px;}}
  .toc a{{color:#2980b9;text-decoration:none;margin:2px 8px 2px 0;display:inline-block;}}
  .q-block{{background:white;border-radius:8px;padding:16px;margin-bottom:28px;
            box-shadow:0 1px 4px rgba(0,0,0,.1);}}
  .q-header{{font-weight:bold;font-size:14px;color:#1a1a2e;margin-bottom:4px;}}
  .q-meta{{font-size:11px;color:#888;margin-bottom:14px;}}
  .summary-table{{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:12px;}}
  .summary-table th{{background:#1a1a2e;color:white;padding:7px 10px;}}
  .best-cell{{font-weight:bold;color:#27ae60;font-size:13px;}}
  .tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;
        font-weight:bold;text-transform:uppercase;letter-spacing:.4px;color:white;}}
  .detail-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;}}
  .detail-card{{border:1px solid #e0e0e0;border-radius:6px;padding:12px;background:#fafafa;}}
  .card-header{{display:flex;align-items:center;justify-content:space-between;
                margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #eee;}}
  .badge{{font-size:13px;font-weight:bold;padding:2px 9px;border-radius:5px;color:white;background:#1a1a2e;}}
  .scores-row{{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;}}
  .score-box{{background:#f0f0f0;border-radius:5px;padding:4px 8px;text-align:center;min-width:60px;}}
  .score-lbl{{font-size:9px;color:#888;display:block;}}
  .score-val{{font-size:14px;font-weight:bold;}}
  .score-stars{{font-size:9px;color:#f39c12;}}
  .v41-section{{margin-top:8px;padding:8px;background:#f0f7ff;border-radius:5px;
                border-left:3px solid #2980b9;}}
  .v43-section{{margin-top:8px;padding:8px;background:#f3fff3;border-radius:5px;
                border-left:3px solid #27ae60;}}
  .judge-title{{font-size:10px;font-weight:bold;margin-bottom:6px;}}
  .judge-badge{{font-size:13px;font-weight:bold;color:white;padding:2px 8px;
                border-radius:4px;float:right;}}
  .v41-title{{color:#2980b9;}} .v41-badge{{background:#2980b9;}}
  .v43-title{{color:#27ae60;}} .v43-badge{{background:#27ae60;}}
  .raisonnement{{font-style:italic;color:#555;font-size:11px;padding:6px 9px;
                 background:#fffde7;border-left:3px solid #f1c40f;border-radius:4px;margin-bottom:6px;}}
  .mislab-table{{font-size:10px;width:100%;margin-top:4px;border-collapse:collapse;}}
  .mislab-table td{{padding:2px 4px;border-bottom:1px solid #e0e0e0;vertical-align:top;}}
  .mislab-oui{{color:#c0392b;font-weight:bold;}}
  .mislab-non{{color:#27ae60;}}
  .et-row{{font-size:10px;color:#555;margin-top:4px;}}
  .et-precis{{color:#27ae60;font-weight:bold;}}
  .et-approx-ok{{color:#e67e22;}}
  .et-approx-nok{{color:#c0392b;font-weight:bold;}}
  .et-omis{{color:#c0392b;}}
  .answer{{font-size:11.5px;line-height:1.55;white-space:pre-wrap;max-height:380px;
           overflow-y:auto;color:#333;border-top:1px solid #eee;padding-top:8px;margin-top:6px;}}
  .meta-bar{{font-size:10px;color:#bbb;text-align:right;margin-top:6px;}}
  .error{{color:#c0392b;font-style:italic;}}
  .delta-plus{{color:#27ae60;font-weight:bold;}}
  .delta-minus{{color:#c0392b;font-weight:bold;}}
</style></head><body>
<h1>Dry-run Judge V4.3 — 7 questions (1 par section)</h1>
<p class="subtitle">Source ablation : {ABLATION_JSON} &nbsp;|&nbsp; Seed={SEED} &nbsp;|&nbsp; {ts}</p>
""")

# Recap global
H.append('<div class="recap"><h2>Scores moyens par config</h2><table>\n')
H.append('<tr><th>Config</th><th>V4.1 (moy.)</th><th>V4.3 (moy.)</th><th>Δ moy.</th></tr>\n')
for ver in VERSIONS:
    c   = _COLORS.get(ver, "#555")
    m41 = mean_v(ver, "v41_score_global")
    m43 = mean_v(ver, "v43_score_global")
    d   = round(m43 - m41, 2) if m41 is not None and m43 is not None else None
    dcls = "delta-minus" if (d is not None and d < 0) else ("delta-plus" if (d is not None and d > 0) else "")
    H.append(f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td>'
             f'<td><b>{fmt(m41)}</b></td><td><b>{fmt(m43)}</b></td>'
             f'<td class="{dcls}"><b>{fmt(d) if d is not None else "—"}</b></td></tr>\n')
H.append('</table></div>\n')

# TOC
H.append('<div class="toc"><b>Navigation</b> : ')
for row in selected_rows:
    e0 = idx[VERSIONS[0]].get(row, {})
    sec_short = (e0.get("section","") or "")[:25]
    H.append(f'<a href="#r{row}">R{row} {esc(sec_short)}</a> ')
H.append('</div>\n')

# Par question
for row in selected_rows:
    e0  = idx[VERSIONS[0]].get(row, {})
    q   = e0.get("question", "")
    sec = e0.get("section", "")
    sub = e0.get("subsection", "") or ""

    H.append(f'<div class="q-block" id="r{row}">\n')
    H.append(f'<div class="q-header">R{row} — {esc(q)}</div>\n')
    H.append(f'<div class="q-meta">{esc(sec)} › {esc(sub)}</div>\n')

    # Tableau récap
    H.append('<table class="summary-table"><tr><th>Config</th>'
             '<th>V4.1 Pert</th><th>V4.1 Fact</th><th>V4.1 Nuance</th><th>V4.1 Q/Q</th>'
             '<th>V4.1 Global</th>'
             '<th>V4.3 Pert</th><th>V4.3 Fact</th><th>V4.3 Nuance</th><th>V4.3 Q/Q</th>'
             '<th>V4.3 Global</th><th>Δ</th><th>Mislabelling</th></tr>\n')

    g43_vals = [idx[v].get(row, {}).get("v43_score_global") for v in VERSIONS]
    best43   = max((x for x in g43_vals if x is not None), default=None)

    for ver in VERSIONS:
        e  = idx[ver].get(row, {})
        c  = _COLORS.get(ver, "#555")
        if e.get("rag_status") != "ok":
            H.append(f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span>'
                     f'</td><td colspan="12" class="error">Erreur RAG</td></tr>\n')
            continue
        def td_dim(key):
            v = e.get(key)
            return f'<td>{v}/5 {stars(v)}</td>' if v is not None else '<td>—</td>'
        sg41 = e.get("v41_score_global")
        sg43 = e.get("v43_score_global")
        d    = round(sg43 - sg41, 2) if sg43 is not None and sg41 is not None else None
        best_cls = ' class="best-cell"' if sg43 and sg43 == best43 else ''
        dcls = "delta-minus" if (d is not None and d < 0) else ("delta-plus" if (d is not None and d > 0) else "")
        # Mislabelling summary
        ml = e.get("v43_mislabelling_detecte", {}) or {}
        flags = [lbl for k, lbl in _MISLAB_LABELS.items() if (ml.get(k, "non") or "non").startswith("oui")]
        ml_str = (", ".join(flags) if flags else "aucun")
        ml_col = "mislab-oui" if flags else "mislab-non"
        H.append(f'<tr><td><span class="tag" style="background:{c}">{esc(ver)}</span></td>')
        for d_key in ("v41_pertinence","v41_fondement_factuel","v41_nuance_incertitude","v41_coherence_qualiquanti"):
            H.append(td_dim(d_key))
        H.append(f'<td><b>{fmt(sg41)}</b></td>')
        for d_key in ("v43_pertinence","v43_fondement_factuel","v43_nuance_incertitude","v43_coherence_qualiquanti"):
            H.append(td_dim(d_key))
        H.append(f'<td{best_cls}><b>{fmt(sg43)}</b></td>')
        H.append(f'<td class="{dcls}"><b>{fmt(d) if d is not None else "—"}</b></td>')
        H.append(f'<td class="{ml_col}" style="font-size:10px">{esc(ml_str)}</td>')
        H.append('</tr>\n')
    H.append('</table>\n')

    # Cartes détail
    H.append('<div class="detail-grid">\n')
    for ver in VERSIONS:
        e  = idx[ver].get(row, {})
        c  = _COLORS.get(ver, "#555")
        sg43 = e.get("v43_score_global")
        sg41 = e.get("v41_score_global")
        d    = round(sg43 - sg41, 2) if sg43 is not None and sg41 is not None else None
        H.append('<div class="detail-card">\n')
        H.append(f'<div class="card-header"><span class="tag" style="background:{c}">{esc(ver)}</span>'
                 f'<span class="badge">{fmt(sg43)} / 5</span></div>\n')

        if e.get("rag_status") != "ok":
            H.append(f'<div class="error">Erreur : {esc(e.get("rag_error","?"))}</div>\n')
        else:
            # Section V4.1
            H.append('<div class="v41-section"><div class="judge-title v41-title">'
                     f'Juge V4.1 <span class="judge-badge v41-badge">{fmt(sg41)}/5</span></div>\n')
            H.append('<div class="scores-row">\n')
            for dk, lbl in [("v41_pertinence","Pert"),("v41_fondement_factuel","Factuel"),
                            ("v41_nuance_incertitude","Nuance"),("v41_coherence_qualiquanti","Q/Q")]:
                v = e.get(dk)
                H.append(f'<div class="score-box"><span class="score-lbl">{lbl}</span>'
                         f'<span class="score-val">{v if v is not None else "?"}</span>'
                         f'<span class="score-stars">{stars(v)}</span></div>\n')
            H.append('</div>\n')
            ets = e.get("v41_elements_traitement", []) or []
            if ets:
                _cls = {"precis":"et-precis","approximation_signalee":"et-approx-ok",
                        "approximation_non_signalee":"et-approx-nok","omis":"et-omis"}
                parts = [f'<span class="{_cls.get(et.get("traitement",""),"")}">'
                         f'{esc(et.get("element",""))}={esc(et.get("traitement",""))}</span>'
                         for et in ets if isinstance(et, dict)]
                H.append(f'<div class="et-row">Éléments : {" · ".join(parts)}</div>\n')
            if e.get("v41_raisonnement"):
                H.append(f'<div class="raisonnement">💬 {esc(e["v41_raisonnement"])}</div>\n')
            H.append('</div>\n')  # end v41-section

            # Section V4.3
            H.append('<div class="v43-section"><div class="judge-title v43-title">'
                     f'Juge V4.3 <span class="judge-badge v43-badge">{fmt(sg43)}/5</span>'
                     f'&nbsp;{delta_badge(d)}</div>\n')
            H.append('<div class="scores-row">\n')
            for dk, lbl in [("v43_pertinence","Pert"),("v43_fondement_factuel","Factuel"),
                            ("v43_nuance_incertitude","Nuance"),("v43_coherence_qualiquanti","Q/Q")]:
                v = e.get(dk)
                H.append(f'<div class="score-box"><span class="score-lbl">{lbl}</span>'
                         f'<span class="score-val">{v if v is not None else "?"}</span>'
                         f'<span class="score-stars">{stars(v)}</span></div>\n')
            H.append('</div>\n')
            # Mislabelling
            ml = e.get("v43_mislabelling_detecte", {}) or {}
            H.append('<table class="mislab-table">\n')
            for k, lbl in _MISLAB_LABELS.items():
                val = (ml.get(k, "non") or "non")
                is_oui = val.startswith("oui")
                cls = "mislab-oui" if is_oui else "mislab-non"
                H.append(f'<tr><td style="width:130px;color:#888">{lbl}</td>'
                         f'<td class="{cls}">{esc(val)}</td></tr>\n')
            H.append('</table>\n')
            # Sources inventaire
            si = e.get("v43_sources_inventaire", []) or []
            if si:
                src_parts = []
                for s in si:
                    t = s.get("type","?")
                    col = "#27ae60" if t == "quali" else "#c0392b"
                    src_parts.append(f'<span style="color:{col}">{esc(s.get("source","?"))}({t})</span>')
                H.append(f'<div style="font-size:10px;margin-top:4px">Sources : {" · ".join(src_parts)}</div>\n')
            if e.get("v43_raisonnement"):
                H.append(f'<div class="raisonnement">💬 {esc(e["v43_raisonnement"])}</div>\n')
            ets43 = e.get("v43_elements_traitement", []) or []
            if ets43:
                _cls = {"precis":"et-precis","approximation_signalee":"et-approx-ok",
                        "approximation_non_signalee":"et-approx-nok","omis":"et-omis"}
                parts = [f'<span class="{_cls.get(et.get("traitement",""),"")}">'
                         f'{esc(et.get("element",""))}={esc(et.get("traitement",""))}</span>'
                         for et in ets43 if isinstance(et, dict)]
                H.append(f'<div class="et-row">Éléments : {" · ".join(parts)}</div>\n')
            H.append('</div>\n')  # end v43-section

            H.append(f'<div class="answer">{esc(e.get("answer",""))}</div>\n')
            H.append(f'<div class="meta-bar">{e.get("n_sources","?")} src | '
                     f'RAG {e.get("rag_elapsed_s","?")}s | '
                     f'V4.3 {e.get("judge_v43_elapsed_s","?")}s</div>\n')
        H.append('</div>\n')  # end detail-card
    H.append('</div>\n</div>\n\n')  # end detail-grid + q-block

H.append('</body></html>\n')

with open(html_out, "w", encoding="utf-8") as f:
    f.write("".join(H))
print(f"HTML → {html_out}")
