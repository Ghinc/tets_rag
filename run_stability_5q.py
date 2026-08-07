"""
run_stability_5q.py
====================
Mesure de stabilité : v_decomp (avec typage) × 3 runs sur les 5 questions
où v_decomp_no_typing montre le plus de variance (|NT_v1 - NT_v2|).

Questions cibles (classées par volatilité NT) :
  Q035 var=2.25 | Q011 var=0.75 | Q025 var=0.50 | Q009 var=0.50 | Q002 var=0.50

Pour chaque question : 3 runs indépendants de v_decomp → V4.3 juge → std/range des scores.

Usage :
  python run_stability_5q.py           # run + HTML
  python run_stability_5q.py --report  # HTML depuis résultats existants
"""
import argparse, json, re, sys, time, datetime, statistics
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")

from rag_ablations import DecompOnlyRAG, DecompNoTypingRAG

import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None
from eval_from_excel import _call_llm, _build_sources_text, _JUDGE_V43_SYSTEM, _parse_judge_v43

# ── Constantes ────────────────────────────────────────────────────────────────
COMPLET     = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
NT2_JSON    = Path("comparaisons_rag/notyping_v2_12q/notyping_v2_results.json")
OUT_DIR     = Path("comparaisons_rag/stability_5q")
RESULTS     = OUT_DIR / "stability_results.json"

# 5 questions les plus volatiles (|NT_v1 - NT_v2|)
TARGET_ROWS = [35, 11, 25, 9, 2]
N_RUNS      = 3
K           = 5
N_SUBQ      = 5

ORIENTATIONS = {2: "mixte", 9: "subjectif", 11: "subjectif", 25: "mixte", 35: "mixte"}
NT_VARIANCE  = {35: 2.25, 11: 0.75, 25: 0.50, 9: 0.50, 2: 0.50}

# ── Juge V4.3 ─────────────────────────────────────────────────────────────────

def judge_v43(question, answer, sources, section, subsection, expected_type):
    src_text = _build_sources_text(sources)
    prompt = (
        f"QUESTION : {question}\n\nSECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{src_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:4000]}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter."
    )
    try:
        raw = _call_llm(_JUDGE_V43_SYSTEM, prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        r = _parse_judge_v43(j)
        r["error"] = None
        return r
    except Exception as e:
        return {"error": str(e), "score_global": None}


def normalize_sources(sources):
    out = []
    for s in sources:
        meta = s.get("meta") or s.get("metadata") or {}
        out.append({
            "content":  s.get("text") or s.get("content") or "",
            "metadata": {**meta, "source_type": s.get("collection", "")},
            "label":    s.get("label") or s.get("collection") or "",
        })
    return out


# ── Chargement données de référence ──────────────────────────────────────────

def load_reference():
    with open(COMPLET, encoding="utf-8") as f:
        data = json.load(f)
    with open(NT2_JSON, encoding="utf-8") as f:
        nt2 = json.load(f)

    dec = {e["excel_row"]: e for e in data["v_decomp"] if e["excel_row"] in TARGET_ROWS}
    nt1 = {e["excel_row"]: e for e in data["v_decomp_no_typing"] if e["excel_row"] in TARGET_ROWS}
    nt2_d = {int(k[1:]): v for k, v in nt2.items() if int(k[1:]) in TARGET_ROWS}
    return dec, nt1, nt2_d


# ── Persistance ───────────────────────────────────────────────────────────────

def load_results():
    if RESULTS.exists():
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(r):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)


def run_key(row, run_i):
    return f"Q{row:03d}_run{run_i}"


def is_complete(entry):
    return isinstance(entry.get("score_global"), (int, float))


# ── Run ───────────────────────────────────────────────────────────────────────

def run(force=False):
    dec_ref, _, _ = load_reference()
    results = load_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initialisation DecompOnlyRAG...")
    rag = DecompOnlyRAG()
    rag.init()
    print("Pipeline prêt.\n")

    total = len(TARGET_ROWS) * N_RUNS
    done = 0
    for row in TARGET_ROWS:
        meta = dec_ref.get(row, {})
        question     = meta.get("question", f"Q{row}")
        section      = meta.get("section", "")
        subsection   = meta.get("subsection", "")
        exp_type     = meta.get("type_reponse_attendue_specifie",
                                "reponse_substantielle_attendue")

        for run_i in range(1, N_RUNS + 1):
            key = run_key(row, run_i)
            existing = results.get(key, {})

            if not force and is_complete(existing):
                done += 1
                print(f"  Q{row:03d} run{run_i} déjà ok → {existing['score_global']}")
                continue

            print(f"  Q{row:03d} run{run_i}/{N_RUNS} RAG (v_decomp)...", end=" ", flush=True)
            t0 = time.time()
            try:
                answer, sources_raw, _, sub_qa, _ = rag.query(
                    question, k=K, n_subquestions=N_SUBQ
                )
                rag_t = round(time.time() - t0, 1)
                print(f"ok ({rag_t}s)", end=" ", flush=True)
            except Exception as e:
                print(f"ERREUR RAG: {e}")
                results[key] = {"rag_status": "error", "error": str(e),
                                "excel_row": row, "run": run_i}
                save_results(results)
                done += 1
                continue

            # Sauvegarde intermédiaire RAG
            results[key] = {
                "excel_row": row, "run": run_i,
                "question": question, "section": section, "subsection": subsection,
                "rag_status": "ok", "answer": answer,
                "sub_questions": sub_qa,
                "sources_raw": [{"content": s.get("text") or s.get("content",""),
                                  "metadata": s.get("meta") or s.get("metadata") or {},
                                  "collection": s.get("collection",""),
                                  "label": s.get("label","")}
                                 for s in sources_raw],
                "n_sources": len(sources_raw),
                "rag_elapsed_s": rag_t,
            }
            save_results(results)

            # Juge
            print("→ juge...", end=" ", flush=True)
            t1 = time.time()
            j = judge_v43(question, answer, normalize_sources(sources_raw),
                          section, subsection, exp_type)
            jt = round(time.time() - t1, 1)
            sg = j.get("score_global")
            print(f"score={sg} ({jt}s)")

            results[key].update({
                "score_global":          sg,
                "pertinence":            j.get("pertinence"),
                "fondement_factuel":     j.get("fondement_factuel"),
                "nuance_incertitude":    j.get("nuance_incertitude"),
                "coherence_qualiquanti": j.get("coherence_qualiquanti"),
                "pertinence_justif":          j.get("pertinence_justif"),
                "fondement_factuel_justif":   j.get("fondement_factuel_justif"),
                "nuance_incertitude_justif":  j.get("nuance_incertitude_justif"),
                "coherence_qualiquanti_justif": j.get("coherence_qualiquanti_justif"),
                "raisonnement":          j.get("raisonnement"),
                "judge_error":           j.get("error"),
                "judge_elapsed_s":       jt,
                "timestamp":             datetime.datetime.now().isoformat(),
            })
            save_results(results)
            done += 1

    print(f"\n{done}/{total} terminés → {RESULTS}")
    return results


# ── HTML ──────────────────────────────────────────────────────────────────────

def _avg(lst):
    lst = [x for x in lst if isinstance(x, (int, float))]
    return round(sum(lst) / len(lst), 3) if lst else None

def _std(lst):
    lst = [x for x in lst if isinstance(x, (int, float))]
    return round(statistics.stdev(lst), 3) if len(lst) >= 2 else None

def _rng(lst):
    lst = [x for x in lst if isinstance(x, (int, float))]
    return round(max(lst) - min(lst), 2) if len(lst) >= 2 else None

def _sc(v):
    if v is None: return '<span class="na">—</span>'
    if isinstance(v, float) and v == int(v): v = int(v)
    hi = v >= 4.5; lo = v < 3
    cls = "shi" if hi else ("slo" if lo else "smid")
    return f'<span class="{cls}">{v}</span>'

def _bar(v, mx=5.0):
    if v is None: return ""
    pct = min(100, round(v / mx * 100))
    return f'<div class="bar-wrap"><div class="bar" style="width:{pct}%"></div><span class="bar-lbl">{v}</span></div>'

def sub_q_list(sub_qa):
    if not sub_qa: return '<em class="muted">—</em>'
    items = []
    for sq in sub_qa:
        idx = sq.get("idx","·")
        q   = sq.get("question","")
        a   = sq.get("answer","").strip()
        a_block = (f'<details class="sq-ans"><summary>Réponse</summary>'
                   f'<div class="sq-ans-body">{a}</div></details>') if a else ""
        items.append(f'<li><span class="sq-i">SQ{idx}</span> {q}{a_block}</li>')
    return f'<ol class="sq-ol">{"".join(items)}</ol>'

def make_html(results, dec_ref, nt1_ref, nt2_ref):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Organise par question
    q_data = {}
    for row in TARGET_ROWS:
        runs = [results.get(run_key(row, i), {}) for i in range(1, N_RUNS+1)]
        scores = [r.get("score_global") for r in runs]
        q_data[row] = {
            "runs":   runs,
            "scores": scores,
            "avg":    _avg(scores),
            "std":    _std(scores),
            "rng":    _rng(scores),
            # références NT
            "nt1":    nt1_ref.get(row, {}).get("score_global"),
            "nt2":    nt2_ref.get(row, {}).get("score_global"),
            "dec_ref": dec_ref.get(row, {}).get("score_global"),  # original single run
            "question": dec_ref.get(row, {}).get("question", f"Q{row}"),
            "orient": ORIENTATIONS.get(row, "?"),
            "nt_var": NT_VARIANCE.get(row, 0),
        }

    # ── Bloc synthèse ──
    def synthesis_block():
        html = '<div class="synth-block">\n'
        html += '<h2>Stabilité v_decomp (3 runs) vs variance no_typing observée</h2>\n'
        html += '<p class="sub">v_decomp exécuté 3 fois indépendamment. '
        html += 'NT_v1 = run original (COMPLET.json). NT_v2 = run avec prompt réécrit. '
        html += 'Plus le range est petit, plus le pipeline est stable.</p>\n'

        html += '<table class="synth-tbl"><thead><tr>'
        html += '<th>Q</th><th>Orient.</th><th>Var. NT<br>(|v1−v2|)</th>'
        html += '<th>v_decomp run1</th><th>run2</th><th>run3</th>'
        html += '<th>Decomp avg</th><th>Decomp std</th><th>Decomp range</th>'
        html += '<th>NT_v1</th><th>NT_v2</th><th>NT range</th>'
        html += '</tr></thead><tbody>\n'

        for row in TARGET_ROWS:
            d = q_data[row]
            sc = d["scores"]
            nt_rng = round(abs((d["nt1"] or 0) - (d["nt2"] or 0)), 2) if d["nt1"] and d["nt2"] else None
            dec_rng_cls = "hi-var" if (d["rng"] or 0) >= 1.0 else ("mid-var" if (d["rng"] or 0) >= 0.5 else "lo-var")
            nt_rng_cls  = "hi-var" if (nt_rng or 0) >= 1.0 else ("mid-var" if (nt_rng or 0) >= 0.5 else "lo-var")
            html += (
                f'<tr>'
                f'<td class="q-num">Q{row:03d}</td>'
                f'<td class="orient-cell orient-{d["orient"]}">{d["orient"][:3].upper()}</td>'
                f'<td class="num {nt_rng_cls}">{d["nt_var"]:.2f}</td>'
            )
            for s in sc:
                html += f'<td class="num">{_sc(s)}</td>'
            d_std = f"{d['std']:.3f}" if d['std'] is not None else "—"
            d_rng = f"{d['rng']:.2f}" if d['rng'] is not None else "—"
            nt_rng_s = f"{nt_rng:.2f}" if nt_rng is not None else "—"
            html += (
                f'<td class="num">{_sc(d["avg"])}</td>'
                f'<td class="num">{d_std}</td>'
                f'<td class="num {dec_rng_cls}">{d_rng}</td>'
                f'<td class="num">{_sc(d["nt1"])}</td>'
                f'<td class="num">{_sc(d["nt2"])}</td>'
                f'<td class="num {nt_rng_cls}">{nt_rng_s}</td>'
                f'</tr>\n'
            )

        # Ligne totaux
        all_dec_std  = _avg([q_data[r]["std"] for r in TARGET_ROWS])
        all_dec_rng  = _avg([q_data[r]["rng"] for r in TARGET_ROWS])
        all_nt_rng   = _avg([abs((q_data[r]["nt1"] or 0)-(q_data[r]["nt2"] or 0))
                              for r in TARGET_ROWS if q_data[r]["nt1"] and q_data[r]["nt2"]])
        moy_avg    = f"{_avg([q_data[r]['avg'] for r in TARGET_ROWS]):.3f}"
        moy_std    = f"{all_dec_std:.3f}" if all_dec_std else "—"
        moy_rng    = f"{all_dec_rng:.2f}" if all_dec_rng else "—"
        moy_nt1    = f"{_avg([q_data[r]['nt1'] for r in TARGET_ROWS]):.3f}"
        moy_nt2    = f"{_avg([q_data[r]['nt2'] for r in TARGET_ROWS]):.3f}"
        moy_ntrng  = f"{all_nt_rng:.2f}" if all_nt_rng else "—"
        html += (
            f'<tr class="total-row">'
            f'<td colspan="3"><strong>MOY</strong></td>'
            f'<td colspan="3"></td>'
            f'<td class="num"><strong>{moy_avg}</strong></td>'
            f'<td class="num"><strong>{moy_std}</strong></td>'
            f'<td class="num"><strong>{moy_rng}</strong></td>'
            f'<td class="num"><strong>{moy_nt1}</strong></td>'
            f'<td class="num"><strong>{moy_nt2}</strong></td>'
            f'<td class="num"><strong>{moy_ntrng}</strong></td>'
            f'</tr>\n'
        )
        html += '</tbody></table>\n</div>\n'
        return html

    # ── Détail par question ──
    def detail_section():
        html = ""
        for row in TARGET_ROWS:
            d = q_data[row]
            html += f'<div class="q-block">\n'
            html += f'<h3>Q{row:03d} — {d["question"]}</h3>\n'
            html += f'<p class="q-meta">Orient : <strong class="orient-{d["orient"]}">{d["orient"]}</strong> · '
            html += f'Var NT observée : <strong>{d["nt_var"]:.2f}</strong> · '
            html += f'Decomp avg={d["avg"]} std={d["std"]} range={d["rng"]}</p>\n'

            # Scores par dimension pour les 3 runs
            dims = [("Pertinence","pertinence"),("Fact. factuel","fondement_factuel"),
                    ("Nuance","nuance_incertitude"),("Coh. Q/Q","coherence_qualiquanti"),
                    ("GLOBAL","score_global")]
            html += '<table class="dim-runs-tbl"><thead><tr><th>Dim.</th>'
            for i in range(1, N_RUNS+1):
                html += f'<th>Run {i}</th>'
            html += '<th>Avg</th><th>Std</th><th>Range</th></tr></thead><tbody>\n'
            for dim_label, dim_key in dims:
                vals = [d["runs"][i].get(dim_key) for i in range(N_RUNS)]
                cls = " class=\"total-dim\"" if dim_label == "GLOBAL" else ""
                html += f'<tr{cls}><td>{dim_label}</td>'
                for v in vals:
                    html += f'<td class="num">{_sc(v)}</td>'
                v_std = f"{_std(vals):.3f}" if _std(vals) is not None else "—"
                v_rng = f"{_rng(vals):.2f}" if _rng(vals) is not None else "—"
                html += (f'<td class="num">{_sc(_avg(vals))}</td>'
                         f'<td class="num">{v_std}</td>'
                         f'<td class="num">{v_rng}</td>'
                         f'</tr>\n')
            html += '</tbody></table>\n'

            # 3 runs détaillés
            for i, run_rec in enumerate(d["runs"], 1):
                sg = run_rec.get("score_global")
                n_sq = len(run_rec.get("sub_questions", []))
                html += (f'<details class="run-detail">'
                         f'<summary>Run {i} — Score {_sc(sg)} — {n_sq} sous-questions</summary>'
                         f'<div class="run-body">\n')

                # Sous-questions
                sub_qa = run_rec.get("sub_questions", [])
                html += f'<h4>Sous-questions générées ({len(sub_qa)})</h4>\n'
                html += sub_q_list(sub_qa)

                # Réponse
                ans = run_rec.get("answer","")
                if ans:
                    html += (f'<details class="ans-d"><summary>Réponse complète</summary>'
                             f'<div class="ans-body">{ans}</div></details>\n')

                # Sources
                srcs = run_rec.get("sources_raw",[])
                if srcs:
                    html += f'<details class="src-d"><summary>Sources ({len(srcs)})</summary><ul class="src-list">\n'
                    for s in srcs:
                        meta = s.get("metadata",{})
                        coll = s.get("collection") or meta.get("source_type","?")
                        com  = meta.get("commune","")
                        lbl  = f"{coll}" + (f" — {com}" if com else "")
                        sq_i = meta.get("sub_question_idx") or s.get("sub_question_idx","")
                        sq_tag = f' <span class="sq-tag">SQ{sq_i}</span>' if sq_i else ""
                        content = (s.get("content") or "")[:400]
                        html += f'<li><strong>{lbl}</strong>{sq_tag}<br><span class="src-txt">{content}</span></li>\n'
                    html += '</ul></details>\n'

                html += '</div></details>\n'
            html += '</div>\n'
        return html

    css = """
<style>
:root {
  --bg:#f8f9fa;--bg2:#fff;--border:#dee2e6;--text:#212529;--muted:#6c757d;
  --acc:#0d6efd;--obj:#0d6efd;--sub:#6f42c1;--mix:#0a9ab5;
  --shi:#198754;--smid:#fd7e14;--slo:#dc3545;
  --lo-var:#198754;--mid-var:#fd7e14;--hi-var:#dc3545;
}
@media(prefers-color-scheme:dark){
  :root{--bg:#121416;--bg2:#1e2124;--border:#2d3035;--text:#e9ecef;--muted:#adb5bd;
    --obj:#74c0fc;--sub:#cc5de8;--mix:#22b8cf;
    --shi:#51cf66;--smid:#ff922b;--slo:#ff6b6b;
    --lo-var:#51cf66;--mid-var:#ff922b;--hi-var:#ff6b6b;}
}
:root[data-theme="light"]{--bg:#f8f9fa;--bg2:#fff;--border:#dee2e6;--text:#212529;--muted:#6c757d;--obj:#0d6efd;--sub:#6f42c1;--mix:#0a9ab5;--shi:#198754;--smid:#fd7e14;--slo:#dc3545;--lo-var:#198754;--mid-var:#fd7e14;--hi-var:#dc3545;}
:root[data-theme="dark"]{--bg:#121416;--bg2:#1e2124;--border:#2d3035;--text:#e9ecef;--muted:#adb5bd;--obj:#74c0fc;--sub:#cc5de8;--mix:#22b8cf;--shi:#51cf66;--smid:#ff922b;--slo:#ff6b6b;--lo-var:#51cf66;--mid-var:#ff922b;--hi-var:#ff6b6b;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:system-ui,sans-serif;font-size:13px;background:var(--bg);color:var(--text);line-height:1.5;padding:24px;max-width:1300px;margin:0 auto;}
h1{font-size:1.5rem;margin-bottom:4px;}
h2{font-size:1.1rem;margin:20px 0 10px;border-bottom:2px solid var(--border);padding-bottom:4px;}
h3{font-size:0.95rem;margin:10px 0 4px;}
h4{font-size:0.85rem;margin:8px 0 4px;color:var(--muted);}
.sub{color:var(--muted);font-size:0.82rem;margin-bottom:12px;}
table{border-collapse:collapse;width:100%;}
th,td{border:1px solid var(--border);padding:4px 7px;text-align:left;font-size:0.78rem;}
th{background:var(--bg);font-weight:600;}
.num{text-align:center;font-variant-numeric:tabular-nums;}
.na{color:var(--muted);}
.shi{color:var(--shi);font-weight:700;}
.smid{color:var(--smid);font-weight:600;}
.slo{color:var(--slo);font-weight:600;}
.lo-var{color:var(--lo-var);}
.mid-var{color:var(--mid-var);}
.hi-var{color:var(--hi-var);font-weight:700;}
.orient-objectif{color:var(--obj);}.orient-subjectif{color:var(--sub);}.orient-mixte{color:var(--mix);}
.orient-cell{text-align:center;font-size:0.72rem;}
.q-num{font-weight:700;text-align:center;}
.total-row{font-weight:700;background:var(--bg2);}
.total-dim{font-weight:700;}
.synth-block{background:var(--bg2);border:2px solid var(--acc);border-radius:8px;padding:18px;margin-bottom:24px;}
.synth-tbl{font-size:0.78rem;}
.q-block{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:14px;margin-bottom:14px;}
.q-meta{font-size:0.8rem;color:var(--muted);margin-bottom:10px;}
.dim-runs-tbl{margin-bottom:10px;font-size:0.78rem;}
.run-detail{margin-bottom:8px;}
.run-detail summary{cursor:pointer;padding:5px 8px;background:var(--bg);border-radius:4px;color:var(--acc);}
.run-body{padding:8px 0;}
.sq-ol{margin:4px 0 8px 16px;}
.sq-ol li{margin-bottom:6px;}
.sq-i{font-weight:700;color:var(--acc);font-size:0.72rem;}
.sq-ans summary{cursor:pointer;font-size:0.75rem;color:var(--muted);}
.sq-ans-body{font-size:0.78rem;white-space:pre-wrap;background:var(--bg);padding:6px;border-radius:3px;margin-top:3px;}
.ans-d summary,.src-d summary{cursor:pointer;font-size:0.78rem;color:var(--muted);}
.ans-body{font-size:0.8rem;white-space:pre-wrap;background:var(--bg);padding:8px;border-radius:4px;margin-top:4px;}
.src-list{list-style:none;margin:6px 0;}
.src-list li{border-bottom:1px solid var(--border);padding:4px 0;font-size:0.75rem;}
.src-txt{color:var(--muted);font-size:0.72rem;}
.sq-tag{background:var(--acc);color:#fff;font-size:0.62rem;padding:1px 3px;border-radius:2px;margin-left:4px;}
details summary{user-select:none;}
.bar-wrap{display:flex;align-items:center;gap:6px;}
.bar{height:8px;background:var(--acc);border-radius:2px;}
.bar-lbl{font-size:0.75rem;}
</style>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8">
<title>Stabilité v_decomp — 5 questions × 3 runs</title>
{css}</head>
<body>
<h1>Stabilité v_decomp (typage) — 3 runs indépendants × 5 questions</h1>
<p class="sub">Questions sélectionnées : les 5 où v_decomp_no_typing montre la plus grande variance entre NT_v1 et NT_v2.
Juge V4.3 (GPT-4o) · Généré le {ts}</p>

{synthesis_block()}

<h2>Détail par question</h2>
{detail_section()}

<p class="sub" style="margin-top:16px">JSON → {RESULTS}</p>
</body></html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = OUT_DIR / f"stability_report_{ts}.html"

    dec_ref, nt1_ref, nt2_ref = load_reference()

    if args.report:
        results = load_results()
    else:
        print(f"Stabilité v_decomp — {len(TARGET_ROWS)} questions × {N_RUNS} runs\n")
        results = run(force=args.force)

    html = make_html(results, dec_ref, nt1_ref, nt2_ref)
    html_path.write_text(html, encoding="utf-8")
    print(f"\nHTML → {html_path}")
    dl = Path.home() / "Downloads" / f"stability_report_{ts}.html"
    dl.write_text(html, encoding="utf-8")
    print(f"     → {dl}")
    print(f"\nRelance : python run_stability_5q.py")


if __name__ == "__main__":
    main()
