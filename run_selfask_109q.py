"""
run_selfask_109q.py — Self-Ask sur 109 questions benchmark, section par section.

Idempotent : relancer reprend à la première question non complète.
Reprise    : python run_selfask_109q.py
Rapport    : généré à la fin → comparaisons_rag/rapport_selfask_109q_{ts}.html
"""
import io, json, sys, time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

# ── Constantes ────────────────────────────────────────────────────────────────

COMPLET   = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
OUT_DIR   = Path("comparaisons_rag/selfask_12q")
DL_DIR    = Path(r"C:\Users\comiti_g\Downloads")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti"]
DIM_LABELS = {"pertinence":"Pertinence","fondement_factuel":"Factuel",
              "nuance_incertitude":"Nuance","coherence_qualiquanti":"Quali/Quanti"}

# ── Ordre des sections du papier ──────────────────────────────────────────────

SECTION_ORDER = [
    "Retrieval mono-commune",
    "Raisonnement comparatif",
    "Raisonnement causal et contre-intuitif",
    "Gestion de l’incertitude et des biais",
    "Gestion de l’absence d’information",
    "Robustesse s\xe9mantique",
    "Limites architecturales",
]

# Label d'affichage pour chaque section
SECTION_LABELS = {
    "Retrieval mono-commune":                        "1 · Retrieval mono-commune",
    "Raisonnement comparatif":                       "2 · Raisonnement comparatif",
    "Raisonnement causal et contre-intuitif":        "3 · Raisonnement causal",
    "Gestion de l’incertitude et des biais":    "4 · Gestion incertitude/biais",
    "Gestion de l’absence d’information":  "5 · Gestion info manquante",
    "Robustesse s\xe9mantique":                      "6 · Robustesse sémantique",
    "Limites architecturales":                       "7 · Limites architecturales",
}

def _norm_section(s: str) -> str:
    """Normalise les variantes d'encodage de noms de section."""
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("\xe9", "é").replace("\xe8", "è").replace("\xe0", "à"))
    # Uniformise les deux variantes de "Gestion d'absence"
    s = s.replace("Gestion d'absence d'information",
                  "Gestion de l'absence d'information")
    return s

# Reconstruire SECTION_ORDER avec les vraies clés normalisées
SECTION_ORDER_NORM = [_norm_section(s) for s in SECTION_ORDER]

# ── Erreurs dures (arrêt propre) ──────────────────────────────────────────────

HARD_ERROR_MARKERS = [
    "401", "authentication", "invalid_api_key", "insufficient_quota",
    "credit", "billing", "account", "disabled", "deactivated",
]

def _is_hard_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in HARD_ERROR_MARKERS)

def _hard_stop(provider: str, row: int, reason: str):
    print(f"\n{'='*70}")
    print(f"ARRÊT PROPRE — Erreur dure {provider}")
    print(f"  Question : Q{row:03d}")
    print(f"  Raison   : {reason}")
    print(f"  Reprise  : python run_selfask_109q.py")
    print(f"{'='*70}")
    sys.exit(2)

# ── Idempotence ───────────────────────────────────────────────────────────────

def _is_complete(row: int) -> bool:
    p = OUT_DIR / f"selfask_q{row:03d}.json"
    if not p.exists():
        return False
    try:
        e = json.loads(p.read_text(encoding="utf-8"))
        return isinstance(e.get("score_global"), (int, float))
    except Exception:
        return False

def _load_existing(row: int) -> dict:
    p = OUT_DIR / f"selfask_q{row:03d}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save(row: int, entry: dict):
    p = OUT_DIR / f"selfask_q{row:03d}.json"
    p.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

# ── Chargement pipeline ───────────────────────────────────────────────────────

def _load_pipeline():
    from rag_selfask import SelfAskRAG
    from eval_from_excel import score_judge_v43
    print("Initialisation SelfAskRAG...")
    pipeline = SelfAskRAG()
    pipeline.init()
    return pipeline, score_judge_v43

# ── Run d'une question ────────────────────────────────────────────────────────

def _run_question(pipeline, score_judge_v43, q: dict, idx: int, total: int) -> dict:
    row          = q["excel_row"]
    question     = q["question"]
    section      = q["section"]
    subsection   = q.get("subsection","")
    expected_type = q.get("type_reponse_attendue_specifie","reponse_substantielle_attendue")

    print(f"\n  [{idx:03d}/{total}] Q{row:03d} [{_norm_section(section)[:30]:<30}] RAG Self-Ask...")

    existing = _load_existing(row)

    # ── Phase 1 : RAG ──────────────────────────────────────────────────────
    if isinstance(existing.get("n_hops"), int) and "final_answer" in existing:
        print(f"           RAG déjà présent — skip RAG, lance juge.")
        final_answer = existing["final_answer"]
        all_sources  = existing.get("all_sources", [])
        hops         = existing.get("hops", [])
        elapsed_rag  = existing.get("elapsed_rag_s", 0)
    else:
        t0 = time.time()
        try:
            final_answer, all_sources, hops = pipeline.query(question)
        except Exception as exc:
            if _is_hard_error(exc):
                _hard_stop("Mistral/Haiku", row, str(exc))
            print(f"           ERREUR RAG (non-dure) : {exc}")
            final_answer, all_sources, hops = f"[ERREUR RAG: {exc}]", [], []
        elapsed_rag = round(time.time() - t0, 1)
        n_hops = len(hops)
        print(f"           ok ({elapsed_rag}s, {n_hops} hops, {len(all_sources)} src)")

        # Sauvegarder phase RAG immédiatement
        entry_rag = {
            "excel_row":     row,
            "question":      question,
            "section":       section,
            "subsection":    subsection,
            "expected_type": expected_type,
            "hops":          hops,
            "n_hops":        n_hops,
            "final_answer":  final_answer,
            "all_sources":   all_sources,
            "elapsed_rag_s": elapsed_rag,
            "meta":          {"max_hops":5,"k":5,"mistral_temp":0.0,
                              "answerer":"haiku","ts":datetime.now().isoformat()},
        }
        # Fusionner avec éventuels champs existants
        for k,v in existing.items():
            if k not in entry_rag:
                entry_rag[k] = v
        _save(row, entry_rag)
        existing = entry_rag

    # ── Phase 2 : Juge ─────────────────────────────────────────────────────
    print(f"           juge V4.3...")
    t1 = time.time()
    try:
        j = score_judge_v43(
            question     = question,
            answer       = final_answer,
            sources      = all_sources,
            section      = _norm_section(section),
            subsection   = subsection,
            expected_type= expected_type,
        )
    except Exception as exc:
        if _is_hard_error(exc):
            _hard_stop("OpenAI/GPT-4o", row, str(exc))
        print(f"           ERREUR juge (non-dure) : {exc}")
        j = {"score_global": None, "error": str(exc)}
    elapsed_judge = round(time.time() - t1, 1)
    sg = j.get("score_global")
    print(f"           score={sg} ({elapsed_judge}s)")

    # Fusionner tout
    entry = dict(existing)
    entry.update(j)
    entry["elapsed_judge_s"] = elapsed_judge
    _save(row, entry)
    return entry

# ── Calcul stats ──────────────────────────────────────────────────────────────

def _avg(vals):
    v = [x for x in vals if isinstance(x,(int,float))]
    return round(sum(v)/len(v),2) if v else None

def _fmt(v, d=2):
    return f"{v:.{d}f}" if isinstance(v,(int,float)) else "—"

def _delta(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)): return round(a-b,2)
    return None

def _sign(v): return "+" if isinstance(v,(int,float)) and v>0 else ""

def _score_cls(v):
    if not isinstance(v,(int,float)): return "na"
    if v>=4.5: return "hi"
    if v>=3.5: return "mid"
    return "lo"

def _delta_cls(d):
    if not isinstance(d,(int,float)): return "na"
    if d>0.25: return "pos"
    if d<-0.25: return "neg"
    return "neu"

def _ml_flag(e):
    if e is None: return False
    fl = e.get("mislabelling_flag")
    if fl is not None: return bool(fl)
    ml = e.get("mislabelling_detecte") or {}
    return any(str(v).lower() not in ("non","false","","null","none") for v in ml.values())

# ── Rapport HTML ──────────────────────────────────────────────────────────────

CSS = """
:root{
  --bg:#f4f5f8;--bg2:#fff;--bg3:#ecedf2;--border:#d0d3de;
  --text:#1a1d2e;--text2:#4a4e6b;--text3:#8891b2;
  --hi:#1e7e44;--hi-bg:#d4edda;--mid:#856404;--mid-bg:#fff3cd;
  --lo:#842029;--lo-bg:#f8d7da;
  --pos:#1e7e44;--pos-bg:#d4edda;--neg:#842029;--neg-bg:#f8d7da;--neu:#4a4e6b;
  --sa:#3a7bd5;--dr:#e07b39;--vk10:#2e7d62;--vk25:#7b52ab;--r:6px;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1f2430;--border:#30363d;
  --text:#c9d1d9;--text2:#8b949e;--text3:#484f58;
  --hi:#3fb950;--hi-bg:#0d2818;--mid:#d29922;--mid-bg:#271e00;
  --lo:#f85149;--lo-bg:#2d0d0c;
  --pos:#3fb950;--pos-bg:#0d2818;--neg:#f85149;--neg-bg:#2d0d0c;--neu:#8b949e;
  --sa:#6ea8fe;--dr:#f4a261;--vk10:#4ade80;--vk25:#c084fc;
}}
:root[data-theme="dark"]{--bg:#0d1117;--bg2:#161b22;--bg3:#1f2430;--border:#30363d;--text:#c9d1d9;--text2:#8b949e;--text3:#484f58;--hi:#3fb950;--hi-bg:#0d2818;--mid:#d29922;--mid-bg:#271e00;--lo:#f85149;--lo-bg:#2d0d0c;--pos:#3fb950;--pos-bg:#0d2818;--neg:#f85149;--neg-bg:#2d0d0c;--neu:#8b949e;--sa:#6ea8fe;--dr:#f4a261;--vk10:#4ade80;--vk25:#c084fc;}
:root[data-theme="light"]{--bg:#f4f5f8;--bg2:#fff;--bg3:#ecedf2;--border:#d0d3de;--text:#1a1d2e;--text2:#4a4e6b;--text3:#8891b2;--hi:#1e7e44;--hi-bg:#d4edda;--mid:#856404;--mid-bg:#fff3cd;--lo:#842029;--lo-bg:#f8d7da;--pos:#1e7e44;--pos-bg:#d4edda;--neg:#842029;--neg-bg:#f8d7da;--neu:#4a4e6b;--sa:#3a7bd5;--dr:#e07b39;--vk10:#2e7d62;--vk25:#7b52ab;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;font-size:14px;background:var(--bg);color:var(--text);line-height:1.55}
.wrap{max-width:1400px;margin:0 auto;padding:1.5rem 1rem}
h1{font-size:1.4rem;font-weight:700;margin-bottom:.3rem}
.meta{font-size:.8rem;color:var(--text2);margin-bottom:2rem}
section{margin-bottom:2.5rem}
h2{font-size:.85rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text2);margin-bottom:.9rem;padding-bottom:.4rem;border-bottom:1px solid var(--border)}
.tscroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:var(--bg3);color:var(--text2);font-weight:600;text-align:left;padding:.4rem .6rem;font-size:.73rem;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
td{padding:.35rem .6rem;border-bottom:1px solid var(--border);vertical-align:middle}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace;white-space:nowrap}
.bold{font-weight:700}
.cfg-sa{color:var(--sa)}.cfg-dr{color:var(--dr)}.cfg-vk10{color:var(--vk10)}.cfg-vk25{color:var(--vk25)}
.sc-hi{color:var(--hi);background:var(--hi-bg);border-radius:3px}
.sc-mid{color:var(--mid);background:var(--mid-bg);border-radius:3px}
.sc-lo{color:var(--lo);background:var(--lo-bg);border-radius:3px}
.sc-na{color:var(--text3)}
.delta-pos{color:var(--pos);font-weight:600}
.delta-neg{color:var(--neg);font-weight:600}
.delta-neu{color:var(--neu)}
.delta-na{color:var(--text3)}
.delta-row td{background:var(--bg3)}
.sec-row td{background:var(--bg3);font-weight:600}
.badge{display:inline-block;padding:.12em .45em;border-radius:3px;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.b1{background:#dbeafe;color:#1d4ed8}.b2{background:#fce7f3;color:#9d174d}
.b3{background:#fef9c3;color:#854d0e}.b4{background:#d1fae5;color:#065f46}
.b5{background:#ede9fe;color:#5b21b6}.b6{background:#ffedd5;color:#9a3412}
.b7{background:#f1f5f9;color:#475569}
@media(prefers-color-scheme:dark){
.b1{background:#1e3a5f;color:#93c5fd}.b2{background:#4a1942;color:#f9a8d4}
.b3{background:#3d2c00;color:#fde047}.b4{background:#064e3b;color:#6ee7b7}
.b5{background:#2e1065;color:#c4b5fd}.b6{background:#431407;color:#fed7aa}
.b7{background:#1e293b;color:#94a3b8}
}
:root[data-theme="dark"] .b1{background:#1e3a5f;color:#93c5fd}
:root[data-theme="dark"] .b2{background:#4a1942;color:#f9a8d4}
:root[data-theme="dark"] .b3{background:#3d2c00;color:#fde047}
:root[data-theme="dark"] .b4{background:#064e3b;color:#6ee7b7}
:root[data-theme="dark"] .b5{background:#2e1065;color:#c4b5fd}
:root[data-theme="dark"] .b6{background:#431407;color:#fed7aa}
:root[data-theme="dark"] .b7{background:#1e293b;color:#94a3b8}
:root[data-theme="light"] .b1{background:#dbeafe;color:#1d4ed8}
:root[data-theme="light"] .b2{background:#fce7f3;color:#9d174d}
:root[data-theme="light"] .b3{background:#fef9c3;color:#854d0e}
:root[data-theme="light"] .b4{background:#d1fae5;color:#065f46}
:root[data-theme="light"] .b5{background:#ede9fe;color:#5b21b6}
:root[data-theme="light"] .b6{background:#ffedd5;color:#9a3412}
:root[data-theme="light"] .b7{background:#f1f5f9;color:#475569}
.q-detail{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);margin-bottom:.4rem;overflow:hidden}
.q-summary{display:flex;align-items:center;gap:.5rem;padding:.55rem .8rem;cursor:pointer;user-select:none;flex-wrap:wrap}
.q-summary::-webkit-details-marker{display:none}
.q-summary::before{content:"▸";color:var(--text3);flex-shrink:0;transition:transform .15s}
details[open] .q-summary::before{transform:rotate(90deg)}
.q-num{font-family:ui-monospace,monospace;font-size:.77rem;color:var(--text2);flex-shrink:0;width:2.8rem}
.q-text{flex:1;font-size:.82rem;font-weight:500;min-width:0}
.chips{display:flex;gap:.22rem;flex-shrink:0}
.chip{font-family:ui-monospace,monospace;font-size:.71rem;padding:.08em .38em;border-radius:3px;background:var(--bg3);font-weight:700}
.q-body{padding:.8rem;border-top:1px solid var(--border)}
.dim-grid{width:auto;margin-bottom:.7rem;font-size:.8rem}
.dim-lbl{color:var(--text2);white-space:nowrap;padding-right:.7rem}
.global-row td{border-top:2px solid var(--border)}
.inner-det{margin-bottom:.4rem;border:1px solid var(--border);border-radius:4px;overflow:hidden}
.inner-det>summary{padding:.35rem .6rem;cursor:pointer;font-size:.78rem;color:var(--text2);font-weight:600;background:var(--bg3);user-select:none}
.inner-det>summary::-webkit-details-marker{display:none}
.inner-det[open]>summary{border-bottom:1px solid var(--border)}
.hop{margin-bottom:.6rem;border-left:3px solid var(--sa);padding-left:.7rem}
.hop-hdr{font-weight:600;font-size:.8rem;margin-bottom:.15rem}
.hop-ia{font-size:.77rem;color:var(--text2);padding:.3rem .5rem;border-radius:3px;background:var(--bg2)}
.ans-box{font-size:.79rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;padding:.6rem;border-radius:0 4px 4px 0;border-left:3px solid var(--border);margin:.45rem 0;background:var(--bg2)}
.ans-sa{border-left-color:var(--sa)}.ans-dr{border-left-color:var(--dr)}
.ans-vk10{border-left-color:var(--vk10)}.ans-vk25{border-left-color:var(--vk25)}
.note{font-size:.75rem;color:var(--text3);font-style:italic;padding:.2rem .4rem}
.sec-hdr{font-size:.9rem;font-weight:700;margin:1.8rem 0 .5rem;color:var(--text2);display:flex;align-items:center;gap:.6rem}
.theme-btn{position:fixed;top:.8rem;right:.8rem;background:var(--bg2);border:1px solid var(--border);color:var(--text);border-radius:20px;padding:.28rem .7rem;cursor:pointer;font-size:.78rem;z-index:100}
.summary-note{font-size:.78rem;color:var(--text2);margin-bottom:.6rem}
"""

BADGE_CSS = ["b1","b2","b3","b4","b5","b6","b7"]

def build_report(all_q_sorted: list, sa_data: dict, dr_data: dict, vk10_data: dict, vk25_data: dict, ts: str) -> str:
    CFGS = [
        ("v_selfask",       "Self-Ask",     "sa"),
        ("v_decomp_raptor", "Decomp+Raptor","dr"),
        ("v_vanilla_k10",   "Vanilla k10",  "vk10"),
        ("v_vanilla_k25",   "Vanilla k25",  "vk25"),
    ]
    cfg_data = {
        "v_selfask":       sa_data,
        "v_decomp_raptor": dr_data,
        "v_vanilla_k10":   vk10_data,
        "v_vanilla_k25":   vk25_data,
    }

    def _get(cfg_key, row, field):
        return cfg_data[cfg_key].get(row,{}).get(field)

    all_rows = [q["excel_row"] for q in all_q_sorted]
    n = len(all_rows)

    # ── Global averages ──────────────────────────────────────────────────────
    def cfg_avgs(cfg_key):
        avgs = {d: _avg([_get(cfg_key,r,d) for r in all_rows]) for d in DIMS}
        avgs["score_global"] = _avg([_get(cfg_key,r,"score_global") for r in all_rows])
        avgs["ml_n"] = sum(1 for r in all_rows if _ml_flag(cfg_data[cfg_key].get(r)))
        return avgs
    all_avgs = {k: cfg_avgs(k) for k,_,_ in CFGS}

    dim_headers = "".join(f"<th>{DIM_LABELS[d]}</th>" for d in DIMS)

    def summary_row(cfg_key, lbl, css):
        avgs = all_avgs[cfg_key]
        dim_cells = "".join(f'<td class="num sc-{_score_cls(avgs[d])}">{_fmt(avgs[d])}</td>' for d in DIMS)
        g = avgs["score_global"]
        return (f'<tr><td class="bold cfg-{css}">{lbl}</td>{dim_cells}'
                f'<td class="num bold sc-{_score_cls(g)}">{_fmt(g)}</td>'
                f'<td class="num">{avgs["ml_n"]}/{n}</td></tr>')

    dr_avgs = all_avgs["v_decomp_raptor"]
    def delta_row(cfg_key, lbl, css):
        avgs = all_avgs[cfg_key]
        dim_cells = ""
        for d in DIMS:
            dv = _delta(avgs[d], dr_avgs[d])
            dim_cells += f'<td class="num delta-{_delta_cls(dv)}">{_sign(dv)}{_fmt(dv)}</td>'
        sg_d = _delta(avgs["score_global"], dr_avgs["score_global"])
        ml_d = avgs["ml_n"] - dr_avgs["ml_n"]
        return (f'<tr class="delta-row"><td class="cfg-{css}">Δ {lbl} − Raptor</td>'
                f'{dim_cells}<td class="num bold delta-{_delta_cls(sg_d)}">{_sign(sg_d)}{_fmt(sg_d)}</td>'
                f'<td class="num">{ml_d:+d}</td></tr>')

    summary_rows = "\n".join(summary_row(k,l,c) for k,l,c in CFGS)
    delta_rows   = "\n".join(delta_row(k,l,c) for k,l,c in CFGS if k!="v_decomp_raptor")

    # ── Per-section table ────────────────────────────────────────────────────
    by_section_norm = {}
    for q in all_q_sorted:
        ns = _norm_section(q["section"])
        by_section_norm.setdefault(ns, []).append(q["excel_row"])

    sec_rows_html = ""
    for i, sec_norm in enumerate(SECTION_ORDER_NORM):
        rows = sorted(by_section_norm.get(sec_norm, []))
        if not rows:
            continue
        badge_c = BADGE_CSS[i % len(BADGE_CSS)]
        lbl = SECTION_LABELS.get(sec_norm, sec_norm)
        cells = ""
        ref_g = _avg([_get("v_decomp_raptor",r,"score_global") for r in rows])
        for cfg_key,_,css in CFGS:
            v = _avg([_get(cfg_key,r,"score_global") for r in rows])
            dv = _delta(v,ref_g) if cfg_key!="v_decomp_raptor" else None
            d_str = (f" <span class='delta-{_delta_cls(dv)}'>{_sign(dv)}{_fmt(dv)}</span>"
                     if dv is not None else "")
            cells += f'<td class="num sc-{_score_cls(v)}">{_fmt(v)}{d_str}</td>'
        sec_rows_html += (f'<tr><td><span class="badge {badge_c}">{lbl}</span></td>'
                          f'<td class="num" style="color:var(--text3)">{len(rows)}</td>'
                          f'{cells}</tr>')

    # ── Per-question detail table + collapsible detail ───────────────────────
    detail_html = ""
    last_sec_norm = None
    for i_sec, sec_norm in enumerate(SECTION_ORDER_NORM):
        rows_in_sec = sorted(by_section_norm.get(sec_norm, []))
        if not rows_in_sec:
            continue
        badge_c = BADGE_CSS[i_sec % len(BADGE_CSS)]
        lbl = SECTION_LABELS.get(sec_norm, sec_norm)
        detail_html += f'<div class="sec-hdr"><span class="badge {badge_c}">{lbl}</span> <span style="color:var(--text3);font-size:.75rem;font-weight:400">({len(rows_in_sec)} questions)</span></div>'

        q_by_row = {q["excel_row"]: q for q in all_q_sorted}
        for row in rows_in_sec:
            q = q_by_row.get(row, {})
            question_text = q.get("question","?")

            # Score chips
            chips = ""
            for cfg_key,_,css in CFGS:
                g = _get(cfg_key, row, "score_global")
                chips += f'<span class="chip cfg-{css}">{_fmt(g)}</span>'

            # Dim grid
            dim_grid = f'<table class="dim-grid"><thead><tr><th>Dim</th>'
            for _,lbl2,_ in CFGS: dim_grid += f'<th>{lbl2}</th>'
            dim_grid += '</tr></thead><tbody>'
            for d in DIMS:
                dim_grid += f'<tr><td class="dim-lbl">{DIM_LABELS[d]}</td>'
                for cfg_key,_,_ in CFGS:
                    v = _get(cfg_key,row,d)
                    dim_grid += f'<td class="num sc-{_score_cls(v)}">{_fmt(v)}</td>'
                dim_grid += '</tr>'
            dim_grid += '<tr class="global-row"><td class="dim-lbl"><strong>Global</strong></td>'
            for cfg_key,_,_ in CFGS:
                g = _get(cfg_key,row,"score_global")
                dim_grid += f'<td class="num bold sc-{_score_cls(g)}">{_fmt(g)}</td>'
            dim_grid += '</tr><tr><td class="dim-lbl">Mislabelling</td>'
            for cfg_key,_,_ in CFGS:
                e = cfg_data[cfg_key].get(row)
                fl = _ml_flag(e)
                dim_grid += f'<td class="num sc-{"lo" if fl else "hi"}">{"✗" if fl else "✓"}</td>'
            dim_grid += '</tr></tbody></table>'

            # Self-Ask hops
            sa_e   = cfg_data["v_selfask"].get(row, {})
            hops   = sa_e.get("hops",[])
            hops_h = ""
            for h in hops:
                srcs = ", ".join(s.get("source_type","?") for s in h.get("sources",[]))
                ia   = h.get("intermediate_answer","")
                hops_h += (f'<div class="hop">'
                           f'<div class="hop-hdr">Hop {h["hop"]} — <em>{h["follow_up"][:120]}{"…" if len(h["follow_up"])>120 else ""}</em></div>'
                           f'<div class="note" style="margin:.1rem 0">Sources: {srcs or "—"}</div>'
                           f'<div class="hop-ia">{ia[:350]}{"…" if len(ia)>350 else ""}</div>'
                           f'</div>')
            if not hops_h:
                hops_h = '<p class="note">0 hop — réponse directe (aucune retrieval)</p>'

            def ans_excerpt(cfg_key, css):
                a = (_get(cfg_key,row,"final_answer") or _get(cfg_key,row,"answer") or "")
                return f'<div class="ans-box ans-{css}">{a[:600]}{"…" if len(a)>600 else ""}</div>'

            detail_html += f"""
<details class="q-detail">
  <summary class="q-summary">
    <span class="q-num">Q{row:03d}</span>
    <span class="q-text">{question_text}</span>
    <span class="chips">{chips}</span>
  </summary>
  <div class="q-body">
    {dim_grid}
    <details class="inner-det">
      <summary>▸ Self-Ask — {len(hops)} hop(s)</summary>
      <div style="padding:.5rem">{hops_h}{ans_excerpt("v_selfask","sa")}</div>
    </details>
    <details class="inner-det">
      <summary>▸ Decomp+Raptor — {cfg_data["v_decomp_raptor"].get(row,{}).get("n_subquestions","?") } sous-q.</summary>
      <div style="padding:.5rem">{ans_excerpt("v_decomp_raptor","dr")}</div>
    </details>
    <details class="inner-det">
      <summary>▸ Vanilla k=10</summary>
      <div style="padding:.5rem">{ans_excerpt("v_vanilla_k10","vk10")}</div>
    </details>
    <details class="inner-det">
      <summary>▸ Vanilla k=25</summary>
      <div style="padding:.5rem">{ans_excerpt("v_vanilla_k25","vk25")}</div>
    </details>
  </div>
</details>"""

    cfg_legend = " · ".join(f'<span class="cfg-{css} bold">{lbl}</span>' for _,lbl,css in CFGS)

    return f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>4 configs RAG — 109 questions benchmark</title>
<style>{CSS}</style>
</head>
<body>
<button class="theme-btn" onclick="const r=document.documentElement,c=r.getAttribute('data-theme');r.setAttribute('data-theme',c==='dark'?'light':'dark')">◑ Thème</button>
<div class="wrap">
<h1>4 configurations RAG — 109 questions benchmark · Juge V4.3</h1>
<div class="meta">{cfg_legend} · Δ = config − Decomp+Raptor · Chips : SA / DR / Vk10 / Vk25 · Généré {ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}</div>

<section>
  <h2>Moyennes globales — {n} questions</h2>
  <div class="tscroll"><table>
    <thead><tr><th>Config</th>{dim_headers}<th>Global</th><th>Mislabelling</th></tr></thead>
    <tbody>{summary_rows}{delta_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>Score global moyen par section</h2>
  <div class="summary-note">Δ en indice = écart vs Decomp+Raptor</div>
  <div class="tscroll"><table>
    <thead><tr><th>Section</th><th>N</th>{"".join(f'<th class="cfg-{css}">{lbl}</th>' for _,lbl,css in CFGS)}</tr></thead>
    <tbody>{sec_rows_html}</tbody>
  </table></div>
</section>

<section>
  <h2>Détail par question</h2>
  <div class="summary-note">Cliquer sur une ligne pour déplier · Chips : <span class="chip cfg-sa">SA</span> <span class="chip cfg-dr">DR</span> <span class="chip cfg-vk10">Vk10</span> <span class="chip cfg-vk25">Vk25</span></div>
  {detail_html}
</section>
</div></body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Charger les 109 questions depuis COMPLET.json (v_decomp_raptor = référence)
    print(f"Chargement COMPLET.json...")
    complet = json.loads(COMPLET.read_text(encoding="utf-8"))
    ref_entries = {e["excel_row"]: e for e in complet.get("v_decomp_raptor", [])}
    dr_data     = ref_entries
    vk10_data   = {e["excel_row"]: e for e in complet.get("v_vanilla_k10", [])}
    vk25_data   = {e["excel_row"]: e for e in complet.get("v_vanilla_k25", [])}

    # Grouper par section normalisée
    by_section = {}
    for row, e in ref_entries.items():
        ns = _norm_section(e.get("section","?"))
        by_section.setdefault(ns, []).append(e)

    # Ordre de traversée : sections papier, questions par excel_row croissant
    all_q_ordered = []
    for sec_norm in SECTION_ORDER_NORM:
        qs = sorted(by_section.get(sec_norm, []), key=lambda x: x["excel_row"])
        all_q_ordered.extend(qs)
    # Ajouter sections non reconnues en fin (sécurité)
    known = set(SECTION_ORDER_NORM)
    for ns, qs in by_section.items():
        if ns not in known:
            all_q_ordered.extend(sorted(qs, key=lambda x: x["excel_row"]))

    print(f"{len(all_q_ordered)} questions chargées.")

    # Compter les questions déjà complètes
    n_done   = sum(1 for q in all_q_ordered if _is_complete(q["excel_row"]))
    n_todo   = len(all_q_ordered) - n_done
    print(f"  Déjà complètes : {n_done} | À traiter : {n_todo}")

    if n_todo == 0:
        print("\nToutes les questions sont déjà complètes. Génération du rapport...")
    else:
        # Charger le pipeline une seule fois
        pipeline, score_judge_v43 = _load_pipeline()

    # ── Boucle section par section ────────────────────────────────────────────
    total = len(all_q_ordered)
    idx   = 0
    for sec_norm in SECTION_ORDER_NORM:
        qs_sec = sorted(by_section.get(sec_norm, []), key=lambda x: x["excel_row"])
        if not qs_sec:
            continue
        lbl = SECTION_LABELS.get(sec_norm, sec_norm)
        print(f"\n{'─'*70}")
        print(f"SECTION : {lbl} ({len(qs_sec)} questions)")
        print(f"{'─'*70}")

        sec_scores = []
        for q in qs_sec:
            idx += 1
            row = q["excel_row"]
            if _is_complete(row):
                e = _load_existing(row)
                sg = e.get("score_global")
                if isinstance(sg,(int,float)):
                    sec_scores.append(sg)
                print(f"  [{idx:03d}/{total}] Q{row:03d} — skip (score={sg})")
                continue
            entry = _run_question(pipeline, score_judge_v43, q, idx, total)
            sg = entry.get("score_global")
            if isinstance(sg,(int,float)):
                sec_scores.append(sg)

        sec_avg = _avg(sec_scores)
        print(f"\n  ✓ Section terminée : {lbl}")
        print(f"    {len(qs_sec)} questions | Self-Ask moy. = {_fmt(sec_avg)} ({len(sec_scores)} scorées)")

    # Sections non reconnues (sécurité)
    known = set(SECTION_ORDER_NORM)
    extra_qs = [q for ns, qs in by_section.items() if ns not in known for q in qs]
    if extra_qs:
        print(f"\n{'─'*70}")
        print(f"SECTION INCONNUE : {len(extra_qs)} questions")
        for q in sorted(extra_qs, key=lambda x: x["excel_row"]):
            idx += 1
            row = q["excel_row"]
            if _is_complete(row):
                print(f"  [{idx:03d}/{total}] Q{row:03d} — skip")
                continue
            _run_question(pipeline, score_judge_v43, q, idx, total)

    # ── Chargement des résultats Self-Ask pour le rapport ─────────────────────
    print(f"\n{'='*70}")
    print("Chargement des résultats Self-Ask pour le rapport...")
    sa_data = {}
    n_complete_sa = 0
    for q in all_q_ordered:
        row = q["excel_row"]
        e = _load_existing(row)
        sa_data[row] = e
        if isinstance(e.get("score_global"),(int,float)):
            n_complete_sa += 1
    print(f"  {n_complete_sa}/{len(all_q_ordered)} questions Self-Ask complètes avec score.")

    # ── Rapport HTML ──────────────────────────────────────────────────────────
    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = build_report(all_q_ordered, sa_data, dr_data, vk10_data, vk25_data, ts2)
    out_html = Path(f"comparaisons_rag/rapport_selfask_109q_{ts2}.html")
    out_html.write_text(html, encoding="utf-8")
    dl_html  = DL_DIR / out_html.name
    try:
        dl_html.write_text(html, encoding="utf-8")
        print(f"\nRapport HTML → {out_html}")
        print(f"            → {dl_html}")
    except Exception:
        print(f"\nRapport HTML → {out_html}")

    # ── Résumé final ──────────────────────────────────────────────────────────
    sa_global = _avg([e.get("score_global") for e in sa_data.values()
                      if isinstance(e.get("score_global"),(int,float))])
    dr_global = _avg([e.get("score_global") for e in dr_data.values()
                      if isinstance(e.get("score_global"),(int,float))])
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ — {n_complete_sa} questions Self-Ask scorées")
    print(f"  Self-Ask global  : {_fmt(sa_global)}")
    print(f"  Decomp+Raptor    : {_fmt(dr_global)}")
    print(f"  Delta SA vs DR   : {_sign(_delta(sa_global,dr_global))}{_fmt(_delta(sa_global,dr_global))}")
    print(f"\nJSON Self-Ask  : {OUT_DIR}/")
    print(f"Rapport HTML   : {out_html}")
    print(f"Reprise        : python run_selfask_109q.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
