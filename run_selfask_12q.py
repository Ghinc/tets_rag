"""
run_selfask_12q.py — Pilote Self-Ask 12 questions + rapport HTML comparatif vs v_decomp_raptor.

Étapes :
  1. Pour chaque question cible, charge le JSON existant (si complet) ou lance Self-Ask + juge V4.3.
  2. Charge les scores v_decomp_raptor depuis COMPLET.json (sans recalculer).
  3. Génère un rapport HTML comparatif.

Idempotent : une question avec score_global présent dans comparaisons_rag/selfask_12q/selfask_q{row:03d}.json
             est sautée.

Relance : python run_selfask_12q.py
"""

import io, json, sys, time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from rag_selfask import SelfAskRAG
from eval_from_excel import score_judge_v43

# ── Constantes ────────────────────────────────────────────────────────────────

OUT_DIR    = Path("comparaisons_rag/selfask_12q")
COMPLET    = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
DL_DIR     = Path(r"C:\Users\comiti_g\Downloads")
TS         = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Orientations : objectif / perception / vaste
TARGET_12Q = [
    {"excel_row": 4,  "orientation": "objectif",   "section": "Retrieval mono-commune", "subsection": "Retrieval factuel et interprétation",        "question": "Quel est le score moyen de bien-être à Ajaccio ?",                                            "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 5,  "orientation": "objectif",   "section": "Retrieval mono-commune", "subsection": "Retrieval factuel et interprétation",        "question": "Quelle dimension OppChoVec obtient la note la plus faible à Ajaccio ?",                        "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 6,  "orientation": "objectif",   "section": "Retrieval mono-commune", "subsection": "Retrieval factuel et interprétation",        "question": "Quel est le score OppChoVec d'Ajaccio, par catégorie ?",                                       "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 8,  "orientation": "objectif",   "section": "Retrieval mono-commune", "subsection": "Retrieval factuel et interprétation",        "question": "Combien d'habitants ont répondu à l'enquête à Ajaccio ?",                                      "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 15, "orientation": "objectif",   "section": "Retrieval mono-commune", "subsection": "Retrieval source-spécifique",                "question": "De combien de services de proximité dispose la ville d'Ajaccio ?",                            "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 9,  "orientation": "perception", "section": "Retrieval mono-commune", "subsection": "Retrieval par sous-population",              "question": "Comment les 18-25 ans ressentent-ils le bien-être ?",                                          "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 10, "orientation": "perception", "section": "Retrieval mono-commune", "subsection": "Retrieval par sous-population",              "question": "Que pensent les entrepreneurs Ajacciens de la qualité de vie ?",                              "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 11, "orientation": "perception", "section": "Retrieval mono-commune", "subsection": "Retrieval par sous-population",              "question": "Que pensent les seniors du bien-être à Ajaccio ?",                                            "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 14, "orientation": "perception", "section": "Retrieval mono-commune", "subsection": "Retrieval source-spécifique",                "question": "Que révèlent les verbatims sur la sécurité à Corte ?",                                        "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 2,  "orientation": "vaste",      "section": "Retrieval mono-commune", "subsection": "Retrieval descriptif global",                "question": "Peut-on considérer Ajaccio comme un territoire favorable au bien-être ?",                      "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 25, "orientation": "vaste",      "section": "Retrieval mono-commune", "subsection": "Analyse multi-source et cohérence des données","question": "Les indicateurs objectifs et les perceptions qualitatives convergent-ils ?",               "expected_type": "reponse_substantielle_attendue"},
    {"excel_row": 35, "orientation": "vaste",      "section": "Raisonnement comparatif","subsection": "Comparaison croisée quanti/quali",           "question": "Observe-t-on un écart significatif entre indicateurs objectifs et perception à Bastia ?",    "expected_type": "reponse_substantielle_attendue"},
]

DIMS = ["pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti"]
DIM_LABELS = {"pertinence": "Pertinence", "fondement_factuel": "Factuel",
              "nuance_incertitude": "Nuance", "coherence_qualiquanti": "Quali/Quanti"}
ORIENT_LABELS = {"objectif": "Objectif", "perception": "Perception", "vaste": "Vaste"}
ORIENT_ROWS = {"objectif": [4,5,6,8,15], "perception": [9,10,11,14], "vaste": [2,25,35]}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ml_flag(ml_dict):
    if not ml_dict:
        return False
    return any(str(v).lower() not in ("non", "false", "", "null", "none")
               for v in ml_dict.values())

def _load_existing(row: int) -> dict | None:
    p = OUT_DIR / f"selfask_q{row:03d}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d.get("score_global"), (int, float)):
            return d
    return None

def _save(row: int, entry: dict):
    p = OUT_DIR / f"selfask_q{row:03d}.json"
    p.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Phase 1 : RAG Self-Ask ────────────────────────────────────────────────────

selfask_results: dict[int, dict] = {}

# Check what's already done
missing = [q for q in TARGET_12Q if _load_existing(q["excel_row"]) is None]

if missing:
    print(f"Initialisation SelfAskRAG ({len(missing)} questions à traiter)...")
    pipeline = SelfAskRAG(max_hops=5, k=5)
    pipeline.init()

    for idx, q in enumerate(TARGET_12Q, 1):
        row = q["excel_row"]
        existing = _load_existing(row)
        if existing:
            selfask_results[row] = existing
            print(f"  [{idx:02d}/12] Q{row:03d} déjà complète (score={existing['score_global']}) — skip.")
            continue

        print(f"\n  [{idx:02d}/12] Q{row:03d} [{q['orientation']:<10}] RAG Self-Ask...")
        t0 = time.time()
        final_answer, all_sources, hops = pipeline.query(q["question"])
        elapsed_rag = round(time.time() - t0, 1)

        entry = {
            "excel_row": row,
            "question": q["question"],
            "section": q["section"],
            "subsection": q["subsection"],
            "orientation": q["orientation"],
            "expected_type": q["expected_type"],
            "hops": hops,
            "n_hops": len(hops),
            "final_answer": final_answer,
            "all_sources": all_sources,
            "elapsed_rag_s": elapsed_rag,
            "meta": {
                "max_hops": 5, "k": 5,
                "model_loop": "mistral-large-latest",
                "temperature_loop": 0.0,
                "model_answerer": "claude-haiku-4-5-20251001",
                "ts": datetime.now().isoformat(),
            },
        }
        _save(row, entry)

        print(f"           ok ({elapsed_rag}s, {len(hops)} hops, {len(all_sources)} src)")
        print(f"           juge V4.3...")
        t1 = time.time()
        judge = score_judge_v43(
            q["question"], final_answer, all_sources,
            q["section"], q["subsection"], q["expected_type"],
        )
        elapsed_j = round(time.time() - t1, 1)
        entry.update(judge)
        entry["elapsed_judge_s"] = elapsed_j
        _save(row, entry)

        g = judge.get("score_global")
        print(f"           score={g} ({elapsed_j}s)")
        selfask_results[row] = entry
else:
    print("Toutes les entrées Self-Ask sont déjà complètes — chargement depuis JSON.")
    for q in TARGET_12Q:
        selfask_results[q["excel_row"]] = _load_existing(q["excel_row"])

# Charger les manquants depuis JSON si pipeline n'a pas tout initialisé
for q in TARGET_12Q:
    row = q["excel_row"]
    if row not in selfask_results:
        selfask_results[row] = _load_existing(row)

print(f"\n{len(selfask_results)}/12 entrées Self-Ask disponibles.")


# ── Phase 2 : Charger v_decomp_raptor depuis COMPLET.json ────────────────────

print("Chargement v_decomp_raptor depuis COMPLET.json...")
complet_data = json.loads(COMPLET.read_text(encoding="utf-8"))
vdr_list = complet_data.get("v_decomp_raptor", [])
vdr_by_row = {e["excel_row"]: e for e in vdr_list}

raptor_results: dict[int, dict] = {}
for q in TARGET_12Q:
    row = q["excel_row"]
    e = vdr_by_row.get(row)
    if e:
        raptor_results[row] = e
    else:
        print(f"  [WARN] Q{row:03d} absent de v_decomp_raptor dans COMPLET.json")


# ── Phase 3 : Génération HTML ─────────────────────────────────────────────────

def _fmt(v, digits=2):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "—"

def _avg(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    return round(sum(v)/len(v), 2) if v else None

def _delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(a - b, 2)
    return None

def _score_cls(v):
    if not isinstance(v, (int, float)): return "na"
    if v >= 4.5: return "hi"
    if v >= 3.5: return "mid"
    return "lo"

def _delta_cls(d):
    if not isinstance(d, (int, float)): return "na"
    if d > 0.25: return "pos"
    if d < -0.25: return "neg"
    return "neu"

def _ml(entry):
    flag = entry.get("mislabelling_flag")
    if flag is None:
        flag = _ml_flag(entry.get("mislabelling_detecte", {}))
    return "✗" if flag else "✓"

def _ml_cls(entry):
    flag = entry.get("mislabelling_flag")
    if flag is None:
        flag = _ml_flag(entry.get("mislabelling_detecte", {}))
    return "lo" if flag else "hi"

def build_html() -> str:
    # ── Aggregate stats ──────────────────────────────────────────────────────
    rows_present = [q["excel_row"] for q in TARGET_12Q
                    if q["excel_row"] in selfask_results and q["excel_row"] in raptor_results]

    def avg_dim(results_dict, dim):
        return _avg([results_dict[r].get(dim) for r in rows_present])

    def avg_global(results_dict):
        return _avg([results_dict[r].get("score_global") for r in rows_present])

    sa_avgs  = {d: avg_dim(selfask_results, d) for d in DIMS}
    sa_avgs["score_global"] = avg_global(selfask_results)
    dr_avgs  = {d: avg_dim(raptor_results, d)  for d in DIMS}
    dr_avgs["score_global"] = avg_global(raptor_results)

    sa_ml_pct = sum(1 for r in rows_present
                    if (selfask_results[r].get("mislabelling_flag") or
                        _ml_flag(selfask_results[r].get("mislabelling_detecte",{}))))
    dr_ml_pct = sum(1 for r in rows_present
                    if _ml_flag(raptor_results[r].get("mislabelling_detecte",{})))
    n = len(rows_present) or 1

    # ── Per-orientation stats ────────────────────────────────────────────────
    orient_stats = {}
    for ori, ori_rows in ORIENT_ROWS.items():
        present = [r for r in ori_rows if r in selfask_results and r in raptor_results]
        orient_stats[ori] = {
            "rows": present,
            "sa":  _avg([selfask_results[r].get("score_global") for r in present]),
            "dr":  _avg([raptor_results[r].get("score_global")  for r in present]),
        }
        orient_stats[ori]["delta"] = _delta(orient_stats[ori]["sa"], orient_stats[ori]["dr"])

    # ── Per-question rows ────────────────────────────────────────────────────
    q_meta = {q["excel_row"]: q for q in TARGET_12Q}

    # ── HTML build ───────────────────────────────────────────────────────────
    def sc(v, extra=""):
        cls = _score_cls(v)
        return f'<td class="num sc-{cls}{" " + extra if extra else ""}">{_fmt(v)}</td>'

    def dc(d, extra=""):
        cls = _delta_cls(d)
        sign = "+" if isinstance(d, (int, float)) and d > 0 else ""
        return f'<td class="num delta-{cls}{" " + extra if extra else ""}">{sign}{_fmt(d)}</td>'

    # Summary table rows
    def summary_row(label, avgs, ml_n, cls=""):
        cells = "".join(f'<td class="num sc-{_score_cls(avgs[d])}">{_fmt(avgs[d])}</td>' for d in DIMS)
        cells += f'<td class="num sc-{_score_cls(avgs["score_global"])} bold">{_fmt(avgs["score_global"])}</td>'
        cells += f'<td class="num">{ml_n}/{n}</td>'
        return f'<tr class="{cls}"><td class="config-name">{label}</td>{cells}</tr>'

    delta_avgs = {d: _delta(sa_avgs[d], dr_avgs[d]) for d in DIMS}
    delta_avgs["score_global"] = _delta(sa_avgs["score_global"], dr_avgs["score_global"])

    def delta_row():
        cells = "".join(f'<td class="num delta-{_delta_cls(delta_avgs[d])}">'
                        f'{"+" if isinstance(delta_avgs[d],(int,float)) and delta_avgs[d]>0 else ""}'
                        f'{_fmt(delta_avgs[d])}</td>' for d in DIMS)
        sg_d = delta_avgs["score_global"]
        sign = "+" if isinstance(sg_d,(int,float)) and sg_d > 0 else ""
        cells += f'<td class="num bold delta-{_delta_cls(sg_d)}">{sign}{_fmt(sg_d)}</td>'
        cells += f'<td class="num delta-{_delta_cls(sa_ml_pct - dr_ml_pct)}">{sa_ml_pct-dr_ml_pct:+d}</td>'
        return f'<tr class="delta-row"><td class="config-name">Δ (Self-Ask − Raptor)</td>{cells}</tr>'

    # Question detail sections
    detail_sections = []
    for q in TARGET_12Q:
        row = q["excel_row"]
        if row not in selfask_results or row not in raptor_results:
            continue
        sa = selfask_results[row]
        dr = raptor_results[row]

        ori_badge = f'<span class="badge badge-{q["orientation"]}">{ORIENT_LABELS[q["orientation"]]}</span>'

        # Score comparison sub-table
        score_rows = ""
        for d in DIMS:
            sa_v, dr_v = sa.get(d), dr.get(d)
            dv = _delta(sa_v, dr_v)
            sign = "+" if isinstance(dv,(int,float)) and dv > 0 else ""
            score_rows += (
                f'<tr>'
                f'<td>{DIM_LABELS[d]}</td>'
                f'<td class="num sc-{_score_cls(sa_v)}">{_fmt(sa_v)}</td>'
                f'<td class="num sc-{_score_cls(dr_v)}">{_fmt(dr_v)}</td>'
                f'<td class="num delta-{_delta_cls(dv)}">{sign}{_fmt(dv)}</td>'
                f'</tr>'
            )
        sg_sa, sg_dr = sa.get("score_global"), dr.get("score_global")
        sg_d = _delta(sg_sa, sg_dr)
        sign = "+" if isinstance(sg_d,(int,float)) and sg_d > 0 else ""
        score_rows += (
            f'<tr class="global-row">'
            f'<td><strong>Global</strong></td>'
            f'<td class="num bold sc-{_score_cls(sg_sa)}">{_fmt(sg_sa)}</td>'
            f'<td class="num bold sc-{_score_cls(sg_dr)}">{_fmt(sg_dr)}</td>'
            f'<td class="num bold delta-{_delta_cls(sg_d)}">{sign}{_fmt(sg_d)}</td>'
            f'</tr>'
        )
        ml_sa = _ml(sa)
        ml_dr = _ml(dr)
        ml_cls_sa = _ml_cls(sa)
        ml_cls_dr = _ml_cls(dr)
        score_rows += (
            f'<tr>'
            f'<td>Mislabelling</td>'
            f'<td class="num sc-{ml_cls_sa}">{ml_sa}</td>'
            f'<td class="num sc-{ml_cls_dr}">{ml_dr}</td>'
            f'<td class="num">—</td>'
            f'</tr>'
        )

        # Self-Ask follow-up chain
        hops_html = ""
        for h in sa.get("hops", []):
            srcs_text = ", ".join(
                f'{s.get("source_type","?")}'
                for s in h.get("sources", [])
            )
            ia = h.get("intermediate_answer", "")
            hops_html += (
                f'<div class="hop">'
                f'<div class="hop-header">Hop {h["hop"]} — <em>{h["follow_up"]}</em></div>'
                f'<div class="hop-src">Sources : {srcs_text or "—"}</div>'
                f'<div class="hop-ia">{ia[:400]}{"…" if len(ia)>400 else ""}</div>'
                f'</div>'
            )
        if not hops_html:
            hops_html = "<p class='no-data'>Réponse directe (0 hop)</p>"

        # Final answer excerpt
        fa = sa.get("final_answer", "")
        fa_html = f'<div class="answer-box">{fa[:800]}{"…" if len(fa)>800 else ""}</div>'

        # v_decomp_raptor answer excerpt + mislabelling detail
        dr_answer = dr.get("answer", "")
        dr_html = f'<div class="answer-box raptor-box">{dr_answer[:800]}{"…" if len(dr_answer)>800 else ""}</div>'
        dr_n_sq = dr.get("n_subquestions", "?")
        dr_ml_detail = dr.get("mislabelling_detecte") or {}
        dr_ml_html = ""
        if dr_ml_detail:
            for rule, val in dr_ml_detail.items():
                if str(val).lower() not in ("non","false","","null","none"):
                    dr_ml_html += f'<div class="ml-detail"><strong>{rule}</strong> : {val}</div>'

        sa_ml_detail = sa.get("mislabelling_detecte") or {}
        sa_ml_html = ""
        if sa_ml_detail:
            for rule, val in sa_ml_detail.items():
                if str(val).lower() not in ("non","false","","null","none"):
                    sa_ml_html += f'<div class="ml-detail"><strong>{rule}</strong> : {val}</div>'

        justifs_sa = "".join(
            f'<div class="justif"><span class="dim-label">{DIM_LABELS[d]}</span> {sa.get(d+"_justif","") or "—"}</div>'
            for d in DIMS
        )
        justifs_dr = "".join(
            f'<div class="justif"><span class="dim-label">{DIM_LABELS[d]}</span> {dr.get(d+"_justif","") or "—"}</div>'
            for d in DIMS
        )

        detail_sections.append(f"""
<details class="q-detail">
  <summary class="q-summary">
    <span class="q-num">Q{row:03d}</span>
    {ori_badge}
    <span class="q-text">{q['question']}</span>
    <span class="q-scores">
      <span class="sc-chip sc-{_score_cls(sg_sa)}">SA {_fmt(sg_sa)}</span>
      <span class="sc-chip sc-{_score_cls(sg_dr)}">DR {_fmt(sg_dr)}</span>
      <span class="sc-chip delta-chip delta-{_delta_cls(sg_d)}">{sign}{_fmt(sg_d)}</span>
    </span>
  </summary>
  <div class="q-body">
    <div class="two-col">
      <!-- Scores -->
      <div class="col-block">
        <h4>Scores juge V4.3</h4>
        <table class="score-table">
          <thead><tr><th>Dimension</th><th>Self-Ask</th><th>Decomp+Raptor</th><th>Δ</th></tr></thead>
          <tbody>{score_rows}</tbody>
        </table>
        {f'<div class="ml-section"><strong>Mislabelling Self-Ask :</strong>{sa_ml_html or "<span class=ok>Aucun</span>"}</div>' if sa_ml_html else ""}
        {f'<div class="ml-section"><strong>Mislabelling Decomp+Raptor :</strong>{dr_ml_html or "<span class=ok>Aucun</span>"}</div>' if dr_ml_html else ""}
      </div>
      <!-- Métadonnées -->
      <div class="col-block">
        <h4>Métadonnées</h4>
        <table class="meta-table">
          <tr><td>Section</td><td>{q['section']}</td></tr>
          <tr><td>Sous-section</td><td>{q['subsection']}</td></tr>
          <tr><td>Self-Ask hops</td><td>{sa.get('n_hops','?')}</td></tr>
          <tr><td>Self-Ask sources</td><td>{len(sa.get('all_sources',[]))}</td></tr>
          <tr><td>Decomp+Raptor SQ</td><td>{dr_n_sq}</td></tr>
          <tr><td>Decomp+Raptor sources</td><td>{dr.get('n_sources','?')}</td></tr>
        </table>
      </div>
    </div>

    <!-- Self-Ask follow-up chain -->
    <details class="inner-details">
      <summary>▸ Self-Ask — séquence de follow-ups ({sa.get('n_hops',0)} hop(s))</summary>
      <div class="hops-container">{hops_html}</div>
      <h4 style="margin-top:1rem">Réponse finale Self-Ask</h4>
      {fa_html}
      {f'<div class="justifs-block"><h4>Justifications juge</h4>{justifs_sa}</div>' if justifs_sa else ""}
    </details>

    <!-- Decomp+Raptor answer -->
    <details class="inner-details">
      <summary>▸ Decomp+Raptor — réponse ({dr_n_sq} sous-questions)</summary>
      <p class="note">Texte des sous-questions non persisté dans COMPLET.json.</p>
      {dr_html}
      {f'<div class="justifs-block"><h4>Justifications juge</h4>{justifs_dr}</div>' if justifs_dr else ""}
    </details>
  </div>
</details>
""")

    # ── Orient table ─────────────────────────────────────────────────────────
    orient_rows_html = ""
    for ori in ["objectif", "perception", "vaste"]:
        s = orient_stats[ori]
        sa_v, dr_v, dv = s["sa"], s["dr"], s["delta"]
        sign = "+" if isinstance(dv,(int,float)) and dv > 0 else ""
        orient_rows_html += (
            f'<tr>'
            f'<td><span class="badge badge-{ori}">{ORIENT_LABELS[ori]}</span></td>'
            f'<td class="num">{", ".join(f"Q{r:03d}" for r in s["rows"])}</td>'
            f'<td class="num sc-{_score_cls(sa_v)}">{_fmt(sa_v)}</td>'
            f'<td class="num sc-{_score_cls(dr_v)}">{_fmt(dr_v)}</td>'
            f'<td class="num delta-{_delta_cls(dv)}">{sign}{_fmt(dv)}</td>'
            f'</tr>'
        )

    details_html = "\n".join(detail_sections)

    # ── Full HTML ─────────────────────────────────────────────────────────────
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-Ask vs Decomp+Raptor — 12 questions</title>
<style>
/* ── Tokens ────────────────────────────────────────────────────── */
:root {{
  --bg:      #f4f5f8;
  --bg2:     #ffffff;
  --bg3:     #ecedf2;
  --border:  #d0d3de;
  --text:    #1a1d2e;
  --text2:   #4a4e6b;
  --text3:   #8891b2;
  --accent:  #4a6fa5;
  --sa:      #3a7bd5;
  --dr:      #e07b39;
  --hi:      #1e7e44;
  --hi-bg:   #d4edda;
  --mid:     #856404;
  --mid-bg:  #fff3cd;
  --lo:      #842029;
  --lo-bg:   #f8d7da;
  --pos:     #1e7e44;
  --pos-bg:  #d4edda;
  --neg:     #842029;
  --neg-bg:  #f8d7da;
  --neu:     #4a4e6b;
  --neu-bg:  #ecedf2;
  --r:       6px;
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --bg:     #0d1117;
    --bg2:    #161b22;
    --bg3:    #1f2430;
    --border: #30363d;
    --text:   #c9d1d9;
    --text2:  #8b949e;
    --text3:  #484f58;
    --hi:     #3fb950;
    --hi-bg:  #0d2818;
    --mid:    #d29922;
    --mid-bg: #271e00;
    --lo:     #f85149;
    --lo-bg:  #2d0d0c;
    --pos:    #3fb950;
    --pos-bg: #0d2818;
    --neg:    #f85149;
    --neg-bg: #2d0d0c;
    --neu:    #8b949e;
    --neu-bg: #1f2430;
  }}
}}
:root[data-theme="dark"]  {{ --bg:#0d1117;--bg2:#161b22;--bg3:#1f2430;--border:#30363d;--text:#c9d1d9;--text2:#8b949e;--text3:#484f58;--hi:#3fb950;--hi-bg:#0d2818;--mid:#d29922;--mid-bg:#271e00;--lo:#f85149;--lo-bg:#2d0d0c;--pos:#3fb950;--pos-bg:#0d2818;--neg:#f85149;--neg-bg:#2d0d0c;--neu:#8b949e;--neu-bg:#1f2430; }}
:root[data-theme="light"] {{ --bg:#f4f5f8;--bg2:#ffffff;--bg3:#ecedf2;--border:#d0d3de;--text:#1a1d2e;--text2:#4a4e6b;--text3:#8891b2;--hi:#1e7e44;--hi-bg:#d4edda;--mid:#856404;--mid-bg:#fff3cd;--lo:#842029;--lo-bg:#f8d7da;--pos:#1e7e44;--pos-bg:#d4edda;--neg:#842029;--neg-bg:#f8d7da;--neu:#4a4e6b;--neu-bg:#ecedf2; }}

/* ── Reset & base ───────────────────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,sans-serif;font-size:14px;background:var(--bg);color:var(--text);line-height:1.55}}
a{{color:var(--accent)}}

/* ── Layout ─────────────────────────────────────────────────────── */
.wrap{{max-width:1200px;margin:0 auto;padding:1.5rem 1rem}}
.page-header{{margin-bottom:2rem}}
.page-header h1{{font-size:1.4rem;font-weight:700;margin-bottom:.3rem}}
.page-header .meta{{font-size:.8rem;color:var(--text2)}}
section{{margin-bottom:2.5rem}}
h2{{font-size:1rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text2);margin-bottom:.9rem;padding-bottom:.4rem;border-bottom:1px solid var(--border)}}
h3{{font-size:.9rem;font-weight:600;margin:.7rem 0 .4rem}}
h4{{font-size:.82rem;font-weight:600;color:var(--text2);margin:.5rem 0 .4rem;text-transform:uppercase;letter-spacing:.04em}}

/* ── Tables ─────────────────────────────────────────────────────── */
.tscroll{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
th{{background:var(--bg3);color:var(--text2);font-weight:600;text-align:left;padding:.45rem .7rem;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}}
td{{padding:.4rem .7rem;border-bottom:1px solid var(--border);vertical-align:top}}
tr:last-child td{{border-bottom:none}}
.num{{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace;white-space:nowrap}}
.bold{{font-weight:700}}
.config-name{{font-weight:600;white-space:nowrap}}

/* ── Score colors ────────────────────────────────────────────────── */
.sc-hi{{color:var(--hi);background:var(--hi-bg);border-radius:3px}}
.sc-mid{{color:var(--mid);background:var(--mid-bg);border-radius:3px}}
.sc-lo{{color:var(--lo);background:var(--lo-bg);border-radius:3px}}
.sc-na{{color:var(--text3)}}

/* ── Delta colors ────────────────────────────────────────────────── */
.delta-pos{{color:var(--pos);font-weight:600}}
.delta-neg{{color:var(--neg);font-weight:600}}
.delta-neu{{color:var(--neu)}}
.delta-na{{color:var(--text3)}}
.delta-row td{{background:var(--bg3);font-weight:600}}

/* ── Score table (detail) ────────────────────────────────────────── */
.score-table td,.score-table th{{padding:.3rem .6rem}}
.global-row td{{border-top:2px solid var(--border);font-size:.9rem}}

/* ── Meta table ──────────────────────────────────────────────────── */
.meta-table td:first-child{{color:var(--text2);font-size:.78rem;white-space:nowrap;padding-right:1rem}}
.meta-table td:last-child{{font-weight:500}}

/* ── Orientation & summary tables ─────────────────────────────────── */
.summary-table th:first-child{{width:200px}}

/* ── Badges ──────────────────────────────────────────────────────── */
.badge{{display:inline-block;padding:.15em .5em;border-radius:3px;font-size:.72rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase}}
.badge-objectif{{background:#dbeafe;color:#1d4ed8}}
.badge-perception{{background:#fce7f3;color:#9d174d}}
.badge-vaste{{background:#d1fae5;color:#065f46}}
@media(prefers-color-scheme:dark){{
  .badge-objectif{{background:#1e3a5f;color:#93c5fd}}
  .badge-perception{{background:#4a1942;color:#f9a8d4}}
  .badge-vaste{{background:#064e3b;color:#6ee7b7}}
}}
:root[data-theme="dark"] .badge-objectif{{background:#1e3a5f;color:#93c5fd}}
:root[data-theme="dark"] .badge-perception{{background:#4a1942;color:#f9a8d4}}
:root[data-theme="dark"] .badge-vaste{{background:#064e3b;color:#6ee7b7}}
:root[data-theme="light"] .badge-objectif{{background:#dbeafe;color:#1d4ed8}}
:root[data-theme="light"] .badge-perception{{background:#fce7f3;color:#9d174d}}
:root[data-theme="light"] .badge-vaste{{background:#d1fae5;color:#065f46}}

/* ── Config chips ─────────────────────────────────────────────────── */
.cfg-sa{{color:var(--sa);font-weight:700}}
.cfg-dr{{color:var(--dr);font-weight:700}}

/* ── Question details ─────────────────────────────────────────────── */
.q-detail{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);margin-bottom:.6rem;overflow:hidden}}
.q-summary{{display:flex;align-items:center;gap:.6rem;padding:.7rem 1rem;cursor:pointer;user-select:none;flex-wrap:wrap}}
.q-summary::-webkit-details-marker{{display:none}}
.q-summary::before{{content:"▸";color:var(--text3);flex-shrink:0;transition:transform .15s}}
details[open] .q-summary::before{{transform:rotate(90deg)}}
.q-num{{font-family:ui-monospace,monospace;font-size:.8rem;color:var(--text2);flex-shrink:0}}
.q-text{{flex:1;font-weight:500;min-width:0}}
.q-scores{{display:flex;gap:.3rem;flex-shrink:0;flex-wrap:wrap}}
.sc-chip{{font-family:ui-monospace,monospace;font-size:.75rem;padding:.1em .45em;border-radius:3px;font-weight:700}}
.delta-chip{{font-family:ui-monospace,monospace;font-size:.75rem;padding:.1em .45em;border-radius:3px;background:var(--bg3)}}

.q-body{{padding:1rem;border-top:1px solid var(--border)}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}}
@media(max-width:700px){{.two-col{{grid-template-columns:1fr}}}}
.col-block{{background:var(--bg3);border-radius:var(--r);padding:.8rem}}

/* ── Inner details ────────────────────────────────────────────────── */
.inner-details{{margin-top:.7rem;background:var(--bg3);border-radius:var(--r);overflow:hidden}}
.inner-details > summary{{padding:.5rem .8rem;cursor:pointer;font-size:.82rem;font-weight:600;color:var(--text2);user-select:none}}
.inner-details > summary::-webkit-details-marker{{display:none}}
.inner-details[open] > summary{{border-bottom:1px solid var(--border)}}

/* ── Hop chain ────────────────────────────────────────────────────── */
.hops-container{{padding:.8rem}}
.hop{{margin-bottom:.8rem;border-left:3px solid var(--sa);padding-left:.8rem}}
.hop-header{{font-weight:600;font-size:.83rem;margin-bottom:.25rem}}
.hop-header em{{font-weight:400;color:var(--text2)}}
.hop-src{{font-size:.74rem;color:var(--text3);margin-bottom:.2rem;font-family:ui-monospace,monospace}}
.hop-ia{{font-size:.8rem;color:var(--text2);background:var(--bg2);border-radius:3px;padding:.4rem .6rem}}

/* ── Answers ──────────────────────────────────────────────────────── */
.answer-box{{background:var(--bg2);border-left:3px solid var(--sa);padding:.7rem;border-radius:0 var(--r) var(--r) 0;font-size:.82rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;margin-top:.5rem}}
.raptor-box{{border-left-color:var(--dr)}}

/* ── Justifications ────────────────────────────────────────────────── */
.justifs-block{{margin-top:.7rem}}
.justif{{font-size:.79rem;margin-bottom:.3rem;color:var(--text2)}}
.dim-label{{font-weight:700;color:var(--text);margin-right:.3rem}}

/* ── Mislabelling ──────────────────────────────────────────────────── */
.ml-section{{margin-top:.5rem;font-size:.78rem}}
.ml-detail{{color:var(--neg);margin-top:.2rem;padding:.25rem .5rem;background:var(--lo-bg);border-radius:3px}}
.ok{{color:var(--pos)}}

/* ── Misc ─────────────────────────────────────────────────────────── */
.note{{font-size:.78rem;color:var(--text3);padding:.4rem .8rem;font-style:italic}}
.no-data{{color:var(--text3);font-size:.8rem;padding:.4rem}}
.justifs-block h4{{margin-top:.6rem}}

/* ── Theme toggle ──────────────────────────────────────────────────── */
.theme-btn{{position:fixed;top:.8rem;right:.8rem;background:var(--bg2);border:1px solid var(--border);color:var(--text);border-radius:20px;padding:.3rem .8rem;cursor:pointer;font-size:.8rem;z-index:100}}
</style>
</head>
<body>
<button class="theme-btn" onclick="
  const r=document.documentElement;
  const cur=r.getAttribute('data-theme');
  r.setAttribute('data-theme',cur==='dark'?'light':'dark');
">◑ Thème</button>

<div class="wrap">

<div class="page-header">
  <h1>Self-Ask vs Decomp+Raptor — Pilote 12 questions</h1>
  <div class="meta">
    Juge V4.3 (GPT-4o) · <span class="cfg-sa">v_selfask</span> : Mistral Large temp=0, max 5 hops, k=5, Haiku answerer ·
    <span class="cfg-dr">v_decomp_raptor</span> : chargé depuis COMPLET.json ·
    Généré le {TS[:4]}-{TS[4:6]}-{TS[6:8]} {TS[9:11]}:{TS[11:13]}
  </div>
</div>

<!-- ── RÉCAP GLOBAL ────────────────────────────────────────────── -->
<section>
  <h2>Récapitulatif global — moyennes sur 12 questions</h2>
  <div class="tscroll">
  <table class="summary-table">
    <thead>
      <tr>
        <th>Config</th>
        {"".join(f"<th>{DIM_LABELS[d]}</th>" for d in DIMS)}
        <th>Global</th>
        <th>Mislabelling</th>
      </tr>
    </thead>
    <tbody>
      {summary_row('<span class="cfg-sa">Self-Ask</span>', sa_avgs, sa_ml_pct)}
      {summary_row('<span class="cfg-dr">Decomp+Raptor</span>', dr_avgs, dr_ml_pct)}
      {delta_row()}
    </tbody>
  </table>
  </div>
</section>

<!-- ── PAR ORIENTATION ─────────────────────────────────────────── -->
<section>
  <h2>Δ Score global par orientation</h2>
  <div class="tscroll">
  <table>
    <thead>
      <tr><th>Orientation</th><th>Questions</th><th>Self-Ask</th><th>Decomp+Raptor</th><th>Δ</th></tr>
    </thead>
    <tbody>
      {orient_rows_html}
    </tbody>
  </table>
  </div>
</section>

<!-- ── DÉTAIL PAR QUESTION ────────────────────────────────────── -->
<section>
  <h2>Détail par question</h2>
  <p style="font-size:.8rem;color:var(--text2);margin-bottom:.8rem">
    SA = Self-Ask · DR = Decomp+Raptor · Δ = SA − DR · Cliquer sur une ligne pour déplier.
  </p>
  {details_html}
</section>

</div><!-- /wrap -->
</body>
</html>"""


html = build_html()

# ── Sauvegarde HTML ───────────────────────────────────────────────────────────

ts_html = datetime.now().strftime("%Y%m%d_%H%M%S")
out_html = OUT_DIR / f"rapport_selfask_vs_raptor_{ts_html}.html"
out_html.write_text(html, encoding="utf-8")

dl_html = DL_DIR / f"rapport_selfask_vs_raptor_{ts_html}.html"
try:
    dl_html.write_text(html, encoding="utf-8")
    print(f"\nHTML → {out_html}")
    print(f"     → {dl_html}")
except Exception:
    print(f"\nHTML → {out_html}")

print("Relance : python run_selfask_12q.py")
