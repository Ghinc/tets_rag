"""
run_raptor_typing_12q.py
========================
Run ciblé : v_decomp_raptor vs v_decomp_raptor_no_typing, 12 questions.
Juge V4.3 STRICTEMENT identique au rapport de référence (run_ablations_103q.py) :
  - GPT-4o, _JUDGE_V43_SYSTEM, _parse_judge_v43, k=5, n_subquestions=5
  - Réponse tronquée à 4000 chars pour le juge (idem référence)

Idempotence : un (config × question) déjà complet (score_global présent) n'est pas refait.

Usage :
  python run_raptor_typing_12q.py           # RAG + juge + HTML
  python run_raptor_typing_12q.py --report  # HTML seulement depuis JSONs existants
"""
import argparse, json, re, sys, time, requests
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Forcer GPT-4o AVANT import eval_from_excel (identique à run_ablations_103q) ──
import importlib
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm

# ── Constantes (gelées, identiques au rapport de référence) ───────────────────
BASE        = "http://localhost:8000/api/query"
HEADERS     = {"Content-Type": "application/json"}
K           = 5
N_SUBQ      = 5
JUDGE_DELAY = 1.0
COMPLET     = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
OUT_DIR     = Path("comparaisons_rag/raptor_typing_12q")

CONFIGS     = ["v_decomp_raptor", "v_decomp_raptor_no_typing"]
TARGET_ROWS = [2, 4, 5, 6, 8, 9, 10, 11, 14, 15, 25, 35]

ORIENTATIONS = {
    2: "vaste",     4: "objective",  5: "objective",  6: "objective",
    8: "objective", 9: "perception", 10: "perception", 11: "perception",
    14: "perception", 15: "objective", 25: "vaste",   35: "vaste",
}
ORIENT_GROUPS = {
    "objective":  [4, 5, 6, 8, 15],
    "perception": [9, 10, 11, 14],
    "vaste":      [2, 25, 35],
}
ORIENT_LABEL = {
    "objective":  "Objectif — QUANTI",
    "perception": "Perception — QUALI",
    "vaste":      "Vaste — BOTH",
}

# ── Judge (copie exacte de run_ablations_103q) ────────────────────────────────
def judge_v43(question: str, answer: str, sources: list,
              section: str, subsection: str, expected_type: str) -> dict:
    sources_text = _build_sources_text(sources)
    user_prompt = (
        f"QUESTION : {question}\n\n"
        f"SECTION : {section}\n\n"
        f"SOUS-SECTION : {subsection}\n\n"
        f"TYPE DE RÉPONSE ATTENDUE : {expected_type}\n\n"
        f"SOURCES FOURNIES AU SYSTÈME :\n{sources_text}\n\n"
        f"RÉPONSE DU SYSTÈME :\n{answer[:4000]}\n\n"
        "Évalue cette réponse selon la procédure et le format spécifiés.\n"
        "Consulte les définitions opérationnelles et la grille AVANT de noter.\n"
        "Réponds UNIQUEMENT avec le JSON demandé, sans texte avant ni après."
    )
    try:
        raw = _call_llm(_JUDGE_V43_SYSTEM, user_prompt, max_tokens=3000, json_mode=True)
        m = re.search(r'\{[\s\S]*\}', raw)
        j = json.loads(m.group()) if m else {}
        result = _parse_judge_v43(j)
        result["judge_error"] = None
        md = result.get("mislabelling_detecte") or {}
        result["mislabelling_flag"] = any(
            str(v).lower() not in ("non", "false", "", "null", "none", "0")
            for v in md.values()
        )
        return result
    except Exception as e:
        return {"judge_error": str(e), "score_global": None, "mislabelling_flag": False}

# ── Métadonnées questions depuis COMPLET ──────────────────────────────────────
def load_meta() -> dict:
    with open(COMPLET, encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    for e in data["v_decomp_raptor"]:
        row = e["excel_row"]
        if row in TARGET_ROWS:
            meta[row] = {
                "question":       e["question"],
                "section":        e.get("section", ""),
                "subsection":     e.get("subsection", ""),
                "expected_type":  "reponse_substantielle_attendue",
            }
    return meta

# ── Appel API avec retry ──────────────────────────────────────────────────────
def call_api(question: str, rag_version: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(
                BASE,
                json={"question": question, "rag_version": rag_version, "k": K},
                headers=HEADERS,
                timeout=360,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  [retry {attempt+1}/{retries}] {e} — attente {wait}s", flush=True)
            if attempt < retries - 1:
                time.sleep(wait)
    return {"error": f"échec après {retries} tentatives", "answer": "", "sources": []}

# ── Un (config × question) ────────────────────────────────────────────────────
def json_path(row: int, cfg: str) -> Path:
    return OUT_DIR / f"Q{row:03d}_{cfg}.json"

def is_complete(row: int, cfg: str) -> bool:
    p = json_path(row, cfg)
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return isinstance(d.get("score_global"), (int, float))
    except Exception:
        return False

def run_single(row: int, cfg: str, meta: dict) -> dict:
    m = meta[row]
    question  = m["question"]
    section   = m["section"]
    subsection = m["subsection"]
    etype     = m["expected_type"]
    orientation = ORIENTATIONS[row]

    cfg_short = "TYP" if "no_typing" not in cfg else "NT "
    print(f"  Q{row:03d} [{cfg_short}] RAG...", end="", flush=True)
    t0 = time.time()
    resp = call_api(question, cfg)
    rag_elapsed = round(time.time() - t0, 1)

    if resp.get("error"):
        entry = {
            "excel_row": row, "config": cfg, "orientation": orientation,
            "question": question, "section": section, "subsection": subsection,
            "rag_status": "error", "rag_error": resp["error"],
            "score_global": None, "mislabelling_flag": False,
            "timestamp": datetime.now().isoformat(),
        }
        print(f" ERREUR: {resp['error']}", flush=True)
        return entry

    raw_sources = resp.get("sources", [])
    sources_for_judge = [
        {
            "content":     s.get("content") or s.get("extrait") or "",
            "metadata":    s.get("metadata", {}),
            "source_type": s.get("source_type", ""),
            "label":       s.get("label", ""),
        }
        for s in raw_sources
    ]
    answer     = resp.get("answer", "")
    sub_q      = resp.get("sub_questions") or []
    sources_mob = resp.get("sources_mobilisees") or []
    print(f" {rag_elapsed}s {len(raw_sources)}src {len(sub_q)}SQ — juge...", end="", flush=True)

    time.sleep(JUDGE_DELAY)
    t1 = time.time()
    scores = judge_v43(question, answer, sources_for_judge, section, subsection, etype)
    judge_elapsed = round(time.time() - t1, 1)

    sg    = scores.get("score_global")
    sg_s  = f"{sg:.2f}" if isinstance(sg, (int, float)) else "?"
    mis   = " [MIS]" if scores.get("mislabelling_flag") else ""
    print(f" V4.3={sg_s}{mis} ({judge_elapsed}s)", flush=True)

    entry = {
        # Identification
        "excel_row":    row,
        "config":       cfg,
        "orientation":  orientation,
        "question":     question,
        "section":      section,
        "subsection":   subsection,
        "expected_type": etype,
        # RAG
        "rag_status":       "ok",
        "rag_elapsed_s":    rag_elapsed,
        "n_sources":        len(raw_sources),
        "n_subquestions":   len(sub_q),
        "answer":           answer,
        "sub_questions":    sub_q,
        "sources":          sources_for_judge,  # contenu intégral
        "sources_mobilisees": sources_mob,
        # Juge V4.3 (sortie complète)
        **scores,
        "judge_elapsed_s":  judge_elapsed,
        # Méta
        "k":              K,
        "judge_model":    "gpt-4o",
        "judge_version":  "V4.3",
        "timestamp":      datetime.now().isoformat(),
    }
    return entry

def load_or_run(row: int, cfg: str, meta: dict, judge_delay_override: float = None) -> dict:
    p = json_path(row, cfg)
    cfg_short = "TYP" if "no_typing" not in cfg else "NT "

    # Déjà complet
    if is_complete(row, cfg):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        sg = d.get("score_global")
        print(f"  Q{row:03d} [{cfg_short}] déjà complet → score={sg:.2f}", flush=True)
        return d

    # RAG déjà fait, juge seulement
    if p.exists():
        with open(p, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("rag_status") == "ok" and existing.get("answer"):
            m = meta[row]
            delay = judge_delay_override if judge_delay_override is not None else JUDGE_DELAY
            print(f"  Q{row:03d} [{cfg_short}] RAG déjà OK — juge seulement (délai {delay}s)...", end="", flush=True)
            if delay > 2:
                time.sleep(delay)
            t1 = time.time()
            scores = judge_v43(
                existing["question"], existing["answer"], existing.get("sources", []),
                m["section"], m["subsection"], m["expected_type"],
            )
            existing.update(scores)
            existing["judge_elapsed_s"] = round(time.time() - t1, 1)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            sg = existing.get("score_global")
            sg_s = f"{sg:.2f}" if isinstance(sg, (int, float)) else "?"
            mis = " [MIS]" if existing.get("mislabelling_flag") else ""
            print(f" V4.3={sg_s}{mis}", flush=True)
            return existing

    # RAG + juge complet
    entry = run_single(row, cfg, meta)
    # Sauvegarder même si juge échoué (RAG réutilisable)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    return entry

# ── HTML ──────────────────────────────────────────────────────────────────────
COLORS = {
    "v_decomp_raptor":           "#2980b9",
    "v_decomp_raptor_no_typing": "#8e44ad",
}

def _sc(s, fmt=".2f"):
    if not isinstance(s, (int, float)):
        return "—"
    return format(s, fmt)

def _avg(entries, key):
    vals = [e[key] for e in entries if isinstance(e.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None

def _mis_count(entries):
    return sum(1 for e in entries if e.get("mislabelling_flag"))

def _delta(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 2)

def _delta_str(d):
    if d is None:
        return "—"
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.2f}"

def _score_color(s):
    if s is None:      return "#94a3b8"
    if s >= 4.5:       return "#16a34a"
    if s >= 3.5:       return "#65a30d"
    if s >= 2.5:       return "#ca8a04"
    return "#dc2626"

def make_html(all_entries: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    TYP = [all_entries[r]["v_decomp_raptor"] for r in TARGET_ROWS
           if all_entries.get(r, {}).get("v_decomp_raptor", {}).get("rag_status") == "ok"]
    NT  = [all_entries[r]["v_decomp_raptor_no_typing"] for r in TARGET_ROWS
           if all_entries.get(r, {}).get("v_decomp_raptor_no_typing", {}).get("rag_status") == "ok"]

    dims = ["pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti", "score_global"]
    dim_lbls = ["Pertinence", "Factuel", "Nuance", "Quali/Q", "Global"]

    # Averages globaux
    avg_typ = {d: _avg(TYP, d) for d in dims}
    avg_nt  = {d: _avg(NT, d) for d in dims}
    mis_typ = _mis_count(TYP)
    mis_nt  = _mis_count(NT)

    # Averages par orientation
    def orient_delta(orient):
        rows = ORIENT_GROUPS[orient]
        t = [all_entries.get(r, {}).get("v_decomp_raptor", {}) for r in rows
             if all_entries.get(r, {}).get("v_decomp_raptor", {}).get("rag_status") == "ok"]
        n = [all_entries.get(r, {}).get("v_decomp_raptor_no_typing", {}) for r in rows
             if all_entries.get(r, {}).get("v_decomp_raptor_no_typing", {}).get("rag_status") == "ok"]
        at = _avg(t, "score_global")
        an = _avg(n, "score_global")
        return at, an, _delta(at, an), _mis_count(t), _mis_count(n)

    od = {o: orient_delta(o) for o in ["objective", "perception", "vaste"]}

    # ── DECISION block ────────────────────────────────────────────────────────
    dg = _delta(avg_typ["score_global"], avg_nt["score_global"])
    dg_str = _delta_str(dg)
    dg_sign = "TYPAGE > NO_TYPING" if (dg or 0) > 0 else ("NO_TYPING > TYPAGE" if (dg or 0) < 0 else "égalité")
    delta_mis = mis_typ - mis_nt
    delta_mis_str = ("+" if delta_mis > 0 else "") + str(delta_mis)

    def orient_row(orient):
        at, an, d, mt, mn = od[orient]
        d_str = _delta_str(d)
        clr = "#16a34a" if (d or 0) > 0 else ("#dc2626" if (d or 0) < 0 else "#64748b")
        lbl = ORIENT_LABEL[orient]
        n = len(ORIENT_GROUPS[orient])
        return (
            f'<tr><td class="ol">{lbl} (n={n})</td>'
            f'<td class="oc">{_sc(at)}</td><td class="oc">{_sc(an)}</td>'
            f'<td class="oc" style="color:{clr};font-weight:700">{d_str}</td>'
            f'<td class="oc">{mt}/{n}</td><td class="oc">{mn}/{n}</td>'
            f'<td class="oc" style="color:{"#dc2626" if (mt-mn)>0 else "#16a34a" if (mt-mn)<0 else "#64748b"};font-weight:600">'
            f'{"+" if (mt-mn)>0 else ""}{mt-mn}</td>'
            f'</tr>'
        )

    decision_html = f"""
<div class="decision-box">
  <div class="decision-title">DÉCISION — Δ bruts juge V4.3</div>
  <div class="decision-note">Δ = v_decomp_raptor − v_decomp_raptor_no_typing (positif = typage meilleur)</div>
  <table class="decision-table">
    <thead>
      <tr>
        <th>Périmètre</th>
        <th>TYPAGE</th><th>NO_TYPING</th><th>Δ global</th>
        <th>Mis. TYPAGE</th><th>Mis. NO_TYPING</th><th>Δ mis.</th>
      </tr>
    </thead>
    <tbody>
      <tr class="global-row">
        <td class="ol"><strong>Global (n=12)</strong></td>
        <td class="oc"><strong>{_sc(avg_typ["score_global"])}</strong></td>
        <td class="oc"><strong>{_sc(avg_nt["score_global"])}</strong></td>
        <td class="oc" style="color:{_score_color(dg)};font-weight:800;font-size:1.1em">{dg_str}</td>
        <td class="oc"><strong>{mis_typ}/12</strong></td>
        <td class="oc"><strong>{mis_nt}/12</strong></td>
        <td class="oc" style="font-weight:700;color:{"#dc2626" if delta_mis>0 else "#16a34a" if delta_mis<0 else "#64748b"}">{delta_mis_str}</td>
      </tr>
      {orient_row("objective")}
      {orient_row("perception")}
      {orient_row("vaste")}
    </tbody>
  </table>
  <div class="decision-caption">
    Δ global = {dg_str} ({dg_sign}) · Δ mislabelling = {delta_mis_str} (positif = plus de mislabelling avec typage)
  </div>
</div>"""

    # ── Tableau récap par config × dimension ─────────────────────────────────
    def summary_row(cfg_label, entries, color, mis_n):
        cells = "".join(
            f'<td style="text-align:center">{_sc(_avg(entries, d))}</td>'
            for d in dims
        )
        mis_pct = round(100*mis_n/len(entries)) if entries else 0
        return (
            f'<tr><td style="color:{color};font-weight:700">{cfg_label}</td>'
            f'<td style="text-align:center">{len(entries)}/12</td>'
            f'{cells}'
            f'<td style="text-align:center;color:#dc2626">{mis_n}/12 ({mis_pct}%)</td>'
            f'</tr>'
        )

    summary_html = (
        summary_row("v_decomp_raptor", TYP, COLORS["v_decomp_raptor"], mis_typ) +
        summary_row("v_decomp_raptor_no_typing", NT, COLORS["v_decomp_raptor_no_typing"], mis_nt)
    )

    # ── Détail par question ───────────────────────────────────────────────────
    def mis_html(entry):
        md = entry.get("mislabelling_detecte") or {}
        if not md:
            return '<span class="mis-non">non</span>'
        parts = []
        for rule, val in md.items():
            oui = str(val).lower() not in ("non", "false", "", "null", "none", "0")
            txt = str(val)[:180]
            parts.append(
                f'<div class="mis-rule">'
                f'<span class="mis-label {"mis-oui" if oui else "mis-non-sm"}">{rule}</span> '
                f'<span class="mis-val">{txt}</span>'
                f'</div>'
            )
        return "".join(parts)

    def render_answer_block(e, cfg_color):
        """Réponse + sous-questions + sources dans des <details> dépliables."""
        if e.get("rag_status") != "ok":
            return '<em style="color:#dc2626">RAG échoué</em>'

        # Réponse finale
        answer = e.get("answer", "").strip()
        ans_block = (
            f'<details class="exp-details"><summary class="exp-sum" style="color:{cfg_color}">Réponse complète</summary>'
            f'<div class="exp-body answer-body">{answer}</div></details>'
        )

        # Sous-questions
        sub_q = e.get("sub_questions") or []
        if sub_q:
            sq_items = ""
            for sq in sub_q:
                sq_ans = sq.get("answer", "").strip()
                sq_ans_block = (
                    f'<details class="sq-inner-det"><summary class="sq-inner-sum">réponse intermédiaire</summary>'
                    f'<div class="sq-inner-body">{sq_ans}</div></details>'
                ) if sq_ans else ""
                sq_items += (
                    f'<li><span class="sq-num-lbl">{sq.get("idx","·")}</span>'
                    f' <span class="sq-q-text">{sq.get("question","")}</span>'
                    f'{sq_ans_block}</li>'
                )
            sq_block = (
                f'<details class="exp-details"><summary class="exp-sum" style="color:{cfg_color}">Sous-questions ({len(sub_q)})</summary>'
                f'<ol class="sq-ol">{sq_items}</ol></details>'
            )
        else:
            sq_block = ""

        # Sources
        sources = e.get("sources") or []
        if sources:
            src_items = ""
            for i, s in enumerate(sources, 1):
                meta = s.get("metadata", {})
                content = s.get("content", "").strip()
                src_type = s.get("source_type") or meta.get("type", "")
                label = s.get("label") or meta.get("view", "") or ""
                commune = meta.get("commune", "")
                sq_idx = meta.get("sub_question_idx", "")
                sq_lbl = f" · SQ{sq_idx}" if sq_idx != "" else ""
                type_cls = "src-quali" if "quali" in src_type.lower() or "verbatim" in src_type.lower() or "entretien" in src_type.lower() else \
                           "src-quanti" if "quanti" in src_type.lower() or "oppchovec" in src_type.lower() or "indicateur" in src_type.lower() else "src-other"
                src_items += (
                    f'<details class="src-det"><summary class="src-sum">'
                    f'<span class="src-badge {type_cls}">{src_type or "?"}</span>'
                    f' {commune}{" — " if commune and label else ""}{label}{sq_lbl}'
                    f'</summary><div class="src-content">{content}</div></details>'
                )
            src_block = (
                f'<details class="exp-details"><summary class="exp-sum" style="color:{cfg_color}">Sources ({len(sources)})</summary>'
                f'<div class="src-list">{src_items}</div></details>'
            )
        else:
            src_block = '<em class="muted-txt">— aucune source —</em>'

        return ans_block + sq_block + src_block

    detail_rows = ""
    for row in TARGET_ROWS:
        orient = ORIENTATIONS[row]
        orient_lbl = ORIENT_LABEL[orient]
        typ_e = all_entries.get(row, {}).get("v_decomp_raptor", {})
        nt_e  = all_entries.get(row, {}).get("v_decomp_raptor_no_typing", {})
        question = typ_e.get("question") or nt_e.get("question") or f"Q{row}"
        subsec   = typ_e.get("subsection") or ""

        def mini_scores(e, color):
            if e.get("rag_status") != "ok":
                return f'<td colspan="6" style="color:red">ERREUR RAG</td>'
            sg = e.get("score_global")
            mis_flag = e.get("mislabelling_flag", False)
            sg_s = f"{sg:.2f}" if isinstance(sg, (int, float)) else "?"
            cells = "".join(
                f'<td style="text-align:center">{_sc(e.get(d))}</td>'
                for d in ["pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti"]
            )
            mis_badge = f'<span class="mis-badge">MIS</span>' if mis_flag else ""
            return (
                f'{cells}'
                f'<td style="text-align:center;font-weight:800;color:{color}">{sg_s}</td>'
                f'<td style="text-align:center">{mis_badge}</td>'
            )

        typ_sg = typ_e.get("score_global")
        nt_sg  = nt_e.get("score_global")
        dq     = _delta(typ_sg, nt_sg)
        dq_s   = _delta_str(dq)
        dq_clr = _score_color(dq)

        raisonnement_typ = (typ_e.get("raisonnement") or "")
        raisonnement_nt  = (nt_e.get("raisonnement") or "")

        detail_rows += f"""
<tr class="q-row">
  <td rowspan="4" class="q-num">Q{row}</td>
  <td rowspan="4" class="q-orient orient-{orient}">{orient_lbl}</td>
  <td rowspan="4" class="q-text" title="{question}">{question[:70]}<br><small class="subsec">{subsec}</small></td>
  <td style="color:{COLORS['v_decomp_raptor']};font-weight:600;white-space:nowrap">TYPAGE</td>
  {mini_scores(typ_e, COLORS['v_decomp_raptor'])}
  <td rowspan="2" style="text-align:center;font-weight:800;color:{dq_clr}">{dq_s}</td>
</tr>
<tr>
  <td style="color:{COLORS['v_decomp_raptor_no_typing']};font-weight:600;white-space:nowrap">NO_TYPING</td>
  {mini_scores(nt_e, COLORS['v_decomp_raptor_no_typing'])}
</tr>
<tr class="mis-row-detail">
  <td colspan="7" class="mis-detail-cell">
    <div class="mis-cols">
      <div class="mis-col"><span class="mis-col-lbl" style="color:{COLORS['v_decomp_raptor']}">TYPAGE — Mislabelling</span> {mis_html(typ_e)}</div>
      <div class="mis-col"><span class="mis-col-lbl" style="color:{COLORS['v_decomp_raptor_no_typing']}">NO_TYPING — Mislabelling</span> {mis_html(nt_e)}</div>
    </div>
    <div class="rais-cols">
      <div class="rais-col"><span class="rais-lbl">Raisonnement TYPAGE :</span> {raisonnement_typ}</div>
      <div class="rais-col"><span class="rais-lbl">Raisonnement NO_TYPING :</span> {raisonnement_nt}</div>
    </div>
  </td>
</tr>
<tr class="content-row">
  <td colspan="7" class="content-cell">
    <div class="content-cols">
      <div class="content-col">
        <div class="content-col-hdr" style="color:{COLORS['v_decomp_raptor']}">TYPAGE</div>
        {render_answer_block(typ_e, COLORS['v_decomp_raptor'])}
      </div>
      <div class="content-col">
        <div class="content-col-hdr" style="color:{COLORS['v_decomp_raptor_no_typing']}">NO_TYPING</div>
        {render_answer_block(nt_e, COLORS['v_decomp_raptor_no_typing'])}
      </div>
    </div>
  </td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Raptor Typing — 12q — V4.3 GPT-4o</title>
<style>
body{{font-family:system-ui,sans-serif;font-size:13px;margin:20px;color:#1e293b;background:#f8fafc}}
h2{{font-size:1.2rem;margin-bottom:4px}}
.subtitle{{color:#64748b;font-size:0.85rem;margin-bottom:20px}}

/* DECISION */
.decision-box{{background:#fff;border:2px solid #1e293b;border-radius:8px;padding:18px 20px;
  margin-bottom:28px;max-width:900px}}
.decision-title{{font-size:1rem;font-weight:800;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}}
.decision-note{{font-size:11px;color:#64748b;margin-bottom:12px}}
.decision-table{{border-collapse:collapse;font-size:13px;width:100%}}
.decision-table th{{background:#f1f5f9;padding:6px 10px;text-align:left;border:1px solid #e2e8f0;
  font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
.decision-table td{{padding:7px 10px;border:1px solid #e2e8f0}}
.ol{{font-weight:600}}
.oc{{text-align:center}}
.global-row td{{background:#f8fafc}}
.decision-caption{{margin-top:10px;font-size:11px;color:#64748b}}

/* Récap */
.recap-table{{border-collapse:collapse;margin-bottom:30px;font-size:13px}}
.recap-table th,td{{border:1px solid #ddd;padding:5px 9px}}
.recap-table th{{background:#f5f5f5;text-align:center}}

/* Détail */
table.detail{{border-collapse:collapse;width:100%;font-size:12.5px;margin-bottom:6px}}
.detail th{{background:#f5f5f5;padding:5px 8px;border:1px solid #ddd;text-align:center;font-size:11px}}
.detail td{{border:1px solid #ddd;padding:4px 8px;vertical-align:top}}
.q-num{{font-weight:800;color:#2980b9;text-align:center;font-size:14px}}
.q-orient{{font-size:10px;text-align:center;font-weight:600;border-radius:4px;padding:2px 6px;
  white-space:nowrap}}
.orient-objective{{background:#dbeafe;color:#1d4ed8}}
.orient-perception{{background:#fce7f3;color:#9d174d}}
.orient-vaste{{background:#e0e7ff;color:#4338ca}}
.q-text{{max-width:280px;font-size:12px}}
.subsec{{color:#94a3b8;font-style:italic}}
.mis-badge{{display:inline-block;background:#fef2f2;color:#dc2626;border:1px solid #fca5a5;
  border-radius:3px;font-size:9px;padding:1px 5px;font-weight:700}}
.mis-detail-cell{{background:#fafafa;padding:8px 12px}}
.mis-cols{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:8px}}
.mis-col-lbl{{font-weight:700;font-size:10px;text-transform:uppercase;color:#64748b;display:block;margin-bottom:3px}}
.mis-rule{{margin-bottom:2px;font-size:11px}}
.mis-label{{display:inline-block;font-size:9px;padding:1px 5px;border-radius:3px;
  font-weight:700;margin-right:4px}}
.mis-oui{{background:#fef2f2;color:#dc2626;border:1px solid #fca5a5}}
.mis-non-sm{{background:#f0fdf4;color:#16a34a;border:1px solid #86efac}}
.mis-non{{color:#16a34a;font-weight:600}}
.mis-val{{color:#374151;font-size:11px}}
.rais-cols{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.rais-lbl{{font-weight:600;color:#64748b;font-size:10px}}
.rais-col{{font-size:11px;color:#374151;line-height:1.5}}
.mis-row-detail td{{background:#fafafa}}
tr.q-row td{{background:#fff}}
/* Contenu dépliable */
.content-row td{{background:#f8fafc;padding:0}}
.content-cell{{padding:0}}
.content-cols{{display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid #e2e8f0}}
.content-col{{padding:12px 14px;border-right:1px solid #e2e8f0}}
.content-col:last-child{{border-right:none}}
.content-col-hdr{{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px}}
.exp-details{{margin-bottom:6px}}
.exp-sum{{font-size:12px;font-weight:600;cursor:pointer;user-select:none;padding:3px 0;list-style:none}}
.exp-sum::-webkit-details-marker{{display:none}}
.exp-sum::before{{content:"▶ ";font-size:9px}}
details[open] > .exp-sum::before{{content:"▼ "}}
.exp-body{{padding:8px 10px;margin-top:4px;font-size:12px;line-height:1.7;color:#1e293b;
  white-space:pre-wrap;word-break:break-word;
  background:#fff;border-left:3px solid #e2e8f0;border-radius:0 4px 4px 0}}
.answer-body{{max-height:600px;overflow-y:auto}}
/* Sous-questions */
.sq-ol{{margin:6px 0 0 16px;padding:0;font-size:12px;line-height:1.5}}
.sq-ol li{{margin-bottom:8px}}
.sq-num-lbl{{font-weight:700;color:#64748b;margin-right:4px}}
.sq-q-text{{font-weight:500}}
.sq-inner-det{{margin-top:4px;margin-left:4px}}
.sq-inner-sum{{font-size:11px;color:#94a3b8;cursor:pointer;list-style:none}}
.sq-inner-sum::-webkit-details-marker{{display:none}}
.sq-inner-sum::before{{content:"▶ "}}
details[open] > .sq-inner-sum::before{{content:"▼ "}}
.sq-inner-body{{padding:6px 8px;margin-top:3px;font-size:11px;line-height:1.6;color:#374151;
  background:#f8fafc;border-left:2px solid #e2e8f0;white-space:pre-wrap;word-break:break-word}}
/* Sources */
.src-list{{padding:4px 0}}
.src-det{{margin-bottom:4px}}
.src-sum{{font-size:11px;cursor:pointer;list-style:none;display:flex;align-items:center;gap:6px;color:#374151}}
.src-sum::-webkit-details-marker{{display:none}}
.src-sum::before{{content:"▶ ";font-size:9px;color:#94a3b8}}
details[open] > .src-sum::before{{content:"▼ ";font-size:9px}}
.src-badge{{display:inline-block;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700;white-space:nowrap}}
.src-quali{{background:#fce7f3;color:#9d174d;border:1px solid #f9a8d4}}
.src-quanti{{background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd}}
.src-other{{background:#f1f5f9;color:#475569;border:1px solid #cbd5e1}}
.src-content{{padding:6px 10px;margin-top:3px;font-size:11px;line-height:1.6;color:#374151;
  background:#fff;border-left:2px solid #e2e8f0;white-space:pre-wrap;word-break:break-word;
  max-height:300px;overflow-y:auto}}
.muted-txt{{color:#94a3b8;font-size:11px}}
</style>
</head>
<body>
<h2>v_decomp_raptor vs v_decomp_raptor_no_typing — 12 questions — Juge V4.3 (GPT-4o)</h2>
<div class="subtitle">Généré le {ts} · k={K} · n_subquestions={N_SUBQ} · Juge : GPT-4o, grille V4.3, json_mode=True</div>

{decision_html}

<h3 style="margin-bottom:8px">Récapitulatif par config</h3>
<table class="recap-table">
  <thead>
    <tr>
      <th style="text-align:left">Config</th><th>N OK</th>
      {''.join(f'<th>{l}</th>' for l in dim_lbls)}
      <th>Mislabelling</th>
    </tr>
  </thead>
  <tbody>{summary_html}</tbody>
</table>

<h3 style="margin-bottom:8px">Détail par question</h3>
<table class="detail">
  <thead>
    <tr>
      <th>#</th><th>Orient.</th><th>Question</th><th>Config</th>
      <th>Pert.</th><th>Fact.</th><th>Nuance</th><th>Q/Q</th><th>Global</th><th>Mis.</th>
      <th>Δ</th>
    </tr>
  </thead>
  <tbody>
  {detail_rows}
  </tbody>
</table>

<p style="color:#94a3b8;font-size:11px;margin-top:16px">
  Rapport généré le {ts} ·
  Juge : GPT-4o · Grille : V4.3 · Sources brutes incluses dans les JSONs ·
  Commande de relance : <code>python run_raptor_typing_12q.py</code>
</p>
</body>
</html>"""
    return html

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="HTML seulement")
    parser.add_argument("--retry-judge-only", action="store_true",
                        help="Rejuge seulement les entrées RAG-ok sans score (60s entre appels)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    meta = load_meta()
    all_entries = {}

    if not args.report:
        judge_delay_override = 60.0 if args.retry_judge_only else None
        lbl = "Retry juge seulement" if args.retry_judge_only else f"Run ciblé"
        print(f"{lbl} : {len(TARGET_ROWS)} questions × {len(CONFIGS)} configs (k={K})\n", flush=True)
        for row in TARGET_ROWS:
            all_entries[row] = {}
            for cfg in CONFIGS:
                all_entries[row][cfg] = load_or_run(row, cfg, meta, judge_delay_override)
    else:
        print("Mode --report : chargement JSONs existants…", flush=True)
        for row in TARGET_ROWS:
            all_entries[row] = {}
            for cfg in CONFIGS:
                p = json_path(row, cfg)
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        all_entries[row][cfg] = json.load(f)
                else:
                    print(f"  MANQUANT : {p.name}", flush=True)
                    all_entries[row][cfg] = {}

    html = make_html(all_entries)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = OUT_DIR / f"rapport_raptor_typing_12q_{ts_file}.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"\n{'='*60}", flush=True)
    print(f"HTML  → {html_path}  ({round(html_path.stat().st_size/1024)} Ko)", flush=True)
    print(f"JSONs → {OUT_DIR}/", flush=True)
    print(f"Relance : python run_raptor_typing_12q.py", flush=True)

if __name__ == "__main__":
    main()
