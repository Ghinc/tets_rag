"""
run_nt_v2_109q.py — Run v_decomp_no_typing (v2) sur les 109 questions du benchmark.

v2 = ablation complète du typage :
  - décomposeur : règle (1) supprimée (v1)
  - answerer    : priorisation par type de sous-question supprimée (v2)
  - synthétiseur : instruction 3 neutralisée + RÈGLE PROPORTIONNALITÉ dé-typée (v2)

Charge v_decomp et v_decomp_no_typing (v1) depuis COMPLET.json.
Génère un rapport HTML complet : résumé + détail collapsible par question.

Usage:
    python run_nt_v2_109q.py            # toutes les 109 questions
    python run_nt_v2_109q.py --max 12   # pilot rapide
"""
import html as _html, json, re, sys, time, requests, argparse
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm

# ── Constants ────────────────────────────────────────────────────────────────
COMPLET     = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
OUT_DIR     = Path("comparaisons_rag/nt_v2")
BASE        = "http://localhost:8000/api/query"
HEADERS     = {"Content-Type": "application/json"}
JUDGE_DELAY = 1.0

HARD_ERROR_MARKERS = [
    "401", "authentication", "invalid_api_key",
    "insufficient_quota", "credit", "billing",
    "account", "disabled", "deactivated",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_hard_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in HARD_ERROR_MARKERS)


def _is_complete(row):
    """Complet = JSON existe + score_global + sub_questions sauvegardées."""
    p = OUT_DIR / f"ntv2_q{row:03d}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return (isinstance(d.get("score_global"), (int, float))
                and isinstance(d.get("sub_questions"), list)
                and len(d["sub_questions"]) > 0)
    except Exception:
        return False


def _load_complete():
    data = json.loads(COMPLET.read_text(encoding="utf-8"))
    result = {"v_decomp": {}, "v_decomp_no_typing_v1": {}}
    if isinstance(data, dict):
        for ver_key, entries in data.items():
            if ver_key == "v_decomp":
                for e in entries:
                    result["v_decomp"][e["excel_row"]] = e
            elif ver_key == "v_decomp_no_typing":
                for e in entries:
                    result["v_decomp_no_typing_v1"][e["excel_row"]] = e
    return result


def _load_questions_from_complet():
    data = json.loads(COMPLET.read_text(encoding="utf-8"))
    rows = {}
    entries = data if isinstance(data, list) else sum(data.values(), [])
    for e in entries:
        r = e.get("excel_row")
        if r and r not in rows:
            rows[r] = {
                "excel_row":  r,
                "section":    e.get("section", ""),
                "subsection": e.get("subsection", ""),
                "question":   e.get("question", ""),
            }
    return [rows[k] for k in sorted(rows)]


def judge_v43(question, answer, sources, section, subsection, expected_type):
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
        if "mislabelling_flag" not in result:
            result["mislabelling_flag"] = any(
                str(v).lower() not in ("non", "false", "", "null", "none")
                for v in result.get("mislabelling_detecte", {}).values()
            )
        return result
    except Exception as ex:
        err = str(ex)
        if "insufficient_quota" in err.lower() or "quota" in err.lower():
            raise RuntimeError(f"QUOTA ÉPUISÉ: {err}")
        return {"judge_error": err, "score_global": None}


def _expected_type(section):
    s = (section or "").lower()
    if "limite" in s and "architect" in s:
        return "limite_architecturale"
    return "reponse_substantielle_attendue"


# ── HTML report ───────────────────────────────────────────────────────────────

def e(s):
    return _html.escape(str(s) if s is not None else "")


def _score_color(v):
    if not isinstance(v, (int, float)):
        return "#888"
    if v >= 4.5:  return "#0d7a55"
    if v >= 3.5:  return "#c96a1a"
    return "#c0392b"


def build_report(complet_data, v2_entries, ts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CFGS = [
        ("v_decomp",          "#27ae60", complet_data["v_decomp"]),
        ("no_typing v1",      "#8e44ad", complet_data["v_decomp_no_typing_v1"]),
        ("no_typing v2",      "#e67e22", v2_entries),
    ]
    DIMS       = ["pertinence", "fondement_factuel", "nuance_incertitude",
                  "coherence_qualiquanti", "score_global"]
    DIM_LABELS = ["Pertinence", "Factuel", "Nuance", "Q/Q", "Global"]

    def gavg(key, edict):
        vals = [e_[key] for e_ in edict.values() if isinstance(e_.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    def gavg_sec(key, edict, sec):
        vals = [e_[key] for e_ in edict.values()
                if isinstance(e_.get(key), (int, float)) and e_.get("section") == sec]
        return round(sum(vals) / len(vals), 2) if vals else None

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = ""
    for label, color, edict in CFGS:
        n = sum(1 for e_ in edict.values() if isinstance(e_.get("score_global"), (int, float)))
        cells = ""
        for d in DIMS:
            v = gavg(d, edict)
            cells += f"<td style='color:{_score_color(v)};font-weight:bold'>{fmt(v)}</td>"
        summary_rows += (
            f"<tr><td style='color:{color};font-weight:bold'>{e(label)}</td>"
            f"<td>{n}/{len(edict)}</td>{cells}</tr>\n"
        )

    # ── Delta row (v2 - v_decomp) ─────────────────────────────────────────────
    delta_cells = ""
    for d in DIMS:
        v2  = gavg(d, v2_entries)
        ref = gavg(d, complet_data["v_decomp"])
        if v2 is not None and ref is not None:
            delta = v2 - ref
            sign  = "+" if delta >= 0 else ""
            col   = "#0d7a55" if delta >= 0 else "#c0392b"
            delta_cells += f"<td style='color:{col}'>{sign}{delta:.2f}</td>"
        else:
            delta_cells += "<td>—</td>"
    summary_rows += (
        f"<tr style='border-top:2px solid #ccc'>"
        f"<td style='color:#555;font-style:italic' colspan=2>Δ v2 − v_decomp</td>"
        f"{delta_cells}</tr>\n"
    )

    # ── Per-section table ─────────────────────────────────────────────────────
    sections = sorted({e_.get("section", "") for e_ in complet_data["v_decomp"].values()})
    sec_rows = ""
    for sec in sections:
        sec_short = (sec or "—")[:45]
        for label, color, edict in CFGS:
            cells = ""
            for d in DIMS:
                v = gavg_sec(d, edict, sec)
                cells += f"<td style='color:{_score_color(v)}'>{fmt(v)}</td>"
            sec_rows += (
                f"<tr><td style='font-size:.8em'>{e(sec_short)}</td>"
                f"<td style='color:{color}'>{e(label)}</td>{cells}</tr>\n"
            )
        sec_rows += "<tr><td colspan=7 style='border:none;padding:1px'></td></tr>\n"

    # ── Per-question collapsible blocks ───────────────────────────────────────
    all_rows = sorted({e_.get("excel_row") for e_ in complet_data["v_decomp"].values()})
    detail_blocks = ""

    for row in all_rows:
        ref_entry = (complet_data["v_decomp"].get(row)
                     or complet_data["v_decomp_no_typing_v1"].get(row)
                     or v2_entries.get(row, {}))
        question  = ref_entry.get("question", "")
        section   = ref_entry.get("section", "")
        subsection = ref_entry.get("subsection", "")

        # Score comparison header
        score_cells = ""
        for label, color, edict in CFGS:
            entry_ = edict.get(row, {})
            sg = entry_.get("score_global")
            mis = " ⚑" if entry_.get("mislabelling_flag") else ""
            score_cells += (
                f"<span style='color:{color};font-weight:bold;margin-right:1rem'>"
                f"{e(label)}: {fmt(sg)}{mis}</span>"
            )

        # v2 details
        v2e = v2_entries.get(row, {})
        sub_questions = v2e.get("sub_questions") or []

        # Sub-questions block
        sq_html = ""
        for i, sq in enumerate(sub_questions, 1):
            sq_text = e(sq.get("question", sq) if isinstance(sq, dict) else sq)
            sq_ans  = e(sq.get("answer", "") if isinstance(sq, dict) else "")
            sq_html += f"""
<div class="sq-block">
  <div class="sq-label">Sous-question {i}</div>
  <div class="sq-q">{sq_text}</div>
  {"<div class='sq-a'>" + sq_ans + "</div>" if sq_ans else ""}
</div>"""

        # Sources block
        sources = v2e.get("sources") or []
        src_rows = ""
        for s in sources:
            meta = s.get("metadata") or {}
            st   = s.get("source_type") or meta.get("source_type") or "?"
            comm = meta.get("commune") or meta.get("view_name") or "—"
            cont = (s.get("content") or "")[:300]
            src_rows += (
                f"<tr><td style='white-space:nowrap;font-size:.75em'>{e(st)}</td>"
                f"<td style='font-size:.75em'>{e(comm)}</td>"
                f"<td style='font-size:.72em;color:#555'>{e(cont)}…</td></tr>\n"
            )
        sources_html = (
            f"<table class='src-table'>"
            f"<tr><th>Type</th><th>Commune/vue</th><th>Extrait</th></tr>"
            f"{src_rows}</table>"
            if src_rows else "<em>Aucune source</em>"
        )

        # Judge details
        judge_html = ""
        for dim, label_d in [
            ("pertinence",         "Pertinence"),
            ("fondement_factuel",  "Fondement factuel"),
            ("nuance_incertitude", "Nuance/incertitude"),
            ("coherence_qualiquanti", "Cohérence Q/Q"),
        ]:
            score_d = v2e.get(dim)
            justif  = v2e.get(f"{dim}_justif") or v2e.get(f"{dim}_justification") or ""
            judge_html += (
                f"<div class='dim-row'>"
                f"<span class='dim-label'>{label_d}</span>"
                f"<span class='dim-score' style='color:{_score_color(score_d)}'>{fmt(score_d)}</span>"
                f"<span class='dim-justif'>{e(justif)}</span>"
                f"</div>"
            )
        raisonnement = v2e.get("raisonnement") or v2e.get("raisonnement_v43") or ""
        mis_details  = v2e.get("mislabelling_detecte") or {}
        mis_html = ""
        if any(str(v).lower() not in ("non", "false", "", "null", "none")
               for v in mis_details.values()):
            mis_html = (
                f"<div class='mis-box'>⚑ Mislabelling détecté : "
                + " | ".join(f"{k}: {v}" for k, v in mis_details.items()
                              if str(v).lower() not in ("non", "false", "", "null", "none"))
                + "</div>"
            )

        detail_blocks += f"""
<details class="q-block">
  <summary>
    <span class="q-num">Q{row}</span>
    <span class="q-sec">{e(section[:30])}</span>
    <span class="q-text">{e(question[:80])}</span>
    <span class="q-scores">{score_cells}</span>
  </summary>
  <div class="q-body">
    <div class="q-full"><b>Question complète :</b> {e(question)}</div>
    <div class="q-meta"><b>Section :</b> {e(section)} &nbsp;·&nbsp; <b>Sous-section :</b> {e(subsection)}</div>

    <details class="inner">
      <summary>Décomposition v2 ({len(sub_questions)} sous-questions)</summary>
      <div class="sq-list">{sq_html if sq_html else "<em>Non disponible</em>"}</div>
    </details>

    <details class="inner">
      <summary>Réponse finale v2</summary>
      <div class="answer-block">{e(v2e.get("answer",""))}</div>
    </details>

    <details class="inner">
      <summary>Sources récupérées ({len(sources)})</summary>
      {sources_html}
    </details>

    <details class="inner">
      <summary>Évaluation juge V4.3</summary>
      <div class="raison"><b>Raisonnement :</b> {e(raisonnement)}</div>
      {judge_html}
      {mis_html}
    </details>
  </div>
</details>"""

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>no_typing v1/v2 vs decomp — 109q — {ts}</title>
<style>
body{{font-family:system-ui,sans-serif;font-size:13px;margin:24px;background:#f8f9fc;color:#1a1a2e;line-height:1.5}}
h2{{font-size:1.1rem;margin-bottom:.3rem}}
h3{{font-size:.9rem;color:#444;margin:1.5rem 0 .4rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}}
table{{border-collapse:collapse;margin-bottom:1.5rem}}
th,td{{border:1px solid #ddd;padding:4px 8px;vertical-align:top}}
th{{background:#efefef;font-size:.78rem}}
tr:hover td{{background:#f0f4ff}}
.sum{{max-width:720px}}
.sec-tbl{{width:100%}}

/* Question blocks */
details.q-block{{margin-bottom:.4rem;border:1px solid #ddd;border-radius:5px;background:#fff}}
details.q-block summary{{
  display:flex;align-items:center;gap:.5rem;padding:.4rem .7rem;cursor:pointer;
  list-style:none;font-size:.82rem;flex-wrap:wrap
}}
details.q-block summary::-webkit-details-marker{{display:none}}
details.q-block[open] summary{{background:#f0f0f8;border-radius:5px 5px 0 0}}
.q-num{{font-weight:700;color:#555;min-width:2.5rem;font-size:.78rem}}
.q-sec{{color:#888;font-size:.73rem;white-space:nowrap;max-width:160px;overflow:hidden;text-overflow:ellipsis}}
.q-text{{flex:1;color:#222;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.q-scores{{display:flex;gap:.4rem;flex-wrap:wrap;font-size:.78rem}}
.q-body{{padding:.6rem 1rem;border-top:1px solid #eee}}
.q-full{{margin-bottom:.4rem;font-size:.82rem}}
.q-meta{{font-size:.75rem;color:#666;margin-bottom:.6rem}}

/* Inner details */
details.inner{{margin:.4rem 0;border:1px solid #e8e8f0;border-radius:4px}}
details.inner summary{{
  padding:.25rem .6rem;cursor:pointer;list-style:none;
  font-size:.78rem;font-weight:600;color:#333;background:#f5f5fb;border-radius:4px
}}
details.inner summary::-webkit-details-marker{{display:none}}
details.inner[open] summary{{border-radius:4px 4px 0 0;border-bottom:1px solid #e8e8f0}}

/* Sub-questions */
.sq-list{{padding:.5rem .7rem}}
.sq-block{{margin:.5rem 0;padding:.4rem .6rem;background:#fafafa;border-left:3px solid #8e44ad;border-radius:3px}}
.sq-label{{font-size:.7rem;font-weight:700;color:#8e44ad;text-transform:uppercase;letter-spacing:.04em}}
.sq-q{{font-size:.82rem;margin:.2rem 0;color:#222}}
.sq-a{{font-size:.78rem;color:#444;margin-top:.3rem;padding-top:.3rem;border-top:1px solid #eee}}

/* Answer */
.answer-block{{padding:.5rem .7rem;white-space:pre-wrap;font-size:.78rem;color:#333;background:#fafafa}}

/* Sources */
.src-table{{width:100%;border-collapse:collapse;font-size:.75rem}}
.src-table th{{background:#f0f0f0;padding:3px 6px}}
.src-table td{{padding:3px 6px;vertical-align:top;border:1px solid #e8e8e8}}

/* Judge dims */
.dim-row{{display:flex;gap:.5rem;align-items:baseline;margin:.3rem 0;font-size:.78rem}}
.dim-label{{font-weight:600;min-width:130px;color:#333}}
.dim-score{{font-weight:700;min-width:2rem}}
.dim-justif{{color:#555;flex:1}}
.raison{{font-size:.78rem;color:#444;margin:.4rem 0;padding:.3rem .5rem;background:#f7f7f0;border-left:3px solid #c96a1a;border-radius:3px}}
.mis-box{{margin:.4rem 0;padding:.3rem .6rem;background:#fee2e2;color:#991b1b;border-radius:3px;font-size:.78rem;border-left:3px solid #991b1b}}
</style>
</head><body>

<h2>v_decomp vs v_decomp_no_typing v1/v2 — {len(all_rows)} questions — Juge V4.3 (GPT-4o)</h2>
<p style="font-size:.8rem;color:#666;margin-bottom:1rem">
  <b>v_decomp</b> : référence &nbsp;·&nbsp;
  <b>no_typing v1</b> : décomposeur déTypé seulement &nbsp;·&nbsp;
  <b>no_typing v2</b> : décomposeur + answerer + synthétiseur déTypés &nbsp;·&nbsp;
  Généré le {ts}
</p>

<h3>Résumé global</h3>
<table class="sum">
<tr><th>Config</th><th>N</th>{''.join(f"<th>{l}</th>" for l in DIM_LABELS)}</tr>
{summary_rows}
</table>

<h3>Par section</h3>
<table class="sec-tbl">
<tr><th>Section</th><th>Config</th>{''.join(f"<th>{l}</th>" for l in DIM_LABELS)}</tr>
{sec_rows}
</table>

<h3>Détail par question <small style="font-weight:normal;color:#888">(cliquer pour déplier)</small></h3>
{detail_blocks}

<p style="color:#888;font-size:.8em;margin-top:1rem">
  ⚑ = mislabelling détecté · JSON bruts : comparaisons_rag/nt_v2/ntv2_q***.json
</p>
</body></html>"""

    p = OUT_DIR / f"nt_v2_rapport_complet_{ts}.html"
    p.write_text(html, encoding="utf-8")
    print(f"\nHTML → {p}", flush=True)
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--judge-delay", type=float, default=JUDGE_DELAY)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        requests.get("http://localhost:8000/", timeout=5).raise_for_status()
    except Exception as ex:
        print(f"[BLOQUANT] Serveur RAG inaccessible : {ex}", flush=True)
        sys.exit(1)

    print("Chargement COMPLET.json...", flush=True)
    complet_data = _load_complete()
    questions = _load_questions_from_complet()
    if args.max:
        questions = questions[:args.max]
    n_q = len(questions)

    n_decomp = sum(1 for e_ in complet_data["v_decomp"].values()
                   if isinstance(e_.get("score_global"), (int, float)))
    n_nt_v1  = sum(1 for e_ in complet_data["v_decomp_no_typing_v1"].values()
                   if isinstance(e_.get("score_global"), (int, float)))
    print(f"v_decomp (COMPLET)         : {n_decomp} scorées", flush=True)
    print(f"v_decomp_no_typing v1      : {n_nt_v1} scorées", flush=True)
    print(f"\nLancement v_decomp_no_typing v2 sur {n_q} questions...\n", flush=True)

    v2_entries = {}
    n_skip = n_done = n_err = 0

    for i, q in enumerate(questions, 1):
        row = q["excel_row"]

        if _is_complete(row):
            p = OUT_DIR / f"ntv2_q{row:03d}.json"
            try:
                v2_entries[row] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
            n_skip += 1
            print(f"  [{i:3}/{n_q}] Q{row:3}  SKIP", flush=True)
            continue

        question = q["question"]
        entry = {
            "excel_row":  row,
            "section":    q["section"],
            "subsection": q["subsection"],
            "question":   question,
        }

        # ── RAG ──────────────────────────────────────────────────────────────
        try:
            t0   = time.time()
            resp = requests.post(BASE,
                                 json={"question": question, "rag_version": "v_decomp_no_typing", "k": 5},
                                 headers=HEADERS, timeout=300)
            rag_elapsed = round(time.time() - t0, 1)
        except Exception as ex:
            if _is_hard_error(ex):
                print(f"\n[HARD STOP] Auth/quota RAG : {ex}", flush=True)
                sys.exit(1)
            entry.update({"rag_status": "exception", "rag_error": str(ex)})
            print(f"  [{i:3}/{n_q}] Q{row:3}  RAG EXCEPTION: {ex}", flush=True)
            (OUT_DIR / f"ntv2_q{row:03d}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            n_err += 1
            continue

        if resp.status_code != 200:
            entry.update({"rag_status": "error", "rag_error": resp.text[:200]})
            print(f"  [{i:3}/{n_q}] Q{row:3}  RAG ERREUR {resp.status_code}", flush=True)
            (OUT_DIR / f"ntv2_q{row:03d}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            n_err += 1
            continue

        data = resp.json()
        raw_sources = data.get("sources", [])
        sub_questions = data.get("sub_questions") or []

        sources_for_judge = [
            {"content":     s.get("content") or s.get("extrait") or "",
             "metadata":    s.get("metadata", {}),
             "source_type": s.get("source_type", ""),
             "label":       s.get("label", "")}
            for s in raw_sources
        ]
        entry.update({
            "rag_status":     "ok",
            "answer":         data.get("answer", ""),
            "sub_questions":  sub_questions,           # liste complète {question, answer}
            "n_subquestions": len(sub_questions),
            "n_sources":      len(raw_sources),
            "rag_elapsed_s":  rag_elapsed,
            "sources":        sources_for_judge,       # content + metadata complets
        })

        # Sauvegarde RAG immédiate
        (OUT_DIR / f"ntv2_q{row:03d}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── Judge V4.3 ────────────────────────────────────────────────────────
        time.sleep(args.judge_delay)
        etype = _expected_type(q["section"])
        try:
            scores = judge_v43(question, entry["answer"], sources_for_judge,
                               q["section"], q["subsection"], etype)
        except RuntimeError as quota_err:
            print(f"\n[HARD STOP] {quota_err}", flush=True)
            sys.exit(1)

        entry.update(scores)
        (OUT_DIR / f"ntv2_q{row:03d}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

        v2_entries[row] = entry
        n_done += 1

        sg_raw = scores.get("score_global")
        sg_str = f"{sg_raw:.2f}" if isinstance(sg_raw, (int, float)) else "?"
        mis    = " [MIS]" if scores.get("mislabelling_flag") else ""
        print(
            f"  [{i:3}/{n_q}] Q{row:3}  V4.3={sg_str}{mis}"
            f"  {len(sub_questions)}SQ  RAG={rag_elapsed}s  {len(raw_sources)}src",
            flush=True,
        )

    # Charger les skips
    for q in questions:
        row = q["excel_row"]
        if row not in v2_entries:
            p = OUT_DIR / f"ntv2_q{row:03d}.json"
            if p.exists():
                try:
                    v2_entries[row] = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass

    # ── Récap console ─────────────────────────────────────────────────────────
    n_scored = sum(1 for e_ in v2_entries.values() if isinstance(e_.get("score_global"), (int, float)))
    print(f"\n{'='*60}", flush=True)
    print(f"v2 : {n_scored}/{n_q} scorées  ({n_skip} skip, {n_done} nouveaux, {n_err} erreurs)")

    def gavg(key, edict):
        vals = [e_[key] for e_ in edict.values() if isinstance(e_.get(key), (int, float))]
        return f"{sum(vals)/len(vals):.2f}" if vals else "—"

    for label, edict in [
        ("v_decomp",                complet_data["v_decomp"]),
        ("v_decomp_no_typing v1",   complet_data["v_decomp_no_typing_v1"]),
        ("v_decomp_no_typing v2",   v2_entries),
    ]:
        print(f"  {label:<28} global={gavg('score_global', edict)}"
              f"  Q/Q={gavg('coherence_qualiquanti', edict)}"
              f"  Nuance={gavg('nuance_incertitude', edict)}")

    build_report(complet_data, v2_entries, ts)


if __name__ == "__main__":
    main()
