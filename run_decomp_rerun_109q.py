"""
run_decomp_rerun_109q.py — Re-run v_decomp avec les prompts à jour sur 109 questions.

Contexte : le v_decomp dans COMPLET.json a été généré avant le commit e4c0dea
(5 juillet 2026) qui a modifié _SYSTEM_ANSWERER et _SYSTEM_SYNTHESIZER_NO_BILAN.
Ce script produit un nouveau run v_decomp homogène avec v1/v2 (même code base).

À la fin : COMPLET.json est mis à jour (clé v_decomp remplacée).

Usage:
    python run_decomp_rerun_109q.py            # 109 questions
    python run_decomp_rerun_109q.py --max 5    # pilot
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

# ── Paths ────────────────────────────────────────────────────────────────────
COMPLET  = Path("comparaisons_rag/ablations_103q_v43_gpt4o_COMPLET.json")
OUT_DIR  = Path("comparaisons_rag/decomp_rerun")
BASE     = "http://localhost:8000/api/query"
HEADERS  = {"Content-Type": "application/json"}
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
    p = OUT_DIR / f"decomp_q{row:03d}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return (isinstance(d.get("score_global"), (int, float))
                and isinstance(d.get("sub_questions"), list)
                and len(d["sub_questions"]) > 0)
    except Exception:
        return False


def _load_complet():
    return json.loads(COMPLET.read_text(encoding="utf-8"))


def _load_questions():
    data = _load_complet()
    rows = {}
    # v_vanilla_k10 est la source canonique stable (jamais écrasée par ce script)
    for e in data.get("v_vanilla_k10", []):
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


# ── Report ───────────────────────────────────────────────────────────────────

def e(s):
    return _html.escape(str(s) if s is not None else "")


def _score_color(v):
    if not isinstance(v, (int, float)):
        return "#888"
    if v >= 4.5:  return "#0d7a55"
    if v >= 3.5:  return "#c96a1a"
    return "#c0392b"


def build_report(new_decomp, v1_data, v2_entries, ts):
    CFGS = [
        ("v_decomp (new)",   "#27ae60", new_decomp),
        ("no_typing v1",     "#8e44ad", v1_data),
        ("no_typing v2",     "#e67e22", v2_entries),
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

    # Summary
    summary_rows = ""
    for label, color, edict in CFGS:
        n = sum(1 for e_ in edict.values() if isinstance(e_.get("score_global"), (int, float)))
        cells = "".join(
            f"<td style='color:{_score_color(gavg(d, edict))};font-weight:bold'>{fmt(gavg(d, edict))}</td>"
            for d in DIMS
        )
        summary_rows += (
            f"<tr><td style='color:{color};font-weight:bold'>{e(label)}</td>"
            f"<td>{n}/{len(edict)}</td>{cells}</tr>\n"
        )

    # Delta rows
    for label, ref_edict in [("Δ v1 − v_decomp", v1_data), ("Δ v2 − v_decomp", v2_entries)]:
        delta_cells = ""
        for d in DIMS:
            v   = gavg(d, ref_edict)
            ref = gavg(d, new_decomp)
            if v is not None and ref is not None:
                delta = v - ref
                sign  = "+" if delta >= 0 else ""
                col   = "#0d7a55" if delta >= 0 else "#c0392b"
                delta_cells += f"<td style='color:{col}'>{sign}{delta:.2f}</td>"
            else:
                delta_cells += "<td>—</td>"
        summary_rows += (
            f"<tr style='border-top:1px solid #ddd'>"
            f"<td style='color:#555;font-style:italic' colspan=2>{e(label)}</td>"
            f"{delta_cells}</tr>\n"
        )

    # Per-section
    sections = sorted({e_.get("section", "") for e_ in new_decomp.values()})
    sec_rows = ""
    for sec in sections:
        sec_short = (sec or "—")[:45]
        for label, color, edict in CFGS:
            cells = "".join(
                f"<td style='color:{_score_color(gavg_sec(d, edict, sec))}'>{fmt(gavg_sec(d, edict, sec))}</td>"
                for d in DIMS
            )
            sec_rows += (
                f"<tr><td style='font-size:.8em'>{e(sec_short)}</td>"
                f"<td style='color:{color}'>{e(label)}</td>{cells}</tr>\n"
            )
        sec_rows += "<tr><td colspan=7 style='border:none;padding:1px'></td></tr>\n"

    # Per-question detail blocks (v_decomp new only for brevity)
    all_rows = sorted(new_decomp.keys())
    detail_blocks = ""
    for row in all_rows:
        entry  = new_decomp[row]
        question   = entry.get("question", "")
        section    = entry.get("section", "")
        subsection = entry.get("subsection", "")

        score_cells = ""
        for label, color, edict in CFGS:
            sg  = edict.get(row, {}).get("score_global")
            mis = " ⚑" if edict.get(row, {}).get("mislabelling_flag") else ""
            score_cells += (
                f"<span style='color:{color};font-weight:bold;margin-right:1rem'>"
                f"{e(label)}: {fmt(sg)}{mis}</span>"
            )

        sub_questions = entry.get("sub_questions") or []
        sq_html = ""
        for i, sq in enumerate(sub_questions, 1):
            sq_text = e(sq.get("question", sq) if isinstance(sq, dict) else sq)
            sq_ans  = e(sq.get("answer", "")   if isinstance(sq, dict) else "")
            sq_html += (
                f"<div class='sq-block'>"
                f"<div class='sq-label'>Sous-question {i}</div>"
                f"<div class='sq-q'>{sq_text}</div>"
                + (f"<div class='sq-a'>{sq_ans}</div>" if sq_ans else "")
                + "</div>"
            )

        sources = entry.get("sources") or []
        src_rows = "".join(
            f"<tr>"
            f"<td style='white-space:nowrap;font-size:.75em'>{e((s.get('source_type') or s.get('metadata',{}).get('source_type','?')))}</td>"
            f"<td style='font-size:.75em'>{e((s.get('metadata',{}).get('commune') or s.get('metadata',{}).get('view_name','—')))}</td>"
            f"<td style='font-size:.72em;color:#555'>{e((s.get('content',''))[:300])}…</td></tr>\n"
            for s in sources
        )
        sources_html = (
            f"<table class='src-table'><tr><th>Type</th><th>Commune/vue</th><th>Extrait</th></tr>"
            f"{src_rows}</table>" if src_rows else "<em>Aucune source</em>"
        )

        judge_html = ""
        for dim, label_d in [
            ("pertinence",           "Pertinence"),
            ("fondement_factuel",    "Fondement factuel"),
            ("nuance_incertitude",   "Nuance/incertitude"),
            ("coherence_qualiquanti","Cohérence Q/Q"),
        ]:
            score_d = entry.get(dim)
            justif  = entry.get(f"{dim}_justif") or entry.get(f"{dim}_justification") or ""
            judge_html += (
                f"<div class='dim-row'>"
                f"<span class='dim-label'>{label_d}</span>"
                f"<span class='dim-score' style='color:{_score_color(score_d)}'>{fmt(score_d)}</span>"
                f"<span class='dim-justif'>{e(justif)}</span>"
                f"</div>"
            )
        raisonnement = entry.get("raisonnement") or entry.get("raisonnement_v43") or ""
        mis_details  = entry.get("mislabelling_detecte") or {}
        mis_html = ""
        if any(str(v).lower() not in ("non", "false", "", "null", "none") for v in mis_details.values()):
            mis_html = (
                f"<div class='mis-box'>⚑ Mislabelling : "
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
    <div class="q-full"><b>Question :</b> {e(question)}</div>
    <div class="q-meta"><b>Section :</b> {e(section)} &nbsp;·&nbsp; {e(subsection)}</div>
    <details class="inner"><summary>Décomposition ({len(sub_questions)} SQ)</summary>
      <div class="sq-list">{sq_html or "<em>Non disponible</em>"}</div></details>
    <details class="inner"><summary>Réponse finale</summary>
      <div class="answer-block">{e(entry.get("answer",""))}</div></details>
    <details class="inner"><summary>Sources ({len(sources)})</summary>{sources_html}</details>
    <details class="inner"><summary>Juge V4.3</summary>
      <div class="raison"><b>Raisonnement :</b> {e(raisonnement)}</div>
      {judge_html}{mis_html}</details>
  </div>
</details>"""

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>v_decomp (new) vs v1/v2 — 109q — {ts}</title>
<style>
body{{font-family:system-ui,sans-serif;font-size:13px;margin:24px;background:#f8f9fc;color:#1a1a2e;line-height:1.5}}
h2{{font-size:1.1rem;margin-bottom:.3rem}}
h3{{font-size:.9rem;color:#444;margin:1.5rem 0 .4rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}}
table{{border-collapse:collapse;margin-bottom:1.5rem}}
th,td{{border:1px solid #ddd;padding:4px 8px;vertical-align:top}}
th{{background:#efefef;font-size:.78rem}}
tr:hover td{{background:#f0f4ff}}
details.q-block{{margin-bottom:.4rem;border:1px solid #ddd;border-radius:5px;background:#fff}}
details.q-block summary{{display:flex;align-items:center;gap:.5rem;padding:.4rem .7rem;cursor:pointer;list-style:none;font-size:.82rem;flex-wrap:wrap}}
details.q-block summary::-webkit-details-marker{{display:none}}
details.q-block[open] summary{{background:#f0f0f8;border-radius:5px 5px 0 0}}
.q-num{{font-weight:700;color:#555;min-width:2.5rem;font-size:.78rem}}
.q-sec{{color:#888;font-size:.73rem;white-space:nowrap;max-width:160px;overflow:hidden;text-overflow:ellipsis}}
.q-text{{flex:1;color:#222;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.q-scores{{display:flex;gap:.4rem;flex-wrap:wrap;font-size:.78rem}}
.q-body{{padding:.6rem 1rem;border-top:1px solid #eee}}
.q-full,.q-meta{{font-size:.82rem;margin-bottom:.4rem}}
.q-meta{{font-size:.75rem;color:#666}}
details.inner{{margin:.4rem 0;border:1px solid #e8e8f0;border-radius:4px}}
details.inner summary{{padding:.25rem .6rem;cursor:pointer;list-style:none;font-size:.78rem;font-weight:600;color:#333;background:#f5f5fb;border-radius:4px}}
details.inner summary::-webkit-details-marker{{display:none}}
details.inner[open] summary{{border-radius:4px 4px 0 0;border-bottom:1px solid #e8e8f0}}
.sq-list{{padding:.5rem .7rem}}
.sq-block{{margin:.5rem 0;padding:.4rem .6rem;background:#fafafa;border-left:3px solid #27ae60;border-radius:3px}}
.sq-label{{font-size:.7rem;font-weight:700;color:#27ae60;text-transform:uppercase;letter-spacing:.04em}}
.sq-q{{font-size:.82rem;margin:.2rem 0;color:#222}}
.sq-a{{font-size:.78rem;color:#444;margin-top:.3rem;padding-top:.3rem;border-top:1px solid #eee}}
.answer-block{{padding:.5rem .7rem;white-space:pre-wrap;font-size:.78rem;color:#333;background:#fafafa}}
.src-table{{width:100%;border-collapse:collapse}}
.src-table th{{background:#f0f0f0;padding:3px 6px}}
.src-table td{{padding:3px 6px;vertical-align:top;border:1px solid #e8e8e8}}
.dim-row{{display:flex;gap:.5rem;align-items:baseline;margin:.3rem 0;font-size:.78rem}}
.dim-label{{font-weight:600;min-width:130px;color:#333}}
.dim-score{{font-weight:700;min-width:2rem}}
.dim-justif{{color:#555;flex:1}}
.raison{{font-size:.78rem;color:#444;margin:.4rem 0;padding:.3rem .5rem;background:#f7f7f0;border-left:3px solid #c96a1a;border-radius:3px}}
.mis-box{{margin:.4rem 0;padding:.3rem .6rem;background:#fee2e2;color:#991b1b;border-radius:3px;font-size:.78rem;border-left:3px solid #991b1b}}
</style></head><body>

<h2>v_decomp (prompts à jour) vs no_typing v1/v2 — {len(all_rows)} questions — Juge V4.3 (GPT-4o)</h2>
<p style="font-size:.8rem;color:#666;margin-bottom:1rem">
  <b>v_decomp (new)</b> : prompts du {ts[:8]} &nbsp;·&nbsp;
  <b>no_typing v1</b> : décomposeur déTypé &nbsp;·&nbsp;
  <b>no_typing v2</b> : décomposeur + answerer + synthétiseur déTypés &nbsp;·&nbsp;
  Généré le {ts}
</p>

<h3>Résumé global</h3>
<table style="max-width:780px">
<tr><th>Config</th><th>N</th>{''.join(f"<th>{l}</th>" for l in DIM_LABELS)}</tr>
{summary_rows}
</table>

<h3>Par section</h3>
<table style="width:100%">
<tr><th>Section</th><th>Config</th>{''.join(f"<th>{l}</th>" for l in DIM_LABELS)}</tr>
{sec_rows}
</table>

<h3>Détail par question <small style="font-weight:normal;color:#888">(cliquer pour déplier — détails v_decomp new)</small></h3>
{detail_blocks}

<p style="color:#888;font-size:.8em;margin-top:1rem">
  ⚑ = mislabelling détecté · JSON bruts : comparaisons_rag/decomp_rerun/decomp_q***.json
</p>
</body></html>"""

    p = OUT_DIR / f"decomp_rerun_rapport_{ts}.html"
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

    # Vérification serveur
    try:
        requests.get("http://localhost:8000/", timeout=5).raise_for_status()
    except Exception as ex:
        print(f"[BLOQUANT] Serveur RAG inaccessible : {ex}", flush=True)
        sys.exit(1)

    questions = _load_questions()
    if args.max:
        questions = questions[:args.max]
    n_q = len(questions)

    # Charger v1 et v2 pour comparaison dans le rapport
    complet_raw = _load_complet()
    v1_data = {e["excel_row"]: e for e in complet_raw.get("v_decomp_no_typing", [])}
    v2_entries = {}
    for p2 in sorted(Path("comparaisons_rag/nt_v2").glob("ntv2_q*.json")):
        try:
            d2 = json.loads(p2.read_text(encoding="utf-8"))
            if isinstance(d2.get("score_global"), (int, float)):
                v2_entries[d2["excel_row"]] = d2
        except Exception:
            pass

    print(f"Lancement v_decomp (prompts à jour) sur {n_q} questions...\n", flush=True)

    new_decomp = {}
    n_skip = n_done = n_err = 0

    for i, q in enumerate(questions, 1):
        row = q["excel_row"]

        if _is_complete(row):
            p_json = OUT_DIR / f"decomp_q{row:03d}.json"
            try:
                new_decomp[row] = json.loads(p_json.read_text(encoding="utf-8"))
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
                                 json={"question": question, "rag_version": "v_decomp", "k": 5},
                                 headers=HEADERS, timeout=300)
            rag_elapsed = round(time.time() - t0, 1)
        except Exception as ex:
            if _is_hard_error(ex):
                print(f"\n[HARD STOP] Auth/quota RAG : {ex}", flush=True)
                sys.exit(1)
            entry.update({"rag_status": "exception", "rag_error": str(ex)})
            print(f"  [{i:3}/{n_q}] Q{row:3}  RAG EXCEPTION: {ex}", flush=True)
            (OUT_DIR / f"decomp_q{row:03d}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            n_err += 1
            continue

        if resp.status_code != 200:
            entry.update({"rag_status": "error", "rag_error": resp.text[:200]})
            print(f"  [{i:3}/{n_q}] Q{row:3}  RAG ERREUR {resp.status_code}", flush=True)
            (OUT_DIR / f"decomp_q{row:03d}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            n_err += 1
            continue

        data = resp.json()
        raw_sources   = data.get("sources", [])
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
            "sub_questions":  sub_questions,
            "n_subquestions": len(sub_questions),
            "n_sources":      len(raw_sources),
            "rag_elapsed_s":  rag_elapsed,
            "sources":        sources_for_judge,
        })

        (OUT_DIR / f"decomp_q{row:03d}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── Judge V4.3 ────────────────────────────────────────────────────────
        time.sleep(args.judge_delay)
        try:
            scores = judge_v43(question, entry["answer"], sources_for_judge,
                               q["section"], q["subsection"], _expected_type(q["section"]))
        except RuntimeError as quota_err:
            print(f"\n[HARD STOP] {quota_err}", flush=True)
            sys.exit(1)

        entry.update(scores)
        (OUT_DIR / f"decomp_q{row:03d}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

        new_decomp[row] = entry
        n_done += 1
        sg_str = f"{scores.get('score_global', '?'):.2f}" if isinstance(scores.get("score_global"), (int, float)) else "?"
        mis    = " [MIS]" if scores.get("mislabelling_flag") else ""
        print(
            f"  [{i:3}/{n_q}] Q{row:3}  V4.3={sg_str}{mis}"
            f"  {len(sub_questions)}SQ  RAG={rag_elapsed}s  {len(raw_sources)}src",
            flush=True,
        )

    # Charger les skips manqués
    for q in questions:
        row = q["excel_row"]
        if row not in new_decomp:
            p_json = OUT_DIR / f"decomp_q{row:03d}.json"
            if p_json.exists():
                try:
                    new_decomp[row] = json.loads(p_json.read_text(encoding="utf-8"))
                except Exception:
                    pass

    # ── Mise à jour COMPLET.json ──────────────────────────────────────────────
    n_scored = sum(1 for e_ in new_decomp.values() if isinstance(e_.get("score_global"), (int, float)))
    if n_scored == n_q:
        complet_raw["v_decomp"] = [new_decomp[r] for r in sorted(new_decomp)]
        COMPLET.write_text(json.dumps(complet_raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nCOMPLET.json mis à jour : v_decomp remplacé ({n_scored} entrées).", flush=True)
    else:
        print(f"\n[ATTENTION] Seulement {n_scored}/{n_q} scorées — COMPLET.json non mis à jour.", flush=True)

    # ── Récap console ─────────────────────────────────────────────────────────
    def gavg(key, edict):
        vals = [e_[key] for e_ in edict.values() if isinstance(e_.get(key), (int, float))]
        return f"{sum(vals)/len(vals):.2f}" if vals else "—"

    print(f"\n{'='*60}")
    for label, edict in [
        ("v_decomp (new)",          new_decomp),
        ("no_typing v1",            v1_data),
        ("no_typing v2",            v2_entries),
    ]:
        print(f"  {label:<28} global={gavg('score_global', edict)}"
              f"  Q/Q={gavg('coherence_qualiquanti', edict)}"
              f"  Nuance={gavg('nuance_incertitude', edict)}")

    build_report(new_decomp, v1_data, v2_entries, ts)


if __name__ == "__main__":
    main()
