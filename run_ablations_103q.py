"""
run_ablations_103q.py — Évaluation complète : 103 questions × 4 configs RAG + judge V4.3 (GPT-4o).

- Charge les questions depuis rag_evaluation_with_metrics_full.xlsx
- Appelle le serveur RAG pour chaque question × config
- Applique le judge V4.3 (GPT-4o) sur chaque réponse
- Sauvegarde un checkpoint JSON après chaque config
- Arrêt propre sur rate limit persistant (quota épuisé) — relançable avec --resume

Usage:
    python run_ablations_103q.py                    # toutes les 103 questions
    python run_ablations_103q.py --max 10           # test sur 10 questions
    python run_ablations_103q.py --resume results.json   # reprise depuis checkpoint
"""
import argparse, json, re, sys, time, requests, openpyxl
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

# ── Forcer GPT-4o comme juge avant d'importer eval_from_excel ─────────────
import importlib
import eval_from_excel as evmod

evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None  # force re-init avec la nouvelle config

from eval_from_excel import _JUDGE_V43_SYSTEM, _parse_judge_v43, _build_sources_text, _call_llm

XLSX      = r"C:\Users\comiti_g\Downloads\rag_evaluation_with_metrics_full.xlsx"
BASE      = "http://localhost:8000/api/query"
HEADERS   = {"Content-Type": "application/json"}
VERSIONS  = ["v_vanilla_k10", "v_vanilla_k25", "v_decomp", "v_decomp_raptor"]
OUT_DIR   = Path("comparaisons_rag")
JUDGE_DELAY = 1.0  # secondes entre appels judge


def _expected_type(section: str) -> str:
    s = (section or "").lower()
    if "limite" in s and "architect" in s:
        return "limite_architecturale"
    if "absence" in s and "information" in s:
        return "reponse_substantielle_attendue"
    return "reponse_substantielle_attendue"


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
        # Booléen synthétique pour les affichages/filtres
        if "mislabelling_flag" not in result:
            result["mislabelling_flag"] = any(
                str(v).lower() not in ("non", "false", "", "null", "none")
                for v in result.get("mislabelling_detecte", {}).values()
            )
        return result
    except Exception as e:
        err = str(e)
        # Quota épuisé → lever pour arrêt propre
        if "insufficient_quota" in err or "quota" in err.lower():
            raise RuntimeError(f"QUOTA ÉPUISÉ: {err}")
        return {"judge_error": err, "score_global": None}


def load_questions(max_q: int | None = None) -> list:
    wb = openpyxl.load_workbook(XLSX)
    ws = wb.active
    rows = []
    for r in range(2, ws.max_row + 1):
        section   = ws.cell(r, 1).value or ""
        subsection = ws.cell(r, 2).value or ""
        question  = ws.cell(r, 3).value or ""
        if not question.strip():
            continue
        rows.append({
            "excel_row":  r - 1,   # 1-based question index
            "section":    section,
            "subsection": subsection,
            "question":   question,
        })
    return rows[:max_q] if max_q else rows


def save_checkpoint(results: dict, ts: str, n_q: int, label: str = "") -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    p = OUT_DIR / f"ablations_{n_q}q_v43_gpt4o_{ts}{label}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [SAUVEGARDE] {p}", flush=True)
    return p


def generate_html(results: dict, ts: str, n_q: int) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    versions = list(results.keys())
    _COLORS = {
        "v_vanilla_k10":   "#c0392b",
        "v_vanilla_k25":   "#e67e22",
        "v_decomp":        "#27ae60",
        "v_decomp_raptor": "#2980b9",
    }
    dims = ["pertinence", "fondement_factuel", "nuance_incertitude",
            "coherence_qualiquanti", "score_global"]
    dim_labels = ["Pertinence", "Factuel", "Nuance", "Quali/Q", "Global"]

    def avg(key, entries):
        vals = [e[key] for e in entries if isinstance(e.get(key), (int, float))]
        return f"{sum(vals)/len(vals):.2f}" if vals else "-"

    # Résumé par config
    summary_rows = ""
    for ver in versions:
        ok = [e for e in results[ver] if e.get("rag_status") == "ok" and not e.get("judge_error")]
        color = _COLORS.get(ver, "#555")
        cells = "".join(f"<td>{avg(d, ok)}</td>" for d in dims)
        summary_rows += f"<tr><td style='color:{color};font-weight:bold'>{ver}</td><td>{len(ok)}/{len(results[ver])}</td>{cells}</tr>\n"

    # Tableau détaillé
    detail_rows = ""
    all_entries = []
    for ver in versions:
        for e in results[ver]:
            all_entries.append((ver, e))

    for ver, e in all_entries:
        color = _COLORS.get(ver, "#555")
        q_short = (e.get("question") or "")[:60]
        status = e.get("rag_status", "?")
        if status != "ok":
            detail_rows += (f"<tr><td style='color:{color}'>{ver}</td><td>{e.get('excel_row','')}</td>"
                            f"<td>{q_short}</td><td colspan=6 style='color:red'>{status}: {e.get('rag_error','')[:60]}</td></tr>\n")
            continue
        sg  = e.get("score_global")
        sg_str = f"{sg:.2f}" if isinstance(sg, (int,float)) else "?"
        mis = "✓" if e.get("mislabelling_flag") else ""
        detail_rows += (
            f"<tr>"
            f"<td style='color:{color}'>{ver}</td>"
            f"<td>{e.get('excel_row','')}</td>"
            f"<td title='{e.get('question','')}' style='max-width:350px;overflow:hidden;white-space:nowrap'>{q_short}</td>"
            f"<td>{e.get('section','')[:20]}</td>"
            f"<td>{avg('pertinence',[e])}</td>"
            f"<td>{avg('fondement_factuel',[e])}</td>"
            f"<td>{avg('nuance_incertitude',[e])}</td>"
            f"<td>{avg('coherence_qualiquanti',[e])}</td>"
            f"<td><b>{sg_str}</b></td>"
            f"<td style='color:#c0392b'>{mis}</td>"
            f"<td style='font-size:0.85em;color:#555'>{e.get('raisonnement_v43','')[:80]}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Ablations RAG — {n_q}q — V4.3 GPT-4o — {ts}</title>
<style>
  body{{font-family:sans-serif;font-size:13px;margin:20px}}
  table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #ddd;padding:4px 7px;vertical-align:top}}
  th{{background:#f5f5f5;position:sticky;top:0}}
  tr:hover{{background:#f9f9f9}}
  .sum{{width:60%;margin-bottom:30px}}
</style>
</head><body>
<h2>Ablations RAG — {n_q} questions — Judge V4.3 (GPT-4o) — {ts}</h2>

<h3>Résumé par config (moyennes V4.3)</h3>
<table class="sum">
<tr><th>Config</th><th>N OK</th>{''.join(f'<th>{l}</th>' for l in dim_labels)}</tr>
{summary_rows}
</table>

<h3>Détail par question</h3>
<table>
<tr>
  <th>Config</th><th>#</th><th>Question</th><th>Section</th>
  <th>Pert.</th><th>Fact.</th><th>Nuance</th><th>Q/Q</th><th>Global</th>
  <th>Mis.</th><th>Raisonnement</th>
</tr>
{detail_rows}
</table>
<p style="color:#888;font-size:0.9em">Généré le {ts} | Juge : GPT-4o + V4.3 | Mis. = mislabelling détecté</p>
</body></html>"""

    p = OUT_DIR / f"ablations_{n_q}q_v43_gpt4o_{ts}.html"
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML → {p}", flush=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Nb de questions (défaut: toutes)")
    parser.add_argument("--judge-delay", type=float, default=JUDGE_DELAY)
    parser.add_argument("--versions", nargs="+", default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Reprendre depuis un fichier JSON checkpoint")
    args = parser.parse_args()

    questions = load_questions(args.max)
    versions  = args.versions if args.versions else VERSIONS
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_q       = len(questions)

    # ── Reprise depuis checkpoint ─────────────────────────────────────────
    results: dict = {}
    if args.resume:
        with open(args.resume, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[REPRISE] {args.resume} — configs déjà présentes: {list(results.keys())}")

    print(f"\n{n_q} questions × {len(versions)} configs = {n_q*len(versions)} appels RAG + judge", flush=True)
    print(f"Juge : GPT-4o + V4.3\n")

    for ver in versions:
        n_ok_existing = sum(1 for e in results.get(ver, []) if e.get("rag_status") == "ok")
        if ver in results and n_ok_existing >= n_q:
            print(f"[SKIP] {ver} déjà complète ({len(results[ver])} entrées)", flush=True)
            continue

        k = 10 if ver == "v_vanilla_k10" else (25 if ver == "v_vanilla_k25" else 5)
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {ver}  (k={k})", flush=True)
        print('='*60, flush=True)

        # Garder les entrées déjà calculées si reprise partielle (succès uniquement)
        done_rows = {e["excel_row"] for e in results.get(ver, []) if e.get("rag_status") == "ok"}
        config_results = [e for e in results.get(ver, []) if e.get("rag_status") == "ok"]

        try:
            for i, q in enumerate(questions, 1):
                if q["excel_row"] in done_rows:
                    print(f"  [{i:3}/{n_q}] Q{q['excel_row']:3}  SKIP (déjà calculé)", flush=True)
                    continue

                question = q["question"]
                etype    = _expected_type(q["section"])
                entry    = {
                    "excel_row":  q["excel_row"],
                    "section":    q["section"],
                    "subsection": q["subsection"],
                    "question":   question,
                }

                # 1. Appel RAG
                try:
                    t0 = time.time()
                    resp = requests.post(BASE,
                                         json={"question": question, "rag_version": ver, "k": k},
                                         headers=HEADERS, timeout=300)
                    rag_elapsed = time.time() - t0

                    if resp.status_code != 200:
                        entry.update({"rag_status": "error", "rag_error": resp.text[:200]})
                        print(f"  [{i:3}/{n_q}] Q{q['excel_row']:3}  RAG ERREUR {resp.status_code}", flush=True)
                        config_results.append(entry)
                        continue

                    data = resp.json()
                    raw_sources = data.get("sources", [])
                    sources_for_judge = [
                        {"content":   s.get("content") or s.get("extrait") or "",
                         "metadata":  s.get("metadata", {}),
                         "source_type": s.get("source_type", ""),
                         "label":     s.get("label", "")}
                        for s in raw_sources
                    ]
                    entry.update({
                        "rag_status":     "ok",
                        "answer":         data.get("answer", ""),
                        "n_sources":      len(raw_sources),
                        "n_subquestions": len(data.get("sub_questions") or []),
                        "rag_elapsed_s":  round(rag_elapsed, 1),
                        "sources":        sources_for_judge,
                    })
                except Exception as e:
                    entry.update({"rag_status": "exception", "rag_error": str(e)})
                    print(f"  [{i:3}/{n_q}] Q{q['excel_row']:3}  RAG EXCEPTION: {e}", flush=True)
                    config_results.append(entry)
                    continue

                # 2. Judge V4.3
                time.sleep(args.judge_delay)
                t0 = time.time()
                scores = judge_v43(question, entry["answer"], sources_for_judge,
                                   q["section"], q["subsection"], etype)
                judge_elapsed = round(time.time() - t0, 1)
                entry.update(scores)
                entry["judge_elapsed_s"] = judge_elapsed

                sg = scores.get("score_global")
                sg_str = f"{sg:.2f}" if isinstance(sg, (int, float)) else "?"
                mis = " [MIS]" if scores.get("mislabelling_flag") else ""
                print(
                    f"  [{i:3}/{n_q}] Q{q['excel_row']:3}  V4.3={sg_str}{mis}"
                    f"  RAG={entry['rag_elapsed_s']}s  judge={judge_elapsed}s"
                    f"  {len(raw_sources)}src",
                    flush=True,
                )
                config_results.append(entry)

        except RuntimeError as quota_err:
            # Quota épuisé : sauvegarder et lever une alerte
            results[ver] = config_results
            save_checkpoint(results, ts, n_q, "_CHECKPOINT_QUOTA")
            print(f"\n{'!'*60}", flush=True)
            print(f"QUOTA ÉPUISÉ — arrêt propre.", flush=True)
            print(f"Relancer avec : python run_ablations_103q.py --resume {OUT_DIR}/ablations_{n_q}q_v43_gpt4o_{ts}_CHECKPOINT_QUOTA.json", flush=True)
            print(f"{'!'*60}", flush=True)
            sys.exit(1)

        results[ver] = config_results
        save_checkpoint(results, ts, n_q)

    # ── Tableau récap ────────────────────────────────────────────────────────
    dims   = ["pertinence","fondement_factuel","nuance_incertitude","coherence_qualiquanti","score_global"]
    labels = ["Pertinence","Factuel","Nuance","Quali/Q","Global"]

    print(f"\n{'─'*75}")
    print(f"RÉSULTATS V4.3 (GPT-4o) — {n_q} questions")
    print(f"{'─'*75}")
    print(f"{'Config':<22} {'N OK':>5}  " + "  ".join(f"{l:>8}" for l in labels))
    print("─"*75)

    for ver in versions:
        ok = [e for e in results[ver] if e.get("rag_status") == "ok" and not e.get("judge_error")]
        def avg(key):
            vals = [e[key] for e in ok if isinstance(e.get(key), (int,float))]
            return f"{sum(vals)/len(vals):.2f}" if vals else "-"
        print(f"  {ver:<20} {len(ok):>5}  " + "  ".join(f"{avg(d):>8}" for d in dims))

    # Mislabelling stats
    print(f"\n{'─'*40}")
    print("Mislabelling détecté par config:")
    for ver in versions:
        ok = [e for e in results[ver] if e.get("rag_status") == "ok"]
        n_mis = sum(1 for e in ok if e.get("mislabelling_flag"))
        print(f"  {ver:<22}: {n_mis}/{len(ok)}")

    # ── Sauvegarde finale ────────────────────────────────────────────────────
    final_json = save_checkpoint(results, ts, n_q, "_FINAL")
    generate_html(results, ts, n_q)
    print(f"\nTerminé. JSON final → {final_json}")


if __name__ == "__main__":
    main()
