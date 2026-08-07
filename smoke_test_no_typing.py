"""
smoke_test_no_typing.py — Smoke test pour v_decomp_no_typing sur 10 questions sélectionnées.

Questions choisies pour couvrir différentes catégories :
  Q001 — qualitative globale bien-être Ajaccio
  Q006 — factuelle OppChoVec par catégorie (lock quanti attendu sans typage)
  Q011 — sous-population seniors Ajaccio
  Q028 — comparative factuelle Ajaccio vs Bastia
  Q034 — crossing quali/quanti convergence (cas le plus intéressant)
  Q041 — contre-intuitive score élevé / mal évalué
  Q044 — absence totale données (Nice)
  Q048 — données partielles insuffisantes
  Q068 — raisonnement causal
  Q083 — limite de l'inférence causale

Usage :
    python smoke_test_no_typing.py
    python smoke_test_no_typing.py --html-only  # régénère le HTML depuis JSON existants
"""

import argparse, json, re, sys, time, requests, openpyxl
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

XLSX        = r"C:\Users\comiti_g\Downloads\rag_evaluation_with_metrics_full.xlsx"
API_BASE    = "http://localhost:8000/api/query"
RAG_VERSION = "v_decomp_no_typing"
K           = 5
OUT_DIR     = Path("smoke_test_no_typing")
CALL_DELAY  = 2.0          # secondes entre appels

TARGET_ROWS = {1, 6, 11, 28, 34, 41, 44, 48, 68, 83}   # 1-indexed question numbers

CATEGORY_LABELS = {
    1:  "Qualitative globale",
    6:  "Factuelle OppChoVec",
    11: "Sous-population seniors",
    28: "Comparative factuelle",
    34: "Crossing quali/quanti",
    41: "Contre-intuitive",
    44: "Absence totale données",
    48: "Données partielles",
    68: "Raisonnement causal",
    83: "Limite inférence causale",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_questions() -> List[Dict]:
    wb = openpyxl.load_workbook(XLSX)
    ws = wb.active
    questions = []
    for r in range(2, ws.max_row + 1):
        idx = r - 1
        if idx not in TARGET_ROWS:
            continue
        q = (ws.cell(r, 3).value or "").strip()
        if not q:
            continue
        questions.append({
            "excel_row":  idx,
            "section":    (ws.cell(r, 1).value or "").strip(),
            "subsection": (ws.cell(r, 2).value or "").strip(),
            "question":   q,
            "category":   CATEGORY_LABELS.get(idx, ""),
        })
    questions.sort(key=lambda x: x["excel_row"])
    return questions


def call_api(question: str, retries: int = 3) -> Dict:
    payload = {"question": question, "rag_version": RAG_VERSION, "k": K}
    for attempt in range(retries):
        try:
            r = requests.post(API_BASE, json=payload, timeout=300)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  [retry {attempt+1}/{retries}] {e} — attente {wait}s")
            if attempt < retries - 1:
                time.sleep(wait)
    return {"error": "echec après retries", "answer": "", "sources": []}


# ── Main run ──────────────────────────────────────────────────────────────────

def run_smoke_test(questions: List[Dict]) -> List[Dict]:
    OUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"\n{'='*60}")
    print(f"SMOKE TEST v_decomp_no_typing — {len(questions)} questions")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions, 1):
        row = q["excel_row"]
        print(f"[{i}/{len(questions)}] Q{row:03d} [{q['category']}]")
        print(f"  {q['question'][:90]}")

        t0 = time.time()
        resp = call_api(q["question"])
        elapsed = time.time() - t0

        if "error" in resp and not resp.get("answer"):
            print(f"  ECHEC : {resp.get('error')}")
            record = {**q, "answer": "", "sub_questions": [], "sources": [],
                      "sources_mobilisees": [], "elapsed_s": elapsed, "error": resp.get("error")}
        else:
            sub_qs = resp.get("sub_questions") or []
            sources = resp.get("sources") or []
            sources_mob = resp.get("sources_mobilisees") or []
            answer = resp.get("answer", "")
            print(f"  OK — {len(sub_qs)} sous-questions, {len(sources)} sources, {elapsed:.1f}s")
            record = {
                **q,
                "answer":             answer,
                "sub_questions":      sub_qs,
                "sources":            sources,
                "sources_mobilisees": sources_mob,
                "elapsed_s":          elapsed,
                "error":              None,
            }

        # Sauvegarde incrémentale
        out_path = OUT_DIR / f"m_nt_{ts}_{row}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        results.append(record)

        if i < len(questions):
            time.sleep(CALL_DELAY)

    print(f"\nRésultats sauvegardés dans {OUT_DIR}/")
    return results


# ── HTML report ───────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def generate_html(results: List[Dict]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sections_html = ""
    for rec in results:
        row      = rec.get("excel_row", "?")
        cat      = _esc(rec.get("category", ""))
        section  = _esc(rec.get("section", ""))
        sub_sec  = _esc(rec.get("subsection", ""))
        question = _esc(rec.get("question", ""))
        answer   = _esc(rec.get("answer", "") or "— (pas de réponse)")
        elapsed  = rec.get("elapsed_s", 0)
        err      = rec.get("error")

        sub_qs      = rec.get("sub_questions") or []
        sources_mob = rec.get("sources_mobilisees") or []

        # Bâtir un dict sq_idx → types mobilisés
        mob_by_sq: Dict[int, List[str]] = {}
        for m in sources_mob:
            idx = m.get("sq", m.get("idx", 0))
            mob_by_sq[idx] = m.get("types", [])

        sub_qs_html = ""
        for sq in sub_qs:
            sq_idx  = sq.get("idx", "?")
            sq_q    = _esc(sq.get("question", ""))
            sq_ans  = _esc(sq.get("answer", "") or "—")
            types   = mob_by_sq.get(sq_idx, [])
            badge   = ""
            for t in types:
                badge += f'<span class="badge">{_esc(t)}</span>'

            sub_qs_html += f"""
<div class="subq">
  <div class="subq-header">
    <span class="subq-num">SQ{sq_idx}</span>
    <span class="subq-text">{sq_q}</span>
    {badge}
  </div>
  <div class="subq-answer">{sq_ans}</div>
</div>"""

        status_class = "error" if err else "ok"
        elapsed_str  = f"{elapsed:.1f}s"

        sections_html += f"""
<section class="question-block {status_class}">
  <div class="q-header">
    <span class="q-id">Q{row:03d}</span>
    <span class="q-cat">{cat}</span>
    <span class="q-elapsed">{elapsed_str}</span>
  </div>
  <div class="q-meta">{section} › {sub_sec}</div>
  <div class="q-text">{question}</div>

  {'<div class="error-box">ERREUR : ' + _esc(err) + '</div>' if err else ''}

  <details class="subq-block" {"open" if not err else ""}>
    <summary>Sous-questions ({len(sub_qs)})</summary>
    {sub_qs_html if sub_qs_html else '<p class="empty">Aucune sous-question</p>'}
  </details>

  <details class="answer-block">
    <summary>Réponse finale de synthèse</summary>
    <div class="answer-text">{answer}</div>
  </details>
</section>"""

    n_ok  = sum(1 for r in results if not r.get("error"))
    n_tot = len(results)

    html = f"""<title>Smoke test v_decomp_no_typing</title>
<meta charset="utf-8">
<style>
  :root {{ --bg: #fff; --fg: #222; --border: #ddd; --subq-bg: #f7f7f7;
           --badge: #8e44ad; --ok: #27ae60; --err: #c0392b; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #1a1a1a; --fg: #e0e0e0; --border: #333; --subq-bg: #252525; }}
  }}
  :root[data-theme="dark"]  {{ --bg: #1a1a1a; --fg: #e0e0e0; --border: #333; --subq-bg: #252525; }}
  :root[data-theme="light"] {{ --bg: #fff; --fg: #222; --border: #ddd; --subq-bg: #f7f7f7; }}
  body {{ background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif;
          line-height: 1.55; max-width: 900px; margin: 0 auto; padding: 1.5rem 1rem; }}
  h1 {{ font-size: 1.4rem; border-bottom: 2px solid var(--badge); padding-bottom: .4rem; }}
  .meta {{ font-size: .85rem; color: #888; margin-bottom: 1.5rem; }}
  .question-block {{ border: 1px solid var(--border); border-radius: 8px; padding: 1rem;
                     margin-bottom: 1.4rem; }}
  .question-block.error {{ border-left: 4px solid var(--err); }}
  .question-block.ok    {{ border-left: 4px solid var(--ok); }}
  .q-header {{ display: flex; gap: .6rem; align-items: center; margin-bottom: .3rem; }}
  .q-id    {{ background: var(--badge); color: #fff; border-radius: 4px;
               padding: 2px 7px; font-weight: bold; font-size: .85rem; }}
  .q-cat   {{ background: #eee; color: #555; border-radius: 4px;
               padding: 2px 7px; font-size: .8rem; }}
  @media (prefers-color-scheme: dark) {{ .q-cat {{ background: #333; color: #aaa; }} }}
  .q-elapsed {{ margin-left: auto; font-size: .8rem; color: #888; }}
  .q-meta  {{ font-size: .8rem; color: #888; margin-bottom: .5rem; }}
  .q-text  {{ font-weight: 600; margin-bottom: .8rem; }}
  .error-box {{ background: #fdecea; border: 1px solid var(--err); border-radius: 4px;
                padding: .5rem .8rem; color: var(--err); font-size: .9rem; margin-bottom: .7rem; }}
  details {{ margin-top: .5rem; }}
  summary  {{ cursor: pointer; font-weight: 600; font-size: .95rem; padding: .3rem 0;
               color: var(--badge); user-select: none; }}
  summary:hover {{ opacity: .8; }}
  .subq       {{ background: var(--subq-bg); border-radius: 6px; padding: .7rem .9rem;
                 margin-top: .6rem; }}
  .subq-header {{ display: flex; align-items: flex-start; gap: .5rem; flex-wrap: wrap;
                   margin-bottom: .4rem; }}
  .subq-num   {{ background: #555; color: #fff; border-radius: 4px;
                  padding: 1px 6px; font-size: .8rem; flex-shrink: 0; }}
  .subq-text  {{ font-weight: 500; flex: 1; }}
  .badge      {{ background: var(--badge); color: #fff; border-radius: 12px;
                  padding: 1px 8px; font-size: .75rem; white-space: nowrap; }}
  .subq-answer {{ font-size: .9rem; color: var(--fg); opacity: .85; white-space: pre-wrap; }}
  .answer-block .answer-text {{ white-space: pre-wrap; font-size: .9rem; margin-top: .5rem; }}
  .empty {{ color: #888; font-style: italic; }}
</style>
<h1>Smoke test — <code>v_decomp_no_typing</code></h1>
<div class="meta">
  Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")} · {n_ok}/{n_tot} questions OK ·
  RAG version : {RAG_VERSION} · k={K}
</div>
{sections_html}
"""

    out_path = OUT_DIR / f"smoke_report_no_typing_{ts}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nRapport HTML : {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-only", action="store_true",
                        help="Régénère le HTML depuis les JSON existants (pas d'appels API)")
    args = parser.parse_args()

    questions = load_questions()
    print(f"{len(questions)} questions chargées depuis Excel.")

    if args.html_only:
        # Charger tous les JSON existants
        json_files = sorted(OUT_DIR.glob("m_nt_*.json"))
        if not json_files:
            sys.exit(f"Aucun JSON trouvé dans {OUT_DIR}/")
        results = []
        for f in json_files:
            with open(f, encoding="utf-8") as fh:
                results.append(json.load(fh))
        results.sort(key=lambda x: x.get("excel_row", 0))
        print(f"{len(results)} résultats chargés depuis {OUT_DIR}/")
    else:
        # Vérifier que le serveur répond
        try:
            r = requests.get("http://localhost:8000/", timeout=5)
            r.raise_for_status()
        except Exception as e:
            sys.exit(f"BLOQUANT — serveur RAG inaccessible : {e}\n"
                     "Démarrez le serveur avant de lancer ce script.")
        results = run_smoke_test(questions)

    html_path = generate_html(results)
    print(f"Ouvrir : {html_path.resolve()}")


if __name__ == "__main__":
    main()
