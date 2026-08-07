"""
smoke_test_compare.py — Compare v_decomp (originel) vs v_decomp_no_typing sur les 10 mêmes questions.

Lit les JSON no_typing depuis smoke_test_no_typing/ et relance v_decomp.
Génère un rapport HTML de comparaison côte-à-côte.

Usage :
    python smoke_test_compare.py             # appels v_decomp + génère HTML
    python smoke_test_compare.py --html-only # régénère HTML depuis JSON existants
"""

import argparse, json, sys, time, requests, openpyxl
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.stdout.reconfigure(encoding="utf-8")

XLSX         = r"C:\Users\comiti_g\Downloads\rag_evaluation_with_metrics_full.xlsx"
API_BASE     = "http://localhost:8000/api/query"
K            = 5
CALL_DELAY   = 2.0

NT_DIR   = Path("smoke_test_no_typing")
CMP_DIR  = Path("smoke_test_compare")

TARGET_ROWS = {1, 6, 11, 28, 34, 41, 44, 48, 68, 83}
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
    return sorted(questions, key=lambda x: x["excel_row"])


def call_api(question: str, rag_version: str, retries: int = 3) -> Dict:
    payload = {"question": question, "rag_version": rag_version, "k": K}
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


def run_decomp(questions: List[Dict]) -> List[Dict]:
    CMP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"\n{'='*60}")
    print(f"SMOKE TEST v_decomp (originel) — {len(questions)} questions")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions, 1):
        row = q["excel_row"]
        print(f"[{i}/{len(questions)}] Q{row:03d} [{q['category']}]")
        print(f"  {q['question'][:90]}")

        t0 = time.time()
        resp = call_api(q["question"], "v_decomp")
        elapsed = time.time() - t0

        if "error" in resp and not resp.get("answer"):
            print(f"  ECHEC : {resp.get('error')}")
            record = {**q, "answer": "", "sub_questions": [], "sources": [],
                      "sources_mobilisees": [], "elapsed_s": elapsed, "error": resp.get("error")}
        else:
            sub_qs  = resp.get("sub_questions") or []
            sources = resp.get("sources") or []
            s_mob   = resp.get("sources_mobilisees") or []
            answer  = resp.get("answer", "")
            print(f"  OK — {len(sub_qs)} SQ, {len(sources)} sources, {elapsed:.1f}s")
            record = {
                **q,
                "answer":             answer,
                "sub_questions":      sub_qs,
                "sources":            sources,
                "sources_mobilisees": s_mob,
                "elapsed_s":          elapsed,
                "error":              None,
            }

        out_path = CMP_DIR / f"m_decomp_{ts}_{row}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        results.append(record)
        if i < len(questions):
            time.sleep(CALL_DELAY)

    return results


def load_json_dir(directory: Path, prefix: str) -> List[Dict]:
    files = sorted(directory.glob(f"{prefix}_*.json"))
    if not files:
        return []
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            records.append(json.load(fh))
    return sorted(records, key=lambda r: r.get("excel_row", 0))


# ── HTML ──────────────────────────────────────────────────────────────────────

def esc(s: str) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")


def sq_panel(sub_qs, sources_mob, version: str) -> str:
    mob_by_sq: Dict[int, List[str]] = {}
    for m in (sources_mob or []):
        idx = m.get("sq", m.get("idx", 0))
        mob_by_sq[idx] = m.get("types", [])

    if not sub_qs:
        return '<p class="empty">Aucune sous-question.</p>'

    color_class = "nt" if "no_typing" in version else "orig"
    html = ""
    for sq in sub_qs:
        sq_idx = sq.get("idx", "?")
        sq_q   = esc(sq.get("question", ""))
        sq_ans = esc((sq.get("answer") or "—").strip())
        types  = mob_by_sq.get(sq_idx, [])
        badges = "".join(f'<span class="badge {color_class}">{esc(t)}</span>' for t in types)
        html += f"""
<div class="sq-item">
  <div class="sq-top">
    <span class="sq-num">SQ{sq_idx}</span>
    <span class="sq-q">{sq_q}</span>
  </div>
  {('<div class="sq-badges">' + badges + '</div>') if badges else ''}
  <details class="sq-det">
    <summary>Réponse intermédiaire</summary>
    <div class="sq-ans">{sq_ans}</div>
  </details>
</div>"""
    return html


def generate_compare_html(orig_recs: List[Dict], nt_recs: List[Dict]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    orig_by_row = {r["excel_row"]: r for r in orig_recs}
    nt_by_row   = {r["excel_row"]: r for r in nt_recs}
    all_rows    = sorted(set(orig_by_row) | set(nt_by_row))

    blocks = ""
    for row in all_rows:
        orig = orig_by_row.get(row, {})
        nt   = nt_by_row.get(row, {})
        rec  = orig or nt

        cat     = esc(rec.get("category", ""))
        section = esc(rec.get("section", ""))
        sub_s   = esc(rec.get("subsection", ""))
        quest   = esc(rec.get("question", ""))

        orig_sqs = orig.get("sub_questions") or []
        nt_sqs   = nt.get("sub_questions") or []
        orig_el  = f"{orig.get('elapsed_s', 0):.1f}s"
        nt_el    = f"{nt.get('elapsed_s', 0):.1f}s"

        orig_panel = sq_panel(orig_sqs, orig.get("sources_mobilisees"), "v_decomp")
        nt_panel   = sq_panel(nt_sqs, nt.get("sources_mobilisees"), "v_decomp_no_typing")

        orig_ans = esc((orig.get("answer") or "").strip())
        nt_ans   = esc((nt.get("answer") or "").strip())

        blocks += f"""
<section class="q-block" id="q{row}">
  <div class="q-wm">Q{row:03d}</div>
  <header class="q-head">
    <div class="q-id-row">
      <span class="q-id">Q{row:03d}</span>
      <span class="q-cat">{cat}</span>
    </div>
    <div class="q-path">{section} › {sub_s}</div>
    <h2 class="q-text">{quest}</h2>
  </header>

  <div class="compare-grid">
    <div class="pane pane-orig">
      <div class="pane-label">v_decomp <span class="pane-elapsed">{orig_el}</span></div>
      <div class="sq-list">{orig_panel}</div>
      <details class="ans-block">
        <summary>Réponse finale</summary>
        <div class="ans-body">{orig_ans}</div>
      </details>
    </div>

    <div class="pane pane-nt">
      <div class="pane-label">v_decomp_no_typing <span class="pane-elapsed">{nt_el}</span></div>
      <div class="sq-list">{nt_panel}</div>
      <details class="ans-block">
        <summary>Réponse finale</summary>
        <div class="ans-body">{nt_ans}</div>
      </details>
    </div>
  </div>
</section>"""

    n_ok_orig = sum(1 for r in orig_recs if not r.get("error"))
    n_ok_nt   = sum(1 for r in nt_recs  if not r.get("error"))
    gen_time  = datetime.now().strftime("%Y-%m-%d %H:%M")

    css = """
:root {
  --bg:         #f0f2f8;
  --surface:    #ffffff;
  --text:       #1e1b4b;
  --muted:      #5b5f7a;
  --border:     #dde1ee;
  --orig:       #2563eb;
  --orig-dim:   #eff6ff;
  --orig-bdr:   #bfdbfe;
  --nt:         #6d28d9;
  --nt-dim:     #ede9fe;
  --nt-bdr:     #ddd6fe;
  --sq-bg:      #f8f8fd;
  --wm:         rgba(99,102,241,.06);
  --mono:       "SF Mono","Cascadia Code","Consolas",monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:       #0e1020; --surface: #181b2e; --text: #e8eaf6; --muted: #8b90b8;
    --border:   #2a2f50;
    --orig:     #60a5fa; --orig-dim: #1e3050; --orig-bdr: #1d4070;
    --nt:       #a78bfa; --nt-dim:   #1e1a40; --nt-bdr:   #312e6a;
    --sq-bg:    #1a1d35; --wm: rgba(167,139,250,.07);
  }
}
:root[data-theme="dark"] {
  --bg:#0e1020;--surface:#181b2e;--text:#e8eaf6;--muted:#8b90b8;--border:#2a2f50;
  --orig:#60a5fa;--orig-dim:#1e3050;--orig-bdr:#1d4070;
  --nt:#a78bfa;--nt-dim:#1e1a40;--nt-bdr:#312e6a;
  --sq-bg:#1a1d35;--wm:rgba(167,139,250,.07);
}
:root[data-theme="light"] {
  --bg:#f0f2f8;--surface:#ffffff;--text:#1e1b4b;--muted:#5b5f7a;--border:#dde1ee;
  --orig:#2563eb;--orig-dim:#eff6ff;--orig-bdr:#bfdbfe;
  --nt:#6d28d9;--nt-dim:#ede9fe;--nt-bdr:#ddd6fe;
  --sq-bg:#f8f8fd;--wm:rgba(99,102,241,.06);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14.5px; line-height: 1.6;
  max-width: 1180px; margin: 0 auto; padding: 2rem 1.25rem 4rem;
}
/* Header */
.report-header { margin-bottom: 2.5rem; padding-bottom: 1.2rem; border-bottom: 2px solid var(--nt); }
.eyebrow {
  font-family: var(--mono); font-size: .72rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--nt); margin-bottom: .4rem;
}
.report-title { font-size: 1.45rem; font-weight: 700; line-height: 1.25; margin-bottom: .6rem; }
.report-title .v-orig { color: var(--orig); font-family: var(--mono); font-size: 1.2rem; }
.report-title .v-nt   { color: var(--nt);   font-family: var(--mono); font-size: 1.2rem; }
.report-meta { display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: .82rem; color: var(--muted); }
/* Legend pills */
.legend { display: flex; gap: .75rem; align-items: center; margin-top: .6rem; flex-wrap: wrap; }
.legend-pill {
  font-size: .75rem; font-weight: 600; font-family: var(--mono);
  padding: 3px 10px; border-radius: 20px;
}
.legend-pill.orig { background: var(--orig-dim); color: var(--orig); border: 1px solid var(--orig-bdr); }
.legend-pill.nt   { background: var(--nt-dim);   color: var(--nt);   border: 1px solid var(--nt-bdr); }
.legend-pill.nt::after { content: " — pas de contrainte de type"; font-weight: 400; }
/* Question blocks */
.q-block {
  position: relative; background: var(--surface);
  border: 1px solid var(--border); border-radius: 10px;
  margin-bottom: 2.2rem; overflow: hidden;
}
.q-wm {
  position: absolute; top: .15rem; right: .6rem;
  font-size: 3.8rem; font-weight: 900; color: var(--wm);
  pointer-events: none; user-select: none;
  font-variant-numeric: tabular-nums; line-height: 1;
}
.q-head { padding: 1rem 1.25rem .75rem; border-bottom: 1px solid var(--border); }
.q-id-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .35rem; }
.q-id {
  font-family: var(--mono); font-size: .75rem; font-weight: 700;
  background: var(--nt); color: #fff; padding: 2px 8px; border-radius: 4px;
  letter-spacing: .04em;
}
.q-cat {
  font-size: .75rem; background: var(--nt-dim); color: var(--nt);
  padding: 2px 8px; border-radius: 4px; font-weight: 500;
}
.q-path { font-size: .76rem; color: var(--muted); margin-bottom: .4rem; }
.q-text { font-size: 1rem; font-weight: 600; line-height: 1.4; text-wrap: balance; }
/* Compare grid */
.compare-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
}
@media (max-width: 720px) { .compare-grid { grid-template-columns: 1fr; } }
.pane {
  padding: .85rem 1rem 1rem;
  min-width: 0;
}
.pane-orig { border-right: 1px solid var(--border); }
.pane-label {
  font-family: var(--mono); font-size: .72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em;
  margin-bottom: .65rem; display: flex; align-items: baseline; gap: .5rem;
}
.pane-orig .pane-label { color: var(--orig); }
.pane-nt   .pane-label { color: var(--nt); }
.pane-elapsed { font-weight: 400; font-size: .7rem; color: var(--muted); letter-spacing: 0; }
/* Sub-questions */
.sq-item {
  background: var(--sq-bg); border-radius: 6px;
  padding: .6rem .75rem; margin-bottom: .5rem;
}
.sq-top { display: flex; align-items: flex-start; gap: .45rem; margin-bottom: .3rem; }
.sq-num {
  font-family: var(--mono); font-size: .68rem; font-weight: 700;
  background: var(--border); color: var(--muted);
  padding: 1px 5px; border-radius: 3px; flex-shrink: 0; margin-top: .15rem;
}
.sq-q { font-size: .86rem; font-weight: 500; line-height: 1.4; color: var(--text); }
.sq-badges { display: flex; flex-wrap: wrap; gap: .3rem; margin: .3rem 0 .2rem 1.5rem; }
.badge {
  font-size: .68rem; padding: 1px 7px; border-radius: 20px; white-space: nowrap; font-weight: 500;
}
.badge.orig { background: var(--orig-dim); color: var(--orig); border: 1px solid var(--orig-bdr); }
.badge.nt   { background: var(--nt-dim);   color: var(--nt);   border: 1px solid var(--nt-bdr); }
details.sq-det { margin-left: 1.5rem; }
details.sq-det > summary {
  cursor: pointer; font-size: .75rem; color: var(--muted); font-weight: 600;
  user-select: none; list-style: none; padding: .15rem 0;
  display: flex; align-items: center; gap: .3rem;
}
details.sq-det > summary::before { content: "▶"; font-size: .55rem; transition: transform .12s; }
details.sq-det[open] > summary::before { transform: rotate(90deg); }
.sq-ans {
  font-size: .82rem; line-height: 1.6; white-space: pre-wrap;
  margin-top: .4rem; padding-top: .4rem; border-top: 1px solid var(--border);
  color: var(--text);
}
/* Final answer */
.ans-block {
  border-top: 1px solid var(--border); margin-top: .4rem; padding-top: .6rem;
}
.ans-block > summary {
  cursor: pointer; font-size: .8rem; font-weight: 600;
  user-select: none; list-style: none;
  display: flex; align-items: center; gap: .3rem;
}
.pane-orig .ans-block > summary { color: var(--orig); }
.pane-nt   .ans-block > summary { color: var(--nt); }
.ans-block > summary::before { content: "▶"; font-size: .55rem; transition: transform .12s; }
.ans-block[open] > summary::before { transform: rotate(90deg); }
.ans-body {
  white-space: pre-wrap; font-size: .83rem; line-height: 1.65;
  margin-top: .6rem; color: var(--text);
}
.empty { color: var(--muted); font-style: italic; font-size: .85rem; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comparaison v_decomp vs v_decomp_no_typing</title>
<style>{css}</style>
</head>
<body>
<header class="report-header">
  <div class="eyebrow">RAG Ablation Study · Comparaison sous-questions</div>
  <h1 class="report-title">
    <span class="v-orig">v_decomp</span>
    &nbsp;vs&nbsp;
    <span class="v-nt">v_decomp_no_typing</span>
    &nbsp;— 10 questions
  </h1>
  <div class="report-meta">
    <span>v_decomp : {n_ok_orig}/{len(orig_recs)} OK</span>
    <span>v_decomp_no_typing : {n_ok_nt}/{len(nt_recs)} OK</span>
    <span>Généré le {gen_time} · k={K}</span>
  </div>
  <div class="legend">
    <span class="legend-pill orig">v_decomp</span>
    <span class="legend-pill nt">v_decomp_no_typing</span>
  </div>
</header>
{blocks}
</body>
</html>"""

    out = CMP_DIR / f"compare_decomp_vs_notyping_{ts}.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nRapport HTML : {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    questions = load_questions()
    print(f"{len(questions)} questions chargées.")

    # Charger les résultats no_typing (existants)
    nt_recs = load_json_dir(NT_DIR, "m_nt")
    print(f"{len(nt_recs)} résultats no_typing chargés depuis {NT_DIR}/")

    if args.html_only:
        orig_recs = load_json_dir(CMP_DIR, "m_decomp")
        if not orig_recs:
            sys.exit(f"Aucun JSON v_decomp trouvé dans {CMP_DIR}/")
        print(f"{len(orig_recs)} résultats v_decomp chargés.")
    else:
        try:
            r = requests.get("http://localhost:8000/", timeout=5)
            r.raise_for_status()
        except Exception as e:
            sys.exit(f"BLOQUANT — serveur RAG inaccessible : {e}")
        orig_recs = run_decomp(questions)

    out = generate_compare_html(orig_recs, nt_recs)
    print(f"Ouvrir : {out.resolve()}")


if __name__ == "__main__":
    main()
