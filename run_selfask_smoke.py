"""
run_selfask_smoke.py — Smoke test Self-Ask sur 2 questions.

  Q004 : objectif simple  ("Quel est le score moyen de bien-être à Ajaccio ?")
  Q002 : question vaste   ("Peut-on considérer Ajaccio comme un territoire favorable au bien-être ?")

Vérifie :
  - boucle séquentielle (Follow up → Intermediate answer → ...)
  - parsing du marqueur final ("So the final answer is:")
  - mapping catégoriel des sources
  - invocation juge V4.3 sur la réponse finale

Idempotent : une question déjà complète (score_global présent) est sautée.
Relance : python run_selfask_smoke.py
"""

import io, json, sys, time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from rag_selfask import SelfAskRAG
from eval_from_excel import score_judge_v43


# ── Affichage trace ───────────────────────────────────────────────────────────

def _print_trace(entry: dict):
    """Affiche la trace complète d'une entrée Self-Ask."""
    row = entry["excel_row"]
    hops = entry.get("hops", [])
    print(f"\n--- TRACE Q{row:03d} ---")
    print(f"  Hops effectués : {len(hops)}")

    for h in hops:
        print(f"\n  [Hop {h['hop']}] Follow up: {h['follow_up']}")
        srcs = h.get("sources", [])
        print(f"    Sources récupérées ({len(srcs)}) :")
        for s in srcs:
            meta = s.get("metadata", {})
            commune = meta.get("commune", "?")
            coll = s.get("source_type", "?")
            label = s.get("label", "")[:55]
            print(f"      • [{coll}] {commune} — {label}")
        ia = h.get("intermediate_answer", "")
        print(f"    Réponse intermédiaire :")
        print(f"      {ia[:300]}{'...' if len(ia) > 300 else ''}")

    fa = entry.get("final_answer", "")
    print(f"\n  Réponse finale ({entry.get('elapsed_rag_s', '?')}s RAG) :")
    print(f"    {fa[:600]}{'...' if len(fa) > 600 else ''}")

    all_src = entry.get("all_sources", [])
    print(f"\n  Sources agrégées ({len(all_src)} uniques) :")
    by_type: dict = {}
    for s in all_src:
        t = s.get("source_type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"      {t}: {n}")

    print(f"\n  Scores juge V4.3 ({entry.get('elapsed_judge_s', '?')}s) :")
    for dim in ("pertinence", "fondement_factuel", "nuance_incertitude", "coherence_qualiquanti"):
        print(f"    {dim:<30} {entry.get(dim)}")
    print(f"    {'score_global':<30} {entry.get('score_global')}")
    print(f"    {'mislabelling_flag':<30} {entry.get('mislabelling_flag')}")
    ml = entry.get("mislabelling_detecte", {})
    if ml and any(str(v).lower() not in ("non", "false", "", "null", "none")
                  for v in ml.values()):
        print(f"    mislabelling_detecte: {ml}")


# ── Cibles du smoke test ─────────────────────────────────────────────────────

SMOKE_TARGETS = [
    {
        "excel_row": 4,
        "section": "Retrieval mono-commune",
        "subsection": "Retrieval factuel et interprétation",
        "question": "Quel est le score moyen de bien-être à Ajaccio ?",
        "expected_type": "reponse_substantielle_attendue",
    },
    {
        "excel_row": 2,
        "section": "Retrieval mono-commune",
        "subsection": "Retrieval descriptif global",
        "question": "Peut-on considérer Ajaccio comme un territoire favorable au bien-être ?",
        "expected_type": "reponse_substantielle_attendue",
    },
]

OUT_DIR = Path("comparaisons_rag/selfask_12q")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Init pipeline ─────────────────────────────────────────────────────────────

print("Initialisation SelfAskRAG...")
pipeline = SelfAskRAG(max_hops=5, k=5)
pipeline.init()

# ── Boucle smoke test ─────────────────────────────────────────────────────────

for q in SMOKE_TARGETS:
    row = q["excel_row"]
    out_path = OUT_DIR / f"selfask_q{row:03d}.json"

    # Idempotence
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(existing.get("score_global"), (int, float)):
            print(f"\nQ{row:03d} déjà complète (score_global={existing['score_global']}) — skip.")
            _print_trace(existing)
            continue

    print(f"\n{'='*70}")
    print(f"  SMOKE TEST Q{row:03d} : {q['question']}")
    print(f"{'='*70}")

    # Phase 1 : RAG Self-Ask
    t0 = time.time()
    final_answer, all_sources, hops = pipeline.query(q["question"])
    elapsed_rag = round(time.time() - t0, 1)

    entry = {
        "excel_row": row,
        "question": q["question"],
        "section": q["section"],
        "subsection": q["subsection"],
        "expected_type": q["expected_type"],
        "hops": hops,
        "n_hops": len(hops),
        "final_answer": final_answer,
        "all_sources": all_sources,
        "elapsed_rag_s": elapsed_rag,
        "meta": {
            "max_hops": 5,
            "k": 5,
            "model_loop": "mistral-large-latest",
            "temperature_loop": 0.0,
            "model_answerer": "claude-haiku-4-5-20251001",
            "ts": datetime.now().isoformat(),
        },
    }
    out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    # Phase 2 : juge V4.3
    print(f"\n  Jugement V4.3...")
    t1 = time.time()
    judge = score_judge_v43(
        q["question"], final_answer, all_sources,
        q["section"], q["subsection"], q["expected_type"],
    )
    elapsed_judge = round(time.time() - t1, 1)

    entry.update(judge)
    entry["elapsed_judge_s"] = elapsed_judge
    out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_trace(entry)


print(f"\n{'='*70}")
print("Smoke test terminé.")
print(f"Fichiers : {OUT_DIR}/selfask_q002.json, {OUT_DIR}/selfask_q004.json")
print("Relance   : python run_selfask_smoke.py")
