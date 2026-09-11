"""
smoke_phase1.py — Smoke test Phase 1 : baseline vs structured extraction (3 questions).

3 questions × 2 modes = 6 appels pipeline (LLM_BACKEND=api).
Aucun appel au juge. Sortie JSON pour construction du rapport vis-à-vis.

Usage :
  python local/smoke_phase1.py
"""

import argparse, os, sys, json, time, pathlib, traceback

os.environ.setdefault("LLM_BACKEND", "api")
os.environ["PHASE1_STRUCTURED"] = "0"   # sera changé entre les runs

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local"))

import rag_v12_local as v12
from rag_v12_local import RaptorSubQuestionPipelineV12, _sq_json_to_prose

OUT = ROOT / "comparaisons_rag" / "smoke_phase1"
OUT.mkdir(parents=True, exist_ok=True)

QUESTIONS = {
    "causal": (
        "Le niveau de bien-être à Ajaccio peut-il s'expliquer par une combinaison "
        "de facteurs environnementaux, socio-économiques et institutionnels ?"
    ),
    "multi_communes": (
        "Quelle commune obtient le score objectif global le plus élevé entre Ajaccio et Bastia ?"
    ),
    "no_data": (
        "Quel est le niveau de bien-être subjectif à Aïti ?"
    ),
}

# La question pour laquelle on montre le texte brut passé au synthétiseur
SYNTH_CAPTURE_KEY = "causal"


def run_question(pipeline, question: str, phase1: bool) -> dict:
    """Lance le pipeline sur une question. Retourne un dict de résultats."""
    v12.PHASE1_STRUCTURED = phase1
    mode = "phase1" if phase1 else "baseline"
    print(f"\n  mode={mode!r} : {question[:60]}...")
    t0 = time.time()
    try:
        result = pipeline.query(question)
        final_answer, sources, scoring, sub_qa, sources_mob, scope, question_type, metrics = result
        elapsed = time.time() - t0

        answer_compl = metrics["steps"]["answer"]["completion_tokens"]
        synth_compl  = metrics["steps"]["synthesize"]["completion_tokens"]
        total_compl  = metrics["total_completion_tokens"]

        # Reconstruire le texte passé au synthétiseur
        # Baseline : sub_qa[i]["answer"] = prose brute
        # Phase 1  : sub_qa[i]["answer"] = JSON brut -> convertir via _sq_json_to_prose
        if phase1:
            synth_input = [
                {
                    "question": e["question"],
                    "json_raw": e["answer"],
                    "prose_for_synth": _sq_json_to_prose(e["answer"]),
                }
                for e in (sub_qa or [])
            ]
        else:
            synth_input = [
                {
                    "question": e["question"],
                    "prose_for_synth": e["answer"],
                }
                for e in (sub_qa or [])
            ]

        # Préparer sources pour le juge (format extrait/metadata déjà compatible)
        sources_for_judge = [
            {k: v for k, v in s.items() if k != "extrait"}
            | {"extrait": str(s.get("extrait", s.get("content", "")))[:500]}
            for s in (sources or [])
        ]

        return {
            "mode": mode,
            "question": question,
            "scope": scope,
            "question_type": question_type,
            "n_subquestions": len(sub_qa) if sub_qa else 0,
            "final_answer": final_answer,
            "elapsed_s": round(elapsed, 1),
            "answer_completion_tokens": answer_compl,
            "synth_completion_tokens":  synth_compl,
            "total_completion_tokens":  total_compl,
            "synth_input": synth_input,
            "sources_retrieved": sources_for_judge,
            "guard": metrics.get("guard", {}),
            "error": None,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ERREUR : {exc}")
        traceback.print_exc()
        return {
            "mode": mode,
            "question": question,
            "final_answer": f"[ERREUR] {exc}",
            "elapsed_s": round(elapsed, 1),
            "answer_completion_tokens": 0,
            "synth_completion_tokens": 0,
            "total_completion_tokens": 0,
            "synth_input": [],
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="", help="Suffixe du fichier de sortie (ex. _run1)")
    args = parser.parse_args()
    run_suffix = args.run  # "" | "_run1" | "_run2" …

    print("\n" + "=" * 70)
    print(f"Smoke test Phase 1 — 3 questions × 2 modes (LLM_BACKEND=api){' [' + run_suffix.strip('_') + ']' if run_suffix else ''}")
    print("=" * 70)

    print("\nChargement pipeline (bge-m3 -> CPU, LLM_BACKEND=api)...")
    t_load = time.time()
    v12.PHASE1_STRUCTURED = False
    pipeline = RaptorSubQuestionPipelineV12()
    pipeline.init()
    print(f"Chargement : {(time.time()-t_load):.0f}s\n")

    results = {}

    for key, question in QUESTIONS.items():
        print(f"\n{'-'*60}")
        print(f"Question [{key}] : {question[:80]}")
        results[key] = {
            "question": question,
            "key": key,
        }
        for phase1 in [False, True]:
            run = run_question(pipeline, question, phase1)
            mode = run["mode"]
            results[key][mode] = run
            print(f"  {mode:8s} : {run['elapsed_s']}s | "
                  f"answer_compl={run['answer_completion_tokens']} | "
                  f"total_compl={run['total_completion_tokens']}")

    # Sauvegarder les résultats bruts
    out_json = OUT / f"smoke_phase1_results{run_suffix}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nRésultats sauvegardés : {out_json}")

    # Afficher un résumé console
    print("\n-- Resume tokens answer ------------------------------------------")
    print(f"{'Question':<15} {'Base compl':>12} {'P1 compl':>10} {'Gain':>8}")
    for key, r in results.items():
        b = r.get("baseline", {}).get("answer_completion_tokens", 0)
        p = r.get("phase1", {}).get("answer_completion_tokens", 0)
        gain = f"-{(1-p/b)*100:.0f}%" if b else "?"
        print(f"  {key:<15} {b:>12} {p:>10} {gain:>8}")


if __name__ == "__main__":
    main()
