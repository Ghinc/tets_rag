"""
run_phase1_structured.py — Run Phase 1 : extraction structurée (JSON compact).

Identique à run_baseline_v12.py mais avec PHASE1_STRUCTURED=1.
Sortie dans comparaisons_rag/phase1_structured/.

Usage :
  set PHASE1_STRUCTURED=1
  python local/run_phase1_structured.py [--start N] [--end N]

LLM_BACKEND doit rester 'api' pour ce run de validation.
"""

import argparse, json, os, sys, time, pathlib, traceback

os.environ["PHASE1_STRUCTURED"] = "1"

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local"))

from rag_v12_local import RaptorSubQuestionPipelineV12, PHASE1_STRUCTURED

CONFIG_NAME = "v12_phase1_structured"
RERUN2 = ROOT / "comparaisons_rag" / "ablations_103q_v43_gpt4o_RERUN2_20260816_130113.json"


def load_questions():
    with open(RERUN2, encoding="utf-8") as f:
        data = json.load(f)
    first_config = next(iter(data.values()))
    return sorted(set(e["question"] for e in first_config))


def run(output_dir: pathlib.Path, start: int = 0, end: int = None):
    if not PHASE1_STRUCTURED:
        print("⚠️  PHASE1_STRUCTURED n'est pas activé — abandon.")
        sys.exit(1)
    if os.environ.get("LLM_BACKEND", "api") != "api":
        print("⚠️  LLM_BACKEND != 'api' — ce run de validation doit tourner en mode api.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json   = output_dir / f"{CONFIG_NAME}.json"
    metrics_json  = output_dir / f"{CONFIG_NAME}_metrics.json"
    progress_json = output_dir / f"{CONFIG_NAME}_progress.json"

    questions = load_questions()
    if end is None:
        end = len(questions)
    questions = questions[start:end]

    print(f"\n{'='*60}")
    print(f"Phase 1 Structurée — {len(questions)} questions ({start}→{end})")
    print(f"Sortie : {output_dir}")
    print('='*60 + "\n")

    existing_entries = {}
    existing_metrics = {}
    if progress_json.exists():
        with open(progress_json, encoding="utf-8") as f:
            progress = json.load(f)
        existing_entries = {e["question"]: e for e in progress.get("entries", [])}
        existing_metrics = progress.get("metrics", {})
        print(f"Reprise : {len(existing_entries)} questions déjà traitées.\n")

    pipeline = RaptorSubQuestionPipelineV12()
    pipeline.init()

    entries = []
    metrics_all = dict(existing_metrics)
    t_run_start = time.time()
    n_done = 0
    n_errors = 0

    for i, question in enumerate(questions, start=start + 1):
        if question in existing_entries:
            entries.append(existing_entries[question])
            print(f"[{i:3d}/{end}] (cache) {question[:70]}")
            n_done += 1
            continue

        print(f"\n[{i:3d}/{end}] {question}")
        t0 = time.time()
        try:
            result = pipeline.query(question)
            final_answer, sources, scoring, sub_qa, sources_mob, scope, question_type, metrics = result

            entry = {
                "question":       question,
                "reponse":        final_answer,
                "scope":          scope,
                "question_type":  question_type,
                "sources":        [
                    {k: v for k, v in s.items() if k != "content"}
                    for s in (sources or [])[:10]
                ],
                "n_subquestions": len(sub_qa) if sub_qa else 0,
                "sub_qa":         sub_qa or [],
            }
            entries.append(entry)
            metrics_all[question] = metrics
            n_done += 1

            elapsed = time.time() - t0
            ans_tok = metrics["steps"]["answer"]["completion_tokens"]
            synth_tok = metrics["steps"]["synthesize"]["completion_tokens"]
            print(f"  ✅  {elapsed:.1f}s  scope={scope}  type={question_type}")
            print(f"     answer_compl={ans_tok}  synth_compl={synth_tok}  "
                  f"total_compl={metrics['total_completion_tokens']}")

        except Exception as exc:
            n_errors += 1
            print(f"  ❌  ERREUR : {exc}")
            traceback.print_exc()
            entry = {
                "question":      question,
                "reponse":       f"[ERREUR] {exc}",
                "scope":         "unknown",
                "question_type": "unknown",
                "sources":       [],
                "n_subquestions": 0,
                "error":         str(exc),
            }
            entries.append(entry)

        progress = {"entries": entries + list(existing_entries.values()),
                    "metrics": metrics_all}
        with open(progress_json, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    output_data = {CONFIG_NAME: entries}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - t_run_start
    print(f"\n{'='*60}")
    print(f"Run terminé : {n_done} OK, {n_errors} erreurs, {total_elapsed/60:.1f} min")

    _print_summary(metrics_all)


def _print_summary(metrics_all: dict):
    valid = [m for m in metrics_all.values() if "steps" in m]
    if not valid:
        return
    steps = ["decompose", "answer", "synthesize"]
    print("\n── Résumé métriques Phase 1 ──────────────────────────────")
    print(f"{'Étape':<16} {'Temps moy (s)':>14} {'Prompt tok moy':>15} "
          f"{'Compl tok moy':>14} {'Appels':>7}")
    for step in steps:
        times   = [m["steps"].get(step, {}).get("elapsed_s", 0)         for m in valid]
        prompts = [m["steps"].get(step, {}).get("prompt_tokens", 0)     for m in valid]
        compls  = [m["steps"].get(step, {}).get("completion_tokens", 0) for m in valid]
        calls   = [m["steps"].get(step, {}).get("n_calls", 0)           for m in valid]
        n = len(valid)
        print(f"  {step:<14} {sum(times)/n:>14.1f} {sum(prompts)/n:>15.0f} "
              f"{sum(compls)/n:>14.0f} {sum(calls)/n:>7.1f}")
    total_compl = [m.get("total_completion_tokens", 0) for m in valid]
    n = len(valid)
    print(f"\n  Completion moy total : {sum(total_compl)/n:.0f} tok/question")
    print(f"  (baseline : ~3744 tok/question)")
    print(f"  Gain estimé : {(1 - sum(total_compl)/(n*3744))*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "comparaisons_rag" / "phase1_structured"))
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--end",    type=int, default=None)
    args = parser.parse_args()
    run(pathlib.Path(args.output), args.start, args.end)
