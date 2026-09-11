"""
run_baseline_v12.py — Run baseline LLM_BACKEND=api sur les 109 questions.

Sauvegarde :
  - JSON compatible eval pipeline : {config_name: [entries]}
  - JSON métriques par question    : {question: metrics}

Usage :
  python local/run_baseline_v12.py [--output DIR] [--start N] [--end N]

Interruption : Ctrl+C. Les questions déjà traitées sont sauvegardées.
"""

import argparse, json, os, sys, time, pathlib, traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local"))

from rag_v12_local import RaptorSubQuestionPipelineV12

# ── Paramètres ────────────────────────────────────────────────────────────────

RERUN2 = ROOT / "comparaisons_rag" / "ablations_103q_v43_gpt4o_RERUN2_20260816_130113.json"
CONFIG_NAME = "v12_baseline_api"

def load_questions():
    with open(RERUN2, encoding="utf-8") as f:
        data = json.load(f)
    first_config = next(iter(data.values()))
    return sorted(set(e["question"] for e in first_config))


def run_baseline(output_dir: pathlib.Path, start: int = 0, end: int = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json    = output_dir / f"{CONFIG_NAME}.json"
    metrics_json   = output_dir / f"{CONFIG_NAME}_metrics.json"
    progress_json  = output_dir / f"{CONFIG_NAME}_progress.json"

    questions = load_questions()
    if end is None:
        end = len(questions)
    questions = questions[start:end]
    print(f"\n{'='*60}")
    print(f"Baseline v12 — LLM_BACKEND={os.environ.get('LLM_BACKEND', 'api')!r}")
    print(f"{len(questions)} questions ({start}→{end})")
    print(f"Sortie : {output_dir}")
    print('='*60 + "\n")

    # Reprendre si un run partiel existe
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

    for i, question in enumerate(questions, start=start+1):
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
                "question":      question,
                "reponse":       final_answer,
                "scope":         scope,
                "question_type": question_type,
                "sources":       [
                    {k: v for k, v in s.items() if k != "content"}
                    for s in (sources or [])[:10]
                ],
                "n_subquestions": len(sub_qa) if sub_qa else 0,
            }
            entries.append(entry)
            metrics_all[question] = metrics
            n_done += 1

            elapsed = time.time() - t0
            print(f"  ✅  {elapsed:.1f}s  |  scope={scope}  type={question_type}")
            print(f"     prompt={metrics['total_prompt_tokens']}  "
                  f"completion={metrics['total_completion_tokens']}")

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

        # Sauvegarde intermédiaire après chaque question
        progress = {"entries": entries + list(existing_entries.values()),
                    "metrics": metrics_all}
        with open(progress_json, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    # ── Sauvegarde finale ─────────────────────────────────────────────────────
    output_data = {CONFIG_NAME: entries}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - t_run_start
    print(f"\n{'='*60}")
    print(f"Run terminé : {n_done} OK, {n_errors} erreurs, {total_elapsed/60:.1f} min")
    print(f"  → {output_json.name}")
    print(f"  → {metrics_json.name}")
    print('='*60)

    _print_summary(metrics_all)


def _print_summary(metrics_all: dict):
    if not metrics_all:
        return
    valid = [m for m in metrics_all.values() if "steps" in m]
    if not valid:
        return

    steps = ["decompose", "answer", "synthesize", "score_dimension"]
    print("\n── Résumé métriques ──────────────────────────────────")
    print(f"{'Étape':<16} {'Temps moy (s)':>14} {'Prompt tok moy':>15} {'Compl tok moy':>14} {'Appels':>7}")
    for step in steps:
        times    = [m["steps"].get(step, {}).get("elapsed_s", 0)      for m in valid]
        prompts  = [m["steps"].get(step, {}).get("prompt_tokens", 0)  for m in valid]
        compls   = [m["steps"].get(step, {}).get("completion_tokens", 0) for m in valid]
        calls    = [m["steps"].get(step, {}).get("n_calls", 0)        for m in valid]
        n = len(valid)
        print(f"  {step:<14} {sum(times)/n:>14.1f} {sum(prompts)/n:>15.0f} "
              f"{sum(compls)/n:>14.0f} {sum(calls)/n:>7.1f}")

    total_times  = [m.get("total_elapsed_s", 0)         for m in valid]
    total_prompt = [m.get("total_prompt_tokens", 0)     for m in valid]
    total_compl  = [m.get("total_completion_tokens", 0) for m in valid]
    n = len(valid)
    print(f"  {'TOTAL':<14} {sum(total_times)/n:>14.1f} {sum(total_prompt)/n:>15.0f} "
          f"{sum(total_compl)/n:>14.0f}")
    print(f"\nTotal tokens prompt   : {sum(total_prompt):,}")
    print(f"Total tokens complétion: {sum(total_compl):,}")
    print(f"Temps total estimé     : {sum(total_times)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "comparaisons_rag" / "baseline_v12"))
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--end",    type=int, default=None)
    args = parser.parse_args()

    if os.environ.get("LLM_BACKEND", "api") != "api":
        print("⚠️  LLM_BACKEND != 'api' — la baseline de référence DOIT tourner en mode api.")
        sys.exit(1)

    run_baseline(pathlib.Path(args.output), args.start, args.end)
