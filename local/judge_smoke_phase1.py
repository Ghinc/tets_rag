"""
judge_smoke_phase1.py — Notation juge V4.3 (gpt-4o) sur les 3 questions du smoke test.

Lit smoke_phase1_results.json, extrait les réponses baseline et phase1,
appelle score_judge_v43 pour chacune, compare avec les scores RERUN2.

Usage :
    python local/judge_smoke_phase1.py

Sortie : comparaisons_rag/smoke_phase1/judge_results.json
"""
import argparse, json, os, re, sys, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configurer le juge gpt-4o (identique à RERUN2)
import eval_from_excel as evmod
evmod.JUDGE_MODEL       = "gpt-4o"
evmod.JUDGE_MODEL_LIGHT = "gpt-4o-mini"
evmod.JUDGE_BASE_URL    = "https://api.openai.com/v1"
evmod.JUDGE_API_KEY_ENV = "OPENAI_API_KEY"
evmod._openai_client    = None

from eval_from_excel import score_judge_v43

_parser = argparse.ArgumentParser()
_parser.add_argument("--run", default="", help="Suffixe du fichier résultats (ex. _run1)")
_args, _ = _parser.parse_known_args()
_RUN_SUFFIX = _args.run

SMOKE_JSON = ROOT / "comparaisons_rag" / "smoke_phase1" / f"smoke_phase1_results{_RUN_SUFFIX}.json"
RERUN2     = ROOT / "comparaisons_rag" / "ablations_103q_v43_gpt4o_RERUN2_20260816_130113.json"
OUT_DIR    = ROOT / "comparaisons_rag" / "smoke_phase1"

# Métadonnées section/subsection récupérées depuis RERUN2 pour les 3 questions
QUESTION_META = {
    "causal": {
        "section":    "Raisonnement causal et contre-intuitif",
        "subsection": "Raisonnement causal multi-facteurs",
        "expected_type": "reponse_substantielle_attendue",
    },
    "multi_communes": {
        "section":    "Raisonnement comparatif",
        "subsection": "Comparaisons avec hiérarchisation explicite",
        "expected_type": "reponse_substantielle_attendue",
    },
    "no_data": {
        "section":    "Gestion d'absence d'information",
        "subsection": "Absence totale de données",
        "expected_type": "reponse_substantielle_attendue",
    },
}


def load_rerun2_entry(question_fragment: str) -> dict:
    with open(RERUN2, encoding="utf-8") as f:
        data = json.load(f)
    config = data.get("v_decomp_raptor", [])
    for e in config:
        if question_fragment[:25].lower() in e.get("question", "").lower():
            return e
    return {}


def judge_answer(key: str, question: str, answer: str, sources: list, mode: str) -> dict:
    meta = QUESTION_META[key]
    print(f"  Jugement {mode} / {key}...", end=" ", flush=True)
    t0 = time.time()
    result = score_judge_v43(
        question=question,
        answer=answer,
        sources=sources,
        section=meta["section"],
        subsection=meta["subsection"],
        expected_type=meta["expected_type"],
    )
    elapsed = time.time() - t0
    score = result.get("score_global")
    print(f"score={score}/5 ({elapsed:.1f}s)")
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("Judge V4.3 (gpt-4o) — Smoke test Phase 1")
    print("=" * 70)

    if not SMOKE_JSON.exists():
        print(f"ERREUR : {SMOKE_JSON} introuvable. Lancer d'abord smoke_phase1.py")
        sys.exit(1)

    with open(SMOKE_JSON, encoding="utf-8") as f:
        smoke = json.load(f)

    # Charger les entrées RERUN2 pour les 3 questions
    rerun2_entries = {
        "causal":        load_rerun2_entry("Le niveau de bien-être à Ajaccio"),
        "multi_communes": load_rerun2_entry("Quelle commune obtient le score objectif"),
        "no_data":       load_rerun2_entry("Quel est le niveau de bien-être subjectif à Aïti"),
    }

    all_results = {}

    for key in ["causal", "multi_communes", "no_data"]:
        q_data = smoke.get(key, {})
        question = q_data.get("question", "")
        print(f"\n[{key}] {question[:70]}")

        key_results = {"question": question, "rerun2_reference": {}, "modes": {}}

        # Référence RERUN2
        ref = rerun2_entries[key]
        key_results["rerun2_reference"] = {
            "score_global":           ref.get("score_global"),
            "pertinence":             ref.get("pertinence"),
            "fondement_factuel":      ref.get("fondement_factuel"),
            "nuance_incertitude":     ref.get("nuance_incertitude"),
            "coherence_qualiquanti":  ref.get("coherence_qualiquanti"),
            "mislabelling_flag":      ref.get("mislabelling_flag", False),
        }

        for mode in ["baseline", "phase1"]:
            mode_data = q_data.get(mode, {})
            answer  = mode_data.get("final_answer", "")
            sources = mode_data.get("sources_retrieved", [])
            if not answer:
                print(f"  {mode}: pas de réponse")
                continue

            j = judge_answer(key, question, answer, sources, mode)
            key_results["modes"][mode] = {
                "answer_completion_tokens": mode_data.get("answer_completion_tokens", 0),
                "total_completion_tokens":  mode_data.get("total_completion_tokens", 0),
                "elapsed_s":                mode_data.get("elapsed_s", 0),
                "score_global":             j.get("score_global"),
                "pertinence":               j.get("pertinence"),
                "fondement_factuel":        j.get("fondement_factuel"),
                "nuance_incertitude":       j.get("nuance_incertitude"),
                "coherence_qualiquanti":    j.get("coherence_qualiquanti"),
                "mislabelling_flag":        j.get("mislabelling_flag", False),
                "raisonnement":             j.get("raisonnement", ""),
                "judge_raw":                j,
            }

        all_results[key] = key_results

    # Sauvegarder
    out_path = OUT_DIR / f"judge_results{_RUN_SUFFIX}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats sauvegardés : {out_path}")

    # Résumé console
    print("\n" + "=" * 70)
    print("RÉSUMÉ — Score global /5  (ref=v_decomp_raptor)")
    print(f"{'Question':<20} {'Ref':>6} {'Baseline':>10} {'Phase1':>8}")
    print("-" * 50)
    for key, r in all_results.items():
        ref_s  = r["rerun2_reference"].get("score_global", "?")
        base_s = r["modes"].get("baseline", {}).get("score_global", "?")
        p1_s   = r["modes"].get("phase1",   {}).get("score_global", "?")
        print(f"  {key:<18} {str(ref_s):>6} {str(base_s):>10} {str(p1_s):>8}")

    print("\nTokens answer (completion) :")
    print(f"{'Question':<20} {'Baseline':>10} {'Phase1':>8} {'Gain':>7}")
    print("-" * 50)
    for key, r in all_results.items():
        b = r["modes"].get("baseline", {}).get("answer_completion_tokens", 0)
        p = r["modes"].get("phase1",   {}).get("answer_completion_tokens", 0)
        gain = f"-{(1-p/b)*100:.0f}%" if b else "?"
        print(f"  {key:<18} {b:>10} {p:>8} {gain:>7}")


if __name__ == "__main__":
    main()
