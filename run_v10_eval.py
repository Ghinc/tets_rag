"""
Script d'evaluation v10 : RAPTOR + Sous-questions + Notation.

Execute le pipeline v10 complet sur N questions de test et sauvegarde
un recapitulatif detaille (JSON + Markdown) avec :
  - les sous-questions generees
  - les reponses aux sous-questions
  - la synthese finale
  - la notation de la dimension evaluee (si applicable)

Usage:
    python run_v10_eval.py
"""

import json
import os
import sys
import io
import time
from datetime import datetime

# Forcer UTF-8 sur stdout (Windows utilise cp1252 par defaut)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from rag_v10_raptor_subq import (
    RaptorSubQuestionPipeline,
    decompose_question,
    answer_subquestion,
    synthesize_answers,
    score_dimension,
    DEFAULT_N_SUBQUESTIONS,
)

# ============================================================
# Questions de test
# ============================================================

QUESTIONS = [
    "Que pensent les seniors du bien-être ?",
    "Quelles sont les dimensions de la qualité de vie privilégiées par les entrepreneurs ?",
    "Quelle est la qualité des plages à Lozzi ?",
    (
        "Les habitants d'Ajaccio ont-ils un intérêt, en termes de qualité de vie, "
        "à aller vivre dans la zone périurbaine ajaccienne ?"
    ),
    "Quels problèmes sont spécifiques à Ajaccio et lesquels sont communs à d'autres communes corses ?",
    "Quelles sont les préoccupations des habitants d'Ajaccio vis-à-vis de leur qualité de vie ?",
]

N_SUBQUESTIONS = DEFAULT_N_SUBQUESTIONS  # 5
K = 5  # chunks evidence par sous-question


# ============================================================
# Execution du pipeline etape par etape (pour capturer les details)
# ============================================================

def run_question(pipeline: RaptorSubQuestionPipeline,
                 question: str,
                 n_subquestions: int = N_SUBQUESTIONS,
                 k: int = K) -> dict:
    """
    Execute le pipeline v10 complet pour une question et retourne un dict detaille.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"QUESTION : {question}")
    print(sep)

    result = {
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "n_subquestions_requested": n_subquestions,
        "k": k,
        "sub_questions": [],
        "sub_answers": [],
        "final_answer": "",
        "scoring": None,
        "error": None,
        "duration_seconds": 0.0,
    }

    t0 = time.time()

    try:
        # Etape 1 : Decomposition
        print(f"\n[1/4] Decomposition en {n_subquestions} sous-questions (Mistral Large)...")
        sub_questions = decompose_question(question, n=n_subquestions)
        result["sub_questions"] = sub_questions
        for i, sq in enumerate(sub_questions, 1):
            print(f"  {i}. {sq}")

        # Etape 2 : Retrieval RAPTOR + reponse Claude Haiku par sous-question
        print(f"\n[2/4] Reponses aux sous-questions (RAPTOR + Claude Haiku)...")
        sub_qa_pairs = []
        for i, sq in enumerate(sub_questions, 1):
            print(f"  [{i}/{len(sub_questions)}] {sq[:80]}...")
            context_str, sources = pipeline.retriever.query(sq, k=k)
            answer = answer_subquestion(sq, context_str)
            sub_qa_pairs.append((sq, answer))
            result["sub_answers"].append({
                "sub_question": sq,
                "answer": answer,
                "n_sources": len(sources),
            })
            print(f"    -> {answer[:150]}...")

        # Etape 3 : Synthese finale Mistral Large
        print(f"\n[3/4] Synthese finale (Mistral Large)...")
        final_answer = synthesize_answers(question, sub_qa_pairs)
        result["final_answer"] = final_answer
        print(f"\nREPONSE FINALE :\n{final_answer}")

        # Etape 4 : Notation de la dimension (optionnelle)
        print(f"\n[4/4] Notation de la dimension (Mistral Large)...")
        scoring = score_dimension(question, final_answer)
        result["scoring"] = scoring
        if scoring["applicable"]:
            print(f"  -> {scoring['dimension']} : {scoring['score']}/5")
            print(f"     {scoring['justification']}")
        else:
            print(f"  -> Non applicable : {scoring['justification'][:80]}")

    except Exception as e:
        result["error"] = str(e)
        print(f"  ERREUR : {e}")

    result["duration_seconds"] = round(time.time() - t0, 1)
    print(f"\n  [Duree : {result['duration_seconds']}s]")

    return result


# ============================================================
# Generation du fichier Markdown recap
# ============================================================

def _score_stars(score: int) -> str:
    """Convertit un score 1-5 en etoiles unicode."""
    return "★" * score + "☆" * (5 - score)


def build_markdown(all_results: list, timestamp: str) -> str:
    lines = [
        "# Evaluation RAG v10 — RAPTOR + Sous-questions + Notation",
        "",
        f"**Date** : {timestamp}",
        f"**Modeles** : Mistral Large (decomposition + synthese + notation) · Claude Haiku (reponses sous-questions)",
        f"**Parametres** : n_subquestions={N_SUBQUESTIONS}, k={K} chunks evidence/sous-question",
        "",
        "---",
        "",
    ]

    for idx, res in enumerate(all_results, 1):
        lines.append(f"## Question {idx}")
        lines.append("")
        lines.append(f"**{res['question']}**")
        lines.append("")

        if res.get("error"):
            lines.append(f"> **ERREUR** : {res['error']}")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        # Sous-questions
        lines.append("### Sous-questions generees (Mistral Large)")
        lines.append("")
        for i, sq in enumerate(res["sub_questions"], 1):
            lines.append(f"{i}. {sq}")
        lines.append("")

        # Reponses aux sous-questions
        lines.append("### Reponses aux sous-questions (RAPTOR + Claude Haiku)")
        lines.append("")
        for sub in res["sub_answers"]:
            lines.append(f"**{sub['sub_question']}**")
            lines.append("")
            lines.append(sub["answer"])
            lines.append("")
            lines.append(f"*(sources RAPTOR : {sub['n_sources']})*")
            lines.append("")

        # Synthese finale
        lines.append("### Synthese finale (Mistral Large)")
        lines.append("")
        lines.append(res["final_answer"])
        lines.append("")

        # Notation
        scoring = res.get("scoring")
        if scoring and scoring.get("applicable"):
            lines.append("### Notation de la dimension")
            lines.append("")
            stars = _score_stars(scoring["score"])
            lines.append(
                f"> **{scoring['dimension']}** : "
                f"**{scoring['score']}/5** {stars}"
            )
            lines.append(">")
            lines.append(f"> {scoring['justification']}")
            lines.append("")
        elif scoring:
            lines.append("### Notation de la dimension")
            lines.append("")
            lines.append(f"> *Notation non applicable : {scoring['justification']}*")
            lines.append("")

        lines.append(f"*Duree totale : {res['duration_seconds']}s*")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "comparaisons_rag"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"v10_eval_{timestamp}.json")
    md_path   = os.path.join(output_dir, f"v10_eval_{timestamp}.md")

    # Init pipeline
    print("Initialisation du pipeline v10...")
    pipeline = RaptorSubQuestionPipeline()
    pipeline.init()

    # Lancer les tests
    all_results = []
    for q in QUESTIONS:
        res = run_question(pipeline, q, n_subquestions=N_SUBQUESTIONS, k=K)
        all_results.append(res)
        # Pause entre questions pour eviter le rate limiting
        time.sleep(3)

    pipeline.close()

    # Sauvegarder JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON sauvegarde : {json_path}")

    # Sauvegarder Markdown
    ts_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content = build_markdown(all_results, ts_readable)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown sauvegarde : {md_path}")

    # Bilan rapide
    print("\n" + "=" * 70)
    print(f"BILAN : {len(all_results)} questions traitees")
    errors = [r for r in all_results if r.get("error")]
    if errors:
        print(f"  {len(errors)} erreur(s) :")
        for r in errors:
            print(f"    - {r['question'][:60]}... -> {r['error']}")
    scored = [r for r in all_results if r.get("scoring") and r["scoring"].get("applicable")]
    if scored:
        print(f"\n  Notes attribuees ({len(scored)}) :")
        for r in scored:
            s = r["scoring"]
            print(f"    {s['dimension']} : {s['score']}/5")
    total_time = sum(r["duration_seconds"] for r in all_results)
    print(f"\n  Duree totale : {total_time:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
