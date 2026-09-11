"""
Run les 5 questions de comparaison LightRAG à travers le pipeline RAG v10.
Sauvegarde les réponses + sources dans pipeline_results.json.
"""
import sys, os, json, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rag_v10_raptor_subq import RaptorSubQuestionPipeline, _build_source_label

QUESTIONS = [
    ("bien_etre_ajaccio",
     "Quels sont les principaux facteurs de bien-être évoqués par les habitants d'Ajaccio ?"),
    ("corte_vs_bastia",
     "Quelles différences observe-t-on entre Corte et Bastia en matière de revenus et de logement ?"),
    ("etudiants_qov",
     "Comment les étudiants perçoivent-ils leur qualité de vie en Corse ?"),
    ("scores_satisfaction",
     "Quelles communes ont les scores de satisfaction les plus élevés dans l'enquête ?"),
    ("environnement_bienetre",
     "Quels liens existent entre l'environnement naturel et le bien-être subjectif en Corse ?"),
]

OUT = pathlib.Path(__file__).parent / "pipeline_results.json"

def main():
    pipeline = RaptorSubQuestionPipeline()
    pipeline.init()

    results = []
    for qid, question in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Q : {question}")
        t0 = time.time()
        try:
            result = pipeline.query(question, k=5, n_subquestions=5)
            # query() returns (final_answer, all_sources, scoring, sub_qa_pairs, sources_mobilisees)
            if len(result) == 5:
                final_answer, all_sources, scoring, sub_qa_pairs, sources_mobilisees = result
            else:
                final_answer, all_sources, scoring, sub_qa_pairs = result
                sources_mobilisees = []

            # Construire la liste de sources lisibles
            sources_readable = []
            seen = set()
            for s in all_sources:
                label = _build_source_label(s)
                key = (label, s.get("commune") or s.get("dim1_value") or "")
                if key not in seen:
                    seen.add(key)
                    sources_readable.append({
                        "label": label,
                        "type": s.get("source_type") or s.get("type") or "",
                        "commune": s.get("commune") or s.get("dim1_value") or "",
                        "view": s.get("view_name") or s.get("view") or "",
                        "text_excerpt": (s.get("text") or s.get("document") or "")[:400],
                        "sub_question_idx": s.get("sub_question_idx"),
                    })

            elapsed = time.time() - t0
            print(f"  → {len(final_answer)} chars en {elapsed:.1f}s")
            results.append({
                "id": qid,
                "question": question,
                "answer": final_answer,
                "sources": sources_readable,
                "sub_qa_pairs": sub_qa_pairs,
                "elapsed_s": round(elapsed, 1),
            })
        except Exception as e:
            import traceback
            print(f"  ERREUR : {e}")
            traceback.print_exc()
            results.append({"id": qid, "question": question, "answer": f"[ERREUR: {e}]", "sources": []})

    payload = {"system": "pipeline_v10_raptor_subq", "results": results}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ pipeline_results.json sauvegardé ({len(results)} questions)")

if __name__ == "__main__":
    main()
