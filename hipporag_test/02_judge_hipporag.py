"""
HippoRAG v2 — Jugement LLM (gpt-4o-mini).
Compare HippoRAG vs Pipeline RAG v10 sur les 5 questions.
Entrées : hipporag_results.json + ../lightrag_test/pipeline_results.json
Sortie  : comparison_hipporag.json
"""
import sys, os, json, pathlib
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

HIPPORAG = pathlib.Path(__file__).parent / "hipporag_results.json"
PIPELINE = ROOT / "lightrag_test" / "pipeline_results.json"
OUT      = pathlib.Path(__file__).parent / "comparison_hipporag.json"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_PROMPT = """Tu es un juge d'évaluation RAG expert. Évalue cette réponse à une question sur une enquête de bien-être en Corse.

Question : {question}

Réponse à évaluer : {answer}

Évalue selon 4 critères (1-5 chacun) :
- pertinence : la réponse traite bien la question posée
- precision_factuelle : les informations semblent exactes et sourcées
- exhaustivite : la réponse couvre bien les aspects importants
- tracabilite : on peut identifier d'où viennent les informations

Réponds UNIQUEMENT avec ce JSON (sans markdown) :
{{
  "pertinence": <1-5>,
  "precision_factuelle": <1-5>,
  "exhaustivite": <1-5>,
  "tracabilite": <1-5>,
  "score_global": <moyenne arrondie à 1 décimale>,
  "commentaire": "<1-2 phrases>"
}}"""


def judge(question: str, answer: str, system_name: str) -> dict:
    print(f"  → Juge {system_name}…", end=" ", flush=True)
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, answer=answer[:6000]
            )}],
            temperature=0,
            max_tokens=400,
        )
        raw = r.choices[0].message.content.strip()
        result = json.loads(raw)
        print(f"note: {result.get('score_global', '?')}")
        return result
    except Exception as e:
        print(f"ERREUR: {e}")
        return {"score_global": 0, "erreur": str(e)}


def main():
    hipporag_data = json.loads(HIPPORAG.read_text(encoding="utf-8"))
    pipeline_data = json.loads(PIPELINE.read_text(encoding="utf-8"))

    pipeline_by_id = {r["id"]: r for r in pipeline_data.get("results", pipeline_data)}

    comparisons = []
    hippo_scores, pipe_scores = [], []

    for hr in hipporag_data["results"]:
        qid      = hr["id"]
        question = hr["question"]
        pr = pipeline_by_id.get(qid, {})
        print(f"\nQ: {question[:60]}…")

        h_eval = judge(question, hr["answer"], "HippoRAG")
        p_eval = judge(question, pr.get("answer", ""), "Pipeline RAG v10")

        h_score = h_eval.get("score_global", 0)
        p_score = p_eval.get("score_global", 0)
        hippo_scores.append(h_score)
        pipe_scores.append(p_score)

        comparisons.append({
            "id": qid,
            "question": question,
            "hipporag": {
                "answer": hr["answer"],
                "sources": hr.get("sources", []),
                "eval": h_eval,
            },
            "pipeline": {
                "answer": pr.get("answer", ""),
                "sources": pr.get("sources", []),
                "eval": p_eval,
            },
        })

    avg_h = round(sum(hippo_scores) / len(hippo_scores), 2) if hippo_scores else 0
    avg_p = round(sum(pipe_scores)  / len(pipe_scores),  2) if pipe_scores else 0

    payload = {
        "systems": {
            "hipporag": {"name": "HippoRAG v2", "avg_score": avg_h},
            "pipeline":  {"name": "Pipeline RAG v10", "avg_score": avg_p},
        },
        "comparisons": comparisons,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ comparison_hipporag.json ({len(comparisons)} questions)")
    print("\n=== RÉSUMÉ ===")
    for c in comparisons:
        hs = c["hipporag"]["eval"].get("score_global", "?")
        ps = c["pipeline"]["eval"].get("score_global", "?")
        print(f"  {c['question'][:55]}…")
        print(f"    HippoRAG : {hs}/5  |  Pipeline : {ps}/5")
    print(f"\n  Moyenne HippoRAG : {avg_h}/5")
    print(f"  Moyenne Pipeline : {avg_p}/5")
    print("\nLancez maintenant 03_build_report.py")


if __name__ == "__main__":
    main()
