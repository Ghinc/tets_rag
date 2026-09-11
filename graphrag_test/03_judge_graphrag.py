"""
Juge gpt-4o-mini : évalue GraphRAG vs Pipeline RAG v10 sur les 5 questions.
Produit comparison_graphrag.json (même format que comparison_results.json de LightRAG).
"""
import json, os, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"

GR_JSON  = pathlib.Path(__file__).parent / "graphrag_results.json"
PL_JSON  = pathlib.Path(__file__).parent.parent / "lightrag_test" / "pipeline_results.json"
OUT      = pathlib.Path(__file__).parent / "comparison_graphrag.json"

SYSTEM_JUDGE = """Tu es un juge expert en évaluation de systèmes de RAG (Retrieval-Augmented Generation) appliqués à des données de qualité de vie. Tu évalues des réponses générées automatiquement à partir de données d'enquêtes citoyennes et d'entretiens sur le bien-être en Corse.

Pour chaque réponse, évalue selon 4 dimensions sur une échelle de 1 à 5 :

1. **Pertinence** (1-5) : La réponse répond-elle directement à la question posée ?
2. **Précision factuelle** (1-5) : Les chiffres et faits cités sont-ils plausibles et cohérents ?
3. **Exhaustivité** (1-5) : La réponse couvre-t-elle les aspects importants de la question ?
4. **Traçabilité** (1-5) : Les sources sont-elles clairement identifiées ?

Réponds UNIQUEMENT avec un JSON valide :
{
  "pertinence": {"score": <1-5>, "justification": "<max 2 phrases>"},
  "precision_factuelle": {"score": <1-5>, "justification": "<max 2 phrases>"},
  "exhaustivite": {"score": <1-5>, "justification": "<max 2 phrases>"},
  "tracabilite": {"score": <1-5>, "justification": "<max 2 phrases>"},
  "note_globale": <moyenne arrondie au dixième>,
  "commentaire_global": "<1-2 phrases synthétiques>"
}"""


def judge_answer(question: str, answer: str, system_name: str) -> dict:
    prompt = f"""Système évalué : {system_name}

Question posée : {question}

Réponse à évaluer :
{answer[:4000]}

Évalue cette réponse selon les 4 dimensions demandées."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_JUDGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            dims = ["pertinence", "precision_factuelle", "exhaustivite", "tracabilite"]
            scores = [data[d]["score"] for d in dims if d in data]
            data["note_globale"] = round(sum(scores) / len(scores), 1) if scores else 0.0
            return data
        except Exception as e:
            print(f"  [RETRY {attempt+1}/3] {e}")
            time.sleep(2 ** attempt)
    return {"error": "judge failed", "note_globale": 0}


def main():
    gr_data = json.loads(GR_JSON.read_text(encoding="utf-8"))
    pl_data = json.loads(PL_JSON.read_text(encoding="utf-8"))

    pl_by_id = {r["id"]: r for r in pl_data["results"]}

    results = []
    for gr_r in gr_data["results"]:
        qid = gr_r["id"]
        question = gr_r["question"]
        pl = pl_by_id.get(qid, {})
        print(f"\nQ: {question[:70]}...")

        gr_answer = gr_r.get("answer", "")
        pl_answer = pl.get("answer", "")

        print("  → Juge GraphRAG...")
        gr_eval = judge_answer(question, gr_answer, "GraphRAG (Microsoft)")
        print(f"     note: {gr_eval.get('note_globale')}")

        print("  → Juge Pipeline RAG v10...")
        pl_eval = judge_answer(question, pl_answer, "Pipeline RAG v10 (RAPTOR + décomposition)")
        print(f"     note: {pl_eval.get('note_globale')}")

        results.append({
            "id": qid,
            "question": question,
            "graphrag": {
                "answer": gr_answer,
                "mode": gr_r.get("mode", "global"),
                "sources": gr_r.get("sources", []),
                "eval": gr_eval,
            },
            "pipeline": {
                "answer": pl_answer,
                "sources": pl.get("sources", []),
                "sub_qa_pairs": pl.get("sub_qa_pairs", []),
                "eval": pl_eval,
            },
        })

    OUT.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ comparison_graphrag.json ({len(results)} questions)")

    print("\n=== RÉSUMÉ ===")
    gr_scores = [r["graphrag"]["eval"].get("note_globale", 0) for r in results]
    pl_scores = [r["pipeline"]["eval"].get("note_globale", 0) for r in results]
    for r in results:
        g = r["graphrag"]["eval"].get("note_globale", "?")
        p = r["pipeline"]["eval"].get("note_globale", "?")
        print(f"  {r['question'][:55]}...")
        print(f"    GraphRAG : {g}/5  |  Pipeline : {p}/5")
    print(f"\n  Moyenne GraphRAG  : {round(sum(gr_scores)/len(gr_scores),2)}/5")
    print(f"  Moyenne Pipeline  : {round(sum(pl_scores)/len(pl_scores),2)}/5")


if __name__ == "__main__":
    main()
