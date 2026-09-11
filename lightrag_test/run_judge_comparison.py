"""
Juge de comparaison LightRAG vs Pipeline RAG v10.
Évalue chaque réponse sur 4 dimensions (gpt-4o-mini).
Entrées : lightrag_results.json + pipeline_results.json
Sortie  : comparison_results.json
"""
import json, os, pathlib, time
from openai import OpenAI
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o-mini"

LIGHTRAG_JSON  = pathlib.Path(__file__).parent / "lightrag_results.json"
PIPELINE_JSON  = pathlib.Path(__file__).parent / "pipeline_results.json"
OUT            = pathlib.Path(__file__).parent / "comparison_results.json"

SYSTEM_JUDGE = """Tu es un juge expert en évaluation de systèmes de RAG (Retrieval-Augmented Generation) appliqués à des données de qualité de vie. Tu évalues des réponses générées automatiquement à partir de données d'enquêtes citoyennes et d'entretiens sur le bien-être en Corse.

Pour chaque réponse, évalue selon 4 dimensions sur une échelle de 1 à 5 :

1. **Pertinence** (1-5) : La réponse répond-elle directement à la question posée ? Reste-t-elle dans le scope demandé sans dériver ?

2. **Précision factuelle** (1-5) : Les chiffres et faits cités sont-ils plausibles et cohérents ? Y a-t-il des incohérences ou affirmations douteuses ? (Note : tu n'as pas accès aux données brutes, évalue la cohérence interne et la plausibilité)

3. **Exhaustivité** (1-5) : La réponse couvre-t-elle les aspects importants de la question ? Manque-t-il des dimensions cruciales ?

4. **Traçabilité** (1-5) : Les sources sont-elles clairement identifiées ? Peut-on distinguer d'où vient chaque information (enquête, verbatim, indicateurs objectifs) ? Les affirmations sont-elles attribuées ?

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
            # Calculer la note globale si absente
            dims = ["pertinence", "precision_factuelle", "exhaustivite", "tracabilite"]
            scores = [data[d]["score"] for d in dims if d in data]
            data["note_globale"] = round(sum(scores) / len(scores), 1) if scores else 0.0
            return data
        except Exception as e:
            print(f"  [RETRY {attempt+1}/3] {e}")
            time.sleep(2 ** attempt)
    return {"error": "judge failed", "note_globale": 0}


def main():
    lr_data = json.loads(LIGHTRAG_JSON.read_text(encoding="utf-8"))
    pl_data = json.loads(PIPELINE_JSON.read_text(encoding="utf-8"))

    lr_by_id = {r["id"]: r for r in lr_data["results"]}
    pl_by_id = {r["id"]: r for r in pl_data["results"]}

    results = []
    qids = [r["id"] for r in lr_data["results"]]
    for qid in qids:
        lr = lr_by_id.get(qid, {})
        pl = pl_by_id.get(qid, {})
        question = lr.get("question") or pl.get("question", "")
        print(f"\nQ: {question[:70]}...")

        lr_answer = lr.get("answer", "")
        pl_answer = pl.get("answer", "")

        print("  → Juge LightRAG...")
        lr_eval = judge_answer(question, lr_answer, "LightRAG (graphe de connaissances)")
        print(f"     note: {lr_eval.get('note_globale')}")

        print("  → Juge Pipeline RAG v10...")
        pl_eval = judge_answer(question, pl_answer, "Pipeline RAG v10 (RAPTOR + décomposition)")
        print(f"     note: {pl_eval.get('note_globale')}")

        results.append({
            "id": qid,
            "question": question,
            "lightrag": {
                "answer": lr_answer,
                "sources": lr.get("sources", []),
                "eval": lr_eval,
            },
            "pipeline": {
                "answer": pl_answer,
                "sources": pl.get("sources", []),
                "sub_qa_pairs": pl.get("sub_qa_pairs", []),
                "eval": pl_eval,
            },
        })

    OUT.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ comparison_results.json sauvegardé ({len(results)} questions)")

    # Résumé rapide
    print("\n=== RÉSUMÉ ===")
    for r in results:
        lr_g = r["lightrag"]["eval"].get("note_globale", "?")
        pl_g = r["pipeline"]["eval"].get("note_globale", "?")
        q = r["question"][:60]
        print(f"  {q}...")
        print(f"    LightRAG : {lr_g}/5  |  Pipeline : {pl_g}/5")

if __name__ == "__main__":
    main()
