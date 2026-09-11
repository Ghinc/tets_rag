"""
Requêtes LightRAG — réponses complètes + sources, sorties JSON pour le rapport HTML.
Les réponses sont en cache → 0 nouveaux tokens facturés.
"""
import asyncio, pathlib, os, json

ROOT = pathlib.Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = pathlib.Path(__file__).parent / "lightrag_storage"
OUT_JSON    = pathlib.Path(__file__).parent / "lightrag_results.json"

rag = LightRAG(
    working_dir=str(WORKING_DIR),
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=openai_embed,
    ),
)

QUESTIONS = [
    ("bien_etre_ajaccio",    "Quels sont les principaux facteurs de bien-être évoqués par les habitants d'Ajaccio ?"),
    ("corte_vs_bastia",      "Quelles différences observe-t-on entre Corte et Bastia en matière de revenus et de logement ?"),
    ("etudiants_qov",        "Comment les étudiants perçoivent-ils leur qualité de vie en Corse ?"),
    ("scores_satisfaction",  "Quelles communes ont les scores de satisfaction les plus élevés dans l'enquête ?"),
    ("environnement_bienetre","Quels liens existent entre l'environnement naturel et le bien-être subjectif en Corse ?"),
]

SOURCES_USED = {
    "bien_etre_ajaccio":     ["verbatims_Ajaccio.txt", "RAPTOR_entretiens_commune_Ajaccio.md", "RAPTOR_enquete_commune_Ajaccio.md", "scores_enquete_par_commune.csv"],
    "corte_vs_bastia":       ["verbatims_Corte.txt", "verbatims_Bastia.txt", "RAPTOR_entretiens_commune_Corte.md", "RAPTOR_entretiens_commune_Bastia.md", "scores_enquete_par_commune.csv"],
    "etudiants_qov":         ["RAPTOR_enquete_profession_Étudiant.md", "RAPTOR_entretiens_commune_Corte.md", "scores_enquete_par_commune.csv"],
    "scores_satisfaction":   ["scores_enquete_par_commune.csv", "RAPTOR_enquete_commune_*.md", "RAPTOR_enquete_global.md"],
    "environnement_bienetre":["verbatims_Ajaccio.txt", "verbatims_Corte.txt", "RAPTOR_entretiens_dim_Environnement_Ajaccio.md", "RAPTOR_entretiens_dim_Environnement_Corte.md"],
}

GRAPH_STATS = {"nodes": 747, "edges": 857, "files_indexed": 23}

async def main():
    await rag.initialize_storages()
    results = []
    for qid, question in QUESTIONS:
        print(f"\nQ: {question}")
        try:
            answer = await rag.aquery(question, param=QueryParam(mode="hybrid"))
            print(f"  → {len(answer)} chars")
        except Exception as e:
            answer = f"[ERREUR: {e}]"
        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "sources": SOURCES_USED.get(qid, []),
        })

    payload = {"graph_stats": GRAPH_STATS, "results": results}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Sauvegardé : {OUT_JSON}")

asyncio.run(main())
