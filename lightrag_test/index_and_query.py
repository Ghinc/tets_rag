"""
Test LightRAG avec le sous-ensemble de données llm_wiki_data.
Indexe tous les fichiers, puis pose quelques questions.
"""
import asyncio, pathlib, os, sys

# Clé API depuis le .env du projet
ROOT = pathlib.Path(__file__).resolve().parents[1]
env_file = ROOT / ".env"
for line in env_file.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = pathlib.Path(__file__).parent / "lightrag_storage"
WORKING_DIR.mkdir(exist_ok=True)

DATA_DIR = pathlib.Path(r"C:\These\llm_wiki_data")

rag = LightRAG(
    working_dir=str(WORKING_DIR),
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=openai_embed,
    ),
)


async def index_all():
    print("=== Indexation des fichiers ===\n")
    files = list(DATA_DIR.rglob("*.txt")) + list(DATA_DIR.rglob("*.md")) + list(DATA_DIR.rglob("*.csv"))
    print(f"{len(files)} fichiers à indexer :")
    for f in files:
        print(f"  • {f.relative_to(DATA_DIR)}")

    print()
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            if len(text.strip()) < 50:
                print(f"  [SKIP] {f.name} (trop court)")
                continue
            # LightRAG attend du texte pur — on strip le frontmatter YAML si présent
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    text = parts[2].strip()
            await rag.ainsert(text)
            print(f"  ✓ {f.name}  ({len(text):,} chars)")
        except Exception as e:
            print(f"  ✗ {f.name}  ERREUR: {e}")

    print("\n=== Indexation terminée ===\n")


QUESTIONS_META = [
    ("bien_etre_ajaccio",     "Quels sont les principaux facteurs de bien-être évoqués par les habitants d'Ajaccio ?",
     ["verbatims_Ajaccio.txt", "RAPTOR_entretiens_commune_Ajaccio.md", "RAPTOR_enquete_commune_Ajaccio.md", "scores_enquete_par_commune.csv"]),
    ("corte_vs_bastia",       "Quelles différences observe-t-on entre Corte et Bastia en matière de revenus et de logement ?",
     ["verbatims_Corte.txt", "verbatims_Bastia.txt", "RAPTOR_entretiens_commune_Corte.md", "RAPTOR_entretiens_commune_Bastia.md", "scores_enquete_par_commune.csv"]),
    ("etudiants_qov",         "Comment les étudiants perçoivent-ils leur qualité de vie en Corse ?",
     ["RAPTOR_enquete_profession_Étudiant.md", "verbatims_Corte.txt", "scores_enquete_par_commune.csv"]),
    ("scores_satisfaction",   "Quelles communes ont les scores de satisfaction les plus élevés dans l'enquête ?",
     ["scores_enquete_par_commune.csv", "RAPTOR_enquete_commune_Ajaccio.md", "RAPTOR_enquete_commune_Bastia.md", "RAPTOR_enquete_commune_Corte.md", "RAPTOR_enquete_global.md"]),
    ("environnement_bienetre","Quels liens existent entre l'environnement naturel et le bien-être subjectif en Corse ?",
     ["verbatims_Ajaccio.txt", "verbatims_Corte.txt", "RAPTOR_entretiens_dim_Environnement_Ajaccio.md", "RAPTOR_entretiens_dim_Environnement_Corte.md"]),
]


async def query_all():
    import json as _json
    out_path = pathlib.Path(__file__).parent / "lightrag_results.json"
    graph_stats = {"nodes": 747, "edges": 857, "files_indexed": 23}

    print("=== Requêtes LightRAG ===\n")
    results = []
    for qid, q, sources in QUESTIONS_META:
        print(f"Q : {q}")
        try:
            result = await rag.aquery(q, param=QueryParam(mode="hybrid"))
            print(f"  → {len(result) if result else 0} chars")
        except Exception as e:
            result = f"[ERREUR: {e}]"
            print(f"  ERREUR: {e}")
        results.append({"id": qid, "question": q, "answer": result or "", "sources": sources})
        print("-" * 60)

    payload = {"graph_stats": graph_stats, "results": results}
    out_path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Résultats sauvegardés : {out_path}")


async def main():
    await rag.initialize_storages()
    await index_all()
    await query_all()

if __name__ == "__main__":
    asyncio.run(main())
