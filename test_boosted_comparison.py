"""
Script de test pour comparer les résultats entre v1, v2 original et v2 boosted

Compare les résultats pour la requête problématique :
"Quelles sont les dimensions du bien-être plébiscitées à ajaccio ?"

Auteur: Claude Code
Date: 2025-11-15
"""

import os
from dotenv import load_dotenv
from rag_v1_class import BasicRAGPipeline
from rag_v2_improved import ImprovedRAGPipeline
from rag_v2_boosted import ImprovedRAGPipeline as ImprovedRAGPipelineWithBoost

# Charger les variables d'environnement
load_dotenv()

def test_version(version_name: str, rag_pipeline, question: str, k: int = 5):
    """
    Teste une version du RAG et affiche les résultats

    Args:
        version_name: Nom de la version (v1, v2, v2_boosted)
        rag_pipeline: Pipeline RAG à tester
        question: Question à poser
        k: Nombre de résultats à récupérer
    """
    print(f"\n{'='*80}")
    print(f"VERSION: {version_name}")
    print(f"{'='*80}\n")

    try:
        if version_name == "v1":
            # v1 n'a pas les options avancées
            answer, results = rag_pipeline.query(question=question, k=k)
        else:
            # v2 et v2_boosted ont les mêmes options
            answer, results = rag_pipeline.query(
                question=question,
                k=k,
                use_reranking=True,
                include_quantitative=False,
                commune_filter="Ajaccio"
            )

        print(f"REPONSE:\n{answer}\n")
        print(f"\nSOURCES RECUPEREES ({len(results)}):")
        print("-" * 80)

        # Compter les sources par type
        source_counts = {}
        for result in results:
            source_type = result.metadata.get('source', 'unknown')
            source_counts[source_type] = source_counts.get(source_type, 0) + 1

        print("\nREPARTITION PAR SOURCE:")
        for source_type, count in sorted(source_counts.items()):
            print(f"  - {source_type}: {count}")

        print("\nDETAIL DES 5 PREMIERS RESULTATS:")
        for i, result in enumerate(results[:5], 1):
            commune = result.metadata.get('commune', result.metadata.get('nom', 'N/A'))
            source = result.metadata.get('source', 'unknown')
            filename = result.metadata.get('filename', 'N/A')

            print(f"\n{i}. Score: {result.score:.4f} | Source: {source} | Commune: {commune}")
            print(f"   Fichier: {filename}")
            print(f"   Extrait: {result.text[:150]}...")

    except Exception as e:
        print(f"ERREUR lors du test de {version_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Point d'entrée principal"""

    # Question problématique
    question = "Quelles sont les dimensions du bien-être plébiscitées à ajaccio ?"

    print("\n" + "="*80)
    print("TEST DE COMPARAISON: V1 vs V2 ORIGINAL vs V2 BOOSTED")
    print("="*80)
    print(f"\nQuestion: {question}")
    print(f"K: 5 résultats")

    # Récupérer la clé API
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERREUR: OPENAI_API_KEY non trouvée dans .env")
        return

    # === TESTER V1 ===
    print("\n\n[1/3] Initialisation RAG v1...")
    try:
        rag_v1 = BasicRAGPipeline(
            openai_api_key=openai_api_key,
            chroma_path="./chroma_txt/",
            collection_name="communes_corses_txt",
            llm_model="gpt-3.5-turbo",
            embedding_model="intfloat/e5-base-v2"
        )
        test_version("v1", rag_v1, question, k=5)
    except Exception as e:
        print(f"Impossible de tester v1: {e}")

    # === TESTER V2 ORIGINAL ===
    print("\n\n[2/3] Initialisation RAG v2 original...")
    try:
        rag_v2 = ImprovedRAGPipeline(
            openai_api_key=openai_api_key,
            chroma_path="./chroma_v2/",
            collection_name="communes_corses_v2",
            quant_data_path="df_mean_by_commune.csv",
            llm_model="gpt-3.5-turbo",
            embedding_model="intfloat/e5-base-v2",
            reranker_model="antoinelouis/crossencoder-camembert-base-mmarcoFR"
        )
        test_version("v2_original", rag_v2, question, k=5)
    except Exception as e:
        print(f"Impossible de tester v2 original: {e}")

    # === TESTER V2 BOOSTED ===
    print("\n\n[3/3] Initialisation RAG v2 boosted...")
    try:
        rag_v2_boosted = ImprovedRAGPipelineWithBoost(
            openai_api_key=openai_api_key,
            chroma_path="./chroma_v2/",
            collection_name="communes_corses_v2",
            quant_data_path="df_mean_by_commune.csv",
            llm_model="gpt-3.5-turbo",
            embedding_model="intfloat/e5-base-v2",
            reranker_model="antoinelouis/crossencoder-camembert-base-mmarcoFR"
        )
        test_version("v2_boosted", rag_v2_boosted, question, k=5)
    except Exception as e:
        print(f"Impossible de tester v2 boosted: {e}")

    print("\n\n" + "="*80)
    print("FIN DES TESTS")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
