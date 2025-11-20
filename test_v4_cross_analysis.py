"""
Script de test pour RAG v4 avec analyse croisée

Teste la capacité de v4 à faire des analyses croisées entre
indicateurs OppChoVec et entretiens/verbatims.
"""

import os
from dotenv import load_dotenv
from rag_v4_cross_analysis import ImprovedRAGPipeline

load_dotenv()

def test_v4_cross_analysis():
    """Teste v4 avec une question nécessitant une analyse croisée"""

    print("="*80)
    print("TEST RAG V4 - ANALYSE CROISÉE")
    print("="*80)

    # Initialiser RAG v4
    print("\nInitialisation de RAG v4...")
    rag = ImprovedRAGPipeline(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chroma_path="./chroma_v2/",
        collection_name="communes_corses_v2",
        quant_data_path="df_mean_by_commune.csv",
        llm_model="gpt-3.5-turbo",
        embedding_model="intfloat/e5-base-v2",
        reranker_model="antoinelouis/crossencoder-camembert-base-mmarcoFR"
    )

    # Question nécessitant une analyse croisée
    question = "Pourquoi le score Vec est bas à Ajaccio ?"

    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")

    # Utiliser la méthode d'analyse croisée
    answer, results = rag.query_with_cross_analysis(
        question=question,
        k=5,
        use_reranking=True,
        include_quantitative=False,
        commune_filter="Ajaccio"
    )

    # Afficher la réponse
    print("\n" + "="*80)
    print("RÉPONSE GÉNÉRÉE")
    print("="*80)
    print(answer)

    # Afficher les sources utilisées
    print("\n" + "="*80)
    print(f"SOURCES UTILISÉES ({len(results)})")
    print("="*80)

    # Compter par type de source
    source_counts = {}
    for result in results:
        source_type = result.metadata.get('source', 'unknown')
        source_counts[source_type] = source_counts.get(source_type, 0) + 1

    print("\nRépartition par source:")
    for source_type, count in sorted(source_counts.items()):
        print(f"  - {source_type}: {count}")

    print("\nDétail des 5 sources:")
    for i, result in enumerate(results[:5], 1):
        source = result.metadata.get('source', 'unknown')
        commune = result.metadata.get('commune', result.metadata.get('nom', 'N/A'))
        print(f"\n{i}. {source} - {commune} (score: {result.score:.4f})")
        print(f"   {result.text[:150]}...")

    print("\n" + "="*80)
    print("TEST TERMINÉ")
    print("="*80)


if __name__ == "__main__":
    test_v4_cross_analysis()
