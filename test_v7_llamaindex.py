"""
Script de test pour RAG v7 (LlamaIndex)

Test rapide des 3 modes:
- router: Routing automatique vector/graph
- sub_question: Décomposition en sous-questions
- hybrid: Fusion vector + graph
"""

import os
from rag_v7_llamaindex import LlamaIndexRAGPipeline


def test_v7():
    """Test rapide de la v7"""
    print("="*80)
    print("TEST RAG v7 - LlamaIndex Pipeline")
    print("="*80)

    # Récupérer API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY manquante dans .env")
        return

    # Initialiser pipeline
    print("\nInitialisation du pipeline...")
    rag = LlamaIndexRAGPipeline(
        openai_api_key=OPENAI_API_KEY,
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=""
    )

    # Tests des 3 modes
    tests = [
        {
            "question": "Quels sont les scores de bien-être à Ajaccio ?",
            "mode": "router",
            "description": "Question spécifique commune → devrait router vers VECTOR"
        },
        {
            "question": "Quelles sont les dimensions du bien-être selon l'ontologie ?",
            "mode": "router",
            "description": "Question conceptuelle → devrait router vers GRAPH"
        },
        {
            "question": "Compare la santé et le logement à Bastia",
            "mode": "sub_question",
            "description": "Question complexe → décomposition en sous-questions"
        },
        {
            "question": "Qu'est-ce que la dimension santé et quels sont ses scores à Ajaccio ?",
            "mode": "hybrid",
            "description": "Question mixte → fusion vector + graph"
        }
    ]

    for i, test in enumerate(tests, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST {i}/4: {test['description']}")
        print(f"Question: {test['question']}")
        print(f"Mode: {test['mode']}")
        print(f"{'='*80}")

        try:
            response, sources = rag.query(
                question=test['question'],
                mode=test['mode']
            )

            print(f"\n✅ RÉPONSE ({len(response)} chars):")
            print(response[:500])
            if len(response) > 500:
                print("... (tronqué)")

            print(f"\n📚 SOURCES ({len(sources)}):")
            for j, source in enumerate(sources[:3], 1):
                print(f"\n  {j}. Score: {source['score']:.3f} | Type: {source['source_type']}")
                print(f"     Metadata: {source['metadata']}")
                print(f"     Content: {source['content'][:150]}...")

        except Exception as e:
            print(f"\n❌ ERREUR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n\n{'='*80}")
    print("TEST TERMINÉ")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_v7()
