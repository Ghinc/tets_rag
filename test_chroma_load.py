"""
Test script to debug ChromaDB loading issue
"""
import os
from rag_v2_improved import ImprovedRAGPipeline

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("="*80)
print("TEST: Initialisation du RAG pipeline")
print("="*80)

try:
    rag = ImprovedRAGPipeline(
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        llm_model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )

    print("\n" + "="*80)
    print("RÉSULTAT DE L'INITIALISATION")
    print("="*80)
    print(f"Documents chargés: {len(rag.documents)}")
    print(f"Hybrid retriever initialisé: {rag.hybrid_retriever is not None}")
    print(f"Collection count: {rag.collection.count()}")

    if len(rag.documents) > 0:
        print(f"\nPremier document:")
        print(f"  - Contenu: {rag.documents[0].page_content[:100]}...")
        print(f"  - Metadata: {rag.documents[0].metadata}")

    print("\n" + "="*80)
    print("TEST DE REQUÊTE")
    print("="*80)

    if rag.hybrid_retriever is not None:
        answer, results = rag.query(
            "Parle-moi de la commune d'Afa",
            k=3,
            use_reranking=True,
            include_quantitative=False
        )

        print(f"\nNombre de résultats: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result.score:.3f}")
            print(f"    Commune: {result.metadata.get('nom', 'N/A')}")
            print(f"    Texte: {result.text[:100]}...")

        print(f"\nRéponse LLM:\n{answer}")
    else:
        print("ERREUR: Hybrid retriever n'a pas été initialisé!")

except Exception as e:
    print(f"\nERREUR: {e}")
    import traceback
    traceback.print_exc()
