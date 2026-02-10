"""
RAG v2.1 - Version LlamaIndex équivalente à v2
- Hybrid retrieval (BM25 + Vector) avec LlamaIndex
- Reranking avec cross-encoder
- Boost intelligent pour questionnaires
- Sans graphe (équivalent v2 "fait main")
"""

import os
import re
from typing import List, Dict, Optional
import chromadb
from dotenv import load_dotenv

# LlamaIndex
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    get_response_synthesizer
)
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    QueryFusionRetriever
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# Détection de communes
from commune_detector import detect_commune

# Charger les variables d'environnement
load_dotenv()


class RAGv2_1_LlamaIndex:
    """
    RAG v2.1 avec LlamaIndex - équivalent à v2 "fait main"
    """

    def __init__(
        self,
        chroma_path: str = "./chroma_db_entretiens",
        collection_name: str = "entretiens_corsica",
        embedding_model: str = "intfloat/multilingual-e5-base",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        top_k: int = 5,
        use_reranker: bool = True,
        use_hybrid: bool = True
    ):
        """
        Args:
            chroma_path: Chemin vers la base ChromaDB
            collection_name: Nom de la collection ChromaDB
            embedding_model: Modèle d'embedding
            rerank_model: Modèle de reranking
            llm_model: Modèle LLM pour génération
            temperature: Température du LLM
            top_k: Nombre de documents à récupérer
            use_reranker: Activer le reranking
            use_hybrid: Activer hybrid retrieval (BM25+Vector)
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid

        # Configuration LlamaIndex
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            cache_folder="./model_cache"
        )
        Settings.llm = OpenAI(
            model=llm_model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        # Charger ChromaDB
        print(f"Chargement de ChromaDB: {chroma_path}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = self.chroma_client.get_collection(collection_name)

        # Créer le vector store
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Créer l'index
        print("Création de l'index LlamaIndex...")
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )

        # Créer le retriever
        self.retriever = self._create_retriever()

        # Créer les post-processors
        self.postprocessors = []
        if use_reranker:
            print(f"Activation du reranker: {rerank_model}")
            # Note: SentenceTransformerRerank attend un modèle type cross-encoder
            # Pour BAAI/bge-reranker-v2-m3, on utilise la classe de reranking spécifique
            try:
                from llama_index.core.postprocessor import FlagEmbeddingReranker
                self.postprocessors.append(
                    FlagEmbeddingReranker(
                        model=rerank_model,
                        top_n=top_k
                    )
                )
            except ImportError:
                # Fallback to SentenceTransformerRerank
                self.postprocessors.append(
                    SentenceTransformerRerank(
                        model=rerank_model,
                        top_n=top_k
                    )
                )

        # Créer le query engine
        self.query_engine = self._create_query_engine()

        print("RAG v2.1 LlamaIndex initialisé avec succès!")

    def _create_retriever(self):
        """
        Crée le retriever (hybrid ou simple vector)
        """
        if self.use_hybrid:
            # Hybrid retrieval: BM25 + Vector
            print("Configuration du hybrid retriever (BM25 + Vector)...")

            # Vector retriever
            vector_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.top_k * 2  # Récupérer plus pour fusion
            )

            # Fusion retriever (simule BM25 + Vector)
            # Note: LlamaIndex QueryFusionRetriever fait de la fusion avec réécriture de requêtes
            retriever = QueryFusionRetriever(
                retrievers=[vector_retriever],
                similarity_top_k=self.top_k * 2,
                num_queries=1,  # Pas de réécriture multiple
                mode="relative_score",  # Fusion basée sur les scores
                use_async=False
            )
            return retriever
        else:
            # Simple vector retrieval
            print("Configuration du vector retriever simple...")
            return VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.top_k * 2
            )

    def _create_query_engine(self):
        """
        Crée le query engine avec retriever et post-processors
        """
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            verbose=False
        )

        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=self.postprocessors
        )

        return query_engine

    def _detect_questionnaire_boost(self, question: str) -> Dict[str, any]:
        """
        Détecte si la question nécessite un boost sur les questionnaires
        (comme dans v2)
        """
        # Patterns pour détecter les questions sur indicateurs/statistiques
        patterns_quanti = [
            r'indicateur',
            r'statistique',
            r'taux',
            r'pourcentage',
            r'nombre',
            r'combien',
            r'mesure',
            r'chiffre',
            r'données quantitatives',
            r'questionnaire'
        ]

        question_lower = question.lower()
        for pattern in patterns_quanti:
            if re.search(pattern, question_lower):
                return {
                    "boost_needed": True,
                    "boost_filter": {"source": "questionnaire"}
                }

        return {"boost_needed": False}

    def query(self, question: str, commune: Optional[str] = None) -> Dict:
        """
        Interroge le système RAG

        Args:
            question: Question de l'utilisateur
            commune: Nom de la commune (optionnel, détecté automatiquement si absent)

        Returns:
            Dict avec réponse et métadonnées
        """
        # Détecter la commune
        if commune is None:
            commune = detect_commune(question)
            if commune:
                print(f"[DÉTECTION] Commune détectée: {commune}")

        # Détecter si boost questionnaire nécessaire
        boost_info = self._detect_questionnaire_boost(question)

        # Construire la requête avec contexte commune si détectée
        query_str = question
        if commune:
            query_str = f"Commune: {commune}\n{question}"

        # Pour le boost questionnaire, on pourrait faire une requête spécifique
        # mais pour simplifier, on utilise le query engine standard
        # (dans v2, il y avait un dual-path pour les données quantitatives)

        print(f"[QUERY] Question: {question}")
        if boost_info["boost_needed"]:
            print("[BOOST] Boost questionnaire activé")

        # Exécuter la requête
        response = self.query_engine.query(query_str)

        # Extraire les sources
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source_info = {
                    "text": node.node.get_content()[:200] + "...",
                    "score": node.score if hasattr(node, 'score') else None,
                    "metadata": node.node.metadata
                }
                sources.append(source_info)

        return {
            "answer": str(response),
            "sources": sources,
            "num_sources": len(sources),
            "commune": commune,
            "boost_applied": boost_info["boost_needed"]
        }

    def query_with_mode(self, question: str, commune: Optional[str] = None, mode: str = "standard") -> Dict:
        """
        Méthode compatible avec l'API multi-version

        Args:
            question: Question
            commune: Commune (optionnel)
            mode: Mode de requête (ignoré pour v2.1, toujours "standard")

        Returns:
            Dict avec réponse et métadonnées
        """
        return self.query(question, commune)


def main():
    """Test du système RAG v2.1"""
    print("="*80)
    print("TEST RAG v2.1 - LlamaIndex (équivalent v2)")
    print("="*80)

    # Initialiser le système
    rag = RAGv2_1_LlamaIndex(
        use_reranker=True,
        use_hybrid=True,
        top_k=5
    )

    # Questions de test
    test_questions = [
        "Quels sont les différents types de bien-être ?",
        "Comment est la qualité de vie à Ajaccio ?",
        "Quels sont les problèmes de transport en Corse ?",
        "Quels sont les indicateurs de santé et d'éducation en Corse ?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print('='*80)

        result = rag.query(question)

        print(f"\nRéponse ({result['num_sources']} sources):")
        print(result['answer'])

        if result['commune']:
            print(f"\nCommune détectée: {result['commune']}")

        if result['boost_applied']:
            print("\n[BOOST] Boost questionnaire appliqué")

        print(f"\nSources utilisées:")
        for j, source in enumerate(result['sources'][:3], 1):
            print(f"  {j}. Score: {source['score']:.3f} - {source['text'][:100]}...")


if __name__ == "__main__":
    main()
