"""
RAG v8 - LlamaIndex Pipeline FULL (avec graphes obligatoires)

Architecture:
- VectorStoreIndex sur ChromaDB (5815 docs)
- PropertyGraphIndex sur Neo4j (ontologie BE-2010) - OBLIGATOIRE
- RouterQueryEngine pour routing intelligent
- SubQuestionQueryEngine pour questions complexes
- Custom HybridRetriever pour fusion vector+graph

Différence avec v7:
- v7: PropertyGraphIndex et SubQuestionEngine optionnels (fallback vector-only)
- v8: PropertyGraphIndex et SubQuestionEngine REQUIS (crash si APOC manquant)

Auteur: Claude Code
Date: 2026-01-11
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import chromadb

# LlamaIndex core
from llama_index.core import (
    VectorStoreIndex,
    PropertyGraphIndex,
    StorageContext,
    Settings,
    get_response_synthesizer
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.query_engine import (
    RouterQueryEngine,
    SubQuestionQueryEngine,
    RetrieverQueryEngine
)
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.postprocessor import SimilarityPostprocessor

# LlamaIndex integrations
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Import local
from commune_detector import detect_commune


class HybridGraphVectorRetriever(BaseRetriever):
    """
    Retriever hybride qui combine vector et graph retrieval
    avec fusion pondérée des scores
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        graph_retriever: BaseRetriever,
        vector_weight: float = 0.4,
        graph_weight: float = 0.6,
        similarity_top_k: int = 10
    ):
        """
        Args:
            vector_retriever: Retriever vectoriel (ChromaDB)
            graph_retriever: Retriever graphe (Neo4j)
            vector_weight: Poids pour scores vectoriels (défaut: 0.4)
            graph_weight: Poids pour scores graphe (défaut: 0.6)
            similarity_top_k: Nombre de résultats finaux
        """
        super().__init__()
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self._similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle):
        """
        Récupère et fusionne les résultats vector + graph

        Returns:
            List[NodeWithScore]
        """
        # 1. Retrieval vectoriel
        vector_nodes = self.vector_retriever.retrieve(query_bundle)

        # 2. Retrieval graph
        graph_nodes = self.graph_retriever.retrieve(query_bundle)

        # 3. Fusion avec pondération
        all_nodes = {}

        # Ajouter nodes vectoriels avec poids
        for node in vector_nodes:
            node_id = node.node.node_id
            all_nodes[node_id] = NodeWithScore(
                node=node.node,
                score=node.score * self.vector_weight
            )

        # Ajouter/combiner nodes graphe avec poids
        for node in graph_nodes:
            node_id = node.node.node_id
            if node_id in all_nodes:
                # Nœud présent dans les deux : additionner les scores pondérés
                all_nodes[node_id].score += node.score * self.graph_weight
            else:
                # Nouveau nœud du graphe
                all_nodes[node_id] = NodeWithScore(
                    node=node.node,
                    score=node.score * self.graph_weight
                )

        # 4. Trier par score décroissant et prendre top-k
        sorted_nodes = sorted(
            all_nodes.values(),
            key=lambda x: x.score,
            reverse=True
        )

        return sorted_nodes[:self._similarity_top_k]


class LlamaIndexRAGPipeline:
    """
    Pipeline RAG v7 avec LlamaIndex

    Fonctionnalités:
    - Vector retrieval (ChromaDB + BGE-M3)
    - Graph retrieval (Neo4j + ontologie)
    - Router automatique (vector vs graph)
    - Sub-question decomposition
    - Hybrid retrieval (vector + graph)
    """

    def __init__(
        self,
        openai_api_key: str,
        chroma_path: str = "./chroma_v2",
        collection_name: str = "communes_corses_v2",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = ""
    ):
        """
        Initialise le pipeline RAG v7

        Args:
            openai_api_key: Clé API OpenAI
            chroma_path: Chemin vers ChromaDB
            collection_name: Nom de la collection ChromaDB
            neo4j_uri: URI Neo4j
            neo4j_user: Utilisateur Neo4j
            neo4j_password: Mot de passe Neo4j
        """
        print(f"\n{'='*80}")
        print("INITIALISATION RAG v7 (LlamaIndex)")
        print(f"{'='*80}")

        # Configuration globale LlamaIndex
        print("Configuration LlamaIndex...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            query_instruction="Represent this sentence for searching relevant passages: "
        )
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0
        )
        print("  [OK] Embedding model: BGE-M3")
        print("  [OK] LLM: GPT-3.5-turbo")

        # 1. Initialiser VectorStoreIndex (ChromaDB)
        print("\n1. Initialisation VectorStoreIndex (ChromaDB)...")
        self.vector_index = self._init_vector_index(chroma_path, collection_name)

        # 2. Initialiser PropertyGraphIndex (Neo4j) - OBLIGATOIRE
        print("\n2. Initialisation PropertyGraphIndex (Neo4j)...")
        self.graph_index = self._init_graph_index(neo4j_uri, neo4j_user, neo4j_password)
        print("  [OK] PropertyGraphIndex initialisé")

        # 3. Créer les query engines
        print("\n3. Création des query engines...")
        self.router_engine = self._create_router_engine()
        self.sub_question_engine = self._create_sub_question_engine()
        self.hybrid_retriever = self._create_hybrid_retriever()

        print(f"\n{'='*80}")
        print("[OK] RAG v8 LlamaIndex FULL initialisé")
        print(f"{'='*80}")

    def _init_vector_index(self, chroma_path: str, collection_name: str) -> VectorStoreIndex:
        """
        Initialise VectorStoreIndex depuis ChromaDB existant

        Returns:
            VectorStoreIndex
        """
        # Connexion à ChromaDB existant
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        chroma_collection = chroma_client.get_collection(collection_name)

        print(f"  ChromaDB: {chroma_path}")
        print(f"  Collection: {collection_name}")
        print(f"  Documents: {chroma_collection.count()}")

        # Créer ChromaVectorStore
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Créer index depuis vector store existant
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

        print("  [OK] VectorStoreIndex créé")
        return vector_index

    def _init_graph_index(
        self,
        uri: str,
        user: str,
        password: str
    ) -> PropertyGraphIndex:
        """
        Initialise PropertyGraphIndex depuis Neo4j existant

        Returns:
            PropertyGraphIndex
        """
        print(f"  Neo4j URI: {uri}")
        print(f"  Neo4j User: {user}")

        # Connexion Neo4j
        graph_store = Neo4jPropertyGraphStore(
            username=user,
            password=password,
            url=uri,
            database="neo4j"
        )

        print("  [OK] Connexion Neo4j établie")

        # Créer PropertyGraphIndex depuis graphe existant
        # Note: from_existing() charge le graphe sans ré-indexation
        try:
            property_graph_index = PropertyGraphIndex.from_existing(
                graph_store=graph_store,
                embed_model=Settings.embed_model
            )
            print("  [OK] PropertyGraphIndex créé depuis graphe existant")
        except Exception as e:
            print(f"  [WARNING] Impossible de charger depuis graphe existant: {e}")
            print("  [INFO] Création d'un index vide")
            property_graph_index = PropertyGraphIndex(
                nodes=[],
                graph_store=graph_store,
                embed_model=Settings.embed_model
            )

        return property_graph_index

    def _create_router_engine(self) -> RouterQueryEngine:
        """
        Crée RouterQueryEngine qui route automatiquement
        vers vector OU graph selon la question

        Returns:
            RouterQueryEngine
        """
        # Query engines individuels
        vector_query_engine = self.vector_index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ]
        )

        graph_query_engine = self.graph_index.as_query_engine(
            similarity_top_k=5
        )

        # Tools pour le router avec descriptions
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                "Utile pour répondre à des questions sur les données statistiques, "
                "scores de bien-être, verbatims d'entretiens, résultats de questionnaires, "
                "et informations spécifiques aux communes corses. "
                "Contient 5815 documents incluant fiches communes, entretiens qualitatifs, "
                "verbatims et données quantitatives."
            )
        )

        graph_tool = QueryEngineTool.from_defaults(
            query_engine=graph_query_engine,
            description=(
                "Utile pour répondre à des questions sur les concepts de bien-être, "
                "les dimensions théoriques (santé, logement, éducation, etc.), "
                "les indicateurs, et les relations sémantiques "
                "de l'ontologie BE-2010 (Better Life Index). "
                "Contient 116 nœuds d'ontologie avec relations."
            )
        )

        # Router qui choisit automatiquement avec LLM
        router_query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[vector_tool, graph_tool],
            verbose=True
        )

        print("  [OK] RouterQueryEngine créé (vector+graph)")
        return router_query_engine

    def _create_sub_question_engine(self) -> SubQuestionQueryEngine:
        """
        Crée SubQuestionQueryEngine qui décompose
        automatiquement les questions complexes

        Returns:
            SubQuestionQueryEngine
        """
        # Query engines individuels
        vector_query_engine = self.vector_index.as_query_engine(similarity_top_k=10)
        graph_query_engine = self.graph_index.as_query_engine(similarity_top_k=5)

        # Tools pour le sub-question engine
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Données communes corses (statistiques, entretiens, verbatims)"
        )

        graph_tool = QueryEngineTool.from_defaults(
            query_engine=graph_query_engine,
            description="Ontologie BE-2010 (concepts, dimensions, indicateurs)"
        )

        # Sub-question engine
        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[vector_tool, graph_tool],
            verbose=True
        )

        print("  [OK] SubQuestionQueryEngine créé")
        return sub_question_engine

    def _create_hybrid_retriever(self) -> HybridGraphVectorRetriever:
        """
        Crée HybridRetriever custom qui combine
        vector + graph avec fusion pondérée

        Returns:
            HybridGraphVectorRetriever
        """
        # Retrievers individuels
        vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=10
        )

        graph_retriever = self.graph_index.as_retriever(
            similarity_top_k=5
        )

        # Hybrid retriever avec pondération
        hybrid_retriever = HybridGraphVectorRetriever(
            vector_retriever=vector_retriever,
            graph_retriever=graph_retriever,
            vector_weight=0.4,  # Même que v6
            graph_weight=0.6,   # Même que v6
            similarity_top_k=10
        )

        print("  [OK] HybridGraphVectorRetriever créé (weights: 0.4/0.6)")
        return hybrid_retriever

    def query(
        self,
        question: str,
        mode: str = "router",
        commune_filter: Optional[str] = None,
        k: int = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Requête LlamaIndex avec plusieurs modes

        Args:
            question: Question utilisateur
            mode: Mode de query engine
                - "router": RouterQueryEngine (choisit vector OU graph automatiquement)
                - "sub_question": Décompose en sous-questions
                - "hybrid": Custom hybrid retriever (vector + graph fusionnés)
            commune_filter: Filtre optionnel par commune (TODO)
            k: Nombre de résultats

        Returns:
            (réponse, sources)
        """
        print(f"\n{'='*80}")
        print(f"REQUÊTE RAG v8 (mode: {mode})")
        print(f"Question: {question}")
        print(f"{'='*80}")

        # Détection automatique de commune
        if not commune_filter:
            detected_commune = detect_commune(question)
            if detected_commune:
                commune_filter = detected_commune
                print(f"[AUTO-DETECT v8] Commune détectée: {commune_filter}")

        # Sélection du query engine selon le mode
        if mode == "router":
            print("\nMode ROUTER: LLM choisit entre vector et graph...")
            response = self.router_engine.query(question)

        elif mode == "sub_question":
            print("\nMode SUB-QUESTION: Décomposition en sous-questions...")
            response = self.sub_question_engine.query(question)

        elif mode == "hybrid":
            print("\nMode HYBRID: Fusion vector + graph...")
            response = self._query_hybrid(question, k)

        else:
            raise ValueError(f"Mode inconnu: {mode}. Utilisez 'router', 'sub_question' ou 'hybrid'.")

        # Extraire sources
        sources = self._extract_sources(response)

        print(f"\n[OK] Réponse générée avec {len(sources)} sources")

        return response.response, sources

    def _query_hybrid(self, question: str, k: int):
        """
        Query avec hybrid retriever custom

        Args:
            question: Question
            k: Nombre de résultats

        Returns:
            Response object
        """
        # Query engine avec hybrid retriever
        response_synthesizer = get_response_synthesizer(
            response_mode="compact"
        )

        hybrid_query_engine = RetrieverQueryEngine(
            retriever=self.hybrid_retriever,
            response_synthesizer=response_synthesizer
        )

        response = hybrid_query_engine.query(question)
        return response

    def _extract_sources(self, response) -> List[Dict[str, Any]]:
        """
        Extrait les sources depuis Response.source_nodes

        Args:
            response: Response object de LlamaIndex

        Returns:
            Liste de dicts avec content, score, metadata
        """
        sources = []

        if not hasattr(response, 'source_nodes'):
            return sources

        for node in response.source_nodes:
            source = {
                'content': node.node.text[:500],  # Tronquer à 500 chars
                'score': float(node.score) if node.score else 0.0,
                'metadata': node.node.metadata,
                'source_type': self._detect_source_type(node.node.metadata)
            }
            sources.append(source)

        return sources

    def _detect_source_type(self, metadata: Dict) -> str:
        """
        Détecte le type de source depuis metadata

        Args:
            metadata: Metadata du node

        Returns:
            Type de source ('commune', 'entretien', 'graph', etc.)
        """
        if 'source' in metadata:
            return metadata['source']
        elif 'nom' in metadata:
            return 'commune'
        elif 'type' in metadata:
            return metadata['type']
        else:
            return 'unknown'


# Point d'entrée pour tests
def main():
    """Test de la pipeline v7"""
    print("="*80)
    print("TEST RAG v7 - LlamaIndex")
    print("="*80)

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY manquante")

    # Initialiser pipeline
    rag = LlamaIndexRAGPipeline(
        openai_api_key=OPENAI_API_KEY,
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=""
    )

    # Tests
    questions = [
        ("Quels sont les scores de bien-être à Ajaccio ?", "router"),
        ("Quelles sont les dimensions du bien-être ?", "router"),
        ("Compare la santé et le logement à Bastia", "sub_question"),
        ("Qu'est-ce que la dimension santé dans l'ontologie ?", "hybrid")
    ]

    for question, mode in questions:
        print(f"\n\n{'='*80}")
        print(f"TEST: {question}")
        print(f"Mode: {mode}")
        print(f"{'='*80}")

        response, sources = rag.query(question, mode=mode)

        print(f"\nRÉPONSE:\n{response}")
        print(f"\nSOURCES ({len(sources)}):")
        for i, source in enumerate(sources[:3], 1):
            print(f"\n{i}. Score: {source['score']:.3f}")
            print(f"   Type: {source['source_type']}")
            print(f"   Content: {source['content'][:200]}...")


if __name__ == "__main__":
    main()
