"""
RAG v7 - LlamaIndex Pipeline

Architecture:
- VectorStoreIndex sur ChromaDB (5815 docs)
- PropertyGraphIndex sur Neo4j (ontologie BE-2010)
- RouterQueryEngine pour routing intelligent
- SubQuestionQueryEngine pour questions complexes
- Custom HybridRetriever pour fusion vector+graph

Auteur: Claude Code
Date: 2026-01-06
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

# Import local
from commune_detector import detect_commune
from cypher_graph_retriever import CypherGraphRetriever  # Custom retriever pour ontologie BE-2010


class CommuneBoostRetriever(BaseRetriever):
    """
    Wrapper retriever qui applique un boost de score pour une commune spécifique
    """

    def __init__(self, base_retriever: BaseRetriever, commune_filter: Optional[str] = None, boost_factor: float = 1.5):
        """
        Args:
            base_retriever: Retriever de base (VectorIndexRetriever)
            commune_filter: Commune à booster (optionnel)
            boost_factor: Facteur de multiplication du score (défaut: 1.5x)
        """
        super().__init__()
        self._base_retriever = base_retriever
        self._commune_filter = commune_filter
        self._boost_factor = boost_factor

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        """
        Récupère et applique le boost de commune
        """
        # Récupérer les résultats du retriever de base
        nodes = self._base_retriever.retrieve(query_bundle)

        # Si pas de filtre de commune, retourner tel quel
        if not self._commune_filter:
            return nodes

        # Appliquer le boost aux documents de la commune
        boosted_nodes = []
        for node_with_score in nodes:
            score = node_with_score.score if node_with_score.score else 0.0

            # Vérifier si le document appartient à la commune
            metadata = node_with_score.node.metadata
            doc_commune = metadata.get('nom') or metadata.get('commune')

            if doc_commune and doc_commune.lower() == self._commune_filter.lower():
                # Appliquer le boost
                score = score * self._boost_factor
                print(f"  [BOOST] {doc_commune}: {node_with_score.score:.3f} → {score:.3f}")

            # Créer un nouveau NodeWithScore avec le score boosté
            boosted_node = NodeWithScore(
                node=node_with_score.node,
                score=score
            )
            boosted_nodes.append(boosted_node)

        # Re-trier par score après boost
        boosted_nodes.sort(key=lambda x: x.score if x.score else 0.0, reverse=True)

        return boosted_nodes


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
        neo4j_password: str = "",
        populate_graph: bool = False
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
            populate_graph: Si True, peuple le graphe depuis les documents (lent, 1ère fois uniquement)
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

        # 2. Initialiser CypherGraphRetriever (Neo4j) - OPTIONNEL
        print("\n2. Initialisation CypherGraphRetriever (Neo4j)...")
        try:
            self.graph_retriever = CypherGraphRetriever(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                similarity_top_k=5
            )
            self.graph_available = True
            print("  [OK] CypherGraphRetriever disponible (ontologie BE-2010)")
        except Exception as e:
            print(f"  [WARNING] CypherGraphRetriever non disponible: {str(e)[:100]}")
            print("  [INFO] V7 fonctionnera en mode vector-only")
            self.graph_retriever = None
            self.graph_available = False

        # 3. Créer les query engines
        print("\n3. Création des query engines...")
        self.router_engine = self._create_router_engine()

        # Sub-question engine: OPTIONNEL (nécessite llama-index-question-gen-openai)
        try:
            self.sub_question_engine = self._create_sub_question_engine()
            self.sub_question_available = True
        except ImportError as e:
            print(f"  [WARNING] SubQuestionQueryEngine non disponible: {e}")
            print("  [INFO] Installez: pip install llama-index-question-gen-openai")
            self.sub_question_engine = None
            self.sub_question_available = False

        self.hybrid_retriever = self._create_hybrid_retriever()

        print(f"\n{'='*80}")
        print("[OK] RAG v7 LlamaIndex initialisé")
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
        password: str,
        populate_from_vector: bool = False
    ) -> PropertyGraphIndex:
        """
        Initialise PropertyGraphIndex depuis Neo4j

        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
            populate_from_vector: Si True, peuple le graphe depuis les documents vectoriels

        Returns:
            PropertyGraphIndex
        """
        print(f"  Neo4j URI: {uri}")
        print(f"  Neo4j User: {user}")

        # Connexion Neo4j avec patch pour Neo4j 5.x
        graph_store = Neo4j5PropertyGraphStore(
            username=user,
            password=password,
            url=uri,
            database="neo4j"
        )

        print("  [OK] Connexion Neo4j établie")

        # Si on veut peupler depuis les documents vectoriels
        if populate_from_vector:
            print("  [INFO] Peuplement du graphe depuis les documents...")
            print("  [INFO] Récupération des documents depuis ChromaDB...")

            # Récupérer tous les documents depuis ChromaDB directement
            # On ne peut pas utiliser docstore car ChromaVectorStore ne le peuple pas
            from llama_index.core.schema import TextNode
            import chromadb

            # Accéder à ChromaDB
            chroma_client = chromadb.PersistentClient(path="./chroma_v2")
            chroma_collection = chroma_client.get_collection("communes_corses_v2")

            # Récupérer tous les documents
            results = chroma_collection.get(include=["documents", "metadatas"])

            # Convertir en TextNodes pour LlamaIndex
            all_nodes = []
            for i, (doc_id, text, metadata) in enumerate(zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            )):
                node = TextNode(
                    text=text,
                    id_=doc_id,
                    metadata=metadata or {}
                )
                all_nodes.append(node)

            print(f"  [INFO] {len(all_nodes)} documents à traiter")

            # ATTENTION: L'extraction prend du temps (appels LLM)
            # On limite à 10 docs pour le test initial
            sample_nodes = all_nodes[:10]
            print(f"  [INFO] Traitement de {len(sample_nodes)} documents (échantillon)")
            print("  [INFO] Extraction des triplets (entités + relations) via LLM...")
            print("  [WARNING] Cette opération va prendre ~1-2 minutes...")

            # Créer PropertyGraphIndex depuis les documents
            # from_documents() est la méthode recommandée pour peupler le graphe
            # Note: Document est déjà importé via llama_index.core

            # Convertir les TextNodes en Documents pour from_documents()
            from llama_index.core.schema import Document

            documents = [
                Document(
                    text=node.text,
                    metadata=node.metadata,
                    id_=node.id_
                )
                for node in sample_nodes
            ]

            print(f"  [INFO] Construction du PropertyGraphIndex avec extraction des triplets...")
            property_graph_index = PropertyGraphIndex.from_documents(
                documents=documents,
                property_graph_store=graph_store,
                embed_model=Settings.embed_model,
                show_progress=True
            )

            print(f"  [OK] PropertyGraphIndex peuplé avec {len(sample_nodes)} documents")

        else:
            # Mode normal : essayer de charger depuis graphe existant
            try:
                property_graph_index = PropertyGraphIndex.from_existing(
                    property_graph_store=graph_store,
                    embed_model=Settings.embed_model
                )
                print("  [OK] PropertyGraphIndex créé depuis graphe existant")
            except Exception as e:
                print(f"  [WARNING] Impossible de charger depuis graphe existant: {e}")
                print("  [INFO] Création d'un index vide")
                # Créer un index vide connecté au graph store
                property_graph_index = PropertyGraphIndex(
                    nodes=[],
                    graph_store=graph_store,
                    embed_model=Settings.embed_model
                )
                print("  [INFO] Pour peupler le graphe, utilisez populate_graph=True")

        return property_graph_index

    def _create_router_engine(self) -> RouterQueryEngine:
        """
        Crée RouterQueryEngine qui route automatiquement
        vers vector OU graph selon la question

        Si graph indisponible, retourne vector-only query engine

        Returns:
            RouterQueryEngine ou query engine vector-only
        """
        # Query engine vectoriel
        vector_query_engine = self.vector_index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ]
        )

        # Si graph indisponible, retourner vector-only
        if not self.graph_available:
            print("  [OK] Router créé (vector-only mode)")
            return vector_query_engine

        # Graph disponible : créer query engine depuis le retriever
        graph_query_engine = RetrieverQueryEngine.from_args(
            retriever=self.graph_retriever,
            response_synthesizer=get_response_synthesizer()
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

        Si graph indisponible, utilise vector-only

        Returns:
            SubQuestionQueryEngine ou query engine vector-only
        """
        # Query engine vectoriel
        vector_query_engine = self.vector_index.as_query_engine(similarity_top_k=10)

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Données communes corses (statistiques, entretiens, verbatims)"
        )

        # Si graph indisponible, retourner vector-only
        if not self.graph_available:
            sub_question_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=[vector_tool],
                verbose=True
            )
            print("  [OK] SubQuestionQueryEngine créé (vector-only mode)")
            return sub_question_engine

        # Graph disponible : créer avec vector + graph
        graph_query_engine = RetrieverQueryEngine.from_args(
            retriever=self.graph_retriever,
            response_synthesizer=get_response_synthesizer()
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

        print("  [OK] SubQuestionQueryEngine créé (vector+graph)")
        return sub_question_engine

    def _create_hybrid_retriever(self):
        """
        Crée HybridRetriever custom qui combine
        vector + graph avec fusion pondérée

        Si graph indisponible, retourne vector-only retriever

        Returns:
            HybridGraphVectorRetriever ou VectorIndexRetriever
        """
        # Vector retriever
        vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=10
        )

        # Si graph indisponible, retourner vector-only
        if not self.graph_available:
            print("  [OK] Hybrid retriever créé (vector-only mode)")
            return vector_retriever

        # Graph disponible : créer hybrid complet avec le custom retriever
        # Hybrid retriever avec pondération
        hybrid_retriever = HybridGraphVectorRetriever(
            vector_retriever=vector_retriever,
            graph_retriever=self.graph_retriever,
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
        print(f"REQUÊTE RAG v7 (mode: {mode})")
        print(f"Question: {question}")
        print(f"{'='*80}")

        # Détection automatique de commune
        if not commune_filter:
            detected_commune = detect_commune(question)
            if detected_commune:
                commune_filter = detected_commune
                print(f"[AUTO-DETECT v7] Commune détectée: {commune_filter}")

        # Sélection du query engine selon le mode
        if mode == "router":
            print("\nMode ROUTER: LLM choisit entre vector et graph...")
            # Si commune détectée, recréer le router avec boost
            if commune_filter:
                response = self._query_router_with_boost(question, commune_filter)
            else:
                response = self.router_engine.query(question)

        elif mode == "sub_question":
            if not self.sub_question_available:
                raise ValueError(
                    "SubQuestionQueryEngine non disponible. "
                    "Installez: pip install llama-index-question-gen-openai"
                )
            print("\nMode SUB-QUESTION: Décomposition en sous-questions...")
            # Si commune détectée, recréer avec boost
            if commune_filter:
                response = self._query_subquestion_with_boost(question, commune_filter)
            else:
                response = self.sub_question_engine.query(question)

        elif mode == "hybrid":
            print("\nMode HYBRID: Fusion vector + graph...")
            response = self._query_hybrid(question, k, commune_filter)

        else:
            raise ValueError(f"Mode inconnu: {mode}. Utilisez 'router', 'sub_question' ou 'hybrid'.")

        # Extraire sources
        sources = self._extract_sources(response)

        print(f"\n[OK] Réponse générée avec {len(sources)} sources")

        return response.response, sources

    def _query_router_with_boost(self, question: str, commune_filter: str):
        """
        Query router avec boost de commune
        """
        # Créer retriever vector avec boost
        base_vector_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=10)
        boosted_vector_retriever = CommuneBoostRetriever(
            base_retriever=base_vector_retriever,
            commune_filter=commune_filter,
            boost_factor=1.5
        )

        # Créer query engines
        vector_query_engine = RetrieverQueryEngine.from_args(
            retriever=boosted_vector_retriever,
            response_synthesizer=get_response_synthesizer()
        )

        graph_query_engine = RetrieverQueryEngine.from_args(
            retriever=self.graph_retriever,
            response_synthesizer=get_response_synthesizer()
        )

        # Créer les tools
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Documents textuels sur les communes corses (questionnaires, rapports)"
        )

        graph_tool = QueryEngineTool.from_defaults(
            query_engine=graph_query_engine,
            description="Ontologie BE-2010 : concepts de bien-être, dimensions théoriques, indicateurs"
        )

        # Router engine
        router_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[vector_tool, graph_tool]
        )

        return router_engine.query(question)

    def _query_subquestion_with_boost(self, question: str, commune_filter: str):
        """
        Query sub-question avec boost de commune
        """
        if not self.sub_question_available:
            raise ValueError("SubQuestionQueryEngine non disponible")

        # Créer retriever vector avec boost
        base_vector_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=10)
        boosted_vector_retriever = CommuneBoostRetriever(
            base_retriever=base_vector_retriever,
            commune_filter=commune_filter,
            boost_factor=1.5
        )

        # Créer query engines
        vector_query_engine = RetrieverQueryEngine.from_args(
            retriever=boosted_vector_retriever,
            response_synthesizer=get_response_synthesizer()
        )

        graph_query_engine = RetrieverQueryEngine.from_args(
            retriever=self.graph_retriever,
            response_synthesizer=get_response_synthesizer()
        )

        # Créer les tools
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Documents textuels sur les communes corses"
        )

        graph_tool = QueryEngineTool.from_defaults(
            query_engine=graph_query_engine,
            description="Ontologie BE-2010"
        )

        # Sub-question engine (import dynamique car package optionnel)
        try:
            from llama_index.question_gen.openai import OpenAIQuestionGenerator
            sub_question_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=[vector_tool, graph_tool],
                question_gen=OpenAIQuestionGenerator.from_defaults()
            )
            return sub_question_engine.query(question)
        except ImportError:
            raise ValueError("llama-index-question-gen-openai package requis")

    def _query_hybrid(self, question: str, k: int, commune_filter: Optional[str] = None):
        """
        Query avec hybrid retriever custom

        Args:
            question: Question
            k: Nombre de résultats
            commune_filter: Commune pour boost (optionnel)

        Returns:
            Response object
        """
        # Si commune détectée, créer retriever avec boost
        if commune_filter:
            base_vector_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=10)
            boosted_vector_retriever = CommuneBoostRetriever(
                base_retriever=base_vector_retriever,
                commune_filter=commune_filter,
                boost_factor=1.5
            )

            # Créer hybrid retriever avec boost
            hybrid_retriever = HybridGraphVectorRetriever(
                vector_retriever=boosted_vector_retriever,
                graph_retriever=self.graph_retriever,
                vector_weight=0.4,
                graph_weight=0.6,
                similarity_top_k=10
            )
        else:
            # Utiliser hybrid retriever par défaut
            hybrid_retriever = self.hybrid_retriever

        # Query engine avec hybrid retriever
        response_synthesizer = get_response_synthesizer(
            response_mode="compact"
        )

        hybrid_query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
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
