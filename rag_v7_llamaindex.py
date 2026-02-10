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
from llama_index.core.tools import QueryEngineTool, ToolMetadata
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
                print(f"  [BOOST] {doc_commune}: {node_with_score.score:.3f} -> {score:.3f}")

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


class RobustRouterEngine:
    """
    Wrapper pour RouterQueryEngine qui gère les erreurs d'encodage charmap
    sur Windows (caractères Unicode comme '→').

    Note: L'erreur charmap peut survenir lors du print interne de LlamaIndex,
    pas uniquement lors de la génération de la réponse. Ce wrapper redirige
    stdout temporairement pour éviter ces erreurs.
    """

    def __init__(self, router_engine, fallback_engine):
        """
        Args:
            router_engine: RouterQueryEngine original
            fallback_engine: Query engine de fallback (vector-only)
        """
        self._engine = router_engine
        self._fallback = fallback_engine

    def query(self, query_str: str):
        """
        Exécute la requête avec gestion des erreurs d'encodage.
        Capture stdout pour éviter les erreurs charmap des prints internes.
        """
        import sys
        import io

        # Rediriger stdout vers un buffer UTF-8 pour éviter les erreurs charmap
        old_stdout = sys.stdout
        try:
            # Créer un buffer qui accepte tout caractère Unicode
            sys.stdout = io.StringIO()
            result = self._engine.query(query_str)
            # Restaurer stdout et ignorer le contenu capturé (qui pourrait contenir des caractères problématiques)
            sys.stdout = old_stdout
            return result
        except UnicodeEncodeError as e:
            sys.stdout = old_stdout
            print(f"  [FALLBACK] Erreur encodage - utilisation vector-only")
            return self._fallback.query(query_str)
        except Exception as e:
            sys.stdout = old_stdout
            if 'charmap' in str(e) or 'encode' in str(e).lower():
                print(f"  [FALLBACK] Erreur charmap - utilisation vector-only")
                return self._fallback.query(query_str)
            raise

    async def aquery(self, query_str: str):
        """Version asynchrone avec fallback"""
        import sys
        import io

        old_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            result = await self._engine.aquery(query_str)
            sys.stdout = old_stdout
            return result
        except (UnicodeEncodeError, Exception) as e:
            sys.stdout = old_stdout
            if isinstance(e, UnicodeEncodeError) or 'charmap' in str(e):
                print(f"  [FALLBACK] Erreur encodage - utilisation vector-only")
                return await self._fallback.aquery(query_str)
            raise


class RobustSubQuestionEngine:
    """
    Wrapper pour SubQuestionQueryEngine qui gère les KeyError causés par
    le LLM qui génère des noms de tools incorrects.

    Le LLM peut générer 'Ontologie BE-2010' au lieu de 'ontology_search',
    causant une KeyError. Ce wrapper intercepte ces erreurs et fait un
    fallback vers une requête directe sur vector_search.
    """

    def __init__(self, sub_question_engine, fallback_engine, tool_name_mapping: Dict[str, str] = None):
        """
        Args:
            sub_question_engine: SubQuestionQueryEngine original
            fallback_engine: Query engine de fallback (vector-only)
            tool_name_mapping: Mapping des noms incorrects vers les noms corrects
        """
        self._engine = sub_question_engine
        self._fallback = fallback_engine
        self._mapping = tool_name_mapping or {
            'Ontologie BE-2010': 'ontology_search',
            'ontologie_be_2010': 'ontology_search',
            'ontologie': 'ontology_search',
            'graph': 'ontology_search',
            'vector': 'vector_search',
            'documents': 'vector_search',
        }

    def query(self, query_str: str):
        """
        Exécute la requête avec gestion des erreurs.
        En cas de KeyError, utilise le fallback engine.
        """
        try:
            return self._engine.query(query_str)
        except KeyError as e:
            # Le LLM a généré un nom de tool incorrect
            error_tool = str(e).strip("'\"")
            print(f"  [FALLBACK] KeyError sur tool '{error_tool}' - utilisation vector-only")
            return self._fallback.query(query_str)
        except Exception as e:
            # Autre erreur - utiliser le fallback
            print(f"  [FALLBACK] Erreur SubQuestion ({type(e).__name__}) - utilisation vector-only")
            return self._fallback.query(query_str)

    async def aquery(self, query_str: str):
        """Version asynchrone avec fallback"""
        try:
            return await self._engine.aquery(query_str)
        except (KeyError, Exception) as e:
            print(f"  [FALLBACK] Erreur SubQuestion ({type(e).__name__}) - utilisation vector-only")
            return await self._fallback.aquery(query_str)


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

        # Tools pour le router avec descriptions et noms explicites
        vector_tool = QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_search",
                description=(
                    "Utile pour repondre a des questions sur les donnees statistiques, "
                    "scores de bien-etre, verbatims d'entretiens, resultats de questionnaires, "
                    "et informations specifiques aux communes corses. "
                    "Contient 5815 documents incluant fiches communes, entretiens qualitatifs, "
                    "verbatims et donnees quantitatives."
                )
            )
        )

        graph_tool = QueryEngineTool(
            query_engine=graph_query_engine,
            metadata=ToolMetadata(
                name="ontology_search",
                description=(
                    "Utile pour repondre a des questions sur les concepts de bien-etre, "
                    "les dimensions theoriques (sante, logement, education, etc.), "
                    "les indicateurs, et les relations semantiques "
                    "de l'ontologie BE-2010 (Better Life Index). "
                    "Contient 116 noeuds d'ontologie avec relations."
                )
            )
        )

        # Router qui choisit automatiquement avec LLM
        # verbose=False pour éviter les erreurs d'encodage charmap sur Windows
        router_query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[vector_tool, graph_tool],
            verbose=False
        )

        # Wrapper robuste avec fallback vers vector-only en cas d'erreur charmap
        robust_router = RobustRouterEngine(
            router_engine=router_query_engine,
            fallback_engine=vector_query_engine
        )

        print("  [OK] RouterQueryEngine créé (vector+graph) avec fallback robuste")
        return robust_router

    def _create_sub_question_engine(self):
        """
        Crée SubQuestionQueryEngine qui décompose
        automatiquement les questions complexes

        Si graph indisponible, utilise vector-only

        Note: Utilise LLMQuestionGenerator (intégré) au lieu de OpenAIQuestionGenerator
        (qui nécessite llama-index-question-gen-openai)

        Returns:
            SubQuestionQueryEngine ou query engine vector-only
        """
        # Import du question generator intégré
        from llama_index.core.question_gen import LLMQuestionGenerator

        # Query engine vectoriel
        vector_query_engine = self.vector_index.as_query_engine(similarity_top_k=10)

        # IMPORTANT: Définir explicitement le nom du tool pour éviter KeyError
        # Le SubQuestionQueryEngine utilise le nom du tool pour router les sous-questions
        vector_tool = QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_search",
                description="Recherche dans les documents textuels des communes corses: statistiques, entretiens, verbatims, données démographiques"
            )
        )

        # Créer le question generator avec le LLM configuré
        question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)

        # Response synthesizer
        response_synthesizer = get_response_synthesizer()

        # Si graph indisponible, retourner vector-only
        if not self.graph_available:
            # Créer SubQuestionQueryEngine manuellement (évite l'import de question-gen-openai)
            sub_question_engine = SubQuestionQueryEngine(
                question_gen=question_gen,
                response_synthesizer=response_synthesizer,
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

        # IMPORTANT: Définir explicitement le nom du tool pour éviter KeyError
        graph_tool = QueryEngineTool(
            query_engine=graph_query_engine,
            metadata=ToolMetadata(
                name="ontology_search",
                description="Recherche dans l'ontologie BE-2010: concepts de bien-être, dimensions théoriques, indicateurs, relations sémantiques"
            )
        )

        # Sub-question engine créé manuellement (évite l'import de question-gen-openai)
        sub_question_engine = SubQuestionQueryEngine(
            question_gen=question_gen,
            response_synthesizer=response_synthesizer,
            query_engine_tools=[vector_tool, graph_tool],
            verbose=True,
            use_async=False  # Évite les problèmes de coroutines
        )

        # Wrapper robuste avec fallback vers vector-only en cas de KeyError
        # (le LLM peut générer 'Ontologie BE-2010' au lieu de 'ontology_search')
        robust_engine = RobustSubQuestionEngine(
            sub_question_engine=sub_question_engine,
            fallback_engine=vector_query_engine
        )

        print("  [OK] SubQuestionQueryEngine créé (vector+graph) avec fallback robuste")
        return robust_engine

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
        Utilise ToolMetadata explicite et verbose=False pour éviter les erreurs
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

        # Créer les tools avec noms explicites
        vector_tool = QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_search",
                description="Documents textuels sur les communes corses: statistiques, entretiens, verbatims"
            )
        )

        graph_tool = QueryEngineTool(
            query_engine=graph_query_engine,
            metadata=ToolMetadata(
                name="ontology_search",
                description="Ontologie BE-2010: concepts de bien-être, dimensions théoriques, indicateurs"
            )
        )

        # Router engine avec verbose=False pour éviter erreurs charmap
        router_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[vector_tool, graph_tool],
            verbose=False
        )

        # Wrapper robuste avec fallback
        robust_router = RobustRouterEngine(
            router_engine=router_engine,
            fallback_engine=vector_query_engine
        )

        return robust_router.query(question)

    def _query_subquestion_with_boost(self, question: str, commune_filter: str):
        """
        Query sub-question avec boost de commune
        Utilise LLMQuestionGenerator (intégré) au lieu de OpenAIQuestionGenerator
        """
        if not self.sub_question_available:
            raise ValueError("SubQuestionQueryEngine non disponible")

        from llama_index.core.question_gen import LLMQuestionGenerator

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

        # Créer les tools avec noms explicites pour éviter KeyError
        vector_tool = QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_search",
                description="Documents textuels sur les communes corses: statistiques, entretiens, verbatims"
            )
        )

        graph_tool = QueryEngineTool(
            query_engine=graph_query_engine,
            metadata=ToolMetadata(
                name="ontology_search",
                description="Ontologie BE-2010: concepts de bien-être, dimensions théoriques, indicateurs"
            )
        )

        # Créer le question generator avec le LLM configuré
        question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)
        response_synthesizer = get_response_synthesizer()

        # Sub-question engine avec wrapper robuste
        sub_question_engine = SubQuestionQueryEngine(
            question_gen=question_gen,
            response_synthesizer=response_synthesizer,
            query_engine_tools=[vector_tool, graph_tool],
            verbose=True,
            use_async=False
        )

        # Wrapper robuste avec fallback
        robust_engine = RobustSubQuestionEngine(
            sub_question_engine=sub_question_engine,
            fallback_engine=vector_query_engine
        )

        return robust_engine.query(question)

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
