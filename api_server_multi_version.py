"""
API FastAPI multi-versions pour les systèmes RAG

Ce serveur permet de choisir dynamiquement entre les versions v1, v2, v3 et v4 du RAG
via l'interface graphique ou les requêtes API.

Endpoints:
- POST /api/query - Poser une question au chatbot (avec choix de version)
- GET /api/health - Vérifier l'état du serveur
- GET /api/versions - Liste des versions disponibles
- GET /docs - Documentation Swagger automatique

Auteur: Claude Code
Date: 2025-11-17
"""

import os
from typing import Optional, List, Dict, Literal
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
from dotenv import load_dotenv

# Imports des différentes versions RAG
from rag_v1_class import BasicRAGPipeline, RetrievalResult as RetrievalResult_v1
from rag_v2_boosted import ImprovedRAGPipeline, RetrievalResult as RetrievalResult_v2
from rag_v3_ontology import RAGPipelineWithOntology, RetrievalResult as RetrievalResult_v3
from rag_v4_cross_analysis import ImprovedRAGPipeline as CrossAnalysisRAGPipeline, RetrievalResult as RetrievalResult_v4
from rag_v5_graphrag_neo4j import GraphRAGPipeline

# Import optionnel de v6 (nécessite torch_geometric)
try:
    from rag_v6_gretriever import GRetrieverRAGPipeline
    V6_AVAILABLE = True
except ImportError:
    print("AVERTISSEMENT: torch_geometric non installé, RAG v6 désactivé")
    GRetrieverRAGPipeline = None
    V6_AVAILABLE = False

# Import optionnel de v7 (nécessite llama-index, graphes optionnels)
try:
    from rag_v7_llamaindex import LlamaIndexRAGPipeline
    V7_AVAILABLE = True
except ImportError as e:
    print(f"AVERTISSEMENT: LlamaIndex non installé, RAG v7 désactivé ({e})")
    LlamaIndexRAGPipeline = None
    V7_AVAILABLE = False

# Import optionnel de v8 (nécessite llama-index + graphes OBLIGATOIRES)
try:
    from rag_v8_llamaindex_full import LlamaIndexRAGPipeline as LlamaIndexRAGPipelineFull
    V8_AVAILABLE = True
except ImportError as e:
    print(f"AVERTISSEMENT: LlamaIndex non installé, RAG v8 désactivé ({e})")
    LlamaIndexRAGPipelineFull = None
    V8_AVAILABLE = False

# Charger les variables d'environnement
load_dotenv()

# === MODELS PYDANTIC ===

class QueryRequest(BaseModel):
    """Modèle de requête pour poser une question"""
    question: str = Field(..., description="Question à poser au chatbot", min_length=1)
    rag_version: Literal["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"] = Field("v2", description="Version du RAG à utiliser")
    k: int = Field(5, description="Nombre de documents à récupérer", ge=1, le=20)
    use_reranking: bool = Field(True, description="Utiliser le reranking (v2/v3/v4 uniquement)")
    include_quantitative: bool = Field(True, description="Inclure les données quantitatives (v2/v3/v4 uniquement)")
    commune_filter: Optional[str] = Field(None, description="Filtrer par commune spécifique")
    use_ontology_enrichment: bool = Field(True, description="Utiliser l'enrichissement ontologique (v3 uniquement)")
    use_cross_analysis: bool = Field(True, description="Activer l'analyse croisée automatique (v4 uniquement)")
    query_mode: Literal["router", "sub_question", "hybrid"] = Field("router", description="Mode de query pour v7/v8 (router/sub_question/hybrid)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Quelles sont les communes avec le meilleur bien-être ?",
                "rag_version": "v2",
                "k": 5,
                "use_reranking": True,
                "include_quantitative": True,
                "use_ontology_enrichment": True
            }
        }
    )


class Source(BaseModel):
    """Modèle pour une source de document"""
    content: str = Field(..., description="Contenu du document")
    score: float = Field(..., description="Score de pertinence")
    metadata: Dict = Field(..., description="Métadonnées du document (commune, source, etc.)")


class QueryResponse(BaseModel):
    """Modèle de réponse à une question"""
    answer: str = Field(..., description="Réponse générée par le chatbot")
    sources: List[Source] = Field(..., description="Sources utilisées pour générer la réponse")
    metadata: Dict = Field(..., description="Métadonnées de la requête")
    rag_version_used: str = Field(..., description="Version du RAG utilisée")
    timestamp: str = Field(..., description="Horodatage de la réponse")


class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check"""
    status: str = Field(..., description="État du serveur")
    rag_v1_initialized: bool = Field(..., description="RAG v1 initialisé")
    rag_v2_initialized: bool = Field(..., description="RAG v2 initialisé")
    rag_v3_initialized: bool = Field(..., description="RAG v3 initialisé")
    rag_v4_initialized: bool = Field(..., description="RAG v4 initialisé")
    rag_v5_initialized: bool = Field(..., description="RAG v5 initialisé")
    rag_v6_initialized: bool = Field(..., description="RAG v6 initialisé")
    rag_v7_initialized: bool = Field(..., description="RAG v7 initialisé")
    timestamp: str = Field(..., description="Horodatage du check")
    version: str = Field(..., description="Version de l'API")


class VersionInfo(BaseModel):
    """Information sur une version du RAG"""
    version: str
    name: str
    description: str
    available: bool
    features: List[str]


# === GESTION DES PIPELINES RAG ===

rag_pipelines = {
    "v1": None,
    "v2": None,
    "v3": None,
    "v4": None,
    "v5": None,
    "v6": None,
    "v7": None
}


def initialize_all_rags():
    """
    Initialise toutes les versions du RAG disponibles

    Tente d'initialiser v1, v2, v3 et v4. Si une version échoue,
    elle reste à None mais les autres sont disponibles.
    """
    global rag_pipelines

    print("\n" + "="*60)
    print("INITIALISATION MULTI-VERSION RAG")
    print("="*60 + "\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non trouvée dans les variables d'environnement. "
            "Veuillez créer un fichier .env avec votre clé API."
        )

    # === INITIALISER RAG v1 ===
    print("\n[1/4] Initialisation RAG v1...")
    try:
        rag_pipelines["v1"] = BasicRAGPipeline(
            openai_api_key=openai_api_key,
            chroma_path="./chroma_txt/",
            collection_name="communes_corses_txt",
            llm_model="gpt-3.5-turbo",
            embedding_model="intfloat/e5-base-v2"
        )
        print("OK RAG v1 initialisé")
    except Exception as e:
        print(f"AVERTISSEMENT: RAG v1 non disponible: {e}")
        rag_pipelines["v1"] = None

    # === INITIALISER RAG v2 ===
    print("\n[2/4] Initialisation RAG v2...")
    try:
        rag_pipelines["v2"] = ImprovedRAGPipeline(
            openai_api_key=openai_api_key,
            chroma_path="./chroma_v2/",
            collection_name="communes_corses_v2",
            quant_data_path="df_mean_by_commune.csv",
            llm_model="gpt-3.5-turbo",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3"
        )
        print("OK RAG v2 initialisé")
    except Exception as e:
        print(f"AVERTISSEMENT: RAG v2 non disponible: {e}")
        rag_pipelines["v2"] = None

    # === INITIALISER RAG v3 ===
    print("\n[3/4] Initialisation RAG v3...")
    try:
        rag_pipelines["v3"] = RAGPipelineWithOntology(
            openai_api_key=openai_api_key,
            ontology_path="ontology_be_2010_bilingue_fr_en.ttl",
            chroma_path="./chroma_v2/",
            collection_name="communes_corses_v2",
            quant_data_path="df_mean_by_commune.csv",
            llm_model="gpt-3.5-turbo",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3"
        )
        print("OK RAG v3 initialisé")
    except Exception as e:
        print(f"AVERTISSEMENT: RAG v3 non disponible: {e}")
        rag_pipelines["v3"] = None

    # === INITIALISER RAG v4 ===
    print("\n[4/6] Initialisation RAG v4...")
    try:
        rag_pipelines["v4"] = CrossAnalysisRAGPipeline(
            openai_api_key=openai_api_key,
            chroma_path="./chroma_v2/",
            collection_name="communes_corses_v2",
            quant_data_path="df_mean_by_commune.csv",
            llm_model="gpt-3.5-turbo",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3"
        )
        print("OK RAG v4 initialisé")
    except Exception as e:
        print(f"AVERTISSEMENT: RAG v4 non disponible: {e}")
        rag_pipelines["v4"] = None

    # === INITIALISER RAG v5 (Graph-RAG Neo4j) ===
    print("\n[5/6] Initialisation RAG v5 (Graph-RAG Neo4j)...")
    try:
        # Utiliser None si NEO4J_PASSWORD n'est pas défini (connexion sans auth)
        neo4j_password = os.getenv("NEO4J_PASSWORD", None)
        if neo4j_password == "":
            neo4j_password = None

        rag_pipelines["v5"] = GraphRAGPipeline(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password=neo4j_password,
            chroma_path="./chroma_v2/",
            collection_name="communes_corses_v2",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            llm_model="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            ontology_path="ontology_be_2010_bilingue_fr_en.ttl"
        )
        print("OK RAG v5 initialisé")
    except Exception as e:
        print(f"AVERTISSEMENT: RAG v5 non disponible: {e}")
        rag_pipelines["v5"] = None

    # === INITIALISER RAG v6 (G-Retriever GNN) ===
    print("\n[6/6] Initialisation RAG v6 (G-Retriever)...")
    if not V6_AVAILABLE:
        print("AVERTISSEMENT: RAG v6 non disponible (torch_geometric manquant)")
        rag_pipelines["v6"] = None
    else:
        try:
            # Utiliser None si NEO4J_PASSWORD n'est pas défini (connexion sans auth)
            neo4j_password = os.getenv("NEO4J_PASSWORD", None)
            if neo4j_password == "":
                neo4j_password = None

            rag_pipelines["v6"] = GRetrieverRAGPipeline(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password=neo4j_password,
                openai_api_key=openai_api_key
            )
            print("OK RAG v6 initialisé")
        except Exception as e:
            print(f"AVERTISSEMENT: RAG v6 non disponible: {e}")
            rag_pipelines["v6"] = None

    # [7/8] Initialisation RAG v7 (LlamaIndex - graphes optionnels)
    print("\n[7/8] Initialisation RAG v7 (LlamaIndex - graphes optionnels)...")
    if not V7_AVAILABLE:
        print("AVERTISSEMENT: RAG v7 non disponible (llama-index manquant)")
        rag_pipelines["v7"] = None
    else:
        try:
            rag_pipelines["v7"] = LlamaIndexRAGPipeline(
                openai_api_key=openai_api_key,
                chroma_path="./chroma_v2",
                collection_name="communes_corses_v2",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password=""
            )
            print("OK RAG v7 initialisé")
        except Exception as e:
            print(f"AVERTISSEMENT: RAG v7 non disponible: {e}")
            import traceback
            traceback.print_exc()
            rag_pipelines["v7"] = None

    # [8/8] Initialisation RAG v8 (LlamaIndex FULL - graphes OBLIGATOIRES)
    print("\n[8/8] Initialisation RAG v8 (LlamaIndex FULL - graphes OBLIGATOIRES)...")
    if not V8_AVAILABLE:
        print("AVERTISSEMENT: RAG v8 non disponible (llama-index manquant)")
        rag_pipelines["v8"] = None
    else:
        try:
            rag_pipelines["v8"] = LlamaIndexRAGPipelineFull(
                openai_api_key=openai_api_key,
                chroma_path="./chroma_v2",
                collection_name="communes_corses_v2",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password=""
            )
            print("OK RAG v8 initialisé")
        except Exception as e:
            print(f"AVERTISSEMENT: RAG v8 non disponible: {e}")
            print("  Note: v8 nécessite APOC Extended installé dans Neo4j")
            import traceback
            traceback.print_exc()
            rag_pipelines["v8"] = None

    # Résumé
    print("\n" + "="*60)
    available = [v for v, p in rag_pipelines.items() if p is not None]
    print(f"SYSTEMES RAG DISPONIBLES: {', '.join(available) if available else 'AUCUN'}")
    print("="*60 + "\n")

    if not available:
        raise RuntimeError("Aucune version du RAG n'a pu être initialisée")


# === GESTION DU CYCLE DE VIE ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    initialize_all_rags()
    yield
    # Shutdown (si nécessaire)
    pass


# === APPLICATION FASTAPI ===

app = FastAPI(
    title="API Chatbot RAG Multi-Version - Qualité de vie en Corse",
    description="API REST permettant de choisir entre les versions v1, v2 et v3 du système RAG",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# === CONFIGURATION CORS ===

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production: spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === ENDPOINTS ===

@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine - Redirige vers la documentation"""
    return {
        "message": "API Chatbot RAG Multi-Version - Qualité de vie en Corse",
        "documentation": "/docs",
        "health_check": "/api/health",
        "versions": "/api/versions",
        "query_endpoint": "/api/query"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé du serveur et des différentes versions RAG

    Returns:
        HealthResponse avec le statut de chaque version
    """
    return HealthResponse(
        status="healthy",
        rag_v1_initialized=rag_pipelines["v1"] is not None,
        rag_v2_initialized=rag_pipelines["v2"] is not None,
        rag_v3_initialized=rag_pipelines["v3"] is not None,
        rag_v4_initialized=rag_pipelines["v4"] is not None,
        rag_v5_initialized=rag_pipelines["v5"] is not None,
        rag_v6_initialized=rag_pipelines["v6"] is not None,
        rag_v7_initialized=rag_pipelines["v7"] is not None,
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )


@app.get("/api/versions", response_model=List[VersionInfo], tags=["Versions"])
async def get_versions():
    """
    Liste les versions disponibles du RAG avec leurs caractéristiques

    Returns:
        Liste des versions avec leurs fonctionnalités
    """
    versions = [
        VersionInfo(
            version="v1",
            name="RAG Basique",
            description="Retrieval vectoriel simple avec génération LLM",
            available=rag_pipelines["v1"] is not None,
            features=[
                "Retrieval vectoriel (e5-base-v2)",
                "Génération avec GPT-3.5-turbo",
                "Simple et rapide"
            ]
        ),
        VersionInfo(
            version="v2",
            name="RAG Amélioré + Boost",
            description="Hybrid retrieval avec boost intelligent pour questionnaires",
            available=rag_pipelines["v2"] is not None,
            features=[
                "Hybrid retrieval (BM25 + Vector)",
                "Reranking avec cross-encoder",
                "Boost intelligent questionnaires",
                "Données quantitatives",
                "Meilleure précision"
            ]
        ),
        VersionInfo(
            version="v3",
            name="RAG avec Ontologie",
            description="v2 + enrichissement sémantique via ontologie",
            available=rag_pipelines["v3"] is not None,
            features=[
                "Toutes les fonctionnalités v2",
                "Enrichissement de requête via ontologie",
                "Compréhension sémantique avancée",
                "Meilleure couverture thématique"
            ]
        ),
        VersionInfo(
            version="v4",
            name="RAG avec Analyse Croisée",
            description="v2 + décomposition de requêtes et analyse croisée multi-sources",
            available=rag_pipelines["v4"] is not None,
            features=[
                "Toutes les fonctionnalités v2",
                "Décomposition automatique de requêtes complexes",
                "Analyse croisée quantitatif/qualitatif",
                "Diversité de sources garantie",
                "Synthèse multi-perspectives"
            ]
        ),
        VersionInfo(
            version="v5",
            name="Graph-RAG avec Neo4j",
            description="RAG avancé avec graphe de connaissances Neo4j",
            available=rag_pipelines["v5"] is not None,
            features=[
                "Graphe de connaissances Neo4j",
                "Retrieval hybride (Vector + Graph)",
                "Embeddings BGE-M3 état de l'art",
                "Enrichissement ontologique",
                "Raisonnement sur graphe"
            ]
        ),
        VersionInfo(
            version="v6",
            name="G-Retriever (GNN)",
            description="v5 + Graph Neural Networks pour retrieval avancé",
            available=rag_pipelines["v6"] is not None,
            features=[
                "Toutes les fonctionnalités v5",
                "Graph Neural Networks (GraphSAGE)",
                "Embeddings de graphe appris",
                "Retrieval basé sur la structure",
                "Performance maximale"
            ]
        ),
        VersionInfo(
            version="v7",
            name="LlamaIndex Pipeline (graphes optionnels)",
            description="Framework LlamaIndex avec routing intelligent - fonctionne sans APOC Extended",
            available=rag_pipelines["v7"] is not None,
            features=[
                "Framework LlamaIndex complet",
                "3 modes: Router / Sub-question / Hybrid",
                "Graphes Neo4j optionnels (fallback vector-only)",
                "Fonctionne sans APOC Extended",
                "Fusion pondérée vector+graph (0.4/0.6) si graphe dispo",
                "Citations natives avec source_nodes",
                "Code simplifié (~200 lignes vs 500+)"
            ]
        ),
        VersionInfo(
            version="v8",
            name="LlamaIndex Pipeline FULL (graphes OBLIGATOIRES)",
            description="Framework LlamaIndex complet avec PropertyGraphIndex Neo4j - NÉCESSITE APOC Extended",
            available=rag_pipelines["v8"] is not None,
            features=[
                "Toutes les fonctionnalités v7",
                "PropertyGraphIndex Neo4j OBLIGATOIRE",
                "Routing automatique (LLM choisit vector ou graph)",
                "Décomposition automatique questions complexes",
                "Fusion pondérée vector+graph (0.4/0.6)",
                "Exploitation complète du graphe de connaissances",
                "NÉCESSITE: APOC Extended installé dans Neo4j"
            ]
        )
    ]
    return versions


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """
    Pose une question au système RAG avec choix de version

    Args:
        request: QueryRequest contenant la question et la version souhaitée

    Returns:
        QueryResponse avec la réponse et les sources

    Raises:
        HTTPException 400: Version non disponible
        HTTPException 500: Erreur lors du traitement
    """
    # Vérifier que la version demandée est disponible
    rag = rag_pipelines.get(request.rag_version)

    if rag is None:
        available_versions = [v for v, p in rag_pipelines.items() if p is not None]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Version '{request.rag_version}' non disponible. Versions disponibles: {', '.join(available_versions)}"
        )

    try:
        print(f"\n[{datetime.now().isoformat()}] Requête {request.rag_version}: {request.question}")

        # Exécuter la requête selon la version
        if request.rag_version == "v1":
            # v1 : simple retrieval
            answer, retrieval_results = rag.query(
                question=request.question,
                k=request.k
            )

        elif request.rag_version == "v2":
            # v2 : hybrid + reranking + quantitatif
            answer, retrieval_results = rag.query(
                question=request.question,
                k=request.k,
                use_reranking=request.use_reranking,
                include_quantitative=request.include_quantitative,
                commune_filter=request.commune_filter
            )

        elif request.rag_version == "v3":
            # v3 : v2 + ontologie
            answer, retrieval_results = rag.query(
                question=request.question,
                k=request.k,
                use_reranking=request.use_reranking,
                include_quantitative=request.include_quantitative,
                commune_filter=request.commune_filter,
                use_ontology_enrichment=request.use_ontology_enrichment
            )

        elif request.rag_version == "v4":
            # v4 : v2 + cross-analysis
            if request.use_cross_analysis:
                answer, retrieval_results = rag.query_with_cross_analysis(
                    question=request.question,
                    k=request.k,
                    use_reranking=request.use_reranking,
                    include_quantitative=request.include_quantitative,
                    commune_filter=request.commune_filter
                )
            else:
                # Fallback to regular query if cross-analysis is disabled
                answer, retrieval_results = rag.query(
                    question=request.question,
                    k=request.k,
                    use_reranking=request.use_reranking,
                    include_quantitative=request.include_quantitative,
                    commune_filter=request.commune_filter
                )

        elif request.rag_version == "v5":
            # v5 : Graph-RAG avec Neo4j
            answer, retrieval_results = rag.query(
                question=request.question,
                k=request.k,
                use_graph=True,
                use_reranking=request.use_reranking,
                commune_filter=request.commune_filter
            )

        elif request.rag_version == "v6":
            # v6 : G-Retriever avec GNN
            answer, retrieval_results = rag.query(
                question=request.question,
                k=request.k,
                use_gnn=True,
                use_reranking=request.use_reranking,
                commune_filter=request.commune_filter
            )

        elif request.rag_version == "v7":
            # v7 : LlamaIndex avec 3 modes (router, sub_question, hybrid) - graphes optionnels
            answer, retrieval_results = rag.query(
                question=request.question,
                mode=request.query_mode,  # router, sub_question, ou hybrid
                commune_filter=request.commune_filter,
                k=request.k
            )

        elif request.rag_version == "v8":
            # v8 : LlamaIndex FULL avec 3 modes (router, sub_question, hybrid) - graphes OBLIGATOIRES
            answer, retrieval_results = rag.query(
                question=request.question,
                mode=request.query_mode,  # router, sub_question, ou hybrid
                commune_filter=request.commune_filter,
                k=request.k
            )

        # Convertir les résultats en format API
        if request.rag_version in ["v7", "v8"]:
            # v7 retourne List[Dict] avec clés: content, score, metadata, source_type
            sources = [
                Source(
                    content=result['content'],
                    score=result['score'],
                    metadata=result['metadata']
                )
                for result in retrieval_results
            ]
        else:
            # v1-v6 retournent List[RetrievalResult]
            sources = [
                Source(
                    content=result.text,
                    score=result.score,
                    metadata=result.metadata
                )
                for result in retrieval_results
            ]

        # Construire la réponse
        response = QueryResponse(
            answer=answer,
            sources=sources,
            metadata={
                "k": request.k,
                "use_reranking": request.use_reranking if request.rag_version != "v1" else False,
                "use_ontology": request.use_ontology_enrichment if request.rag_version == "v3" else False,
                "use_cross_analysis": request.use_cross_analysis if request.rag_version == "v4" else False,
                "include_quantitative": request.include_quantitative if request.rag_version != "v1" else False,
                "commune_filter": request.commune_filter,
                "num_sources": len(sources)
            },
            rag_version_used=request.rag_version,
            timestamp=datetime.now().isoformat()
        )

        print(f"[{datetime.now().isoformat()}] Réponse générée ({request.rag_version}) avec {len(sources)} sources")

        return response

    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ERREUR ({request.rag_version}): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement avec {request.rag_version}: {str(e)}"
        )


# === POINT D'ENTRÉE ===

if __name__ == "__main__":
    # Configuration du serveur
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    print("\n" + "="*60)
    print("DÉMARRAGE DU SERVEUR API MULTI-VERSION")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Documentation: http://localhost:{port}/docs")
    print("="*60 + "\n")

    # Démarrer le serveur
    uvicorn.run(
        "api_server_multi_version:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
