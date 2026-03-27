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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
from dotenv import load_dotenv

# Imports des différentes versions RAG
from rag_v1_class import BasicRAGPipeline, RetrievalResult as RetrievalResult_v1
from rag_v2_boosted import ImprovedRAGPipeline, RetrievalResult as RetrievalResult_v2
try:
    from rag_v2_1_llamaindex import RAGv2_1_LlamaIndex
    V2_1_AVAILABLE = True
except ImportError as e:
    print(f"AVERTISSEMENT: RAG v2.1 non disponible ({e})")
    RAGv2_1_LlamaIndex = None
    V2_1_AVAILABLE = False
from rag_v2_2_portrait import PortraitRAGPipeline
from rag_v3_ontology import RAGPipelineWithOntology, RetrievalResult as RetrievalResult_v3
from rag_v4_cross_analysis import ImprovedRAGPipeline as CrossAnalysisRAGPipeline, RetrievalResult as RetrievalResult_v4
try:
    from rag_v5_graphrag_neo4j import GraphRAGPipeline
    V5_AVAILABLE = True
except ImportError as e:
    print(f"AVERTISSEMENT: RAG v5 non disponible ({e})")
    GraphRAGPipeline = None
    V5_AVAILABLE = False

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

# Import optionnel de v9 (RAPTOR-lite)
try:
    from rag_v9_raptor import RaptorRetriever
    V9_AVAILABLE = True
except ImportError as e:
    print(f"AVERTISSEMENT: RAPTOR non disponible ({e})")
    RaptorRetriever = None
    V9_AVAILABLE = False

# Import optionnel de v10 (RAPTOR + Sous-questions)
try:
    from rag_v10_raptor_subq import RaptorSubQuestionPipeline
    V10_AVAILABLE = True
except ImportError as e:
    print(f"AVERTISSEMENT: RAG v10 non disponible ({e})")
    RaptorSubQuestionPipeline = None
    V10_AVAILABLE = False

# Charger les variables d'environnement
load_dotenv()

# === MODELS PYDANTIC ===

class QueryRequest(BaseModel):
    """Modèle de requête pour poser une question"""
    question: str = Field(..., description="Question à poser au chatbot", min_length=1)
    rag_version: Literal["v1", "v2", "v2.1", "v2.2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"] = Field("v2", description="Version du RAG à utiliser")
    k: int = Field(5, description="Nombre de documents à récupérer", ge=1, le=20)
    use_reranking: bool = Field(True, description="Utiliser le reranking (v2/v3/v4 uniquement)")
    include_quantitative: bool = Field(True, description="Inclure les données quantitatives (v2/v3/v4 uniquement)")
    commune_filter: Optional[str] = Field(None, description="Filtrer par commune spécifique")
    use_ontology_enrichment: bool = Field(True, description="Utiliser l'enrichissement ontologique (v3 uniquement)")
    use_cross_analysis: bool = Field(True, description="Activer l'analyse croisée automatique (v4 uniquement)")
    query_mode: Literal["router", "sub_question", "hybrid"] = Field("router", description="Mode de query pour v7/v8 (router/sub_question/hybrid)")
    llm_model: str = Field("gpt-3.5-turbo", description="Modèle LLM à utiliser (gpt-3.5-turbo, gpt-4, gpt-4o, gpt-4o-mini)")
    # Filtres portrait pour v2.2
    auto_detect_portrait: bool = Field(True, description="Détecter automatiquement les filtres portrait (v2.2)")
    portrait_age_min: Optional[int] = Field(None, description="Âge minimum pour filtrer les verbatims (v2.2)", ge=15, le=100)
    portrait_age_max: Optional[int] = Field(None, description="Âge maximum pour filtrer les verbatims (v2.2)", ge=15, le=100)
    portrait_genre: Optional[str] = Field(None, description="Genre pour filtrer les verbatims: Homme/Femme (v2.2)")
    portrait_profession: Optional[str] = Field(None, description="Profession pour filtrer les verbatims (v2.2)")
    portrait_dimension: Optional[str] = Field(None, description="Dimension qualité de vie pour filtrer (v2.2)")
    n_subquestions: int = Field(5, ge=1, le=8, description="Nombre de sous-questions à générer (v10 uniquement)")

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
    context: Optional[str] = Field(None, description="Contexte complet passé au LLM (pour debug/export)")
    sub_questions: Optional[List[Dict]] = Field(None, description="Sous-questions et réponses intermédiaires (v10)")


class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check"""
    status: str = Field(..., description="État du serveur")
    rag_v1_initialized: bool = Field(..., description="RAG v1 initialisé")
    rag_v2_initialized: bool = Field(..., description="RAG v2 initialisé")
    rag_v2_1_initialized: bool = Field(..., description="RAG v2.1 initialisé")
    rag_v2_2_initialized: bool = Field(..., description="RAG v2.2 (Portrait) initialisé")
    rag_v3_initialized: bool = Field(..., description="RAG v3 initialisé")
    rag_v4_initialized: bool = Field(..., description="RAG v4 initialisé")
    rag_v5_initialized: bool = Field(..., description="RAG v5 initialisé")
    rag_v6_initialized: bool = Field(..., description="RAG v6 initialisé")
    rag_v7_initialized: bool = Field(..., description="RAG v7 initialisé")
    rag_v8_initialized: bool = Field(..., description="RAG v8 initialisé")
    rag_v9_initialized: bool = Field(..., description="RAG v9 (RAPTOR) initialisé")
    rag_v10_initialized: bool = Field(..., description="RAG v10 (RAPTOR + sous-questions) initialisé")
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
    "v2.1": None,
    "v2.2": None,
    "v3": None,
    "v4": None,
    "v5": None,
    "v6": None,
    "v7": None,
    "v8": None,
    "v9": None,
    "v10": None
}

# Mots-clés déclenchant la recherche dans oppchovec_scores
_OPPCHOVEC_KEYWORDS = [
    "oppchovec", "score", "scores",
    "opportunités", "opportunite", "opportunites",
    "choix", "vécu", "vecu",
    "indice", "indicateur", "indicateurs",
    "classement", "classé", "classee", "rang",
    "bien placé", "bien place", "mal placé", "mal place",
    "meilleur", "moins bon", "moins bonne",
    "mieux noté", "moins bien noté",
]


def _is_oppchovec_question(question: str) -> bool:
    """Retourne True si la question porte (en partie) sur les scores OppChoVec."""
    q = question.lower()
    return any(kw in q for kw in _OPPCHOVEC_KEYWORDS)


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
    # Désactivé : chroma_txt corrompu (Rust panic sur sqlite bindings)
    print("\n[1/10] RAG v1 désactivé (chroma_txt corrompu)")
    rag_pipelines["v1"] = None

    # === INITIALISER RAG v2 ===
    # Désactivé : chroma_v2 corrompu (Rust panic sur sqlite bindings)
    print("\n[2/10] RAG v2 désactivé (chroma_v2 corrompu)")
    rag_pipelines["v2"] = None

    # === INITIALISER RAG v2.1 (LlamaIndex équivalent v2) ===
    print("\n[2.1/9] Initialisation RAG v2.1 (LlamaIndex)...")
    if not V2_1_AVAILABLE:
        print("AVERTISSEMENT: RAG v2.1 non disponible (llama-index manquant)")
        rag_pipelines["v2.1"] = None
    else:
        try:
            rag_pipelines["v2.1"] = RAGv2_1_LlamaIndex(
                chroma_path="./chroma_v2",
                collection_name="communes_corses_v2",
                embedding_model="BAAI/bge-m3",
                rerank_model="BAAI/bge-reranker-v2-m3",
                llm_model="gpt-3.5-turbo",
                top_k=5,
                use_reranker=True,
                use_hybrid=True
            )
            print("OK RAG v2.1 initialisé")
        except Exception as e:
            print(f"AVERTISSEMENT: RAG v2.1 non disponible: {e}")
            import traceback
            traceback.print_exc()
            rag_pipelines["v2.1"] = None

    # === INITIALISER RAG v2.2 (Portrait) ===
    print("\n[2.2/10] Initialisation RAG v2.2 (Portrait)...")
    try:
        rag_pipelines["v2.2"] = PortraitRAGPipeline(
            chroma_path="./chroma_portrait",
            collection_name="portrait_verbatims",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            llm_model="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            quant_data_path="df_mean_by_commune.csv"
        )
        # Afficher les stats portrait
        stats = rag_pipelines["v2.2"].get_portrait_stats()
        if stats.get('count', 0) > 0:
            print(f"OK RAG v2.2 initialisé ({stats['count']} verbatims portrait)")
        else:
            print("OK RAG v2.2 initialisé (pas de verbatims portrait indexés)")
    except Exception as e:
        print(f"AVERTISSEMENT: RAG v2.2 non disponible: {e}")
        import traceback
        traceback.print_exc()
        rag_pipelines["v2.2"] = None

    # === RAG v3, v4, v5 ===
    # Désactivés : utilisent chroma_v2 qui provoque un Rust panic fatal
    print("\n[3-5/10] RAG v3, v4, v5 désactivés (chroma_v2 corrompu)")
    rag_pipelines["v3"] = None
    rag_pipelines["v4"] = None
    rag_pipelines["v5"] = None

    # === INITIALISER RAG v6 (G-Retriever GNN) ===
    print("\n[6/9] Initialisation RAG v6 (G-Retriever)...")
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
                chroma_path="./chroma_portrait",
                collection_name="portrait_verbatims",
                openai_api_key=openai_api_key
            )
            print("OK RAG v6 initialisé")
        except Exception as e:
            print(f"AVERTISSEMENT: RAG v6 non disponible: {e}")
            rag_pipelines["v6"] = None

    # [7/9] Initialisation RAG v7 (LlamaIndex - graphes optionnels)
    print("\n[7/9] Initialisation RAG v7 (LlamaIndex - graphes optionnels)...")
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

    # [8/9] Initialisation RAG v8 (LlamaIndex FULL - graphes OBLIGATOIRES)
    print("\n[8/9] Initialisation RAG v8 (LlamaIndex FULL - graphes OBLIGATOIRES)...")
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

    # [9/11] Initialisation RAG v9 (RAPTOR-lite)
    print("\n[9/11] Initialisation RAG v9 (RAPTOR-lite)...")
    if not V9_AVAILABLE:
        print("AVERTISSEMENT: RAG v9 non disponible (import échoué)")
        rag_pipelines["v9"] = None
    else:
        try:
            raptor = RaptorRetriever(
                chroma_path="./chroma_portrait",
                source_collection="portrait_verbatims",
                summary_collection="raptor_summaries",
            )
            raptor.init()
            rag_pipelines["v9"] = raptor
            print(f"OK RAG v9 initialisé ({raptor.summary_count} synthèses RAPTOR)")
        except Exception as e:
            print(f"AVERTISSEMENT: RAG v9 non disponible: {e}")
            rag_pipelines["v9"] = None

    # [10/11] Initialisation RAG v10 (RAPTOR + Sous-questions)
    print("\n[10/11] Initialisation RAG v10 (RAPTOR + Sous-questions)...")
    if not V10_AVAILABLE:
        print("AVERTISSEMENT: RAG v10 non disponible (import échoué)")
        rag_pipelines["v10"] = None
    else:
        try:
            v10 = RaptorSubQuestionPipeline(
                chroma_path="./chroma_portrait",
                source_collection="portrait_verbatims",
                summary_collection="raptor_summaries",
            )
            v10.init()
            rag_pipelines["v10"] = v10
            print(f"OK RAG v10 initialisé")
        except Exception as e:
            print(f"AVERTISSEMENT: RAG v10 non disponible: {e}")
            rag_pipelines["v10"] = None

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
    description="API REST permettant de choisir entre les versions v1, v2, v2.1, v2.2, v3, v4, v5, v6, v7, v8 du système RAG",
    version="2.1.0",
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


# === FICHIER HTML FRONTEND ===

@app.get("/frontend", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend():
    """Sert l'interface graphique HTML"""
    html_path = os.path.join(os.path.dirname(__file__), "example_frontend_multi_version.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


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
        rag_v2_1_initialized=rag_pipelines["v2.1"] is not None,
        rag_v2_2_initialized=rag_pipelines["v2.2"] is not None,
        rag_v3_initialized=rag_pipelines["v3"] is not None,
        rag_v4_initialized=rag_pipelines["v4"] is not None,
        rag_v5_initialized=rag_pipelines["v5"] is not None,
        rag_v6_initialized=rag_pipelines["v6"] is not None,
        rag_v7_initialized=rag_pipelines["v7"] is not None,
        rag_v8_initialized=rag_pipelines["v8"] is not None,
        rag_v9_initialized=rag_pipelines["v9"] is not None,
        rag_v10_initialized=rag_pipelines["v10"] is not None,
        timestamp=datetime.now().isoformat(),
        version="2.2.0"
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
            version="v2.1",
            name="RAG LlamaIndex (équivalent v2)",
            description="Version LlamaIndex du RAG v2 pour comparaison",
            available=rag_pipelines["v2.1"] is not None,
            features=[
                "Framework LlamaIndex",
                "Hybrid retrieval (BM25 + Vector)",
                "Reranking avec SentenceTransformer",
                "Boost intelligent questionnaires",
                "Comparaison avec v2 'fait main'"
            ]
        ),
        VersionInfo(
            version="v2.2",
            name="RAG Portrait (filtres démographiques)",
            description="v2 + filtrage par profil démographique (âge, genre, profession)",
            available=rag_pipelines["v2.2"] is not None,
            features=[
                "Toutes les fonctionnalités v2",
                "Filtrage par tranche d'âge (15-24, 25-34, 35-49, 50-64, 65+)",
                "Filtrage par genre (Homme/Femme)",
                "Filtrage par profession (9 catégories)",
                "Filtrage par dimension qualité de vie",
                "Auto-détection des filtres dans la question",
                "Requêtes ciblées: 'Que pensent les jeunes de la santé ?'"
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
        ),
        VersionInfo(
            version="v9",
            name="RAPTOR-lite (synthèses analytiques)",
            description="Synthèses pré-calculées par groupes démographiques avec fallback hiérarchique",
            available=rag_pipelines["v9"] is not None,
            features=[
                "349 synthèses pré-calculées (6 vues analytiques)",
                "Vues 1D: âge, profession, commune",
                "Vues 2D: âge×profession, âge×commune, profession×commune",
                "Détection automatique des dimensions dans la question",
                "Fallback hiérarchique (2D → 1D → sémantique)",
                "Synthèse + verbatims evidence",
                "Idéal pour questions analytiques (ex: 'Que pensent les jeunes ?')"
            ]
        ),
        VersionInfo(
            version="v10",
            name="RAPTOR + Sous-questions (pipeline 3 étapes)",
            description="Décomposition Mistral Large → réponses Claude Haiku par sous-question → synthèse Mistral Large",
            available=rag_pipelines["v10"] is not None,
            features=[
                "Décomposition automatique en N sous-questions (Mistral Large)",
                "Retrieval RAPTOR-lite par sous-question",
                "Réponse ciblée par sous-question (Claude Haiku)",
                "Synthèse finale avec filtrage du bruit (Mistral Large)",
                "Configurable : n_subquestions (1-8, défaut 5)",
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
        print(f"\n[{datetime.now().isoformat()}] Requête {request.rag_version} (LLM: {request.llm_model}): {request.question}")
        v10_scoring = {"applicable": False}  # valeur par defaut pour les versions != v10
        v10_sub_qa = None

        # Changer le modèle LLM si différent du défaut
        if hasattr(rag, 'llm_model'):
            rag.llm_model = request.llm_model
        if hasattr(rag, 'llm') and hasattr(rag.llm, 'model'):
            rag.llm.model = request.llm_model

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

        elif request.rag_version == "v2.1":
            # v2.1 : LlamaIndex équivalent v2
            result = rag.query(
                question=request.question,
                commune=request.commune_filter
            )
            answer = result["answer"]
            # Convertir les sources au format attendu
            retrieval_results = []
            for source in result["sources"]:
                retrieval_results.append(type('obj', (object,), {
                    'text': source['text'],
                    'score': source['score'] if source['score'] is not None else 0.0,
                    'metadata': source['metadata']
                })())

        elif request.rag_version == "v2.2":
            # v2.2 : Portrait avec filtres démographiques
            # Construire les filtres portrait explicites si fournis
            portrait_filters = None
            if not request.auto_detect_portrait:
                # Mode manuel: utiliser les filtres fournis par l'utilisateur
                has_explicit_filter = any([
                    request.portrait_age_min,
                    request.portrait_age_max,
                    request.portrait_genre,
                    request.portrait_profession,
                    request.portrait_dimension
                ])
                if has_explicit_filter:
                    portrait_filters = {
                        'has_portrait_filter': True,
                        'age_min': request.portrait_age_min,
                        'age_max': request.portrait_age_max,
                        'genre': request.portrait_genre,
                        'profession': request.portrait_profession,
                        'dimension': request.portrait_dimension
                    }

            answer, retrieval_results, detected_filters = rag.query(
                question=request.question,
                k=request.k,
                use_reranking=request.use_reranking,
                include_quantitative=request.include_quantitative,
                commune_filter=request.commune_filter,
                portrait_filters=portrait_filters,
                auto_detect_filters=request.auto_detect_portrait
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

        elif request.rag_version == "v9":
            # v9 : RAPTOR-lite — synthèses analytiques pré-calculées
            # Le retriever retourne (context_str, sources_list)
            # On passe le contexte à un LLM pour générer la réponse
            context_str, raptor_sources = rag.query(
                question=request.question,
                k=request.k
            )
            # Enrichissement OppChoVec si la question y fait référence
            opp_docs = []
            if _is_oppchovec_question(request.question):
                opp_docs = rag.query_oppchovec(request.question, k=3)
                if opp_docs:
                    opp_block = "\n\n[Scores OppChoVec (quanti) par commune]\n" + "\n\n".join(
                        d["text"] for d in opp_docs
                    )
                    context_str = context_str + opp_block
            # Générer une réponse via LLM
            from openai import OpenAI
            llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            llm_response = llm_client.chat.completions.create(
                model=request.llm_model,
                messages=[
                    {"role": "system", "content": (
                        "Tu es un assistant spécialisé dans l'analyse de la qualité de vie en Corse. "
                        "Réponds à la question en te basant UNIQUEMENT sur le contexte fourni. "
                        "Les sources sont étiquetées (quali) pour les verbatims et synthèses de perceptions citoyennes, "
                        "et (quanti) pour les scores et indicateurs chiffrés (OppChoVec, enquêtes). "
                        "Pour une question portant sur des indicateurs chiffrés, appuie-toi prioritairement sur les sources (quanti). "
                        "Pour une question de perception ou d'opinion, priorise les sources (quali). "
                        "Pour une question mixte, utilise les deux de façon complémentaire. "
                        "Cite des extraits quand c'est pertinent. Sois factuel et nuancé."
                    )},
                    {"role": "user", "content": f"Contexte :\n{context_str}\n\nQuestion : {request.question}"}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            answer = llm_response.choices[0].message.content
            # Fusionner sources RAPTOR + sources OppChoVec
            opp_sources = [
                {
                    "rank": len(raptor_sources) + i,
                    "source_type": "oppchovec_score",
                    "commune": d["meta"].get("commune"),
                    "oppchovec_0_10": d["meta"].get("oppchovec_0_10"),
                    "opp_0_10": d["meta"].get("opp_0_10"),
                    "cho_0_10": d["meta"].get("cho_0_10"),
                    "vec_0_10": d["meta"].get("vec_0_10"),
                    "extrait": d["text"][:400],
                }
                for i, d in enumerate(opp_docs)
            ]
            retrieval_results = raptor_sources + opp_sources

        elif request.rag_version == "v10":
            # v10 : RAPTOR + Sous-questions + Notation
            # Pré-fetch OppChoVec pour l'injecter dans chaque sous-question si pertinent
            opp_docs_v10 = []
            extra_ctx_v10 = ""
            if _is_oppchovec_question(request.question):
                opp_docs_v10 = rag.retriever.query_oppchovec(request.question, k=3)
                if opp_docs_v10:
                    extra_ctx_v10 = "[Données OppChoVec (quanti)]\n" + "\n\n".join(d["text"] for d in opp_docs_v10)
            answer, retrieval_results, v10_scoring, v10_sub_qa = rag.query(
                question=request.question,
                k=request.k,
                n_subquestions=request.n_subquestions,
                extra_context=extra_ctx_v10,
            )
            # Enrichissement OppChoVec : ajout des scores en sources + complément de réponse
            if opp_docs_v10:
                opp_sources = [
                    {
                        "rank": len(retrieval_results) + i,
                        "source_type": "oppchovec_score",
                        "commune": d["meta"].get("commune"),
                        "oppchovec_0_10": d["meta"].get("oppchovec_0_10"),
                        "opp_0_10": d["meta"].get("opp_0_10"),
                        "cho_0_10": d["meta"].get("cho_0_10"),
                        "vec_0_10": d["meta"].get("vec_0_10"),
                        "extrait": d["text"][:400],
                    }
                    for i, d in enumerate(opp_docs_v10)
                ]
                retrieval_results = retrieval_results + opp_sources
                # Ajouter un appendice factuel avec les scores (communes seulement, pas la méthodologie)
                commune_docs = [d for d in opp_docs_v10 if d["meta"].get("oppchovec_0_10") is not None]
                if commune_docs:
                    scores_lines = "\n".join(
                        f"- {d['meta']['commune']} : OppChoVec={d['meta']['oppchovec_0_10']:.2f}/10 "
                        f"(Opp={d['meta']['opp_0_10']:.2f}, Cho={d['meta']['cho_0_10']:.2f}, "
                        f"Vec={d['meta']['vec_0_10']:.2f})"
                        for d in commune_docs
                    )
                    answer = answer + f"\n\n**Scores OppChoVec (données chiffrées) :**\n{scores_lines}"

        # Convertir les résultats en format API
        if request.rag_version in ("v9", "v10"):
            # v9 retourne List[Dict] avec clés: rank, type/commune, extrait, etc.
            sources = [
                Source(
                    content=result.get('extrait', ''),
                    score=1.0 - result.get('rank', 0) * 0.1,  # score décroissant par rang
                    metadata={k: v for k, v in result.items() if k != 'extrait'}
                )
                for result in retrieval_results
            ]
        elif request.rag_version in ["v7", "v8"]:
            # v7/v8 retourne List[Dict] avec clés: content, score, metadata, source_type
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

        # Construire le contexte complet (ce qui est passé au LLM)
        context_parts = [
            f"=== CONTEXTE PASSÉ AU LLM ({request.rag_version.upper()}) ===",
            f"Question: {request.question}",
            f"Date: {datetime.now().isoformat()}",
            f"Nombre de sources: {len(sources)}",
            "",
            "=== SOURCES UTILISÉES ===",
            ""
        ]

        for i, source in enumerate(sources, 1):
            context_parts.append(f"--- Source {i} (score: {source.score:.4f}) ---")
            # Ajouter les métadonnées pertinentes
            if source.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in source.metadata.items() if v and k not in ['content'])
                if meta_str:
                    context_parts.append(f"Métadonnées: {meta_str}")
            context_parts.append(f"Contenu:\n{source.content}")
            context_parts.append("")

        context_parts.append("=== FIN DU CONTEXTE ===")
        full_context = "\n".join(context_parts)

        # Construire les metadonnees (scoring v10 si applicable)
        metadata = {
            "k": request.k,
            "use_reranking": request.use_reranking if request.rag_version != "v1" else False,
            "use_ontology": request.use_ontology_enrichment if request.rag_version == "v3" else False,
            "use_cross_analysis": request.use_cross_analysis if request.rag_version == "v4" else False,
            "include_quantitative": request.include_quantitative if request.rag_version != "v1" else False,
            "commune_filter": request.commune_filter,
            "num_sources": len(sources)
        }
        if request.rag_version == "v10" and v10_scoring.get("applicable"):
            metadata["scoring"] = {
                "dimension": v10_scoring["dimension"],
                "score": v10_scoring["score"],
                "score_max": 5,
                "justification": v10_scoring["justification"],
            }

        # Construire la réponse
        response = QueryResponse(
            answer=answer,
            sources=sources,
            context=full_context,
            metadata=metadata,
            rag_version_used=request.rag_version,
            timestamp=datetime.now().isoformat(),
            sub_questions=v10_sub_qa if request.rag_version == "v10" else None
        )

        print(f"[{datetime.now().isoformat()}] Réponse générée ({request.rag_version}) avec {len(sources)} sources")

        return response

    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ERREUR ({request.rag_version}): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement avec {request.rag_version}: {str(e)}"
        )


# === ENDPOINTS EXPLORATION CORPUS ===

@app.get("/api/browse/filters", tags=["Browse"])
async def get_browse_filters():
    """Retourne les valeurs distinctes disponibles pour filtrer les verbatims."""
    try:
        import chromadb as _chromadb
        chroma = _chromadb.PersistentClient(path="./chroma_portrait")
        col = chroma.get_collection("portrait_verbatims")
        all_metas = col.get(include=["metadatas"])["metadatas"]

        age_order = ["15-24", "25-34", "35-49", "50-64", "65+"]

        def sort_age(lst):
            return sorted(lst, key=lambda x: age_order.index(x) if x in age_order else 99)

        return {
            "communes":    sorted(set(m["nom"]        for m in all_metas if m.get("nom"))),
            "genres":      sorted(set(m["genre"]      for m in all_metas if m.get("genre"))),
            "age_ranges":  sort_age(list(set(m["age_range"]   for m in all_metas if m.get("age_range")))),
            "professions": sorted(set(m["profession"] for m in all_metas if m.get("profession"))),
            "dimensions":  sorted(set(m["dimension"]  for m in all_metas if m.get("dimension"))),
            "total":       len(all_metas),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/browse/verbatims", tags=["Browse"])
async def browse_verbatims(
    commune:    Optional[str] = None,
    genre:      Optional[str] = None,
    age_range:  Optional[str] = None,
    profession: Optional[str] = None,
    dimension:  Optional[str] = None,
):
    """Retourne tous les verbatims correspondant aux filtres (non paginé — 690 docs max)."""
    try:
        import chromadb as _chromadb
        chroma = _chromadb.PersistentClient(path="./chroma_portrait")
        col = chroma.get_collection("portrait_verbatims")

        conditions = []
        if commune:    conditions.append({"nom":        {"$eq": commune}})
        if genre:      conditions.append({"genre":      {"$eq": genre}})
        if age_range:  conditions.append({"age_range":  {"$eq": age_range}})
        if profession: conditions.append({"profession": {"$eq": profession}})
        if dimension:  conditions.append({"dimension":  {"$eq": dimension}})

        kwargs: dict = {"include": ["documents", "metadatas"]}
        if len(conditions) == 1:
            kwargs["where"] = conditions[0]
        elif len(conditions) > 1:
            kwargs["where"] = {"$and": conditions}

        res = col.get(**kwargs)

        verbatims = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(res["documents"], res["metadatas"])
        ]
        verbatims.sort(key=lambda v: (
            v["metadata"].get("nom", ""),
            v["metadata"].get("age_exact", 0) or 0,
        ))

        return {"total": len(verbatims), "verbatims": verbatims}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/browse/summaries", tags=["Browse"])
async def browse_summaries():
    """Retourne toutes les synthèses RAPTOR groupées par vue analytique."""
    try:
        import chromadb as _chromadb
        chroma = _chromadb.PersistentClient(path="./chroma_portrait")
        col = chroma.get_collection("raptor_summaries")

        res = col.get(include=["documents", "metadatas"])

        grouped: Dict[str, list] = {}
        for doc, meta in zip(res["documents"], res["metadatas"]):
            view = meta.get("view_name", "unknown")
            grouped.setdefault(view, []).append({"content": doc, "metadata": meta})

        for view in grouped:
            grouped[view].sort(key=lambda x: (
                x["metadata"].get("dim1_value", ""),
                x["metadata"].get("dim2_value", ""),
            ))

        view_order = [
            "age_range*profession", "age_range*commune", "profession*commune",
            "age_range", "profession", "commune",
        ]
        ordered: Dict[str, list] = {v: grouped[v] for v in view_order if v in grouped}
        ordered.update({v: grouped[v] for v in grouped if v not in view_order})

        return {"total": sum(len(v) for v in ordered.values()), "views": ordered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/browse/quantitative", tags=["Browse"])
async def browse_quantitative(
    commune:   Optional[str] = None,
    genre:     Optional[str] = None,
    age_range: Optional[str] = None,
):
    """Retourne les données quantitatives du questionnaire (273 répondants)."""
    import pandas as pd
    import numpy as np

    CSV_PATH = "./donnees_brutes/sortie_questionnaire_traited.csv"
    try:
        df = pd.read_csv(CSV_PATH, index_col=0)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Fichier CSV introuvable")

    col_a = "Dans quelle commune résidez-vous ? ( A à S)"
    col_b = "Dans quelle commune résidez-vous ? (T à Z)"
    df["_commune"] = (df[col_a].fillna("") + df[col_b].fillna("")).str.strip()

    if commune:   df = df[df["_commune"] == commune]
    if genre:     df = df[df["genre"] == genre]
    if age_range: df = df[df["Catégorie age"] == age_range]

    score_map = {
        "Les services de transports":                                          "Transports",
        "L'accès à l'éducation":                                               "Education",
        "La couverture des réseaux téléphoniques":                             "Réseaux",
        "Les institutions étatiques (niveau de confiance)":                    "Institutions",
        "Le tourisme (ressenti localement)":                                   "Tourisme",
        "La sécurité":                                                         "Sécurité",
        "L'offre de santé ":                                                   "Santé",
        "Votre situation professionnelle":                                     "Emploi",
        "Vos revenus":                                                         "Revenus",
        "La répartition de votre temps entre travail et temps personnel":      "Temps pro/perso",
        "Votre logement":                                                      "Logement",
        "L'offre de services autour de chez vous":                             "Services",
        "Votre accès à la culture":                                            "Culture",
        "Vous sentez-vous bien entouré ?":                                     "Entourage",
        "Vous sentez-vous impliqué dans la vie locale de votre commune ?":     "Implication locale",
    }

    def safe(val):
        if val is None: return None
        if isinstance(val, float) and np.isnan(val): return None
        s = str(val).strip()
        return None if s in ("", "nan") else val

    rows = []
    for _, row in df.iterrows():
        r = {
            "id":                  safe(row.get("ID")),
            "commune":             row["_commune"] or None,
            "genre":               safe(row.get("genre")),
            "age":                 safe(row.get("Age")),
            "age_range":           safe(row.get("Catégorie age")),
            "profession":          safe(row.get("situation socioprofessionnelle")),
            "dimensions_choisies": safe(row.get(
                "Pour vous, qu'est-ce qui est important pour votre qualité de vie ? Choisissez 3 images"
            )),
            "bonheur":          safe(row.get(
                "Sur une échelle de 1 à 5, pourriez-vous estimer à quel point vous êtes heureux ces derniers temps ?"
            )),
            "qualite_vie":      safe(row.get(
                "Sur une échelle de 1 à 5, pourriez-vous évaluer votre qualité de vie ces derniers temps ?"
            )),
            "confiance_avenir": safe(row.get(
                "Sur une échelle de 1 à 5, pourriez-vous évaluer votre confiance en l'avenir ?"
            )),
            "scores": {short: safe(row.get(long)) for long, short in score_map.items()},
        }
        rows.append(r)

    def mean_score(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(float(v) for v in vals) / len(vals), 2) if vals else None

    stats = {
        "bonheur_moyen":         mean_score("bonheur"),
        "qualite_vie_moyenne":   mean_score("qualite_vie"),
        "confiance_avenir_moy":  mean_score("confiance_avenir"),
        "communes_disponibles":  sorted(set(r["commune"] for r in rows if r["commune"])),
        "age_ranges_disponibles": sorted(set(r["age_range"] for r in rows if r["age_range"])),
    }

    return {"total": len(rows), "stats": stats, "rows": rows}


@app.get("/api/browse/quanti_summaries", tags=["Browse"])
async def browse_quanti_summaries():
    """Retourne les synthèses RAPTOR quantitatives groupées par vue analytique."""
    try:
        import chromadb as _chromadb
        chroma = _chromadb.PersistentClient(path="./chroma_portrait")
        col = chroma.get_collection("raptor_quanti_summaries")
        res = col.get(include=["documents", "metadatas"])

        grouped: Dict[str, list] = {}
        for doc, meta in zip(res["documents"], res["metadatas"]):
            view = meta.get("view_name", "unknown")
            grouped.setdefault(view, []).append({"content": doc, "metadata": meta})

        for view in grouped:
            grouped[view].sort(key=lambda x: (
                x["metadata"].get("dim1_value", ""),
                x["metadata"].get("dim2_value", ""),
            ))

        view_order = [
            "age_range*profession", "age_range*commune", "profession*commune",
            "age_range", "profession", "commune",
        ]
        ordered: Dict[str, list] = {v: grouped[v] for v in view_order if v in grouped}
        ordered.update({v: grouped[v] for v in grouped if v not in view_order})

        return {"total": sum(len(v) for v in ordered.values()), "views": ordered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", tags=["Browse"])
async def get_stats():
    """
    Retourne les effectifs réels par dimension d'analyse.
    Toutes les valeurs proviennent des données brutes (ChromaDB + CSV).
    Rien n'est inventé ni estimé.
    """
    from collections import Counter
    import chromadb as _chromadb
    import pandas as pd

    result = {}

    # ── 1. Verbatims (portrait_verbatims, ChromaDB) ───────────────────────────
    try:
        chroma = _chromadb.PersistentClient(path="./chroma_portrait")
        col = chroma.get_collection("portrait_verbatims")
        all_metas = col.get(include=["metadatas"])["metadatas"]

        result["verbatims"] = {
            "total":          len(all_metas),
            "par_commune":    dict(Counter(m["nom"]        for m in all_metas if m.get("nom")).most_common()),
            "par_age_range":  dict(Counter(m["age_range"]  for m in all_metas if m.get("age_range")).most_common()),
            "par_profession": dict(Counter(m["profession"] for m in all_metas if m.get("profession")).most_common()),
            "par_dimension":  dict(Counter(m["dimension"]  for m in all_metas if m.get("dimension")).most_common()),
            "par_genre":      dict(Counter(m["genre"]      for m in all_metas if m.get("genre")).most_common()),
        }
    except Exception as e:
        result["verbatims"] = {"error": str(e)}

    # ── 2. Synthèses RAPTOR (raptor_summaries, ChromaDB) ──────────────────────
    try:
        chroma2 = _chromadb.PersistentClient(path="./chroma_portrait")
        col2 = chroma2.get_collection("raptor_summaries")
        all_sum_metas = col2.get(include=["metadatas"])["metadatas"]

        par_vue: Dict[str, dict] = {}
        for m in all_sum_metas:
            vn = m.get("view_name", "inconnu")
            nc = int(m.get("num_chunks", 0))
            if vn not in par_vue:
                par_vue[vn] = {"total": 0, "chunks_total": 0}
            par_vue[vn]["total"]        += 1
            par_vue[vn]["chunks_total"] += nc

        vue_order = [
            "age_range*profession", "age_range*commune", "profession*commune",
            "dimension*commune", "dimension*age_range", "dimension*profession",
            "age_range", "profession", "commune", "dimension",
        ]
        par_vue_ordered = {}
        for vn in vue_order:
            if vn in par_vue:
                d = par_vue[vn]
                par_vue_ordered[vn] = {
                    "total":        d["total"],
                    "chunks_moyen": round(d["chunks_total"] / d["total"], 1),
                }
        for vn, d in par_vue.items():
            if vn not in par_vue_ordered:
                par_vue_ordered[vn] = {
                    "total":        d["total"],
                    "chunks_moyen": round(d["chunks_total"] / d["total"], 1),
                }

        result["raptor"] = {
            "total":   len(all_sum_metas),
            "par_vue": par_vue_ordered,
        }
    except Exception as e:
        result["raptor"] = {"error": str(e)}

    # ── 3. Répondants enquête (CSV) ───────────────────────────────────────────
    CSV_PATH = "./donnees_brutes/sortie_questionnaire_traited.csv"
    try:
        df = pd.read_csv(CSV_PATH, index_col=0)
        col_a = "Dans quelle commune résidez-vous ? ( A à S)"
        col_b = "Dans quelle commune résidez-vous ? (T à Z)"
        df["_commune"] = (df[col_a].fillna("") + df[col_b].fillna("")).str.strip()

        age_order = ["15-24", "25-34", "35-49", "50-64", "65+"]
        age_counts = dict(Counter(df["Catégorie age"].dropna()))
        age_counts_ordered = {k: age_counts[k] for k in age_order if k in age_counts}
        age_counts_ordered.update({k: v for k, v in age_counts.items() if k not in age_counts_ordered})

        result["enquete"] = {
            "total":          len(df),
            "par_commune":    dict(Counter(df["_commune"].replace("", None).dropna()).most_common()),
            "par_age_range":  age_counts_ordered,
            "par_profession": dict(Counter(df["situation socioprofessionnelle"].dropna()).most_common()),
            "par_genre":      dict(Counter(df["genre"].dropna()).most_common()),
        }
    except Exception as e:
        result["enquete"] = {"error": str(e)}

    return result


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
