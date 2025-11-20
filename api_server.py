"""
API FastAPI pour le système RAG v3 avec enrichissement ontologique

Ce serveur expose le système RAG via une API REST pour être consommé
par un front-end externe.

Endpoints:
- POST /api/query - Poser une question au chatbot
- GET /api/health - Vérifier l'état du serveur
- GET /docs - Documentation Swagger automatique

Auteur: Claude Code
Date: 2025-11-15
"""

import os
from typing import Optional, List, Dict
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
from dotenv import load_dotenv

# Import du système RAG v3
#from rag_v3_ontology import RAGPipelineWithOntology, RetrievalResult
from rag_v2_improved import ImprovedRAGPipeline, RetrievalResult


# Charger les variables d'environnement
load_dotenv()

# === MODELS PYDANTIC ===

class QueryRequest(BaseModel):
    """Modèle de requête pour poser une question"""
    question: str = Field(..., description="Question à poser au chatbot", min_length=1)
    k: int = Field(5, description="Nombre de documents à récupérer", ge=1, le=20)
    use_reranking: bool = Field(True, description="Utiliser le reranking pour améliorer les résultats")
    include_quantitative: bool = Field(True, description="Inclure les données quantitatives")
    commune_filter: Optional[str] = Field(None, description="Filtrer par commune spécifique")
    use_ontology_enrichment: bool = Field(True, description="Utiliser l'enrichissement ontologique")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Quelles sont les communes avec le meilleur bien-être ?",
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
    timestamp: str = Field(..., description="Horodatage de la réponse")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Selon les données disponibles, les communes de Corse...",
                "sources": [
                    {
                        "content": "Extrait d'entretien pertinent...",
                        "score": 0.85,
                        "metadata": {"commune": "Ajaccio", "source": "entretien"}
                    }
                ],
                "metadata": {
                    "k": 5,
                    "use_reranking": True,
                    "use_ontology": True,
                    "num_sources": 5
                },
                "timestamp": "2025-11-15T10:30:00"
            }
        }
    )


class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check"""
    status: str = Field(..., description="État du serveur")
    rag_initialized: bool = Field(..., description="Le système RAG est-il initialisé")
    timestamp: str = Field(..., description="Horodatage du check")
    version: str = Field(..., description="Version de l'API")


# === GESTION DU CYCLE DE VIE ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    initialize_rag()
    yield
    # Shutdown (si nécessaire)
    pass


# === APPLICATION FASTAPI ===

app = FastAPI(
    title="API Chatbot RAG v3 - Qualité de vie en Corse",
    description="API REST pour interroger le système RAG enrichi par ontologie sur la qualité de vie en Corse",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# === CONFIGURATION CORS ===

# Pour les tests en local, on autorise toutes les origines
# En production, spécifier les domaines autorisés
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production: ["http://localhost:3000", "https://votre-domaine.com"]
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, etc.
    allow_headers=["*"],  # Headers autorisés
)

# === INITIALISATION DU RAG ===

rag_pipeline: Optional[RAGPipelineWithOntology] = None


def initialize_rag():
    """
    Initialise le pipeline RAG au démarrage du serveur

    Charge:
    - L'API key OpenAI depuis .env
    - La base ChromaDB
    - L'ontologie
    - Les modèles d'embeddings et de reranking
    """
    global rag_pipeline

    print("\n" + "="*60)
    print("INITIALISATION DU SYSTÈME RAG v3")
    print("="*60 + "\n")

    # Vérifier la présence de l'API key OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non trouvée dans les variables d'environnement. "
            "Veuillez créer un fichier .env avec votre clé API."
        )

    # Chemins par défaut (modifiables via variables d'environnement)
    ontology_path = os.getenv("ONTOLOGY_PATH", "ontology_be_2010.ttl")
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_v2/")
    collection_name = os.getenv("COLLECTION_NAME", "communes_corses_v2")
    quant_data_path = os.getenv("QUANT_DATA_PATH", "df_mean_by_commune.csv")

    try:
        # Initialiser le pipeline RAG v3
        # IMPORTANT: Le modèle d'embeddings doit correspondre à celui utilisé lors de l'indexation
        # La base actuelle a été indexée avec "intfloat/e5-base-v2" (768 dimensions)
        rag_pipeline = ImprovedRAGPipeline(
        openai_api_key=openai_api_key,
    # ontology_path=ontology_path,  # ← Retirer cette ligne (pas dans v2)
        chroma_path=chroma_path,
        collection_name=collection_name,
        quant_data_path=quant_data_path,
        llm_model="gpt-3.5-turbo",
        embedding_model="intfloat/e5-base-v2",
        reranker_model="antoinelouis/crossencoder-camembert-base-mmarcoFR"
)

        print("\n" + "="*60)
        print("OK SYSTEME RAG v3 INITIALISE AVEC SUCCES")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print(f"\nERREUR lors de l'initialisation du RAG: {e}\n")
        raise


# === ENDPOINTS ===

@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine - Redirige vers la documentation"""
    return {
        "message": "API Chatbot RAG v3 - Qualité de vie en Corse",
        "documentation": "/docs",
        "health_check": "/api/health",
        "query_endpoint": "/api/query"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé du serveur et du système RAG

    Returns:
        HealthResponse avec le statut du serveur
    """
    return HealthResponse(
        status="healthy" if rag_pipeline is not None else "unhealthy",
        rag_initialized=rag_pipeline is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """
    Pose une question au système RAG et retourne la réponse

    Args:
        request: QueryRequest contenant la question et les paramètres

    Returns:
        QueryResponse avec la réponse et les sources

    Raises:
        HTTPException 503: Si le système RAG n'est pas initialisé
        HTTPException 500: Si une erreur se produit lors du traitement
    """
    # Vérifier que le RAG est initialisé
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le système RAG n'est pas encore initialisé. Veuillez réessayer dans quelques instants."
        )

    try:
        print(f"\n[{datetime.now().isoformat()}] Nouvelle requête: {request.question}")

        """"
        # Exécuter la requête RAG
        answer, retrieval_results = rag_pipeline.query(
            question=request.question,
            k=request.k,
            use_reranking=request.use_reranking,
            include_quantitative=request.include_quantitative,
            commune_filter=request.commune_filter,
            use_ontology_enrichment=request.use_ontology_enrichment
        )"""

        answer, retrieval_results = rag_pipeline.query(
            question=request.question,
            k=request.k,
            use_reranking=request.use_reranking,
            include_quantitative=request.include_quantitative,
            commune_filter=request.commune_filter
            # Retirer use_ontology_enrichment
        )

        # Convertir les résultats en format API
        sources = [
            Source(
                content=result.text,  # L'attribut est 'text' et non 'content'
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
                "use_reranking": request.use_reranking,
                "use_ontology": request.use_ontology_enrichment,
                "include_quantitative": request.include_quantitative,
                "commune_filter": request.commune_filter,
                "num_sources": len(sources)
            },
            timestamp=datetime.now().isoformat()
        )

        print(f"[{datetime.now().isoformat()}] Réponse générée avec {len(sources)} sources")

        return response

    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ERREUR: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement de la requête: {str(e)}"
        )


# === POINT D'ENTRÉE ===

if __name__ == "__main__":
    # Configuration du serveur
    host = os.getenv("API_HOST", "0.0.0.0")  # 0.0.0.0 pour accepter les connexions externes
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    print("\n" + "="*60)
    print("DÉMARRAGE DU SERVEUR API")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Documentation: http://localhost:{port}/docs")
    print("="*60 + "\n")

    # Démarrer le serveur
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
