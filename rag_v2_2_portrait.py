"""
RAG v2.2 - Version avec filtres portrait
Étend RAG v2 avec:
- Filtrage par âge, genre, profession, dimension
- Détection automatique des filtres dans les questions
- Requêtes spécifiques aux verbatims portrait
"""

import os
import re
import pickle
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Détection de communes et portraits
from commune_detector import detect_commune
from portrait_detector import detect_portrait_filters

# Embeddings et modèles
from sentence_transformers import SentenceTransformer, CrossEncoder

# ChromaDB
import chromadb
from chromadb.utils import embedding_functions

# LangChain pour chunking et retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# OpenAI
import openai

# Importer les classes de base du RAG v2
from rag_v2_improved import (
    RetrievalResult,
    ImprovedSemanticChunker,
    FrenchEmbeddingModel,
    CrossEncoderReranker,
    QuantitativeDataHandler,
    load_ontology_mapping,
    enrich_metadata_with_ontology,
    enrich_all_metadatas
)


class PortraitHybridRetriever:
    """
    Retriever hybride étendu avec support des filtres portrait
    """

    def __init__(self, chroma_collection, documents: List[Document],
                 dense_weight: float = 0.6, sparse_weight: float = 0.4):
        """
        Args:
            chroma_collection: Collection ChromaDB pour recherche dense
            documents: Liste de Documents LangChain pour BM25
            dense_weight: Poids de la recherche dense (0-1)
            sparse_weight: Poids de la recherche sparse (0-1)
        """
        self.chroma_collection = chroma_collection
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Initialiser BM25
        print("Initialisation du retriever BM25...")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 20  # Récupérer top 20 pour fusion

        # Stocker les documents pour référence
        self.documents = documents
        self.doc_map = {doc.metadata.get('id', i): doc for i, doc in enumerate(documents)}

    def _build_portrait_where_clause(self,
                                     commune_filter: Optional[str] = None,
                                     portrait_filters: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """
        Construit la clause WHERE ChromaDB pour les filtres portrait

        Args:
            commune_filter: Filtre par commune
            portrait_filters: Filtres portrait (age_min, age_max, genre, profession, dimension)

        Returns:
            Clause WHERE pour ChromaDB ou None
        """
        conditions = []

        # Filtre par commune
        if commune_filter:
            conditions.append({"nom": commune_filter})

        # Filtres portrait
        if portrait_filters and portrait_filters.get('has_portrait_filter'):
            # Forcer la recherche sur les verbatims portrait uniquement
            conditions.append({"source": "portrait_verbatim"})

            # Filtre par âge (range)
            if portrait_filters.get('age_min') is not None:
                conditions.append({"age_exact": {"$gte": portrait_filters['age_min']}})

            if portrait_filters.get('age_max') is not None:
                conditions.append({"age_exact": {"$lte": portrait_filters['age_max']}})

            # Filtre par genre
            if portrait_filters.get('genre'):
                conditions.append({"genre": portrait_filters['genre']})

            # Filtre par profession
            if portrait_filters.get('profession'):
                conditions.append({"profession": portrait_filters['profession']})

            # Filtre par dimension
            if portrait_filters.get('dimension'):
                conditions.append({"dimension": portrait_filters['dimension']})

        # Construire la clause WHERE
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def retrieve(self, query: str, query_embedding: np.ndarray,
                k: int = 10,
                commune_filter: Optional[str] = None,
                portrait_filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        Retrieval hybride avec filtres portrait

        Args:
            query: Requête textuelle
            query_embedding: Embedding de la requête
            k: Nombre de résultats à retourner
            commune_filter: Filtre par commune (optionnel)
            portrait_filters: Filtres portrait (optionnel)

        Returns:
            Liste de RetrievalResult triés par score
        """
        # 1. Construire la clause WHERE
        where_clause = self._build_portrait_where_clause(commune_filter, portrait_filters)

        # 2. Recherche dense (ChromaDB)
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": 20,
            "include": ["documents", "metadatas", "distances"]
        }

        if where_clause:
            query_params["where"] = where_clause
            # Afficher les filtres actifs
            filters_info = []
            if commune_filter:
                filters_info.append(f"commune={commune_filter}")
            if portrait_filters and portrait_filters.get('has_portrait_filter'):
                if portrait_filters.get('age_min') or portrait_filters.get('age_max'):
                    age_range = f"{portrait_filters.get('age_min', '?')}-{portrait_filters.get('age_max', '?')} ans"
                    filters_info.append(f"âge={age_range}")
                if portrait_filters.get('genre'):
                    filters_info.append(f"genre={portrait_filters['genre']}")
                if portrait_filters.get('profession'):
                    filters_info.append(f"profession={portrait_filters['profession']}")
                if portrait_filters.get('dimension'):
                    filters_info.append(f"dimension={portrait_filters['dimension']}")
            print(f"[FILTRES] {', '.join(filters_info)}")

        try:
            dense_results = self.chroma_collection.query(**query_params)
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur recherche ChromaDB: {e}")
            # Fallback: recherche sans filtres
            query_params.pop("where", None)
            dense_results = self.chroma_collection.query(**query_params)

        # 3. Recherche sparse (BM25) - pas de filtrage ici, on filtre après fusion
        sparse_docs = self.bm25_retriever.invoke(query)

        # Filtrer les résultats BM25 si portrait_filters est actif
        if portrait_filters and portrait_filters.get('has_portrait_filter'):
            filtered_sparse = []
            for doc in sparse_docs:
                meta = doc.metadata
                if meta.get('source') != 'portrait_verbatim':
                    continue

                # Vérifier les autres filtres
                match = True
                if portrait_filters.get('age_min') and meta.get('age_exact', -1) < portrait_filters['age_min']:
                    match = False
                if portrait_filters.get('age_max') and meta.get('age_exact', 999) > portrait_filters['age_max']:
                    match = False
                if portrait_filters.get('genre') and meta.get('genre') != portrait_filters['genre']:
                    match = False
                if portrait_filters.get('profession') and meta.get('profession') != portrait_filters['profession']:
                    match = False
                if portrait_filters.get('dimension') and meta.get('dimension') != portrait_filters['dimension']:
                    match = False

                if match:
                    filtered_sparse.append(doc)

            sparse_docs = filtered_sparse

        # 4. Normaliser les scores
        dense_scores = self._normalize_dense_scores(dense_results['distances'][0] if dense_results['distances'] and dense_results['distances'][0] else [])
        sparse_scores = self._normalize_bm25_scores(sparse_docs)

        # 5. Fusionner les résultats
        merged_results = self._merge_results(
            dense_results, dense_scores,
            sparse_docs, sparse_scores
        )

        # 6. Trier et retourner top k
        merged_results.sort(key=lambda x: x.score, reverse=True)

        return merged_results[:k]

    def _normalize_dense_scores(self, distances: List[float]) -> List[float]:
        """
        Normalise les distances L2 en scores de similarité [0, 1]
        """
        if not distances:
            return []

        max_dist = max(distances) if distances else 1.0
        scores = [1.0 - (d / max_dist) for d in distances]
        return scores

    def _normalize_bm25_scores(self, documents: List[Document]) -> List[float]:
        """
        Les scores BM25 sont déjà normalisés par le retriever
        """
        scores = [1.0 / (i + 1) for i in range(len(documents))]
        return scores

    def _merge_results(self, dense_results: Dict, dense_scores: List[float],
                      sparse_docs: List[Document], sparse_scores: List[float]) -> List[RetrievalResult]:
        """
        Fusionne les résultats dense et sparse avec pondération
        """
        results_map = {}

        # Ajouter résultats dense
        if dense_results['documents'] and dense_results['documents'][0]:
            for i, (doc, metadata, score) in enumerate(zip(
                dense_results['documents'][0],
                dense_results['metadatas'][0],
                dense_scores
            )):
                doc_id = metadata.get('id', f"dense_{i}")
                weighted_score = score * self.dense_weight

                results_map[doc_id] = RetrievalResult(
                    text=doc,
                    metadata=metadata,
                    score=weighted_score,
                    source_type='dense'
                )

        # Ajouter résultats sparse (fusion si déjà présent)
        for doc, score in zip(sparse_docs, sparse_scores):
            doc_id = doc.metadata.get('id', hash(doc.page_content))
            weighted_score = score * self.sparse_weight

            if doc_id in results_map:
                results_map[doc_id].score += weighted_score
                results_map[doc_id].source_type = 'hybrid'
            else:
                results_map[doc_id] = RetrievalResult(
                    text=doc.page_content,
                    metadata=doc.metadata,
                    score=weighted_score,
                    source_type='sparse'
                )

        return list(results_map.values())


class PortraitPromptBuilder:
    """
    Constructeur de prompts adapté aux requêtes portrait
    """

    SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de données sur la qualité de vie en Corse.

Tu as accès à:
1. Des extraits d'entretiens avec des habitants
2. Des verbatims d'enquêtes avec profils démographiques (âge, genre, profession)
3. Des données quantitatives sur des indicateurs de qualité de vie

PRINCIPES:
- Base tes réponses UNIQUEMENT sur les informations fournies
- Cite tes sources quand c'est pertinent
- Distingue clairement données qualitatives et quantitatives
- Si les données sont insuffisantes, indique-le explicitement
- Quand tu analyses des verbatims filtrés par profil, mentionne le profil concerné
- Réponds de manière concise et factuelle
"""

    @staticmethod
    def build_portrait_prompt(question: str,
                              qualitative_context: List[RetrievalResult],
                              portrait_filters: Optional[Dict[str, Any]] = None,
                              quantitative_data: Optional[pd.DataFrame] = None,
                              statistics: Optional[Dict] = None) -> str:
        """
        Construit un prompt RAG avec contexte portrait

        Args:
            question: Question de l'utilisateur
            qualitative_context: Résultats de retrieval
            portrait_filters: Filtres portrait actifs
            quantitative_data: DataFrame avec données quantitatives
            statistics: Statistiques descriptives

        Returns:
            Prompt formaté
        """
        prompt_parts = []

        # 1. Contexte portrait (si filtres actifs)
        if portrait_filters and portrait_filters.get('has_portrait_filter'):
            prompt_parts.append("=== CONTEXTE DÉMOGRAPHIQUE ===\n")
            prompt_parts.append("Analyse basée sur les verbatims filtrés par:")

            if portrait_filters.get('age_min') or portrait_filters.get('age_max'):
                age_min = portrait_filters.get('age_min', 15)
                age_max = portrait_filters.get('age_max', 100)
                if portrait_filters.get('age_range'):
                    prompt_parts.append(f"- Tranche d'âge: {portrait_filters['age_range']} ({age_min}-{age_max} ans)")
                else:
                    prompt_parts.append(f"- Âge: {age_min}-{age_max} ans")

            if portrait_filters.get('genre'):
                prompt_parts.append(f"- Genre: {portrait_filters['genre']}")

            if portrait_filters.get('profession'):
                prompt_parts.append(f"- Profession: {portrait_filters['profession']}")

            if portrait_filters.get('dimension'):
                prompt_parts.append(f"- Thème qualité de vie: {portrait_filters['dimension']}")

            prompt_parts.append("")

        # 2. Contexte qualitatif (verbatims/entretiens)
        prompt_parts.append("=== EXTRAITS / VERBATIMS ===\n")

        for i, result in enumerate(qualitative_context, 1):
            metadata = result.metadata
            commune = metadata.get('nom', metadata.get('commune', 'N/A'))
            source = metadata.get('source', 'N/A')

            prompt_parts.append(f"\n[Extrait {i}]")
            prompt_parts.append(f"Commune: {commune}")

            # Afficher les infos portrait si disponibles
            if source == 'portrait_verbatim':
                genre = metadata.get('genre', 'N/A')
                age = metadata.get('age_exact', 'N/A')
                age_range = metadata.get('age_range', '')
                profession = metadata.get('profession', 'N/A')
                dimension = metadata.get('dimension', 'N/A')

                prompt_parts.append(f"Profil: {genre}, {age} ans ({age_range}), {profession}")
                prompt_parts.append(f"Thème: {dimension}")
            else:
                prompt_parts.append(f"Source: {source}")
                if 'num_entretien' in metadata:
                    prompt_parts.append(f"Entretien n°{metadata['num_entretien']}")

            prompt_parts.append(f"Pertinence: {result.score:.3f}")
            prompt_parts.append(f"\nContenu:\n{result.text}\n")
            prompt_parts.append("-" * 80)

        # 3. Contexte quantitatif (si disponible)
        if quantitative_data is not None and not quantitative_data.empty:
            prompt_parts.append("\n\n=== DONNÉES QUANTITATIVES ===\n")
            quant_handler = QuantitativeDataHandler()
            table = quant_handler.format_as_table(quantitative_data)
            prompt_parts.append(table)

        # 4. Statistiques descriptives
        if statistics:
            prompt_parts.append("\n\n=== STATISTIQUES DESCRIPTIVES ===\n")
            for indicator, stats in statistics.items():
                prompt_parts.append(f"\n{indicator}:")
                prompt_parts.append(f"  - Moyenne: {stats['mean']:.2f}")
                prompt_parts.append(f"  - Médiane: {stats['median']:.2f}")
                prompt_parts.append(f"  - Min-Max: {stats['min']:.2f} - {stats['max']:.2f}")

        # 5. Question
        prompt_parts.append(f"\n\n=== QUESTION ===\n{question}")

        # 6. Instructions adaptées
        prompt_parts.append("\n\n=== INSTRUCTIONS ===")
        if portrait_filters and portrait_filters.get('has_portrait_filter'):
            prompt_parts.append("Réponds à la question en te basant sur les verbatims du profil démographique indiqué.")
            prompt_parts.append("Mentionne les caractéristiques du profil dans ta réponse quand c'est pertinent.")
        else:
            prompt_parts.append("Réponds à la question de manière claire et concise en te basant sur les informations ci-dessus.")

        prompt_parts.append("Cite tes sources quand c'est pertinent (commune, profil, données quantitatives).")
        prompt_parts.append("Si les informations sont insuffisantes, indique-le clairement.")

        return "\n".join(prompt_parts)


class PortraitRAGPipeline:
    """
    Pipeline RAG v2.2 avec support des filtres portrait
    """

    def __init__(self,
                 chroma_path: str = "./chroma_portrait",
                 collection_name: str = "portrait_verbatims",
                 embedding_model: str = "BAAI/bge-m3",
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 llm_model: str = "gpt-3.5-turbo",
                 openai_api_key: Optional[str] = None,
                 quant_data_path: str = "df_mean_by_commune.csv"):
        """
        Initialise le pipeline RAG portrait

        Args:
            chroma_path: Chemin vers la base ChromaDB portrait (défaut: ./chroma_portrait)
            collection_name: Nom de la collection portrait (défaut: portrait_verbatims)
            embedding_model: Modèle d'embeddings (défaut: BAAI/bge-m3)
            reranker_model: Modèle de reranking (défaut: BAAI/bge-reranker-v2-m3)
            llm_model: Modèle LLM pour la génération (défaut: gpt-3.5-turbo)
            openai_api_key: Clé API OpenAI
            quant_data_path: Chemin vers les données quantitatives
        """
        # Configuration OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key

        self.llm_model = llm_model

        # Composants du pipeline
        self.chunker = ImprovedSemanticChunker(chunk_size=500, chunk_overlap=100)
        self.embed_model = FrenchEmbeddingModel(embedding_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.quant_handler = QuantitativeDataHandler(quant_data_path)

        # ChromaDB - Base séparée pour les verbatims portrait
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Récupérer ou créer la collection portrait
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Documents pour BM25
        self.documents = []
        self.hybrid_retriever = None

        # Charger automatiquement le cache s'il existe
        self._load_cache_if_exists()

    def _load_cache_if_exists(self):
        """
        Charge automatiquement les documents depuis ChromaDB portrait
        """
        try:
            collection_count = self.collection.count()
            if collection_count > 0:
                print(f"ChromaDB portrait contient {collection_count} verbatims")

                # Charger les documents depuis ChromaDB
                print("Chargement des documents depuis ChromaDB portrait...")
                chroma_data = self.collection.get(include=['documents', 'metadatas'])

                if chroma_data['documents']:
                    self.documents = []
                    for doc_text, metadata in zip(chroma_data['documents'], chroma_data['metadatas']):
                        self.documents.append(Document(
                            page_content=doc_text,
                            metadata=metadata
                        ))

                    print(f"OK {len(self.documents)} documents reconstruits depuis ChromaDB")

                    print("Initialisation du retriever hybride portrait...")
                    self.hybrid_retriever = PortraitHybridRetriever(
                        self.collection,
                        self.documents,
                        dense_weight=0.6,
                        sparse_weight=0.4
                    )
                    print("OK Retriever hybride portrait initialisé")
            else:
                print("Aucun document dans ChromaDB.")

        except Exception as e:
            print(f"AVERTISSEMENT Erreur lors de la vérification de ChromaDB: {e}")

    def query(self, question: str,
             k: int = 5,
             use_reranking: bool = True,
             include_quantitative: bool = True,
             commune_filter: Optional[str] = None,
             portrait_filters: Optional[Dict[str, Any]] = None,
             auto_detect_filters: bool = True) -> Tuple[str, List[RetrievalResult], Dict[str, Any]]:
        """
        Effectue une requête RAG avec support portrait

        Args:
            question: Question de l'utilisateur
            k: Nombre de résultats à retourner
            use_reranking: Utiliser le reranking
            include_quantitative: Inclure les données quantitatives
            commune_filter: Filtrer par commune (optionnel)
            portrait_filters: Filtres portrait explicites (optionnel)
            auto_detect_filters: Détecter automatiquement les filtres

        Returns:
            (réponse, résultats_retrieval, filtres_détectés)
        """
        if self.hybrid_retriever is None:
            raise ValueError("Le pipeline n'a pas été initialisé. Indexez d'abord les documents.")

        detected_filters = {}

        # 1. Détecter automatiquement la commune
        detected_commune = detect_commune(question)
        if detected_commune:
            print(f"[AUTO-DETECT] Commune détectée: {detected_commune}")
            commune_filter = detected_commune

        # 2. Détecter automatiquement les filtres portrait
        if auto_detect_filters and portrait_filters is None:
            portrait_filters = detect_portrait_filters(question)
            if portrait_filters.get('has_portrait_filter'):
                print(f"[AUTO-DETECT] Filtres portrait détectés:")
                if portrait_filters.get('age_min') or portrait_filters.get('age_max'):
                    print(f"  - Âge: {portrait_filters.get('age_min', '?')}-{portrait_filters.get('age_max', '?')} ans")
                if portrait_filters.get('genre'):
                    print(f"  - Genre: {portrait_filters['genre']}")
                if portrait_filters.get('profession'):
                    print(f"  - Profession: {portrait_filters['profession']}")
                if portrait_filters.get('dimension'):
                    print(f"  - Dimension: {portrait_filters['dimension']}")

        detected_filters = {
            'commune': commune_filter,
            'portrait': portrait_filters
        }

        # 3. Encoder la requête
        query_embedding = self.embed_model.encode_query(question)

        # 4. Retrieval hybride avec filtres portrait
        print("Retrieval hybride portrait...")
        results = self.hybrid_retriever.retrieve(
            question,
            query_embedding,
            k=k*2,
            commune_filter=commune_filter,
            portrait_filters=portrait_filters
        )

        # 5. Reranking (optionnel)
        if use_reranking and results:
            print("Reranking...")
            results = self.reranker.rerank(question, results, top_k=k)
        else:
            results = results[:k]

        # 6. Récupérer les données quantitatives (optionnel)
        quant_data = None
        stats = None
        if include_quantitative:
            commune = commune_filter
            if not commune and results:
                commune = results[0].metadata.get('nom') or results[0].metadata.get('commune')

            if commune:
                quant_data = self.quant_handler.query_structured_data(commune=commune)
                if not quant_data.empty:
                    stats = self.quant_handler.extract_statistics(quant_data)

        # 7. Construire le prompt
        prompt = PortraitPromptBuilder.build_portrait_prompt(
            question,
            results,
            portrait_filters,
            quant_data,
            stats
        )

        # 8. Générer la réponse avec LLM
        print("Génération de la réponse...")
        response = self._generate_response(prompt)

        return response, results, detected_filters

    def _generate_response(self, prompt: str) -> str:
        """
        Génère une réponse avec le LLM
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": PortraitPromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Erreur lors de la génération: {str(e)}"

    def get_portrait_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur les verbatims portrait indexés
        """
        try:
            portrait_results = self.collection.get(
                where={"source": "portrait_verbatim"},
                include=["metadatas"]
            )

            if not portrait_results['metadatas']:
                return {"count": 0}

            metadatas = portrait_results['metadatas']

            # Statistiques
            stats = {
                "count": len(metadatas),
                "genres": {},
                "age_ranges": {},
                "professions": {},
                "dimensions": {},
                "communes": {}
            }

            for m in metadatas:
                # Genre
                genre = m.get('genre', 'Non spécifié')
                stats['genres'][genre] = stats['genres'].get(genre, 0) + 1

                # Tranche d'âge
                age_range = m.get('age_range', 'Non spécifié')
                stats['age_ranges'][age_range] = stats['age_ranges'].get(age_range, 0) + 1

                # Profession
                profession = m.get('profession', 'Non spécifié')
                stats['professions'][profession] = stats['professions'].get(profession, 0) + 1

                # Dimension
                dimension = m.get('dimension', 'Non spécifié')
                stats['dimensions'][dimension] = stats['dimensions'].get(dimension, 0) + 1

                # Commune
                commune = m.get('nom', 'Non spécifié')
                stats['communes'][commune] = stats['communes'].get(commune, 0) + 1

            return stats

        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation du pipeline RAG v2.2 portrait
    """

    # Configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("[AVERTISSEMENT] Clé API OpenAI non trouvée. Les réponses LLM ne fonctionneront pas.")

    # Initialiser le pipeline
    print("=" * 60)
    print("INITIALISATION DU PIPELINE RAG v2.2 PORTRAIT")
    print("=" * 60)

    rag = PortraitRAGPipeline(
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        llm_model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )

    # Afficher les statistiques portrait
    print("\n" + "=" * 60)
    print("STATISTIQUES DES VERBATIMS PORTRAIT")
    print("=" * 60)

    stats = rag.get_portrait_stats()
    if stats.get('count', 0) > 0:
        print(f"\nTotal verbatims portrait: {stats['count']}")
        print(f"\nRépartition par genre:")
        for genre, count in stats['genres'].items():
            print(f"  - {genre}: {count}")
        print(f"\nRépartition par tranche d'âge:")
        for age_range, count in stats['age_ranges'].items():
            print(f"  - {age_range}: {count}")
        print(f"\nTop 5 professions:")
        sorted_professions = sorted(stats['professions'].items(), key=lambda x: -x[1])[:5]
        for profession, count in sorted_professions:
            print(f"  - {profession}: {count}")
    else:
        print("\nAucun verbatim portrait indexé.")
        print("Exécutez d'abord: python index_portrait_verbatims.py")

    # Questions de test
    test_questions = [
        "Que pensent les jeunes de 18-25 ans de la santé ?",
        "Quel est l'avis des étudiants sur le logement ?",
        "Comment les femmes perçoivent-elles les transports ?",
        "Quelles sont les priorités des retraités à Bastia ?",
        "Les hommes salariés sont-ils satisfaits de leur qualité de vie ?",
    ]

    print("\n" + "=" * 60)
    print("TESTS DE REQUÊTES PORTRAIT")
    print("=" * 60)

    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print('=' * 60)

        try:
            response, results, detected_filters = rag.query(
                question,
                k=3,
                use_reranking=True,
                include_quantitative=False,
                auto_detect_filters=True
            )

            print("\n--- RÉSULTATS DE RETRIEVAL ---")
            for i, result in enumerate(results, 1):
                meta = result.metadata
                print(f"\n[{i}] Score: {result.score:.3f} | Type: {result.source_type}")
                if meta.get('source') == 'portrait_verbatim':
                    print(f"    Profil: {meta.get('genre')}, {meta.get('age_exact')} ans, {meta.get('profession')}")
                    print(f"    Dimension: {meta.get('dimension')}")
                print(f"    Commune: {meta.get('nom', 'N/A')}")
                print(f"    Texte: {result.text[:100]}...")

            print("\n--- RÉPONSE GÉNÉRÉE ---")
            print(response)

        except Exception as e:
            print(f"Erreur: {e}")

        print("\n")
