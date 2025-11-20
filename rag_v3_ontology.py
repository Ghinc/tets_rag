"""
RAG v3 - Système RAG avec enrichissement basé sur l'ontologie

Ce module implémente un système RAG qui utilise l'ontologie du bien-être
pour enrichir les requêtes avant le retrieval.

Différences avec RAG v2:
- Enrichissement automatique des requêtes via l'ontologie
- Ajout de termes sémantiquement liés issus de l'ontologie
- Meilleure compréhension du contexte via les dimensions

Architecture:
1. Query Enrichment: Enrichit la requête avec des termes de l'ontologie
2. Hybrid Retrieval: Comme v2 (BM25 + Vector Search)
3. Reranking: Comme v2 (Cross-encoder)
4. Response Generation: Comme v2 + mention de l'ontologie

Auteur: Claude Code
Date: 2025-10-28
"""

import os
import sys
from typing import List, Tuple, Optional, Dict
import pickle
import pandas as pd
import chromadb
import openai

# Imports des modules v2
from rag_v2_improved import (
    ImprovedSemanticChunker,
    FrenchEmbeddingModel,
    HybridRetriever,
    CrossEncoderReranker,
    QuantitativeDataHandler,
    RetrievalResult,
    ImprovedPromptBuilder
)

# Imports des modules ontology
from ontology_parser import OntologyParser
from query_enricher import QueryEnricher


class OntologyEnhancedPromptBuilder:
    """
    Constructeur de prompts enrichi avec l'ontologie

    Hérite du comportement de ImprovedPromptBuilder mais ajoute
    des informations sur les dimensions identifiées par l'ontologie.
    """

    SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de données sur la qualité de vie en Corse.

Tu as accès à:
1. Des extraits d'entretiens avec des habitants
2. Des données quantitatives sur des indicateurs de qualité de vie
3. Des informations contextuelles (Wikipedia) sur les communes
4. Une ontologie du bien-être qui structure les dimensions de la qualité de vie

PRINCIPES:
- Base tes réponses UNIQUEMENT sur les informations fournies
- Cite tes sources quand c'est pertinent
- Distingue clairement données qualitatives et quantitatives
- Si les données sont insuffisantes, indique-le explicitement
- Réponds de manière concise et factuelle
"""

    @staticmethod
    def build_rag_prompt(question: str,
                        qualitative_context: List[RetrievalResult],
                        quantitative_data: Optional[pd.DataFrame] = None,
                        statistics: Optional[Dict] = None,
                        ontology_metadata: Optional[Dict] = None) -> str:
        """
        Construit un prompt RAG avec enrichissement ontologique

        Args:
            question: Question de l'utilisateur
            qualitative_context: Résultats de retrieval (entretiens)
            quantitative_data: DataFrame avec données quantitatives
            statistics: Statistiques descriptives
            ontology_metadata: Métadonnées de l'ontologie (dimensions, etc.)

        Returns:
            Prompt formaté
        """
        prompt_parts = []

        # 0. Contexte ontologique (si disponible)
        if ontology_metadata and ontology_metadata.get('dimension_labels'):
            prompt_parts.append("=== CONTEXTE ONTOLOGIQUE ===\n")
            prompt_parts.append(f"Cette question concerne les dimensions suivantes du bien-etre territorial:")
            for dim_label in ontology_metadata['dimension_labels']:
                prompt_parts.append(f"  - {dim_label}")
            prompt_parts.append("")

        # 1. Contexte qualitatif (entretiens)
        prompt_parts.append("=== EXTRAITS D'ENTRETIENS ===\n")

        for i, result in enumerate(qualitative_context, 1):
            metadata = result.metadata
            commune = metadata.get('nom', metadata.get('commune', 'N/A'))
            source = metadata.get('source', 'N/A')

            prompt_parts.append(f"\n[Extrait {i}]")
            prompt_parts.append(f"Commune: {commune}")
            prompt_parts.append(f"Source: {source}")

            if 'num_entretien' in metadata:
                prompt_parts.append(f"Entretien n {metadata['num_entretien']}")

            prompt_parts.append(f"Pertinence: {result.score:.3f}")
            prompt_parts.append(f"\nContenu:\n{result.text}\n")
            prompt_parts.append("-" * 80)

        # 2. Contexte quantitatif (si disponible)
        if quantitative_data is not None and not quantitative_data.empty:
            prompt_parts.append("\n\n=== DONNEES QUANTITATIVES ===\n")

            quant_handler = QuantitativeDataHandler()
            table = quant_handler.format_as_table(quantitative_data)
            prompt_parts.append(table)

        # 3. Statistiques descriptives
        if statistics:
            prompt_parts.append("\n\n=== STATISTIQUES DESCRIPTIVES ===\n")
            for indicator, stats in statistics.items():
                prompt_parts.append(f"\n{indicator}:")
                prompt_parts.append(f"  - Moyenne: {stats['mean']:.2f}")
                prompt_parts.append(f"  - Mediane: {stats['median']:.2f}")
                prompt_parts.append(f"  - Min-Max: {stats['min']:.2f} - {stats['max']:.2f}")
                prompt_parts.append(f"  - Ecart-type: {stats['std']:.2f}")

        # 4. Question
        prompt_parts.append(f"\n\n=== QUESTION ===\n{question}")

        # 5. Instructions
        prompt_parts.append("\n\n=== INSTRUCTIONS ===")
        prompt_parts.append("Reponds a la question de maniere claire et concise en te basant sur les informations ci-dessus.")
        prompt_parts.append("Cite tes sources quand c'est pertinent (commune, n entretien, donnees quantitatives).")
        prompt_parts.append("Si les informations sont insuffisantes, indique-le clairement.")

        return "\n".join(prompt_parts)


class RAGPipelineWithOntology:
    """
    Pipeline RAG v3 - Avec enrichissement par ontologie

    Ce pipeline enrichit les requêtes avec l'ontologie du bien-être
    avant d'effectuer le retrieval, ce qui améliore la pertinence des résultats.
    """

    def __init__(self,
                 chroma_path: str = "./chroma_v2",
                 collection_name: str = "communes_corses_v2",
                 embedding_model: str = "intfloat/e5-base-v2",
                 reranker_model: str = "antoinelouis/crossencoder-camembert-base-mmarcoFR",
                 llm_model: str = "gpt-3.5-turbo",
                 openai_api_key: Optional[str] = None,
                 quant_data_path: str = "df_mean_by_commune.csv",
                 ontology_path: str = "ontology_be_2010_bilingue_fr_en.ttl"):
        """
        Initialise le pipeline RAG v3 avec ontologie

        Args:
            chroma_path: Chemin vers la base ChromaDB
            collection_name: Nom de la collection
            embedding_model: Modèle d'embeddings
            reranker_model: Modèle de reranking
            llm_model: Modèle LLM pour la génération
            openai_api_key: Clé API OpenAI
            quant_data_path: Chemin vers les données quantitatives
            ontology_path: Chemin vers l'ontologie
        """
        # Configuration OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key

        self.llm_model = llm_model

        # Composants du pipeline (identiques à v2)
        self.chunker = ImprovedSemanticChunker(chunk_size=500, chunk_overlap=100)
        self.embed_model = FrenchEmbeddingModel(embedding_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.quant_handler = QuantitativeDataHandler(quant_data_path)

        # Nouveaux composants pour l'ontologie
        print("Initialisation de l'ontologie...")
        self.ontology_parser = OntologyParser(ontology_path)
        self.ontology_parser.extract_all()
        self.query_enricher = QueryEnricher(self.ontology_parser)
        print("OK Ontologie initialisee")

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Documents pour BM25 (chargés depuis le cache)
        self.documents = []
        self.hybrid_retriever = None

        # Charger le cache
        self._load_cache_if_exists()

    def _load_cache_if_exists(self):
        """
        Charge le cache d'embeddings s'il existe (même cache que v2)
        """
        cache_path = "embeddings_v2.pkl"

        if not os.path.exists(cache_path):
            print("Aucun cache trouve. Vous devrez appeler ingest_documents() pour indexer les documents.")
            return

        print(f"Chargement du cache depuis {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.documents = cache_data.get('documents', [])

            if self.documents:
                print(f"OK {len(self.documents)} documents charges depuis le cache")

                # Initialiser le hybrid retriever
                print("Initialisation du retriever hybride depuis le cache...")
                self.hybrid_retriever = HybridRetriever(
                    self.collection,
                    self.documents
                )
                print("OK Retriever hybride initialise")

        except Exception as e:
            print(f"ERREUR lors du chargement du cache : {e}")

    def query(self, question: str,
              k: int = 5,
              use_reranking: bool = True,
              include_quantitative: bool = True,
              commune_filter: Optional[str] = None,
              use_ontology_enrichment: bool = True) -> Tuple[str, List[RetrievalResult]]:
        """
        Effectue une requête RAG complète avec enrichissement ontologique

        Args:
            question: Question de l'utilisateur
            k: Nombre de résultats à retourner
            use_reranking: Utiliser le reranking
            include_quantitative: Inclure les données quantitatives
            commune_filter: Filtrer par commune (optionnel)
            use_ontology_enrichment: Utiliser l'enrichissement ontologique (True par défaut pour v3)

        Returns:
            (réponse, résultats_retrieval)
        """
        if self.hybrid_retriever is None:
            raise ValueError("Le pipeline n'a pas ete initialise avec des documents. Appelez ingest_documents() d'abord.")

        # 0. Enrichissement de la requête avec l'ontologie (NOUVEAU dans v3)
        ontology_metadata = None
        query_to_use = question

        if use_ontology_enrichment:
            print("Enrichissement avec l'ontologie...")
            enrichment_result = self.query_enricher.enrich_query(question)
            query_to_use = enrichment_result["enriched_query"]
            ontology_metadata = enrichment_result["metadata"]
            print(f"  Dimensions: {enrichment_result['metadata']['dimension_labels']}")

        # 1. Encoder la requête (enrichie)
        query_embedding = self.embed_model.encode_query(query_to_use)

        # 2. Retrieval hybride
        print("Retrieval hybride...")
        results = self.hybrid_retriever.retrieve(
            query_to_use,
            query_embedding,
            k=k*2
        )

        # 3. Reranking (optionnel)
        if use_reranking:
            print("Reranking...")
            # Reranker avec la question ORIGINALE, pas la version enrichie
            results = self.reranker.rerank(question, results, top_k=k)
        else:
            results = results[:k]

        # 3.5. Appliquer le malus Wikipedia
        WIKI_PENALTY = 0.3  # Diviser par 3.33 le score des sources Wikipedia
        for result in results:
            if result.metadata.get('source') == 'wiki':
                result.score *= WIKI_PENALTY

        # 4. Récupérer les données quantitatives (optionnel)
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

        # 5. Construire le prompt (avec métadonnées ontologiques)
        prompt = OntologyEnhancedPromptBuilder.build_rag_prompt(
            question,
            results,
            quant_data,
            stats,
            ontology_metadata  # NOUVEAU
        )

        # 6. Générer la réponse avec LLM
        print("Generation de la reponse...")
        response = self._generate_response(prompt)

        return response, results

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
                    {"role": "system", "content": OntologyEnhancedPromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Erreur lors de la generation : {str(e)}"


def main():
    """
    Fonction principale pour tester le RAG v3
    """
    print("="*80)
    print("RAG V3 - Test avec ontologie")
    print("="*80)

    # Clé API OpenAI
    import os
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("❌ Clé API OpenAI non trouvée. Définissez la variable d'environnement OPENAI_API_KEY")

    # Initialiser le pipeline
    print("\nInitialisation du RAG v3...")
    rag = RAGPipelineWithOntology(
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        openai_api_key=OPENAI_API_KEY
    )

    # Questions de test
    test_questions = [
        "Combien y a-t-il de medecins a Afa ?",
        "Comment est le logement a Ajaccio ?",
        "Quels sont les problemes de transport a Bastia ?",
        "Quelles sont les dimensions du bien-être ?",
        "Sur quel type de données d'appuie le calcul du bine-être subjectif ?",
        "Quels indicateurs combinent des dimensions du bien-être objectif et subjectif ?"
    ]

    print("\n" + "="*80)
    print("TESTS")
    print("="*80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] {question}")
        print("-"*80)

        try:
            response, results = rag.query(
                question,
                k=3,
                use_ontology_enrichment=True
            )

            print(f"\nReponse:")
            print(response)
            print(f"\nNombre de chunks recuperes: {len(results)}")

        except Exception as e:
            print(f"ERREUR: {e}")


if __name__ == "__main__":
    main()
