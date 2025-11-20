"""
RAG v2 - Version améliorée avec:
- Chunking sémantique avec overlap
- Embedding français optimisé
- Hybrid search (BM25 + dense)
- Reranking avec cross-encoder
- Dual-path pour données quantitatives
- Prompt engineering amélioré
"""

import os
import re
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

# Embeddings et modèles
from sentence_transformers import SentenceTransformer, CrossEncoder

# ChromaDB
import chromadb
from chromadb.config import Settings

# LangChain pour chunking et retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# OpenAI
import openai


@dataclass
class RetrievalResult:
    """Résultat de retrieval enrichi"""
    text: str
    metadata: Dict
    score: float
    source_type: str  # 'dense', 'sparse', 'reranked'


class ImprovedSemanticChunker:
    """
    Chunker sémantique amélioré pour entretiens semi-directifs
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Taille cible des chunks en caractères
            chunk_overlap: Overlap entre chunks (20-30% recommandé)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Splitter récursif qui respecte la structure
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Double saut de ligne (fin de paragraphe)
                "\n",    # Simple saut de ligne
                ". ",    # Fin de phrase
                ", ",    # Virgule
                " ",     # Espace
                ""       # Caractère par caractère en dernier recours
            ],
            keep_separator=True
        )

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        Découpe le texte en chunks avec overlap

        Args:
            text: Texte à découper
            metadata: Métadonnées à associer à chaque chunk

        Returns:
            Liste de Documents LangChain
        """
        if metadata is None:
            metadata = {}

        # Nettoyage du texte
        text = text.strip()

        # Chunking
        chunks = self.splitter.split_text(text)

        # Créer des Documents avec métadonnées
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata['chunk_index'] = i
            doc_metadata['total_chunks'] = len(chunks)

            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))

        return documents

    def chunk_interview_qa(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        Chunking spécialisé pour entretiens Q/R
        Garde les paires question-réponse ensemble
        """
        if metadata is None:
            metadata = {}

        # Pattern pour détecter Q: ... R: ...
        qa_pattern = r'Q\s*[:：]\s*(.*?)(?=R\s*[:：])(R\s*[:：]\s*.*?)(?=Q\s*[:：]|$)'

        qa_pairs = re.findall(qa_pattern, text, re.DOTALL | re.IGNORECASE)

        if not qa_pairs:
            # Pas de structure Q/R détectée, utiliser chunking standard
            return self.chunk_text(text, metadata)

        # Créer un document par paire Q/R
        documents = []
        for i, (question, response) in enumerate(qa_pairs):
            qa_text = f"{question.strip()}\n{response.strip()}"

            doc_metadata = metadata.copy()
            doc_metadata['chunk_index'] = i
            doc_metadata['total_chunks'] = len(qa_pairs)
            doc_metadata['type'] = 'qa_pair'

            documents.append(Document(
                page_content=qa_text,
                metadata=doc_metadata
            ))

        return documents


class FrenchEmbeddingModel:
    """
    Wrapper pour modèle d'embeddings optimisé pour le français
    """

    def __init__(self, model_name: str = "dangvantuan/sentence-camembert-large"):
        """
        Args:
            model_name: Nom du modèle HuggingFace
                Options:
                - 'dangvantuan/sentence-camembert-large' (recommandé pour français)
                - 'intfloat/multilingual-e5-large' (multilingual plus performant)
                - 'intfloat/e5-base-v2' (votre modèle actuel, baseline)
        """
        print(f"Chargement du modèle d'embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Déterminer si le modèle nécessite des préfixes (famille e5)
        self.use_prefixes = 'e5' in model_name.lower()

    def encode_documents(self, texts: List[str], batch_size: int = 128,
                        show_progress: bool = True) -> np.ndarray:
        """
        Encode des documents en embeddings
        """
        if self.use_prefixes:
            texts = [f"passage: {text}" for text in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode une requête en embedding
        """
        if self.use_prefixes:
            query = f"query: {query}"

        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding


class HybridRetriever:
    """
    Retriever hybride combinant recherche dense (sémantique) et sparse (BM25)
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

    def retrieve(self, query: str, query_embedding: np.ndarray,
                k: int = 10) -> List[RetrievalResult]:
        """
        Retrieval hybride avec fusion des scores et boost intelligent

        Args:
            query: Requête textuelle
            query_embedding: Embedding de la requête
            k: Nombre de résultats à retourner

        Returns:
            Liste de RetrievalResult triés par score
        """
        # 1. Recherche dense (ChromaDB)
        dense_results = self.chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=20,
            include=["documents", "metadatas", "distances"]
        )

        # 2. Recherche sparse (BM25)
        sparse_docs = self.bm25_retriever.invoke(query)

        # 3. Normaliser les scores
        dense_scores = self._normalize_dense_scores(dense_results['distances'][0])
        sparse_scores = self._normalize_bm25_scores(sparse_docs)

        # 4. Fusionner les résultats
        merged_results = self._merge_results(
            dense_results, dense_scores,
            sparse_docs, sparse_scores
        )

        # 5. Appliquer le boost intelligent pour be_verbatims
        merged_results = self._apply_source_boost(merged_results, query)

        # 6. Trier et retourner top k
        merged_results.sort(key=lambda x: x.score, reverse=True)

        return merged_results[:k]

    def _normalize_dense_scores(self, distances: List[float]) -> List[float]:
        """
        Normalise les distances L2 en scores de similarité [0, 1]
        Distance plus faible = score plus élevé
        """
        if not distances:
            return []

        # Convertir distance en similarité
        max_dist = max(distances) if distances else 1.0
        scores = [1.0 - (d / max_dist) for d in distances]

        return scores

    def _normalize_bm25_scores(self, documents: List[Document]) -> List[float]:
        """
        Les scores BM25 sont déjà normalisés par le retriever
        """
        # BM25Retriever ne retourne pas les scores directement
        # On utilise un score uniforme décroissant basé sur le rang
        scores = [1.0 / (i + 1) for i in range(len(documents))]
        return scores

    def _apply_source_boost(self, results: List[RetrievalResult], question: str) -> List[RetrievalResult]:
        """
        Applique un boost intelligent aux sources selon les mots-clés de la question
        Applique aussi un boost géographique TRÈS FORT pour la commune mentionnée
        Applique aussi un malus aux sources Wikipedia (toujours)

        Détection automatique de la source demandée :
        - Verbatims : verbatim, questionnaire, répondant, perception, opinion
        - Entretiens : entretien, interview, témoignage, raconte
        - OppChoVec : score, indicateur, opportunité, choix, vécu, opp/cho/vec
        - Wiki : contexte, histoire, géographie, présentation
        - Communes : Ajaccio, Bastia, etc. (boost x10 pour la commune mentionnée)
        """
        question_lower = question.lower()

        # Liste des communes corses principales
        communes_corses = [
            'ajaccio', 'bastia', 'porto-vecchio', 'corte', 'calvi', 'bonifacio',
            'propriano', 'sartène', 'ile-rousse', 'ghisonaccia', 'porto', 'piana',
            'aleria', 'cervione', 'saint-florent', 'sollacaro', 'sisco', 'olmeto',
            'zonza', 'levie', 'santa-maria-di-lota', 'vescovato', 'lucciana'
        ]

        # Détecter la commune mentionnée dans la question
        mentioned_commune = None
        for commune in communes_corses:
            if commune in question_lower:
                mentioned_commune = commune
                break

        # Dictionnaire de mots-clés par type de source
        source_keywords = {
            'be_verbatims': [
                'verbatim', 'questionnaire', 'répondant', 'plébiscit',
                'dimension', 'perception', 'opinion', 'pens', 'ressent',
                'sélectionn', 'priorit', 'préfér'
            ],
            'entretien': [
                'entretien', 'interview', 'témoignage', 'raconte', 'dit', 'explique',
                'parole', 'propos', 'discours', 'récit'
            ],
            'oppchovec': [
                'score', 'indicateur', 'opportunité', 'choix', 'vécu', 'oppchovec',
                'opp', 'cho', 'vec',
                'opp1', 'opp2', 'opp3', 'opp4', 'chx1', 'chx2', 'vec1', 'vec2',
                'éducation', 'inégalité', 'mobilité', 'connectivité', 'numérique',
                'transport', 'logement', 'emploi', 'santé', 'insécurité'
            ],
            'wiki': [
                'contexte', 'histoire', 'géographie', 'présentation',
                'population', 'superficie', 'arrondissement', 'canton'
            ]
        }

        # Détecter quelles sources sont demandées
        requested_sources = {}
        for source_type, keywords in source_keywords.items():
            if any(kw in question_lower for kw in keywords):
                requested_sources[source_type] = True

        # Boost/Penalty multipliers
        SOURCE_BOOST = 5.0  # Boost fort pour les sources explicitement demandées
        COMMUNE_BOOST = 10.0  # Boost TRÈS FORT pour la commune mentionnée
        WIKI_PENALTY = 0.3  # Malus pour Wikipedia

        for result in results:
            source = result.metadata.get('source')

            # Récupérer le nom de la commune du document
            doc_commune = result.metadata.get('commune') or result.metadata.get('nom', '')
            doc_commune_lower = doc_commune.lower() if doc_commune else ''

            # BOOST GÉOGRAPHIQUE (prioritaire) - si une commune est mentionnée
            if mentioned_commune and mentioned_commune in doc_commune_lower:
                result.score *= COMMUNE_BOOST

            # Appliquer le boost si la source est demandée
            if source in requested_sources:
                result.score *= SOURCE_BOOST

            # Appliquer le malus Wikipedia (toujours, sauf si explicitement demandé)
            if source == 'wiki' and 'wiki' not in requested_sources:
                result.score *= WIKI_PENALTY

        return results

    def _merge_results(self, dense_results: Dict, dense_scores: List[float],
                      sparse_docs: List[Document], sparse_scores: List[float]) -> List[RetrievalResult]:
        """
        Fusionne les résultats dense et sparse avec pondération
        """
        results_map = {}

        # Ajouter résultats dense
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
                # Fusionner les scores
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


class CrossEncoderReranker:
    """
    Reranker utilisant un cross-encoder pour affiner les résultats
    """

    def __init__(self, model_name: str = "antoinelouis/crossencoder-camembert-base-mmarcoFR"):
        """
        Args:
            model_name: Modèle de cross-encoder
                Options:
                - 'antoinelouis/crossencoder-camembert-base-mmarcoFR' (français)
                - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (multilingual, rapide)
        """
        print(f"Chargement du reranker: {model_name}")
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[RetrievalResult],
               top_k: int = 5) -> List[RetrievalResult]:
        """
        Rerank les résultats en utilisant le cross-encoder

        Args:
            query: Requête originale
            results: Résultats à reranker
            top_k: Nombre de résultats à retourner

        Returns:
            Liste de résultats reranked
        """
        if not results:
            return []

        # Préparer les paires (query, document)
        pairs = [[query, result.text] for result in results]

        # Calculer les scores de pertinence
        scores = self.reranker.predict(pairs)

        # Mettre à jour les scores et trier
        reranked_results = []
        for result, score in zip(results, scores):
            result.score = float(score)
            result.source_type = 'reranked'
            reranked_results.append(result)

        reranked_results.sort(key=lambda x: x.score, reverse=True)

        return reranked_results[:top_k]


class QuantitativeDataHandler:
    """
    Gestionnaire pour données quantitatives structurées
    Dual-path: requête SQL/DataFrame + textification pour contexte
    """

    def __init__(self, quant_data_path: Optional[str] = None):
        """
        Args:
            quant_data_path: Chemin vers les données quantitatives (CSV/Excel)
        """
        self.df = None
        if quant_data_path and os.path.exists(quant_data_path):
            self.df = pd.read_csv(quant_data_path)

    def query_structured_data(self, commune: Optional[str] = None,
                             indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Requête structurée sur les données quantitatives

        Args:
            commune: Nom de la commune (filtre optionnel)
            indicators: Liste d'indicateurs à récupérer

        Returns:
            DataFrame avec les résultats
        """
        if self.df is None:
            return pd.DataFrame()

        result = self.df.copy()

        # Filtrer par commune
        if commune and 'commune' in result.columns:
            result = result[result['commune'].str.lower() == commune.lower()]

        # Filtrer par indicateurs
        if indicators:
            # Supposer que les indicateurs sont des colonnes
            available_cols = [col for col in indicators if col in result.columns]
            if available_cols:
                result = result[['commune'] + available_cols]

        return result

    def format_as_table(self, df: pd.DataFrame) -> str:
        """
        Formate un DataFrame en tableau markdown pour le contexte LLM
        """
        if df.empty:
            return "Aucune donnée quantitative disponible."

        return df.to_markdown(index=False)

    def extract_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Extrait des statistiques descriptives
        """
        if df.empty:
            return {}

        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'std': df[col].std()
            }

        return stats


class ImprovedPromptBuilder:
    """
    Constructeur de prompts amélioré pour entretiens semi-directifs
    """

    SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de données sur la qualité de vie en Corse.

Tu as accès à:
1. Des extraits d'entretiens avec des habitants
2. Des données quantitatives sur des indicateurs de qualité de vie
3. Des informations contextuelles (Wikipedia) sur les communes

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
                        statistics: Optional[Dict] = None) -> str:
        """
        Construit un prompt RAG complet

        Args:
            question: Question de l'utilisateur
            qualitative_context: Résultats de retrieval (entretiens)
            quantitative_data: DataFrame avec données quantitatives
            statistics: Statistiques descriptives

        Returns:
            Prompt formaté
        """
        prompt_parts = []

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
                prompt_parts.append(f"Entretien n°{metadata['num_entretien']}")

            prompt_parts.append(f"Pertinence: {result.score:.3f}")
            prompt_parts.append(f"\nContenu:\n{result.text}\n")
            prompt_parts.append("-" * 80)

        # 2. Contexte quantitatif (si disponible)
        if quantitative_data is not None and not quantitative_data.empty:
            prompt_parts.append("\n\n=== DONNÉES QUANTITATIVES ===\n")

            quant_handler = QuantitativeDataHandler()
            table = quant_handler.format_as_table(quantitative_data)
            prompt_parts.append(table)

        # 3. Statistiques descriptives
        if statistics:
            prompt_parts.append("\n\n=== STATISTIQUES DESCRIPTIVES ===\n")
            for indicator, stats in statistics.items():
                prompt_parts.append(f"\n{indicator}:")
                prompt_parts.append(f"  - Moyenne: {stats['mean']:.2f}")
                prompt_parts.append(f"  - Médiane: {stats['median']:.2f}")
                prompt_parts.append(f"  - Min-Max: {stats['min']:.2f} - {stats['max']:.2f}")
                prompt_parts.append(f"  - Écart-type: {stats['std']:.2f}")

        # 4. Question
        prompt_parts.append(f"\n\n=== QUESTION ===\n{question}")

        # 5. Instructions
        prompt_parts.append("\n\n=== INSTRUCTIONS ===")
        prompt_parts.append("Réponds à la question de manière claire et concise en te basant sur les informations ci-dessus.")
        prompt_parts.append("Cite tes sources quand c'est pertinent (commune, n° entretien, données quantitatives).")
        prompt_parts.append("Si les informations sont insuffisantes, indique-le clairement.")

        return "\n".join(prompt_parts)


class ImprovedRAGPipeline:
    """
    Pipeline RAG amélioré - Version 2
    """

    def __init__(self,
                 chroma_path: str = "./chroma_v2",
                 collection_name: str = "communes_corses_v2",
                 embedding_model: str = "dangvantuan/sentence-camembert-large",
                 reranker_model: str = "antoinelouis/crossencoder-camembert-base-mmarcoFR",
                 llm_model: str = "gpt-3.5-turbo",
                 openai_api_key: Optional[str] = None,
                 quant_data_path: str = "df_mean_by_commune.csv"):
        """
        Initialise le pipeline RAG amélioré
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

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )

        # Documents pour BM25 (chargés lors de l'indexation)
        self.documents = []
        self.hybrid_retriever = None

        # Charger automatiquement le cache s'il existe
        self._load_cache_if_exists()

    def _load_cache_if_exists(self):
        """
        Charge automatiquement le cache d'embeddings s'il existe
        """
        cache_path = "embeddings_v2.pkl"

        if not os.path.exists(cache_path):
            print("Aucun cache trouvé. Vous devrez appeler ingest_documents() pour indexer les documents.")
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
                    self.documents,
                    dense_weight=0.6,
                    sparse_weight=0.4
                )
                print("OK Retriever hybride initialise")
            else:
                print("AVERTISSEMENT Cache vide, aucun document trouve")

        except Exception as e:
            print(f"AVERTISSEMENT Erreur lors du chargement du cache: {e}")
            print("Vous devrez appeler ingest_documents() pour indexer les documents.")

    def ingest_documents(self, texts: List[str], metadatas: List[Dict],
                        use_qa_chunking: bool = False, save_cache: bool = True):
        """
        Ingère des documents dans le pipeline

        Args:
            texts: Liste de textes à ingérer
            metadatas: Métadonnées associées à chaque texte
            use_qa_chunking: Utiliser le chunking Q/R pour entretiens
            save_cache: Sauvegarder les embeddings en cache
        """
        print(f"Ingestion de {len(texts)} documents...")

        all_documents = []

        # 1. Chunking
        print("Chunking des documents...")
        for text, metadata in tqdm(zip(texts, metadatas), total=len(texts)):
            if use_qa_chunking and metadata.get('source') == 'entretien':
                chunks = self.chunker.chunk_interview_qa(text, metadata)
            else:
                chunks = self.chunker.chunk_text(text, metadata)

            all_documents.extend(chunks)

        print(f"Total de chunks créés: {len(all_documents)}")

        # 2. Génération des embeddings
        print("Génération des embeddings...")
        texts_to_embed = [doc.page_content for doc in all_documents]
        embeddings = self.embed_model.encode_documents(texts_to_embed)

        # 3. Préparation pour ChromaDB
        ids = [f"{doc.metadata.get('source', 'doc')}_{doc.metadata.get('nom', 'unknown')}_{i}"
               for i, doc in enumerate(all_documents)]

        # Ajouter les IDs aux métadonnées
        for doc, doc_id in zip(all_documents, ids):
            doc.metadata['id'] = doc_id

        # 4. Indexation dans ChromaDB
        print("Indexation dans ChromaDB...")
        batch_size = 5000
        for i in tqdm(range(0, len(all_documents), batch_size)):
            batch_docs = all_documents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            self.collection.add(
                documents=[doc.page_content for doc in batch_docs],
                embeddings=batch_embeddings.tolist(),
                metadatas=[doc.metadata for doc in batch_docs],
                ids=batch_ids
            )

        # 5. Stocker les documents pour BM25
        self.documents = all_documents

        # 6. Initialiser le hybrid retriever
        print("Initialisation du retriever hybride...")
        self.hybrid_retriever = HybridRetriever(
            self.collection,
            self.documents,
            dense_weight=0.6,
            sparse_weight=0.4
        )

        # 7. Cache (optionnel)
        if save_cache:
            cache_path = "embeddings_v2.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'documents': all_documents,
                    'ids': ids
                }, f)
            print(f"Cache sauvegardé: {cache_path}")

        print("Ingestion terminée!")

    def query(self, question: str,
             k: int = 5,
             use_reranking: bool = True,
             include_quantitative: bool = True,
             commune_filter: Optional[str] = None) -> Tuple[str, List[RetrievalResult]]:
        """
        Effectue une requête RAG complète

        Args:
            question: Question de l'utilisateur
            k: Nombre de résultats à retourner
            use_reranking: Utiliser le reranking
            include_quantitative: Inclure les données quantitatives
            commune_filter: Filtrer par commune (optionnel)

        Returns:
            (réponse, résultats_retrieval)
        """
        if self.hybrid_retriever is None:
            raise ValueError("Le pipeline n'a pas été initialisé avec des documents. Appelez ingest_documents() d'abord.")

        # 1. Encoder la requête
        query_embedding = self.embed_model.encode_query(question)

        # 2. Retrieval hybride
        print("Retrieval hybride...")
        results = self.hybrid_retriever.retrieve(
            question,
            query_embedding,
            k=k*2  # Récupérer plus pour le reranking
        )

        # 3. Reranking (optionnel)
        if use_reranking:
            print("Reranking...")
            results = self.reranker.rerank(question, results, top_k=k)
        else:
            results = results[:k]

        # 4. Récupérer les données quantitatives (optionnel)
        quant_data = None
        stats = None
        if include_quantitative:
            # Extraire la commune de la question ou des résultats
            commune = commune_filter
            if not commune and results:
                commune = results[0].metadata.get('nom') or results[0].metadata.get('commune')

            if commune:
                quant_data = self.quant_handler.query_structured_data(commune=commune)
                if not quant_data.empty:
                    stats = self.quant_handler.extract_statistics(quant_data)

        # 5. Construire le prompt
        prompt = ImprovedPromptBuilder.build_rag_prompt(
            question,
            results,
            quant_data,
            stats
        )

        # 6. Générer la réponse avec LLM
        print("Génération de la réponse...")
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
                    {"role": "system", "content": ImprovedPromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Erreur lors de la génération: {str(e)}"


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_interview_data(directory_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Charge les données d'entretiens depuis un répertoire

    Args:
        directory_path: Chemin vers le répertoire contenant les fichiers d'entretiens

    Returns:
        (texts, metadatas)
    """
    texts = []
    metadatas = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory_path, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Extraire les métadonnées du nom de fichier
            # Format attendu: entretien_commune_num.txt
            parts = filename.replace('.txt', '').split('_')

            metadata = {
                'source': 'entretien',
                'filename': filename
            }

            if len(parts) >= 2:
                metadata['commune'] = parts[1]
            if len(parts) >= 3:
                metadata['num_entretien'] = parts[2]

            texts.append(text)
            metadatas.append(metadata)

    return texts, metadatas


def compare_with_baseline(question: str,
                         baseline_results: List[str],
                         improved_results: List[RetrievalResult]):
    """
    Compare les résultats du RAG amélioré avec le baseline
    """
    print("\n" + "="*80)
    print("COMPARAISON BASELINE vs IMPROVED")
    print("="*80)

    print(f"\nQuestion: {question}\n")

    print("-" * 40)
    print("BASELINE (RAG v1):")
    print("-" * 40)
    for i, result in enumerate(baseline_results[:3], 1):
        print(f"\n[{i}] {result[:200]}...")

    print("\n" + "-" * 40)
    print("IMPROVED (RAG v2):")
    print("-" * 40)
    for i, result in enumerate(improved_results[:3], 1):
        print(f"\n[{i}] Score: {result.score:.3f} | Source: {result.source_type}")
        print(f"Commune: {result.metadata.get('nom', result.metadata.get('commune', 'N/A'))}")
        print(f"Texte: {result.text[:200]}...")

    print("\n" + "="*80)


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation du pipeline amélioré
    """

    # Configuration
    import os
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("❌ Clé API OpenAI non trouvée. Définissez la variable d'environnement OPENAI_API_KEY")

    # Initialiser le pipeline
    print("Initialisation du pipeline RAG amélioré...")
    rag = ImprovedRAGPipeline(
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        embedding_model="dangvantuan/sentence-camembert-large",  # Français optimisé
        reranker_model="antoinelouis/crossencoder-camembert-base-mmarcoFR",
        llm_model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )

    # Charger et ingérer les données
    # Option 1: Depuis un répertoire d'entretiens
    # texts, metadatas = load_interview_data("./entretiens")

    # Option 2: Données de démonstration
    demo_texts = [
        """Q: Comment évaluez-vous la qualité de vie à Ajaccio?
        R: Globalement, je trouve qu'Ajaccio offre une bonne qualité de vie. Le cadre est magnifique,
        la mer est proche, et il y a une vraie douceur de vivre. Par contre, les transports en commun
        sont vraiment insuffisants, surtout en été avec les touristes.""",

        """Q: Quels sont les principaux problèmes que vous rencontrez au quotidien?
        R: Le problème principal c'est le logement. Les prix sont devenus prohibitifs, surtout pour
        les jeunes. Il y a aussi un manque de structures pour les enfants, peu de crèches disponibles.""",

        """Q: Comment percevez-vous l'accès aux soins à Bastia?
        R: L'accès aux soins est correct en ville, on a un bon hôpital. Mais dès qu'on s'éloigne un peu,
        ça devient compliqué. Les délais pour voir un spécialiste sont très longs."""
    ]

    demo_metadatas = [
        {'source': 'entretien', 'nom': 'Ajaccio', 'num_entretien': '1'},
        {'source': 'entretien', 'nom': 'Ajaccio', 'num_entretien': '2'},
        {'source': 'entretien', 'nom': 'Bastia', 'num_entretien': '1'}
    ]

    # Ingestion
    print("\n" + "="*80)
    print("INGESTION DES DONNÉES")
    print("="*80)
    rag.ingest_documents(
        demo_texts,
        demo_metadatas,
        use_qa_chunking=True,
        save_cache=True
    )

    # Requêtes de test
    test_questions = [
        "Quels sont les problèmes de transport à Ajaccio?",
        "Comment est l'accès aux soins en Corse?",
        "Quelles sont les difficultés liées au logement?"
    ]

    print("\n" + "="*80)
    print("REQUÊTES DE TEST")
    print("="*80)

    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print('='*80)

        response, results = rag.query(
            question,
            k=3,
            use_reranking=True,
            include_quantitative=False  # Pas de données quant dans cet exemple
        )

        print("\n--- RÉSULTATS DE RETRIEVAL ---")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result.score:.3f} | Type: {result.source_type}")
            print(f"Commune: {result.metadata.get('nom', 'N/A')}")
            print(f"Texte: {result.text[:150]}...")

        print("\n--- RÉPONSE GÉNÉRÉE ---")
        print(response)
        print("\n")
