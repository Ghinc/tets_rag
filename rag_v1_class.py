"""
RAG v1 - Version avec classe (wrapper)

Ce module encapsule la logique de rag_v1_2904.py dans une classe
pour permettre son utilisation avec l'API, sans modifier la logique originale.

La logique RAG reste identique à v1 : simple retrieval + génération LLM.
"""

import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


@dataclass
class RetrievalResult:
    """Résultat de retrieval simple"""
    text: str
    metadata: Dict
    score: float
    source_type: str = 'vector'  # Toujours 'vector' pour v1


class BasicRAGPipeline:
    """
    Pipeline RAG v1 - Version basique

    Fonctionnalités :
    - Simple retrieval vectoriel (pas de BM25, pas de reranking)
    - Modèle d'embeddings : intfloat/e5-base-v2
    - Génération avec OpenAI GPT

    Cette version est identique à rag_v1_2904.py mais encapsulée dans une classe.
    """

    def __init__(
        self,
        openai_api_key: str,
        chroma_path: str = "./chroma_txt/",
        collection_name: str = "communes_corses_txt",
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "intfloat/e5-base-v2"
    ):
        """
        Initialise le pipeline RAG v1

        Args:
            openai_api_key: Clé API OpenAI
            chroma_path: Chemin vers la base ChromaDB
            collection_name: Nom de la collection ChromaDB
            llm_model: Modèle LLM à utiliser
            embedding_model: Modèle d'embeddings (par défaut e5-base-v2)
        """
        print("\n" + "="*60)
        print("INITIALISATION RAG v1 (BASIQUE)")
        print("="*60)

        # Stocker les paramètres
        self.openai_api_key = openai_api_key
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model

        # Initialiser OpenAI
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Charger le modèle d'embeddings
        print(f"\nChargement du modèle d'embeddings: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model)
        print("OK Modèle d'embeddings chargé")

        # Connexion à ChromaDB
        print(f"\nConnexion à ChromaDB: {chroma_path}")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            doc_count = self.collection.count()
            print(f"OK Collection '{collection_name}' chargée ({doc_count} documents)")
        except Exception as e:
            raise RuntimeError(
                f"Impossible de charger la collection '{collection_name}'. "
                f"Assurez-vous qu'elle a été créée avec rag_v1_2904.py. Erreur: {e}"
            )

        print("\n" + "="*60)
        print("OK RAG v1 INITIALISE")
        print("="*60 + "\n")

    def query(
        self,
        question: str,
        k: int = 5,
        **kwargs  # Accepte les autres paramètres pour compatibilité API mais les ignore
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Effectue une requête RAG v1 simple

        Args:
            question: Question de l'utilisateur
            k: Nombre de résultats à retourner
            **kwargs: Paramètres additionnels (ignorés pour v1)

        Returns:
            Tuple (réponse_générée, liste_de_sources)
        """
        # 1. Générer l'embedding de la question
        # Note: e5-base-v2 nécessite le préfixe "query:"
        query_text = f"query: {question}"
        query_embedding = self.embed_model.encode([query_text]).tolist()

        # 2. Recherche vectorielle dans ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )

        # 3. Extraire les documents et métadonnées
        retrieved_docs = results["documents"][0]
        retrieved_metadatas = results.get("metadatas", [[{}] * len(retrieved_docs)])[0]
        retrieved_distances = results.get("distances", [[0.0] * len(retrieved_docs)])[0]

        # Construire le contexte pour le prompt
        retrieved_context = "\n\n".join(retrieved_docs)

        # 4. Générer le prompt (identique à v1)
        prompt = f"""Tu es un conseiller municipal. Ton but est de donner des informations sur la qualité de vie dans les communes Corses, pour guider les politiques publiques, en te basant uniquement sur les informations suivantes :

{retrieved_context}

Question : {question}
Réponse :
"""

        # 5. Appeler OpenAI pour générer la réponse
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant utile et factuel. Ne réponds qu'avec les informations données."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )

            answer = response.choices[0].message.content

        except Exception as e:
            answer = f"Erreur lors de la génération : {e}"

        # 6. Construire les résultats de retrieval
        retrieval_results = [
            RetrievalResult(
                text=doc,
                metadata=meta if meta else {},
                score=1.0 / (1.0 + distance),  # Convertir distance en score de similarité
                source_type='vector'
            )
            for doc, meta, distance in zip(retrieved_docs, retrieved_metadatas, retrieved_distances)
        ]

        # 7. Appliquer le malus Wikipedia
        WIKI_PENALTY = 0.3  # Diviser par 3.33 le score des sources Wikipedia
        for result in retrieval_results:
            if result.metadata.get('source') == 'wiki':
                result.score *= WIKI_PENALTY

        return answer, retrieval_results

    def ingest_documents(self, texts: List[str], metadatas: List[Dict], **kwargs):
        """
        Méthode d'ingestion (pour compatibilité API)

        Note: Pour v1, l'indexation se fait normalement via rag_v1_2904.py
        Cette méthode est fournie pour compatibilité mais non recommandée.
        """
        raise NotImplementedError(
            "L'ingestion pour RAG v1 doit se faire via rag_v1_2904.py. "
            "Utilisez les fonctions natives du script original."
        )


# ============================================================================
# FONCTION DE COMPATIBILITÉ AVEC LE SCRIPT ORIGINAL
# ============================================================================

def ask_rag_with_llm_v1(
    collection,
    question: str,
    openai_api_key: str,
    n_chunks: int = 5,
    model_name: str = "gpt-3.5-turbo"
) -> str:
    """
    Fonction wrapper pour compatibilité avec rag_v1_2904.py

    Cette fonction reproduit exactement le comportement de ask_rag_with_llm()
    du script original.

    Args:
        collection: Collection ChromaDB
        question: Question à poser
        openai_api_key: Clé API OpenAI
        n_chunks: Nombre de chunks à récupérer
        model_name: Modèle OpenAI à utiliser

    Returns:
        Réponse générée par le LLM
    """
    # Charger le modèle d'embeddings
    embed_model = SentenceTransformer("intfloat/e5-base-v2")
    query_embedding = embed_model.encode([f"query: {question}"]).tolist()

    # Recherche dans ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_chunks
    )

    retrieved_docs = results["documents"][0]
    retrieved_context = "\n\n".join(retrieved_docs)

    # Construire le prompt
    prompt = f"""Tu es un conseiller municipal. Ton but est de donner des informations sur la qualité de vie dans les communes Corses, pour guider les politiques publiques, en te basant uniquement sur les informations suivantes :

{retrieved_context}

Question : {question}
Réponse :
"""

    # Appeler OpenAI
    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Tu es un assistant utile et factuel. Ne réponds qu'avec les informations données."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content
    return answer


if __name__ == "__main__":
    """
    Test du pipeline RAG v1 avec classe
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Récupérer la clé API
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY non trouvée dans .env")
        exit(1)

    # Initialiser le pipeline
    rag = BasicRAGPipeline(
        openai_api_key=openai_api_key,
        chroma_path="./chroma_txt/",
        collection_name="communes_corses_txt"
    )

    # Tester avec une question
    question = "Quelles sont les communes avec le meilleur bien-être ?"
    print(f"\nQuestion : {question}\n")

    answer, sources = rag.query(question, k=5)

    print("="*60)
    print("REPONSE")
    print("="*60)
    print(answer)

    print("\n" + "="*60)
    print(f"SOURCES ({len(sources)})")
    print("="*60)
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}] Score: {source.score:.3f}")
        print(f"Texte: {source.text[:200]}...")
