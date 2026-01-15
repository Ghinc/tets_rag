"""
Script pour importer toutes les communes depuis communes_chatbot/ dans ChromaDB
"""
import os
from rag_v2_improved import ImprovedRAGPipeline

def load_communes_from_directory(directory_path: str):
    """Charge tous les fichiers .txt des communes"""
    texts = []
    metadatas = []

    print(f"Chargement des communes depuis {directory_path}...")

    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory_path, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Extraire le nom de la commune depuis le fichier
            commune_name = filename.replace('.txt', '')

            metadata = {
                'source': 'fiche_commune',
                'nom': commune_name,
                'type': 'scores_detailles',
                'filename': filename
            }

            texts.append(text)
            metadatas.append(metadata)
            print(f"  [OK] {commune_name}")

    print(f"\nTotal: {len(texts)} communes chargées")
    return texts, metadatas


if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY manquante")

    # Initialiser le pipeline
    print("="*80)
    print("IMPORT DES COMMUNES DANS CHROMADB")
    print("="*80)

    rag = ImprovedRAGPipeline(
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        llm_model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )

    # Charger toutes les communes
    texts, metadatas = load_communes_from_directory("./communes_chatbot")

    # Ingérer dans ChromaDB
    print(f"\n{'='*80}")
    print("INGESTION DANS CHROMADB")
    print(f"{'='*80}\n")

    rag.ingest_documents(texts, metadatas)

    print(f"\n{'='*80}")
    print("IMPORT TERMINE !")
    print(f"{'='*80}")
    print(f"[OK] {len(texts)} communes importees avec succes")
    print(f"[OK] Collection: communes_corses_v2")
    print(f"[OK] Modele d'embedding: BAAI/bge-m3 (1024 dimensions)")
