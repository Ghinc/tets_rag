"""
Script de réindexation ChromaDB avec enrichissement des métadonnées ontologie.

Ce script:
1. Charge tous les documents de rag_be/
2. Enrichit les métadonnées avec les identifiants de l'ontologie (source_ontology_mapping.json)
3. Réindexe dans ChromaDB (nouvelle collection ou remplacement)

Usage:
    python reindex_with_ontology.py
"""

import os
import json
import shutil
from typing import List, Dict, Tuple
from datetime import datetime

# Importer les fonctions du RAG v2
from rag_v2_improved import (
    ImprovedRAGPipeline,
    load_ontology_mapping,
    enrich_all_metadatas
)

from dotenv import load_dotenv
load_dotenv()


def load_rag_be_data(rag_be_path: str = "rag_be") -> Tuple[List[str], List[Dict]]:
    """
    Charge tous les fichiers texte du dossier rag_be.

    Args:
        rag_be_path: Chemin vers le dossier rag_be

    Returns:
        (texts, metadatas) - listes de textes et métadonnées
    """
    texts = []
    metadatas = []

    if not os.path.exists(rag_be_path):
        raise FileNotFoundError(f"Dossier non trouvé: {rag_be_path}")

    files = sorted([f for f in os.listdir(rag_be_path) if f.endswith('.txt')])
    print(f"Chargement de {len(files)} fichiers depuis {rag_be_path}/")

    for filename in files:
        filepath = os.path.join(rag_be_path, filename)
        commune_name = filename.replace('.txt', '')

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Déterminer le type de contenu
        # Les fichiers contiennent à la fois des données quantitatives et des verbatims
        metadata = {
            'source': 'enquete',  # Type principal
            'nom': commune_name,
            'filename': filename,
            'type_contenu': 'survey_with_verbatims'
        }

        texts.append(text)
        metadatas.append(metadata)

    return texts, metadatas


def load_entretiens_data(entretiens_path: str = "entretiens_lea.txt") -> Tuple[List[str], List[Dict]]:
    """
    Charge les entretiens depuis le fichier entretiens_lea.txt.

    Returns:
        (texts, metadatas) - listes de textes et métadonnées
    """
    import re

    texts = []
    metadatas = []

    if not os.path.exists(entretiens_path):
        print(f"Fichier d'entretiens non trouvé: {entretiens_path}")
        return texts, metadatas

    with open(entretiens_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern pour extraire chaque entretien
    # Format: ### Commune : X | Entretien Y
    pattern = r'### Commune : ([^|]+) \| Entretien (\d+)\s*\n(.*?)(?=### Commune|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    print(f"Chargement de {len(matches)} entretiens depuis {entretiens_path}")

    for commune_name, entretien_num, entretien_text in matches:
        commune_name = commune_name.strip()
        entretien_text = entretien_text.strip()

        if not entretien_text:
            continue

        metadata = {
            'source': 'entretien',
            'nom': commune_name,
            'commune': commune_name,
            'num_entretien': entretien_num,
            'type_contenu': 'interview'
        }

        texts.append(entretien_text)
        metadatas.append(metadata)

    return texts, metadatas


def main():
    """Fonction principale de réindexation."""
    print("=" * 70)
    print("RÉINDEXATION CHROMADB AVEC MÉTADONNÉES ONTOLOGIE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Clé API OpenAI non trouvée. Définissez OPENAI_API_KEY")

    CHROMA_PATH = "./chroma_v2"
    COLLECTION_NAME = "communes_corses_v2_onto"  # Nouvelle collection avec ontologie
    MAPPING_PATH = "source_ontology_mapping.json"

    # 1. Charger le mapping ontologie
    print("\n1. Chargement du mapping ontologie...")
    if not os.path.exists(MAPPING_PATH):
        print(f"   ERREUR: {MAPPING_PATH} non trouvé!")
        print("   Exécutez d'abord: python populate_communes.py")
        return

    ontology_mapping = load_ontology_mapping(MAPPING_PATH)
    print(f"   OK {len(ontology_mapping.get('sources', {}))} entrées de mapping chargées")

    # 2. Charger les données
    print("\n2. Chargement des données...")

    # 2.1 Données d'enquête (rag_be/)
    survey_texts, survey_metadatas = load_rag_be_data("rag_be")
    print(f"   - {len(survey_texts)} fichiers d'enquête chargés")

    # 2.2 Entretiens
    interview_texts, interview_metadatas = load_entretiens_data("entretiens_lea.txt")
    print(f"   - {len(interview_texts)} entretiens chargés")

    # Combiner
    all_texts = survey_texts + interview_texts
    all_metadatas = survey_metadatas + interview_metadatas
    print(f"   Total: {len(all_texts)} documents")

    # 3. Enrichir les métadonnées avec l'ontologie
    print("\n3. Enrichissement des métadonnées avec identifiants ontologie...")
    all_metadatas = enrich_all_metadatas(all_metadatas, ontology_mapping)

    # Vérifier l'enrichissement
    enriched_count = sum(1 for m in all_metadatas if 'ontology_source_id' in m)
    print(f"   OK {enriched_count}/{len(all_metadatas)} documents enrichis avec ID ontologie")

    # Afficher quelques exemples
    print("\n   Exemples de métadonnées enrichies:")
    for i, m in enumerate(all_metadatas[:3]):
        print(f"   [{i+1}] {m.get('nom', 'N/A')} ({m.get('source', 'N/A')})")
        print(f"       - ontology_source_id: {m.get('ontology_source_id', 'N/A')[:20]}...")
        print(f"       - ontology_source_uri: {m.get('ontology_source_uri', 'N/A')}")
        print(f"       - insee_code: {m.get('insee_code', 'N/A')}")

    # 4. Supprimer l'ancienne collection si elle existe
    print(f"\n4. Préparation de ChromaDB...")

    # Vérifier si une ancienne collection existe
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"   Suppression de l'ancienne collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        print("   OK Collection supprimée")

    # 5. Initialiser le pipeline RAG
    print("\n5. Initialisation du pipeline RAG v2...")
    rag = ImprovedRAGPipeline(
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        llm_model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY
    )

    # 6. Ingestion avec les métadonnées enrichies
    print("\n6. Ingestion des documents dans ChromaDB...")
    rag.ingest_documents(
        all_texts,
        all_metadatas,
        use_qa_chunking=True,  # Chunking Q/R pour les entretiens
        save_cache=True
    )

    # 7. Vérification
    print("\n7. Vérification de l'indexation...")
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    print(f"   OK {count} chunks indexés dans ChromaDB")

    # Vérifier que les métadonnées ontologie sont bien présentes
    sample = collection.peek(limit=3)
    print("\n   Exemple de métadonnées dans ChromaDB:")
    for i, meta in enumerate(sample['metadatas']):
        print(f"   [{i+1}] {meta.get('nom', 'N/A')}")
        print(f"       - ontology_source_id: {meta.get('ontology_source_id', 'NON PRÉSENT')[:30] if meta.get('ontology_source_id') else 'NON PRÉSENT'}...")
        print(f"       - ontology_source_uri: {meta.get('ontology_source_uri', 'NON PRÉSENT')}")

    print("\n" + "=" * 70)
    print("RÉINDEXATION TERMINÉE!")
    print("=" * 70)
    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Chemin ChromaDB: {CHROMA_PATH}")
    print(f"Chunks indexés: {count}")
    print(f"\nLes métadonnées contiennent maintenant:")
    print("  - ontology_source_id: UUID unique de la source dans l'ontologie")
    print("  - ontology_source_uri: URI pour requêtes SPARQL (ex: :source_survey_Ajaccio)")
    print("  - insee_code: Code INSEE de la commune")


if __name__ == "__main__":
    main()
