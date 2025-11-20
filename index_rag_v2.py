"""
Script pour indexer toutes les données réelles dans le RAG v2

Indexe :
- Données quantitatives (rag_quanti/)
- Données bien-être + verbatims (rag_be/)
- Données Wikipedia (communes_corses_wiki.txt)
- Données CSV texte (communes_text.txt)
- Entretiens (si disponibles)
"""

import os
import glob
import re
from typing import List, Dict
from dotenv import load_dotenv
from rag_v2_boosted import ImprovedRAGPipeline

# Configuration
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_rag_folder(folder_path: str, source_type: str) -> tuple[List[str], List[Dict]]:
    """
    Charge tous les fichiers .txt d'un dossier

    Args:
        folder_path: Chemin du dossier
        source_type: Type de source (quanti, be_verbatims, etc.)

    Returns:
        (texts, metadatas)
    """
    texts = []
    metadatas = []

    if not os.path.exists(folder_path):
        print(f"  Dossier {folder_path} introuvable")
        return texts, metadatas

    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    print(f"  Trouvé {len(txt_files)} fichiers dans {folder_path}")

    for filepath in txt_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        if not text:
            continue

        # Extraire le nom de la commune du nom de fichier
        filename = os.path.basename(filepath)
        commune_name = filename.replace(".txt", "").replace("_", " ").replace("-", " ")

        texts.append(text)
        metadatas.append({
            'source': source_type,
            'nom': commune_name,
            'filename': filename
        })

    return texts, metadatas


def load_wiki_data(filepath: str) -> tuple[List[str], List[Dict]]:
    """
    Charge les données Wikipedia

    Args:
        filepath: Chemin du fichier wiki

    Returns:
        (texts, metadatas)
    """
    texts = []
    metadatas = []

    if not os.path.exists(filepath):
        print(f"  Fichier {filepath} introuvable")
        return texts, metadatas

    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Découper par blocs
    blocs = re.split(r"\n---+\n", raw_text.strip())
    print(f"  Trouvé {len(blocs)} blocs Wikipedia")

    for i, bloc in enumerate(blocs):
        match = re.search(r"### (.+?)\n\nRésumé : (.+?)\n\nDescription : (.+)", bloc, re.DOTALL)
        if match:
            nom, resume, description = match.groups()
            full_text = f"{resume.strip()}\n\n{description.strip()}"

            texts.append(full_text)
            metadatas.append({
                'source': 'wiki',
                'nom': nom.strip(),
                'résumé': resume.strip()
            })

    return texts, metadatas


def load_csv_text(filepath: str) -> tuple[List[str], List[Dict]]:
    """
    Charge les données CSV converties en texte

    Args:
        filepath: Chemin du fichier texte

    Returns:
        (texts, metadatas)
    """
    texts = []
    metadatas = []

    if not os.path.exists(filepath):
        print(f"  Fichier {filepath} introuvable")
        return texts, metadatas

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  Trouvé {len(lines)} lignes de données CSV")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Extraire le nom de la commune
        match = re.match(r"La commune de ([^ ]+)", line)
        commune_name = match.group(1) if match else f"commune_{i}"

        texts.append(line)
        metadatas.append({
            'source': 'csv_text',
            'nom': commune_name
        })

    return texts, metadatas


def main():
    """
    Fonction principale d'indexation
    """
    print("="*80)
    print("INDEXATION DES DONNÉES RÉELLES DANS RAG V2")
    print("="*80)

    all_texts = []
    all_metadatas = []

    # 1. Charger les données quantitatives
    print("\n[1] Chargement des données quantitatives (rag_quanti/)...")
    texts, metas = load_rag_folder("rag_quanti", "quanti")
    all_texts.extend(texts)
    all_metadatas.extend(metas)
    print(f"  OK {len(texts)} documents chargés")

    # 2. Charger les données bien-être + verbatims
    print("\n[2] Chargement des données bien-être (rag_be/)...")
    texts, metas = load_rag_folder("rag_be", "be_verbatims")
    all_texts.extend(texts)
    all_metadatas.extend(metas)
    print(f"  OK {len(texts)} documents chargés")

    # 3. Charger les données Wikipedia
    print("\n[3] Chargement des données Wikipedia...")
    texts, metas = load_wiki_data("communes_corses_wiki.txt")
    all_texts.extend(texts)
    all_metadatas.extend(metas)
    print(f"  OK {len(texts)} documents chargés")

    # 4. Charger les données CSV texte
    print("\n[4] Chargement des données CSV texte...")
    texts, metas = load_csv_text("communes_text.txt")
    all_texts.extend(texts)
    all_metadatas.extend(metas)
    print(f"  OK {len(texts)} documents chargés")

    # 5. Charger les entretiens (si disponibles)
    print("\n[5] Chargement des entretiens...")
    entretien_files = glob.glob("entretiens*.txt")
    if entretien_files:
        for filepath in entretien_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if text:
                all_texts.append(text)
                all_metadatas.append({
                    'source': 'entretien',
                    'filename': os.path.basename(filepath)
                })
        print(f"  OK {len(entretien_files)} fichiers d'entretiens chargés")
    else:
        print("  Aucun fichier d'entretien trouvé")

    # 6. Charger les données OppChoVec
    print("\n[6] Chargement des données OppChoVec (communes_chatbot/)...")
    texts, metas = load_rag_folder("communes_chatbot", "oppchovec")
    all_texts.extend(texts)
    all_metadatas.extend(metas)
    print(f"  OK {len(texts)} documents chargés")

    # Afficher le résumé
    print(f"\n{'='*80}")
    print(f"TOTAL: {len(all_texts)} documents à indexer")
    print(f"{'='*80}")

    # Sources
    sources_count = {}
    for meta in all_metadatas:
        source = meta.get('source', 'unknown')
        sources_count[source] = sources_count.get(source, 0) + 1

    print("\nRépartition par source:")
    for source, count in sorted(sources_count.items()):
        print(f"  - {source}: {count} documents")

    # Initialiser le pipeline RAG v2
    print(f"\n{'='*80}")
    print("INITIALISATION DU PIPELINE RAG V2")
    print(f"{'='*80}")

    rag = ImprovedRAGPipeline(
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        embedding_model="intfloat/e5-base-v2",  # Modèle plus rapide
        reranker_model="antoinelouis/crossencoder-camembert-base-mmarcoFR",
        llm_model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )

    # Indexer les documents
    print(f"\n{'='*80}")
    print("INDEXATION")
    print(f"{'='*80}")

    rag.ingest_documents(
        all_texts,
        all_metadatas,
        use_qa_chunking=False,  # Pas de chunking Q/R spécial
        save_cache=True
    )

    print(f"\n{'='*80}")
    print("INDEXATION TERMINÉE")
    print(f"{'='*80}")
    print("\nLe fichier embeddings_v2.pkl a été créé avec toutes les données réelles.")
    print("Vous pouvez maintenant utiliser le RAG v2 avec toutes les données des communes corses.")

    # Test rapide
    print(f"\n{'='*80}")
    print("TEST RAPIDE")
    print(f"{'='*80}")

    test_question = "Quelles sont les dimensions du bien-être à Grossetto Prugna ?"
    print(f"\nQuestion: {test_question}")

    response, results = rag.query(test_question, k=3, use_reranking=True, include_quantitative=True)

    print(f"\nNombre de chunks récupérés: {len(results)}")
    print(f"\nPremier chunk:")
    if results:
        print(f"  Commune: {results[0].metadata.get('nom', 'N/A')}")
        print(f"  Source: {results[0].metadata.get('source', 'N/A')}")
        print(f"  Texte: {results[0].text[:200]}...")

    print(f"\nRéponse:")
    print(response[:500] + "..." if len(response) > 500 else response)


if __name__ == "__main__":
    main()
