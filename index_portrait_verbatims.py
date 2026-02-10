"""
Script d'indexation des verbatims avec métadonnées portrait dans ChromaDB
Ajoute les 690 verbatims avec: genre, age, profession, dimension
"""
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import os

# Configuration
CHROMA_PATH = "./chroma_portrait"  # Base séparée pour les verbatims portrait
COLLECTION_NAME = "portrait_verbatims"
EMBEDDING_MODEL = "BAAI/bge-m3"
CSV_PATH = "donnees_brutes/verbatims_par_commune_avec_portrait/_tous_verbatims_avec_portrait.csv"

def age_to_int(age_value):
    """Convertit l'âge en entier"""
    if pd.isna(age_value):
        return None
    age_str = str(age_value).strip()
    if age_str.lower() == "plus de 70 ans":
        return 71
    try:
        return int(float(age_str))
    except ValueError:
        return None

def age_to_range(age: int) -> str:
    """Convertit l'âge en catégorie"""
    if age is None:
        return "Non spécifié"
    if age < 25:
        return "15-24"  # Jeunes
    elif age < 35:
        return "25-34"  # Jeunes adultes
    elif age < 50:
        return "35-49"  # Adultes
    elif age < 65:
        return "50-64"  # Jeunes seniors
    else:
        return "65+"    # Seniors

def age_range_label(age_range: str) -> str:
    """Retourne le label humain de la tranche d'âge"""
    labels = {
        "15-24": "Jeunes",
        "25-34": "Jeunes adultes",
        "35-49": "Adultes",
        "50-64": "Jeunes seniors",
        "65+": "Seniors"
    }
    return labels.get(age_range, "Non spécifié")

def main():
    print("=" * 60)
    print("INDEXATION DES VERBATIMS PORTRAIT")
    print("=" * 60)

    # 1. Charger les données
    print(f"\n1. Chargement des données depuis {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    print(f"   {len(df)} verbatims chargés")

    # Afficher les statistiques
    print(f"\n   Statistiques:")
    print(f"   - Communes: {df['commune'].nunique()}")
    print(f"   - Genres: {df['genre'].unique().tolist()}")
    print(f"   - Professions: {df['profession'].nunique()}")
    print(f"   - Dimensions: {df['dimension'].nunique()}")

    # 2. Charger le modèle d'embeddings
    print(f"\n2. Chargement du modèle d'embeddings: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("   OK Modèle chargé")

    # 3. Préparer les documents et métadonnées
    print("\n3. Préparation des documents...")
    documents = []
    metadatas = []
    ids = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Traitement"):
        verbatim = str(row['verbatim']).strip()
        if not verbatim or verbatim.lower() in ['nan', 'none', '']:
            continue

        # Convertir l'âge
        age_int = age_to_int(row['age'])
        age_cat = age_to_range(age_int)

        # Construire le document avec contexte
        doc_text = f"Verbatim d'un(e) {row['genre']} de {age_int} ans ({age_range_label(age_cat)}), {row['profession']}, à {row['commune']}, sur le thème '{row['dimension']}': {verbatim}"

        # Métadonnées pour ChromaDB
        metadata = {
            'source': 'portrait_verbatim',
            'nom': str(row['commune']),
            'genre': str(row['genre']) if pd.notna(row['genre']) else 'Non spécifié',
            'age_exact': age_int if age_int is not None else -1,  # ChromaDB n'accepte pas None
            'age_range': age_cat,
            'profession': str(row['profession']) if pd.notna(row['profession']) else 'Non spécifié',
            'dimension': str(row['dimension']) if pd.notna(row['dimension']) else 'Non spécifié',
            'matching_certain': bool(row['matching_certain']) if pd.notna(row.get('matching_certain')) else True,
            'choix_numero': int(row['choix_numero']) if pd.notna(row.get('choix_numero')) else 0,
        }

        doc_id = f"portrait_{row['commune']}_{idx}"

        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(doc_id)

    print(f"   {len(documents)} documents préparés")

    # 4. Générer les embeddings
    print("\n4. Génération des embeddings...")
    # Préfixer pour BGE-M3
    texts_to_embed = [f"passage: {doc}" for doc in documents]
    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True)
    print(f"   Shape: {embeddings.shape}")

    # 5. Connexion à ChromaDB
    print(f"\n5. Connexion à ChromaDB: {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Créer ou récupérer la collection portrait
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    existing_count = collection.count()
    print(f"   Collection '{COLLECTION_NAME}' prête ({existing_count} documents existants)")

    # 6. Supprimer les anciens verbatims portrait (si existants)
    if existing_count > 0:
        print("\n6. Nettoyage des anciens verbatims portrait...")
        try:
            existing_ids = collection.get(include=[])['ids']
            if existing_ids:
                print(f"   Suppression de {len(existing_ids)} anciens verbatims...")
                collection.delete(ids=existing_ids)
                print("   OK Anciens verbatims supprimés")
        except Exception as e:
            print(f"   Note: {e}")
    else:
        print("\n6. Pas d'anciens verbatims à supprimer")

    # 7. Ajouter les nouveaux documents
    print(f"\n7. Ajout de {len(documents)} verbatims portrait...")

    # Ajouter par batch
    batch_size = 500
    for i in tqdm(range(0, len(documents), batch_size), desc="   Indexation"):
        batch_docs = documents[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size].tolist()
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    # 8. Vérification
    print("\n8. Vérification...")
    final_count = collection.count()
    portrait_results = collection.get(
        where={"source": "portrait_verbatim"},
        include=["metadatas"]
    )
    portrait_count = len(portrait_results['ids'])

    print(f"   Total documents dans la collection: {final_count}")
    print(f"   Dont verbatims portrait: {portrait_count}")

    # Statistiques des métadonnées
    if portrait_results['metadatas']:
        genres = set(m.get('genre') for m in portrait_results['metadatas'])
        age_ranges = set(m.get('age_range') for m in portrait_results['metadatas'])
        professions = set(m.get('profession') for m in portrait_results['metadatas'])
        dimensions = set(m.get('dimension') for m in portrait_results['metadatas'])

        print(f"\n   Répartition des métadonnées:")
        print(f"   - Genres: {genres}")
        print(f"   - Tranches d'âge: {age_ranges}")
        print(f"   - Professions: {len(professions)} catégories")
        print(f"   - Dimensions: {len(dimensions)} catégories")

    print("\n" + "=" * 60)
    print("INDEXATION TERMINÉE")
    print("=" * 60)

if __name__ == "__main__":
    main()
