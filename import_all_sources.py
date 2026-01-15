"""
Script pour importer TOUTES les sources de données dans ChromaDB :
- Fiches de communes (déjà fait)
- Entretiens qualitatifs
- Questionnaires
- Verbatims par commune
- Données quantitatives CSV
"""
import os
import pandas as pd
from rag_v2_improved import ImprovedRAGPipeline


def load_entretiens(filepath: str = "./entretiens_lea.txt"):
    """Charge les entretiens qualitatifs"""
    print(f"\nChargement des entretiens depuis {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Diviser par entretien (chercher des séparateurs)
    # Pour l'instant, on va le chunker intelligemment
    chunks = []
    metadatas = []

    # Découper en paragraphes de ~1000 caractères
    lines = content.split('\n')
    current_chunk = []
    current_size = 0
    chunk_index = 0

    for line in lines:
        current_chunk.append(line)
        current_size += len(line)

        if current_size >= 1000:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
            metadatas.append({
                'source': 'entretien',
                'type': 'qualitatif',
                'chunk_index': chunk_index,
                'id': f'entretien_{chunk_index}'
            })
            chunk_index += 1
            current_chunk = []
            current_size = 0

    # Ajouter le dernier chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append(chunk_text)
        metadatas.append({
            'source': 'entretien',
            'type': 'qualitatif',
            'chunk_index': chunk_index,
            'id': f'entretien_{chunk_index}'
        })

    print(f"  [OK] {len(chunks)} chunks d'entretiens créés")
    return chunks, metadatas


def load_questionnaires(filepath: str = "./sortie_questionnaire_traited.csv"):
    """Charge les résultats de questionnaires"""
    print(f"\nChargement des questionnaires depuis {filepath}...")

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        df = pd.read_csv(filepath, encoding='latin1')

    chunks = []
    metadatas = []

    # Grouper par commune si possible
    if 'commune' in df.columns or 'Commune' in df.columns:
        commune_col = 'commune' if 'commune' in df.columns else 'Commune'

        for commune, group in df.groupby(commune_col):
            # Créer un résumé des réponses pour cette commune
            text = f"Questionnaire - {commune}\n\n"
            text += group.to_string(index=False)

            chunks.append(text)
            metadatas.append({
                'source': 'questionnaire',
                'type': 'quanti',
                'nom': commune,
                'id': f'questionnaire_{commune}'
            })
    else:
        # Pas de colonne commune, chunker par lignes
        chunk_size = 50
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            text = f"Questionnaire (lignes {i} à {i+len(chunk_df)})\n\n"
            text += chunk_df.to_string(index=False)

            chunks.append(text)
            metadatas.append({
                'source': 'questionnaire',
                'type': 'quanti',
                'chunk_index': i // chunk_size,
                'id': f'questionnaire_chunk_{i//chunk_size}'
            })

    print(f"  [OK] {len(chunks)} chunks de questionnaires créés")
    return chunks, metadatas


def load_verbatims(filepath: str = "./verbatims_by_commune.csv"):
    """Charge les verbatims par commune"""
    print(f"\nChargement des verbatims depuis {filepath}...")

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        df = pd.read_csv(filepath, encoding='latin1')

    chunks = []
    metadatas = []

    # Chercher la colonne commune
    commune_col = None
    for col in ['commune', 'Commune', 'nom', 'Nom']:
        if col in df.columns:
            commune_col = col
            break

    if commune_col:
        for idx, row in df.iterrows():
            commune = row[commune_col]
            # Concaténer toutes les colonnes pour cette ligne
            text = f"Verbatims - {commune}\n\n"
            for col in df.columns:
                if col != commune_col and pd.notna(row[col]):
                    text += f"{col}: {row[col]}\n"

            chunks.append(text)
            metadatas.append({
                'source': 'verbatim',
                'type': 'qualitatif',
                'nom': commune,
                'id': f'verbatim_{commune}'
            })
    else:
        # Pas de colonne commune identifiée
        for idx, row in df.iterrows():
            text = f"Verbatim {idx}\n\n"
            for col in df.columns:
                if pd.notna(row[col]):
                    text += f"{col}: {row[col]}\n"

            chunks.append(text)
            metadatas.append({
                'source': 'verbatim',
                'type': 'qualitatif',
                'chunk_index': idx,
                'id': f'verbatim_{idx}'
            })

    print(f"  [OK] {len(chunks)} verbatims créés")
    return chunks, metadatas


def load_data_quanti(filepath: str = "./data_comp_finalede2604_4_cleaned.csv"):
    """Charge les données quantitatives supplémentaires"""
    print(f"\nChargement des données quantitatives depuis {filepath}...")

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        df = pd.read_csv(filepath, encoding='latin1')

    chunks = []
    metadatas = []

    # Chercher la colonne commune
    commune_col = None
    for col in ['commune', 'Commune', 'nom', 'Nom']:
        if col in df.columns:
            commune_col = col
            break

    if commune_col:
        for commune, group in df.groupby(commune_col):
            text = f"Données quantitatives - {commune}\n\n"
            text += group.to_string(index=False)

            chunks.append(text)
            metadatas.append({
                'source': 'data_quanti',
                'type': 'quanti',
                'nom': commune,
                'id': f'data_quanti_{commune}'
            })
    else:
        # Chunker par blocs
        chunk_size = 100
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            text = f"Données quantitatives (lignes {i} à {i+len(chunk_df)})\n\n"
            text += chunk_df.to_string(index=False)

            chunks.append(text)
            metadatas.append({
                'source': 'data_quanti',
                'type': 'quanti',
                'chunk_index': i // chunk_size,
                'id': f'data_quanti_chunk_{i//chunk_size}'
            })

    print(f"  [OK] {len(chunks)} chunks de données quantitatives créés")
    return chunks, metadatas


if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY manquante")

    print("="*80)
    print("IMPORT DE TOUTES LES SOURCES DANS CHROMADB")
    print("="*80)

    # Initialiser le pipeline
    rag = ImprovedRAGPipeline(
        chroma_path="./chroma_v2",
        collection_name="communes_corses_v2",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        llm_model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )

    # Collecter tous les documents
    all_texts = []
    all_metadatas = []

    # 1. Entretiens
    if os.path.exists("./entretiens_lea.txt"):
        texts, metas = load_entretiens()
        all_texts.extend(texts)
        all_metadatas.extend(metas)

    # 2. Questionnaires
    if os.path.exists("./sortie_questionnaire_traited.csv"):
        texts, metas = load_questionnaires()
        all_texts.extend(texts)
        all_metadatas.extend(metas)

    # 3. Verbatims
    if os.path.exists("./verbatims_by_commune.csv"):
        texts, metas = load_verbatims()
        all_texts.extend(texts)
        all_metadatas.extend(metas)

    # 4. Données quantitatives
    if os.path.exists("./data_comp_finalede2604_4_cleaned.csv"):
        texts, metas = load_data_quanti()
        all_texts.extend(texts)
        all_metadatas.extend(metas)

    # Ingérer dans ChromaDB
    print(f"\n{'='*80}")
    print("INGESTION DANS CHROMADB")
    print(f"{'='*80}\n")
    print(f"Total documents à ingérer: {len(all_texts)}")

    if all_texts:
        rag.ingest_documents(all_texts, all_metadatas)

        print(f"\n{'='*80}")
        print("IMPORT TERMINE !")
        print(f"{'='*80}")
        print(f"[OK] {len(all_texts)} nouveaux documents importes")
        print(f"[OK] Collection: communes_corses_v2")
        print(f"[OK] Modele d'embedding: BAAI/bge-m3 (1024 dimensions)")
    else:
        print("\nAucun nouveau document à importer")
