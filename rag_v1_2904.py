"""
Script RAG pour le traitement des données de communes corses
Conversion du notebook rag_v1_2904.ipynb
"""

import pandas as pd
import re
import os
import pickle
import glob
from collections import defaultdict

# Imports pour le RAG
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk

# Imports pour Wikipedia
import wikipedia
import wikipediaapi
import time

# Imports pour RDF (optionnel)
import json
from rdflib import Graph

# Imports pour OpenAI
import openai


# ============================================================================
# PARTIE 1 : TRANSFORMATION DES DONNÉES CSV EN TEXTE
# ============================================================================

def row_to_text(row):
    """Transforme une ligne de données en texte descriptif"""
    return (
       f"La commune de {row['libell_x']} compte une part de {row['part_des_cadres_et_prof_intellectuelles_sup_dans_le_nb_demplois_au_lt_2020']}% de cadres et professions intellectuelles supérieures, "
        f"{row['part_des_ouvriers_dans_le_nb_demplois_au_lt_2020']}% d'ouvriers, avec un taux d'activité de {row['taux_dactivit_par_tranche_d_ge_2020']}%. "
        f"{row['part_des_emplois_sal_dans_le_nb_demplois_au_lt_2020']}% des emplois y sont salariés. "
        f"On y trouve {row['point_de_contact_postal_en_nombre_2021']} points de contact postal, {row['hypermarch_supermarch_en_nombre_2021']} hypermarchés ou supermarchés, "
        f"{row['sup_rette_picerie_en_nombre_2021']} supérettes ou épiceries, {row['boulangerie_en_nombre_2021']} boulangeries et {row['boucherie_charcuterie_en_nombre_en_nombre_2021']} boucheries-charcuteries. "
        f"En matière d'éducation, la commune dispose de {row['cole_maternelle_en_nombre_2021']} écoles maternelles, {row['cole_l_mentaire_en_nombre_2021']} écoles élémentaires, "
        f"{row['coll_ge_en_nombre_2021']} collèges et {row['lyc_e_en_nombre_2021']} lycées. "
        f"Elle possède {row['service_durgences_en_nombre_2021']} services d'urgence, {row['m_decin_g_n_raliste_en_nombre_2021']} médecins généralistes, "
        f"{row['pharmacie_en_nombre_2021']} pharmacies, {row['infirmier_en_nombre_2021']} infirmiers et {row['cr_che_en_nombre_2021']} crèches. "
        f"La sécurité est assurée par {row['police_gendarmerie_en_nombre_2021']} postes de police ou gendarmerie. "
        f"Il y a également {row['p_le_emploi_r_seau_de_proximit_et_antennes_ifs_en_nombre_2021']} agences Pôle emploi. "
        f"Le taux de pauvreté est de {row['taux_de_pauvret_2020']}%, la part de logements vacants est de {row['part_des_logements_vacants_dans_le_total_des_logements_2020']}%, "
        f"et celle des résidences secondaires est de {row['part_des_r_s_secondaires_y_compris_les_logements_occasionnels_dans_le_total_des_logements_2020']}%. "
        f"Le taux de coups et blessures volontaires est de {row['coups_et_blessures_volontaires_taux_2022']} pour 1 000 habitants. "
        f"Le prix moyen au m² est de {row['prixm2moyen']}€, pour une surface moyenne des logements de {row['surfacemoy']} m².")


def dataset_to_texts(csv_path, output_path):
    """Convertit un dataset CSV en fichier texte descriptif"""
    df = pd.read_csv(csv_path)
    texts = df.apply(row_to_text, axis=1)

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in texts:
            f.write(line + '\n')

    print(f"{len(texts)} descriptions générées et enregistrées dans {output_path}.")


# ============================================================================
# PARTIE 2 : TRAITEMENT DU QUESTIONNAIRE
# ============================================================================

def process_questionnaire(csv_path):
    """Traite le questionnaire et retourne les dataframes nécessaires"""
    # Chargement du fichier
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Suppression des 7 premières lignes
    df = df.iloc[8:].reset_index(drop=True)

    # Suppression des lignes de test
    df = df.drop([8, 10], errors='ignore').reset_index(drop=True)

    # Renommage de la colonne dimensions_qdv avec vérification
    old_col_name = "Pour vous, qu'est-ce qui est important pour votre qualité de vie ? Choisissez 3 images"
    if old_col_name in df.columns:
        df.rename(columns={old_col_name: "dimensions_qdv"}, inplace=True)
        print(f"✅ Colonne '{old_col_name}' renommée en 'dimensions_qdv'")
    else:
        # Recherche d'une colonne similaire
        possible_cols = [col for col in df.columns if "qualité de vie" in col.lower() or "images" in col.lower()]
        if possible_cols:
            print(f"⚠️ Colonne exacte introuvable. Colonnes similaires trouvées : {possible_cols}")
            df.rename(columns={possible_cols[0]: "dimensions_qdv"}, inplace=True)
            print(f"✅ Colonne '{possible_cols[0]}' renommée en 'dimensions_qdv'")
        else:
            print(f"❌ Colonne 'dimensions_qdv' introuvable dans le fichier")
            print(f"Colonnes disponibles : {list(df.columns)}")

    # Fusion des colonnes communes
    commune_col_1 = "Dans quelle commune résidez-vous ? ( A à S)"
    commune_col_2 = "Dans quelle commune résidez-vous ? (T à Z)"

    if commune_col_1 in df.columns and commune_col_2 in df.columns:
        df["commune"] = df[commune_col_1].combine_first(df[commune_col_2])
    elif commune_col_1 in df.columns:
        df["commune"] = df[commune_col_1]
    elif commune_col_2 in df.columns:
        df["commune"] = df[commune_col_2]
    else:
        # Recherche d'une colonne avec "commune"
        commune_cols = [col for col in df.columns if "commune" in col.lower()]
        if commune_cols:
            df["commune"] = df[commune_cols[0]]
            print(f"⚠️ Utilisation de la colonne : {commune_cols[0]}")
        else:
            raise ValueError("❌ Impossible de trouver une colonne 'commune' dans le fichier")

    # Suppression des colonnes originales si elles existent
    cols_to_drop = [commune_col_1, commune_col_2]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # Transformation des colonnes ordinales
    ORDINAL_ENCODINGS = {
        ("Très peu satisfait", "Peu satisfait", "Neutre", "Satisfait", "Très satisfait"): [0, 1, 2, 3, 4],
        ("Très peu entouré", "Peu entouré", "Moyennement entouré", "Bien entouré", "Très bien entouré"): [0, 1, 2, 3, 4],
        ("Très peu impliqué", "Peu impliqué", "Moyenement impliqué", "Impliqué", "Très impliqué"): [0, 1, 2, 3, 4],
        ("Oui", "Non"): [0, 1],
        ("Jamais", "Rarement", "Souvent", "Toujours"): [0, 1, 2, 3],
    }

    df_encoded, ordinal_columns, encoding_maps = detect_and_encode_ordinal_columns(df, ORDINAL_ENCODINGS)

    print("Colonnes détectées comme ordinales :", ordinal_columns)

    return df_encoded


def detect_and_encode_ordinal_columns(df, encodings):
    """Détecte et encode les colonnes ordinales"""
    ordinal_cols = []
    encoders = {}

    for col in df.columns:
        unique_values = set(df[col].dropna().unique())
        for label_set, encoding in encodings.items():
            if unique_values.issubset(set(label_set)):
                ordinal_cols.append(col)
                mapping = dict(zip(label_set, encoding))
                encoders[col] = mapping
                df[col + "_encoded"] = df[col].map(mapping)
                break

    return df, ordinal_cols, encoders


def calculate_dimensions_by_commune(df):
    """Calcule le pourcentage de chaque dimension par commune"""
    # Vérification que les colonnes nécessaires existent
    if "dimensions_qdv" not in df.columns:
        print("❌ La colonne 'dimensions_qdv' n'existe pas dans le DataFrame")
        print(f"Colonnes disponibles : {list(df.columns)}")
        # Retourner des DataFrames vides
        return pd.DataFrame(columns=["commune", "dimensions_qdv", "count", "total_respondants", "percentage"]), pd.Series(dtype=int, name="total_respondants")

    if "commune" not in df.columns:
        print("❌ La colonne 'commune' n'existe pas dans le DataFrame")
        return pd.DataFrame(columns=["commune", "dimensions_qdv", "count", "total_respondants", "percentage"]), pd.Series(dtype=int, name="total_respondants")

    df_dimensions = df[["commune", "dimensions_qdv"]].dropna().copy()

    if df_dimensions.empty:
        print("⚠️ Aucune donnée trouvée pour les dimensions de qualité de vie")
        return pd.DataFrame(columns=["commune", "dimensions_qdv", "count", "total_respondants", "percentage"]), pd.Series(dtype=int, name="total_respondants")

    df_dimensions["dimensions_qdv"] = df_dimensions["dimensions_qdv"].str.split(";")
    df_exploded = df_dimensions.explode("dimensions_qdv")
    df_exploded["dimensions_qdv"] = df_exploded["dimensions_qdv"].str.strip()

    respondants_per_commune = df.groupby("commune").size().rename("total_respondants")

    dimension_counts = (
        df_exploded.groupby(["commune", "dimensions_qdv"]).size().rename("count").reset_index()
    )

    dimension_counts = dimension_counts.merge(respondants_per_commune, on="commune")
    dimension_counts["percentage"] = (
        dimension_counts["count"] / dimension_counts["total_respondants"] * 100
    ).round(2)

    return dimension_counts, respondants_per_commune


def create_verbatims_dataset(df):
    """Crée un dataset regroupant les verbatims par commune"""
    # Renommage avec vérification des colonnes
    rename_map = {
        "Justifiez brièvement le choix de votre première\xa0image (5 mots).": "verbatim_1",
        "Justifiez brièvement le choix de votre deuxième image (5 mots).": "verbatim_2",
        "Justifiez brièvement le choix de votre troisième image (5 mots).": "verbatim_3",
    }

    # Ne renommer que les colonnes qui existent
    rename_map_filtered = {old: new for old, new in rename_map.items() if old in df.columns}
    if rename_map_filtered:
        df.rename(columns=rename_map_filtered, inplace=True)
    else:
        # Chercher des colonnes similaires
        verbatim_cols = [col for col in df.columns if "justifiez" in col.lower() or "verbatim" in col.lower()]
        print(f"⚠️ Colonnes verbatims exactes introuvables. Colonnes similaires : {verbatim_cols}")

    # Vérifier quelles colonnes verbatim existent
    available_verbatim_cols = [col for col in ["verbatim_1", "verbatim_2", "verbatim_3"] if col in df.columns]
    if not available_verbatim_cols:
        print("❌ Aucune colonne verbatim trouvée")
        return pd.DataFrame(columns=["commune", "verbatim"])

    if "commune" not in df.columns:
        print("❌ La colonne 'commune' n'existe pas dans le DataFrame")
        return pd.DataFrame(columns=["commune", "verbatim"])

    verbatims_df = df[["commune"] + available_verbatim_cols].copy()

    verbatims_long = verbatims_df.melt(
        id_vars="commune",
        value_vars=available_verbatim_cols,
        var_name="verbatim_source",
        value_name="verbatim"
    )

    verbatims_long["verbatim"] = verbatims_long["verbatim"].astype(str).str.strip()
    verbatims_long = verbatims_long[verbatims_long["verbatim"].notnull() & (verbatims_long["verbatim"] != "")]

    verbatims_by_commune = (
        verbatims_long.groupby("commune")["verbatim"]
        .apply(lambda x: list(x))
        .reset_index()
    )

    return verbatims_by_commune


def calculate_mean_by_commune(df):
    """Calcule les moyennes des données numériques par commune"""
    df_copy = df.copy()
    numerical_cols = df_copy.select_dtypes(include=['number']).columns.tolist()

    if 'commune' in numerical_cols:
        numerical_cols.remove('commune')

    cols_to_keep = ['commune'] + numerical_cols
    df_numerical = df_copy[cols_to_keep]
    df_mean_by_commune = df_numerical.groupby('commune').mean().reset_index()

    return df_mean_by_commune


def rename_columns_for_readability(df_mean_by_commune):
    """Renomme les colonnes pour plus de lisibilité"""
    df_mean_by_commune = df_mean_by_commune.rename(columns={
        'Sur une échelle de 1 à 5, pourriez-vous estimer à quel point vous êtes heureux ces derniers temps ?': 'Score bonheur',
        'Sur une échelle de 1 à 5,\xa0pourriez-vous évaluer votre qualité de vie ces derniers temps ?': 'Score qualité de vie',
        "Sur une échelle de 1 à 5, pourriez-vous évaluer votre confiance en l'avenir ?": 'Score confiance avenir',
        "Les services de transports_encoded": "Transports",
        "L'accès à l'éducation_encoded": "Éducation",
        "La couverture des réseaux téléphoniques_encoded": "Réseaux téléphoniques",
        "Les institutions étatiques (niveau de confiance)_encoded": "Institutions",
        "Le tourisme (ressenti localement)_encoded": "Tourisme",
        "La sécurité_encoded": "Sécurité",
        "L'offre de santé _encoded": "Santé",
        "Votre situation professionnelle_encoded": "Situation pro",
        "Vos revenus_encoded": "Revenus",
        "La répartition de votre temps entre travail et temps personnel_encoded": "Temps travail/perso",
        "Votre logement_encoded": "Logement",
        "L'offre de services autour de chez vous_encoded": "Services locaux",
        "Votre accès à la culture_encoded": "Culture",
        "Vous sentez-vous bien entouré ?_encoded": "Soutien social",
        "Êtes-vous impliqué dans des activités associatives ?_encoded": "Vie associative"
    })
    return df_mean_by_commune


def generate_quanti_text(commune, df_mean):
    """Génère un texte descriptif des indicateurs quantitatifs"""
    row = df_mean[df_mean['commune'] == commune]
    if row.empty:
        return f"Commune : {commune}\n\nAucune donnée quantitative disponible."

    row = row.iloc[0]
    text = f"Commune : {commune}\n\nMoyennes des indicateurs de qualité de vie :\n"
    for col in df_mean.columns:
        if col not in ['commune', 'ID']:
            text += f"- {col} : {row[col]:.2f}\n"
    return text


def generate_be_text(commune, df_dims, df_verbatims):
    """Génère un texte descriptif des dimensions du bien-être"""
    dims = df_dims[df_dims['commune'] == commune]
    verb_row = df_verbatims[df_verbatims['commune'] == commune]

    text = f"Commune : {commune}\n\nDimensions du bien-être identifiées :\n"

    if dims.empty:
        text += "- Aucune dimension renseignée.\n"
    else:
        for _, row in dims.iterrows():
            dim = row['dimensions_qdv']
            pct = row['percentage']
            text += f"- {dim} : sélectionnée par {pct:.1f}% des répondants\n"

    text += "\nVerbatims associés :\n"
    if verb_row.empty or not verb_row.iloc[0]['verbatim']:
        text += "- Aucun verbatim disponible.\n"
    else:
        for i, v in enumerate(verb_row.iloc[0]['verbatim'], 1):
            text += f"{i}. {v}\n"
    return text


def generate_rag_files(df_mean_by_commune, dimension_counts, verbatims_by_commune):
    """Génère les fichiers RAG-friendly par commune"""
    output_dir_quanti = "rag_quanti"
    output_dir_be = "rag_be"
    os.makedirs(output_dir_quanti, exist_ok=True)
    os.makedirs(output_dir_be, exist_ok=True)

    communes = set(df_mean_by_commune['commune']).union(
        dimension_counts['commune']
    ).union(verbatims_by_commune['commune'])

    for commune in communes:
        safe_filename = commune.replace(" ", "_").replace("/", "-")

        # Texte quantitatif
        quanti_text = generate_quanti_text(commune, df_mean_by_commune)
        with open(os.path.join(output_dir_quanti, f"{safe_filename}.txt"), "w", encoding="utf-8") as f:
            f.write(quanti_text)

        # Texte dimensions + verbatims
        be_text = generate_be_text(commune, dimension_counts, verbatims_by_commune)
        with open(os.path.join(output_dir_be, f"{safe_filename}.txt"), "w", encoding="utf-8") as f:
            f.write(be_text)

    print("✅ Tous les fichiers RAG-friendly ont été générés par commune.")


# ============================================================================
# PARTIE 3 : SCRAPING WIKIPEDIA (OPTIONNEL)
# ============================================================================

def scrape_wikipedia(csv_path, enable_scraping=False):
    """Scrape les pages Wikipedia des communes corses"""
    if not enable_scraping:
        print("⏭️ Scraping Wikipedia désactivé")
        return

    df = pd.read_csv(csv_path)
    communes_list = sorted(df["libell_x"].dropna().unique().tolist())

    wikipedia.set_lang("fr")
    wiki_wiki = wikipediaapi.Wikipedia(
        language='fr',
        user_agent='MonProjetScraping/1.0 (contact@example.com)'
    )

    def get_commune_page(commune):
        suffixes = [
            " (Corse-du-Sud)",
            " (Haute-Corse)",
            " (commune)",
            ""
        ]

        for suffix in suffixes:
            title = commune + suffix
            page = wiki_wiki.page(title)

            if not page.exists():
                continue

            text = page.text.strip().lower()
            intro = text[:200]

            if "peut désigner" in intro or "peut faire référence à" in intro:
                print(f"⚠️ Page d'ambiguïté détectée pour {title}")
                continue

            if "corse" in page.summary.lower() or "corse" in page.title.lower():
                print(f"✔️ Page valide trouvée : {title}")
                return page

        print(f"❌ Aucune page correcte trouvée pour {commune}")
        return None

    data = []
    for commune in communes_list:
        print(f"Scraping {commune}...")
        page = get_commune_page(commune)

        if page:
            summary = page.summary
            full_content = page.text
        else:
            summary = None
            full_content = None

        data.append({
            "commune": commune,
            "résumé": summary,
            "contenu_wiki": full_content
        })

        time.sleep(1)

    df = pd.DataFrame(data)
    df.to_csv("communes_corse_wikipedia.csv", index=False, encoding='utf-8')
    print("✅ Scraping terminé.")


def process_wiki_to_text(wiki_csv_path, output_path):
    """Convertit les données Wikipedia en fichier texte"""
    if not os.path.exists(wiki_csv_path):
        print(f"⏭️ Fichier {wiki_csv_path} introuvable, étape ignorée")
        return

    df_wiki = pd.read_csv(wiki_csv_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df_wiki.iterrows():
            nom = row["commune"]
            resume = row["résumé"]
            description = row["contenu_wiki"]

            bloc = f"""### {nom}

Résumé : {resume}

Description : {description}

---

"""
            f.write(bloc)

    print(f"✅ Fichier texte généré avec succès à : {output_path}")


# ============================================================================
# PARTIE 4 : RAG - INDEXATION ET RECHERCHE
# ============================================================================

def chunk_text(text, source_type="default"):
    """Découpe un texte en chunks de taille variable selon le type"""
    nltk.download("punkt", quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize

    max_words = {
        "entretien": 80,
        "wiki": 150,
        "quanti": 50,
        "be_verbatims": 60,
        "csv_text": 100,
        "default": 100
    }.get(source_type, 100)

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_len = len(words)
        else:
            current_chunk.extend(words)
            current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_embeddings_and_index():
    """Crée les embeddings et indexe dans ChromaDB"""
    chroma_client = chromadb.PersistentClient(path="./chroma_txt")
    collection = chroma_client.get_or_create_collection(name="communes_corses_txt")

    if os.path.exists("embeddings.pkl"):
        print("✅ Chargement des embeddings depuis le cache...")
        with open("embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        documents = data["documents"]
        embeddings = data["embeddings"]
        metadatas = data["metadatas"]
        ids = data["ids"]
    else:
        print("🚀 Génération des embeddings...")
        documents = []
        metadatas = []
        ids = []
        chunk_counter = 0

        # Traitement du fichier CSV texte
        csv_text_path = "communes_text.txt"
        if os.path.exists(csv_text_path):
            with open(csv_text_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"La commune de ([^ ]+)", line)
                commune_name = match.group(1) if match else f"commune_{i}"
                chunks = chunk_text(line, source_type="csv_text")

                for j, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "nom": commune_name,
                        "source": "csv_text",
                        "chunk": j
                    })
                    ids.append(f"csv_text_commune_{i}_chunk_{j}")
                    chunk_counter += 1

        # Traitement du fichier wiki
        txt_path = "communes_corses_wiki.txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            blocs = re.split(r"\n---+\n", raw_text.strip())

            for i, bloc in enumerate(blocs):
                match = re.search(r"### (.+?)\n\nRésumé : (.+?)\n\nDescription : (.+)", bloc, re.DOTALL)
                if match:
                    nom, resume, description = match.groups()
                    full_text = f"{resume.strip()}\n\n{description.strip()}"
                    chunks = chunk_text(full_text, source_type="wiki")

                    for j, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "nom": nom.strip(),
                            "résumé": resume.strip(),
                            "source": "wiki",
                            "chunk": j
                        })
                        ids.append(f"commune_{i}_chunk_{j}")
                        chunk_counter += 1

        # Intégration des fichiers rag_quanti et rag_be
        def add_rag_fichiers_txt(folder_path, source_label):
            if not os.path.exists(folder_path):
                return

            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
            for path in txt_files:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if not text:
                        continue

                filename = os.path.basename(path)
                nom_commune = filename.replace(".txt", "").replace("_", " ").strip()

                documents.append(text)
                metadatas.append({
                    "nom": nom_commune,
                    "source": source_label,
                    "chunk": 0
                })
                ids.append(f"{source_label}_{nom_commune.replace(' ', '_')}_chunk_0")
                print(f"📄 Ajouté : {filename} | source={source_label}")

        add_rag_fichiers_txt("rag_quanti", "quanti")
        add_rag_fichiers_txt("rag_be", "be_verbatims")

        # Embedding par batches
        print("🔄 Encodage des documents...")
        model = SentenceTransformer("intfloat/e5-base-v2")
        documents_prefixed = [f"passage: {doc}" for doc in documents]

        embeddings = []
        batch_size = 128
        for i in tqdm(range(0, len(documents_prefixed), batch_size), desc="Encodage"):
            batch = documents_prefixed[i:i + batch_size]
            batch_embeddings = model.encode(batch).tolist()
            embeddings.extend(batch_embeddings)

        # Sauvegarde du cache
        with open("embeddings.pkl", "wb") as f:
            pickle.dump({
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "ids": ids
            }, f)
        print(f"✅ Embeddings sauvegardés ({chunk_counter} chunks)")

    # Indexation par batches
    print("🔄 Indexation dans ChromaDB...")
    batch_size = 5000
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexation"):
        collection.add(
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )

    print(f"✅ Indexation réussie ! {len(documents)} chunks ajoutés.")
    return collection


def search_rag(collection, question, n_results=10):
    """Effectue une recherche dans la base RAG"""
    model = SentenceTransformer("intfloat/e5-base-v2")
    query_vector = model.encode([f"query: {question}"]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )

    return results

def ask_rag_with_llm(collection, question, openai_api_key=None, n_chunks=5, model_name="gpt-3.5-turbo"):
    """Pose une question au RAG avec génération de réponse via OpenAI"""
    # Utiliser la clé API fournie en paramètre, sinon chercher dans les variables d'environnement
    if openai_api_key:
        openai.api_key = openai_api_key
    elif os.environ.get("OPENAI_API_KEY"):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    else:
        raise ValueError("❌ Clé API OpenAI non fournie. Définissez la variable d'environnement OPENAI_API_KEY ou passez-la en paramètre.")

    embed_model = SentenceTransformer("intfloat/e5-base-v2")
    query_embedding = embed_model.encode([f"query: {question}"]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_chunks
    )

    retrieved_docs = results["documents"][0]
    retrieved_context = "\n\n".join(retrieved_docs)

    prompt = f"""Tu es un conseiller municipal. Ton but est de donner des informations sur la qualité de vie dans les communes Corses, pour guider les politiques publiques, en te basant uniquement sur les informations suivantes :

{retrieved_context}

Question : {question}
Réponse :
"""

    from openai import OpenAI
    client = OpenAI(api_key=openai.api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Tu es un assistant utile et factuel. Ne réponds qu'avec les informations données."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    return answer


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale d'exécution"""
    print("=" * 80)
    print("SCRIPT RAG - COMMUNES CORSES")
    print("=" * 80)

    # Configuration
    DATA_CSV_PATH = "data_comp_finalede2604_4_cleaned.csv"
    QUESTIONNAIRE_CSV_PATH = "sortie_questionnaire_traited.csv"

    # Étape 1 : Transformation des données CSV en texte
    print("\n[1] Transformation des données CSV en texte...")
    if os.path.exists(DATA_CSV_PATH):
        dataset_to_texts(DATA_CSV_PATH, "communes_text.txt")
    else:
        print(f"⚠️ Fichier {DATA_CSV_PATH} introuvable")

    # Étape 2 : Traitement du questionnaire
    print("\n[2] Traitement du questionnaire...")
    if os.path.exists(QUESTIONNAIRE_CSV_PATH):
        try:
            df = process_questionnaire(QUESTIONNAIRE_CSV_PATH)
            dimension_counts, respondants_per_commune = calculate_dimensions_by_commune(df)
            verbatims_by_commune = create_verbatims_dataset(df)
            df_mean_by_commune = calculate_mean_by_commune(df)
            df_mean_by_commune = rename_columns_for_readability(df_mean_by_commune)

            # Vérifier que respondants_per_commune n'est pas vide
            if not respondants_per_commune.empty:
                df_mean_by_commune = df_mean_by_commune.merge(respondants_per_commune.reset_index(), on="commune", how="left")
            else:
                print("⚠️ Aucun répondant trouvé par commune")

            # Génération des fichiers RAG
            generate_rag_files(df_mean_by_commune, dimension_counts, verbatims_by_commune)

            # Sauvegarde des CSV
            df_mean_by_commune.to_csv("df_mean_by_commune.csv", index=False, encoding="utf-8")
            verbatims_by_commune.to_csv("verbatims_by_commune.csv", index=False, encoding="utf-8")
            dimension_counts.to_csv("dimension_counts.csv", index=False, encoding="utf-8")
            print("✅ Fichiers CSV sauvegardés")
        except Exception as e:
            print(f"❌ Erreur lors du traitement du questionnaire : {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Fichier {QUESTIONNAIRE_CSV_PATH} introuvable")

    # Étape 3 : Scraping Wikipedia (désactivé par défaut)
    print("\n[3] Scraping Wikipedia...")
    scrape_wikipedia(DATA_CSV_PATH, enable_scraping=False)

    # Étape 4 : Conversion Wikipedia en texte
    print("\n[4] Conversion Wikipedia en texte...")
    process_wiki_to_text("communes_corse_wikipedia.csv", "communes_corses_wiki.txt")

    # Étape 5 : Création des embeddings et indexation
    print("\n[5] Création des embeddings et indexation...")
    try:
        collection = create_embeddings_and_index()

        # Étape 6 : Exemple de recherche
        print("\n[6] Exemple de recherche...")
        question = "Combien y a-t-il de médecins à Afa ?"
        results = search_rag(collection, question, n_results=5)

        print(f"\n🔍 Question : {question}")
        print("\n📋 Résultats :")
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            print(f"\n🔹 Commune : {metadata.get('nom', 'N/A')} | Source : {metadata.get('source', 'N/A')}")
            print(f"Extrait : {doc[:200]}...")
    except Exception as e:
        print(f"❌ Erreur lors de la création des embeddings ou de la recherche : {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ SCRIPT TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    main()
