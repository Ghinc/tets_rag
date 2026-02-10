"""
Script pour créer des fichiers de verbatims par commune avec les données portrait (âge, sexe, profession)
"""
import pandas as pd
import os
from collections import defaultdict

# Charger le fichier CSV
df = pd.read_csv('donnees_brutes/sortie_questionnaire_traited.csv', encoding='utf-8')

# Utiliser les indices de colonnes pour éviter les problèmes d'encodage
# Index: 2=genre, 3=age, 4=profession, 5=commune(A-S), 6=commune(T-Z), 8=verbatim1, 9=verbatim2, 10=verbatim3
cols = df.columns.tolist()
print(f"Nombre de colonnes: {len(cols)}")

col_genre = cols[2]  # 'genre'
col_age = cols[3]    # 'age'
col_profession = cols[4]  # 'situation socioprofessionnelle'
col_commune_as = cols[5]  # commune A-S
col_commune_tz = cols[6]  # commune T-Z

# Colonnes de verbatims (justifications) - indices 8, 9, 10
verbatim_cols = [cols[8], cols[9], cols[10]]
print(f"Colonnes verbatim: {verbatim_cols}")

# Colonne des dimensions de qualité de vie choisies (index 7)
col_dimensions = cols[7]  # "Pour vous, qu'est-ce qui est important pour votre qualité de vie ?"
print(f"Colonne dimensions: {col_dimensions}")

# Créer le dossier de sortie
output_dir = 'donnees_brutes/verbatims_par_commune_avec_portrait'
os.makedirs(output_dir, exist_ok=True)

# Dictionnaire pour stocker les verbatims par commune
verbatims_by_commune = defaultdict(list)

# Parcourir chaque ligne
for idx, row in df.iterrows():
    # Déterminer la commune (peut être dans l'une ou l'autre colonne)
    commune = row[col_commune_as] if pd.notna(row[col_commune_as]) and str(row[col_commune_as]).strip() else row[col_commune_tz]

    if pd.isna(commune) or str(commune).strip() == '':
        continue

    commune = str(commune).strip()

    # Extraire les métadonnées
    genre = row[col_genre] if pd.notna(row[col_genre]) else 'Non spécifié'
    age = row[col_age] if pd.notna(row[col_age]) else 'Non spécifié'
    profession = row[col_profession] if pd.notna(row[col_profession]) else 'Non spécifié'

    # Extraire les dimensions de qualité de vie (séparées par ;)
    dimensions_raw = row[col_dimensions] if pd.notna(row[col_dimensions]) else ''
    dimensions_list = [d.strip() for d in str(dimensions_raw).split(';') if d.strip()]

    # Extraire les verbatims
    for i, col in enumerate(verbatim_cols, 1):
        verbatim = row[col] if pd.notna(row[col]) else ''
        verbatim = str(verbatim).strip()

        # Associer la dimension correspondante (i-1 car index 0-based)
        dimension = dimensions_list[i-1] if i-1 < len(dimensions_list) else 'Non spécifié'

        if verbatim and verbatim.lower() not in ['nan', '', 'none']:
            verbatims_by_commune[commune].append({
                'genre': genre,
                'age': age,
                'profession': profession,
                'dimension': dimension,
                'verbatim': verbatim,
                'choix_numero': i
            })

# Créer un fichier par commune
print(f"Création de fichiers pour {len(verbatims_by_commune)} communes...")

for commune, verbatims in verbatims_by_commune.items():
    # Nettoyer le nom de commune pour le nom de fichier
    safe_commune = commune.replace('/', '-').replace('\\', '-').replace(':', '-')
    filename = os.path.join(output_dir, f'{safe_commune}.txt')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"VERBATIMS - COMMUNE DE {commune.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Nombre de verbatims: {len(verbatims)}\n\n")

        for i, v in enumerate(verbatims, 1):
            f.write(f"--- Verbatim {i} ---\n")
            f.write(f"Genre: {v['genre']}\n")
            f.write(f"Âge: {v['age']}\n")
            f.write(f"Profession: {v['profession']}\n")
            f.write(f"Dimension qualité de vie: {v['dimension']}\n")
            f.write(f"Verbatim: {v['verbatim']}\n\n")

    print(f"  - {commune}: {len(verbatims)} verbatims")

# Créer aussi un fichier CSV global avec toutes les données
all_data = []
for commune, verbatims in verbatims_by_commune.items():
    for v in verbatims:
        all_data.append({
            'commune': commune,
            'genre': v['genre'],
            'age': v['age'],
            'profession': v['profession'],
            'dimension': v['dimension'],
            'verbatim': v['verbatim'],
            'choix_numero': v['choix_numero']
        })

df_output = pd.DataFrame(all_data)
df_output.to_csv(os.path.join(output_dir, '_tous_verbatims_avec_portrait.csv'), index=False, encoding='utf-8')
print(f"\nFichier CSV global créé: _tous_verbatims_avec_portrait.csv")
print(f"Total: {len(all_data)} verbatims")
