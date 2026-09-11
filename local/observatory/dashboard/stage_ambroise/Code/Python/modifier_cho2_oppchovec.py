"""
Script pour créer le fichier CHO2 basé sur la proportion de Français 18+
pour les calculs OppChoVec.

Ce script lit les données de proportion d'étrangers 18+ et calcule la proportion
de Français 18+ (ayant potentiellement le droit de vote).
"""

import pandas as pd
from pathlib import Path

# Chemins des fichiers
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "proportion_etrangers_18plus_corse.xlsx"
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "Données" / "Corse_Commune"
OUTPUT_FILE = OUTPUT_DIR / "Cho2_francais_18plus.xlsx"

def creer_cho2_francais_18plus():
    """
    Crée le fichier CHO2 basé sur la proportion de Français 18+.
    """
    print(f"Lecture du fichier : {INPUT_FILE}")

    # Lire les données d'étrangers 18+
    df = pd.read_excel(INPUT_FILE)

    print(f"Données chargées : {len(df)} communes")
    print(f"Colonnes disponibles : {list(df.columns)}")

    # Calculer la proportion de Français 18+
    df['Proportion_Francais_18plus'] = 100 - df['Proportion_Etrangers_18plus']

    # Limiter les valeurs à l'intervalle [0, 100]
    df['Proportion_Francais_18plus'] = df['Proportion_Francais_18plus'].clip(0, 100)

    # Créer le dataframe de sortie avec les colonnes attendues par oppchovec.py
    df_output = pd.DataFrame({
        'Zone': df['Commune'],
        'Code commune': df['Code_INSEE'],
        'Proportion_Francais_18plus': df['Proportion_Francais_18plus']
    })

    # Trier par code commune
    df_output = df_output.sort_values('Code commune').reset_index(drop=True)

    # Créer le répertoire de sortie s'il n'existe pas
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le fichier
    print(f"\nSauvegarde vers : {OUTPUT_FILE}")
    df_output.to_excel(OUTPUT_FILE, index=False)

    # Afficher un aperçu des résultats
    print(f"\nAperçu des données créées :")
    print(df_output.head(10))

    print(f"\nStatistiques de Proportion_Francais_18plus :")
    print(df_output['Proportion_Francais_18plus'].describe())

    print(f"\nFichier cree avec succes : {OUTPUT_FILE}")
    print(f"  Nombre de communes : {len(df_output)}")

if __name__ == "__main__":
    creer_cho2_francais_18plus()
