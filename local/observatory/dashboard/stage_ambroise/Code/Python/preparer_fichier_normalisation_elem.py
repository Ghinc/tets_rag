"""
Script pour preparer le fichier oppchovec_0_norm.xlsx avec 3 sheets (Opp, Cho, Vec)
a partir des resultats d'oppchovec.py
"""

import pandas as pd
from pathlib import Path

def creer_fichier_3_sheets():
    """
    Cree le fichier oppchovec_0_norm.xlsx avec 3 sheets a partir de df_indicateur.xlsx
    """
    print("="*80)
    print("PREPARATION FICHIER POUR NORMALISATION ELEMENTAIRE")
    print("="*80)

    # Charger le fichier d'indicateurs genere par oppchovec.py
    print("\nChargement de df_indicateur.xlsx...")
    df = pd.read_excel('df_indicateur.xlsx')

    print(f"  {len(df)} communes chargees")
    print(f"  Colonnes: {list(df.columns)}")

    # Creer les 3 DataFrames pour chaque dimension
    print("\nCreation des feuilles par dimension...")

    # Feuille Opp
    df_opp = df[['Zone', 'Opp1', 'Opp2', 'Opp3', 'Opp4']].copy()
    print(f"  - Feuille Opp: {len(df_opp)} lignes, {len(df_opp.columns)} colonnes")

    # Feuille Cho
    df_cho = df[['Zone', 'Cho1', 'Cho2']].copy()
    print(f"  - Feuille Cho: {len(df_cho)} lignes, {len(df_cho.columns)} colonnes")

    # Feuille Vec
    df_vec = df[['Zone', 'Vec1', 'Vec2', 'Vec3', 'Vec4']].copy()
    print(f"  - Feuille Vec: {len(df_vec)} lignes, {len(df_vec.columns)} colonnes")

    # Ecrire le fichier Excel avec 3 feuilles
    output_file = 'oppchovec_0_norm.xlsx'
    print(f"\nEcriture du fichier {output_file}...")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_opp.to_excel(writer, sheet_name='Opp', index=False)
        df_cho.to_excel(writer, sheet_name='Cho', index=False)
        df_vec.to_excel(writer, sheet_name='Vec', index=False)

    print(f"\nFichier cree avec succes: {output_file}")
    print(f"  - 3 feuilles: Opp, Cho, Vec")
    print(f"  - {len(df)} communes")

    print("\n" + "="*80)
    print("TRAITEMENT TERMINE")
    print("="*80)

if __name__ == "__main__":
    creer_fichier_3_sheets()
