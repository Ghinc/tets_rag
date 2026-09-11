"""
Génère data_normalisation_elem.json pour les cartes web
à partir de oppchovec_normalisation_elementaire.xlsx
"""

import pandas as pd
import json
import sys
from pathlib import Path

# Ajouter le chemin du répertoire Python
sys.path.insert(0, str(Path(__file__).parent.parent / "Python"))


def generer_data_normalisation_elem():
    """
    Génère data_normalisation_elem.json depuis oppchovec_normalisation_elementaire.xlsx
    """
    print("=" * 80)
    print("GENERATION data_normalisation_elem.json")
    print("=" * 80)
    print()

    fichier_excel = "../Python/oppchovec_normalisation_elementaire.xlsx"

    # Charger toutes les feuilles
    print("Chargement des donnees depuis Excel...")
    df_synthese = pd.read_excel(fichier_excel, sheet_name='Synthese')
    df_opp = pd.read_excel(fichier_excel, sheet_name='Opp')
    df_cho = pd.read_excel(fichier_excel, sheet_name='Cho')
    df_vec = pd.read_excel(fichier_excel, sheet_name='Vec')

    print(f"  {len(df_synthese)} communes chargees")

    # Fusionner toutes les données
    df_complet = df_synthese.merge(
        df_opp[['Zone', 'Opp1', 'Opp1_norm', 'Opp2', 'Opp2_norm', 'Opp3', 'Opp3_norm', 'Opp4', 'Opp4_norm']],
        on='Zone'
    )
    df_complet = df_complet.merge(
        df_cho[['Zone', 'Cho1', 'Cho1_norm', 'Cho2', 'Cho2_norm']],
        on='Zone'
    )
    df_complet = df_complet.merge(
        df_vec[['Zone', 'Vec1', 'Vec1_norm', 'Vec2', 'Vec2_norm', 'Vec3', 'Vec3_norm', 'Vec4', 'Vec4_norm']],
        on='Zone'
    )

    # Convertir en dictionnaire
    data_dict = df_complet.to_dict('index')

    # Sauvegarder en JSON
    print("\nExport vers data_normalisation_elem.json...")
    with open('data_normalisation_elem.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    print(f"  {len(data_dict)} communes exportees")
    print(f"  {len(df_complet.columns)} colonnes")
    print("  Fichier: data_normalisation_elem.json")

    # Aussi créer data_scores_normalisation_elem.json (version simplifiée pour les cartes)
    print("\nExport vers data_scores_normalisation_elem.json...")
    df_scores = df_synthese[['Zone', 'OppChoVec_1_10', 'Score_Opp_1_10', 'Score_Cho_1_10', 'Score_Vec_1_10']].copy()
    df_scores.columns = ['Zone', 'OppChoVec', 'Score_Opp', 'Score_Cho', 'Score_Vec']
    data_scores = df_scores.to_dict('index')

    with open('data_scores_normalisation_elem.json', 'w', encoding='utf-8') as f:
        json.dump(data_scores, f, indent=2, ensure_ascii=False)

    print(f"  {len(data_scores)} communes exportees")
    print("  Fichier: data_scores_normalisation_elem.json")

    print("\n" + "=" * 80)
    print("GENERATION TERMINEE")
    print("=" * 80)

    return df_complet


def main():
    df = generer_data_normalisation_elem()
    return df


if __name__ == "__main__":
    df = main()
