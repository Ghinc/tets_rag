"""
Met à jour les fichiers JSON pour les cartes web avec les nouvelles données OppChoVec
"""

import pandas as pd
import json
import sys
from pathlib import Path

# Ajouter le chemin du répertoire Python
sys.path.insert(0, str(Path(__file__).parent.parent / "Python"))


def normaliser_1_10(serie):
    """
    Normalise une série entre 1 et 10
    """
    min_val = serie.min()
    max_val = serie.max()

    if max_val == min_val:
        return pd.Series([5.5] * len(serie), index=serie.index)

    return ((serie - min_val) / (max_val - min_val)) * 9 + 1


def generer_data_scores(fichier_excel):
    """
    Génère data_scores.json à partir du fichier Excel

    Args:
        fichier_excel: Chemin vers oppchovec_0_norm.xlsx
    """
    print("Generation de data_scores.json...")

    # Charger les données depuis la feuille Synthese
    df = pd.read_excel(fichier_excel, sheet_name='Synthese')

    # Utiliser les scores déjà normalisés 1-10
    df_export = df[['Zone', 'OppChoVec_1_10', 'Score_Opp_1_10', 'Score_Cho_1_10', 'Score_Vec_1_10']].copy()
    df_export.columns = ['Zone', 'OppChoVec', 'Score_Opp', 'Score_Cho', 'Score_Vec']

    # Convertir en dictionnaire (index = numéro de ligne)
    data_dict = df_export.to_dict('index')

    # Sauvegarder en JSON
    with open('data_scores.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    print(f"  - {len(df_export)} communes exportees")
    print("  - Fichier: data_scores.json")


def generer_data_indicateurs(fichier_excel):
    """
    Génère data_indicateurs.json avec tous les indicateurs

    Args:
        fichier_excel: Chemin vers oppchovec_0_norm.xlsx
    """
    print("\nGeneration de data_indicateurs.json...")

    # Charger toutes les feuilles
    df_synthese = pd.read_excel(fichier_excel, sheet_name='Synthese')
    df_opp = pd.read_excel(fichier_excel, sheet_name='Opp')
    df_cho = pd.read_excel(fichier_excel, sheet_name='Cho')
    df_vec = pd.read_excel(fichier_excel, sheet_name='Vec')

    # Fusionner toutes les données
    df_complet = df_synthese.merge(df_opp[['Zone', 'Opp1', 'Opp2', 'Opp3', 'Opp4']], on='Zone')
    df_complet = df_complet.merge(df_cho[['Zone', 'Cho1', 'Cho2']], on='Zone')
    df_complet = df_complet.merge(df_vec[['Zone', 'Vec1', 'Vec2', 'Vec3', 'Vec4']], on='Zone')

    # Renommer les colonnes normalisées pour plus de clarté
    df_export = df_complet[[
        'Zone',
        'OppChoVec_1_10', 'OppChoVec',
        'Score_Opp_1_10', 'Score_Opp',
        'Score_Cho_1_10', 'Score_Cho',
        'Score_Vec_1_10', 'Score_Vec',
        'Opp1', 'Opp2', 'Opp3', 'Opp4',
        'Cho1', 'Cho2',
        'Vec1', 'Vec2', 'Vec3', 'Vec4'
    ]].copy()

    # Convertir en dictionnaire
    data_dict = df_export.to_dict('index')

    # Sauvegarder en JSON
    with open('data_indicateurs.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    print(f"  - {len(df_export)} communes exportees")
    print(f"  - {len(df_export.columns)} colonnes")
    print("  - Fichier: data_indicateurs.json")


def main():
    """
    Point d'entree principal
    """
    print("=" * 80)
    print("MISE A JOUR DES DONNEES POUR LES CARTES WEB")
    print("=" * 80)
    print()

    fichier_excel = "../Python/oppchovec_0_norm.xlsx"

    if not Path(fichier_excel).exists():
        print(f"ERREUR: Fichier {fichier_excel} introuvable!")
        return

    # Générer les fichiers JSON
    generer_data_scores(fichier_excel)
    generer_data_indicateurs(fichier_excel)

    print("\n" + "=" * 80)
    print("MISE A JOUR TERMINEE")
    print("=" * 80)
    print()
    print("Fichiers generes:")
    print("  - data_scores.json (pour les cartes)")
    print("  - data_indicateurs.json (pour les details)")
    print()
    print("Prochaine etape: Recalculer les seuils de Jenks")
    print("  python calculer_seuils_jenks.py")


if __name__ == "__main__":
    main()
