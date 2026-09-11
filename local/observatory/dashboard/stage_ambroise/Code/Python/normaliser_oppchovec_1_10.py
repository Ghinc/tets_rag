"""
Script pour normaliser OppChoVec et les dimensions entre 0 et 10
et créer un fichier Excel multi-feuilles
"""

import pandas as pd
import numpy as np


def normaliser_0_10(serie):
    """
    Normalise une série entre 0 et 10

    Args:
        serie: pandas Series à normaliser

    Returns:
        pandas Series normalisée entre 0 et 10
    """
    min_val = serie.min()
    max_val = serie.max()

    if max_val == min_val:
        return pd.Series([5.0] * len(serie), index=serie.index)

    # Normalisation: (x - min) / (max - min) * 10
    # Pour avoir des valeurs entre 0 et 10
    return ((serie - min_val) / (max_val - min_val)) * 10


def creer_excel_multi_feuilles(fichier_entree, fichier_sortie):
    """
    Crée un fichier Excel multi-feuilles avec normalisation 0-10

    Args:
        fichier_entree: Chemin vers oppchovec_0_norm.xlsx
        fichier_sortie: Chemin vers le fichier de sortie
    """
    print("=" * 80)
    print("NORMALISATION OPPCHOVEC ET DIMENSIONS ENTRE 0 ET 10")
    print("=" * 80)
    print()

    # Charger les données
    print(f"Chargement de {fichier_entree}...")
    df = pd.read_excel(fichier_entree)

    # Normaliser OppChoVec et les dimensions entre 0 et 10
    print("Normalisation entre 0 et 10...")
    df['OppChoVec_0_10'] = normaliser_0_10(df['OppChoVec'])
    df['Score_Opp_0_10'] = normaliser_0_10(df['Score_Opp'])
    df['Score_Cho_0_10'] = normaliser_0_10(df['Score_Cho'])
    df['Score_Vec_0_10'] = normaliser_0_10(df['Score_Vec'])

    # Créer le fichier Excel avec plusieurs feuilles
    print(f"\nCreation de {fichier_sortie}...")
    with pd.ExcelWriter(fichier_sortie, engine='openpyxl') as writer:

        # FEUILLE 1: SYNTHESE
        print("  - Feuille 'Synthese'...")
        colonnes_synthese = [
            'Zone',
            'OppChoVec_0_10', 'OppChoVec',
            'Score_Opp_0_10', 'Score_Opp',
            'Score_Cho_0_10', 'Score_Cho',
            'Score_Vec_0_10', 'Score_Vec'
        ]
        # Filtrer les colonnes qui existent réellement
        colonnes_synthese_existantes = [col for col in colonnes_synthese if col in df.columns]
        df_synthese = df[colonnes_synthese_existantes].copy()
        df_synthese.to_excel(writer, sheet_name='Synthese', index=False)

        # FEUILLE 2: OPP (Opportunités)
        print("  - Feuille 'Opp'...")
        colonnes_opp = ['Zone', 'Score_Opp_0_10', 'Score_Opp']
        colonnes_opp_indicateurs = ['Opp1', 'Opp2', 'Opp3', 'Opp4']
        colonnes_opp += [col for col in colonnes_opp_indicateurs if col in df.columns]
        df_opp = df[colonnes_opp].copy()
        df_opp.to_excel(writer, sheet_name='Opp', index=False)

        # FEUILLE 3: CHO (Choix)
        print("  - Feuille 'Cho'...")
        colonnes_cho = ['Zone', 'Score_Cho_0_10', 'Score_Cho']
        colonnes_cho_indicateurs = ['Cho1', 'Cho2']
        colonnes_cho += [col for col in colonnes_cho_indicateurs if col in df.columns]
        df_cho = df[colonnes_cho].copy()
        df_cho.to_excel(writer, sheet_name='Cho', index=False)

        # FEUILLE 4: VEC (Vécu)
        print("  - Feuille 'Vec'...")
        colonnes_vec = ['Zone', 'Score_Vec_0_10', 'Score_Vec']
        colonnes_vec_indicateurs = ['Vec1', 'Vec2', 'Vec3', 'Vec4']
        colonnes_vec += [col for col in colonnes_vec_indicateurs if col in df.columns]
        df_vec = df[colonnes_vec].copy()
        df_vec.to_excel(writer, sheet_name='Vec', index=False)

    print(f"\nFichier cree avec succes: {fichier_sortie}")

    # Générer les statistiques descriptives normalisées
    print("\n" + "=" * 80)
    print("GENERATION DES STATISTIQUES DESCRIPTIVES")
    print("=" * 80)

    stats = df['OppChoVec_0_10'].describe()
    df_stats = pd.DataFrame(stats)
    df_stats.columns = ['OppChoVec_0_10']

    fichier_stats = 'stats_descriptives_normalisées_V.xlsx'
    df_stats.to_excel(fichier_stats)
    print(f"  [OK] {fichier_stats}")

    # Afficher des statistiques
    print("\n" + "=" * 80)
    print("STATISTIQUES DES SCORES NORMALISES (echelle 0-10)")
    print("=" * 80)

    for col in ['OppChoVec_0_10', 'Score_Opp_0_10', 'Score_Cho_0_10', 'Score_Vec_0_10']:
        print(f"\n{col}:")
        print(f"  Min:    {df[col].min():.2f}")
        print(f"  Max:    {df[col].max():.2f}")
        print(f"  Moyenne: {df[col].mean():.2f}")
        print(f"  Mediane: {df[col].median():.2f}")

    # Top 10
    print("\n" + "=" * 80)
    print("TOP 10 COMMUNES (OppChoVec normalise 0-10)")
    print("=" * 80)
    df_sorted = df.sort_values('OppChoVec_0_10', ascending=False)
    for idx in range(min(10, len(df_sorted))):
        row = df_sorted.iloc[idx]
        print(f"{idx+1:2d}. {row['Zone']:30s} - OppChoVec: {row['OppChoVec_0_10']:5.2f}/10")

    return df


def main():
    """
    Point d'entree principal
    """
    fichier_entree = "oppchovec_resultats_V.xlsx"
    fichier_sortie = "oppchovec_resultats_V_0_10.xlsx"

    df = creer_excel_multi_feuilles(fichier_entree, fichier_sortie)

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)

    return df


if __name__ == "__main__":
    df = main()
