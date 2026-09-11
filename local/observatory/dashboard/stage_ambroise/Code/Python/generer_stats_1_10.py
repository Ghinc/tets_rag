"""
Script pour générer les statistiques descriptives normalisées entre 1 et 10
"""

import pandas as pd


def normaliser_1_10(serie):
    """
    Normalise une série entre 1 et 10

    Args:
        serie: pandas Series à normaliser

    Returns:
        pandas Series normalisée entre 1 et 10
    """
    min_val = serie.min()
    max_val = serie.max()

    if max_val == min_val:
        return pd.Series([5.5] * len(serie), index=serie.index)

    # Normalisation: (x - min) / (max - min) * 9 + 1
    # Pour avoir des valeurs entre 1 et 10
    return ((serie - min_val) / (max_val - min_val)) * 9 + 1


def main():
    """
    Point d'entrée principal
    """
    print("=" * 80)
    print("GENERATION DES STATISTIQUES DESCRIPTIVES (échelle 1-10)")
    print("=" * 80)
    print()

    # Charger les données
    fichier_entree = "oppchovec_resultats_V.xlsx"
    print(f"Chargement de {fichier_entree}...")
    df = pd.read_excel(fichier_entree)

    # Normaliser entre 1 et 10
    print("Normalisation entre 1 et 10...")
    df['OppChoVec_1_10'] = normaliser_1_10(df['OppChoVec'])
    df['Score_Opp_1_10'] = normaliser_1_10(df['Score_Opp'])
    df['Score_Cho_1_10'] = normaliser_1_10(df['Score_Cho'])
    df['Score_Vec_1_10'] = normaliser_1_10(df['Score_Vec'])

    # Calculer les statistiques descriptives
    print("\nCalcul des statistiques descriptives...")
    stats = df['OppChoVec_1_10'].describe()
    df_stats = pd.DataFrame(stats)
    df_stats.columns = ['OppChoVec_1_10']

    # Sauvegarder les statistiques
    fichier_stats = 'stats_descriptives_normalisées_1_10_V.xlsx'
    df_stats.to_excel(fichier_stats)
    print(f"  [OK] {fichier_stats}")

    # Afficher les statistiques à l'écran
    print("\n" + "=" * 80)
    print("STATISTIQUES DES SCORES NORMALISES (échelle 1-10)")
    print("=" * 80)

    print(f"\nOppChoVec_1_10:")
    print(f"  Min:     {df['OppChoVec_1_10'].min():.2f}")
    print(f"  Max:     {df['OppChoVec_1_10'].max():.2f}")
    print(f"  Moyenne: {df['OppChoVec_1_10'].mean():.2f}")
    print(f"  Mediane: {df['OppChoVec_1_10'].median():.2f}")
    print(f"  Ecart-type: {df['OppChoVec_1_10'].std():.2f}")
    print(f"  Q1 (25%): {df['OppChoVec_1_10'].quantile(0.25):.2f}")
    print(f"  Q3 (75%): {df['OppChoVec_1_10'].quantile(0.75):.2f}")

    print(f"\nScore_Opp_1_10:")
    print(f"  Min:     {df['Score_Opp_1_10'].min():.2f}")
    print(f"  Max:     {df['Score_Opp_1_10'].max():.2f}")
    print(f"  Moyenne: {df['Score_Opp_1_10'].mean():.2f}")
    print(f"  Mediane: {df['Score_Opp_1_10'].median():.2f}")

    print(f"\nScore_Cho_1_10:")
    print(f"  Min:     {df['Score_Cho_1_10'].min():.2f}")
    print(f"  Max:     {df['Score_Cho_1_10'].max():.2f}")
    print(f"  Moyenne: {df['Score_Cho_1_10'].mean():.2f}")
    print(f"  Mediane: {df['Score_Cho_1_10'].median():.2f}")

    print(f"\nScore_Vec_1_10:")
    print(f"  Min:     {df['Score_Vec_1_10'].min():.2f}")
    print(f"  Max:     {df['Score_Vec_1_10'].max():.2f}")
    print(f"  Moyenne: {df['Score_Vec_1_10'].mean():.2f}")
    print(f"  Mediane: {df['Score_Vec_1_10'].median():.2f}")

    # Top 10
    print("\n" + "=" * 80)
    print("TOP 10 COMMUNES (OppChoVec normalise 1-10)")
    print("=" * 80)
    df_sorted = df.sort_values('OppChoVec_1_10', ascending=False)
    for idx in range(min(10, len(df_sorted))):
        row = df_sorted.iloc[idx]
        print(f"{idx+1:2d}. {row['Zone']:30s} - OppChoVec: {row['OppChoVec_1_10']:5.2f}/10")

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)


if __name__ == "__main__":
    main()
