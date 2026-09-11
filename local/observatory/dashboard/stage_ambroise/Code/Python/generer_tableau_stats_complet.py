"""
Script pour générer un tableau complet de statistiques descriptives
avec normalisation 0-10 et 1-10
"""

import pandas as pd
import numpy as np


def calculer_gini(serie):
    """
    Calcule le coefficient de Gini

    Args:
        serie: pandas Series

    Returns:
        Coefficient de Gini (entre 0 et 1)
    """
    # Supprimer les valeurs NaN
    serie_clean = serie.dropna()

    if len(serie_clean) == 0:
        return np.nan

    # Trier les valeurs
    sorted_values = np.sort(serie_clean.values)
    n = len(sorted_values)

    # Calculer le coefficient de Gini
    # Formule: (2 * Σ(i * x_i)) / (n * Σx_i) - (n + 1) / n
    cumsum = np.cumsum(sorted_values)
    sum_values = cumsum[-1]

    if sum_values == 0:
        return 0

    # Index commence à 1 pour la formule
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * sum_values) - (n + 1) / n

    return gini


def normaliser_0_10(serie):
    """Normalise une série entre 0 et 10"""
    min_val = serie.min()
    max_val = serie.max()

    if max_val == min_val:
        return pd.Series([5.0] * len(serie), index=serie.index)

    return ((serie - min_val) / (max_val - min_val)) * 10


def normaliser_1_10(serie):
    """Normalise une série entre 1 et 10"""
    min_val = serie.min()
    max_val = serie.max()

    if max_val == min_val:
        return pd.Series([5.5] * len(serie), index=serie.index)

    return ((serie - min_val) / (max_val - min_val)) * 9 + 1


def calculer_statistiques(serie, nom):
    """
    Calcule toutes les statistiques pour une série

    Args:
        serie: pandas Series
        nom: nom de la série

    Returns:
        Dictionnaire avec toutes les statistiques
    """
    return {
        'Indicateur': nom,
        'Min.': serie.min(),
        '1er quartile': serie.quantile(0.25),
        'Moyenne': serie.mean(),
        'Médiane': serie.median(),
        '3e quartile': serie.quantile(0.75),
        'Max.': serie.max(),
        'Ecart-type': serie.std(),
        'Coefficient de Gini': calculer_gini(serie)
    }


def main():
    """
    Point d'entrée principal
    """
    print("=" * 80)
    print("GENERATION DU TABLEAU DE STATISTIQUES COMPLET")
    print("=" * 80)
    print()

    # Charger les données
    fichier_entree = "oppchovec_resultats_V.xlsx"
    print(f"Chargement de {fichier_entree}...")
    df = pd.read_excel(fichier_entree)
    print(f"  {len(df)} communes chargées")

    # Préparer les listes pour stocker les statistiques
    stats_0_10 = []
    stats_1_10 = []

    # ====================================================================
    # NORMALISATION 0-10
    # ====================================================================
    print("\nCalcul des statistiques avec normalisation 0-10...")

    # OppChoVec
    oppchovec_norm = normaliser_0_10(df['OppChoVec'])
    stats_0_10.append(calculer_statistiques(df['OppChoVec'], 'OppChoVec'))
    stats_0_10.append(calculer_statistiques(oppchovec_norm, 'OppChoVec normalisé'))

    # Opp
    opp_norm = normaliser_0_10(df['Score_Opp'])
    stats_0_10.append(calculer_statistiques(df['Score_Opp'], 'Opp'))
    stats_0_10.append(calculer_statistiques(opp_norm, 'Opp normalisé'))

    # Cho
    cho_norm = normaliser_0_10(df['Score_Cho'])
    stats_0_10.append(calculer_statistiques(df['Score_Cho'], 'Cho'))
    stats_0_10.append(calculer_statistiques(cho_norm, 'Cho normalisé'))

    # Vec
    vec_norm = normaliser_0_10(df['Score_Vec'])
    stats_0_10.append(calculer_statistiques(df['Score_Vec'], 'Vec'))
    stats_0_10.append(calculer_statistiques(vec_norm, 'Vec normalisé'))

    # ====================================================================
    # NORMALISATION 1-10
    # ====================================================================
    print("Calcul des statistiques avec normalisation 1-10...")

    # OppChoVec
    oppchovec_norm = normaliser_1_10(df['OppChoVec'])
    stats_1_10.append(calculer_statistiques(df['OppChoVec'], 'OppChoVec'))
    stats_1_10.append(calculer_statistiques(oppchovec_norm, 'OppChoVec normalisé'))

    # Opp
    opp_norm = normaliser_1_10(df['Score_Opp'])
    stats_1_10.append(calculer_statistiques(df['Score_Opp'], 'Opp'))
    stats_1_10.append(calculer_statistiques(opp_norm, 'Opp normalisé'))

    # Cho
    cho_norm = normaliser_1_10(df['Score_Cho'])
    stats_1_10.append(calculer_statistiques(df['Score_Cho'], 'Cho'))
    stats_1_10.append(calculer_statistiques(cho_norm, 'Cho normalisé'))

    # Vec
    vec_norm = normaliser_1_10(df['Score_Vec'])
    stats_1_10.append(calculer_statistiques(df['Score_Vec'], 'Vec'))
    stats_1_10.append(calculer_statistiques(vec_norm, 'Vec normalisé'))

    # ====================================================================
    # CREATION DU FICHIER EXCEL
    # ====================================================================
    print("\nCréation du fichier Excel...")

    # Créer les DataFrames
    df_stats_0_10 = pd.DataFrame(stats_0_10)
    df_stats_1_10 = pd.DataFrame(stats_1_10)

    # Créer le fichier Excel avec 2 feuilles
    fichier_sortie = 'tableau_statistiques_complet.xlsx'
    with pd.ExcelWriter(fichier_sortie, engine='openpyxl') as writer:
        df_stats_0_10.to_excel(writer, sheet_name='Normalisation 0-10', index=False)
        df_stats_1_10.to_excel(writer, sheet_name='Normalisation 1-10', index=False)

    print(f"  [OK] {fichier_sortie}")

    # ====================================================================
    # AFFICHAGE DES RESULTATS
    # ====================================================================
    print("\n" + "=" * 80)
    print("STATISTIQUES - NORMALISATION 0-10")
    print("=" * 80)
    print(df_stats_0_10.to_string(index=False))

    print("\n" + "=" * 80)
    print("STATISTIQUES - NORMALISATION 1-10")
    print("=" * 80)
    print(df_stats_1_10.to_string(index=False))

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)
    print(f"\nFichier généré: {fichier_sortie}")
    print("  - Feuille 1: Normalisation 0-10")
    print("  - Feuille 2: Normalisation 1-10")


if __name__ == "__main__":
    main()
