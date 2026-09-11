"""
Script pour régénérer toute l'application avec normalisation 1-10
- Normalisation des scores 1-10
- Calcul des seuils Jenks
- Génération des données pour les cartes
- Calcul LISA 1% et 5%
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


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
    return ((serie - min_val) / (max_val - min_val)) * 9 + 1


def jenks_breaks(values, n_classes=5):
    """
    Calcule les seuils de Jenks (Natural Breaks)

    Args:
        values: array des valeurs
        n_classes: nombre de classes

    Returns:
        Liste des seuils
    """
    values = np.array(sorted(values))
    n = len(values)

    if n <= n_classes:
        return list(values)

    # Matrices pour la programmation dynamique
    mat1 = np.zeros((n + 1, n_classes + 1))
    mat2 = np.zeros((n + 1, n_classes + 1))

    # Initialisation
    for i in range(1, n_classes + 1):
        mat1[1, i] = 1
        mat2[1, i] = 0
        for j in range(2, n + 1):
            mat2[j, i] = float('inf')

    # Calcul
    for l in range(2, n + 1):
        sum_val = 0
        sum_sq = 0
        w = 0

        for m in range(1, l + 1):
            i3 = l - m + 1
            val = values[i3 - 1]

            sum_val += val
            sum_sq += val * val
            w += 1

            variance = sum_sq - (sum_val * sum_val) / w

            if i3 != 1:
                for j in range(2, n_classes + 1):
                    if mat2[l, j] >= variance + mat2[i3 - 1, j - 1]:
                        mat1[l, j] = i3
                        mat2[l, j] = variance + mat2[i3 - 1, j - 1]

        mat1[l, 1] = 1
        mat2[l, 1] = variance

    # Extraction des seuils
    breaks = []
    k = n
    for j in range(n_classes, 0, -1):
        breaks.insert(0, values[int(mat1[k, j]) - 1])
        k = int(mat1[k, j]) - 1

    return breaks


def main():
    """
    Point d'entrée principal
    """
    print("=" * 80)
    print("REGENERATION COMPLETE DE L'APPLICATION (NORMALISATION 1-10)")
    print("=" * 80)
    print()

    # ========================================================================
    # ETAPE 1: CHARGER LES DONNEES
    # ========================================================================
    print("ETAPE 1: Chargement des données")
    print("-" * 80)

    fichier_resultats = "oppchovec_resultats_V.xlsx"
    fichier_indicateurs = "df_indicateur.xlsx"

    print(f"  Chargement de {fichier_resultats}...")
    df_resultats = pd.read_excel(fichier_resultats)
    print(f"    {len(df_resultats)} communes chargées")

    print(f"  Chargement de {fichier_indicateurs}...")
    df_indicateurs = pd.read_excel(fichier_indicateurs, index_col=0)
    print(f"    {len(df_indicateurs)} communes chargées")

    # ========================================================================
    # ETAPE 2: NORMALISER LES SCORES 1-10
    # ========================================================================
    print("\nETAPE 2: Normalisation des scores (1-10)")
    print("-" * 80)

    df_resultats['OppChoVec_1_10'] = normaliser_1_10(df_resultats['OppChoVec'])
    df_resultats['Score_Opp_1_10'] = normaliser_1_10(df_resultats['Score_Opp'])
    df_resultats['Score_Cho_1_10'] = normaliser_1_10(df_resultats['Score_Cho'])
    df_resultats['Score_Vec_1_10'] = normaliser_1_10(df_resultats['Score_Vec'])

    print("  [OK] Scores normalisés 1-10")
    print(f"    OppChoVec: {df_resultats['OppChoVec_1_10'].min():.2f} - {df_resultats['OppChoVec_1_10'].max():.2f}")
    print(f"    Opp:       {df_resultats['Score_Opp_1_10'].min():.2f} - {df_resultats['Score_Opp_1_10'].max():.2f}")
    print(f"    Cho:       {df_resultats['Score_Cho_1_10'].min():.2f} - {df_resultats['Score_Cho_1_10'].max():.2f}")
    print(f"    Vec:       {df_resultats['Score_Vec_1_10'].min():.2f} - {df_resultats['Score_Vec_1_10'].max():.2f}")

    # ========================================================================
    # ETAPE 3: CALCULER LES SEUILS JENKS
    # ========================================================================
    print("\nETAPE 3: Calcul des seuils Jenks (5 classes)")
    print("-" * 80)

    seuils_jenks = {}

    for col in ['OppChoVec_1_10', 'Score_Opp_1_10', 'Score_Cho_1_10', 'Score_Vec_1_10']:
        values = df_resultats[col].dropna().values
        breaks = jenks_breaks(values, n_classes=5)
        seuils_jenks[col] = breaks
        print(f"  {col}:")
        print(f"    Seuils: {[f'{b:.2f}' for b in breaks]}")

    # Sauvegarder les seuils Jenks
    fichier_jenks = 'seuils_jenks_1_10.json'
    with open(fichier_jenks, 'w', encoding='utf-8') as f:
        json.dump(seuils_jenks, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] {fichier_jenks}")

    # ========================================================================
    # ETAPE 4: GENERER LES DONNEES POUR LES CARTES
    # ========================================================================
    print("\nETAPE 4: Génération des données pour les cartes")
    print("-" * 80)

    # Fusionner les résultats avec les indicateurs
    df_complet = df_resultats.copy()
    df_complet.set_index('Zone', inplace=True)

    # Ajouter les indicateurs
    for col in df_indicateurs.columns:
        if col not in df_complet.columns:
            df_complet[col] = df_indicateurs[col]

    # Créer le dictionnaire de données
    data_dict = df_complet.to_dict(orient='index')

    # Sauvegarder les données
    fichier_data = 'data_scores_1_10.json'
    with open(fichier_data, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {fichier_data}")

    # ========================================================================
    # ETAPE 5: PREPARER LES DONNEES POUR LISA
    # ========================================================================
    print("\nETAPE 5: Préparation des données pour LISA")
    print("-" * 80)

    # Créer un mapping code_commune -> nom_commune
    fichier_mapping = "../../Données/Corse_Commune/mapping_communes.csv"
    df_mapping = pd.read_csv(fichier_mapping)
    nom_to_code = dict(zip(df_mapping['nom_commune'], df_mapping['code_commune']))

    # Préparer les données avec les codes communes
    df_lisa = df_resultats[['Zone', 'OppChoVec_1_10', 'Score_Opp_1_10',
                             'Score_Cho_1_10', 'Score_Vec_1_10']].copy()
    df_lisa['code_commune'] = df_lisa['Zone'].map(nom_to_code)

    # Sauvegarder le fichier de données pour LISA
    fichier_lisa_data = 'data_pour_lisa_1_10.csv'
    df_lisa.to_csv(fichier_lisa_data, index=False)
    print(f"  [OK] {fichier_lisa_data}")
    print(f"    {len(df_lisa)} communes préparées")
    print(f"    Colonnes: {list(df_lisa.columns)}")

    # ========================================================================
    # ETAPE 6: EXPORTER LES FICHIERS POUR LE WEB
    # ========================================================================
    print("\nETAPE 6: Export vers le dossier WEB")
    print("-" * 80)

    web_dir = Path("../WEB")
    if not web_dir.exists():
        print("  ATTENTION: Le dossier ../WEB n'existe pas")
    else:
        # Copier les fichiers
        import shutil

        shutil.copy(fichier_data, web_dir / 'data_scores_1_10.json')
        print(f"  [OK] data_scores_1_10.json -> WEB/")

        shutil.copy(fichier_jenks, web_dir / 'seuils_jenks_1_10.json')
        print(f"  [OK] seuils_jenks_1_10.json -> WEB/")

    # ========================================================================
    # ETAPE 7: GENERER LE FICHIER EXCEL RECAPITULATIF
    # ========================================================================
    print("\nETAPE 7: Génération du fichier Excel récapitulatif")
    print("-" * 80)

    fichier_excel = 'oppchovec_1_10_complet.xlsx'
    with pd.ExcelWriter(fichier_excel, engine='openpyxl') as writer:
        # Feuille 1: Synthèse
        df_synthese = df_resultats[['Zone', 'OppChoVec_1_10', 'Score_Opp_1_10',
                                     'Score_Cho_1_10', 'Score_Vec_1_10']].copy()
        df_synthese.to_excel(writer, sheet_name='Synthese', index=False)

        # Feuille 2: OppChoVec détaillé
        df_opp_detail = df_resultats[['Zone', 'OppChoVec_1_10', 'OppChoVec',
                                       'Score_Opp_1_10', 'Score_Opp',
                                       'Score_Cho_1_10', 'Score_Cho',
                                       'Score_Vec_1_10', 'Score_Vec']].copy()
        df_opp_detail.to_excel(writer, sheet_name='OppChoVec', index=False)

    print(f"  [OK] {fichier_excel}")

    # ========================================================================
    # RESUME FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUME FINAL")
    print("=" * 80)
    print("\nFichiers générés:")
    print(f"  1. {fichier_data} - Données complètes pour les cartes")
    print(f"  2. {fichier_jenks} - Seuils Jenks pour les cartes")
    print(f"  3. {fichier_lisa_data} - Données pour analyses LISA")
    print(f"  4. {fichier_excel} - Fichier Excel récapitulatif")

    print("\nTop 10 communes (normalisation 1-10):")
    top10 = df_resultats.nlargest(10, 'OppChoVec_1_10')
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"  {i:2d}. {row['Zone']:30s} - {row['OppChoVec_1_10']:.2f}/10")

    print("\nProchaines étapes:")
    print("  1. Exécuter lisa_normalisation_elementaire_1pct.py avec data_pour_lisa_1_10.csv")
    print("  2. Modifier le script pour utiliser les colonnes *_1_10")
    print("  3. Générer LISA 1% et 5%")
    print("  4. Copier les résultats vers le dossier WEB")

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)


if __name__ == "__main__":
    main()
