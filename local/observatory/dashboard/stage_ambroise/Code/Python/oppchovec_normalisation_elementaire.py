"""
Calcul OppChoVec avec normalisation des indicateurs élémentaires (0-1)
puis normalisation finale de l'indice (1-10)

Méthode :
1. Normaliser Opp1-4, Cho1-2, Vec1-4 entre 0 et 1
2. Calculer Score_Opp, Score_Cho, Score_Vec (moyennes pondérées)
3. Calculer OppChoVec avec formule classique
4. Normaliser OppChoVec entre 1 et 10
"""

import pandas as pd
import numpy as np
from oppchovec import ALPHA, BETA, PONDERATIONS, POND_FINALE


def normaliser_0_1(serie):
    """
    Normalise une série entre 0 et 1
    """
    min_val = serie.min()
    max_val = serie.max()

    if max_val == min_val:
        return pd.Series([0.5] * len(serie), index=serie.index)

    return (serie - min_val) / (max_val - min_val)


def normaliser_1_10(serie):
    """
    Normalise une série entre 1 et 10
    """
    min_val = serie.min()
    max_val = serie.max()

    if max_val == min_val:
        return pd.Series([5.5] * len(serie), index=serie.index)

    return ((serie - min_val) / (max_val - min_val)) * 9 + 1


def calc_dik(indicateurs_normalises, ponderations):
    """
    Calcule le score d'une dimension (moyenne pondérée)

    Args:
        indicateurs_normalises: Dict des indicateurs normalisés {nom: valeur}
        ponderations: Dict des pondérations {nom: poids}

    Returns:
        Score de la dimension
    """
    valeurs = np.array([indicateurs_normalises.get(k, 0) for k in ponderations.keys()])
    poids = np.array(list(ponderations.values()))

    if poids.sum() == 0:
        return 0

    return (valeurs * poids).sum() / poids.sum()


def calc_oppchovec(dik, pk_values):
    """
    Calcule l'indice OppChoVec final
    Formule: (1/3) * [Σ(pk × dik^β)]^(α/β)
    """
    somme_ponderee = (pk_values * (dik ** BETA)).sum()
    oppchovec = (1/3) * (somme_ponderee ** (ALPHA / BETA))
    return oppchovec


def calculer_oppchovec_normalisation_elementaire():
    """
    Calcule OppChoVec avec normalisation élémentaire
    """
    print("=" * 80)
    print("CALCUL OPPCHOVEC - NORMALISATION ELEMENTAIRE")
    print("=" * 80)
    print()

    # Charger les données depuis oppchovec_0_norm.xlsx (feuille Opp pour avoir tous les indicateurs)
    print("Chargement des donnees...")
    df_opp = pd.read_excel('oppchovec_0_norm.xlsx', sheet_name='Opp')
    df_cho = pd.read_excel('oppchovec_0_norm.xlsx', sheet_name='Cho')
    df_vec = pd.read_excel('oppchovec_0_norm.xlsx', sheet_name='Vec')

    # Fusionner toutes les données
    df = df_opp[['Zone', 'Opp1', 'Opp2', 'Opp3', 'Opp4']].copy()
    df = df.merge(df_cho[['Zone', 'Cho1', 'Cho2']], on='Zone')
    df = df.merge(df_vec[['Zone', 'Vec1', 'Vec2', 'Vec3', 'Vec4']], on='Zone')

    print(f"  {len(df)} communes chargees")

    # Étape 1: Normaliser chaque indicateur élémentaire entre 0 et 1
    print("\nNormalisation des indicateurs elementaires entre 0 et 1...")

    indicateurs = ['Opp1', 'Opp2', 'Opp3', 'Opp4', 'Cho1', 'Cho2', 'Vec1', 'Vec2', 'Vec3', 'Vec4']

    df_normalise = df.copy()
    for ind in indicateurs:
        if ind in df.columns:
            df_normalise[f'{ind}_norm'] = normaliser_0_1(df[ind])
            print(f"  - {ind}: [{df[ind].min():.2f}, {df[ind].max():.2f}] -> [0, 1]")

    # Étape 2: Calculer les scores des dimensions avec indicateurs normalisés
    print("\nCalcul des scores des dimensions...")

    scores_opp = []
    scores_cho = []
    scores_vec = []

    for idx, row in df_normalise.iterrows():
        # Score Opp
        opp_vals = {k: row[f'{k}_norm'] for k in ['Opp1', 'Opp2', 'Opp3', 'Opp4']
                    if f'{k}_norm' in row.index}
        score_opp = calc_dik(opp_vals, PONDERATIONS['Opp'])
        scores_opp.append(score_opp)

        # Score Cho
        cho_vals = {k: row[f'{k}_norm'] for k in ['Cho1', 'Cho2']
                    if f'{k}_norm' in row.index}
        score_cho = calc_dik(cho_vals, PONDERATIONS['Cho'])
        scores_cho.append(score_cho)

        # Score Vec
        vec_vals = {k: row[f'{k}_norm'] for k in ['Vec1', 'Vec2', 'Vec3', 'Vec4']
                    if f'{k}_norm' in row.index}
        score_vec = calc_dik(vec_vals, PONDERATIONS['Vec'])
        scores_vec.append(score_vec)

    df_normalise['Score_Opp'] = scores_opp
    df_normalise['Score_Cho'] = scores_cho
    df_normalise['Score_Vec'] = scores_vec

    print(f"  Score_Opp: [{min(scores_opp):.4f}, {max(scores_opp):.4f}]")
    print(f"  Score_Cho: [{min(scores_cho):.4f}, {max(scores_cho):.4f}]")
    print(f"  Score_Vec: [{min(scores_vec):.4f}, {max(scores_vec):.4f}]")

    # Étape 3: Calculer OppChoVec
    print("\nCalcul de l'indice OppChoVec...")

    oppchovec_values = []
    for idx, row in df_normalise.iterrows():
        dik = np.array([row['Score_Opp'], row['Score_Cho'], row['Score_Vec']])
        pk = np.array(POND_FINALE)
        oppchovec = calc_oppchovec(dik, pk)
        oppchovec_values.append(oppchovec)

    df_normalise['OppChoVec'] = oppchovec_values

    print(f"  OppChoVec: [{min(oppchovec_values):.4f}, {max(oppchovec_values):.4f}]")

    # Étape 4: Normaliser OppChoVec entre 1 et 10
    print("\nNormalisation finale de OppChoVec entre 1 et 10...")
    df_normalise['OppChoVec_1_10'] = normaliser_1_10(df_normalise['OppChoVec'])

    # Normaliser aussi les scores des dimensions entre 1 et 10
    df_normalise['Score_Opp_1_10'] = normaliser_1_10(df_normalise['Score_Opp'])
    df_normalise['Score_Cho_1_10'] = normaliser_1_10(df_normalise['Score_Cho'])
    df_normalise['Score_Vec_1_10'] = normaliser_1_10(df_normalise['Score_Vec'])

    # Trier par OppChoVec décroissant
    df_normalise = df_normalise.sort_values('OppChoVec_1_10', ascending=False).reset_index(drop=True)

    print("\nStatistiques finales:")
    print(f"  Nombre de communes: {len(df_normalise)}")
    print(f"  OppChoVec (1-10) moyen: {df_normalise['OppChoVec_1_10'].mean():.2f}")
    print(f"  OppChoVec (1-10) min: {df_normalise['OppChoVec_1_10'].min():.2f} ({df_normalise.loc[df_normalise['OppChoVec_1_10'].idxmin(), 'Zone']})")
    print(f"  OppChoVec (1-10) max: {df_normalise['OppChoVec_1_10'].max():.2f} ({df_normalise.loc[df_normalise['OppChoVec_1_10'].idxmax(), 'Zone']})")

    return df_normalise


def exporter_excel(df, fichier='oppchovec_normalisation_elementaire.xlsx'):
    """
    Exporte vers Excel avec plusieurs feuilles
    Chaque feuille est triée par son indicateur principal en ordre décroissant
    """
    print(f"\nExport vers {fichier}...")

    with pd.ExcelWriter(fichier, engine='openpyxl') as writer:
        # Feuille Synthèse - Tri par OppChoVec_1_10 décroissant
        colonnes_synthese = [
            'Zone',
            'OppChoVec_1_10', 'OppChoVec',
            'Score_Opp_1_10', 'Score_Opp',
            'Score_Cho_1_10', 'Score_Cho',
            'Score_Vec_1_10', 'Score_Vec'
        ]
        df_synthese = df[colonnes_synthese].copy()
        df_synthese = df_synthese.sort_values('OppChoVec_1_10', ascending=False)
        df_synthese.to_excel(writer, sheet_name='Synthese', index=False)
        print("  - Feuille 'Synthese' (tri par OppChoVec)")

        # Feuille Opp - Tri par Score_Opp_1_10 décroissant
        colonnes_opp = ['Zone', 'Score_Opp_1_10', 'Score_Opp',
                        'Opp1', 'Opp1_norm', 'Opp2', 'Opp2_norm',
                        'Opp3', 'Opp3_norm', 'Opp4', 'Opp4_norm']
        df_opp = df[colonnes_opp].copy()
        df_opp = df_opp.sort_values('Score_Opp_1_10', ascending=False)
        df_opp.to_excel(writer, sheet_name='Opp', index=False)
        print("  - Feuille 'Opp' (tri par Score_Opp)")

        # Feuille Cho - Tri par Score_Cho_1_10 décroissant
        colonnes_cho = ['Zone', 'Score_Cho_1_10', 'Score_Cho',
                        'Cho1', 'Cho1_norm', 'Cho2', 'Cho2_norm']
        df_cho = df[colonnes_cho].copy()
        df_cho = df_cho.sort_values('Score_Cho_1_10', ascending=False)
        df_cho.to_excel(writer, sheet_name='Cho', index=False)
        print("  - Feuille 'Cho' (tri par Score_Cho)")

        # Feuille Vec - Tri par Score_Vec_1_10 décroissant
        colonnes_vec = ['Zone', 'Score_Vec_1_10', 'Score_Vec',
                        'Vec1', 'Vec1_norm', 'Vec2', 'Vec2_norm',
                        'Vec3', 'Vec3_norm', 'Vec4', 'Vec4_norm']
        df_vec = df[colonnes_vec].copy()
        df_vec = df_vec.sort_values('Score_Vec_1_10', ascending=False)
        df_vec.to_excel(writer, sheet_name='Vec', index=False)
        print("  - Feuille 'Vec' (tri par Score_Vec)")

    print(f"  Fichier cree: {fichier}")

    # Afficher le top 10
    print("\n" + "=" * 80)
    print("TOP 10 COMMUNES (OppChoVec avec normalisation elementaire)")
    print("=" * 80)
    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        print(f"{idx+1:2d}. {row['Zone']:30s} - OppChoVec: {row['OppChoVec_1_10']:5.2f}/10")


def main():
    """
    Point d'entrée principal
    """
    # Calculer avec normalisation élémentaire
    df_resultats = calculer_oppchovec_normalisation_elementaire()

    # Exporter vers Excel
    exporter_excel(df_resultats)

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)

    return df_resultats


if __name__ == "__main__":
    df = main()
