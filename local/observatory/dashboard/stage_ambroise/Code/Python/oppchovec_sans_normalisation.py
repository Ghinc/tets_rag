"""
Calcul de l'indice OppChoVec SANS normalisation des indicateurs bruts

Ce script calcule l'indice OppChoVec en utilisant directement les valeurs brutes
des indicateurs (Opp1-4, Cho1-2, Vec1-4) sans les normaliser entre 0 et 1.
"""

import pandas as pd
import numpy as np
from oppchovec import (
    ALPHA, BETA, PONDERATIONS, POND_FINALE,
    charger_donnees_commune, charger_toutes_communes, charger_mapping_communes,
    calculer_indicateurs
)


def calc_dik_sans_normalisation(v_ijk_dict, p_jk_dict):
    """
    Calcule le score d'une dimension SANS normalisation préalable
    (utilise directement les valeurs brutes)
    """
    valeurs = np.array([v_ijk_dict.get(k, 0) for k in p_jk_dict.keys()])
    poids = np.array(list(p_jk_dict.values()))

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


def calculer_oppchovec_sans_normalisation():
    """
    Calcule l'indice OppChoVec SANS normaliser les indicateurs bruts
    """
    print("=" * 80)
    print("CALCUL OPPCHOVEC SANS NORMALISATION DES INDICATEURS")
    print("=" * 80)
    print()

    print("Chargement de la liste des communes...")
    communes = charger_toutes_communes()
    print(f"Traitement de {len(communes)} communes...")

    # Étape 1: Charger et calculer les indicateurs bruts
    print("Calcul des indicateurs bruts...")
    data_indicateurs = {}

    for zone in communes:
        donnees = charger_donnees_commune(zone)
        indicateurs = calculer_indicateurs(donnees)
        data_indicateurs[zone] = indicateurs

    # Convertir en DataFrame
    df_indicateurs = pd.DataFrame.from_dict(data_indicateurs, orient='index')
    df_indicateurs.index.name = 'Zone'

    # Mapper les codes INSEE vers les noms de communes
    print("Conversion des codes INSEE vers les noms de communes...")
    mapping = charger_mapping_communes()
    df_indicateurs.index = df_indicateurs.index.map(lambda x: mapping.get(x, x))

    # Étape 2: SKIP LA NORMALISATION
    print("Calcul des dimensions SANS normalisation préalable...")

    # Étape 3: Calculer les scores avec les valeurs BRUTES
    resultats = []

    for zone in df_indicateurs.index:
        indicateurs_bruts = df_indicateurs.loc[zone]

        # Score Opp (avec valeurs brutes)
        opp_vals = {}
        for k in PONDERATIONS['Opp'].keys():
            if k in indicateurs_bruts.index and not pd.isna(indicateurs_bruts[k]):
                opp_vals[k] = indicateurs_bruts[k]
        score_opp = calc_dik_sans_normalisation(opp_vals, PONDERATIONS['Opp']) if opp_vals else 0

        # Score Cho (avec valeurs brutes)
        cho_vals = {}
        for k in PONDERATIONS['Cho'].keys():
            if k in indicateurs_bruts.index and not pd.isna(indicateurs_bruts[k]):
                cho_vals[k] = indicateurs_bruts[k]
        score_cho = calc_dik_sans_normalisation(cho_vals, PONDERATIONS['Cho']) if cho_vals else 0

        # Score Vec (avec valeurs brutes)
        vec_vals = {}
        for k in PONDERATIONS['Vec'].keys():
            if k in indicateurs_bruts.index and not pd.isna(indicateurs_bruts[k]):
                vec_vals[k] = indicateurs_bruts[k]
        score_vec = calc_dik_sans_normalisation(vec_vals, PONDERATIONS['Vec']) if vec_vals else 0

        # Calculer OppChoVec
        dik = np.array([score_opp, score_cho, score_vec])
        pk = np.array(POND_FINALE)
        oppchovec = calc_oppchovec(dik, pk)

        # Construire la ligne de résultat
        resultat = {
            'Zone': zone,
            'OppChoVec': oppchovec,
            'Score_Opp': score_opp,
            'Score_Cho': score_cho,
            'Score_Vec': score_vec,
        }

        # Ajouter tous les indicateurs bruts disponibles
        for ind in df_indicateurs.columns:
            resultat[ind] = indicateurs_bruts[ind] if ind in indicateurs_bruts.index else np.nan

        resultats.append(resultat)

    # Créer le DataFrame final
    df_resultats = pd.DataFrame(resultats)

    # Trier par OppChoVec décroissant
    df_resultats = df_resultats.sort_values('OppChoVec', ascending=False).reset_index(drop=True)

    print("Calcul terminé!")
    print(f"\nNombre de communes: {len(df_resultats)}")
    print(f"OppChoVec moyen: {df_resultats['OppChoVec'].mean():.4f}")
    print(f"OppChoVec min: {df_resultats['OppChoVec'].min():.4f} ({df_resultats.loc[df_resultats['OppChoVec'].idxmin(), 'Zone']})")
    print(f"OppChoVec max: {df_resultats['OppChoVec'].max():.4f} ({df_resultats.loc[df_resultats['OppChoVec'].idxmax(), 'Zone']})")

    return df_resultats


def exporter_resultats(df_resultats, fichier="oppchovec_0_norm.xlsx"):
    """
    Exporte les résultats vers un fichier Excel
    """
    print(f"\nExport des résultats vers {fichier}...")
    df_resultats.to_excel(fichier, index=False)
    print(f"Export termine: {fichier}")

    # Afficher le top 10
    print("\n" + "=" * 80)
    print("Top 10 des communes (OppChoVec SANS normalisation des indicateurs):")
    print("=" * 80)
    for idx in range(min(10, len(df_resultats))):
        row = df_resultats.iloc[idx]
        print(f"{idx+1:2d}. {row['Zone']:30s} - OppChoVec: {row['OppChoVec']:10.4f}")

    print("\n" + "=" * 80)
    print("COMPARAISON DES DIMENSIONS (Top 3):")
    print("=" * 80)
    for idx in range(min(3, len(df_resultats))):
        row = df_resultats.iloc[idx]
        print(f"\n{idx+1}. {row['Zone']}")
        print(f"   Score_Opp: {row['Score_Opp']:10.2f}")
        print(f"   Score_Cho: {row['Score_Cho']:10.2f}")
        print(f"   Score_Vec: {row['Score_Vec']:10.2f}")
        print(f"   -> OppChoVec: {row['OppChoVec']:8.4f}")


def main():
    """
    Point d'entrée principal
    """
    # Calculer OppChoVec sans normalisation
    df_resultats = calculer_oppchovec_sans_normalisation()

    # Exporter vers Excel
    exporter_resultats(df_resultats, fichier="oppchovec_0_norm.xlsx")

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    main()
