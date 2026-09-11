"""
Calcul de l'indice OppChoVec SANS normalisation des indicateurs bruts

Ce script calcule l'indice OppChoVec en utilisant directement les valeurs brutes
des indicateurs (Opp1-4, Cho1-2, Vec1-4) sans les normaliser entre 0 et 1.
"""

import pandas as pd
import numpy as np
from cmath import exp as cexp
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Ajouter le répertoire du script oppchovec.py au path
sys.path.insert(0, str(Path(__file__).parent / "stage_ambroise/Code/Python"))

# Importer les fonctions de calcul et constantes depuis oppchovec.py
from oppchovec import (
    ALPHA, BETA, PONDERATIONS, POND_FINALE, POND_VEC3,
    calc_opp1, calc_opp2, calc_opp3, calc_opp4,
    calc_cho1, calc_cho2,
    calc_vec1, calc_vec2, calc_vec3, calc_vec4,
    charger_donnees_commune, charger_toutes_communes, charger_mapping_communes,
    calculer_indicateurs
)


def calc_dik_sans_normalisation(v_ijk_dict: Dict[str, float], p_jk_dict: Dict[str, float]) -> float:
    """
    Calcule le score d'une dimension SANS normalisation préalable
    (utilise directement les valeurs brutes)

    Args:
        v_ijk_dict: Dictionnaire des valeurs BRUTES des indicateurs
        p_jk_dict: Dictionnaire des pondérations des indicateurs

    Returns:
        Score de la dimension (moyenne pondérée des valeurs brutes)
    """
    valeurs = np.array([v_ijk_dict[k] for k in p_jk_dict.keys()])
    poids = np.array(list(p_jk_dict.values()))

    if poids.sum() == 0:
        return 0

    return (valeurs * poids).sum() / poids.sum()


def calc_oppchovec(dik: np.ndarray, pk_values: np.ndarray) -> float:
    """
    Calcule l'indice OppChoVec final

    Formule: (1/3) * [Σ(pk × dik^β)]^(α/β)

    Args:
        dik: Array des scores des 3 dimensions [Opp, Cho, Vec]
        pk_values: Array des pondérations finales [1, 1, 1]

    Returns:
        Indice OppChoVec final
    """
    somme_ponderee = (pk_values * (dik ** BETA)).sum()
    oppchovec = (1/3) * (somme_ponderee ** (ALPHA / BETA))
    return oppchovec


def calculer_oppchovec_sans_normalisation(communes: List[str] = None) -> pd.DataFrame:
    """
    Calcule l'indice OppChoVec SANS normaliser les indicateurs bruts

    Args:
        communes: Liste des communes à traiter (None = toutes)

    Returns:
        DataFrame avec tous les résultats (indicateurs bruts + dimensions + OppChoVec)
    """
    print("=" * 80)
    print("CALCUL OPPCHOVEC SANS NORMALISATION DES INDICATEURS")
    print("=" * 80)
    print()

    print("Chargement de la liste des communes...")
    if communes is None:
        communes = charger_toutes_communes()

    print(f"Traitement de {len(communes)} communes...")

    # Étape 1: Charger et calculer les indicateurs bruts pour toutes les communes
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

    # Étape 2: SKIP LA NORMALISATION - On utilise directement les valeurs brutes
    print("Calcul des dimensions SANS normalisation préalable...")

    # Étape 3: Calculer les scores des dimensions avec les valeurs BRUTES
    resultats = []

    for zone in df_indicateurs.index:
        # Récupérer les indicateurs bruts
        indicateurs_bruts = df_indicateurs.loc[zone]

        # Score Opp (avec valeurs brutes)
        opp_vals = {k: indicateurs_bruts.get(k, 0) for k in PONDERATIONS['Opp'].keys()
                    if k in df_indicateurs.columns}
        score_opp = calc_dik_sans_normalisation(opp_vals, PONDERATIONS['Opp'])

        # Score Cho (avec valeurs brutes)
        cho_vals = {k: indicateurs_bruts.get(k, 0) for k in PONDERATIONS['Cho'].keys()
                    if k in df_indicateurs.columns}
        score_cho = calc_dik_sans_normalisation(cho_vals, PONDERATIONS['Cho'])

        # Score Vec (avec valeurs brutes)
        vec_vals = {k: indicateurs_bruts.get(k, 0) for k in PONDERATIONS['Vec'].keys()
                    if k in df_indicateurs.columns}
        score_vec = calc_dik_sans_normalisation(vec_vals, PONDERATIONS['Vec'])

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

        # Ajouter tous les indicateurs bruts
        for ind in ['Opp1', 'Opp2', 'Opp3', 'Opp4', 'Cho1', 'Cho2', 'Vec1', 'Vec2', 'Vec3', 'Vec4']:
            if ind in indicateurs_bruts:
                resultat[ind] = indicateurs_bruts[ind]
            else:
                resultat[ind] = np.nan

        resultats.append(resultat)

    # Créer le DataFrame final
    df_resultats = pd.DataFrame(resultats)

    # Réorganiser les colonnes dans l'ordre souhaité
    colonnes_ordre = (
        ['Zone', 'OppChoVec', 'Score_Opp', 'Score_Cho', 'Score_Vec'] +
        ['Opp1', 'Opp2', 'Opp3', 'Opp4'] +
        ['Cho1', 'Cho2'] +
        ['Vec1', 'Vec2', 'Vec3', 'Vec4']
    )
    df_resultats = df_resultats[colonnes_ordre]

    # Trier par OppChoVec décroissant
    df_resultats = df_resultats.sort_values('OppChoVec', ascending=False)

    print("Calcul terminé!")
    print(f"\nNombre de communes: {len(df_resultats)}")
    print(f"OppChoVec moyen: {df_resultats['OppChoVec'].mean():.4f}")
    print(f"OppChoVec min: {df_resultats['OppChoVec'].min():.4f} ({df_resultats.loc[df_resultats['OppChoVec'].idxmin(), 'Zone']})")
    print(f"OppChoVec max: {df_resultats['OppChoVec'].max():.4f} ({df_resultats.loc[df_resultats['OppChoVec'].idxmax(), 'Zone']})")

    return df_resultats


def exporter_resultats(df_resultats: pd.DataFrame, fichier: str = "oppchovec_0_norm.xlsx"):
    """
    Exporte les résultats vers un fichier Excel

    Args:
        df_resultats: DataFrame des résultats
        fichier: Nom du fichier de sortie
    """
    print(f"\nExport des résultats vers {fichier}...")
    df_resultats.to_excel(fichier, index=False)
    print(f"✓ Export terminé: {fichier}")

    # Afficher le top 10
    print("\nTop 10 des communes (OppChoVec sans normalisation):")
    print("=" * 80)
    for i, row in df_resultats.head(10).iterrows():
        print(f"{i+1:2d}. {row['Zone']:30s} - OppChoVec: {row['OppChoVec']:10.4f}")

    print("\n" + "=" * 80)
    print("COMPARAISON DES DIMENSIONS (Top 3):")
    print("=" * 80)
    for i, row in df_resultats.head(3).iterrows():
        print(f"\n{i+1}. {row['Zone']}")
        print(f"   Score_Opp: {row['Score_Opp']:8.2f}")
        print(f"   Score_Cho: {row['Score_Cho']:8.2f}")
        print(f"   Score_Vec: {row['Score_Vec']:8.2f}")
        print(f"   → OppChoVec: {row['OppChoVec']:8.4f}")


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
