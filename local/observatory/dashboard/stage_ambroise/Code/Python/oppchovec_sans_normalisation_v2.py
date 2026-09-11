"""
Calcul de l'indice OppChoVec SANS normalisation des indicateurs bruts
Version 2 : avec nouveau calcul de Cho1 utilisant exp(-pourcentage)

Cho1 = exp(-pourcentage_population_quartiers_prioritaires)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from oppchovec import (
    ALPHA, BETA, PONDERATIONS, POND_FINALE, POND_VEC3,
    calc_opp1, calc_opp2, calc_opp3, calc_opp4,
    calc_cho2,
    calc_vec1, calc_vec2, calc_vec3, calc_vec4,
    charger_donnees_commune, charger_toutes_communes, charger_mapping_communes,
    INPUT_FILES
)

# Charger les données de population
DATA_DIR = Path("../../Données/Corse_Commune")
POPULATION_FILE = DATA_DIR / "population_corse.xlsx"


def charger_population():
    """
    Charge les données de population des communes corses

    Returns:
        Dict[str, float]: Dictionnaire {code_commune: population}
    """
    try:
        df_pop = pd.read_excel(POPULATION_FILE)
        return dict(zip(df_pop['code_commune'], df_pop['population_2022']))
    except Exception as e:
        print(f"Erreur lors du chargement de la population: {e}")
        return {}


def calc_cho1_v2(nb_personnes: float, population_totale: float) -> float:
    """
    Nouveau calcul de Cho1 : exp(-pourcentage_quartiers_prioritaires)

    Args:
        nb_personnes: Nombre de personnes dans les quartiers prioritaires
        population_totale: Population totale de la commune

    Returns:
        exp(-pourcentage) où pourcentage = nb_personnes / population_totale
    """
    if population_totale == 0:
        return 1.0  # Si pas de population, score maximal

    if nb_personnes == 0:
        return 1.0  # exp(-0) = 1

    pourcentage = nb_personnes / population_totale
    return np.exp(-pourcentage)


def calculer_indicateurs_v2(donnees: dict, population_dict: dict) -> dict:
    """
    Calcule tous les indicateurs pour une commune (version avec nouveau Cho1)

    Args:
        donnees: Dictionnaire des données brutes de la commune
        population_dict: Dictionnaire des populations par code INSEE

    Returns:
        Dictionnaire des indicateurs calculés
    """
    indicateurs = {}

    # Récupérer le code INSEE pour la population
    zone = donnees.get('Zone', '')
    population_totale = population_dict.get(zone, 0)

    # Opp1: Niveau d'éducation
    if 'Opp1' in donnees:
        indicateurs['Opp1'] = calc_opp1(donnees['Opp1'])

    # Opp2: Diversité sociale (Theil)
    if 'Opp2' in donnees:
        indicateurs['Opp2'] = donnees['Opp2']

    # Opp3: Accès au transport
    v_p_key = 'Opp3_Part des ménages ayant au moins 1 voiture 2021'
    g_j_key = 'Opp3_Accès aux réseaux de transport'
    if v_p_key in donnees and g_j_key in donnees:
        zone_nom = donnees.get('Zone', '')
        indicateurs['Opp3'] = calc_opp3(donnees[v_p_key], donnees[g_j_key], zone_nom)

    # Opp4: Connectivité numérique (Internet + 4G + Fibre)
    r_key = 'Opp4_Proportion de population avec débit > 30Mb/s'
    t_key = 'Opp4_Proportion de population couverte par la 4G'
    f_key = 'Opp4_fibre'
    if r_key in donnees and t_key in donnees and f_key in donnees:
        indicateurs['Opp4'] = calc_opp4(donnees[r_key], donnees[t_key], donnees[f_key])

    # Cho1: Quartiers prioritaires (NOUVEAU CALCUL)
    if 'Cho1' in donnees:
        nb_personnes = donnees['Cho1']
        indicateurs['Cho1'] = calc_cho1_v2(nb_personnes, population_totale)

    # Cho2: Droit de vote
    if 'Cho2' in donnees:
        indicateurs['Cho2'] = calc_cho2(donnees['Cho2'])

    # Vec1: Revenu médian
    if 'Vec1' in donnees:
        indicateurs['Vec1'] = calc_vec1(donnees['Vec1'])

    # Vec2: Qualité du logement
    v21_key = 'Vec2_pers_par_piece_moy'
    v22_sdb_key = 'Vec2_pct_avec_sdb'
    v22_chauffage_key = 'Vec2_pct_chauffage'
    v23_key = 'Vec2_pct_maisons'
    if all(k in donnees for k in [v21_key, v22_sdb_key, v22_chauffage_key, v23_key]):
        indicateurs['Vec2'] = calc_vec2(
            donnees[v21_key],
            donnees[v22_sdb_key],
            donnees[v22_chauffage_key],
            donnees[v23_key]
        )

    # Vec3: Stabilité de l'emploi
    vec3_keys = [
        'Vec3_Emploi stable (5)',
        'Vec3_Contrat à durée déterminée (4)',
        'Vec3_Contrat ponctuel (3)',
        'Vec3_Chomeur (1)',
        'Vec3_Emploi aidé (2)'
    ]
    if all(k in donnees for k in vec3_keys):
        p_vec = [donnees[k] for k in vec3_keys]
        indicateurs['Vec3'] = calc_vec3(p_vec, POND_VEC3)

    # Vec4: Accès aux commerces
    if 'Vec4' in donnees:
        indicateurs['Vec4'] = calc_vec4(donnees['Vec4'])

    return indicateurs


def calc_dik_sans_normalisation(v_ijk_dict, p_jk_dict):
    """
    Calcule le score d'une dimension SANS normalisation préalable
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


def calculer_oppchovec_sans_normalisation_v2():
    """
    Calcule l'indice OppChoVec SANS normaliser les indicateurs bruts
    Version 2 : avec nouveau calcul de Cho1
    """
    print("=" * 80)
    print("CALCUL OPPCHOVEC SANS NORMALISATION - VERSION 2")
    print("Nouveau Cho1: exp(-pourcentage_quartiers_prioritaires)")
    print("=" * 80)
    print()

    # Charger les données de population
    print("Chargement des données de population...")
    population_dict = charger_population()
    print(f"Population chargee pour {len(population_dict)} communes")

    print("Chargement de la liste des communes...")
    communes = charger_toutes_communes()
    print(f"Traitement de {len(communes)} communes...")

    # Étape 1: Charger et calculer les indicateurs bruts
    print("Calcul des indicateurs bruts avec nouveau Cho1...")
    data_indicateurs = {}

    for zone in communes:
        donnees = charger_donnees_commune(zone)
        indicateurs = calculer_indicateurs_v2(donnees, population_dict)
        data_indicateurs[zone] = indicateurs

    # Convertir en DataFrame
    df_indicateurs = pd.DataFrame.from_dict(data_indicateurs, orient='index')
    df_indicateurs.index.name = 'Zone'

    # Mapper les codes INSEE vers les noms de communes
    print("Conversion des codes INSEE vers les noms de communes...")
    mapping = charger_mapping_communes()
    df_indicateurs.index = df_indicateurs.index.map(lambda x: mapping.get(x, x))

    # Étape 2: SKIP LA NORMALISATION
    print("Calcul des dimensions SANS normalisation prealable...")

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

    print("Calcul termine!")
    print(f"\nNombre de communes: {len(df_resultats)}")
    print(f"OppChoVec moyen: {df_resultats['OppChoVec'].mean():.4f}")
    print(f"OppChoVec min: {df_resultats['OppChoVec'].min():.4f} ({df_resultats.loc[df_resultats['OppChoVec'].idxmin(), 'Zone']})")
    print(f"OppChoVec max: {df_resultats['OppChoVec'].max():.4f} ({df_resultats.loc[df_resultats['OppChoVec'].idxmax(), 'Zone']})")

    return df_resultats


def exporter_resultats(df_resultats, fichier="oppchovec_0_norm.xlsx"):
    """
    Exporte les résultats vers un fichier Excel
    """
    print(f"\nExport des resultats vers {fichier}...")
    df_resultats.to_excel(fichier, index=False)
    print(f"Export termine: {fichier}")

    # Afficher le top 10
    print("\n" + "=" * 80)
    print("Top 10 des communes (OppChoVec SANS normalisation, nouveau Cho1):")
    print("=" * 80)
    for idx in range(min(10, len(df_resultats))):
        row = df_resultats.iloc[idx]
        print(f"{idx+1:2d}. {row['Zone']:30s} - OppChoVec: {row['OppChoVec']:10.4f}")

    print("\n" + "=" * 80)
    print("COMPARAISON DES DIMENSIONS (Top 5):")
    print("=" * 80)
    for idx in range(min(5, len(df_resultats))):
        row = df_resultats.iloc[idx]
        print(f"\n{idx+1}. {row['Zone']}")
        print(f"   Score_Opp: {row['Score_Opp']:10.2f}")
        print(f"   Score_Cho: {row['Score_Cho']:10.2f}  (Cho1={row.get('Cho1', 'N/A'):.4f} si disponible)")
        print(f"   Score_Vec: {row['Score_Vec']:10.2f}")
        print(f"   -> OppChoVec: {row['OppChoVec']:8.4f}")


def main():
    """
    Point d'entrée principal
    """
    # Calculer OppChoVec sans normalisation (version 2 avec nouveau Cho1)
    df_resultats = calculer_oppchovec_sans_normalisation_v2()

    # Exporter vers Excel
    exporter_resultats(df_resultats, fichier="oppchovec_0_norm.xlsx")

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)


if __name__ == "__main__":
    main()
