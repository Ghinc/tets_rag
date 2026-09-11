"""
OppChoVec - Calcul de l'indice de bien-être objectif pour les communes de Corse

Cet indice mesure la qualité de vie selon trois dimensions:
- Opp (Opportunités) - Accès aux opportunités
- Cho (Choix) - Liberté de choix
- Vec (Vécu) - Qualité de vie vécue

Méthodologie basée sur les travaux de Lise Bourdeau-Lepage
Utilise uniquement des données "froides" (objectives) de sources officielles
"""

import pandas as pd
import numpy as np
from cmath import exp as cexp
import json
from pathlib import Path
from typing import Dict, List, Tuple


# ==============================================================================
# CONFIGURATION ET CONSTANTES
# ==============================================================================

# Paramètres de l'indice OppChoVec
ALPHA = 2.5  # Paramètre d'aversion à la pauvreté
BETA = 1.5   # Paramètre de complémentarité entre dimensions

# Pondérations des indicateurs au sein de chaque dimension
PONDERATIONS = {
    'Opp': {'Opp1': 0.25, 'Opp2': 0.25, 'Opp3': 0.25, 'Opp4': 0.25},
    'Cho': {'Cho1': 0.50, 'Cho2': 0.50},
    'Vec': {'Vec1': 0.25, 'Vec2': 0.25, 'Vec3': 0.25, 'Vec4': 0.25}
}

# Pondérations finales des dimensions
POND_FINALE = [1, 1, 1]  # Opp, Cho, Vec

# Chemins des fichiers d'entrée
DATA_DIR = Path("../../Données/Corse_Commune")
INPUT_FILES = {
    'Opp1': DATA_DIR / "Opp1.xlsx",
    'Opp2': DATA_DIR / "Opp2.xlsx",
    'Opp3': DATA_DIR / "Opp3.xlsx",
    'Opp4': DATA_DIR / "Opp4.xlsx",
    'Cho1': DATA_DIR / "Cho1_personnes.xlsx",
    'Cho2': DATA_DIR / "Cho2.xlsx",
    'Vec1': DATA_DIR / "Vec1.xlsx",
    'Vec2': DATA_DIR / "vec2_lb.csv",
    'Vec3': DATA_DIR / "Vec3.xlsx",
    'Vec4': Path("services_accessibles_20min_local.csv"),  # Services accessibles < 20 min (OSRM local)
}

# Fichier de mapping code INSEE -> nom commune
MAPPING_FILE = DATA_DIR / "mapping_communes.csv"

# Colonnes spécifiques par fichier
COLONNES = {
    'Opp1': "Niveau d'education moyen",
    'Opp2': "Indice de Theil",
    'Opp3': ["Part des ménages ayant au moins 1 voiture 2021", "Accès aux réseaux de transport"],
    'Opp4': ["Proportion de population avec débit > 30Mb/s", "Proportion de population couverte par la 4G"],
    'Cho1': "Nb_Personnes_Quartiers_Prioritaires",
    'Cho2': "Proportion",
    'Vec1': "Médiane du niveau de vie 2021",
    'Vec2': ["pers_par_piece_moy", "pct_avec_sdb", "pct_chauffage", "pct_maisons"],
    'Vec3': ["Emploi stable (5)", "Contrat à durée déterminée (4)", "Contrat ponctuel (3)",
             "Chomeur (1)", "Emploi aidé (2)"],
    'Vec4': "nb_services_20min",  # Nombre de services accessibles < 20 min
}

# Pondérations pour Vec3 (stabilité d'emploi)
POND_VEC3 = [1, 0.75, 0.5, 0.25, 0]


# ==============================================================================
# FONCTIONS DE CALCUL DES INDICATEURS
# ==============================================================================

def calc_opp1(e: float) -> float:
    """
    Indicateur Opp1: Niveau d'éducation moyen

    Args:
        e: Niveau d'éducation moyen (échelle 1-7)

    Returns:
        Score d'éducation
    """
    return e


def calc_opp2(d_jour: float, d_nuit: float) -> float:
    """
    Indicateur Opp2: Diversité sociale (indice de Theil)

    Args:
        d_jour: Indice de Theil de jour
        d_nuit: Indice de Theil de nuit

    Returns:
        Moyenne des indices de Theil
    """
    return (d_jour + d_nuit) / 2


def calc_opp3(v_p: float, g_j: float, zone: str = "") -> float:
    """
    Indicateur Opp3: Accès au transport (voiture ou transports en commun)

    Args:
        v_p: Proportion de ménages avec voiture
        g_j: Accès aux transports en commun (0 ou 1)
        zone: Code INSEE ou nom de la commune

    Returns:
        Score de mobilité
        = v_p + 100 si g_j > 0 + 100 si zone a un vrai réseau de bus
    """
    score = v_p

    # +100 si la commune possède un réseau de transports en commun
    if g_j > 0:
        score += 100

    # +100 pour Ajaccio, Bastia et Porto-Vecchio (vrai réseau de bus)
    # Codes INSEE: 2A004 (Ajaccio), 2B033 (Bastia), 2A247 (Porto-Vecchio)
    villes_reseau_bus = ['2A004', '2B033', '2A247', 'Ajaccio', 'Bastia', 'Porto-Vecchio']
    if zone in villes_reseau_bus:
        score += 100

    return score


def calc_opp4(r: float, t: float) -> float:
    """
    Indicateur Opp4: Accès à Internet et 4G

    Args:
        r: Couverture Internet
        t: Couverture 4G

    Returns:
        Score de connectivité numérique
    """
    return (r + t) / 2


def calc_cho1(nb_personnes: float) -> float:
    """
    Indicateur Cho1: Quartiers prioritaires

    Args:
        nb_personnes: Nombre de personnes vivant dans les quartiers prioritaires

    Returns:
        Score inversé: -log(nb_personnes) si nb_personnes > 0, sinon 0
        (moins de personnes dans quartiers prioritaires = meilleur score)
    """
    if nb_personnes > 0:
        return -np.log(nb_personnes)
    else:
        return 0


def calc_cho2(d_v: float) -> float:
    """
    Indicateur Cho2: Accès au droit de vote

    Args:
        d_v: Proportion de personnes inscrites/ayant le droit de vote

    Returns:
        Proportion de personnes inscrites (inscrits sur les listes électorales)
    """
    return d_v


def calc_vec1(rf: float) -> float:
    """
    Indicateur Vec1: Niveau de revenu médian

    Args:
        rf: Revenu fiscal médian

    Returns:
        Revenu médian
    """
    return rf


def calc_vec2(v21: float, v22_sdb: float, v22_chauffage: float, v23: float) -> float:
    """
    Indicateur Vec2: Qualité du logement

    Args:
        v21: Nombre moyen de personnes par pièce
        v22_sdb: Proportion de logements avec salle de bain
        v22_chauffage: Proportion de logements avec chauffage
        v23: Proportion de maisons individuelles

    Returns:
        Score de qualité du logement (moyenne de 3 composantes)
    """
    v22 = (v22_sdb + v22_chauffage) / 2  # Moyenne de sdb et chauffage
    return (cexp(-v21).real + v22 + v23) / 3


def calc_vec3(p_vec: List[float], valeur: List[float]) -> float:
    """
    Indicateur Vec3: Stabilité de l'emploi

    Args:
        p_vec: Proportions de chaque catégorie d'emploi
        valeur: Pondérations de stabilité pour chaque catégorie

    Returns:
        Score pondéré de stabilité d'emploi
    """
    if sum(p_vec) == 0:
        return 0
    return sum(p * v for p, v in zip(p_vec, valeur)) / sum(p_vec)


def calc_vec4(n_services: float) -> float:
    """
    Indicateur Vec4: Accès aux services de vie courante

    Nombre de services accessibles à moins de 20 minutes en voiture
    Calculé avec distance géodésique × 1.6 / vitesse 45 km/h (ajusté Corse)

    Args:
        n_services: Nombre de services accessibles < 20 minutes

    Returns:
        Nombre de services accessibles
    """
    return n_services


# ==============================================================================
# FONCTIONS DE NORMALISATION ET AGRÉGATION
# ==============================================================================

def normaliser(x: float, min_x: float, max_x: float) -> float:
    """
    Normalise une valeur entre 0 et 1

    Args:
        x: Valeur à normaliser
        min_x: Valeur minimale de l'échantillon
        max_x: Valeur maximale de l'échantillon

    Returns:
        Valeur normalisée entre 0 et 1
    """
    if max_x == min_x:
        return 0
    return (x - min_x) / (max_x - min_x)


def calc_dik(v_ijk_dict: Dict[str, float], p_jk_dict: Dict[str, float]) -> float:
    """
    Calcule le score d'une dimension (moyenne pondérée des indicateurs normalisés)

    Args:
        v_ijk_dict: Dictionnaire des valeurs normalisées des indicateurs
        p_jk_dict: Dictionnaire des pondérations des indicateurs

    Returns:
        Score de la dimension (0 à 1)
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
    où α = 2.5 (aversion à la pauvreté)
       β = 1.5 (complémentarité entre dimensions)

    Args:
        dik: Array des scores des 3 dimensions [Opp, Cho, Vec]
        pk_values: Array des pondérations finales [1, 1, 1]

    Returns:
        Indice OppChoVec final
    """
    somme_ponderee = (pk_values * (dik ** BETA)).sum()
    oppchovec = (1/3) * (somme_ponderee ** (ALPHA / BETA))
    return oppchovec


# ==============================================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# ==============================================================================

def charger_donnees_commune(zone: str) -> Dict:
    """
    Charge toutes les données d'une commune depuis les fichiers Excel/CSV

    Args:
        zone: Nom de la commune ou code INSEE (ex: "Ajaccio" ou "2A004")

    Returns:
        Dictionnaire contenant toutes les données brutes de la commune
    """
    donnees = {'Zone': zone}

    for indicateur, fichier in INPUT_FILES.items():
        try:
            # Déterminer si c'est un fichier CSV ou Excel
            is_csv = str(fichier).endswith('.csv')

            if is_csv:
                df = pd.read_csv(fichier)
                # Pour les CSV, chercher par code_commune
                if 'code_commune' in df.columns:
                    ligne = df[df['code_commune'] == zone]
                else:
                    ligne = df[df['Zone'] == zone]
            else:
                df = pd.read_excel(fichier)
                # Pour les Excel, chercher d'abord par Code commune, sinon par Zone
                if 'Code commune' in df.columns:
                    ligne = df[df['Code commune'] == zone]
                else:
                    ligne = df[df['Zone'] == zone]

            if ligne.empty:
                print(f"Attention: {zone} non trouvée dans {fichier}")
                continue

            col_names = COLONNES[indicateur]

            if isinstance(col_names, list):
                # Plusieurs colonnes à récupérer
                for col in col_names:
                    if col in ligne.columns:
                        donnees[f"{indicateur}_{col}"] = ligne[col].values[0]
            else:
                # Une seule colonne
                if col_names in ligne.columns:
                    donnees[indicateur] = ligne[col_names].values[0]

        except Exception as e:
            print(f"Erreur lors du chargement de {fichier} pour {zone}: {e}")

    return donnees


def charger_toutes_communes() -> List[str]:
    """
    Récupère la liste de toutes les communes depuis vec2_lb.csv (codes INSEE)

    Returns:
        Liste des codes INSEE des communes (ex: ["2A001", "2A004", ...])
    """
    try:
        df = pd.read_csv(INPUT_FILES['Vec2'])
        return df['code_commune'].tolist()
    except Exception as e:
        print(f"Erreur lors du chargement de la liste des communes: {e}")
        return []


def charger_mapping_communes() -> Dict[str, str]:
    """
    Charge le mapping code INSEE -> nom de commune

    Returns:
        Dictionnaire {code_insee: nom_commune}
    """
    try:
        df_mapping = pd.read_csv(MAPPING_FILE)
        return dict(zip(df_mapping['code_commune'], df_mapping['nom_commune']))
    except Exception as e:
        print(f"Erreur lors du chargement du mapping: {e}")
        return {}


# ==============================================================================
# FONCTIONS DE CALCUL DES INDICATEURS POUR UNE COMMUNE
# ==============================================================================

def calculer_indicateurs(donnees: Dict) -> Dict[str, float]:
    """
    Calcule tous les indicateurs pour une commune

    Args:
        donnees: Dictionnaire des données brutes de la commune

    Returns:
        Dictionnaire des indicateurs calculés
    """
    indicateurs = {}

    # Opp1: Niveau d'éducation
    if 'Opp1' in donnees:
        indicateurs['Opp1'] = calc_opp1(donnees['Opp1'])

    # Opp2: Diversité sociale (Theil)
    if 'Opp2' in donnees:
        # Le fichier contient directement l'indice de Theil
        indicateurs['Opp2'] = donnees['Opp2']

    # Opp3: Accès au transport
    v_p_key = 'Opp3_Part des ménages ayant au moins 1 voiture 2021'
    g_j_key = 'Opp3_Accès aux réseaux de transport'
    if v_p_key in donnees and g_j_key in donnees:
        zone_nom = donnees.get('Zone', '')
        indicateurs['Opp3'] = calc_opp3(donnees[v_p_key], donnees[g_j_key], zone_nom)

    # Opp4: Connectivité numérique
    r_key = 'Opp4_Proportion de population avec débit > 30Mb/s'
    t_key = 'Opp4_Proportion de population couverte par la 4G'
    if r_key in donnees and t_key in donnees:
        indicateurs['Opp4'] = calc_opp4(donnees[r_key], donnees[t_key])

    # Cho1: Quartiers prioritaires
    if 'Cho1' in donnees:
        indicateurs['Cho1'] = calc_cho1(donnees['Cho1'])

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


# ==============================================================================
# FONCTION PRINCIPALE DE TRAITEMENT
# ==============================================================================

def calculer_oppchovec_complet(communes: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcule l'indice OppChoVec pour toutes les communes

    Args:
        communes: Liste des communes à traiter (None = toutes)

    Returns:
        Tuple de 3 DataFrames:
        - df_indicateurs: Tous les indicateurs calculés par commune
        - df_dimensions: Scores des 3 dimensions par commune
        - df_oppchovec: Indice OppChoVec final par commune
    """
    print("Chargement de la liste des communes...")
    if communes is None:
        communes = charger_toutes_communes()

    print(f"Traitement de {len(communes)} communes...")

    # Étape 1: Charger et calculer les indicateurs pour toutes les communes
    print("Calcul des indicateurs...")
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

    # Étape 2: Normaliser tous les indicateurs
    print("Normalisation des indicateurs...")
    df_normalise = df_indicateurs.copy()

    for col in df_indicateurs.columns:
        min_val = df_indicateurs[col].min()
        max_val = df_indicateurs[col].max()
        df_normalise[col] = df_indicateurs[col].apply(
            lambda x: normaliser(x, min_val, max_val)
        )

    # Étape 3: Calculer les scores des dimensions
    print("Calcul des scores des dimensions...")
    scores_dimensions = {}

    for zone in df_normalise.index:
        scores = {}

        # Score Opp
        opp_vals = {k: df_normalise.loc[zone, k] for k in PONDERATIONS['Opp'].keys()
                    if k in df_normalise.columns}
        scores['Score_Opp'] = calc_dik(opp_vals, PONDERATIONS['Opp'])

        # Score Cho
        cho_vals = {k: df_normalise.loc[zone, k] for k in PONDERATIONS['Cho'].keys()
                    if k in df_normalise.columns}
        scores['Score_Cho'] = calc_dik(cho_vals, PONDERATIONS['Cho'])

        # Score Vec
        vec_vals = {k: df_normalise.loc[zone, k] for k in PONDERATIONS['Vec'].keys()
                    if k in df_normalise.columns}
        scores['Score_Vec'] = calc_dik(vec_vals, PONDERATIONS['Vec'])

        scores_dimensions[zone] = scores

    df_dimensions = pd.DataFrame.from_dict(scores_dimensions, orient='index')
    df_dimensions.index.name = 'Zone'

    # Étape 4: Calculer l'indice OppChoVec final
    print("Calcul de l'indice OppChoVec final...")
    resultats_oppchovec = {}

    for zone in df_dimensions.index:
        dik = np.array([
            df_dimensions.loc[zone, 'Score_Opp'],
            df_dimensions.loc[zone, 'Score_Cho'],
            df_dimensions.loc[zone, 'Score_Vec']
        ])
        pk = np.array(POND_FINALE)

        oppchovec = calc_oppchovec(dik, pk)
        resultats_oppchovec[zone] = {
            'OppChoVec': oppchovec,
            'Score_Opp': dik[0],
            'Score_Cho': dik[1],
            'Score_Vec': dik[2]
        }

    df_oppchovec = pd.DataFrame.from_dict(resultats_oppchovec, orient='index')
    df_oppchovec.index.name = 'Zone'

    print("Calcul terminé!")
    return df_indicateurs, df_dimensions, df_oppchovec


# ==============================================================================
# FONCTIONS D'EXPORT
# ==============================================================================

def exporter_resultats(df_indicateurs: pd.DataFrame,
                      df_dimensions: pd.DataFrame,
                      df_oppchovec: pd.DataFrame,
                      suffixe: str = "V"):
    """
    Exporte tous les résultats vers des fichiers Excel et JSON

    Args:
        df_indicateurs: DataFrame des indicateurs
        df_dimensions: DataFrame des scores de dimensions
        df_oppchovec: DataFrame des résultats OppChoVec
        suffixe: Suffixe à ajouter aux noms de fichiers (par défaut "V")
    """
    print("\nExport des résultats...")

    # Export des indicateurs
    fichier_indicateurs = f"df_indicateur.xlsx"
    df_indicateurs.to_excel(fichier_indicateurs)
    print(f"  - {fichier_indicateurs}")

    # Export des dimensions
    fichier_dimensions = f"dimensions_{suffixe}.xlsx"
    df_dimensions.to_excel(fichier_dimensions)
    print(f"  - {fichier_dimensions}")

    # Export OppChoVec
    fichier_oppchovec = f"oppchovec_resultats_{suffixe}.xlsx"
    df_oppchovec.to_excel(fichier_oppchovec)
    print(f"  - {fichier_oppchovec}")

    # Export OppChoVec normalisé
    df_oppchovec_norm = df_oppchovec.copy()
    min_opp = df_oppchovec['OppChoVec'].min()
    max_opp = df_oppchovec['OppChoVec'].max()
    df_oppchovec_norm['OppChoVec_Normalise'] = df_oppchovec['OppChoVec'].apply(
        lambda x: normaliser(x, min_opp, max_opp)
    )

    fichier_oppchovec_norm = f"oppchovec_resultats_normalisées_{suffixe}.xlsx"
    df_oppchovec_norm.to_excel(fichier_oppchovec_norm)
    print(f"  - {fichier_oppchovec_norm}")

    # Export JSON des données brutes
    data_dict = df_indicateurs.to_dict(orient='index')
    fichier_json = "data_indicateurs.json"
    with open(fichier_json, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    print(f"  - {fichier_json}")

    # Statistiques descriptives
    stats = df_oppchovec_norm['OppChoVec_Normalise'].describe()
    df_stats = pd.DataFrame(stats)
    df_stats.columns = ['OppChoVec_Normalise']

    fichier_stats = f"stats_descriptives_normalisées_{suffixe}.xlsx"
    df_stats.to_excel(fichier_stats)
    print(f"  - {fichier_stats}")

    print("\nRésumé des résultats:")
    print(f"  Nombre de communes: {len(df_oppchovec)}")
    print(f"  OppChoVec moyen: {df_oppchovec['OppChoVec'].mean():.4f}")
    print(f"  OppChoVec min: {df_oppchovec['OppChoVec'].min():.4f} ({df_oppchovec['OppChoVec'].idxmin()})")
    print(f"  OppChoVec max: {df_oppchovec['OppChoVec'].max():.4f} ({df_oppchovec['OppChoVec'].idxmax()})")

    # Top 10
    print("\nTop 10 des communes:")
    top10 = df_oppchovec.nlargest(10, 'OppChoVec')
    for i, (zone, row) in enumerate(top10.iterrows(), 1):
        print(f"  {i:2d}. {zone:30s} - {row['OppChoVec']:.4f}")


def exporter_comparaison(df_oppchovec: pd.DataFrame,
                        communes_ciblees: List[str],
                        suffixe: str = "V"):
    """
    Exporte une comparaison pour des communes ciblées

    Args:
        df_oppchovec: DataFrame des résultats OppChoVec
        communes_ciblees: Liste des communes à comparer
        suffixe: Suffixe à ajouter au nom de fichier
    """
    # Filtrer les communes ciblées qui existent dans les résultats
    communes_existantes = [c for c in communes_ciblees if c in df_oppchovec.index]

    if not communes_existantes:
        print("Aucune commune ciblée trouvée dans les résultats")
        return

    df_comparaison = df_oppchovec.loc[communes_existantes]
    fichier_comparaison = f"Comparaison_{suffixe}.xlsx"
    df_comparaison.to_excel(fichier_comparaison)
    print(f"\nComparaison exportée: {fichier_comparaison}")


# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ==============================================================================

def main():
    """
    Fonction principale qui orchestre le calcul de l'indice OppChoVec
    """
    print("=" * 80)
    print("CALCUL DE L'INDICE OPPCHOVEC - COMMUNES DE CORSE")
    print("=" * 80)
    print()

    # Calculer l'indice pour toutes les communes
    df_indicateurs, df_dimensions, df_oppchovec = calculer_oppchovec_complet()

    # Exporter tous les résultats
    exporter_resultats(df_indicateurs, df_dimensions, df_oppchovec)

    # Communes ciblées pour comparaison
    communes_ciblees = [
        "Ajaccio", "Corte", "Bastia", "Alata", "Appieto",
        "Ghisonaccia", "Propriano", "Bonifacio", "Aléria",
        "Lucciana", "Calvi", "Chisa", "Altiani", "Pietralba",
        "Santa-Maria-di-Lota"
    ]

    exporter_comparaison(df_oppchovec, communes_ciblees)

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINÉ")
    print("=" * 80)


if __name__ == "__main__":
    main()
