"""
Simulation de l'impact de l'amélioration des transports en commun (Opp3)
sur l'indice OppChoVec

Teste l'impact d'ajouter +100 au score Opp3 pour les communes avec
beaucoup de transports en commun.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Communes avec beaucoup de transports en commun
COMMUNES_TRANSPORTS = [
    'Ajaccio',
    'Bastia',
    'Corte',
    "L'Ile-Rousse",
    'Ghisonaccia',
    'Moriani-Plage',  # Attention au nom exact
    'Calvi'
]

# Bonus à ajouter à Opp3 (transports)
BONUS_TRANSPORT = 100

# Paramètres OppChoVec (identiques à oppchovec.py)
ALPHA = 2.5
BETA = 1.5

PONDERATIONS = {
    'Opp': {'Opp1': 0.25, 'Opp2': 0.25, 'Opp3': 0.25, 'Opp4': 0.25},
    'Cho': {'Cho1': 0.50, 'Cho2': 0.50},
    'Vec': {'Vec1': 0.25, 'Vec2': 0.25, 'Vec3': 0.25, 'Vec4': 0.25}
}

POND_FINALE = [1, 1, 1]


# ==============================================================================
# FONCTIONS DE CALCUL
# ==============================================================================

def normaliser(x, min_x, max_x):
    """Normalise une valeur entre 0 et 1"""
    if max_x == min_x:
        return 0
    return (x - min_x) / (max_x - min_x)


def calc_dik(v_ijk_dict, p_jk_dict):
    """Calcule le score d'une dimension"""
    valeurs = np.array([v_ijk_dict[k] for k in p_jk_dict.keys() if k in v_ijk_dict])
    poids = np.array([p_jk_dict[k] for k in p_jk_dict.keys() if k in v_ijk_dict])

    if poids.sum() == 0:
        return 0

    return (valeurs * poids).sum() / poids.sum()


def calc_oppchovec(dik, pk_values):
    """Calcule l'indice OppChoVec final"""
    somme_ponderee = (pk_values * (dik ** BETA)).sum()
    oppchovec = (1/3) * (somme_ponderee ** (ALPHA / BETA))
    return oppchovec


# ==============================================================================
# FONCTIONS PRINCIPALES
# ==============================================================================

def charger_donnees():
    """Charge les données actuelles"""
    print("Chargement des données...")

    with open('data_indicateurs.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  {len(data)} communes chargées")
    return data


def appliquer_simulation(data_original):
    """Applique la simulation : ajoute +100 à Opp3 pour certaines communes"""
    data_simule = {}
    communes_modifiees = []
    communes_non_trouvees = []

    for commune, indicateurs in data_original.items():
        # Copier les données
        data_simule[commune] = indicateurs.copy()

        # Vérifier si la commune doit être modifiée
        commune_match = None
        for c_transport in COMMUNES_TRANSPORTS:
            # Recherche flexible (ignore casse et tirets)
            if (c_transport.lower().replace('-', '').replace("'", '') in
                commune.lower().replace('-', '').replace("'", '')):
                commune_match = c_transport
                break

        if commune_match:
            # Modifier Opp3
            if 'Opp3' in data_simule[commune]:
                ancienne_valeur = data_simule[commune]['Opp3']
                data_simule[commune]['Opp3'] = ancienne_valeur + BONUS_TRANSPORT
                communes_modifiees.append({
                    'commune': commune,
                    'ancienne_valeur': ancienne_valeur,
                    'nouvelle_valeur': ancienne_valeur + BONUS_TRANSPORT
                })

    # Vérifier les communes non trouvées
    for c in COMMUNES_TRANSPORTS:
        if not any(c.lower().replace('-', '').replace("'", '') in
                   cm['commune'].lower().replace('-', '').replace("'", '')
                   for cm in communes_modifiees):
            communes_non_trouvees.append(c)

    return data_simule, communes_modifiees, communes_non_trouvees


def calculer_indices(data_indicateurs):
    """Calcule les indices OppChoVec pour un ensemble de données"""
    print("\nCalcul des indices...")

    # Étape 1: Normaliser les indicateurs
    df = pd.DataFrame.from_dict(data_indicateurs, orient='index')
    df_normalise = df.copy()

    for col in df.columns:
        if col in ['Opp1', 'Opp2', 'Opp3', 'Opp4', 'Cho1', 'Cho2',
                   'Vec1', 'Vec2', 'Vec3', 'Vec4']:
            min_val = df[col].min()
            max_val = df[col].max()
            df_normalise[col] = df[col].apply(lambda x: normaliser(x, min_val, max_val))

    # Étape 2: Calculer les scores des dimensions
    scores_dimensions = {}

    for commune in df_normalise.index:
        scores = {}

        # Score Opp
        opp_vals = {k: df_normalise.loc[commune, k] for k in PONDERATIONS['Opp'].keys()
                    if k in df_normalise.columns}
        scores['Score_Opp'] = calc_dik(opp_vals, PONDERATIONS['Opp'])

        # Score Cho
        cho_vals = {k: df_normalise.loc[commune, k] for k in PONDERATIONS['Cho'].keys()
                    if k in df_normalise.columns}
        scores['Score_Cho'] = calc_dik(cho_vals, PONDERATIONS['Cho'])

        # Score Vec
        vec_vals = {k: df_normalise.loc[commune, k] for k in PONDERATIONS['Vec'].keys()
                    if k in df_normalise.columns}
        scores['Score_Vec'] = calc_dik(vec_vals, PONDERATIONS['Vec'])

        scores_dimensions[commune] = scores

    # Étape 3: Calculer OppChoVec final
    resultats = {}

    for commune, scores in scores_dimensions.items():
        dik = np.array([scores['Score_Opp'], scores['Score_Cho'], scores['Score_Vec']])
        pk = np.array(POND_FINALE)

        oppchovec = calc_oppchovec(dik, pk)

        resultats[commune] = {
            'OppChoVec': oppchovec,
            'Score_Opp': scores['Score_Opp'],
            'Score_Cho': scores['Score_Cho'],
            'Score_Vec': scores['Score_Vec']
        }

    return resultats


def comparer_resultats(resultats_avant, resultats_apres, communes_modifiees):
    """Compare les résultats avant/après simulation"""
    print("\nCréation du tableau comparatif...")

    comparaison = []

    # Pour toutes les communes
    for commune in resultats_avant.keys():
        avant = resultats_avant[commune]
        apres = resultats_apres[commune]

        # Vérifier si modifiée
        modifiee = any(cm['commune'] == commune for cm in communes_modifiees)

        comparaison.append({
            'Commune': commune,
            'Modifiée': 'OUI' if modifiee else 'Non',
            'OppChoVec_Avant': avant['OppChoVec'],
            'OppChoVec_Après': apres['OppChoVec'],
            'OppChoVec_Différence': apres['OppChoVec'] - avant['OppChoVec'],
            'OppChoVec_Variation_%': ((apres['OppChoVec'] - avant['OppChoVec']) / avant['OppChoVec'] * 100) if avant['OppChoVec'] > 0 else 0,
            'Score_Opp_Avant': avant['Score_Opp'],
            'Score_Opp_Après': apres['Score_Opp'],
            'Score_Opp_Différence': apres['Score_Opp'] - avant['Score_Opp'],
            'Score_Cho_Avant': avant['Score_Cho'],
            'Score_Vec_Avant': avant['Score_Vec']
        })

    df_comparaison = pd.DataFrame(comparaison)

    # Trier par variation décroissante
    df_comparaison = df_comparaison.sort_values('OppChoVec_Différence', ascending=False)

    return df_comparaison


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main():
    """Fonction principale"""
    print("="*80)
    print("SIMULATION - IMPACT DE L'AMÉLIORATION DES TRANSPORTS EN COMMUN")
    print("="*80)
    print(f"\nScénario: +{BONUS_TRANSPORT} au score Opp3 (Transports)")
    print(f"Communes concernées: {', '.join(COMMUNES_TRANSPORTS)}")

    # 1. Charger les données
    data_original = charger_donnees()

    # 2. Appliquer la simulation
    print("\nApplication de la simulation...")
    data_simule, communes_modifiees, communes_non_trouvees = appliquer_simulation(data_original)

    print(f"\n  Communes modifiées: {len(communes_modifiees)}")
    for cm in communes_modifiees:
        print(f"    - {cm['commune']:25s}: Opp3 {cm['ancienne_valeur']:.1f} -> {cm['nouvelle_valeur']:.1f}")

    if communes_non_trouvees:
        print(f"\n  [ATTENTION] Communes non trouvees: {', '.join(communes_non_trouvees)}")

    # 3. Calculer les indices AVANT simulation
    print("\n" + "="*80)
    print("CALCUL DES INDICES - SCÉNARIO ACTUEL")
    print("="*80)
    resultats_avant = calculer_indices(data_original)

    # 4. Calculer les indices APRÈS simulation
    print("\n" + "="*80)
    print("CALCUL DES INDICES - AVEC AMÉLIORATION TRANSPORTS")
    print("="*80)
    resultats_apres = calculer_indices(data_simule)

    # 5. Comparer les résultats
    df_comparaison = comparer_resultats(resultats_avant, resultats_apres, communes_modifiees)

    # 6. Export
    print("\n" + "="*80)
    print("EXPORT DES RÉSULTATS")
    print("="*80)

    fichier_excel = "simulation_impact_transports.xlsx"

    with pd.ExcelWriter(fichier_excel, engine='openpyxl') as writer:
        # Feuille 1: Comparaison complète
        df_comparaison.to_excel(writer, sheet_name='Comparaison_Complète', index=False)

        # Feuille 2: Focus sur communes modifiées
        df_modifiees = df_comparaison[df_comparaison['Modifiée'] == 'OUI']
        df_modifiees.to_excel(writer, sheet_name='Communes_Modifiées', index=False)

        # Feuille 3: Top 20 plus grandes variations
        df_top20 = df_comparaison.head(20)
        df_top20.to_excel(writer, sheet_name='Top_20_Variations', index=False)

    print(f"\n  [OK] {fichier_excel}")

    # 7. Afficher les résultats clés
    print("\n" + "="*80)
    print("RÉSULTATS CLÉS")
    print("="*80)

    print("\nCommunes modifiees - Impact sur OppChoVec:")
    for _, row in df_modifiees.iterrows():
        print(f"  {row['Commune']:25s}: {row['OppChoVec_Avant']:.4f} -> {row['OppChoVec_Après']:.4f} "
              f"({row['OppChoVec_Variation_%']:+.2f}%)")

    print("\nStatistiques globales:")
    print(f"  Variation moyenne (toutes communes):     {df_comparaison['OppChoVec_Différence'].mean():+.6f}")
    print(f"  Variation moyenne (communes modifiées):  {df_modifiees['OppChoVec_Différence'].mean():+.6f}")
    print(f"  Plus forte hausse: {df_comparaison.iloc[0]['Commune']} ({df_comparaison.iloc[0]['OppChoVec_Variation_%']:+.2f}%)")

    # Impact sur Score_Opp
    print("\nImpact sur le Score Opp (Opportunites):")
    for _, row in df_modifiees.iterrows():
        print(f"  {row['Commune']:25s}: {row['Score_Opp_Avant']:.4f} -> {row['Score_Opp_Après']:.4f} "
              f"({row['Score_Opp_Différence']:+.4f})")

    print("\n" + "="*80)
    print("SIMULATION TERMINÉE")
    print("="*80)
    print(f"\nConsultez {fichier_excel} pour l'analyse complète.")

    return df_comparaison


if __name__ == "__main__":
    main()
