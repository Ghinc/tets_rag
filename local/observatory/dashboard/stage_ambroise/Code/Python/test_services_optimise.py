"""
Version OPTIMISÉE pour test sur quelques communes

Stratégie d'optimisation :
1. Pré-filtrage par distance géodésique (rayon ~30-40 km)
2. Utilisation OSRM uniquement pour les services potentiellement accessibles
3. Checkpoint system pour sauvegarde progressive
4. Comparaison avec résultats actuels

Test sur : Ajaccio, Bastia, Corte, Bonifacio, Calvi
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from math import radians, cos, sin, asin, sqrt


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Communes à tester
COMMUNES_TEST = [
    'Ajaccio',
    'Bastia',
    'Corte',
    'Bonifacio',
    'Calvi'
]

# Fichiers
COORDS_FILE = "communes_corse_coordonnees.csv"
BPE_FILE = "bpe_corse.csv"
CHECKPOINT_FILE = "test_services_checkpoint.pkl"
OUTPUT_FILE = "test_services_optimise_resultats.csv"
COMPARAISON_FILE = "services_accessibles_20min.csv"

# API OSRM
OSRM_URL = "http://router.project-osrm.org/table/v1/driving/"

# Paramètres
SEUIL_TEMPS_MIN = 20
RAYON_PREFILTRE_KM = 40  # Distance géodésique max pour pré-filtrage
BATCH_SIZE = 20
PAUSE_ENTRE_REQUETES = 1.5
MAX_RETRIES = 3


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calcule la distance géodésique entre deux points (formule de Haversine)

    Args:
        lon1, lat1: Coordonnées du point 1 (degrés)
        lon2, lat2: Coordonnées du point 2 (degrés)

    Returns:
        Distance en kilomètres
    """
    # Convertir en radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Formule de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Rayon de la Terre en km
    r = 6371

    return c * r


def prefiltre_services_par_distance(commune_coords: Tuple[float, float],
                                    services_coords: List[Tuple[float, float, int]],
                                    rayon_km: float) -> List[int]:
    """
    Pré-filtre les services par distance géodésique

    Args:
        commune_coords: (longitude, latitude) de la commune
        services_coords: Liste de (longitude, latitude, index) des services
        rayon_km: Rayon maximum en km

    Returns:
        Liste des indices des services dans le rayon
    """
    lon_commune, lat_commune = commune_coords
    services_proches = []

    for lon_service, lat_service, idx in services_coords:
        distance = haversine(lon_commune, lat_commune, lon_service, lat_service)
        if distance <= rayon_km:
            services_proches.append(idx)

    return services_proches


# ==============================================================================
# CHECKPOINT MANAGER
# ==============================================================================

class CheckpointManager:
    """Gère la sauvegarde et reprise des calculs"""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.data = self.load()

    def load(self) -> Dict:
        """Charge le checkpoint s'il existe"""
        if Path(self.checkpoint_file).exists():
            print(f"[CHECKPOINT] Reprise depuis {self.checkpoint_file}")
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"  -> {len(data.get('resultats', []))} communes deja traitees")
                return data
            except Exception as e:
                print(f"  -> Erreur chargement: {e}")
                return {'resultats': [], 'communes_traitees': set()}
        return {'resultats': [], 'communes_traitees': set()}

    def save(self, resultats: List[Dict], commune: str):
        """Sauvegarde l'état actuel"""
        communes_traitees = self.data.get('communes_traitees', set())
        communes_traitees.add(commune)

        data = {
            'resultats': resultats,
            'communes_traitees': communes_traitees,
            'timestamp': time.time()
        }
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[ERREUR] Sauvegarde: {e}")

    def delete(self):
        """Supprime le checkpoint"""
        if Path(self.checkpoint_file).exists():
            Path(self.checkpoint_file).unlink()


# ==============================================================================
# CALCUL OSRM
# ==============================================================================

def calculer_temps_osrm_batch(commune_coords: Tuple[float, float],
                              services_coords: List[Tuple[float, float]],
                              max_retries: int = MAX_RETRIES) -> List[Optional[float]]:
    """
    Calcule les temps de trajet OSRM par batch

    Args:
        commune_coords: (longitude, latitude) de la commune
        services_coords: Liste de (longitude, latitude) des services

    Returns:
        Liste des temps en minutes (None si échec)
    """
    if not services_coords:
        return []

    temps = [None] * len(services_coords)

    # Traiter par batch
    for i in range(0, len(services_coords), BATCH_SIZE):
        batch = services_coords[i:i+BATCH_SIZE]

        # Construire requête
        all_coords = [commune_coords] + batch
        coords_str = ";".join([f"{lon},{lat}" for lon, lat in all_coords])

        url = f"{OSRM_URL}{coords_str}?sources=0&destinations=1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20"[:len(coords_str)+100]

        # Retry logic
        for tentative in range(max_retries):
            try:
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if data.get('code') == 'Ok':
                        durations = data.get('durations', [[]])[0]

                        # Convertir secondes en minutes
                        for j, duration in enumerate(durations):
                            if duration is not None:
                                temps[i + j] = duration / 60.0

                        time.sleep(PAUSE_ENTRE_REQUETES)
                        break  # Succès
                    else:
                        print(f"    Code OSRM: {data.get('code')}")

                elif response.status_code == 429:
                    print(f"    Rate limit (429), pause 10s...")
                    time.sleep(10)
                else:
                    print(f"    Status {response.status_code}")

            except requests.exceptions.Timeout:
                print(f"    Timeout (tentative {tentative+1}/{max_retries})")
                time.sleep(5)
            except Exception as e:
                print(f"    Erreur: {e}")
                time.sleep(5)

        # Afficher progression
        progress = min(100, (i + len(batch)) / len(services_coords) * 100)
        print(f"      {progress:.0f}% des services traites")

    return temps


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def calculer_services_pour_commune(commune: str,
                                   coords_commune: Tuple[float, float],
                                   df_services: pd.DataFrame) -> Dict:
    """
    Calcule le nombre de services accessibles pour une commune

    Args:
        commune: Nom de la commune
        coords_commune: (longitude, latitude)
        df_services: DataFrame des services avec coordonnées

    Returns:
        Dictionnaire avec résultats
    """
    print(f"\n  {commune}")
    print(f"    Coordonnees: {coords_commune}")

    # Étape 1: Pré-filtrage géodésique
    print(f"    Etape 1: Pre-filtrage (rayon {RAYON_PREFILTRE_KM} km)...")

    services_coords_indexed = [
        (row['longitude'], row['latitude'], idx)
        for idx, row in df_services.iterrows()
    ]

    indices_proches = prefiltre_services_par_distance(
        coords_commune,
        services_coords_indexed,
        RAYON_PREFILTRE_KM
    )

    print(f"      {len(indices_proches)} services dans le rayon (sur {len(df_services)})")

    if len(indices_proches) == 0:
        return {
            'commune': commune,
            'nb_services_20min': 0,
            'nb_services_prefiltres': 0,
            'methode': 'geodesique_seul'
        }

    # Étape 2: Calcul OSRM pour services pré-filtrés
    print(f"    Etape 2: Calcul OSRM pour services pre-filtres...")

    services_prefiltres = df_services.iloc[indices_proches]
    coords_services = [
        (row['longitude'], row['latitude'])
        for _, row in services_prefiltres.iterrows()
    ]

    temps_trajets = calculer_temps_osrm_batch(coords_commune, coords_services)

    # Étape 3: Compter services < 20 min
    nb_accessibles = sum(1 for t in temps_trajets if t is not None and t <= SEUIL_TEMPS_MIN)

    print(f"      -> {nb_accessibles} services accessibles en < 20 min")

    return {
        'commune': commune,
        'nb_services_20min': nb_accessibles,
        'nb_services_prefiltres': len(indices_proches),
        'nb_services_osrm_ok': sum(1 for t in temps_trajets if t is not None),
        'methode': 'optimise'
    }


def main():
    """Fonction principale"""
    print("="*80)
    print("TEST VERSION OPTIMISÉE - CALCUL SERVICES ACCESSIBLES")
    print("="*80)
    print(f"\nCommunes à tester: {', '.join(COMMUNES_TEST)}")
    print(f"Stratégie: Pré-filtrage géodésique ({RAYON_PREFILTRE_KM} km) + OSRM")

    # Charger les données
    print("\n1. Chargement des données...")

    # Communes
    if not Path(COORDS_FILE).exists():
        print(f"[ERREUR] Fichier {COORDS_FILE} introuvable")
        return

    df_communes = pd.read_csv(COORDS_FILE)
    print(f"  {len(df_communes)} communes chargées")

    # Services BPE
    if not Path(BPE_FILE).exists():
        print(f"[ERREUR] Fichier {BPE_FILE} introuvable")
        print("  Exécutez d'abord services_accessibles_20min.py pour télécharger la BPE")
        return

    df_services = pd.read_csv(BPE_FILE)

    # Vérifier/ajouter coordonnées (gérer majuscules/minuscules)
    if 'LONGITUDE' in df_services.columns and 'LATITUDE' in df_services.columns:
        df_services = df_services.rename(columns={'LONGITUDE': 'longitude', 'LATITUDE': 'latitude'})
    elif 'longitude' not in df_services.columns or 'latitude' not in df_services.columns:
        print("  [ATTENTION] Coordonnées manquantes dans BPE")
        return

    # Filtrer services avec coordonnées valides
    df_services = df_services.dropna(subset=['longitude', 'latitude'])
    print(f"  {len(df_services)} services avec coordonnées")

    # Checkpoint manager
    checkpoint = CheckpointManager(CHECKPOINT_FILE)

    # Résultats
    resultats = checkpoint.data.get('resultats', [])
    communes_traitees = checkpoint.data.get('communes_traitees', set())

    # Calcul pour chaque commune
    print("\n2. Calcul des services accessibles...")

    for commune_nom in COMMUNES_TEST:
        if commune_nom in communes_traitees:
            print(f"\n  {commune_nom} [DEJA TRAITE - SKIP]")
            continue

        # Trouver la commune
        commune_row = df_communes[df_communes['nom_commune'] == commune_nom]

        if commune_row.empty:
            print(f"\n  {commune_nom} [NON TROUVÉ]")
            continue

        coords = (
            commune_row['longitude'].values[0],
            commune_row['latitude'].values[0]
        )

        # Calculer
        try:
            resultat = calculer_services_pour_commune(commune_nom, coords, df_services)
            resultats.append(resultat)

            # Checkpoint
            checkpoint.save(resultats, commune_nom)

        except Exception as e:
            print(f"    [ERREUR] {e}")
            continue

    # Sauvegarder résultats finaux
    print("\n3. Sauvegarde des résultats...")

    df_resultats = pd.DataFrame(resultats)
    df_resultats.to_csv(OUTPUT_FILE, index=False)
    print(f"  [OK] {OUTPUT_FILE}")

    # Comparaison avec résultats actuels
    print("\n4. Comparaison avec résultats actuels...")

    if Path(COMPARAISON_FILE).exists():
        df_actuel = pd.read_csv(COMPARAISON_FILE)

        print("\n" + "="*80)
        print("COMPARAISON DES RÉSULTATS")
        print("="*80)

        for _, row in df_resultats.iterrows():
            commune = row['commune']
            nb_nouveau = row['nb_services_20min']

            # Trouver dans ancien
            ancien_row = df_actuel[df_actuel['code_commune'].str.contains(commune, case=False, na=False) |
                                   df_actuel['nom_commune'].str.contains(commune, case=False, na=False)]

            if not ancien_row.empty:
                nb_ancien = ancien_row['nb_services_20min'].values[0]
                diff = nb_nouveau - nb_ancien
                diff_pct = (diff / nb_ancien * 100) if nb_ancien > 0 else 0

                print(f"\n{commune:20s}:")
                print(f"  Ancien (simplifié):  {nb_ancien:5d} services")
                print(f"  Nouveau (optimisé):  {nb_nouveau:5d} services")
                print(f"  Différence:          {diff:+5d} ({diff_pct:+.1f}%)")
            else:
                print(f"\n{commune:20s}: {nb_nouveau} services (pas de comparaison)")

    # Supprimer checkpoint
    checkpoint.delete()

    print("\n" + "="*80)
    print("TEST TERMINÉ")
    print("="*80)
    print(f"\nRésultats: {OUTPUT_FILE}")
    print("\nSi les résultats sont satisfaisants, on peut:")
    print("  1. Appliquer à toutes les communes")
    print("  2. Ajuster les paramètres (rayon, batch size, etc.)")


if __name__ == "__main__":
    main()
