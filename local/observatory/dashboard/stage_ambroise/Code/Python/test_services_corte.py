"""
Test ultra-simplifié sur CORTE uniquement
Pour valider que l'approche fonctionne
"""

import pandas as pd
import numpy as np
import requests
import time
from math import radians, cos, sin, asin, sqrt


# Configuration
COMMUNE_TEST = 'Corte'
COORDS_FILE = "communes_corse_coordonnees.csv"
BPE_FILE = "bpe_corse.csv"
OSRM_URL = "http://router.project-osrm.org/table/v1/driving/"
SEUIL_TEMPS_MIN = 20
RAYON_PREFILTRE_KM = 15  # Adapté aux routes corses (pas d'autoroutes)
BATCH_SIZE = 15
PAUSE_ENTRE_REQUETES = 2.0


def haversine(lon1, lat1, lon2, lat2):
    """Distance géodésique en km"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def main():
    print("="*80)
    print(f"TEST SIMPLE - {COMMUNE_TEST}")
    print("="*80)

    # Charger commune
    df_communes = pd.read_csv(COORDS_FILE)
    commune_row = df_communes[df_communes['nom_commune'] == COMMUNE_TEST]

    if commune_row.empty:
        print(f"[ERREUR] {COMMUNE_TEST} non trouvée")
        return

    coords_commune = (
        commune_row['longitude'].values[0],
        commune_row['latitude'].values[0]
    )

    print(f"\n{COMMUNE_TEST}: {coords_commune}")

    # Charger services
    df_services = pd.read_csv(BPE_FILE)

    if 'LONGITUDE' in df_services.columns:
        df_services = df_services.rename(columns={'LONGITUDE': 'longitude', 'LATITUDE': 'latitude'})

    df_services = df_services.dropna(subset=['longitude', 'latitude'])
    print(f"\n{len(df_services)} services totaux")

    # Pré-filtrage géodésique
    print(f"\nPré-filtrage (rayon {RAYON_PREFILTRE_KM} km)...")

    distances = []
    for _, row in df_services.iterrows():
        dist = haversine(
            coords_commune[0], coords_commune[1],
            row['longitude'], row['latitude']
        )
        distances.append(dist)

    df_services['distance_km'] = distances
    df_proches = df_services[df_services['distance_km'] <= RAYON_PREFILTRE_KM].copy()

    print(f"  -> {len(df_proches)} services dans le rayon")

    if len(df_proches) == 0:
        print("\nAucun service dans le rayon")
        return

    # Calcul OSRM
    print(f"\nCalcul OSRM par batch de {BATCH_SIZE}...")

    services_accessibles = []
    total_batches = (len(df_proches) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(df_proches), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        print(f"\n  Batch {batch_num}/{total_batches}...")

        batch = df_proches.iloc[i:i+BATCH_SIZE]

        # Construire requête
        coords_batch = [(row['longitude'], row['latitude']) for _, row in batch.iterrows()]
        all_coords = [coords_commune] + coords_batch

        coords_str = ";".join([f"{lon},{lat}" for lon, lat in all_coords])
        url = f"{OSRM_URL}{coords_str}?sources=0&destinations=" + ";".join([str(j) for j in range(1, len(all_coords))])

        try:
            print(f"    Requête OSRM... ", end='')
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if data.get('code') == 'Ok':
                    durations = data.get('durations', [[]])[0]

                    # Compter services accessibles
                    for j, duration in enumerate(durations):
                        if duration is not None:
                            temps_min = duration / 60.0
                            if temps_min <= SEUIL_TEMPS_MIN:
                                services_accessibles.append({
                                    'distance_km': batch.iloc[j]['distance_km'],
                                    'temps_min': temps_min
                                })

                    print(f"OK ({len([d for d in durations if d and d/60 <= SEUIL_TEMPS_MIN])} accessibles)")
                else:
                    print(f"Code: {data.get('code')}")

            elif response.status_code == 429:
                print("Rate limit - pause 10s")
                time.sleep(10)
            else:
                print(f"Status {response.status_code}")

            time.sleep(PAUSE_ENTRE_REQUETES)

        except Exception as e:
            print(f"Erreur: {e}")

    # Résultats
    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)

    print(f"\n{COMMUNE_TEST}:")
    print(f"  Services pré-filtrés (<{RAYON_PREFILTRE_KM}km):  {len(df_proches)}")
    print(f"  Services accessibles (<{SEUIL_TEMPS_MIN}min):   {len(services_accessibles)}")

    if services_accessibles:
        temps = [s['temps_min'] for s in services_accessibles]
        print(f"\nStatistiques temps de trajet:")
        print(f"  Minimum:  {min(temps):.1f} min")
        print(f"  Maximum:  {max(temps):.1f} min")
        print(f"  Moyenne:  {sum(temps)/len(temps):.1f} min")

    # Comparer avec résultats actuels
    print("\n" + "="*80)
    print("COMPARAISON AVEC RÉSULTATS ACTUELS")
    print("="*80)

    try:
        df_actuel = pd.read_csv("services_accessibles_20min.csv")
        ancien_row = df_actuel[df_actuel['nom_commune'] == COMMUNE_TEST]

        if not ancien_row.empty:
            nb_ancien = ancien_row['nb_services_20min'].values[0]
            nb_nouveau = len(services_accessibles)
            diff = nb_nouveau - nb_ancien

            print(f"\n{COMMUNE_TEST}:")
            print(f"  Méthode actuelle (simplifiée):  {nb_ancien} services")
            print(f"  Méthode optimisée (OSRM):       {nb_nouveau} services")
            print(f"  Différence:                      {diff:+d} ({diff/nb_ancien*100:+.1f}%)")

            print("\nInterprétation:")
            if abs(diff/nb_ancien) < 0.1:
                print("  -> Résultats très proches (< 10% différence)")
                print("  -> La méthode simplifiée semble correcte")
            elif diff > 0:
                print("  -> OSRM trouve plus de services")
                print("  -> La méthode simplifiée sous-estime")
            else:
                print("  -> OSRM trouve moins de services")
                print("  -> La méthode simplifiée sur-estime")
        else:
            print(f"{COMMUNE_TEST} non trouvée dans résultats actuels")

    except FileNotFoundError:
        print("Fichier services_accessibles_20min.csv non trouvé")


if __name__ == "__main__":
    main()
