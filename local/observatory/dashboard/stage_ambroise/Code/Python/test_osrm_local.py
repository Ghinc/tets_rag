"""
Script de test pour vérifier que le serveur OSRM local fonctionne correctement.

Lance des tests basiques avant de lancer le calcul complet.
"""

import requests
import time


def test_osrm_local():
    """
    Teste la connexion au serveur OSRM local
    """
    print("="*80)
    print("TEST SERVEUR OSRM LOCAL")
    print("="*80)
    print()

    # 1. Test de connexion basique
    print("[1/3] Test de connexion au serveur...")
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        print(f"  ✓ Serveur accessible (code {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("  ✗ ERREUR: Impossible de se connecter à localhost:5000")
        print("  → Le serveur OSRM est-il lancé ?")
        print("  → Lancez start_osrm_server.bat dans un autre terminal")
        return False
    except Exception as e:
        print(f"  ✗ ERREUR: {e}")
        return False

    print()

    # 2. Test de calcul simple (Ajaccio -> Bastia)
    print("[2/3] Test calcul de trajet (Ajaccio → Bastia)...")
    try:
        # Coordonnées: Ajaccio (8.7379, 41.9270) et Bastia (9.4496, 42.7027)
        url = "http://localhost:5000/table/v1/driving/8.7379,41.9270;9.4496,42.7027?sources=0&destinations=1"

        start = time.time()
        response = requests.get(url, timeout=10)
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()

            if data.get('code') == 'Ok':
                duration = data['durations'][0][0]  # en secondes
                duration_min = duration / 60.0

                print(f"  ✓ Calcul réussi en {elapsed*1000:.0f} ms")
                print(f"  ✓ Temps de trajet Ajaccio → Bastia: {duration_min:.1f} minutes")
                print(f"    (distance réelle ~150 km par N193, ~2h-2h30 en réalité)")

                if duration_min < 60 or duration_min > 200:
                    print(f"  ⚠ ATTENTION: Le temps semble anormal ({duration_min:.1f} min)")
                    print("    Vérifiez que les données OSM Corse sont bien chargées")

            else:
                print(f"  ✗ ERREUR OSRM: {data.get('code')}")
                print(f"     Message: {data.get('message', 'N/A')}")
                return False
        else:
            print(f"  ✗ ERREUR HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"  ✗ ERREUR: {e}")
        return False

    print()

    # 3. Test de calcul matriciel (5 communes × 5 destinations)
    print("[3/3] Test calcul matriciel (5×5 points)...")
    try:
        # 5 communes corses (sample)
        coords = [
            "8.7379,41.9270",  # Ajaccio
            "9.4496,42.7027",  # Bastia
            "9.1519,42.3093",  # Corte
            "8.7435,41.5816",  # Bonifacio
            "8.8767,42.5524",  # Calvi
        ]

        coords_str = ";".join(coords)
        url = f"http://localhost:5000/table/v1/driving/{coords_str}?sources=0;1;2;3;4&destinations=0;1;2;3;4"

        start = time.time()
        response = requests.get(url, timeout=10)
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()

            if data.get('code') == 'Ok':
                durations = data['durations']
                print(f"  ✓ Calcul matriciel réussi en {elapsed*1000:.0f} ms")
                print(f"  ✓ Matrice 5×5 = 25 trajets calculés")

                # Afficher quelques exemples
                communes = ["Ajaccio", "Bastia", "Corte", "Bonifacio", "Calvi"]
                print(f"\n  Exemples de temps de trajet (en minutes):")
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            temps = durations[i][j] / 60.0
                            print(f"    {communes[i]:12s} → {communes[j]:12s} : {temps:6.1f} min")

            else:
                print(f"  ✗ ERREUR OSRM: {data.get('code')}")
                return False
        else:
            print(f"  ✗ ERREUR HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"  ✗ ERREUR: {e}")
        return False

    print()
    print("="*80)
    print("✓ TOUS LES TESTS SONT REUSSIS !")
    print("="*80)
    print()
    print("Vous pouvez maintenant lancer:")
    print("  python services_accessibles_osrm_local.py")
    print()

    return True


if __name__ == "__main__":
    success = test_osrm_local()

    if not success:
        print()
        print("="*80)
        print("TESTS ECHOUES")
        print("="*80)
        print()
        print("Pour lancer le serveur OSRM local:")
        print("  1. Ouvrir un nouveau terminal")
        print("  2. Exécuter: start_osrm_server.bat")
        print("  3. Attendre que le serveur affiche 'running and waiting for requests'")
        print("  4. Relancer ce test")
        print()
