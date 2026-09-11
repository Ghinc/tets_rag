"""
Script de comparaison entre les résultats OSRM local et API publique

Compare les fichiers :
- services_accessibles_20min.csv (API publique)
- services_accessibles_20min_local.csv (OSRM local)

Affiche les différences et statistiques pour valider la cohérence.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def comparer_resultats():
    """
    Compare les résultats des deux méthodes de calcul
    """
    print("="*80)
    print("COMPARAISON RESULTATS API PUBLIQUE vs OSRM LOCAL")
    print("="*80)
    print()

    # Charger les fichiers
    fichier_api = "services_accessibles_20min.csv"
    fichier_local = "services_accessibles_20min_local.csv"

    # Vérifier que les fichiers existent
    if not Path(fichier_api).exists():
        print(f"❌ Fichier {fichier_api} non trouvé")
        print("   Lancez d'abord: python services_accessibles_20min.py")
        return

    if not Path(fichier_local).exists():
        print(f"❌ Fichier {fichier_local} non trouvé")
        print("   Lancez d'abord: python services_accessibles_osrm_local.py")
        return

    print(f"✓ Chargement des fichiers...")
    df_api = pd.read_csv(fichier_api, encoding='utf-8')
    df_local = pd.read_csv(fichier_local, encoding='utf-8')

    print(f"  API publique : {len(df_api)} communes")
    print(f"  OSRM local   : {len(df_local)} communes")
    print()

    # Fusionner sur le code commune
    df_compare = pd.merge(
        df_api,
        df_local,
        on='code_commune',
        suffixes=('_api', '_local')
    )

    print(f"✓ {len(df_compare)} communes en commun")
    print()

    # 1. Comparer le nombre de services accessibles
    print("-"*80)
    print("1. NOMBRE DE SERVICES ACCESSIBLES (<20 min)")
    print("-"*80)

    df_compare['diff_services'] = df_compare['nb_services_20min_local'] - df_compare['nb_services_20min_api']
    df_compare['diff_services_pct'] = (df_compare['diff_services'] / df_compare['nb_services_20min_api']) * 100

    print(f"\nDifférence moyenne : {df_compare['diff_services'].mean():.1f} services")
    print(f"Différence médiane : {df_compare['diff_services'].median():.1f} services")
    print(f"Différence en % : {df_compare['diff_services_pct'].mean():.1f}%")

    print(f"\nÉcart-type : {df_compare['diff_services'].std():.1f}")
    print(f"Min : {df_compare['diff_services'].min():.0f}")
    print(f"Max : {df_compare['diff_services'].max():.0f}")

    # Communes avec les plus grandes différences
    print("\n📊 Top 10 communes avec les plus grandes différences (en valeur absolue):")
    df_compare['diff_abs'] = df_compare['diff_services'].abs()
    top_diff = df_compare.nlargest(10, 'diff_abs')[['nom_commune_api', 'nb_services_20min_api', 'nb_services_20min_local', 'diff_services', 'diff_services_pct']]

    for i, (idx, row) in enumerate(top_diff.iterrows(), 1):
        print(f"  {i:2d}. {row['nom_commune_api']:25s} : API={row['nb_services_20min_api']:5.0f} | Local={row['nb_services_20min_local']:5.0f} | Diff={row['diff_services']:+5.0f} ({row['diff_services_pct']:+5.1f}%)")

    # 2. Comparer le temps vers le service le plus proche
    print()
    print("-"*80)
    print("2. TEMPS VERS LE SERVICE LE PLUS PROCHE (minutes)")
    print("-"*80)

    df_compare_temps = df_compare.dropna(subset=['temps_service_plus_proche_min_api', 'temps_service_plus_proche_min_local'])

    if len(df_compare_temps) > 0:
        df_compare_temps['diff_temps_proche'] = df_compare_temps['temps_service_plus_proche_min_local'] - df_compare_temps['temps_service_plus_proche_min_api']

        print(f"\nDifférence moyenne : {df_compare_temps['diff_temps_proche'].mean():.2f} minutes")
        print(f"Différence médiane : {df_compare_temps['diff_temps_proche'].median():.2f} minutes")

        print(f"\nÉcart-type : {df_compare_temps['diff_temps_proche'].std():.2f}")
        print(f"Min : {df_compare_temps['diff_temps_proche'].min():.2f}")
        print(f"Max : {df_compare_temps['diff_temps_proche'].max():.2f}")

    # 3. Corrélation
    print()
    print("-"*80)
    print("3. CORRELATION ENTRE LES DEUX METHODES")
    print("-"*80)

    correlation_services = df_compare['nb_services_20min_api'].corr(df_compare['nb_services_20min_local'])
    print(f"\nCorrélation (nombre de services) : {correlation_services:.4f}")

    if correlation_services > 0.95:
        print("  ✓ Très forte corrélation - Les résultats sont très cohérents")
    elif correlation_services > 0.90:
        print("  ✓ Forte corrélation - Les résultats sont cohérents")
    elif correlation_services > 0.80:
        print("  ⚠ Corrélation modérée - Quelques différences notables")
    else:
        print("  ❌ Corrélation faible - Vérifier la configuration")

    if len(df_compare_temps) > 0:
        correlation_temps = df_compare_temps['temps_service_plus_proche_min_api'].corr(df_compare_temps['temps_service_plus_proche_min_local'])
        print(f"Corrélation (temps au plus proche) : {correlation_temps:.4f}")

    # 4. Distribution des différences
    print()
    print("-"*80)
    print("4. DISTRIBUTION DES DIFFERENCES")
    print("-"*80)

    bins = [(-float('inf'), -100), (-100, -50), (-50, -10), (-10, 10), (10, 50), (50, 100), (100, float('inf'))]
    bin_labels = ["<-100", "-100 à -50", "-50 à -10", "-10 à +10", "+10 à +50", "+50 à +100", ">+100"]

    print("\nRépartition des communes par différence de services:")
    for (low, high), label in zip(bins, bin_labels):
        if low == -float('inf'):
            count = (df_compare['diff_services'] < high).sum()
        elif high == float('inf'):
            count = (df_compare['diff_services'] >= low).sum()
        else:
            count = ((df_compare['diff_services'] >= low) & (df_compare['diff_services'] < high)).sum()

        pct = (count / len(df_compare)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:15s} : {count:3d} communes ({pct:5.1f}%) {bar}")

    # 5. Résumé et recommandations
    print()
    print("="*80)
    print("RESUME")
    print("="*80)

    diff_moy_abs = df_compare['diff_services'].abs().mean()

    if diff_moy_abs < 50 and correlation_services > 0.95:
        print("\n✅ VALIDATION REUSSIE")
        print("Les résultats des deux méthodes sont très cohérents.")
        print("Vous pouvez utiliser OSRM local en toute confiance.")

    elif diff_moy_abs < 100 and correlation_services > 0.90:
        print("\n✓ VALIDATION OK")
        print("Les résultats sont globalement cohérents avec quelques différences mineures.")
        print("Ces différences peuvent être dues à :")
        print("  - Des mises à jour différentes des données OpenStreetMap")
        print("  - Des algorithmes de routage légèrement différents")
        print("OSRM local est utilisable.")

    else:
        print("\n⚠ DIFFERENCES IMPORTANTES DETECTEES")
        print("Les résultats montrent des différences notables.")
        print("Vérifications recommandées :")
        print("  1. Les données OSM Corse sont-elles bien chargées ?")
        print("  2. Le serveur OSRM est-il bien configuré pour la Corse ?")
        print("  3. Les deux fichiers ont-ils été générés avec les mêmes coordonnées ?")

    print()

    # 6. Export du rapport de comparaison
    fichier_comparaison = "comparaison_api_vs_local.csv"
    colonnes_export = [
        'code_commune',
        'nom_commune_api',
        'nb_services_20min_api',
        'nb_services_20min_local',
        'diff_services',
        'diff_services_pct',
        'temps_service_plus_proche_min_api',
        'temps_service_plus_proche_min_local'
    ]

    df_compare[colonnes_export].to_csv(fichier_comparaison, index=False, encoding='utf-8')
    print(f"📁 Rapport détaillé exporté : {fichier_comparaison}")
    print()


if __name__ == "__main__":
    comparer_resultats()
