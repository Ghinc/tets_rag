#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtrage des routes nationales et departementales du reseau routier de Corse
"""

import geopandas as gpd
import os
import sys

# Configuration pour eviter les erreurs d'encodage sur Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
input_file = "stage_ambroise/Données/reseau-routier-de-corse.shp"
output_file = "stage_ambroise/Code/WEB/routes.geojson"

print(f"Chargement du reseau routier depuis {input_file}...")
routes = gpd.read_file(input_file)

print(f"Nombre total de routes: {len(routes)}")
print(f"\nColonnes disponibles: {routes.columns.tolist()}")

# Examiner les valeurs de class_adm
print("\n=== Classification administrative (class_adm) ===")
try:
    print(routes['class_adm'].value_counts())
except Exception as e:
    print(f"Erreur: {e}")
    # Essayer de voir les valeurs uniques
    print("Valeurs uniques:", routes['class_adm'].unique())

# Examiner aussi num_route pour identifier les N et D
print("\n=== Numeros de routes (num_route) - top 30 ===")
try:
    print(routes['num_route'].value_counts().head(30))
except Exception as e:
    print(f"Erreur: {e}")

# Filtrer les routes nationales et departementales
# On cherche les routes avec class_adm contenant "Nationale" ou "Departementale"
# Ou avec num_route commencant par N ou D

routes_nat_dep = routes[
    (routes['class_adm'].str.contains('National', case=False, na=False)) |
    (routes['class_adm'].str.contains('D.partemental', case=False, na=False)) |
    (routes['class_adm'].str.contains('Departemental', case=False, na=False)) |
    (routes['num_route'].str.startswith('N', na=False)) |
    (routes['num_route'].str.startswith('D', na=False))
].copy()

print(f"\n=== Resultats du filtrage ===")
print(f"Routes nationales et departementales: {len(routes_nat_dep)}")
print(f"Routes supprimees: {len(routes) - len(routes_nat_dep)}")
print(f"Pourcentage conserve: {len(routes_nat_dep)/len(routes)*100:.1f}%")

if len(routes_nat_dep) > 0:
    print("\n=== Distribution des routes conservees ===")
    print("Par class_adm:")
    print(routes_nat_dep['class_adm'].value_counts())
    print("\nPar num_route (top 20):")
    print(routes_nat_dep['num_route'].value_counts().head(20))

    # Sauvegarder
    print(f"\nSauvegarde dans {output_file}...")
    # Convertir en WGS84 si necessaire
    if routes_nat_dep.crs.to_epsg() != 4326:
        routes_nat_dep = routes_nat_dep.to_crs(epsg=4326)

    routes_nat_dep.to_file(output_file, driver='GeoJSON')

    # Verifier la taille du fichier
    taille_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Taille du fichier: {taille_mb:.2f} MB")
    print("Termine!")
else:
    print("\nAUCUNE route nationale ou departementale trouvee!")
    print("Verification manuelle necessaire des criteres de filtrage.")
