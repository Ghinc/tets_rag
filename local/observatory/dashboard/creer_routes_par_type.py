#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation de fichiers GeoJSON separes par type de route
- Routes nationales
- Routes departementales
- Routes communales
- Toutes les routes
"""

import geopandas as gpd
import os
import sys

# Configuration pour eviter les erreurs d'encodage sur Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
input_file = "stage_ambroise/Données/reseau-routier-de-corse.shp"
output_dir = "stage_ambroise/Code/WEB"

print(f"Chargement du reseau routier depuis {input_file}...")
routes = gpd.read_file(input_file)

# Convertir en WGS84 si necessaire
if routes.crs.to_epsg() != 4326:
    routes = routes.to_crs(epsg=4326)

print(f"Nombre total de routes: {len(routes)}")

# Filtrer par type
routes_nationales = routes[
    (routes['class_adm'].str.contains('National', case=False, na=False)) |
    (routes['num_route'].str.startswith('N', na=False))
].copy()

routes_departementales = routes[
    (routes['class_adm'].str.contains('D.partemental', case=False, na=False)) |
    (routes['class_adm'].str.contains('Departemental', case=False, na=False)) |
    (routes['num_route'].str.startswith('D', na=False))
].copy()

routes_communales = routes[
    (routes['class_adm'].str.contains('Sans objet', case=False, na=False))
].copy()

# Filtrer les routes maritimes pour toutes les categories
print("\nFiltrage des routes maritimes...")
communes = gpd.read_file("stage_ambroise/Données/Commune_Corse.geojson")
if communes.crs.to_epsg() != 4326:
    communes = communes.to_crs(epsg=4326)

zone_terrestre = communes.geometry.buffer(0.01).union_all()

def filtrer_maritimes(routes_gdf):
    if len(routes_gdf) == 0:
        return routes_gdf
    routes_terrestres = routes_gdf[routes_gdf.geometry.intersects(zone_terrestre)].copy()
    nb_supprimees = len(routes_gdf) - len(routes_terrestres)
    print(f"  Routes maritimes supprimees: {nb_supprimees}")
    return routes_terrestres

routes_nationales = filtrer_maritimes(routes_nationales)
routes_departementales = filtrer_maritimes(routes_departementales)
routes_communales = filtrer_maritimes(routes_communales)
routes_toutes = filtrer_maritimes(routes)

# Afficher les statistiques
print(f"\n=== Statistiques ===")
print(f"Routes nationales: {len(routes_nationales)}")
print(f"Routes departementales: {len(routes_departementales)}")
print(f"Routes communales: {len(routes_communales)}")
print(f"Total (sans maritimes): {len(routes_toutes)}")

# Sauvegarder chaque type
fichiers = {
    'routes_nationales.geojson': routes_nationales,
    'routes_departementales.geojson': routes_departementales,
    'routes_communales.geojson': routes_communales,
    'routes_toutes.geojson': routes_toutes
}

for nom_fichier, gdf in fichiers.items():
    chemin = os.path.join(output_dir, nom_fichier)
    print(f"\nSauvegarde de {nom_fichier}...")
    gdf.to_file(chemin, driver='GeoJSON')
    taille_mb = os.path.getsize(chemin) / (1024 * 1024)
    print(f"  Taille: {taille_mb:.2f} MB ({len(gdf)} routes)")

print("\nTermine!")
