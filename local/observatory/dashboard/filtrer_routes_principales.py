#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtrage des routes principales du réseau routier de Corse
Conserve uniquement les routes de longueur significative (> 1km)
"""

import geopandas as gpd
import os

# Configuration
input_file = "stage_ambroise/Code/WEB/routes.geojson"
output_file = "stage_ambroise/Code/WEB/routes.geojson"
seuil_longueur_km = 1.0  # Longueur minimale en km

print(f"Chargement du reseau routier depuis {input_file}...")
routes = gpd.read_file(input_file)

print(f"Nombre total de routes: {len(routes)}")

# Calculer la longueur en km (en supposant WGS84)
# Pour WGS84, on convertit en projection metrique pour calculer la longueur
routes_utm = routes.to_crs(epsg=32632)  # UTM zone 32N pour la Corse
routes['longueur_km'] = routes_utm.geometry.length / 1000

print(f"\nStatistiques de longueur:")
print(f"  Longueur moyenne: {routes['longueur_km'].mean():.2f} km")
print(f"  Longueur médiane: {routes['longueur_km'].median():.2f} km")
print(f"  Longueur min: {routes['longueur_km'].min():.2f} km")
print(f"  Longueur max: {routes['longueur_km'].max():.2f} km")

# Filtrer les routes principales
routes_principales = routes[routes['longueur_km'] >= seuil_longueur_km].copy()

# Supprimer la colonne temporaire de longueur
routes_principales = routes_principales.drop(columns=['longueur_km'])

print(f"\nFiltrage avec seuil de {seuil_longueur_km} km:")
print(f"  Routes conservees: {len(routes_principales)}")
print(f"  Routes supprimees: {len(routes) - len(routes_principales)}")
print(f"  Pourcentage conserve: {len(routes_principales)/len(routes)*100:.1f}%")

# Sauvegarder
print(f"\nSauvegarde dans {output_file}...")
routes_principales.to_file(output_file, driver='GeoJSON')

# Verifier la taille du fichier
taille_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"Taille du fichier: {taille_mb:.2f} MB")
print("Termine !")
