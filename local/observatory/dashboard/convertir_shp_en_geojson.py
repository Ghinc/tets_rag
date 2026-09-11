#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour convertir un fichier Shapefile (.shp) en GeoJSON
Usage: python convertir_shp_en_geojson.py input.shp output.geojson
"""

import sys
import os
import geopandas as gpd

# Configuration pour l'encodage sur Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def convertir_shp_vers_geojson(fichier_shp, fichier_geojson):
    """
    Convertit un fichier Shapefile en GeoJSON

    Args:
        fichier_shp: Chemin vers le fichier .shp
        fichier_geojson: Chemin de sortie pour le fichier .geojson
    """
    print(f"[*] Lecture du fichier Shapefile: {fichier_shp}")

    # Lire le fichier Shapefile
    gdf = gpd.read_file(fichier_shp)

    print(f"[OK] {len(gdf)} entites geographiques chargees")
    print(f"[INFO] Type de geometrie: {gdf.geometry.geom_type.unique()}")
    print(f"[INFO] Systeme de coordonnees: {gdf.crs}")

    # Reprojeter en WGS84 (EPSG:4326) si nécessaire pour compatibilité web
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        print(f"[*] Reprojection de {gdf.crs.to_epsg()} vers WGS84 (EPSG:4326)")
        gdf = gdf.to_crs(epsg=4326)

    # Afficher les colonnes disponibles
    print(f"[INFO] Colonnes disponibles: {list(gdf.columns)}")

    # Sauvegarder en GeoJSON
    print(f"[*] Sauvegarde en GeoJSON: {fichier_geojson}")
    gdf.to_file(fichier_geojson, driver='GeoJSON')

    # Afficher la taille du fichier
    taille = os.path.getsize(fichier_geojson) / (1024 * 1024)  # En Mo
    print(f"[OK] Conversion reussie ! Taille: {taille:.2f} Mo")

    return gdf

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convertir_shp_en_geojson.py input.shp output.geojson")
        print("\nExemple:")
        print('  python convertir_shp_en_geojson.py "routes.shp" "routes.geojson"')
        sys.exit(1)

    fichier_shp = sys.argv[1]
    fichier_geojson = sys.argv[2]

    try:
        convertir_shp_vers_geojson(fichier_shp, fichier_geojson)
    except Exception as e:
        print(f"[ERREUR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
