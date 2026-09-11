#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour filtrer les routes maritimes et ne garder que les routes terrestres
"""

import sys
import os
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np

# Configuration pour l'encodage sur Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def filtrer_routes_maritimes(fichier_routes, fichier_communes, fichier_sortie):
    """
    Filtre les routes pour ne garder que celles qui sont proches des communes (routes terrestres)

    Args:
        fichier_routes: Chemin vers le GeoJSON des routes
        fichier_communes: Chemin vers le GeoJSON des communes
        fichier_sortie: Chemin de sortie pour le fichier filtré
    """
    print("[*] Chargement des routes...")
    routes = gpd.read_file(fichier_routes)
    print(f"[OK] {len(routes)} routes chargees")

    print("[*] Chargement des communes...")
    communes = gpd.read_file(fichier_communes)
    print(f"[OK] {len(communes)} communes chargees")

    # Créer un buffer autour des communes pour inclure les routes côtières
    # Buffer de 1000m (0.01 degré ~ 1km à cette latitude)
    print("[*] Creation d'une zone tampon autour des terres...")
    zone_terrestre = communes.unary_union.buffer(0.01)

    # Filtrer les routes qui intersectent la zone terrestre
    print("[*] Filtrage des routes terrestres...")
    routes_terrestres = routes[routes.geometry.intersects(zone_terrestre)]

    nb_routes_supprimees = len(routes) - len(routes_terrestres)
    print(f"[INFO] {nb_routes_supprimees} routes maritimes supprimees")
    print(f"[OK] {len(routes_terrestres)} routes terrestres conservees")

    # Sauvegarder
    print(f"[*] Sauvegarde des routes terrestres: {fichier_sortie}")
    routes_terrestres.to_file(fichier_sortie, driver='GeoJSON')

    # Afficher la taille du fichier
    taille = os.path.getsize(fichier_sortie) / (1024 * 1024)
    print(f"[OK] Fichier sauvegarde ! Taille: {taille:.2f} Mo")

    return routes_terrestres

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python filtrer_routes_terrestres.py routes.geojson communes.geojson output.geojson")
        print("\nExemple:")
        print('  python filtrer_routes_terrestres.py "routes.geojson" "Commune_Corse.geojson" "routes_terrestres.geojson"')
        sys.exit(1)

    fichier_routes = sys.argv[1]
    fichier_communes = sys.argv[2]
    fichier_sortie = sys.argv[3]

    try:
        filtrer_routes_maritimes(fichier_routes, fichier_communes, fichier_sortie)
    except Exception as e:
        print(f"[ERREUR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
