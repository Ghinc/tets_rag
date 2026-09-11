"""
Analyse LISA pour OppChoVec avec normalisation 0-10
Seuils de significativité : 1% et 5%
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import json
from libpysal.weights import Queen
from esda.moran import Moran_Local, Moran
import requests

# Fixer le seed pour reproductibilité
SEED = 42
np.random.seed(SEED)


def charger_donnees(fichier='data_pour_lisa_0_10.csv'):
    """Charge les données depuis le fichier CSV"""
    print(f"Chargement des données depuis {fichier}...")
    df = pd.read_csv(fichier)
    print(f"  {len(df)} communes chargées")
    print(f"  Colonnes: {df.columns.tolist()}")
    return df


def recuperer_geometries_communes():
    """Récupère les géométries des communes de Corse depuis l'API"""
    print("\nRécupération des géométries des communes de Corse...")

    departements = ['2A', '2B']
    communes_list = []

    for dept in departements:
        print(f"  Téléchargement des communes du département {dept}...")
        url = f"https://geo.api.gouv.fr/departements/{dept}/communes?format=geojson&geometry=contour"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(f"    {len(data['features'])} communes trouvées")
            communes_list.extend(data['features'])
        else:
            print(f"    Erreur lors du téléchargement: {response.status_code}")

    gdf = gpd.GeoDataFrame.from_features(communes_list, crs='EPSG:4326')
    print(f"  Total: {len(gdf)} communes chargées")

    return gdf


def joindre_donnees(df_data, gdf_geo):
    """Joint les données avec les géométries"""
    print("\nJointure des données...")

    gdf = gdf_geo.merge(df_data, left_on='nom', right_on='Zone', how='inner')
    print(f"  {len(gdf)} communes après jointure")

    return gdf


def calculer_matrice_poids(gdf):
    """Calcule la matrice de poids spatial (contiguïté de type Queen)"""
    print("\nCalcul de la matrice de poids spatial (Queen contiguity)...")

    gdf_proj = gdf.to_crs('EPSG:2154')

    w = Queen.from_dataframe(gdf_proj, use_index=False)
    w.transform = 'r'

    print(f"  Matrice de poids créée")
    print(f"  Nombre moyen de voisins: {w.mean_neighbors:.2f}")
    print(f"  Nombre min de voisins: {w.min_neighbors}")
    print(f"  Nombre max de voisins: {w.max_neighbors}")

    return w, gdf_proj


def calculer_lisa(gdf, w, variable='OppChoVec_0_10', alpha=0.01):
    """
    Calcule l'indicateur LISA (Moran local)

    Args:
        gdf: GeoDataFrame
        w: Matrice de poids spatial
        variable: Nom de la variable à analyser
        alpha: Seuil de significativité (0.01 ou 0.05)

    Returns:
        lisa, gdf, moran_global
    """
    print(f"\nCalcul de l'indicateur LISA pour '{variable}' (alpha={alpha})...")

    y = gdf[variable].values

    # Moran global
    moran_global = Moran(y, w)
    print(f"\nMoran global:")
    print(f"  I = {moran_global.I:.4f}")
    print(f"  p-value = {moran_global.p_sim:.4f}")

    # Moran local
    lisa = Moran_Local(y, w, seed=SEED, permutations=9999)

    # Ajouter les résultats
    suffix = f"_{int(alpha*100)}pct"
    gdf[f'lisa_I{suffix}'] = lisa.Is
    gdf[f'lisa_p{suffix}'] = lisa.p_sim
    gdf[f'lisa_q{suffix}'] = lisa.q
    gdf[f'lisa_sig{suffix}'] = lisa.p_sim < alpha

    # Classification
    gdf[f'lisa_cluster{suffix}'] = 'Non significatif'
    gdf.loc[(gdf[f'lisa_q{suffix}'] == 1) & (gdf[f'lisa_sig{suffix}']), f'lisa_cluster{suffix}'] = 'HH (High-High)'
    gdf.loc[(gdf[f'lisa_q{suffix}'] == 2) & (gdf[f'lisa_sig{suffix}']), f'lisa_cluster{suffix}'] = 'LH (Low-High)'
    gdf.loc[(gdf[f'lisa_q{suffix}'] == 3) & (gdf[f'lisa_sig{suffix}']), f'lisa_cluster{suffix}'] = 'LL (Low-Low)'
    gdf.loc[(gdf[f'lisa_q{suffix}'] == 4) & (gdf[f'lisa_sig{suffix}']), f'lisa_cluster{suffix}'] = 'HL (High-Low)'

    print(f"\nRésultats LISA (alpha={alpha}):")
    print(f"  Communes significatives: {gdf[f'lisa_sig{suffix}'].sum()} ({gdf[f'lisa_sig{suffix}'].sum()/len(gdf)*100:.1f}%)")
    print(f"\nRépartition par type de cluster:")
    print(gdf[f'lisa_cluster{suffix}'].value_counts())

    return lisa, gdf, moran_global


def exporter_pour_web(gdf, variable, alpha_list=[0.01, 0.05]):
    """Exporte les résultats LISA pour l'application web"""
    print(f"\nExport des résultats pour le web...")

    for alpha in alpha_list:
        suffix = f"_{int(alpha*100)}pct"
        output_file = f"lisa_oppchovec_0_10_{int(alpha*100)}pct.json"

        # Créer le dictionnaire de données
        data_dict = {}
        for idx, row in gdf.iterrows():
            commune = row['nom']
            data_dict[commune] = {
                'cluster': row[f'lisa_cluster{suffix}'],
                'p_value': float(row[f'lisa_p{suffix}']),
                'I': float(row[f'lisa_I{suffix}']),
                'significatif': bool(row[f'lisa_sig{suffix}']),
                'quadrant': int(row[f'lisa_q{suffix}'])
            }

        # Sauvegarder
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

        print(f"  [OK] {output_file}")

    return data_dict


def main():
    """Fonction principale"""
    print("=" * 80)
    print("ANALYSE LISA - OPPCHOVEC NORMALISATION 0-10 (1% et 5%)")
    print("=" * 80)
    print()

    # Charger les données
    df = charger_donnees('data_pour_lisa_0_10.csv')

    # Récupérer les géométries
    gdf_geo = recuperer_geometries_communes()

    # Joindre les données
    gdf = joindre_donnees(df, gdf_geo)

    # Calculer la matrice de poids
    w, gdf_proj = calculer_matrice_poids(gdf)

    # LISA à 1%
    print("\n" + "=" * 80)
    print("ANALYSE LISA - SEUIL DE SIGNIFICATIVITE 1%")
    print("=" * 80)
    lisa_1pct, gdf, moran_1pct = calculer_lisa(gdf, w, 'OppChoVec_0_10', alpha=0.01)

    # LISA à 5%
    print("\n" + "=" * 80)
    print("ANALYSE LISA - SEUIL DE SIGNIFICATIVITE 5%")
    print("=" * 80)
    lisa_5pct, gdf, moran_5pct = calculer_lisa(gdf, w, 'OppChoVec_0_10', alpha=0.05)

    # Exporter pour le web
    exporter_pour_web(gdf, 'OppChoVec_0_10', alpha_list=[0.01, 0.05])

    # Sauvegarder le GeoDataFrame complet
    print("\n" + "=" * 80)
    print("SAUVEGARDE DES RESULTATS")
    print("=" * 80)

    # Sauvegarder en CSV
    output_csv = 'lisa_resultats_0_10.csv'
    gdf_export = gdf.drop(columns=['geometry'])
    gdf_export.to_csv(output_csv, index=False)
    print(f"  [OK] {output_csv}")

    # Copier vers WEB
    print("\nCopie vers ../WEB...")
    import shutil
    from pathlib import Path

    web_dir = Path("../WEB")
    if web_dir.exists():
        for alpha in [1, 5]:
            src = f"lisa_oppchovec_0_10_{alpha}pct.json"
            if Path(src).exists():
                shutil.copy(src, web_dir / src)
                print(f"  [OK] {src} -> WEB/")

    print("\n" + "=" * 80)
    print("TRAITEMENT TERMINE")
    print("=" * 80)


if __name__ == "__main__":
    main()
