"""
Script pour exporter les clusters LISA en JSON pour l'interface web
Calcule la VRAIE analyse LISA avec autocorrélation spatiale
SEED FIXÉ pour reproductibilité
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import json
import requests
from libpysal.weights import Queen
from esda.moran import Moran_Local, Moran

# Fixer le seed pour reproductibilité des permutations de Monte Carlo
SEED = 42
np.random.seed(SEED)


def charger_donnees_excel(fichier='../Python/oppchovec_normalisation_elementaire_trie.xlsx'):
    """Charge les données depuis le fichier Excel"""
    print(f"Chargement des données depuis {fichier}...")
    df = pd.read_excel(fichier, sheet_name='Synthese')
    print(f"  {len(df)} communes chargées")
    return df


def recuperer_geometries_communes():
    """Récupère les géométries des communes de Corse depuis l'API"""
    print("\nRécupération des géométries des communes de Corse...")

    departements = ['2A', '2B']
    communes_list = []

    for dept in departements:
        print(f"  Téléchargement du département {dept}...")
        url = f"https://geo.api.gouv.fr/departements/{dept}/communes?format=geojson&geometry=contour"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(f"    {len(data['features'])} communes trouvées")
            communes_list.extend(data['features'])
        else:
            print(f"    Erreur: {response.status_code}")

    gdf = gpd.GeoDataFrame.from_features(communes_list, crs='EPSG:4326')
    print(f"  Total: {len(gdf)} communes chargées")

    return gdf


def joindre_donnees(df_data, gdf_geo):
    """Joint les données Excel avec les géométries"""
    print("\nJointure des données...")
    gdf = gdf_geo.merge(df_data, left_on='nom', right_on='Zone', how='inner')
    print(f"  {len(gdf)} communes après jointure")

    print("\nStatistiques OppChoVec_1_10:")
    print(gdf['OppChoVec_1_10'].describe())

    return gdf


def calculer_lisa(gdf, variable='OppChoVec_1_10'):
    """Calcule l'analyse LISA (vraie autocorrélation spatiale)"""
    print(f"\nCalcul LISA pour '{variable}'...")

    # Projeter en Lambert 93
    gdf_proj = gdf.to_crs('EPSG:2154')

    # Créer matrice de poids (Queen contiguity)
    w = Queen.from_dataframe(gdf_proj)
    w.transform = 'r'  # Row-standardized

    print(f"  Voisins moyens: {w.mean_neighbors:.2f}")
    print(f"  Voisins min: {w.min_neighbors}, max: {w.max_neighbors}")

    # Extraire les valeurs
    y = gdf[variable].values

    # Moran global
    moran_global = Moran(y, w)
    print(f"\nMoran global I = {moran_global.I:.4f} (p = {moran_global.p_sim:.4f})")

    # Moran local (LISA)
    lisa = Moran_Local(y, w)

    # Ajouter résultats au GeoDataFrame
    gdf['lisa_I'] = lisa.Is  # Indicateur local
    gdf['lisa_p'] = lisa.p_sim  # P-value
    gdf['lisa_q'] = lisa.q  # Quadrant
    gdf['lisa_sig'] = lisa.p_sim < 0.05  # Significatif à 5%

    # Classifier les clusters
    gdf['lisa_cluster'] = 'Non significatif'
    gdf.loc[(gdf['lisa_q'] == 1) & (gdf['lisa_sig']), 'lisa_cluster'] = 'HH (High-High)'
    gdf.loc[(gdf['lisa_q'] == 2) & (gdf['lisa_sig']), 'lisa_cluster'] = 'LH (Low-High)'
    gdf.loc[(gdf['lisa_q'] == 3) & (gdf['lisa_sig']), 'lisa_cluster'] = 'LL (Low-Low)'
    gdf.loc[(gdf['lisa_q'] == 4) & (gdf['lisa_sig']), 'lisa_cluster'] = 'HL (High-Low)'

    print(f"\nCommunes significatives: {gdf['lisa_sig'].sum()} ({gdf['lisa_sig'].sum()/len(gdf)*100:.1f}%)")
    print("\nRépartition par cluster:")
    print(gdf['lisa_cluster'].value_counts())

    return gdf, moran_global


def exporter_json(gdf, moran_global, fichier_sortie='lisa_clusters.json'):
    """Exporte les clusters LISA en JSON"""
    print(f"\nExport vers {fichier_sortie}...")

    # Créer dictionnaire des clusters par commune
    clusters = {}
    for idx, row in gdf.iterrows():
        clusters[row['nom']] = {
            'cluster': row['lisa_cluster'],
            'lisa_I': float(row['lisa_I']),
            'p_value': float(row['lisa_p']),
            'significatif': bool(row['lisa_sig']),
            'oppchovec': float(row['OppChoVec_1_10'])
        }

    # Créer structure complète avec métadonnées
    resultat = {
        'metadata': {
            'description': 'Analyse LISA (Local Indicators of Spatial Association) pour les communes de Corse',
            'variable': 'OppChoVec_1_10',
            'methode': 'Moran Local avec matrice Queen contiguity',
            'moran_global_I': float(moran_global.I),
            'moran_global_p': float(moran_global.p_sim),
            'seuil_significativite': 0.05,
            'nb_communes': len(gdf),
            'nb_significatives': int(gdf['lisa_sig'].sum()),
            'pourcent_significatives': float(gdf['lisa_sig'].sum() / len(gdf) * 100)
        },
        'statistiques': {
            'HH (High-High)': int((gdf['lisa_cluster'] == 'HH (High-High)').sum()),
            'LL (Low-Low)': int((gdf['lisa_cluster'] == 'LL (Low-Low)').sum()),
            'HL (High-Low)': int((gdf['lisa_cluster'] == 'HL (High-Low)').sum()),
            'LH (Low-High)': int((gdf['lisa_cluster'] == 'LH (Low-High)').sum()),
            'Non significatif': int((gdf['lisa_cluster'] == 'Non significatif').sum())
        },
        'clusters': clusters
    }

    # Sauvegarder
    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        json.dump(resultat, f, ensure_ascii=False, indent=2)

    print(f"  OK - {len(clusters)} communes exportees")
    print(f"  OK - Moran I global: {moran_global.I:.4f}")
    print(f"  OK - Fichier: {fichier_sortie}")


def main():
    """Fonction principale"""
    print("=" * 60)
    print("EXPORT DES CLUSTERS LISA POUR INTERFACE WEB")
    print("=" * 60)

    # 1. Charger les données
    df = charger_donnees_excel()

    # 2. Récupérer géométries
    gdf_geo = recuperer_geometries_communes()

    # 3. Joindre
    gdf = joindre_donnees(df, gdf_geo)

    # 4. Calculer LISA
    gdf, moran_global = calculer_lisa(gdf)

    # 5. Exporter JSON
    exporter_json(gdf, moran_global)

    print("\n" + "=" * 60)
    print("EXPORT TERMINE AVEC SUCCES")
    print("=" * 60)


if __name__ == '__main__':
    main()
