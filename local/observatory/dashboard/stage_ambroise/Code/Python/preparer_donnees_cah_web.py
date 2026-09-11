"""
Prépare les données de la CAH pour le site web:
- Copie le graphique des écarts standardisés (3 clusters)
- Génère le fichier JSON avec les clusters pour la carte
"""

import json
import shutil
import pandas as pd
import geopandas as gpd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

def copier_graphique():
    """Copie le graphique des écarts standardisés vers le dossier WEB"""
    print("[INFO] Copie du graphique des ecarts standardises...")

    source = '../OUTPUT/cah_3d_ecarts_std_3_clusters.png'
    dest = '../WEB/cah_3_clusters_ecarts.png'

    shutil.copy(source, dest)
    print(f"  [OK] Graphique copie vers {dest}")

def generer_json_clusters():
    """Génère le fichier JSON avec les clusters pour la carte"""
    print("\n[INFO] Generation du fichier JSON des clusters...")

    # Charger les données
    with open('../WEB/data_scores_0_10.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    gdf = gpd.read_file('../WEB/Commune_Corse.geojson')

    df_scores = pd.DataFrame.from_dict(data, orient='index')
    df_scores.index.name = 'commune'
    df_scores.reset_index(inplace=True)

    gdf['nom_clean'] = gdf['nom'].str.strip()
    gdf_merged = gdf.merge(df_scores, left_on='nom_clean', right_on='commune', how='inner')

    print(f"  [OK] {len(gdf_merged)} communes fusionnees")

    # Calculer les clusters
    X = gdf_merged[['Score_Opp_0_10', 'Score_Cho_0_10', 'Score_Vec_0_10']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method='ward')
    clusters = fcluster(Z, t=3, criterion='maxclust')

    print(f"  [OK] 3 clusters calcules")

    # Créer le dictionnaire de résultats
    resultats = {}
    for idx, row in gdf_merged.iterrows():
        commune = row['nom_clean']
        cluster = int(clusters[idx])

        resultats[commune] = {
            'cluster': cluster,
            'Score_Opp': float(row['Score_Opp_0_10']),
            'Score_Cho': float(row['Score_Cho_0_10']),
            'Score_Vec': float(row['Score_Vec_0_10']),
            'OppChoVec': float(row['OppChoVec_0_10'])
        }

    # Calculer les stats par cluster
    stats_clusters = []
    for i in range(1, 4):
        cluster_data = gdf_merged[clusters == i]
        stats_clusters.append({
            'cluster': i,
            'n_communes': len(cluster_data),
            'opp_moyen': float(cluster_data['Score_Opp_0_10'].mean()),
            'cho_moyen': float(cluster_data['Score_Cho_0_10'].mean()),
            'vec_moyen': float(cluster_data['Score_Vec_0_10'].mean()),
            'oppchovec_moyen': float(cluster_data['OppChoVec_0_10'].mean())
        })

    output = {
        'metadata': {
            'description': 'Clusters de la CAH 3D (3 clusters)',
            'methode': 'Classification Hierarchique Ascendante',
            'variables': ['Score_Opp_0_10', 'Score_Cho_0_10', 'Score_Vec_0_10'],
            'distance': 'Euclidienne',
            'linkage': 'Ward',
            'n_clusters': 3
        },
        'clusters': resultats,
        'stats': stats_clusters
    }

    # Sauvegarder
    output_path = '../WEB/cah_3_clusters.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Fichier JSON sauvegarde: {output_path}")

    # Afficher les stats
    print("\n  [INFO] Statistiques des clusters:")
    for stat in stats_clusters:
        print(f"    Cluster {stat['cluster']} (n={stat['n_communes']}): "
              f"Opp={stat['opp_moyen']:.2f}, Cho={stat['cho_moyen']:.2f}, Vec={stat['vec_moyen']:.2f}")

def main():
    print("=" * 80)
    print("  PREPARATION DES DONNEES CAH POUR LE SITE WEB")
    print("=" * 80)

    copier_graphique()
    generer_json_clusters()

    print("\n" + "=" * 80)
    print("[OK] PREPARATION TERMINEE")
    print("     - Graphique copie dans WEB/")
    print("     - JSON des clusters genere")
    print("=" * 80)

if __name__ == "__main__":
    main()
