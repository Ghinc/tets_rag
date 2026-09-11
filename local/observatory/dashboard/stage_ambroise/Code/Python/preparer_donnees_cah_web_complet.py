"""
Prépare les données de la CAH pour le site web:
- Copie les graphiques des écarts standardisés (3 et 5 clusters)
- Génère les fichiers JSON avec les clusters pour la carte (3 et 5 clusters)
"""

import json
import shutil
import pandas as pd
import geopandas as gpd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

def copier_graphiques():
    """Copie les graphiques des écarts standardisés vers le dossier WEB"""
    print("[INFO] Copie des graphiques des ecarts standardises...")

    # Copier le graphique 3 clusters
    source_3 = '../OUTPUT/cah_3d_ecarts_std_3_clusters.png'
    dest_3 = '../WEB/cah_3_clusters_ecarts.png'
    shutil.copy(source_3, dest_3)
    print(f"  [OK] Graphique 3 clusters copie vers {dest_3}")

    # Copier le graphique 5 clusters
    source_5 = '../OUTPUT/cah_3d_ecarts_std_5_clusters.png'
    dest_5 = '../WEB/cah_5_clusters_ecarts.png'
    shutil.copy(source_5, dest_5)
    print(f"  [OK] Graphique 5 clusters copie vers {dest_5}")

def generer_json_clusters(n_clusters, output_filename):
    """Génère le fichier JSON avec les clusters pour la carte"""
    print(f"\n[INFO] Generation du fichier JSON pour {n_clusters} clusters...")

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
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

    print(f"  [OK] {n_clusters} clusters calcules")

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
    for i in range(1, n_clusters + 1):
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
            'description': f'Clusters de la CAH 3D ({n_clusters} clusters)',
            'methode': 'Classification Hierarchique Ascendante',
            'variables': ['Score_Opp_0_10', 'Score_Cho_0_10', 'Score_Vec_0_10'],
            'distance': 'Euclidienne',
            'linkage': 'Ward',
            'n_clusters': n_clusters
        },
        'clusters': resultats,
        'stats': stats_clusters
    }

    # Sauvegarder
    output_path = f'../WEB/{output_filename}'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Fichier JSON sauvegarde: {output_path}")

    # Afficher les stats
    print(f"\n  [INFO] Statistiques des {n_clusters} clusters:")
    for stat in stats_clusters:
        print(f"    Cluster {stat['cluster']} (n={stat['n_communes']}): "
              f"Opp={stat['opp_moyen']:.2f}, Cho={stat['cho_moyen']:.2f}, Vec={stat['vec_moyen']:.2f}")

def main():
    print("=" * 80)
    print("  PREPARATION DES DONNEES CAH POUR LE SITE WEB (3 ET 5 CLUSTERS)")
    print("=" * 80)

    # Vérifier que les graphiques existent
    import os
    if not os.path.exists('../OUTPUT/cah_3d_ecarts_std_5_clusters.png'):
        print("\n[ATTENTION] Le graphique pour 5 clusters n'existe pas encore.")
        print("            Veuillez d'abord executer graphique_ecarts_types_cah.py")
        print("            avec n_clusters_list = [3, 4, 5]")
        return

    copier_graphiques()
    generer_json_clusters(3, 'cah_3_clusters.json')
    generer_json_clusters(5, 'cah_5_clusters.json')

    print("\n" + "=" * 80)
    print("[OK] PREPARATION TERMINEE")
    print("     - Graphiques copies dans WEB/")
    print("     - JSON des clusters generes (3 et 5 clusters)")
    print("=" * 80)

if __name__ == "__main__":
    main()
