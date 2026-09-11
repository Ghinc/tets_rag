"""
Génère des graphiques montrant l'écart en nombre d'écarts-types à la moyenne
pour chaque dimension (Opp, Cho, Vec) par cluster

Pour 3 et 4 clusters de la CAH 3D
"""

import json
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

def charger_donnees():
    """Charge les données"""
    print("[INFO] Chargement des donnees...")

    with open('../WEB/data_scores_0_10.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    gdf = gpd.read_file('../WEB/Commune_Corse.geojson')

    df_scores = pd.DataFrame.from_dict(data, orient='index')
    df_scores.index.name = 'commune'
    df_scores.reset_index(inplace=True)

    gdf['nom_clean'] = gdf['nom'].str.strip()
    gdf_merged = gdf.merge(df_scores, left_on='nom_clean', right_on='commune', how='inner')

    print(f"  [OK] {len(gdf_merged)} communes fusionnees")

    return gdf_merged

def recalculer_clusters(gdf, n_clusters_list):
    """Recalcule les clusters"""
    print("\n[INFO] Calcul de la CAH 3D...")

    X = gdf[['Score_Opp_0_10', 'Score_Cho_0_10', 'Score_Vec_0_10']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method='ward')

    clusters_dict = {}
    for n in n_clusters_list:
        clusters = fcluster(Z, t=n, criterion='maxclust')
        clusters_dict[n] = clusters
        print(f"  [OK] {n} clusters calcules")

    return clusters_dict, X, scaler

def calculer_ecarts_standardises(gdf, clusters_dict, scaler):
    """Calcule les écarts standardisés pour chaque cluster"""
    print("\n[INFO] Calcul des ecarts standardises...")

    # Moyennes et écarts-types globaux (sur les données originales 0-10)
    dimensions = ['Score_Opp_0_10', 'Score_Cho_0_10', 'Score_Vec_0_10']
    moyennes_globales = {dim: gdf[dim].mean() for dim in dimensions}
    std_globaux = {dim: gdf[dim].std() for dim in dimensions}

    print(f"\n  [INFO] Statistiques globales:")
    for dim in dimensions:
        print(f"    {dim}: moyenne={moyennes_globales[dim]:.2f}, std={std_globaux[dim]:.2f}")

    resultats = {}

    for n, clusters in clusters_dict.items():
        print(f"\n  [INFO] Analyse pour {n} clusters:")
        gdf_temp = gdf.copy()
        gdf_temp['cluster'] = clusters

        ecarts = []

        for i in range(1, n + 1):
            cluster_data = gdf_temp[gdf_temp['cluster'] == i]
            n_communes = len(cluster_data)

            # Calculer les moyennes du cluster
            moyennes_cluster = {dim: cluster_data[dim].mean() for dim in dimensions}

            # Calculer les écarts en nombre d'écarts-types
            ecarts_std = {}
            for dim in dimensions:
                ecart = (moyennes_cluster[dim] - moyennes_globales[dim]) / std_globaux[dim]
                ecarts_std[dim] = ecart

            ecarts.append({
                'cluster': i,
                'n_communes': n_communes,
                'ecart_opp': ecarts_std['Score_Opp_0_10'],
                'ecart_cho': ecarts_std['Score_Cho_0_10'],
                'ecart_vec': ecarts_std['Score_Vec_0_10']
            })

            print(f"    Cluster {i} (n={n_communes}): "
                  f"Opp={ecarts_std['Score_Opp_0_10']:+.2f}std, "
                  f"Cho={ecarts_std['Score_Cho_0_10']:+.2f}std, "
                  f"Vec={ecarts_std['Score_Vec_0_10']:+.2f}std")

        resultats[n] = pd.DataFrame(ecarts)

    return resultats, moyennes_globales, std_globaux

def generer_graphique(resultats, n_clusters, output_path):
    """Génère le graphique des écarts standardisés"""
    print(f"\n[INFO] Generation du graphique pour {n_clusters} clusters...")

    df = resultats[n_clusters]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Préparer les données
    clusters = df['cluster'].values
    x_pos = np.arange(len(clusters))
    width = 0.25

    # Créer les barres
    bars1 = ax.bar(x_pos - width, df['ecart_opp'], width,
                   label='Opportunités', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x_pos, df['ecart_cho'], width,
                   label='Choix', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x_pos + width, df['ecart_vec'], width,
                   label='Vecu', color='#2ecc71', alpha=0.8)

    # Ligne horizontale à 0 (moyenne globale)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Moyenne globale')

    # Ajouter les valeurs sur les barres
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:+.2f}std',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Configuration des axes
    ax.set_ylabel('Ecart a la moyenne globale (en nombre d\'ecarts-types)',
                  fontsize=12, weight='bold')
    ax.set_xlabel('Cluster', fontsize=12, weight='bold')
    ax.set_title(f'Profil standardise des {n_clusters} clusters - CAH 3D\n'
                f'Ecart moyen par dimension (en ecarts-types)',
                fontsize=14, weight='bold', pad=20)

    # Personnaliser les ticks X
    ax.set_xticks(x_pos)
    labels_x = [f"Cluster {c}\n(n={n})" for c, n in zip(df['cluster'], df['n_communes'])]
    ax.set_xticklabels(labels_x)

    # Légende
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Grille
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)

    # Ajuster les marges
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Graphique sauvegarde: {output_path}")

def generer_graphique_radar_comparatif(resultats, output_path):
    """Génère un graphique radar comparant 3 et 4 clusters"""
    print(f"\n[INFO] Generation du graphique radar comparatif...")

    fig = plt.figure(figsize=(16, 7))

    categories = ['Opp', 'Cho', 'Vec']
    n_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Palette de couleurs
    colors_3 = ['#e74c3c', '#3498db', '#2ecc71']
    colors_4 = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # Subplot 1: 3 clusters
    ax1 = fig.add_subplot(121, projection='polar')
    df_3 = resultats[3]

    for idx, row in df_3.iterrows():
        values = [row['ecart_opp'], row['ecart_cho'], row['ecart_vec']]
        values += values[:1]

        ax1.plot(angles, values, 'o-', linewidth=2.5,
                label=f"Cluster {row['cluster']} (n={row['n_communes']})",
                color=colors_3[idx], markersize=8)
        ax1.fill(angles, values, alpha=0.15, color=colors_3[idx])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=12, weight='bold')
    ax1.set_ylim(-2, 2)
    ax1.set_yticks([-2, -1, 0, 1, 2])
    ax1.set_yticklabels(['-2std', '-1std', '0', '+1std', '+2std'], size=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('3 clusters\nEcarts standardises', size=14, weight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # Subplot 2: 4 clusters
    ax2 = fig.add_subplot(122, projection='polar')
    df_4 = resultats[4]

    for idx, row in df_4.iterrows():
        values = [row['ecart_opp'], row['ecart_cho'], row['ecart_vec']]
        values += values[:1]

        ax2.plot(angles, values, 'o-', linewidth=2.5,
                label=f"Cluster {row['cluster']} (n={row['n_communes']})",
                color=colors_4[idx], markersize=8)
        ax2.fill(angles, values, alpha=0.15, color=colors_4[idx])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=12, weight='bold')
    ax2.set_ylim(-2, 2)
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_yticklabels(['-2std', '-1std', '0', '+1std', '+2std'], size=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('4 clusters\nEcarts standardises', size=14, weight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Graphique radar sauvegarde: {output_path}")

def main():
    print("=" * 80)
    print("  GRAPHIQUES DES ÉCARTS STANDARDISÉS - CAH 3D")
    print("=" * 80)

    # Charger les données
    gdf = charger_donnees()

    # Calculer les clusters pour 3 et 4
    n_clusters_list = [3, 4]
    clusters_dict, X, scaler = recalculer_clusters(gdf, n_clusters_list)

    # Calculer les écarts standardisés
    resultats, moyennes, stds = calculer_ecarts_standardises(gdf, clusters_dict, scaler)

    # Générer les graphiques en barres
    output_dir = '../OUTPUT'
    for n in n_clusters_list:
        output_path = f"{output_dir}/cah_3d_ecarts_std_{n}_clusters.png"
        generer_graphique(resultats, n, output_path)

    # Générer le graphique radar comparatif
    output_radar = f"{output_dir}/cah_3d_ecarts_std_comparatif_radar.png"
    generer_graphique_radar_comparatif(resultats, output_radar)

    print("\n" + "=" * 80)
    print("[OK] GENERATION TERMINEE")
    print(f"     - Graphiques en barres: 2 fichiers")
    print(f"     - Graphique radar comparatif: 1 fichier")
    print("=" * 80)

if __name__ == "__main__":
    main()
