"""
Génère les profils des clusters pour la CAH 3D (Opp, Cho, Vec)
Visualisations: radar charts, bar charts, heatmaps
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

def recalculer_clusters(gdf):
    """Recalcule les clusters pour 2 à 8"""
    print("\n[INFO] Recalcul des clusters...")

    X = gdf[['Score_Opp_0_10', 'Score_Cho_0_10', 'Score_Vec_0_10']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method='ward')

    clusters_dict = {}
    for n in range(2, 9):
        clusters = fcluster(Z, t=n, criterion='maxclust')
        clusters_dict[n] = clusters
        print(f"  [OK] Clusters recalcules pour {n} clusters")

    return clusters_dict

def calculer_profils(gdf, clusters_dict):
    """Calcule les profils moyens pour chaque cluster"""
    print("\n[INFO] Calcul des profils...")

    profils_dict = {}

    for n, clusters in clusters_dict.items():
        gdf_temp = gdf.copy()
        gdf_temp['cluster'] = clusters

        profils = []
        for i in range(1, n + 1):
            cluster_data = gdf_temp[gdf_temp['cluster'] == i]
            profils.append({
                'cluster': i,
                'n_communes': len(cluster_data),
                'Score_Opp': cluster_data['Score_Opp_0_10'].mean(),
                'Score_Cho': cluster_data['Score_Cho_0_10'].mean(),
                'Score_Vec': cluster_data['Score_Vec_0_10'].mean(),
                'OppChoVec': cluster_data['OppChoVec_0_10'].mean()
            })

        profils_dict[n] = pd.DataFrame(profils)
        print(f"  [OK] Profils calcules pour {n} clusters")

    return profils_dict

def generer_radar_chart(profils_df, n, output_path):
    """Génère un radar chart (spider chart) pour les profils"""

    categories = ['Opp', 'Cho', 'Vec']
    n_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set3(np.linspace(0, 1, n))

    for idx, row in profils_df.iterrows():
        values = [row['Score_Opp'], row['Score_Cho'], row['Score_Vec']]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=f"Cluster {row['cluster']} (n={row['n_communes']})",
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_yticklabels(['0', '2', '4', '6', '8', '10'], size=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.title(f'Profils des {n} clusters - Radar Chart\nVariables: Opp, Cho, Vec',
              size=16, weight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generer_bar_chart(profils_df, n, output_path):
    """Génère un bar chart avec les profils"""

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(profils_df))
    width = 0.25

    colors_opp = '#e74c3c'
    colors_cho = '#3498db'
    colors_vec = '#2ecc71'

    bars1 = ax.bar(x - width, profils_df['Score_Opp'], width, label='Opp', color=colors_opp, alpha=0.8)
    bars2 = ax.bar(x, profils_df['Score_Cho'], width, label='Cho', color=colors_cho, alpha=0.8)
    bars3 = ax.bar(x + width, profils_df['Score_Vec'], width, label='Vec', color=colors_vec, alpha=0.8)

    ax.set_xlabel('Cluster', fontsize=12, weight='bold')
    ax.set_ylabel('Score moyen (0-10)', fontsize=12, weight='bold')
    ax.set_title(f'Profils des {n} clusters - Scores moyens Opp, Cho, Vec',
                fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{row['cluster']}\n(n={row['n_communes']})" for _, row in profils_df.iterrows()])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 10)

    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generer_heatmap(profils_df, n, output_path):
    """Génère une heatmap des profils"""

    fig, ax = plt.subplots(figsize=(10, 8))

    data = profils_df[['Score_Opp', 'Score_Cho', 'Score_Vec']].T

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)

    ax.set_xticks(np.arange(len(profils_df)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([f"Cluster {row['cluster']}\n(n={row['n_communes']})" for _, row in profils_df.iterrows()])
    ax.set_yticklabels(['Opp', 'Cho', 'Vec'])

    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=11, weight='bold')

    # Ajouter les valeurs dans les cellules
    for i in range(3):
        for j in range(len(profils_df)):
            text = ax.text(j, i, f'{data.iloc[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=11, weight='bold')

    ax.set_title(f'Heatmap des profils - {n} clusters\nVariables: Opp, Cho, Vec',
                fontsize=14, weight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score (0-10)', rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generer_profil_complet(profils_df, n, output_path):
    """Génère un graphique complet avec plusieurs vues"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Bar chart
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(profils_df))
    width = 0.25

    ax1.bar(x - width, profils_df['Score_Opp'], width, label='Opp', color='#e74c3c', alpha=0.8)
    ax1.bar(x, profils_df['Score_Cho'], width, label='Cho', color='#3498db', alpha=0.8)
    ax1.bar(x + width, profils_df['Score_Vec'], width, label='Vec', color='#2ecc71', alpha=0.8)

    ax1.set_ylabel('Score moyen (0-10)', fontsize=11, weight='bold')
    ax1.set_title(f'Profils des {n} clusters - Vue complete', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{row['cluster']}\n(n={row['n_communes']})" for _, row in profils_df.iterrows()])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 10)

    # 2. Line plot par dimension
    ax2 = fig.add_subplot(gs[1, 0])
    clusters_idx = profils_df['cluster'].values
    ax2.plot(clusters_idx, profils_df['Score_Opp'], 'o-', linewidth=2, markersize=8, label='Opp', color='#e74c3c')
    ax2.plot(clusters_idx, profils_df['Score_Cho'], 's-', linewidth=2, markersize=8, label='Cho', color='#3498db')
    ax2.plot(clusters_idx, profils_df['Score_Vec'], '^-', linewidth=2, markersize=8, label='Vec', color='#2ecc71')
    ax2.set_xlabel('Cluster', fontsize=11, weight='bold')
    ax2.set_ylabel('Score moyen', fontsize=11, weight='bold')
    ax2.set_title('Evolution par dimension', fontsize=12, weight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 10)

    # 3. OppChoVec global
    ax3 = fig.add_subplot(gs[1, 1])
    colors_oppchovec = plt.cm.viridis(profils_df['OppChoVec'] / 10)
    bars = ax3.bar(profils_df['cluster'], profils_df['OppChoVec'], color=colors_oppchovec, alpha=0.8)
    ax3.set_xlabel('Cluster', fontsize=11, weight='bold')
    ax3.set_ylabel('OppChoVec moyen', fontsize=11, weight='bold')
    ax3.set_title('Score OppChoVec global', fontsize=12, weight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 10)

    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, weight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 80)
    print("  PROFILS DES CLUSTERS - CAH 3D (OPP, CHO, VEC)")
    print("=" * 80)

    # Charger les données
    gdf = charger_donnees()

    # Recalculer les clusters
    clusters_dict = recalculer_clusters(gdf)

    # Calculer les profils
    profils_dict = calculer_profils(gdf, clusters_dict)

    # Générer les graphiques
    print("\n[INFO] Generation des graphiques...")
    output_dir = '../OUTPUT'

    for n, profils_df in profils_dict.items():
        # Radar chart
        generer_radar_chart(profils_df, n, f"{output_dir}/profils_3d_radar_{n}_clusters.png")
        print(f"  [OK] Radar chart: profils_3d_radar_{n}_clusters.png")

        # Bar chart
        generer_bar_chart(profils_df, n, f"{output_dir}/profils_3d_bar_{n}_clusters.png")
        print(f"  [OK] Bar chart: profils_3d_bar_{n}_clusters.png")

        # Heatmap
        generer_heatmap(profils_df, n, f"{output_dir}/profils_3d_heatmap_{n}_clusters.png")
        print(f"  [OK] Heatmap: profils_3d_heatmap_{n}_clusters.png")

        # Profil complet
        generer_profil_complet(profils_df, n, f"{output_dir}/profils_3d_complet_{n}_clusters.png")
        print(f"  [OK] Profil complet: profils_3d_complet_{n}_clusters.png")

    # Afficher exemple
    print("\n" + "=" * 80)
    print("EXEMPLE: PROFILS POUR 7 CLUSTERS")
    print("=" * 80)
    print(profils_dict[7].to_string(index=False))

    print("\n" + "=" * 80)
    print("[OK] GENERATION TERMINEE")
    print(f"     {len(profils_dict) * 4} graphiques crees (4 types x {len(profils_dict)} configurations)")
    print("=" * 80)

if __name__ == "__main__":
    main()
