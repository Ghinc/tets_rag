"""
Génère les profils des clusters pour OppChoVec
Affiche les moyennes de Score_Opp, Score_Cho et Score_Vec par cluster

Pour chaque nombre de clusters (2 à 8):
- Radar chart (graphique en étoile)
- Bar chart (diagramme en barres)
"""

import json
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

def charger_donnees(json_path, geojson_path):
    """Charge les données des indicateurs et le GeoJSON"""
    print("[INFO] Chargement des donnees...")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gdf = gpd.read_file(geojson_path)

    # Créer DataFrame avec les scores
    df_scores = pd.DataFrame.from_dict(data, orient='index')
    df_scores.index.name = 'commune'
    df_scores.reset_index(inplace=True)

    # Fusionner avec GeoDataFrame
    gdf['nom_clean'] = gdf['nom'].str.strip()
    gdf_merged = gdf.merge(df_scores, left_on='nom_clean', right_on='commune', how='inner')

    print(f"  [OK] {len(gdf_merged)} communes fusionnees")

    return gdf_merged

def recalculer_clusters(gdf):
    """Recalcule les clusters (même méthode que classification_hierarchique.py)"""
    print("\n[INFO] Recalcul des clusters...")

    # Extraire OppChoVec normalisé 0-10
    X = gdf[['OppChoVec_0_10']].values

    # Standardiser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # CAH - Ward nécessite les observations, pas la matrice de distances
    Z = linkage(X_scaled, method='ward')

    # Calculer clusters pour 2 à 8
    clusters_dict = {}
    for n in range(2, 9):
        clusters = fcluster(Z, t=n, criterion='maxclust')
        clusters_dict[n] = clusters

    print(f"  [OK] Clusters recalcules pour n=2 a 8")

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
    """Génère un radar chart pour un nombre de clusters"""

    # Préparer les données
    categories = ['Score_Opp', 'Score_Cho', 'Score_Vec']
    N = len(categories)

    # Calculer les angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Couleurs
    colors_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                     '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Tracer chaque cluster
    for idx, row in profils_df.iterrows():
        values = [row['Score_Opp'], row['Score_Cho'], row['Score_Vec']]
        values += values[:1]  # Fermer le polygone

        ax.plot(angles, values, 'o-', linewidth=2,
               label=f"Cluster {row['cluster']} (n={row['n_communes']})",
               color=colors_palette[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_palette[idx])

    # Configurer les labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_yticklabels(['0', '2', '4', '6', '8', '10'], size=10)
    ax.grid(True)

    # Titre et légende
    plt.title(f'Profils des clusters - {n} clusters\n' +
             f'Moyennes des scores par cluster',
             size=14, weight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generer_bar_chart(profils_df, n, output_path):
    """Génère un bar chart pour un nombre de clusters"""

    # Couleurs
    colors_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                     '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(14, 8))

    # Positions des barres
    x = np.arange(len(profils_df))
    width = 0.25

    # Tracer les barres pour chaque variable
    bars1 = ax.bar(x - width, profils_df['Score_Opp'], width,
                   label='Score Opportunites', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, profils_df['Score_Cho'], width,
                   label='Score Cho', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, profils_df['Score_Vec'], width,
                   label='Score Vec', color='#2ecc71', alpha=0.8)

    # Ajouter les valeurs sur les barres
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    # Configuration
    ax.set_xlabel('Clusters', fontsize=12, weight='bold')
    ax.set_ylabel('Scores moyens (0-10)', fontsize=12, weight='bold')
    ax.set_title(f'Profils des clusters - {n} clusters\n' +
                f'Comparaison des scores moyens par dimension',
                fontsize=14, weight='bold')

    # Labels des clusters avec effectifs
    cluster_labels = [f"Cluster {row['cluster']}\n(n={row['n_communes']})"
                     for _, row in profils_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels, fontsize=10)

    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 10.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generer_heatmap(profils_df, n, output_path):
    """Génère une heatmap pour un nombre de clusters"""

    fig, ax = plt.subplots(figsize=(10, n * 0.8 + 2))

    # Préparer les données pour la heatmap
    data = profils_df[['Score_Opp', 'Score_Cho', 'Score_Vec']].values.T

    # Créer la heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)

    # Labels
    ax.set_xticks(np.arange(len(profils_df)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([f"Cluster {i+1}\n(n={profils_df.iloc[i]['n_communes']})"
                        for i in range(len(profils_df))], fontsize=10)
    ax.set_yticklabels(['Score Opp', 'Score Cho', 'Score Vec'], fontsize=11)

    # Ajouter les valeurs dans les cellules
    for i in range(3):
        for j in range(len(profils_df)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                         ha="center", va="center", color="black",
                         fontsize=10, weight='bold')

    # Titre et colorbar
    ax.set_title(f'Heatmap des profils - {n} clusters\n' +
                f'Scores moyens par dimension et par cluster',
                fontsize=14, weight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Score moyen (0-10)', rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generer_profil_complet(profils_df, n, output_path):
    """Génère un graphique complet avec les 4 variables (incluant OppChoVec)"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Couleurs
    colors_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                     '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    # --- Graphique 1: Bar chart avec les 4 variables ---
    x = np.arange(len(profils_df))
    width = 0.2

    bars1 = ax1.bar(x - 1.5*width, profils_df['OppChoVec'], width,
                    label='OppChoVec (composite)', color='#9b59b6', alpha=0.9)
    bars2 = ax1.bar(x - 0.5*width, profils_df['Score_Opp'], width,
                    label='Score Opportunites', color='#e74c3c', alpha=0.9)
    bars3 = ax1.bar(x + 0.5*width, profils_df['Score_Cho'], width,
                    label='Score Choix', color='#3498db', alpha=0.9)
    bars4 = ax1.bar(x + 1.5*width, profils_df['Score_Vec'], width,
                    label='Score Vécu', color='#2ecc71', alpha=0.9)

    ax1.set_xlabel('Clusters', fontsize=11, weight='bold')
    ax1.set_ylabel('Scores moyens (0-10)', fontsize=11, weight='bold')
    ax1.set_title('Profil complet par cluster', fontsize=12, weight='bold')

    cluster_labels = [f"C{row['cluster']}\n(n={row['n_communes']})"
                     for _, row in profils_df.iterrows()]
    ax1.set_xticks(x)
    ax1.set_xticklabels(cluster_labels, fontsize=9)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_ylim(0, 10.5)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # --- Graphique 2: Line plot pour voir les tendances ---
    for idx, row in profils_df.iterrows():
        values = [row['OppChoVec'], row['Score_Opp'], row['Score_Cho'], row['Score_Vec']]
        ax2.plot(['OppChoVec', 'Opp', 'Cho', 'Vec'], values,
                'o-', linewidth=2.5, markersize=8,
                label=f"Cluster {row['cluster']} (n={row['n_communes']})",
                color=colors_palette[idx])

    ax2.set_xlabel('Variables', fontsize=11, weight='bold')
    ax2.set_ylabel('Scores moyens (0-10)', fontsize=11, weight='bold')
    ax2.set_title('Evolution des scores par cluster', fontsize=12, weight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.set_ylim(0, 10.5)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Analyse des profils - {n} clusters',
                fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 80)
    print("  PROFILS DES CLUSTERS - OPP, CHO, VEC")
    print("=" * 80)

    # Chemins
    json_path = '../WEB/data_scores_0_10.json'
    geojson_path = '../WEB/Commune_Corse.geojson'
    output_dir = '../OUTPUT'

    # Charger les données
    gdf = charger_donnees(json_path, geojson_path)

    # Recalculer les clusters
    clusters_dict = recalculer_clusters(gdf)

    # Calculer les profils
    profils_dict = calculer_profils(gdf, clusters_dict)

    # Générer les graphiques pour chaque nombre de clusters
    print("\n[INFO] Generation des graphiques...")

    for n in range(2, 9):
        profils_df = profils_dict[n]

        # Radar chart
        generer_radar_chart(profils_df, n, f"{output_dir}/profils_radar_{n}_clusters.png")
        print(f"  [OK] Radar chart: profils_radar_{n}_clusters.png")

        # Bar chart
        generer_bar_chart(profils_df, n, f"{output_dir}/profils_bar_{n}_clusters.png")
        print(f"  [OK] Bar chart: profils_bar_{n}_clusters.png")

        # Heatmap
        generer_heatmap(profils_df, n, f"{output_dir}/profils_heatmap_{n}_clusters.png")
        print(f"  [OK] Heatmap: profils_heatmap_{n}_clusters.png")

        # Profil complet
        generer_profil_complet(profils_df, n, f"{output_dir}/profils_complet_{n}_clusters.png")
        print(f"  [OK] Profil complet: profils_complet_{n}_clusters.png")

    # Afficher un exemple de profil
    print("\n" + "=" * 80)
    print("EXEMPLE: PROFILS POUR 7 CLUSTERS")
    print("=" * 80)
    print(profils_dict[7].to_string(index=False))

    print("\n" + "=" * 80)
    print("[OK] GENERATION TERMINEE")
    print(f"     {7 * 4} graphiques crees (4 types x 7 configurations)")
    print("=" * 80)

if __name__ == "__main__":
    main()
