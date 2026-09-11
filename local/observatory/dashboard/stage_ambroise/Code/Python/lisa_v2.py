"""
Analyse LISA (Local Indicators of Spatial Association) pour les communes de Corse
Version 2 : Utilise oppchovec_0_norm.xlsx avec les nouvelles données
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran_Local, Moran
import requests
from shapely.geometry import shape

# Configuration
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


def charger_donnees_excel(fichier='oppchovec_0_norm.xlsx'):
    """Charge les données depuis le fichier Excel (feuille Synthese)"""
    print(f"Chargement des donnees depuis {fichier}...")
    df = pd.read_excel(fichier, sheet_name='Synthese')
    print(f"  {len(df)} communes chargees")
    print(f"  Colonnes: {df.columns.tolist()}")
    return df


def recuperer_geometries_communes():
    """Recupere les geometries des communes de Corse depuis l'API geo.data.gouv.fr"""
    print("\nRecuperation des geometries des communes de Corse...")

    # Liste des departements de Corse: 2A (Corse-du-Sud) et 2B (Haute-Corse)
    departements = ['2A', '2B']
    communes_list = []

    for dept in departements:
        print(f"  Telechargement des communes du departement {dept}...")
        url = f"https://geo.api.gouv.fr/departements/{dept}/communes?format=geojson&geometry=contour"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(f"    {len(data['features'])} communes trouvees")
            communes_list.extend(data['features'])
        else:
            print(f"    Erreur lors du telechargement: {response.status_code}")

    # Creer un GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(communes_list, crs='EPSG:4326')
    print(f"  Total: {len(gdf)} communes chargees")

    return gdf


def joindre_donnees(df_data, gdf_geo):
    """Joint les donnees Excel avec les geometries"""
    print("\nJointure des donnees...")

    # Joindre sur le nom de commune
    gdf = gdf_geo.merge(df_data, left_on='nom', right_on='Zone', how='inner')
    print(f"  {len(gdf)} communes apres jointure")

    # Afficher les statistiques descriptives
    print("\nStatistiques descriptives de OppChoVec_1_10:")
    print(gdf['OppChoVec_1_10'].describe())

    return gdf


def calculer_matrice_poids(gdf):
    """Calcule la matrice de poids spatial (contiguïté de type Queen)"""
    print("\nCalcul de la matrice de poids spatial (Queen contiguity)...")

    # Projeter en Lambert 93 pour la France metropolitaine
    gdf_proj = gdf.to_crs('EPSG:2154')

    # Creer la matrice de poids (Queen: partage un cote ou un sommet)
    w = Queen.from_dataframe(gdf_proj)

    # Standardiser les poids (row-standardized)
    w.transform = 'r'

    print(f"  Matrice de poids creee")
    print(f"  Nombre moyen de voisins: {w.mean_neighbors:.2f}")
    print(f"  Nombre min de voisins: {w.min_neighbors}")
    print(f"  Nombre max de voisins: {w.max_neighbors}")

    return w, gdf_proj


def calculer_lisa(gdf, w, variable='OppChoVec_1_10'):
    """Calcule l'indicateur LISA (Moran local)"""
    print(f"\nCalcul de l'indicateur LISA pour '{variable}'...")

    # Extraire les valeurs
    y = gdf[variable].values

    # Calculer le Moran global pour comparaison
    moran_global = Moran(y, w)
    print(f"\nMoran global (autocorrelation spatiale globale):")
    print(f"  I = {moran_global.I:.4f}")
    print(f"  p-value = {moran_global.p_sim:.4f}")

    # Calculer le Moran local
    lisa = Moran_Local(y, w)

    # Ajouter les resultats au GeoDataFrame
    gdf['lisa_I'] = lisa.Is  # Indicateur local pour chaque commune
    gdf['lisa_p'] = lisa.p_sim  # P-value (test de permutation)
    gdf['lisa_q'] = lisa.q  # Quadrant (HH, LL, LH, HL)
    gdf['lisa_sig'] = lisa.p_sim < 0.05  # Significatif a 5%

    # Creer une classification combinee (quadrant + significativite)
    gdf['lisa_cluster'] = 'Non significatif'
    gdf.loc[(gdf['lisa_q'] == 1) & (gdf['lisa_sig']), 'lisa_cluster'] = 'HH (High-High)'
    gdf.loc[(gdf['lisa_q'] == 2) & (gdf['lisa_sig']), 'lisa_cluster'] = 'LH (Low-High)'
    gdf.loc[(gdf['lisa_q'] == 3) & (gdf['lisa_sig']), 'lisa_cluster'] = 'LL (Low-Low)'
    gdf.loc[(gdf['lisa_q'] == 4) & (gdf['lisa_sig']), 'lisa_cluster'] = 'HL (High-Low)'

    print(f"\nResultats LISA:")
    print(f"  Communes avec autocorrelation spatiale significative: {gdf['lisa_sig'].sum()} ({gdf['lisa_sig'].sum()/len(gdf)*100:.1f}%)")
    print(f"\nRepartition par type de cluster:")
    print(gdf['lisa_cluster'].value_counts())

    return lisa, gdf


def visualiser_resultats(gdf, variable='OppChoVec_1_10'):
    """Cree des visualisations des resultats LISA"""
    print("\nCreation des visualisations...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # 1. Carte de la variable originale
    ax1 = axes[0, 0]
    gdf.plot(column=variable, cmap='YlOrRd', legend=True,
             ax=ax1, edgecolor='black', linewidth=0.5)
    ax1.set_title(f'Distribution spatiale de {variable}', fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')

    # 2. Carte des indicateurs LISA (I local)
    ax2 = axes[0, 1]
    gdf.plot(column='lisa_I', cmap='RdBu_r', legend=True,
             ax=ax2, edgecolor='black', linewidth=0.5)
    ax2.set_title('Indicateurs LISA (Moran Local I)', fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')

    # 3. Carte de significativite
    ax3 = axes[1, 0]
    gdf.plot(column='lisa_sig', cmap='RdYlGn', legend=True,
             ax=ax3, edgecolor='black', linewidth=0.5,
             categorical=True)
    ax3.set_title('Significativite statistique (p < 0.05)', fontsize=14, fontweight='bold', pad=20)
    ax3.axis('off')

    # 4. Carte des clusters LISA
    ax4 = axes[1, 1]
    colors = {
        'Non significatif': 'lightgray',
        'HH (High-High)': 'red',
        'LL (Low-Low)': 'blue',
        'LH (Low-High)': 'lightblue',
        'HL (High-Low)': 'pink'
    }

    labels_descriptifs = {
        'Non significatif': 'Non significatif',
        'HH (High-High)': 'HH : Valeurs elevees, voisins eleves',
        'LL (Low-Low)': 'LL : Valeurs faibles, voisins faibles',
        'LH (Low-High)': 'LH : Valeurs faibles, voisins eleves',
        'HL (High-Low)': 'HL : Valeurs elevees, voisins faibles'
    }

    for cluster_type, color in colors.items():
        data = gdf[gdf['lisa_cluster'] == cluster_type]
        if len(data) > 0:
            data.plot(ax=ax4, color=color, edgecolor='black',
                     linewidth=0.5, label=labels_descriptifs[cluster_type])

    ax4.set_title('Clusters LISA (significatifs a p < 0.05)', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper left', fontsize=8, frameon=True, fancybox=True, shadow=True)
    ax4.axis('off')

    plt.tight_layout(pad=3.0)
    plt.savefig('lisa_analysis_v2.png', dpi=300, bbox_inches='tight')
    print("  Carte sauvegardee: lisa_analysis_v2.png")

    # Creer une image separee pour la carte des clusters uniquement
    fig_cluster, ax_cluster = plt.subplots(1, 1, figsize=(12, 10))

    for cluster_type, color in colors.items():
        data = gdf[gdf['lisa_cluster'] == cluster_type]
        if len(data) > 0:
            data.plot(ax=ax_cluster, color=color, edgecolor='black',
                     linewidth=0.5, label=labels_descriptifs[cluster_type])

    ax_cluster.set_title('Clusters LISA - Indice OppChoVec en Corse (sans normalisation)',
                         fontsize=16, fontweight='bold', pad=20)
    ax_cluster.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True,
                     title='Types de clusters', title_fontsize=12)
    ax_cluster.axis('off')

    plt.tight_layout()
    plt.savefig('lisa_clusters_v2.png', dpi=300, bbox_inches='tight')
    print("  Carte des clusters sauvegardee: lisa_clusters_v2.png")

    plt.show()


def exporter_resultats(gdf, nom_fichier='resultats_lisa_v2.xlsx'):
    """Exporte les resultats dans un fichier Excel"""
    print(f"\nExport des resultats vers {nom_fichier}...")

    # Selectionner les colonnes pertinentes
    colonnes = ['code', 'nom', 'Zone', 'OppChoVec_1_10', 'OppChoVec',
                'Score_Opp_1_10', 'Score_Cho_1_10', 'Score_Vec_1_10',
                'lisa_I', 'lisa_p', 'lisa_q', 'lisa_sig', 'lisa_cluster']

    # Filtrer les colonnes qui existent
    colonnes_existantes = [col for col in colonnes if col in gdf.columns]

    df_export = gdf[colonnes_existantes].copy()
    df_export = df_export.sort_values('lisa_p')  # Trier par p-value

    # Supprimer la geometrie pour l'export Excel
    if 'geometry' in df_export.columns:
        df_export = df_export.drop(columns=['geometry'])

    df_export.to_excel(nom_fichier, index=False)
    print(f"  {len(df_export)} lignes exportees")


def main():
    """Fonction principale"""
    print("="*80)
    print("ANALYSE LISA - Communes de Corse (Version 2 - Sans normalisation)")
    print("="*80)

    try:
        # 1. Charger les donnees Excel
        df_data = charger_donnees_excel('oppchovec_0_norm.xlsx')

        # 2. Recuperer les geometries des communes
        gdf_geo = recuperer_geometries_communes()

        # 3. Joindre les donnees
        gdf = joindre_donnees(df_data, gdf_geo)

        # 4. Calculer la matrice de poids spatial
        w, gdf_proj = calculer_matrice_poids(gdf)

        # 5. Calculer l'indicateur LISA
        lisa, gdf_result = calculer_lisa(gdf_proj, w, variable='OppChoVec_1_10')

        # 6. Visualiser les resultats
        visualiser_resultats(gdf_result, variable='OppChoVec_1_10')

        # 7. Exporter les resultats
        exporter_resultats(gdf_result, nom_fichier='resultats_lisa_v2.xlsx')

        print("\n" + "="*80)
        print("ANALYSE TERMINEE AVEC SUCCES!")
        print("="*80)

        return gdf_result, lisa, w

    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    gdf_result, lisa, w = main()
