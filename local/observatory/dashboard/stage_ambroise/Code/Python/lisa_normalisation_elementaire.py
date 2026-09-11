"""
Analyse LISA (Local Indicators of Spatial Association) pour les communes de Corse
Version 3 : Utilise oppchovec_normalisation_elementaire.xlsx avec normalisation 1-10
Ameliorations : Legendes visibles et mise en page optimisee
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from libpysal.weights import Queen
from esda.moran import Moran_Local, Moran
import requests
from shapely.geometry import shape
import json

# Fixer le seed pour reproductibilité
SEED = 42
np.random.seed(SEED)

# Configuration
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


def charger_donnees_excel(fichier='oppchovec_normalisation_elementaire_trie.xlsx'):
    """Charge les donnees depuis le fichier Excel (feuille Synthese)"""
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
    """Calcule la matrice de poids spatial (contiguite de type Queen)"""
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

    return lisa, gdf, moran_global


def visualiser_resultats(gdf, variable='OppChoVec_1_10'):
    """Cree des visualisations des resultats LISA avec legendes ameliorees"""
    print("\nCreation des visualisations...")

    # Creer la figure avec des sous-graphiques
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15,
                         left=0.05, right=0.95, top=0.95, bottom=0.05)

    # 1. Carte de la variable originale
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = gdf.plot(column=variable, cmap='YlOrRd', legend=False,
                   ax=ax1, edgecolor='black', linewidth=0.3)
    ax1.set_title(f'Distribution spatiale de {variable}',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.axis('off')

    # Ajouter colorbar personnalisee
    sm1 = plt.cm.ScalarMappable(cmap='YlOrRd',
                                norm=plt.Normalize(vmin=gdf[variable].min(),
                                                  vmax=gdf[variable].max()))
    sm1._A = []
    cbar1 = fig.colorbar(sm1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(f'{variable}', rotation=270, labelpad=20, fontsize=11)

    # 2. Carte des indicateurs LISA (I local)
    ax2 = fig.add_subplot(gs[0, 1])
    gdf.plot(column='lisa_I', cmap='RdBu_r', legend=False,
             ax=ax2, edgecolor='black', linewidth=0.3)
    ax2.set_title('Indicateurs LISA (Moran Local I)',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.axis('off')

    # Ajouter colorbar personnalisee
    sm2 = plt.cm.ScalarMappable(cmap='RdBu_r',
                                norm=plt.Normalize(vmin=gdf['lisa_I'].min(),
                                                  vmax=gdf['lisa_I'].max()))
    sm2._A = []
    cbar2 = fig.colorbar(sm2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Indice de Moran Local', rotation=270, labelpad=20, fontsize=11)

    # 3. Carte de significativite
    ax3 = fig.add_subplot(gs[1, 0])

    # Creer une colormap pour la significativite
    colors_sig = ['#d73027', '#1a9850']  # Rouge pour non-sig, vert pour sig
    cmap_sig = ListedColormap(colors_sig)

    gdf.plot(column='lisa_sig', cmap=cmap_sig, legend=False,
             ax=ax3, edgecolor='black', linewidth=0.3)
    ax3.set_title('Significativite statistique (p < 0.05)',
                  fontsize=14, fontweight='bold', pad=15)
    ax3.axis('off')

    # Legende personnalisee
    legend_elements = [
        Patch(facecolor='#d73027', edgecolor='black', label='Non significatif (p >= 0.05)'),
        Patch(facecolor='#1a9850', edgecolor='black', label='Significatif (p < 0.05)')
    ]
    ax3.legend(handles=legend_elements, loc='lower left',
              fontsize=10, frameon=True, fancybox=True, shadow=True)

    # 4. Carte des clusters LISA
    ax4 = fig.add_subplot(gs[1, 1])

    colors = {
        'Non significatif': '#d9d9d9',
        'HH (High-High)': '#d73027',
        'LL (Low-Low)': '#4575b4',
        'LH (Low-High)': '#abd9e9',
        'HL (High-Low)': '#fdae61'
    }

    labels_descriptifs = {
        'Non significatif': 'Non significatif',
        'HH (High-High)': 'HH : Valeurs elevees entourees de valeurs elevees',
        'LL (Low-Low)': 'LL : Valeurs faibles entourees de valeurs faibles',
        'LH (Low-High)': 'LH : Valeurs faibles entourees de valeurs elevees',
        'HL (High-Low)': 'HL : Valeurs elevees entourees de valeurs faibles'
    }

    for cluster_type, color in colors.items():
        data = gdf[gdf['lisa_cluster'] == cluster_type]
        if len(data) > 0:
            data.plot(ax=ax4, color=color, edgecolor='black',
                     linewidth=0.3, label=labels_descriptifs[cluster_type])

    ax4.set_title('Clusters LISA (significatifs a p < 0.05)',
                  fontsize=14, fontweight='bold', pad=15)
    ax4.legend(loc='lower left', fontsize=9, frameon=True,
              fancybox=True, shadow=True)
    ax4.axis('off')

    plt.savefig('lisa_analysis_normalisation_elementaire.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("  Carte sauvegardee: lisa_analysis_normalisation_elementaire.png")

    # Creer une image separee pour la carte des clusters uniquement (avec legende amelioree)
    fig_cluster = plt.figure(figsize=(14, 12))
    ax_cluster = fig_cluster.add_subplot(111)

    for cluster_type, color in colors.items():
        data = gdf[gdf['lisa_cluster'] == cluster_type]
        if len(data) > 0:
            data.plot(ax=ax_cluster, color=color, edgecolor='black',
                     linewidth=0.4)

    ax_cluster.set_title('Clusters LISA - Indice OppChoVec en Corse\n(Normalisation elementaire 1-10)',
                         fontsize=18, fontweight='bold', pad=25)
    ax_cluster.axis('off')

    # Creer une legende personnalisee plus grande et visible
    legend_elements = []
    for cluster_type, color in colors.items():
        if len(gdf[gdf['lisa_cluster'] == cluster_type]) > 0:
            count = len(gdf[gdf['lisa_cluster'] == cluster_type])
            label = f"{labels_descriptifs[cluster_type]} (n={count})"
            legend_elements.append(Patch(facecolor=color, edgecolor='black',
                                        linewidth=1, label=label))

    legend = ax_cluster.legend(handles=legend_elements,
                              loc='lower center',
                              bbox_to_anchor=(0.5, -0.08),
                              fontsize=12,
                              frameon=True,
                              fancybox=True,
                              shadow=True,
                              title='Types de clusters',
                              title_fontsize=14,
                              ncol=2)
    legend.set_zorder(1000)  # Mettre la legende au premier plan

    plt.tight_layout()
    plt.savefig('lisa_clusters_normalisation_elementaire.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("  Carte des clusters sauvegardee: lisa_clusters_normalisation_elementaire.png")

    plt.show()


def exporter_resultats(gdf, nom_fichier='resultats_lisa_normalisation_elementaire.xlsx'):
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


def exporter_clusters_json(gdf, moran_global, fichier='../WEB/lisa_clusters.json'):
    """Exporte les clusters LISA en JSON pour l'interface web"""
    print(f"\nExport des clusters LISA vers {fichier}...")

    # Creer dictionnaire des clusters par commune
    clusters = {}
    for idx, row in gdf.iterrows():
        clusters[row['nom']] = {
            'cluster': row['lisa_cluster'],
            'lisa_I': float(row['lisa_I']),
            'p_value': float(row['lisa_p']),
            'significatif': bool(row['lisa_sig']),
            'oppchovec': float(row['OppChoVec_1_10'])
        }

    # Creer structure complete avec metadonnees
    resultat = {
        'metadata': {
            'description': 'Analyse LISA (Local Indicators of Spatial Association) pour les communes de Corse',
            'variable': 'OppChoVec_1_10',
            'methode': 'Moran Local avec matrice Queen contiguity',
            'seed': SEED,
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
    with open(fichier, 'w', encoding='utf-8') as f:
        json.dump(resultat, f, ensure_ascii=False, indent=2)

    print(f"  OK - {len(clusters)} communes exportees")
    print(f"  OK - Moran I global: {moran_global.I:.4f}")
    print(f"  OK - Fichier: {fichier}")


def main():
    """Fonction principale"""
    print("="*80)
    print("ANALYSE LISA - Communes de Corse")
    print("Version Normalisation Elementaire (1-10)")
    print("="*80)

    try:
        # 1. Charger les donnees Excel
        df_data = charger_donnees_excel('oppchovec_normalisation_elementaire_trie.xlsx')

        # 2. Recuperer les geometries des communes
        gdf_geo = recuperer_geometries_communes()

        # 3. Joindre les donnees
        gdf = joindre_donnees(df_data, gdf_geo)

        # 4. Calculer la matrice de poids spatial
        w, gdf_proj = calculer_matrice_poids(gdf)

        # 5. Calculer l'indicateur LISA
        lisa, gdf_result, moran_global = calculer_lisa(gdf_proj, w, variable='OppChoVec_1_10')

        # 6. Visualiser les resultats
        visualiser_resultats(gdf_result, variable='OppChoVec_1_10')

        # 7. Exporter les resultats
        exporter_resultats(gdf_result,
                          nom_fichier='resultats_lisa_normalisation_elementaire.xlsx')

        # 8. Exporter les clusters JSON pour l'interface web
        exporter_clusters_json(gdf_result, moran_global)

        print("\n" + "="*80)
        print("ANALYSE TERMINEE AVEC SUCCES!")
        print("="*80)

        return gdf_result, lisa, w, moran_global

    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    gdf_result, lisa, w, moran_global = main()
