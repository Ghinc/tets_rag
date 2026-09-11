"""
Génère les données LISA pour l'indice OppChoVec avec pondération Betti et al. (2008)
Produit : lisa_data_betti.js et lisa_data_betti_1pct.js
"""
import json, math, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran_Local, Moran

SEED = 42
np.random.seed(SEED)

WEB_DIR = 'c:/These/oppchovec_visu/stage_ambroise/Code/WEB/'
GEOJSON  = 'c:/These/oppchovec_visu/stage_ambroise/Données/Commune_Corse.geojson'
DATA_JSON = WEB_DIR + 'data_scores_0_10.json'

# === Charger données ===
with open(DATA_JSON, encoding='utf-8') as f:
    data = json.load(f)

communes = list(data.keys())
alpha, beta = 2.5, 1.5

def mean(x): return sum(x)/len(x)
def std(x): m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def pearson(a,b):
    n=len(a); ma,mb=mean(a),mean(b)
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da=math.sqrt(sum((v-ma)**2 for v in a)); db=math.sqrt(sum((v-mb)**2 for v in b))
    return num/(da*db) if da*db else 0
def cv(x): m=mean(x); return std(x)/m if m else 0
def norm010(d):
    mn,mx=min(d.values()),max(d.values())
    return {c:(d[c]-mn)/(mx-mn)*10 if mx!=mn else 5 for c in communes}

# === Poids Betti (scores 0-1) ===
dims_keys = ['Score_Opp','Score_Cho','Score_Vec']
sc01 = {k:[data[c][k] for c in communes] for k in dims_keys}
p1=[cv(sc01[k]) for k in dims_keys]
corrs=[[pearson(sc01[dims_keys[i]],sc01[dims_keys[j]]) for j in range(3)] for i in range(3)]
p2=[1/mean([abs(corrs[i][j]) for j in range(3)]) for i in range(3)]
pkRaw=[p1[i]*p2[i] for i in range(3)]; s=sum(pkRaw)
pk=[v/s for v in pkRaw]
print(f"Poids Betti: Opp={pk[0]:.4f} Cho={pk[1]:.4f} Vec={pk[2]:.4f}")

# OppChoVec Betti 0-10
raw_betti={c:(1/3)*sum(pk[i]*data[c][dims_keys[i]]**beta for i in range(3))**(alpha/beta) for c in communes}
occ010=norm010(raw_betti)
print(f"OppChoVec_Betti_0_10 : moy={mean(list(occ010.values())):.4f}  min={min(occ010.values()):.4f}  max={max(occ010.values()):.4f}")

# === Charger GeoJSON ===
print(f"\nChargement GeoJSON...")
gdf = gpd.read_file(GEOJSON)
print(f"  Colonnes disponibles : {gdf.columns.tolist()}")
print(f"  {len(gdf)} communes dans le GeoJSON")

# Identifier la colonne nom
nom_col = next((c for c in ['nom','NOM','NOM_COM','name','NAME','Zone'] if c in gdf.columns), None)
print(f"  Colonne nom utilisée : {nom_col}")

# Joindre avec les données Betti
import pandas as pd
df_betti = pd.DataFrame({'nom': list(occ010.keys()), 'OppChoVec_Betti_0_10': list(occ010.values())})
gdf = gdf.merge(df_betti, left_on=nom_col, right_on='nom', how='inner')
print(f"  {len(gdf)} communes après jointure")

# === Matrice de poids spatiale (Queen) ===
print("\nCalcul de la matrice Queen...")
gdf_proj = gdf.to_crs('EPSG:2154')
w = Queen.from_dataframe(gdf_proj)
w.transform = 'r'
print(f"  Voisins moyens : {w.mean_neighbors:.2f}")

# === LISA 5% ===
def run_lisa(seuil, label):
    print(f"\nCalcul LISA (seuil p<{seuil})...")
    y = gdf_proj['OppChoVec_Betti_0_10'].values
    moran_global = Moran(y, w)
    print(f"  Moran I global = {moran_global.I:.4f}  p = {moran_global.p_sim:.4f}")

    lisa = Moran_Local(y, w, seed=SEED)

    sig = lisa.p_sim < seuil
    cluster_map = {1:'HH (High-High)', 2:'LH (Low-High)', 3:'LL (Low-Low)', 4:'HL (High-Low)'}

    clusters = {}
    for i, row in gdf_proj.iterrows():
        nom = row[nom_col] if nom_col in row else row['nom']
        is_sig = bool(sig[list(gdf_proj.index).index(i)])
        q = int(lisa.q[list(gdf_proj.index).index(i)])
        cluster_type = cluster_map.get(q, 'Non significatif') if is_sig else 'Non significatif'
        clusters[nom] = {
            'cluster': cluster_type,
            'lisa_I': float(lisa.Is[list(gdf_proj.index).index(i)]),
            'p_value': float(lisa.p_sim[list(gdf_proj.index).index(i)]),
            'significatif': is_sig,
            'oppchovec': float(occ010.get(nom, 0))
        }

    nb_sig = sum(1 for v in clusters.values() if v['significatif'])
    stats = {k:sum(1 for v in clusters.values() if v['cluster']==k)
             for k in ['HH (High-High)','LL (Low-Low)','HL (High-Low)','LH (Low-High)','Non significatif']}

    print(f"  Communes significatives : {nb_sig} ({nb_sig/len(clusters)*100:.1f}%)")
    for k,v in stats.items():
        if v>0: print(f"    {k}: {v}")

    return {
        'metadata': {
            'description': f'Analyse LISA (Local Indicators of Spatial Association) - OppChoVec Betti et al. (2008)',
            'variable': 'OppChoVec_Betti_0_10',
            'poids_betti': {'Opp': round(pk[0],4), 'Cho': round(pk[1],4), 'Vec': round(pk[2],4)},
            'methode': 'Moran Local avec matrice Queen contiguity',
            'seed': SEED,
            'moran_global_I': float(moran_global.I),
            'moran_global_p': float(moran_global.p_sim),
            'seuil_significativite': seuil,
            'nb_communes': len(clusters),
            'nb_significatives': nb_sig,
            'pourcent_significatives': nb_sig/len(clusters)*100
        },
        'statistiques': stats,
        'clusters': clusters
    }

result_5pct  = run_lisa(0.05, '5%')
result_1pct  = run_lisa(0.01, '1%')

# === Écrire les fichiers JS ===
for result, varname, fname, label in [
    (result_5pct, 'LISA_DATA_BETTI',     WEB_DIR+'lisa_data_betti.js',     '5%'),
    (result_1pct, 'LISA_DATA_BETTI_1PCT', WEB_DIR+'lisa_data_betti_1pct.js', '1%'),
]:
    content = (
        f"// Données LISA Betti et al. (2008) - seuil {label} - SEED={SEED}\n"
        f"// Généré automatiquement - Ne pas modifier manuellement\n\n"
        f"const {varname} = " + json.dumps(result, ensure_ascii=False, indent=2) + ";\n"
    )
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nOK -> {fname}")

print("\nTerminé !")
