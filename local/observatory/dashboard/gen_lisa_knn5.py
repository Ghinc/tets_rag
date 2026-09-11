"""
Génère les 4 fichiers LISA avec matrice KNN k=5 et 9999 permutations
(cohérent avec le tableau Moran de l'article)

Produit :
  lisa_data.js            → LISA égal 5%
  lisa_data_1pct.js       → LISA égal 1%
  lisa_data_betti.js      → LISA Betti 5%
  lisa_data_betti_1pct.js → LISA Betti 1%
"""
import json, math, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

import geopandas as gpd
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran_Local, Moran

SEED = 42
np.random.seed(SEED)

WEB_DIR  = 'c:/These/oppchovec_visu/stage_ambroise/Code/WEB/'
GEOJSON  = 'c:/These/oppchovec_visu/stage_ambroise/Données/Commune_Corse.geojson'
DATA_JSON = WEB_DIR + 'data_scores_0_10.json'

# === Charger données ===
with open(DATA_JSON, encoding='utf-8') as f:
    data = json.load(f)

communes = list(data.keys())
alpha, beta = 2.5, 1.5

def mean(x): return sum(x)/len(x)
def std(x):  m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def pearson(a, b):
    n=len(a); ma,mb=mean(a),mean(b)
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da=math.sqrt(sum((v-ma)**2 for v in a))
    db=math.sqrt(sum((v-mb)**2 for v in b))
    return num/(da*db) if da*db else 0
def cv(x): m=mean(x); return std(x)/m if m else 0
def norm010(d):
    mn,mx=min(d.values()),max(d.values())
    return {c:(d[c]-mn)/(mx-mn)*10 if mx!=mn else 5 for c in communes}

# === Poids Betti (scores 0-1) ===
dims_keys = ['Score_Opp', 'Score_Cho', 'Score_Vec']
sc01  = {k: [data[c][k] for c in communes] for k in dims_keys}
p1    = [cv(sc01[k]) for k in dims_keys]
corrs = [[pearson(sc01[dims_keys[i]], sc01[dims_keys[j]]) for j in range(3)] for i in range(3)]
p2    = [1/mean([abs(corrs[i][j]) for j in range(3)]) for i in range(3)]
pkRaw = [p1[i]*p2[i] for i in range(3)]; s = sum(pkRaw)
pk_betti = [v/s for v in pkRaw]
pk_egal  = [1, 1, 1]
print(f"Poids Betti : Opp={pk_betti[0]:.4f}  Cho={pk_betti[1]:.4f}  Vec={pk_betti[2]:.4f}")

# === OppChoVec brut → normalisé 0-10 ===
def calc_occ(pkv):
    return {c: (1/3)*sum(pkv[i]*data[c][dims_keys[i]]**beta for i in range(3))**(alpha/beta)
            for c in communes}

occ_egal_010  = norm010(calc_occ(pk_egal))
occ_betti_010 = norm010(calc_occ(pk_betti))

# === Charger GeoJSON ===
print("\nChargement GeoJSON...")
gdf = gpd.read_file(GEOJSON)
nom_col = next((c for c in ['nom','NOM','NOM_COM','name','NAME','Zone'] if c in gdf.columns), None)
print(f"  Colonne nom : {nom_col} | {len(gdf)} communes")

df_join = pd.DataFrame({
    'nom': communes,
    'OppChoVec_Egal_0_10':  [occ_egal_010[c]  for c in communes],
    'OppChoVec_Betti_0_10': [occ_betti_010[c] for c in communes],
})
gdf = gdf.merge(df_join, left_on=nom_col, right_on='nom', how='inner')
print(f"  {len(gdf)} communes après jointure")

# === Matrice KNN k=5 ===
print("\nCalcul matrice KNN k=5...")
gdf_proj = gdf.to_crs('EPSG:2154')
w = KNN.from_dataframe(gdf_proj, k=5)
w.transform = 'r'
print(f"  Voisins moyens : {w.mean_neighbors:.2f} (= 5 par construction)")

# === Fonction LISA ===
def run_lisa(col, seuil, label, poids_info):
    print(f"\nCalcul LISA  [{col}]  seuil={seuil}  permutations=9999 ...")
    y = gdf_proj[col].values

    moran_global = Moran(y, w)
    print(f"  Moran I = {moran_global.I:.4f}   p = {moran_global.p_sim:.4f}")

    lisa = Moran_Local(y, w, permutations=9999, seed=SEED)

    sig = lisa.p_sim < seuil
    cluster_map = {1:'HH (High-High)', 2:'LH (Low-High)', 3:'LL (Low-Low)', 4:'HL (High-Low)'}

    clusters = {}
    for pos, (_, row) in enumerate(gdf_proj.iterrows()):
        nom = row[nom_col]
        is_sig = bool(sig[pos])
        q = int(lisa.q[pos])
        clusters[nom] = {
            'cluster':     cluster_map.get(q, 'Non significatif') if is_sig else 'Non significatif',
            'lisa_I':      float(lisa.Is[pos]),
            'p_value':     float(lisa.p_sim[pos]),
            'significatif': is_sig,
            'oppchovec':   float(y[pos])
        }

    nb_sig = sum(1 for v in clusters.values() if v['significatif'])
    stats  = {k: sum(1 for v in clusters.values() if v['cluster']==k)
              for k in ['HH (High-High)','LL (Low-Low)','HL (High-Low)','LH (Low-High)','Non significatif']}

    print(f"  Significatifs : {nb_sig} ({nb_sig/len(clusters)*100:.1f}%)")
    for k, v in stats.items():
        if v > 0: print(f"    {k}: {v}")

    return {
        'metadata': {
            'description': f'LISA - {col} - seuil {label} - KNN k=5 - 9999 permutations',
            'variable': col,
            'poids': poids_info,
            'methode': 'Moran Local - KNN k=5 - row-standardized',
            'permutations': 9999,
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

# === Lancer les 4 analyses ===
betti_info = {'type':'betti', 'Opp':round(pk_betti[0],4), 'Cho':round(pk_betti[1],4), 'Vec':round(pk_betti[2],4)}
egal_info  = {'type':'egal',  'pk':[1,1,1]}

res_egal_5pct  = run_lisa('OppChoVec_Egal_0_10',  0.05, '5%', egal_info)
res_egal_1pct  = run_lisa('OppChoVec_Egal_0_10',  0.01, '1%', egal_info)
res_betti_5pct = run_lisa('OppChoVec_Betti_0_10', 0.05, '5%', betti_info)
res_betti_1pct = run_lisa('OppChoVec_Betti_0_10', 0.01, '1%', betti_info)

# === Écrire les fichiers JS ===
outputs = [
    (res_egal_5pct,  'LISA_DATA',           WEB_DIR+'lisa_data.js',           'égal 5%'),
    (res_egal_1pct,  'LISA_DATA_1PCT',       WEB_DIR+'lisa_data_1pct.js',      'égal 1%'),
    (res_betti_5pct, 'LISA_DATA_BETTI',      WEB_DIR+'lisa_data_betti.js',     'Betti 5%'),
    (res_betti_1pct, 'LISA_DATA_BETTI_1PCT', WEB_DIR+'lisa_data_betti_1pct.js','Betti 1%'),
]
for result, varname, fname, label in outputs:
    content = (
        f"// Données LISA {label} — KNN k=5 — 9999 permutations — SEED={SEED}\n"
        f"// Généré automatiquement — Ne pas modifier manuellement\n\n"
        f"const {varname} = " + json.dumps(result, ensure_ascii=False, indent=2) + ";\n"
    )
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nOK → {fname}")

print("\n✓ Terminé !")
